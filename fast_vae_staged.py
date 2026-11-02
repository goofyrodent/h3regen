"""Experimental CPU-staged StreamRender-H3 decoder.

This keeps temporal activations and Q/K/V tensors in system RAM between attention
operations. Attention is still computed using the original decoder's temporal and
spatial windows and per-block global tokens. It is deliberately a separate class:
the shipped LeanNonCausalViTDecoder is not modified.

Experimental: numerical equivalence and practical performance must be checked on
the target GPU before production use.
"""
from __future__ import annotations

import logging
import time

import torch
from torch.nn import functional as F

from .fast_vae_decoder import (
    NonCausalViTDecoder, OUT_CH, PATCH, PATCH_T,
    _apply_qk_norm, _apply_rotary_pos_emb_impl,
)

ROWS = 16384
_LOG = logging.getLogger("OpenH3Regenerate")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _host_memory_snapshot():
    """Best-effort process/system RAM snapshot; psutil is optional."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 2**30, psutil.virtual_memory().used / 2**30
    except Exception:
        return None, None


def _fmt_gb(value):
    return "n/a" if value is None else f"{value:.2f} GB"


def _cpu_linear_chunks(module, src_cpu, scale=None):
    """Apply a GPU linear module to CPU-resident rows, returning CPU rows."""
    flat = src_cpu.reshape(-1, src_cpu.shape[-1])
    pieces = []
    for start in range(0, flat.shape[0], ROWS):
        part = flat[start:start + ROWS].to(module.weight.device, non_blocking=False)
        y = module(part)
        if scale is not None:
            y = y * scale
        pieces.append(y.to("cpu"))
        del part, y
    return torch.cat(pieces, 0).view(*src_cpu.shape[:-1], -1)


class StagedNonCausalViTDecoder(NonCausalViTDecoder):
    """RAM-staged variant preserving the decoder's native attention geometry."""

    def _diag_begin(self):
        """Start a measured phase. Synchronisation makes timings meaningful."""
        dev = self._diag_device
        _sync(dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        return time.perf_counter()

    def _diag_end(self, name, started):
        """Finish a phase and record time plus its own GPU memory high-water mark."""
        dev = self._diag_device
        _sync(dev)
        elapsed = time.perf_counter() - started
        if dev.type == "cuda":
            peak_a = torch.cuda.max_memory_allocated(dev) / 2**30
            peak_r = torch.cuda.max_memory_reserved(dev) / 2**30
            cur_a = torch.cuda.memory_allocated(dev) / 2**30
            cur_r = torch.cuda.memory_reserved(dev) / 2**30
            free_b, total_b = torch.cuda.mem_get_info(dev)
            free_g = free_b / 2**30
            self._diag_peak_alloc = max(self._diag_peak_alloc, peak_a)
            self._diag_peak_reserved = max(self._diag_peak_reserved, peak_r)
            detail = (f"{name}={elapsed:.2f}s"
                      f"(peak {peak_a:.2f}/{peak_r:.2f} GB, current {cur_a:.2f}/{cur_r:.2f} GB, "
                      f"free {free_g:.2f} GB)")
        else:
            detail = f"{name}={elapsed:.2f}s(GPU metrics n/a)"
        self._diag_current_phases.append(detail)

    def _ff_cpu(self, blk, x_cpu):
        flat = x_cpu.reshape(-1, x_cpu.shape[-1])
        pieces = []
        for start in range(0, flat.shape[0], ROWS):
            t = flat[start:start + ROWS].to(blk.norm2.weight.device)
            gate, up = blk.ff.w1(blk.norm2(t.float()).to(t.dtype)).chunk(2, -1)
            y = blk.ff.w2(F.silu(gate) * up)
            pieces.append(y.to("cpu"))
            del t, gate, up, y
        return torch.cat(pieces, 0).view_as(x_cpu)

    def _residual_cpu(self, x_cpu, delta_cpu):
        # Match the reference's promotion when the fp32 layer scale promotes bf16.
        dtype = torch.promote_types(x_cpu.dtype, delta_cpu.dtype)
        if dtype != x_cpu.dtype:
            x_cpu = x_cpu.to(dtype)
        flat_x = x_cpu.reshape(-1, x_cpu.shape[-1])
        flat_d = delta_cpu.reshape(-1, delta_cpu.shape[-1])
        for start in range(0, flat_x.shape[0], ROWS):
            flat_x[start:start + ROWS].add_(flat_d[start:start + ROWS].to(dtype))
        return x_cpu

    def _layer_staged(self, li, blk, x_cpu, g):
        B, T, H, W, nb, S = self._geom
        dev, tb, nh = g.device, self.block, self.heads
        s, st = self._shifts(li)

        # Global-token Q/K/V stay small and resident on the GPU.
        phase = self._diag_begin()
        hg = blk.norm1(g.float()).to(g.dtype)
        qg, kg, vg = blk.attn.to_qkv(hg).view(B, nb, S, nh, 3 * self.dh).chunk(3, -1)
        if blk.attn.norm_q is not None:
            qg = _apply_qk_norm(blk.attn.norm_q, qg)
            kg = _apply_qk_norm(blk.attn.norm_k, kg)
        self._diag_end("global_qkv", phase)

        # Compute local Q/K/V one frame at a time and immediately stage them in RAM.
        phase = self._diag_begin()
        qs = ks = vs = None
        for t in range(T):
            xt = x_cpu[:, t].to(dev)
            h = blk.norm1(xt.float()).to(xt.dtype)
            qt, kt, vt = blk.attn.to_qkv(h).view(B, H, W, nh, 3 * self.dh).chunk(3, -1)
            if blk.attn.norm_q is not None:
                qt = _apply_qk_norm(blk.attn.norm_q, qt)
                kt = _apply_qk_norm(blk.attn.norm_k, kt)
            # Same window-local RoPE coordinates as _rope(T,...), without a clip-sized table.
            rope = self._rope(1, H, W, s, (t + st) % tb, dev)
            rope = tuple(r.expand(B, -1, -1, -1) for r in rope)
            qt = _apply_rotary_pos_emb_impl(qt.reshape(B, H * W, nh, self.dh), rope).view(B, H, W, nh, self.dh)
            kt = _apply_rotary_pos_emb_impl(kt.reshape(B, H * W, nh, self.dh), rope).view(B, H, W, nh, self.dh)
            if qs is None:
                shape = (B, T, H, W, nh, self.dh)
                qs = torch.empty(shape, dtype=qt.dtype, device="cpu")
                ks, vs = torch.empty_like(qs), torch.empty_like(qs)
            qs[:, t].copy_(qt.to("cpu")); ks[:, t].copy_(kt.to("cpu")); vs[:, t].copy_(vt.to("cpu"))
            del xt, h, qt, kt, vt, rope
        self._diag_end("local_qkv_rope_cpu_stage", phase)

        # Window attention. Keep partitioned Q/K/V in system RAM and transfer only a
        # small batch of spatial windows to the GPU at a time. This preserves the
        # original window order, validity mask, and global-token keys while lowering
        # the temporary GPU working set.
        o_cpu = torch.empty((B, T, H, W, nh * self.dh), dtype=qs.dtype, device="cpu")
        nt = (T + st + tb - 1) // tb
        phase = self._diag_begin()
        spatial_batch = int(getattr(self, "spatial_batch", 4))  # one item is one 16x16 window
        if spatial_batch < 1:
            raise ValueError(f"spatial_batch must be >= 1, got {spatial_batch}")
        for j in range(nt):
            lo, hi = max(0, j * tb - st), min(T, (j + 1) * tb - st)
            off = lo + st - j * tb
            # Partition on CPU: the large padded/window tensors never need to coexist
            # on the GPU. _partition is device-agnostic and preserves reference order.
            qw_cpu, vmask_cpu, info = self._partition(qs[:, lo:hi], s, off)
            kw_cpu, _, _ = self._partition(ks[:, lo:hi], s, off)
            vw_cpu, _, _ = self._partition(vs[:, lo:hi], s, off)
            nwin = info[8] * info[9]
            b0, b1 = lo // tb, (hi - 1) // tb
            qw_cpu = qw_cpu.view(B, nwin, *qw_cpu.shape[1:])
            kw_cpu = kw_cpu.view(B, nwin, *kw_cpu.shape[1:])
            vw_cpu = vw_cpu.view(B, nwin, *vw_cpu.shape[1:])
            ow_parts = []
            for ws in range(0, nwin, spatial_batch):
                we = min(nwin, ws + spatial_batch)
                count = we - ws
                qw = qw_cpu[:, ws:we].reshape(B * count, *qw_cpu.shape[2:]).to(dev)
                kw = kw_cpu[:, ws:we].reshape(B * count, *kw_cpu.shape[2:]).to(dev)
                vw = vw_cpu[:, ws:we].reshape(B * count, *vw_cpu.shape[2:]).to(dev)
                kgw = kg[:, [b0, b1]].reshape(B, 1, 2 * S, nh, self.dh).expand(B, count, 2 * S, nh, self.dh)
                vgw = vg[:, [b0, b1]].reshape(B, 1, 2 * S, nh, self.dh).expand(B, count, 2 * S, nh, self.dh)
                gmask = torch.tensor([True] * S + [b1 != b0] * S, device=dev).expand(count, 2 * S)
                Kc = torch.cat([kw, kgw.reshape(B * count, 2 * S, nh, self.dh)], 1).transpose(1, 2)
                Vc = torch.cat([vw, vgw.reshape(B * count, 2 * S, nh, self.dh)], 1).transpose(1, 2)
                vmask = vmask_cpu[ws:we].to(dev).repeat(B, 1)
                kmask = torch.cat([vmask, gmask.repeat(B, 1)], 1)[:, None, None, :]
                ow = F.scaled_dot_product_attention(qw.transpose(1, 2), Kc, Vc, attn_mask=kmask).transpose(1, 2)
                ow_parts.append(ow.to("cpu").view(B, count, *ow.shape[1:]))
                del qw, kw, vw, kgw, vgw, Kc, Vc, vmask, kmask, ow
            ow_cpu = torch.cat(ow_parts, dim=1).reshape(B * nwin, *ow_parts[0].shape[2:])
            outw = self._unpartition(ow_cpu, info, s, off).reshape(B, hi - lo, H, W, nh * self.dh)
            o_cpu[:, lo:hi].copy_(outw)
            del qw_cpu, kw_cpu, vw_cpu, ow_parts, ow_cpu, outw
        self._diag_end(f"window_attention_{nt}_windows_chunked", phase)

        # Each global-token group attends to its native five-latent block.
        phase = self._diag_begin()
        og = []
        for b in range(nb):
            lo, hi = b * tb, min(T, (b + 1) * tb)
            kb = ks[:, lo:hi].to(dev).reshape(B, -1, nh, self.dh)
            vb = vs[:, lo:hi].to(dev).reshape(B, -1, nh, self.dh)
            Kg = torch.cat([kb, kg[:, b]], 1).transpose(1, 2)
            Vg = torch.cat([vb, vg[:, b]], 1).transpose(1, 2)
            og.append(F.scaled_dot_product_attention(qg[:, b].transpose(1, 2), Kg, Vg).transpose(1, 2).reshape(B, 1, S, -1))
            del kb, vb, Kg, Vg
        og = torch.cat(og, 1)
        del qs, ks, vs
        self._diag_end(f"global_attention_{nb}_blocks", phase)

        # Linear projections and FFNs operate in small GPU row batches; activations return to RAM.
        phase = self._diag_begin()
        attn_delta = _cpu_linear_chunks(blk.attn.to_out, o_cpu, blk.scale1)
        x_cpu = self._residual_cpu(x_cpu, attn_delta)
        del o_cpu, attn_delta
        self._diag_end("attention_output_projection_residual", phase)

        phase = self._diag_begin()
        g = g + blk.attn.to_out(og) * blk.scale1
        self._diag_end("global_output_projection", phase)

        phase = self._diag_begin()
        ff_delta = self._ff_cpu(blk, x_cpu)
        x_cpu = self._residual_cpu(x_cpu, ff_delta * blk.scale2.detach().cpu())
        del ff_delta
        self._diag_end("feed_forward_cpu_staged", phase)

        phase = self._diag_begin()
        g = g + self._ff_gpu(blk, g) * blk.scale2
        self._diag_end("global_feed_forward", phase)
        return x_cpu, g

    @staticmethod
    def _ff_gpu(blk, x):
        gate, up = blk.ff.w1(blk.norm2(x.float()).to(x.dtype)).chunk(2, -1)
        return blk.ff.w2(F.silu(gate) * up)

    def forward(self, z, sink=None):
        decode_started = time.perf_counter()
        B, T, _, H, W = z.shape
        dev, tb = z.device, self.block
        nb = (T + tb - 1) // tb
        self._diag_device = dev
        self._diag_peak_alloc = 0.0
        self._diag_peak_reserved = 0.0
        if dev.type == "cuda":
            _sync(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        rss0, sys0 = _host_memory_snapshot()
        rss_peak, sys_peak = rss0, sys0
        layer_times = []
        phase_times = {}
        compute_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else z.dtype
        raw = z.float() * self.latents_std.view(1, 1, -1, 1, 1) + self.latents_mean.view(1, 1, -1, 1, 1)
        raw = self.post_quant_conv(raw.transpose(1, 2)).permute(0, 2, 3, 4, 1)
        # Embed one frame at a time; retain the resulting activation in CPU RAM.
        phase_started = time.perf_counter()
        x_cpu = None
        for t in range(T):
            with torch.autocast("cuda", enabled=False):
                xt = self.x_embedder(raw[:, t].float())
            xt = xt.to(compute_dtype)
            if x_cpu is None:
                x_cpu = torch.empty((B, T, H, W, xt.shape[-1]), dtype=xt.dtype, device="cpu")
            x_cpu[:, t].copy_(xt.to("cpu"))
            del xt
        del raw
        _sync(dev)
        phase_times["embedding + CPU staging"] = time.perf_counter() - phase_started
        D = x_cpu.shape[-1]
        S = self.register_tokens.shape[1] + 1
        g = torch.cat([self.register_tokens.expand(B, -1, -1), torch.zeros(B, 1, D, device=dev, dtype=compute_dtype)], 1)
        g = g.unsqueeze(1).expand(B, nb, S, D).contiguous()
        self._geom = (B, T, H, W, nb, S)
        layer_phase_reports = []
        for li, blk in enumerate(self.blocks):
            _sync(dev)
            layer_started = time.perf_counter()
            self._diag_current_phases = []
            x_cpu, g = self._layer_staged(li, blk, x_cpu, g)
            _sync(dev)
            elapsed = time.perf_counter() - layer_started
            layer_times.append(elapsed)
            rss_now, sys_now = _host_memory_snapshot()
            if rss_now is not None:
                rss_peak = max(rss_peak or 0.0, rss_now)
            if sys_now is not None:
                sys_peak = max(sys_peak or 0.0, sys_now)
            layer_phase_reports.append(
                f"L{li + 1}={elapsed:.2f}s; processRAM={_fmt_gb(rss_now)}; "
                f"systemRAM={_fmt_gb(sys_now)}: " + "; ".join(self._diag_current_phases)
            )
            if li == 0 or (li + 1) % 4 == 0 or li + 1 == len(self.blocks):
                if dev.type == "cuda":
                    alloc = torch.cuda.memory_allocated(dev) / 2**30
                    reserv = torch.cuda.memory_reserved(dev) / 2**30
                    gpu_note = f"GPU alloc/reserved {alloc:.2f}/{reserv:.2f} GB"
                else:
                    gpu_note = "GPU memory n/a"
                _LOG.info("[H3 staged diagnostics] layer %02d/%02d: %.2fs; %s; process RAM %s; system RAM used %s",
                          li + 1, len(self.blocks), elapsed, gpu_note, _fmt_gb(rss_now), _fmt_gb(sys_now))

        phase_started = time.perf_counter()
        keep = [PATCH_T - (self.frames_to_trim if t % tb == 0 else 0) for t in range(T)]
        if sink is None:
            out = torch.empty(B, OUT_CH, sum(keep), H * PATCH, W * PATCH, device=dev, dtype=torch.float32)
            sink = lambda f, p: out.__setitem__((slice(None), slice(None), slice(f, f + p.shape[2])), p)
        else:
            out = None
        f = 0
        for t in range(T):
            xt = x_cpu[:, t].to(dev)
            xt = self.norm_out(xt)
            with torch.autocast("cuda", enabled=False):
                pt = self.proj_out(xt.float())
            pt = pt.view(B, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(0, 3, 4, 1, 5, 2, 6)
            pt = pt.reshape(B, OUT_CH, PATCH_T, H * PATCH, W * PATCH)[:, :, PATCH_T - keep[t]:]
            sink(f, pt)
            f += keep[t]
            del xt, pt
        _sync(dev)
        phase_times["output projection + sink"] = time.perf_counter() - phase_started
        total = time.perf_counter() - decode_started
        if dev.type == "cuda":
            # Per-phase peak values are aggregated because each measured phase resets
            # CUDA's peak counters to isolate its own temporary allocations.
            peak_alloc = self._diag_peak_alloc
            peak_reserved = self._diag_peak_reserved
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
            gpu_summary = (f"peak allocated/reserved across measured phases {peak_alloc:.2f}/{peak_reserved:.2f} GB; "
                           f"free at end {free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GB")
        else:
            gpu_summary = "CUDA metrics unavailable"
        rss_end, sys_end = _host_memory_snapshot()
        layer_summary = ", ".join(f"L{i+1}:{v:.2f}s" for i, v in enumerate(layer_times))
        phase_summary = ", ".join(f"{k}={v:.2f}s" for k, v in phase_times.items())
        self.last_decode_report = (
            f"input={tuple(z.shape)}; output_frames={sum(keep)}; total={total:.2f}s; "
            f"{gpu_summary}; process RAM start/peak/end={_fmt_gb(rss0)}/{_fmt_gb(rss_peak)}/{_fmt_gb(rss_end)}; "
            f"system RAM used start/peak/end={_fmt_gb(sys0)}/{_fmt_gb(sys_peak)}/{_fmt_gb(sys_end)}; "
            f"phases[{phase_summary}]; layer_seconds[{layer_summary}]; "
            f"layer_phase_breakdown[{ ' || '.join(layer_phase_reports) }]"
        )
        _LOG.info("[H3 staged diagnostics] DECODE COMPLETE: %s", self.last_decode_report)
        return out
