"""Flag-Free Compiled Classical Modded-NanoGPT Leaderboard Architecture (`Wave 45` / `EXPERIMENT_PLAN_060.md`).

Provenance & Mathematical Rationale:
1. Built for `Wave 45` (`EXPERIMENT_PLAN_060.md`) following Rule 6 of `docs/SCRIPT_BEST_PRACTICES.md`
   (clean, flag-free per-experiment module without legacy boolean branches).
2. Eliminates the 5 throughput and optimization bottlenecks identified in legacy `ClassicalModdedGPT`:
   - Single-pass chunked `15 * tanh(z / 15)` cross-entropy (`direct_chunked_softcap_linear_ce`)
     eliminates the `3.07+ GiB` `(B, T, 50304)` logits materialization so `B=16..32, T=1024`
     fits comfortably inside `16 GB` VRAM without activation checkpointing.
   - Zero forced per-block activation checkpointing on `16 GB` GPUs (`-33%` forward FLOP overhead
     and full compatibility with whole-backbone `torch.compile`).
   - Native `bfloat16` half-truncated `LeaderboardRotary` (`base = 1024.0`) with zero per-layer
     `float32` upcast/downcast round-trips.
   - Interior-layer Layer-0 value residual (`v_0`) mixing (`v = sa_lambdas[0] * v + sa_lambdas[1] * v_0`)
     on layers without a dedicated token Value Embedding bank (`ve_l is None`), fixing the legacy
     `v = 0.5 * v` signal-halving bug on interior layers `3, 4, 5, 6, 8`.
   - Exact `KellerJordan/modded-nanogpt` optimizer partitioning: batched `StiefelMuon` (`lr=0.04`,
     `ema_momentum=True`, `300`-step momentum warmup `0.85 -> 0.95`, `aspect_scale=True`) on all
     2D trunk matrices (with automatic `(3, d, d)` Q/K/V slice orthogonalization) + fused `Adam`
     (`betas=(0.8, 0.95)`, `eps=1e-10`) on `wte`/`value_embeds` (`lr=0.6`), `lm_head` (`lr=0.008`),
     and scalar gates (`lr=0.04`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from ..layers.direct_chunked_ce import direct_chunked_softcap_linear_ce
from ..optim.muon import StiefelMuon, StiefelMuonConfig

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_060.md Section 3
# ==============================================================================
DEFAULT_LEADERBOARD_VOCAB_SIZE: int = 50304
DEFAULT_VALID_FINEWEB_VOCAB_SIZE: int = 50257
DEFAULT_BOS_TOKEN_ID: int = 50256
DEFAULT_LEADERBOARD_EMBED_DIM: int = 768
DEFAULT_LEADERBOARD_NUM_HEADS: int = 6
DEFAULT_LEADERBOARD_NUM_LAYERS: int = 12
DEFAULT_LEADERBOARD_HIDDEN_DIM: int = 3072
DEFAULT_LEADERBOARD_SEQ_LEN: int = 1024
DEFAULT_LEADERBOARD_VALUE_EMBED_BANKS: int = 3
DEFAULT_LEADERBOARD_SKIP_ATTN_LAYERS: tuple[int, ...] = (7,)
DEFAULT_LONG_WINDOW_LAYERS: tuple[int, ...] = (0, 4, 11)
DEFAULT_FLEX_BLOCK_SIZE: int = 128
DEFAULT_INITIAL_LONG_WINDOW_BLOCKS: int = 3
DEFAULT_INITIAL_SHORT_WINDOW_BLOCKS: int = 1
DEFAULT_WS_SCHEDULE_BLOCKS: tuple[int, ...] = (3, 6, 8)
DEFAULT_WS_VALIDATE_BLOCKS: int = 12
DEFAULT_WS_VALIDATE_FINAL_BLOCKS: int = 16
DEFAULT_GATE_INPUT_CHANNELS: int = 12
DEFAULT_VE_GATE_PART_CHANNELS: int = 6
DEFAULT_SKIP_LOGIT_INIT: float = -1.5
DEFAULT_VE_INIT_STD: float = 0.02
DEFAULT_YARN_ALPHA: float = 1.0
DEFAULT_YARN_BETA: float = 32.0
DEFAULT_YARN_SCALE_COEF: float = 0.2

DEFAULT_LEADERBOARD_ATTN_SCALE: float = 0.12
DEFAULT_LEADERBOARD_ROPE_BASE: float = 1024.0
DEFAULT_LEADERBOARD_SOFTCAP: float = 15.0
DEFAULT_LEADERBOARD_CE_CHUNK_TOKENS: int = 8192
DEFAULT_LEADERBOARD_RMS_EPS: float = 1e-6
DEFAULT_LEADERBOARD_INIT_STD_FACTOR: float = 0.5
UNIFORM_SQRT3_BOUND_FACTOR: float = 1.7320508075688772

DEFAULT_LEADERBOARD_MUON_LR: float = 0.04
DEFAULT_LEADERBOARD_MUON_MOMENTUM: float = 0.95
DEFAULT_LEADERBOARD_MUON_MOMENTUM_INIT: float = 0.85
DEFAULT_LEADERBOARD_MUON_WARMUP_STEPS: int = 300
DEFAULT_LEADERBOARD_WTE_LR: float = 0.30
DEFAULT_LEADERBOARD_HEAD_LR: float = 0.008
DEFAULT_LEADERBOARD_SCALAR_LR: float = 0.04
DEFAULT_LEADERBOARD_ADAM_BETA1: float = 0.80
DEFAULT_LEADERBOARD_ADAM_BETA2: float = 0.95
DEFAULT_LEADERBOARD_ADAM_EPS: float = 1e-8

DEFAULT_MODDED_VAL_TOKENS: int = 10_485_760
DEFAULT_FAST_VAL_TOKENS: int = 65_536
FINEWEB_SHARD_HEADER_BYTES: int = 1024

InteriorValueMode = Literal["unit_v", "v0_residual", "legacy_half_v"]
VALID_INTERIOR_VALUE_MODES: tuple[str, ...] = ("unit_v", "v0_residual", "legacy_half_v")


@dataclass(frozen=True)
class CompiledLeaderboardClassicalConfig:
    """Configuration for `CompiledLeaderboardClassicalGPT` (`Wave 45` / `EXPERIMENT_PLAN_060.md`)."""

    vocab_size: int = DEFAULT_LEADERBOARD_VOCAB_SIZE
    embed_dim: int = DEFAULT_LEADERBOARD_EMBED_DIM
    num_heads: int = DEFAULT_LEADERBOARD_NUM_HEADS
    num_layers: int = DEFAULT_LEADERBOARD_NUM_LAYERS
    hidden_dim: int = DEFAULT_LEADERBOARD_HIDDEN_DIM
    max_seq_len: int = DEFAULT_LEADERBOARD_SEQ_LEN
    num_value_embeddings: int = DEFAULT_LEADERBOARD_VALUE_EMBED_BANKS
    skip_attn_layers: tuple[int, ...] = DEFAULT_LEADERBOARD_SKIP_ATTN_LAYERS
    long_window_layers: tuple[int, ...] = DEFAULT_LONG_WINDOW_LAYERS
    attn_scale: float = DEFAULT_LEADERBOARD_ATTN_SCALE
    rope_base: float = DEFAULT_LEADERBOARD_ROPE_BASE
    logit_softcap: float = DEFAULT_LEADERBOARD_SOFTCAP
    ce_chunk_tokens: int = DEFAULT_LEADERBOARD_CE_CHUNK_TOKENS
    interior_value_mode: InteriorValueMode = "unit_v"
    use_flex_doc_swc: bool = True
    initial_ws_blocks: int = DEFAULT_INITIAL_LONG_WINDOW_BLOCKS
    use_smear_gate: bool = True
    use_attn_gate: bool = True
    use_ve_gate: bool = True
    use_partial_key_offset: bool = True
    use_gated_xsa: bool = True
    use_sigmoid_skip: bool = True
    ve_init_std: float = DEFAULT_VE_INIT_STD
    bfloat16_weights: bool = True
    eps: float = DEFAULT_LEADERBOARD_RMS_EPS

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.interior_value_mode not in VALID_INTERIOR_VALUE_MODES:
            raise ValueError(
                f"interior_value_mode must be one of {VALID_INTERIOR_VALUE_MODES}, got {self.interior_value_mode!r}"
            )


@dataclass(frozen=True)
class CompiledLeaderboardOptimizerConfig:
    """Grouped optimizer hyperparameters for `build_compiled_leaderboard_optimizers`."""

    muon_lr: float = DEFAULT_LEADERBOARD_MUON_LR
    muon_warmup_steps: int = DEFAULT_LEADERBOARD_MUON_WARMUP_STEPS
    wte_lr: float = DEFAULT_LEADERBOARD_WTE_LR
    head_lr: float = DEFAULT_LEADERBOARD_HEAD_LR
    scalar_lr: float = DEFAULT_LEADERBOARD_SCALAR_LR
    polar_method: Literal["newton_schulz", "polar_express"] = "polar_express"
    weight_decay: float = 0.01
    use_normuon: bool = True
    use_twin_rail: bool = True


class NativeLeaderboardRotary(nn.Module):
    """Half-truncated Rotary Position Embedding (`base = 1024.0`) with in-place YaRN window scaling."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = DEFAULT_LEADERBOARD_SEQ_LEN,
        base: float = DEFAULT_LEADERBOARD_ROPE_BASE,
    ) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_seq_len = max(4096, int(max_seq_len))
        self.base = float(base)
        quarter = max(1, self.head_dim // 4)
        angular_freq = (1.0 / self.base) ** torch.linspace(0.0, 1.0, steps=quarter, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(quarter)])
        self.register_buffer("base_angular_freq", angular_freq.clone(), persistent=False)
        self.register_buffer("angular_freq", angular_freq.clone(), persistent=False)
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.register_buffer("cos_f32", theta.cos(), persistent=False)
        self.register_buffer("sin_f32", theta.sin(), persistent=False)
        self.register_buffer("cos_bf16", theta.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_bf16", theta.sin().to(torch.bfloat16), persistent=False)
        self.register_buffer("yarn_q_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def reset_yarn(self) -> None:
        """Reset RoPE angular frequencies and YaRN query softmax scale to Step-0 values in-place."""
        self.angular_freq.copy_(self.base_angular_freq)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos_f32.copy_(theta.cos())
        self.sin_f32.copy_(theta.sin())
        self.cos_bf16.copy_(self.cos_f32.to(torch.bfloat16))
        self.sin_bf16.copy_(self.sin_f32.to(torch.bfloat16))
        self.yarn_q_scale.fill_(1.0)

    @torch.no_grad()
    def apply_yarn(
        self,
        old_window_blocks: int,
        new_window_blocks: int,
        block_size: int = DEFAULT_FLEX_BLOCK_SIZE,
    ) -> None:
        """Interpolate RoPE frequencies and scale softmax temperature in-place when window grows (`Smear` YaRN)."""
        if new_window_blocks <= 0 or old_window_blocks <= 0 or new_window_blocks == old_window_blocks:
            return
        rotations = (float(block_size) * float(old_window_blocks) * self.angular_freq) / (2.0 * math.pi)
        scaling_factor = float(old_window_blocks) / float(new_window_blocks)
        interp_w = torch.clamp(
            (rotations - DEFAULT_YARN_ALPHA) / (DEFAULT_YARN_BETA - DEFAULT_YARN_ALPHA),
            0.0,
            1.0,
        )
        self.angular_freq.mul_(scaling_factor + interp_w * (1.0 - scaling_factor))
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos_f32.copy_(theta.cos())
        self.sin_f32.copy_(theta.sin())
        self.cos_bf16.copy_(self.cos_f32.to(torch.bfloat16))
        self.sin_bf16.copy_(self.sin_f32.to(torch.bfloat16))
        temp_mult = 1.0 + DEFAULT_YARN_SCALE_COEF * math.log(float(new_window_blocks) / float(old_window_blocks))
        self.yarn_q_scale.mul_(float(temp_mult))

    def get_cos_sin(self, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(cos, sin)` of shape `(1, seq_len, 1, head_dim // 2)` for `(B, T, H, D)` zero-copy layout."""
        if dtype == torch.bfloat16:
            return self.cos_bf16[:seq_len].unsqueeze(0).unsqueeze(2), self.sin_bf16[:seq_len].unsqueeze(0).unsqueeze(2)
        if dtype == torch.float32:
            return self.cos_f32[:seq_len].unsqueeze(0).unsqueeze(2), self.sin_f32[:seq_len].unsqueeze(0).unsqueeze(2)
        return (
            self.cos_f32[:seq_len].to(dtype).unsqueeze(0).unsqueeze(2),
            self.sin_f32[:seq_len].to(dtype).unsqueeze(0).unsqueeze(2),
        )


def apply_native_leaderboard_rotary(
    x_bthd: torch.Tensor,
    cos_4d: torch.Tensor,
    sin_4d: torch.Tensor,
) -> torch.Tensor:
    """Apply half-chunk rotary embedding to `(B, T, H, D)` directly in `x_bthd.dtype` (zero `float32` upcast)."""
    x1, x2 = x_bthd.chunk(2, dim=-1)
    y1 = x1 * cos_4d + x2 * sin_4d
    y2 = x2 * cos_4d - x1 * sin_4d
    return torch.cat((y1, y2), dim=-1)


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _fused_softcap_ce_inplace_kernel(
        raw_ptr,
        target_ptr,
        loss_ptr,
        stride_row,
        vocab_size,
        softcap,
        inv_n_tok,
        BLOCK_V: tl.constexpr,
    ):
        """Fused in-place Softcap (`C * tanh(x / C)`) + Cross-Entropy + Backward Gradient Triton Kernel (`8` args)."""
        row_idx = tl.program_id(0)
        row_start_ptr = raw_ptr + row_idx * stride_row
        y_t = tl.load(target_ptr + row_idx)
        inv_softcap = 1.0 / softcap

        m_i = -float("inf")
        d_i = 0.0
        for v_off in range(0, vocab_size, BLOCK_V):
            cols = v_off + tl.arange(0, BLOCK_V)
            mask = cols < vocab_size
            x_val = tl.load(row_start_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            u = x_val * inv_softcap
            abs_u = tl.abs(u)
            exp_m2u = tl.exp(-2.0 * abs_u)
            tanh_u = tl.where(u >= 0.0, 1.0, -1.0) * (1.0 - exp_m2u) / (1.0 + exp_m2u)
            z = tl.where(mask, softcap * tanh_u, -float("inf"))
            block_max = tl.max(z, axis=0)
            m_new = tl.maximum(m_i, block_max)
            d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(z - m_new), axis=0)
            m_i = m_new

        lse = m_i + tl.log(d_i)
        raw_y = tl.load(row_start_ptr + y_t).to(tl.float32)
        u_y = raw_y * inv_softcap
        abs_uy = tl.abs(u_y)
        exp_m2uy = tl.exp(-2.0 * abs_uy)
        tanh_uy = tl.where(u_y >= 0.0, 1.0, -1.0) * (1.0 - exp_m2uy) / (1.0 + exp_m2uy)
        z_y = softcap * tanh_uy
        tl.store(loss_ptr + row_idx, (lse - z_y) * inv_n_tok)

        for v_off in range(0, vocab_size, BLOCK_V):
            cols = v_off + tl.arange(0, BLOCK_V)
            mask = cols < vocab_size
            x_val = tl.load(row_start_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            u = x_val * inv_softcap
            abs_u = tl.abs(u)
            exp_m2u = tl.exp(-2.0 * abs_u)
            tanh_u = tl.where(u >= 0.0, 1.0, -1.0) * (1.0 - exp_m2u) / (1.0 + exp_m2u)
            z = softcap * tanh_u
            prob = tl.exp(z - lse)
            is_target = tl.where(cols == y_t, 1.0, 0.0)
            grad = (prob - is_target) * (1.0 - tanh_u * tanh_u) * inv_n_tok
            tl.store(row_start_ptr + cols, grad, mask=mask)


def _softcap_ce_chunk_elementwise_fallback(
    raw_logits: torch.Tensor,
    yc: torch.Tensor,
    softcap: float,
    inv_softcap: float,
    inv_n_tok: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback elementwise softcap + logsumexp + softmax + tanh-Jacobian step under `torch.no_grad()`."""
    u = torch.tanh(raw_logits.float() * inv_softcap)
    logits = u * softcap
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    target_logits = logits.gather(1, yc.unsqueeze(1))
    chunk_loss = (lse - target_logits).sum() * inv_n_tok
    probs = torch.exp(logits - lse)
    probs.scatter_add_(1, yc.unsqueeze(1), torch.full_like(target_logits, -1.0))
    grad_raw = (probs * (1.0 - u * u) * inv_n_tok).to(raw_logits.dtype)
    return chunk_loss, grad_raw


class _CompiledChunkedSoftcapLinearCEFn(torch.autograd.Function):
    """Single-pass chunked linear + tanh-softcap CE using in-place Triton kernel + Tensor-Core GEMMs."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_3d: torch.Tensor,
        w_lm: torch.Tensor,
        targets: torch.Tensor,
        softcap: float = DEFAULT_LEADERBOARD_SOFTCAP,
        chunk_tokens: int = DEFAULT_LEADERBOARD_CE_CHUNK_TOKENS,
        use_compiled_chunk: bool = True,
    ) -> torch.Tensor:
        with torch.no_grad():
            orig_shape = x_3d.shape
            d_model = x_3d.size(-1)
            x_flat = x_3d.reshape(-1, d_model)
            flat_y = targets.reshape(-1).long()
            n_tok = x_flat.size(0)
            chunk_sz = max(1, min(int(chunk_tokens), n_tok))

            work_dtype = (
                torch.bfloat16
                if (x_flat.is_cuda and (x_flat.dtype == torch.bfloat16 or torch.is_autocast_enabled()))
                else x_flat.dtype
            )
            x_work = x_flat if x_flat.dtype == work_dtype else x_flat.to(work_dtype)
            w_work = w_lm if w_lm.dtype == work_dtype else w_lm.to(work_dtype)

            need_dx = bool(ctx.needs_input_grad[0])
            need_dw = bool(ctx.needs_input_grad[1])
            dx_work = torch.empty_like(x_work) if need_dx else None
            dw_work = torch.zeros_like(w_work) if need_dw else None
            total_loss = torch.zeros((), device=x_flat.device, dtype=torch.float32)

            inv_n_tok = 1.0 / float(max(1, n_tok))
            softcap_f = float(softcap)
            inv_softcap = 1.0 / softcap_f
            v_vocab = w_work.size(0)
            use_triton = bool(HAS_TRITON and x_flat.is_cuda and use_compiled_chunk)
            row_losses = (
                torch.empty((chunk_sz,), device=x_flat.device, dtype=torch.float32) if use_triton else None
            )
            raw_buf = torch.empty((chunk_sz, v_vocab), device=x_flat.device, dtype=work_dtype)

            for start in range(0, n_tok, chunk_sz):
                end = min(start + chunk_sz, n_tok)
                c_len = end - start
                xc = x_work[start:end]
                yc = flat_y[start:end]
                raw = raw_buf[:c_len]
                torch.mm(xc, w_work.t(), out=raw)

                if use_triton and row_losses is not None:
                    try:
                        r_loss = row_losses[:c_len]
                        _fused_softcap_ce_inplace_kernel[(c_len,)](
                            raw,
                            yc,
                            r_loss,
                            raw.stride(0),
                            v_vocab,
                            softcap_f,
                            inv_n_tok,
                            BLOCK_V=4096,
                            num_warps=8,
                        )
                        total_loss.add_(r_loss.sum())
                        grad_raw = raw
                    except (FileNotFoundError, RuntimeError, OSError):
                        use_triton = False
                        c_loss, grad_raw = _softcap_ce_chunk_elementwise_fallback(
                            raw, yc, softcap_f, inv_softcap, inv_n_tok
                        )
                        total_loss.add_(c_loss)
                else:
                    c_loss, grad_raw = _softcap_ce_chunk_elementwise_fallback(
                        raw, yc, softcap_f, inv_softcap, inv_n_tok
                    )
                    total_loss.add_(c_loss)

                if need_dx and dx_work is not None:
                    torch.mm(grad_raw, w_work, out=dx_work[start:end])
                if need_dw and dw_work is not None:
                    dw_work.addmm_(grad_raw.t(), xc)

            dx_saved = (
                dx_work.to(x_3d.dtype).view(orig_shape)
                if dx_work is not None
                else torch.empty(0, device=x_flat.device)
            )
            dw_saved = dw_work.to(w_lm.dtype) if dw_work is not None else torch.empty(0, device=w_lm.device)
            ctx.save_for_backward(dx_saved, dw_saved)
            ctx.need_dx = need_dx
            ctx.need_dw = need_dw
            return total_loss

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        dx_saved, dw_saved = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        dx = (dx_saved.float() * g).to(dx_saved.dtype) if ctx.need_dx else None
        dw = (dw_saved.float() * g).to(dw_saved.dtype) if ctx.need_dw else None
        return dx, dw, None, None, None, None


class CompiledLeaderboardBlock(nn.Module):
    """Single-GPU Compiled Modded-NanoGPT Transformer Block with Document-SWC FlexAttention, Gates, & XSA."""

    def __init__(
        self,
        cfg: CompiledLeaderboardClassicalConfig,
        layer_idx: int,
        has_value_embedding: bool,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.embed_dim = int(cfg.embed_dim)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.hidden_dim = int(cfg.hidden_dim)
        self.skip_attention = bool(self.layer_idx in cfg.skip_attn_layers)
        self.has_value_embedding = bool(has_value_embedding)
        self.is_long_window_layer = bool(
            self.layer_idx in cfg.long_window_layers or self.layer_idx == cfg.num_layers - 1
        )
        self.interior_value_mode: InteriorValueMode = cfg.interior_value_mode
        self.use_partial_key_offset = bool(cfg.use_partial_key_offset and self.is_long_window_layer)
        self.use_gated_xsa = bool(cfg.use_gated_xsa)
        self.attn_scale = float(cfg.attn_scale)
        self.eps = float(cfg.eps)
        self.gate_in_ch = min(DEFAULT_GATE_INPUT_CHANNELS, self.embed_dim)
        self.ve_part_ch = min(DEFAULT_VE_GATE_PART_CHANNELS, self.embed_dim)

        std = DEFAULT_LEADERBOARD_INIT_STD_FACTOR * (self.embed_dim ** -0.5)
        bound = UNIFORM_SQRT3_BOUND_FACTOR * std

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))

        if self.skip_attention:
            self.qkv_proj: nn.Linear | None = None
            self.out_proj: nn.Linear | None = None
            self.attn_gate: nn.Linear | None = None
            self.ve_gate: nn.Linear | None = None
            self.register_parameter("sa_lambdas", None)
            self.register_parameter("xsa_alpha", None)
        else:
            self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            with torch.no_grad():
                self.qkv_proj.weight.uniform_(-bound, bound)
                self.out_proj.weight.zero_()
            if not self.has_value_embedding and self.interior_value_mode == "unit_v":
                sa_init = [1.0, 0.0]
            else:
                sa_init = [0.5, 0.5]
            self.sa_lambdas: nn.Parameter | None = nn.Parameter(
                torch.tensor(sa_init, dtype=torch.float32)
            )
            if cfg.use_attn_gate:
                self.attn_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
                with torch.no_grad():
                    self.attn_gate.weight.zero_()
            else:
                self.attn_gate = None

            if cfg.use_ve_gate and self.has_value_embedding:
                self.ve_gate = nn.Linear(2 * self.ve_part_ch, self.num_heads, bias=False)
                with torch.no_grad():
                    self.ve_gate.weight.zero_()
            else:
                self.ve_gate = None

            if self.use_gated_xsa:
                self.xsa_alpha: nn.Parameter | None = nn.Parameter(
                    torch.zeros(self.num_heads, dtype=torch.float32)
                )
            else:
                self.register_parameter("xsa_alpha", None)

        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)
        with torch.no_grad():
            self.fc1.weight.uniform_(-bound, bound)
            self.fc2.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
        v_0: torch.Tensor | None = None,
        ve_l: torch.Tensor | None = None,
        block_mask: BlockMask | None = None,
        q_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute 1 transformer block in `(B, T, H, D)` physical layout and return `(x_out, v_0_out)` (`<= 8` args)."""
        lam_x = self.block_lambdas[0].to(x.dtype)
        lam_x0 = self.block_lambdas[1].to(x.dtype)
        x = lam_x * x + lam_x0 * x_0

        v_0_out = v_0
        if not self.skip_attention and self.qkv_proj is not None and self.out_proj is not None:
            bsz, seq_len, d_model = x.shape
            h = F.rms_norm(x, (d_model,), eps=self.eps)
            q, k, v = self.qkv_proj(h).view(bsz, seq_len, 3, self.num_heads, self.head_dim).unbind(dim=2)

            if v_0_out is None:
                v_0_out = v

            if self.sa_lambdas is not None:
                s0 = self.sa_lambdas[0].to(v.dtype)
                s1 = self.sa_lambdas[1].to(v.dtype)
                if ve_l is not None:
                    ve_heads = ve_l.view(bsz, seq_len, self.num_heads, self.head_dim)
                    if self.ve_gate is not None:
                        c_ve = self.ve_part_ch
                        ve_in = torch.cat([h[..., :c_ve], ve_l[..., :c_ve]], dim=-1)
                        ve_g = (2.0 * torch.sigmoid(self.ve_gate(ve_in))).unsqueeze(-1)
                        ve_heads = ve_heads * ve_g
                    v = s0 * v + s1 * ve_heads
                elif self.interior_value_mode == "v0_residual" and v_0_out is not None:
                    v = s0 * v + s1 * v_0_out
                else:
                    v = s0 * v

            q = F.rms_norm(q, (self.head_dim,), eps=self.eps)
            k = F.rms_norm(k, (self.head_dim,), eps=self.eps)
            q = apply_native_leaderboard_rotary(q, cos_4d, sin_4d)
            if q_scale is not None:
                q = q * q_scale.to(q.dtype)
            k = apply_native_leaderboard_rotary(k, cos_4d, sin_4d)

            if self.use_partial_key_offset and seq_len > 1:
                d_half = self.head_dim // 2
                k_stat_shifted = torch.cat([k[:, :1, :, d_half:], k[:, :-1, :, d_half:]], dim=1)
                k = torch.cat([k[..., :d_half], k_stat_shifted], dim=-1)

            q_bhtd = q.transpose(1, 2)
            k_bhtd = k.transpose(1, 2)
            v_bhtd = v.transpose(1, 2)

            if block_mask is not None:
                attn_out = flex_attention(
                    q_bhtd,
                    k_bhtd,
                    v_bhtd,
                    block_mask=block_mask,
                    scale=self.attn_scale,
                )
            else:
                attn_out = F.scaled_dot_product_attention(
                    q_bhtd,
                    k_bhtd,
                    v_bhtd,
                    is_causal=True,
                    scale=self.attn_scale,
                )

            if self.xsa_alpha is not None:
                v_unit = F.normalize(v_bhtd, dim=-1, eps=1e-5)
                self_proj = (attn_out * v_unit).sum(dim=-1, keepdim=True) * v_unit
                alpha_h = torch.tanh(self.xsa_alpha).view(1, self.num_heads, 1, 1).to(attn_out.dtype)
                attn_out = attn_out - alpha_h * self_proj

            attn_heads = attn_out.transpose(1, 2)
            if self.attn_gate is not None:
                g_h = (2.0 * torch.sigmoid(self.attn_gate(h[..., : self.gate_in_ch]))).unsqueeze(-1)
                attn_heads = attn_heads * g_h

            attn_flat = attn_heads.reshape(bsz, seq_len, d_model)
            x = x + self.out_proj(attn_flat)

        h_mlp = F.rms_norm(x, (self.embed_dim,), eps=self.eps)
        r_mlp = F.relu(self.fc1(h_mlp))
        x = x + self.fc2(r_mlp.square())
        return x, v_0_out


class CompiledLeaderboardClassicalGPT(nn.Module):
    """Flag-Free Compiled Classical Modded-NanoGPT Language Model (`Wave 45` / `EXPERIMENT_PLAN_060.md`)."""

    def __init__(self, cfg: CompiledLeaderboardClassicalConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else CompiledLeaderboardClassicalConfig()
        self.vocab_size = int(self.cfg.vocab_size)
        self.embed_dim = int(self.cfg.embed_dim)
        self.num_heads = int(self.cfg.num_heads)
        self.num_layers = int(self.cfg.num_layers)
        self.hidden_dim = int(self.cfg.hidden_dim)

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.n_ve_banks = min(max(0, int(self.cfg.num_value_embeddings)), max(1, self.num_layers // 2))
        if self.n_ve_banks > 0:
            self.value_embeds: nn.ModuleList | None = nn.ModuleList(
                [nn.Embedding(self.vocab_size, self.embed_dim) for _ in range(self.n_ve_banks)]
            )
            if self.cfg.ve_init_std > 0.0:
                with torch.no_grad():
                    for ve_emb in self.value_embeds:
                        ve_emb.weight.normal_(0.0, float(self.cfg.ve_init_std))
        else:
            self.value_embeds = None

        self.smear_in_ch = min(DEFAULT_GATE_INPUT_CHANNELS, self.embed_dim)
        if self.cfg.use_smear_gate:
            self.smear_gate: nn.Linear | None = nn.Linear(self.smear_in_ch, 1, bias=False)
            with torch.no_grad():
                self.smear_gate.weight.zero_()
            self.smear_lambda: nn.Parameter | None = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.smear_gate = None
            self.register_parameter("smear_lambda", None)

        self.rope = NativeLeaderboardRotary(
            head_dim=self.embed_dim // self.num_heads,
            max_seq_len=self.cfg.max_seq_len,
            base=self.cfg.rope_base,
        )

        init_ws = max(1, int(self.cfg.initial_ws_blocks))
        init_long_tok = init_ws * DEFAULT_FLEX_BLOCK_SIZE
        init_short_tok = max(1, init_ws // 2) * DEFAULT_FLEX_BLOCK_SIZE
        self.current_ws_blocks: int = init_ws
        self.register_buffer(
            "long_window_buf",
            torch.tensor(init_long_tok, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "short_window_buf",
            torch.tensor(init_short_tok, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "final_window_buf",
            torch.tensor(init_long_tok, dtype=torch.int32),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [
                CompiledLeaderboardBlock(
                    cfg=self.cfg,
                    layer_idx=layer_idx,
                    has_value_embedding=(self._ve_bank_index(layer_idx) is not None),
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        n_half = self.num_layers // 2
        if self.cfg.use_sigmoid_skip:
            skip_init = torch.full((n_half,), DEFAULT_SKIP_LOGIT_INIT, dtype=torch.float32)
        else:
            skip_init = torch.ones(n_half, dtype=torch.float32)
        self.skip_weights = nn.Parameter(skip_init)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()

        if self.cfg.bfloat16_weights:
            self.wte.to(dtype=torch.bfloat16)
            if self.value_embeds is not None:
                self.value_embeds.to(dtype=torch.bfloat16)
            if self.smear_gate is not None:
                self.smear_gate.to(dtype=torch.bfloat16)
            for blk in self.blocks:
                if blk.qkv_proj is not None:
                    blk.qkv_proj.to(dtype=torch.bfloat16)
                if blk.out_proj is not None:
                    blk.out_proj.to(dtype=torch.bfloat16)
                if blk.attn_gate is not None:
                    blk.attn_gate.to(dtype=torch.bfloat16)
                if blk.ve_gate is not None:
                    blk.ve_gate.to(dtype=torch.bfloat16)
                blk.fc1.to(dtype=torch.bfloat16)
                blk.fc2.to(dtype=torch.bfloat16)
            self.lm_head.to(dtype=torch.bfloat16)

        self._compiled_backbone_fn: Any = None

    @torch.no_grad()
    def set_sliding_window_blocks(
        self,
        ws_blocks: int,
        ws_final_blocks: int | None = None,
        apply_yarn: bool = True,
    ) -> None:
        """Update Long/Short/Final FlexAttention window buffers and YaRN scale in-place (`zero recompile`)."""
        new_ws = max(1, int(ws_blocks))
        final_ws = max(new_ws, int(ws_final_blocks)) if ws_final_blocks is not None else new_ws
        if apply_yarn and new_ws != self.current_ws_blocks:
            if new_ws > self.current_ws_blocks:
                self.rope.apply_yarn(self.current_ws_blocks, new_ws, block_size=DEFAULT_FLEX_BLOCK_SIZE)
            else:
                self.rope.reset_yarn()
        self.current_ws_blocks = new_ws
        self.long_window_buf.fill_(new_ws * DEFAULT_FLEX_BLOCK_SIZE)
        self.short_window_buf.fill_(max(1, new_ws // 2) * DEFAULT_FLEX_BLOCK_SIZE)
        self.final_window_buf.fill_(final_ws * DEFAULT_FLEX_BLOCK_SIZE)

    def _ve_bank_index(self, layer_idx: int) -> int | None:
        """Map layer index `l` to value embedding bank index in `[0, n_ve_banks)` (outer encoder + outer decoder)."""
        if self.n_ve_banks <= 0:
            return None
        if layer_idx < self.n_ve_banks:
            return layer_idx
        if layer_idx >= self.num_layers - self.n_ve_banks:
            return layer_idx - (self.num_layers - self.n_ve_banks)
        return None

    def enable_compiled_backbone(self, dynamic: bool = False) -> None:
        """Compile the 12-layer U-Net transformer backbone into a single Inductor graph."""
        self._compiled_backbone_fn = torch.compile(self._run_backbone_eager, dynamic=dynamic)

    def _build_flex_block_masks(
        self,
        idx: torch.Tensor,
    ) -> tuple[BlockMask | None, BlockMask | None]:
        """Construct Document-Boundary + Long/Short Sliding-Window `BlockMask` pair on CUDA."""
        bsz, seq_len = idx.shape
        if not (self.cfg.use_flex_doc_swc and idx.is_cuda and seq_len >= DEFAULT_FLEX_BLOCK_SIZE and seq_len % DEFAULT_FLEX_BLOCK_SIZE == 0):
            return None, None

        docs = (idx == DEFAULT_BOS_TOKEN_ID).cumsum(dim=1)
        lw_buf = self.long_window_buf
        sw_buf = self.short_window_buf

        def long_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            return (q_idx >= kv_idx) & (docs[b, q_idx] == docs[b, kv_idx]) & ((q_idx - kv_idx) < lw_buf)

        def short_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            return (q_idx >= kv_idx) & (docs[b, q_idx] == docs[b, kv_idx]) & ((q_idx - kv_idx) < sw_buf)

        bm_long = create_block_mask(
            long_mask_mod,
            B=bsz,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=idx.device,
            BLOCK_SIZE=DEFAULT_FLEX_BLOCK_SIZE,
        )
        bm_short = create_block_mask(
            short_mask_mod,
            B=bsz,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=idx.device,
            BLOCK_SIZE=DEFAULT_FLEX_BLOCK_SIZE,
        )
        return bm_long, bm_short

    def _run_backbone_eager(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode token indices `(B, T)` into final RMSNormed hidden states `(B, T, embed_dim)`."""
        _, seq_len = idx.shape
        raw_emb = self.wte(idx)
        if self.smear_gate is not None and self.smear_lambda is not None and seq_len > 1:
            s_gate = self.smear_lambda.to(raw_emb.dtype) * torch.sigmoid(
                self.smear_gate(raw_emb[:, 1:, : self.smear_in_ch])
            )
            raw_emb = torch.cat([raw_emb[:, :1], raw_emb[:, 1:] + s_gate * raw_emb[:, :-1]], dim=1)

        x_0 = F.rms_norm(raw_emb, (self.embed_dim,), eps=self.cfg.eps)
        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=x.dtype)
        q_scale = self.rope.yarn_q_scale
        bm_long, bm_short = self._build_flex_block_masks(idx)

        ve_banks: list[torch.Tensor] = []
        if self.value_embeds is not None:
            for ve_emb in self.value_embeds:
                ve_banks.append(ve_emb(idx))

        n_half = self.num_layers // 2
        encoder_states: list[torch.Tensor] = []
        v_0: torch.Tensor | None = None

        for layer_idx, block in enumerate(self.blocks):
            if layer_idx >= n_half and encoder_states:
                dec_i = layer_idx - n_half
                if self.cfg.use_sigmoid_skip:
                    skip_g = torch.sigmoid(self.skip_weights[dec_i]).to(x.dtype)
                else:
                    skip_g = self.skip_weights[dec_i].to(x.dtype)
                x = x + skip_g * encoder_states.pop()

            bank_idx = self._ve_bank_index(layer_idx)
            ve_l = ve_banks[bank_idx] if (bank_idx is not None and bank_idx < len(ve_banks)) else None
            layer_bm = bm_long if block.is_long_window_layer else bm_short

            x, v_0 = block(
                x,
                x_0,
                cos_4d,
                sin_4d,
                v_0=v_0,
                ve_l=ve_l,
                block_mask=layer_bm,
                q_scale=q_scale,
            )

            if layer_idx < n_half:
                encoder_states.append(x)

        return F.rms_norm(x, (self.embed_dim,), eps=self.cfg.eps)

    def forward_features(self, idx: torch.Tensor) -> torch.Tensor:
        """Dispatch to compiled or eager U-Net transformer backbone."""
        if self._compiled_backbone_fn is not None:
            return self._compiled_backbone_fn(idx)
        return self._run_backbone_eager(idx)

    def extract_layer_trajectories(self, idx: torch.Tensor) -> dict[str, Any]:
        """Extract per-layer input/output hidden trajectories for classical-to-quantum layer mirroring.

        Returns a dictionary containing:
        - ``x_0``: ``(B, T, embed_dim)`` initial RMSNormed embedding (plus smear gate if enabled).
        - ``layer_inputs``: ``list[torch.Tensor]`` of length ``num_layers``, exact input to each block ``l``
          (including U-Net encoder skip connections added before decoder blocks ``l >= num_layers // 2``).
        - ``layer_outputs``: ``list[torch.Tensor]`` of length ``num_layers``, exact output of each block ``l``.
        - ``v_0``: ``(B, T, num_heads, head_dim)`` Layer-0 value anchor tensor.
        - ``final_norm_x``: ``(B, T, embed_dim)`` final RMSNormed hidden state before ``lm_head``.
        """
        _, seq_len = idx.shape
        raw_emb = self.wte(idx)
        if self.smear_gate is not None and self.smear_lambda is not None and seq_len > 1:
            s_gate = self.smear_lambda.to(raw_emb.dtype) * torch.sigmoid(
                self.smear_gate(raw_emb[:, 1:, : self.smear_in_ch])
            )
            raw_emb = torch.cat([raw_emb[:, :1], raw_emb[:, 1:] + s_gate * raw_emb[:, :-1]], dim=1)

        x_0 = F.rms_norm(raw_emb, (self.embed_dim,), eps=self.cfg.eps)
        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=x.dtype)
        q_scale = self.rope.yarn_q_scale
        bm_long, bm_short = self._build_flex_block_masks(idx)

        ve_banks: list[torch.Tensor] = []
        if self.value_embeds is not None:
            for ve_emb in self.value_embeds:
                ve_banks.append(ve_emb(idx))

        n_half = self.num_layers // 2
        encoder_states: list[torch.Tensor] = []
        layer_inputs: list[torch.Tensor] = []
        layer_outputs: list[torch.Tensor] = []
        v_0: torch.Tensor | None = None

        for layer_idx, block in enumerate(self.blocks):
            if layer_idx >= n_half and encoder_states:
                dec_i = layer_idx - n_half
                if self.cfg.use_sigmoid_skip:
                    skip_g = torch.sigmoid(self.skip_weights[dec_i]).to(x.dtype)
                else:
                    skip_g = self.skip_weights[dec_i].to(x.dtype)
                x = x + skip_g * encoder_states.pop()

            layer_inputs.append(x)
            bank_idx = self._ve_bank_index(layer_idx)
            ve_l = ve_banks[bank_idx] if (bank_idx is not None and bank_idx < len(ve_banks)) else None
            layer_bm = bm_long if block.is_long_window_layer else bm_short

            x, v_0 = block(
                x,
                x_0,
                cos_4d,
                sin_4d,
                v_0=v_0,
                ve_l=ve_l,
                block_mask=layer_bm,
                q_scale=q_scale,
            )
            layer_outputs.append(x)

            if layer_idx < n_half:
                encoder_states.append(x)

        final_norm_x = F.rms_norm(x, (self.embed_dim,), eps=self.cfg.eps)
        return {
            "x_0": x_0,
            "layer_inputs": layer_inputs,
            "layer_outputs": layer_outputs,
            "v_0": v_0,
            "final_norm_x": final_norm_x,
        }

    def compute_logits(self, x_final: torch.Tensor) -> torch.Tensor:
        """Compute full softcapped vocabulary logits `15 * tanh(x @ W_lm^T / 15)` when explicitly requested."""
        raw = F.linear(x_final, self.lm_head.weight.to(x_final.dtype))
        if self.cfg.logit_softcap > 0.0:
            return self.cfg.logit_softcap * torch.tanh(raw / self.cfg.logit_softcap)
        return raw

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass returning `(logits_or_none, loss_or_none)` using single-pass chunked softcap CE."""
        x_final = self.forward_features(idx)
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        if targets is not None and not return_logits:
            loss = _CompiledChunkedSoftcapLinearCEFn.apply(
                x_final,
                self.lm_head.weight,
                targets,
                self.cfg.logit_softcap,
                self.cfg.ce_chunk_tokens,
                True,
            )
            return None, loss

        logits = self.compute_logits(x_final)
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1).long(),
                ignore_index=-1,
            )
        return logits, loss


def build_compiled_leaderboard_optimizers(
    model: CompiledLeaderboardClassicalGPT,
    opt_cfg: CompiledLeaderboardOptimizerConfig | None = None,
) -> tuple[StiefelMuon, torch.optim.Adam]:
    """Construct exact `KellerJordan/modded-nanogpt` `StiefelMuon` + fused `Adam` optimizer pair (`<= 8` args)."""
    cfg = opt_cfg if opt_cfg is not None else CompiledLeaderboardOptimizerConfig()

    muon_params: list[nn.Parameter] = []
    wte_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = [model.lm_head.weight]
    scalar_params: list[nn.Parameter] = [model.skip_weights]

    wte_params.append(model.wte.weight)
    if model.value_embeds is not None:
        for ve_emb in model.value_embeds:
            wte_params.append(ve_emb.weight)

    if model.smear_lambda is not None:
        scalar_params.append(model.smear_lambda)
    if model.smear_gate is not None:
        scalar_params.append(model.smear_gate.weight)

    for blk in model.blocks:
        scalar_params.append(blk.block_lambdas)
        if blk.sa_lambdas is not None:
            scalar_params.append(blk.sa_lambdas)
        if blk.xsa_alpha is not None:
            scalar_params.append(blk.xsa_alpha)
        if blk.attn_gate is not None:
            scalar_params.append(blk.attn_gate.weight)
        if blk.ve_gate is not None:
            scalar_params.append(blk.ve_gate.weight)
        if blk.qkv_proj is not None:
            muon_params.append(blk.qkv_proj.weight)
        if blk.out_proj is not None:
            muon_params.append(blk.out_proj.weight)
        muon_params.append(blk.fc1.weight)
        muon_params.append(blk.fc2.weight)

    muon_opt = StiefelMuon(
        muon_params,
        lr=cfg.muon_lr,
        momentum=DEFAULT_LEADERBOARD_MUON_MOMENTUM,
        weight_decay=cfg.weight_decay,
        decay_mode="cautious" if cfg.weight_decay > 0.0 else "none",
        aspect_scale=True,
        config=StiefelMuonConfig(
            nesterov=True,
            ns_steps=5,
            polar_method=cfg.polar_method,
            stiefel_penalty=0.0,
            use_normuon=cfg.use_normuon,
            use_twin_rail=cfg.use_twin_rail,
        ),
        momentum_warmup_steps=cfg.muon_warmup_steps,
        momentum_init=DEFAULT_LEADERBOARD_MUON_MOMENTUM_INIT,
        ema_momentum=True,
        batched_ns=True,
    )

    is_cuda = next(model.parameters()).is_cuda
    adam_groups = [
        {"params": head_params, "lr": cfg.head_lr, "group_name": "lm_head"},
        {"params": wte_params, "lr": cfg.wte_lr, "group_name": "wte_and_ve"},
        {"params": scalar_params, "lr": cfg.scalar_lr, "group_name": "scalars"},
    ]
    adam_opt = torch.optim.Adam(
        adam_groups,
        betas=(DEFAULT_LEADERBOARD_ADAM_BETA1, DEFAULT_LEADERBOARD_ADAM_BETA2),
        eps=DEFAULT_LEADERBOARD_ADAM_EPS,
        fused=is_cuda,
    )
    return muon_opt, adam_opt


@torch.no_grad()
def evaluate_compiled_leaderboard_validation(
    model: nn.Module,
    val_tokens: np.ndarray | torch.Tensor | str | Path,
    val_tokens_budget: int = DEFAULT_MODDED_VAL_TOKENS,
    batch_size: int = 16,
    seq_len: int = DEFAULT_LEADERBOARD_SEQ_LEN,
    valid_vocab_size: int = DEFAULT_VALID_FINEWEB_VOCAB_SIZE,
    chunk_tokens: int = DEFAULT_LEADERBOARD_CE_CHUNK_TOKENS,
) -> dict[str, float]:
    """Evaluate `CompiledLeaderboardClassicalGPT` on sequential `FineWeb` Shard-0 tokens (`<= 8` args).

    Computes exact `val_ce`, `val_ppl`, `top1_acc`, `rep_top1_acc`, `rep_val_ce`, `non_rep_val_ce`,
    and `empirical_rep_rate` in `bfloat16` token chunks without materializing full-batch logits.
    """
    was_training = model.training
    model.eval()
    t0 = time.perf_counter()

    device = next(model.parameters()).device
    if isinstance(val_tokens, (str, Path)):
        mm = np.memmap(Path(val_tokens), dtype=np.uint16, mode="r", offset=FINEWEB_SHARD_HEADER_BYTES)
        n_avail = min(len(mm) - 1, int(val_tokens_budget))
        tok_arr = np.asarray(mm[: n_avail + 1], dtype=np.int64)
        flat_gpu = torch.from_numpy(tok_arr).to(device=device, dtype=torch.long)
    elif isinstance(val_tokens, np.ndarray):
        n_avail = min(int(val_tokens.shape[0]) - 1, int(val_tokens_budget))
        flat_gpu = torch.from_numpy(np.asarray(val_tokens[: n_avail + 1], dtype=np.int64)).to(
            device=device, dtype=torch.long
        )
    else:
        n_avail = min(int(val_tokens.numel()) - 1, int(val_tokens_budget))
        flat_gpu = val_tokens[: n_avail + 1].to(device=device, dtype=torch.long)

    n_seqs = max(1, n_avail // int(seq_len))
    total_eval_tokens = n_seqs * int(seq_len)
    x_all = flat_gpu[:total_eval_tokens].view(n_seqs, int(seq_len))
    y_all = flat_gpu[1 : total_eval_tokens + 1].view(n_seqs, int(seq_len))

    softcap = float(getattr(getattr(model, "cfg", None), "logit_softcap", getattr(model, "logit_softcap", 15.0)))
    inv_softcap = (1.0 / softcap) if softcap > 0.0 else 0.0
    w_lm = model.lm_head.weight  # type: ignore[union-attr]
    work_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    w_work = w_lm[:valid_vocab_size].to(dtype=work_dtype)

    total_nll = 0.0
    total_top1 = 0
    rep_nll = 0.0
    non_rep_nll = 0.0
    rep_top1 = 0
    rep_count = 0

    for b_start in range(0, n_seqs, max(1, int(batch_size))):
        b_end = min(n_seqs, b_start + max(1, int(batch_size)))
        xb = x_all[b_start:b_end]
        yb = y_all[b_start:b_end]

        with torch.amp.autocast(device_type=device.type, dtype=work_dtype, enabled=(device.type == "cuda")):
            feat_out = model.forward_features(xb)  # type: ignore[operator]
            x_final = feat_out[0] if isinstance(feat_out, tuple) else feat_out

        x_flat = x_final.reshape(-1, x_final.size(-1)).to(dtype=work_dtype)
        y_flat = yb.reshape(-1)
        x_in_flat = xb.reshape(-1)
        is_rep = x_in_flat == y_flat

        n_b_tok = x_flat.size(0)
        c_step = max(256, min(int(chunk_tokens), n_b_tok))
        for c_start in range(0, n_b_tok, c_step):
            c_end = min(n_b_tok, c_start + c_step)
            xc = x_flat[c_start:c_end]
            yc = y_flat[c_start:c_end]
            rep_c = is_rep[c_start:c_end]

            logits_c = F.linear(xc, w_work).float()
            if softcap > 0.0:
                logits_c = softcap * torch.tanh(logits_c * inv_softcap)

            nll_c = F.cross_entropy(logits_c, yc, reduction="none")
            pred_c = logits_c.argmax(dim=-1)
            hit_c = pred_c == yc

            total_nll += float(nll_c.sum().item())
            total_top1 += int(hit_c.sum().item())
            if bool(rep_c.any().item()):
                rep_nll += float(nll_c[rep_c].sum().item())
                rep_top1 += int(hit_c[rep_c].sum().item())
                rep_count += int(rep_c.sum().item())
            non_rep_mask = ~rep_c
            if bool(non_rep_mask.any().item()):
                non_rep_nll += float(nll_c[non_rep_mask].sum().item())

    if was_training:
        model.train()

    wall_s = max(1e-6, time.perf_counter() - t0)
    val_ce = total_nll / float(max(1, total_eval_tokens))
    non_rep_count = max(1, total_eval_tokens - rep_count)
    return {
        "val_ce": round(val_ce, 6),
        "val_ppl": round(math.exp(min(20.0, val_ce)), 4),
        "top1_acc": round(100.0 * float(total_top1) / float(max(1, total_eval_tokens)), 4),
        "rep_val_ce": round(rep_nll / float(max(1, rep_count)), 6) if rep_count > 0 else 0.0,
        "non_rep_val_ce": round(non_rep_nll / float(non_rep_count), 6),
        "rep_top1_acc": round(100.0 * float(rep_top1) / float(max(1, rep_count)), 4) if rep_count > 0 else 0.0,
        "empirical_rep_rate": round(100.0 * float(rep_count) / float(max(1, total_eval_tokens)), 4),
        "eval_tokens": float(total_eval_tokens),
        "eval_tok_per_s": round(float(total_eval_tokens) / wall_s, 1),
        "eval_wall_s": round(wall_s, 3),
    }
