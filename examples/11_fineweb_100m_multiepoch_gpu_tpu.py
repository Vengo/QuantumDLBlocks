"""Example 11: Full 100M-Token Multi-Epoch FineWeb Quantum Slowrun Benchmark (GPU & TPU).

Benchmarks the 10-Qubit (`d = 1024 = 2^10`, ~70.68M parameters) Next-Gen Slowrun
`QuantumBornGPT` architecture over multiple full passes of the 100,000,000-token
FineWeb training shard (`fineweb_train_000001.bin`, `299,827,200` total tokens across
3 full epochs at `65,536` tokens/step) on both:

1. **CUDA GPU (`--backend cuda`)**:
   - Fused Triton `flash_born_attention` kernel (64x64 Tensor Core MMA tiles + atomic-free
     SRAM-accumulated backward `dQ` and `dK/dV` kernels, zero `[B, H, T, T]` materialization)
   - `torch.compile` (TorchInductor fusion across Isometric Hilbert Lift, Temporal Bigram
     Dispersion, 2-Tap Causal QK Wave Filter, Unitary RoPE, `TwoBeamQuantumInterference` MLP,
     Mach-Zehnder U-Net Layer-Skip Interferometry, and Stiefel-Tied POVM Readout)
   - Compiled 5-step Newton-Schulz `StiefelMuon` polar factor + BFloat16 Riemannian Stiefel
     retraction (`retract_every=4`) + Checkpointed BFloat16 Stiefel-POVM Cross-Entropy.

2. **Cloud TPU (`--backend tpu`)**:
   - Pure JAX/XLA `@jax.jit(donate_argnums=(0, 1))` whole-step compilation (fusing forward,
     backward, global gradient norm clipping, 5-step BFloat16 MXU Newton-Schulz `StiefelMuon`,
     Continuous Riemannian Stiefel Penalty, and `AdamW` into a single XLA HLO executable)
   - Rematerialized Fused MXU Born Attention (`@jax.checkpoint`) + Scatter-Free Factorized
     MXU Stiefel-POVM Cross-Entropy (`L_i = logsumexp(beta * z_i @ M^T) - beta * <z_i, M_{y_i}>`).
"""

from __future__ import annotations

import argparse
from functools import partial
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "tpu"),
        default="auto",
        help="Accelerator backend to run ('auto' detects CUDA GPU or JAX TPU).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of full 100M-token epochs to train (default: 3 = 300M total tokens).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/data/fineweb10B"),
        help="Directory containing train_full_100M.bin and val_slice.bin.",
    )
    return parser.parse_args()


def run_cuda_benchmark(num_epochs: int = 3, data_dir: Path = Path("/content/data/fineweb10B")) -> list[dict[str, float]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    from qdlblocks.functional import l2_normalize, zeropower_via_newtonschulz5
    from qdlblocks.models.born_gpt import QuantumBornGPT
    import qdlblocks.optim.muon as _muon_mod
    from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    _muon_mod.zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="default")

    def _chunk_ce_fn(zc: torch.Tensor, M_seed: torch.Tensor, inv_t: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
        logits_c = F.linear(zc, M_seed) * inv_t.to(zc.dtype)
        return F.cross_entropy(logits_c, yc)

    def forward_backbone(m: QuantumBornGPT, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        psi_0 = m.hilbert_lift(idx)
        if m.training and m.depolarizing_jitter > 0.0:
            noise = torch.randn_like(psi_0) * (m.depolarizing_jitter / math.sqrt(m.embed_dim))
            psi_0 = l2_normalize(psi_0 + noise, dim=-1, eps=m.eps)
        if m.use_bigram_smear and T > 1 and m.smear_embed_gate is not None:
            psi_prev = F.pad(psi_0[:, :-1, :], (0, 0, 1, 0))
            eta = (0.06 * torch.tanh(m.smear_embed_gate)).to(psi_0.dtype)
            psi_0 = l2_normalize(psi_0 + eta * psi_prev, dim=-1, eps=m.eps)

        psi = psi_0
        encoder_states: list[torch.Tensor] = []
        n_half = m.num_layers // 2
        cos, sin = m.rope.get_cos_sin(T)
        for layer_idx, block in enumerate(m.blocks):
            psi = block(psi, cos=cos, sin=sin, psi_0=psi_0)
            if m.use_unet_interferometry and m.unet_gates is not None and m.unet_scales is not None:
                if layer_idx < n_half:
                    encoder_states.append(psi)
                elif layer_idx >= m.num_layers - n_half:
                    enc_idx = m.num_layers - 1 - layer_idx
                    skip_psi = encoder_states[enc_idx]
                    u_scale = (0.35 * torch.tanh(m.unet_scales[enc_idx])).to(psi.dtype)
                    skip_rot = F.linear(skip_psi, m.unet_gates[enc_idx].to(psi.dtype))
                    psi = l2_normalize(
                        psi + u_scale * l2_normalize(skip_rot, dim=-1, eps=m.eps),
                        dim=-1,
                        eps=m.eps,
                    )

        psi_final = l2_normalize(m.readout_rot(psi), dim=-1, eps=m.eps)
        phi_seed = m.hilbert_lift.collapse_to_seed(psi_final).reshape(-1, m.seed_dim)
        M_seed = l2_normalize(m.wte.weight + m.povm_head_seed_delta, dim=1, eps=m.eps).to(phi_seed.dtype)
        inv_t = torch.clamp(m.tied_inv_temp, min=1.0, max=60.0)
        return phi_seed, M_seed, inv_t

    compiled_backbone = torch.compile(forward_backbone, mode="default")
    compiled_chunk_ce = torch.compile(_chunk_ce_fn, mode="default")

    def compute_step_loss(m: QuantumBornGPT, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        phi_seed, M_seed, inv_t = compiled_backbone(m, idx)
        flat_y = targets.reshape(-1)
        chunk_sz = 16384
        n_chunks = phi_seed.size(0) // chunk_sz
        total_loss = torch.zeros((), device=phi_seed.device, dtype=torch.float32)
        for i in range(n_chunks):
            zc = phi_seed[i * chunk_sz : (i + 1) * chunk_sz]
            yc = flat_y[i * chunk_sz : (i + 1) * chunk_sz]
            total_loss = total_loss + checkpoint(compiled_chunk_ce, zc, M_seed, inv_t, yc, use_reentrant=False)
        return total_loss / n_chunks

    @torch.no_grad()
    def compute_isometry_error(model: nn.Module) -> float:
        errs: list[float] = []
        for name, p in model.named_parameters():
            if p.ndim == 2 and not any(k in name for k in ("wte", "povm_head", "embed", "unet_scales", "smear")):
                w = p.detach().float()
                r, c = w.shape
                if r >= c:
                    gram = w.t() @ w
                    eye = torch.eye(c, device=w.device, dtype=torch.float32)
                    err = torch.norm(gram - eye, p="fro") / math.sqrt(c)
                else:
                    gram = w @ w.t()
                    eye = torch.eye(r, device=w.device, dtype=torch.float32)
                    err = torch.norm(gram - eye, p="fro") / math.sqrt(r)
                errs.append(float(err.item()))
        return float(np.mean(errs)) if errs else 0.0

    @torch.no_grad()
    def evaluate_validation(
        model: QuantumBornGPT,
        val_tokens: np.ndarray,
        batch_size: int = 64,
        block_size: int = 512,
        num_batches: int = 10,
    ) -> dict[str, float]:
        model.eval()
        losses: list[float] = []
        top1_hits = 0
        top5_hits = 0
        total_toks = 0
        all_ranks: list[np.ndarray] = []

        stride = batch_size * block_size
        for b_idx in range(num_batches):
            base = b_idx * stride
            chunk = val_tokens[base : base + stride + 1].astype(np.int64)
            x_np = chunk[:-1].reshape(batch_size, block_size)
            y_np = chunk[1:].reshape(batch_size, block_size)
            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                phi_seed, M_seed, inv_t = compiled_backbone(model, x)
                flat_y = y.reshape(-1)
                chunk_sz = 8192
                for start in range(0, phi_seed.size(0), chunk_sz):
                    end = min(start + chunk_sz, phi_seed.size(0))
                    zc = phi_seed[start:end]
                    yc = flat_y[start:end]
                    logits_c = F.linear(zc, M_seed) * inv_t.to(zc.dtype)
                    loss_c = F.cross_entropy(logits_c, yc)
                    losses.append(float(loss_c.item()))

                    top5 = torch.topk(logits_c, k=5, dim=-1).indices
                    top1_hits += int((top5[:, 0] == yc).sum().item())
                    top5_hits += int((top5 == yc.unsqueeze(1)).any(dim=-1).sum().item())
                    total_toks += int(yc.numel())

                    if b_idx < 2 and start == 0:
                        sub_logits = logits_c[:1024]
                        sub_y = yc[:1024]
                        target_scores = sub_logits.gather(1, sub_y.unsqueeze(1))
                        ranks = (sub_logits > target_scores).sum(dim=-1) + 1
                        all_ranks.append(ranks.cpu().numpy())

        model.train()
        val_ce = float(np.mean(losses))
        val_ppl = float(math.exp(min(val_ce, 20.0)))
        med_rank = float(np.median(np.concatenate(all_ranks))) if all_ranks else -1.0
        iso_err = compute_isometry_error(model)
        return {
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "top1_acc": 100.0 * top1_hits / max(1, total_toks),
            "top5_acc": 100.0 * top5_hits / max(1, total_toks),
            "med_rank": med_rank,
            "iso_err": iso_err,
        }

    print("=" * 90, flush=True)
    print("FULL 100M-TOKEN MULTI-EPOCH QUANTUM SLOWRUN BENCHMARK — COLAB GPU (BLACKWELL G4)", flush=True)
    print(
        f"Device: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | Fused Triton Flash-Born + torch.compile",
        flush=True,
    )
    print("=" * 90, flush=True)

    train_tokens = np.fromfile(data_dir / "train_full_100M.bin", dtype=np.uint16, offset=1024)
    val_tokens = np.fromfile(data_dir / "val_slice.bin", dtype=np.uint16, offset=1024)
    print(f"Loaded Train Shard: {len(train_tokens):,} tokens | Val Shard: {len(val_tokens):,} tokens", flush=True)

    torch.manual_seed(42)
    np.random.seed(42)

    B = 128
    T = 512
    tok_per_step = B * T
    steps_per_epoch = (len(train_tokens) - 1) // tok_per_step
    total_steps = steps_per_epoch * num_epochs
    eval_every = steps_per_epoch // 2

    model = QuantumBornGPT(
        vocab_size=50304,
        embed_dim=1024,
        seed_dim=128,
        num_heads=16,
        num_layers=6,
        hidden_dim=1536,
        block_size=T,
        readout_mode="tied_stiefel",
        learnable_temperature=True,
        value_residual_mixing=True,
        input_highway_mixing=True,
        use_bigram_smear=True,
        use_qk_wave_filter=True,
        use_unet_interferometry=True,
        use_two_beam_mlp=True,
        depolarizing_jitter=0.015,
        use_flash_born=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Architecture: 10-Qubit QuantumBornGPT (d=1024, H=16, L=6, d_ff=1536x2, d_seed=128) | Params: {n_params:,}",
        flush=True,
    )
    print(
        f"Schedule: {num_epochs} Full Epochs x {steps_per_epoch:,} steps/epoch = {total_steps:,} steps ({total_steps * tok_per_step:,} total tokens)",
        flush=True,
    )

    muon_params, adamw_params = split_muon_adamw_params(
        model, adamw_name_patterns=("wte", "povm_head", "embed", "unet_scales", "smear")
    )
    base_muon_lr = 0.025
    base_adamw_lr = 0.004
    opt_muon = StiefelMuon(
        muon_params,
        lr=base_muon_lr,
        momentum=0.95,
        weight_decay=0.01,
        decay_mode="cautious",
        stiefel_penalty=0.005,
        ns_steps=5,
        retract_every=4,
    )
    opt_adamw = torch.optim.AdamW(
        adamw_params, lr=base_adamw_lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True
    )

    print("Compiling model & StiefelMuon kernels via TorchInductor + Triton...", flush=True)
    t_comp0 = time.perf_counter()
    x_warm = torch.randint(0, 50257, (B, T), device=device, dtype=torch.long)
    y_warm = torch.randint(0, 50257, (B, T), device=device, dtype=torch.long)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_w = compute_step_loss(model, x_warm, y_warm)
    loss_w.backward()
    opt_muon.zero_grad(set_to_none=True)
    opt_adamw.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"Compilation complete in {time.perf_counter() - t_comp0:.2f}s", flush=True)

    torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, float]] = []

    val0 = evaluate_validation(model, val_tokens)
    print(
        f"[Epoch 0.00 | Step    0 |   0.00M tok] Val CE: {val0['val_ce']:.4f} | Val PPL: {val0['val_ppl']:8.1f} | "
        f"Top-1: {val0['top1_acc']:5.2f}% | Top-5: {val0['top5_acc']:5.2f}% | MedRank: {val0['med_rank']:5.0f} | IsoErr: {val0['iso_err']:.4f}",
        flush=True,
    )
    history.append({"epoch": 0.0, "step": 0, "tokens_M": 0.0, "train_ce": val0["val_ce"], "tok_per_s": 0.0, **val0})

    warmup_steps = 150
    running_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    running_steps = 0
    train_t0 = time.perf_counter()
    window_t0 = time.perf_counter()
    window_tokens = 0

    global_step = 0
    for ep in range(num_epochs):
        rng = np.random.default_rng(42 + ep)
        batch_order = rng.permutation(steps_per_epoch)

        for b_pos in range(steps_per_epoch):
            global_step += 1
            if global_step <= warmup_steps:
                lr_mult = global_step / float(warmup_steps)
            else:
                prog = (global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                lr_mult = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))

            for g in opt_muon.param_groups:
                g["lr"] = base_muon_lr * lr_mult
            for g in opt_adamw.param_groups:
                g["lr"] = base_adamw_lr * lr_mult

            chunk_idx = int(batch_order[b_pos])
            start_tok = chunk_idx * tok_per_step
            buf = train_tokens[start_tok : start_tok + tok_per_step + 1].astype(np.int64)
            x = torch.from_numpy(buf[:-1].reshape(B, T)).to(device, non_blocking=True)
            y = torch.from_numpy(buf[1:].reshape(B, T)).to(device, non_blocking=True)

            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compute_step_loss(model, x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adamw.step()

            running_loss_t += loss.detach().float()
            running_steps += 1
            window_tokens += tok_per_step

            if (global_step == eval_every + (ep * steps_per_epoch)) or (global_step == (ep + 1) * steps_per_epoch):
                torch.cuda.synchronize()
                dt_w = time.perf_counter() - window_t0
                tps = window_tokens / max(dt_w, 1e-6)
                train_ce = float(running_loss_t.item()) / max(1, running_steps)
                running_loss_t.zero_()
                running_steps = 0

                val_metrics = evaluate_validation(model, val_tokens)
                ep_frac = round((global_step / float(steps_per_epoch)) * 2.0) / 2.0
                tok_m = (global_step * tok_per_step) / 1e6
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"[Epoch {ep_frac:4.2f} | Step {global_step:4d} | {tok_m:6.2f}M tok] "
                    f"Train CE: {train_ce:.4f} | Val CE: {val_metrics['val_ce']:.4f} | Val PPL: {val_metrics['val_ppl']:6.1f} | "
                    f"Top-1: {val_metrics['top1_acc']:5.2f}% | Top-5: {val_metrics['top5_acc']:5.2f}% | "
                    f"MedRank: {val_metrics['med_rank']:4.0f} | IsoErr: {val_metrics['iso_err']:.4f} | "
                    f"{tps:,.0f} tok/s | VRAM: {peak_gb:.2f} GB",
                    flush=True,
                )
                history.append(
                    {
                        "epoch": ep_frac,
                        "step": global_step,
                        "tokens_M": tok_m,
                        "train_ce": train_ce,
                        "tok_per_s": tps,
                        "peak_vram_gb": peak_gb,
                        **val_metrics,
                    }
                )
                Path("/content/gpu_100M_results.json").write_text(json.dumps(history, indent=2))
                window_t0 = time.perf_counter()
                window_tokens = 0

    total_time = time.perf_counter() - train_t0
    print("=" * 90, flush=True)
    print(f"COMPLETED {num_epochs} FULL 100M-TOKEN EPOCHS ON GPU IN {total_time:.1f}s ({total_time/60:.2f} min)", flush=True)
    print("=" * 90, flush=True)
    return history


def run_tpu_benchmark(num_epochs: int = 3, data_dir: Path = Path("/content/data/fineweb10B")) -> list[dict[str, float]]:
    import jax
    import jax.numpy as jnp

    def l2_norm_bf16(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
        x_bf16 = x.astype(jnp.bfloat16)
        sq = jnp.sum(x_bf16 * x_bf16, axis=axis, keepdims=True, dtype=jnp.float32)
        return x_bf16 * jax.lax.rsqrt(jnp.maximum(sq, eps * eps)).astype(jnp.bfloat16)

    def qr_ortho(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        raw = jax.random.normal(key, shape, dtype=jnp.float32)
        m, n = shape[0], shape[1]
        if m >= n:
            q, _ = jnp.linalg.qr(raw)
            return q
        q_t, _ = jnp.linalg.qr(raw.T)
        return q_t.T

    def precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[jax.Array, jax.Array]:
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (jnp.arange(0, half, dtype=jnp.float32) / half))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        return (
            jnp.cos(freqs)[None, None, :, :].astype(jnp.bfloat16),
            jnp.sin(freqs)[None, None, :, :].astype(jnp.bfloat16),
        )

    def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def newton_schulz5_bf16(G: jax.Array, steps: int = 5) -> jax.Array:
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.astype(jnp.bfloat16)
        transposed = X.shape[0] > X.shape[1]
        if transposed:
            X = X.T
        norm = jnp.sqrt(jnp.sum(X * X, dtype=jnp.float32)) + 1e-7
        X = (X.astype(jnp.float32) / norm).astype(jnp.bfloat16)
        for _ in range(steps):
            A = jnp.matmul(X, X.T, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            B_mat = b * A + c * jnp.matmul(A, A, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            X = a * X + jnp.matmul(B_mat, X, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        if transposed:
            X = X.T
        return X.astype(jnp.float32)

    def stiefel_retract_grad(W: jax.Array) -> jax.Array:
        W_bf16 = W.astype(jnp.bfloat16)
        r, c = W.shape
        if r >= c:
            gram = jnp.matmul(W_bf16.T, W_bf16, preferred_element_type=jnp.float32)
            diff = gram - jnp.eye(c, dtype=jnp.float32)
            denom = 1.0 + (jnp.sqrt(jnp.sum(diff * diff)) / math.sqrt(c))
            grad = jnp.matmul(W_bf16, (diff / denom).astype(jnp.bfloat16), preferred_element_type=jnp.float32)
        else:
            gram = jnp.matmul(W_bf16, W_bf16.T, preferred_element_type=jnp.float32)
            diff = gram - jnp.eye(r, dtype=jnp.float32)
            denom = 1.0 + (jnp.sqrt(jnp.sum(diff * diff)) / math.sqrt(r))
            grad = jnp.matmul((diff / denom).astype(jnp.bfloat16), W_bf16, preferred_element_type=jnp.float32)
        return grad

    print("=" * 90, flush=True)
    print("FULL 100M-TOKEN MULTI-EPOCH QUANTUM SLOWRUN BENCHMARK — COLAB TPU (CLOUD TPU v5e-1)", flush=True)
    print(f"JAX {jax.__version__} | Backend: {jax.default_backend()} | Devices: {jax.devices()}", flush=True)
    print("=" * 90, flush=True)

    train_tokens = np.fromfile(data_dir / "train_full_100M.bin", dtype=np.uint16, offset=1024)
    val_tokens = np.fromfile(data_dir / "val_slice.bin", dtype=np.uint16, offset=1024)
    print(f"Loaded Train Shard: {len(train_tokens):,} tokens | Val Shard: {len(val_tokens):,} tokens", flush=True)

    V = 50304
    D_seed = 128
    D = 1024
    H = 16
    D_h = 64
    L = 6
    D_mlp = 1536
    T = 512
    B_micro = 64
    grad_accum = 2
    B = B_micro * grad_accum

    tok_per_step = B * T
    steps_per_epoch = (len(train_tokens) - 1) // tok_per_step
    total_steps = steps_per_epoch * num_epochs
    eval_every = steps_per_epoch // 2

    key = jax.random.PRNGKey(42)
    k_wte, k_lift, k_blocks = jax.random.split(key, 3)

    raw_wte = jax.random.normal(k_wte, (V, D_seed), dtype=jnp.float32) * (1.0 / math.sqrt(D_seed))
    layer_keys = jax.random.split(k_blocks, L)

    def init_block(k: jax.Array) -> dict[str, jax.Array]:
        kq, kk, kv, ko, ka, kb, kd = jax.random.split(k, 7)
        return {
            "w_q": qr_ortho(kq, (D, D)),
            "w_k": qr_ortho(kk, (D, D)),
            "w_v": qr_ortho(kv, (D, D)),
            "w_o": qr_ortho(ko, (D, D)),
            "w_a": qr_ortho(ka, (D, D_mlp)),
            "w_b": qr_ortho(kb, (D, D_mlp)),
            "w_down": qr_ortho(kd, (D_mlp, D)),
            "head_scale": jnp.full((1, H, 1, 1), math.sqrt(D_h), dtype=jnp.float32),
            "v_mix_logit": jnp.array(-1.1, dtype=jnp.float32),
            "q_prev_mix": jnp.zeros((1, H, 1, 1), dtype=jnp.float32),
            "k_prev_mix": jnp.zeros((1, H, 1, 1), dtype=jnp.float32),
            "highway_logit": jnp.array(-2.0, dtype=jnp.float32),
        }

    params = {
        "wte": raw_wte,
        "povm_delta": jnp.zeros((V, D_seed), dtype=jnp.float32),
        "w_lift": qr_ortho(k_lift, (D_seed, D)),
        "readout_rot": jnp.eye(D, dtype=jnp.float32),
        "smear_gate": jnp.zeros((1, 1, D), dtype=jnp.float32),
        "tied_inv_temp": jnp.array(1.0 / 0.035, dtype=jnp.float32),
        "unet_gates": jnp.stack([jnp.eye(D, dtype=jnp.float32) for _ in range(L // 2)]),
        "unet_scales": jnp.full((L // 2,), 0.18, dtype=jnp.float32),
        "blocks": jax.vmap(init_block)(layer_keys),
    }

    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
    print(
        f"Architecture: 10-Qubit QuantumBornGPT (d=1024, H=16, L=6, d_ff=1536x2, d_seed=128) | Params: {n_params:,}",
        flush=True,
    )
    print(
        f"Schedule: {num_epochs} Full Epochs x {steps_per_epoch:,} steps/epoch = {total_steps:,} steps ({total_steps * tok_per_step:,} total tokens)",
        flush=True,
    )

    cos_rope, sin_rope = precompute_rope(T, D_h)
    causal_mask_bf16 = jnp.tril(jnp.ones((T, T), dtype=jnp.bfloat16))[None, None, :, :]

    def encode_backbone(p: dict, x_idx: jax.Array, rng_key: jax.Array | None, is_training: bool) -> tuple[jax.Array, jax.Array, jax.Array]:
        B_sz, T_sz = x_idx.shape
        psi_seed = l2_norm_bf16(p["wte"][x_idx])
        psi_0 = l2_norm_bf16(jnp.matmul(psi_seed, p["w_lift"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
        if is_training and rng_key is not None:
            noise = jax.random.normal(rng_key, psi_0.shape, dtype=jnp.bfloat16) * jnp.bfloat16(0.015 / math.sqrt(D))
            psi_0 = l2_norm_bf16(psi_0 + noise)
        psi_prev = jnp.pad(psi_0[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
        eta = (0.06 * jnp.tanh(p["smear_gate"])).astype(jnp.bfloat16)
        psi_0 = l2_norm_bf16(psi_0 + eta * psi_prev)

        @jax.checkpoint
        def apply_one_block(psi_in: jax.Array, blk: dict) -> jax.Array:
            gamma = jax.nn.sigmoid(blk["highway_logit"]).astype(jnp.bfloat16)
            x_h = l2_norm_bf16((1.0 - gamma) * psi_in + gamma * psi_0)

            q = jnp.matmul(x_h, blk["w_q"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16).reshape(B_sz, T_sz, H, D_h).swapaxes(1, 2)
            k = jnp.matmul(x_h, blk["w_k"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16).reshape(B_sz, T_sz, H, D_h).swapaxes(1, 2)
            v = jnp.matmul(x_h, blk["w_v"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16).reshape(B_sz, T_sz, H, D_h).swapaxes(1, 2)

            v_0 = jnp.matmul(psi_0, blk["w_v"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16).reshape(B_sz, T_sz, H, D_h).swapaxes(1, 2)
            alpha_v = jax.nn.sigmoid(blk["v_mix_logit"]).astype(jnp.bfloat16)
            v = (1.0 - alpha_v) * v + alpha_v * v_0

            q_prev = jnp.pad(q[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0)))
            k_prev = jnp.pad(k[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0)))
            qm = (0.06 * jnp.tanh(blk["q_prev_mix"])).astype(jnp.bfloat16)
            km = (0.06 * jnp.tanh(blk["k_prev_mix"])).astype(jnp.bfloat16)
            q = q + qm * q_prev
            k = k + km * k_prev

            q = apply_rope(q, cos_rope, sin_rope)
            k = apply_rope(k, cos_rope, sin_rope)
            q, k = l2_norm_bf16(q), l2_norm_bf16(k)

            scale = jnp.clip(blk["head_scale"], 1.0, 40.0).astype(jnp.bfloat16)
            s = jnp.einsum("bhid,bhjd->bhij", q * scale, k, preferred_element_type=jnp.bfloat16)
            a_pos = jnp.maximum(s, jnp.bfloat16(0.0)) * causal_mask_bf16
            prob = a_pos * a_pos
            z_den = jnp.sum(prob, axis=-1, keepdims=True, dtype=jnp.float32) + 1e-6
            o_num = jnp.einsum("bhij,bhjd->bhid", prob, v, preferred_element_type=jnp.float32)
            attn_out = (o_num / z_den).astype(jnp.bfloat16).swapaxes(1, 2).reshape(B_sz, T_sz, D)
            attn_proj = jnp.matmul(attn_out, blk["w_o"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16)
            x_mid = l2_norm_bf16(x_h + attn_proj)

            a_beam = jnp.matmul(x_mid, blk["w_a"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16)
            b_beam = l2_norm_bf16(jnp.matmul(x_mid, blk["w_b"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            interfered = a_beam * b_beam * jnp.bfloat16(math.sqrt(D_mlp))
            mlp_out = l2_norm_bf16(jnp.matmul(interfered, blk["w_down"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            return l2_norm_bf16(x_mid + mlp_out)

        enc_blks = jax.tree_util.tree_map(lambda x: x[: L // 2], p["blocks"])
        dec_blks = jax.tree_util.tree_map(lambda x: x[L // 2 :], p["blocks"])

        def scan_enc(psi_c, blk):
            psi_next = apply_one_block(psi_c, blk)
            return psi_next, psi_next

        psi, enc_skips = jax.lax.scan(scan_enc, psi_0, enc_blks)
        rev_skips = jnp.flip(enc_skips, axis=0)
        rev_gates = jnp.flip(p["unet_gates"], axis=0)
        rev_scales = jnp.flip(p["unet_scales"], axis=0)

        def scan_dec(psi_c, dec_in):
            blk, skip_psi, u_gate, u_scale = dec_in
            psi_next = apply_one_block(psi_c, blk)
            us = (0.35 * jnp.tanh(u_scale)).astype(jnp.bfloat16)
            skip_rot = l2_norm_bf16(jnp.matmul(skip_psi, u_gate.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
            psi_next = l2_norm_bf16(psi_next + us * skip_rot)
            return psi_next, None

        psi, _ = jax.lax.scan(scan_dec, psi, (dec_blks, rev_skips, rev_gates, rev_scales))
        psi_final = l2_norm_bf16(jnp.matmul(psi, p["readout_rot"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16))
        phi_seed = l2_norm_bf16(jnp.matmul(psi_final, p["w_lift"].T.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16)).reshape(-1, D_seed)
        M_seed = l2_norm_bf16(p["wte"] + p["povm_delta"])
        return phi_seed, M_seed, p["tied_inv_temp"]

    def microbatch_loss(p: dict, x_mb: jax.Array, y_mb: jax.Array, rng_key: jax.Array) -> jax.Array:
        phi_seed, M_seed, inv_t = encode_backbone(p, x_mb, rng_key, True)
        N = phi_seed.shape[0]
        num_chunks = 4
        chunk_sz = N // num_chunks
        phi_c = phi_seed.reshape(num_chunks, chunk_sz, D_seed)
        beta_f = jnp.clip(inv_t.astype(jnp.float32), 1.0, 60.0)
        beta_bf16 = beta_f.astype(jnp.bfloat16)

        flat_y = y_mb.reshape(-1)
        target_M = M_seed[flat_y]
        target_logits = jnp.sum(phi_seed.astype(jnp.float32) * target_M.astype(jnp.float32), axis=-1) * beta_f
        mean_target = jnp.mean(target_logits)

        @jax.checkpoint
        def lse_chunk(carry, zc):
            logits = jnp.matmul(zc, M_seed.T, preferred_element_type=jnp.bfloat16) * beta_bf16
            max_l = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
            shifted = (logits - max_l).astype(jnp.float32)
            lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1)) + jnp.squeeze(max_l.astype(jnp.float32), axis=-1)
            return carry + jnp.sum(lse), None

        tot_lse, _ = jax.lax.scan(lse_chunk, jnp.zeros((), dtype=jnp.float32), phi_c)
        return (tot_lse / float(N)) - mean_target

    muon_block_keys = ("w_q", "w_k", "w_v", "w_o", "w_a", "w_b", "w_down")
    muon_top_keys = ("w_lift", "readout_rot", "unet_gates")

    def is_muon_leaf(top_k: str, sub_k: str | None) -> bool:
        if top_k == "blocks":
            return sub_k in muon_block_keys
        return top_k in muon_top_keys

    opt_state = {
        "step": jnp.array(0, dtype=jnp.int32),
        "mom": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v_adam": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(p: dict, state: dict, x_full: jax.Array, y_full: jax.Array, lr_mult: jax.Array, rng_key: jax.Array):
        mb_keys = jax.random.split(rng_key, grad_accum)
        val_grad_fn = jax.value_and_grad(microbatch_loss)

        def accum_body(carry, inputs):
            loss_acc, grad_acc = carry
            x_mb, y_mb, k_mb = inputs
            l_mb, g_mb = val_grad_fn(p, x_mb, y_mb, k_mb)
            new_grad = jax.tree_util.tree_map(lambda a, b: a + b / float(grad_accum), grad_acc, g_mb)
            return (loss_acc + l_mb / float(grad_accum), new_grad), None

        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, p)
        (loss_val, grads), _ = jax.lax.scan(
            accum_body, (jnp.zeros((), dtype=jnp.float32), zero_grads), (x_full, y_full, mb_keys)
        )

        sq_norm = sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree_util.tree_leaves(grads))
        g_norm = jnp.sqrt(sq_norm + 1e-12)
        clip_scale = jnp.minimum(1.0, 1.0 / g_norm)
        grads = jax.tree_util.tree_map(lambda g: g * clip_scale, grads)

        step_num = state["step"] + 1
        do_retract = (step_num % 4) == 0
        lr_muon = 0.025 * lr_mult
        lr_adam = 0.004 * lr_mult
        b1_corr = 1.0 - (0.9 ** step_num.astype(jnp.float32))
        b2_corr = 1.0 - (0.95 ** step_num.astype(jnp.float32))

        def update_muon_2d(w: jax.Array, g: jax.Array, m: jax.Array) -> tuple[jax.Array, jax.Array]:
            new_m = 0.95 * m + g
            u = newton_schulz5_bf16(new_m, steps=5)
            cwd_mask = (w * u > 0.0).astype(jnp.float32)
            w_next = w - (lr_muon * 0.01) * (w * cwd_mask)
            r_grad = jax.lax.cond(
                do_retract,
                stiefel_retract_grad,
                lambda x: jnp.zeros_like(x),
                w_next,
            )
            w_next = w_next - (0.005 * 4.0) * r_grad - lr_muon * u
            return w_next, new_m

        def update_adam(w: jax.Array, g: jax.Array, m: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            new_m = 0.9 * m + 0.1 * g
            new_v = 0.95 * v + 0.05 * (g * g)
            m_hat = new_m / b1_corr
            v_hat = new_v / b2_corr
            w_next = w * (1.0 - lr_adam * 0.01) - lr_adam * (m_hat / (jnp.sqrt(v_hat) + 1e-8))
            return w_next, new_m, new_v

        new_p = {}
        new_mom = {}
        new_v = {}

        for top_k in p:
            if top_k == "blocks":
                new_p["blocks"] = {}
                new_mom["blocks"] = {}
                new_v["blocks"] = {}
                for sub_k in p["blocks"]:
                    w_l = p["blocks"][sub_k]
                    g_l = grads["blocks"][sub_k]
                    m_l = state["mom"]["blocks"][sub_k]
                    v_l = state["v_adam"]["blocks"][sub_k]
                    if is_muon_leaf("blocks", sub_k):
                        w_u, m_u = jax.vmap(update_muon_2d)(w_l, g_l, m_l)
                        new_p["blocks"][sub_k] = w_u
                        new_mom["blocks"][sub_k] = m_u
                        new_v["blocks"][sub_k] = v_l
                    else:
                        w_u, m_u, v_u = update_adam(w_l, g_l, m_l, v_l)
                        new_p["blocks"][sub_k] = w_u
                        new_mom["blocks"][sub_k] = m_u
                        new_v["blocks"][sub_k] = v_u
            elif top_k == "unet_gates":
                w_u, m_u = jax.vmap(update_muon_2d)(p[top_k], grads[top_k], state["mom"][top_k])
                new_p[top_k] = w_u
                new_mom[top_k] = m_u
                new_v[top_k] = state["v_adam"][top_k]
            elif is_muon_leaf(top_k, None):
                w_u, m_u = update_muon_2d(p[top_k], grads[top_k], state["mom"][top_k])
                new_p[top_k] = w_u
                new_mom[top_k] = m_u
                new_v[top_k] = state["v_adam"][top_k]
            else:
                w_u, m_u, v_u = update_adam(p[top_k], grads[top_k], state["mom"][top_k], state["v_adam"][top_k])
                new_p[top_k] = w_u
                new_mom[top_k] = m_u
                new_v[top_k] = v_u

        new_state = {"step": step_num, "mom": new_mom, "v_adam": new_v}
        return new_p, new_state, loss_val

    @jax.jit
    def eval_batch_step(p: dict, x_val: jax.Array, y_val: jax.Array):
        phi_seed, M_seed, inv_t = encode_backbone(p, x_val, None, False)
        N = phi_seed.shape[0]
        num_chunks = 4
        chunk_sz = N // num_chunks
        phi_c = phi_seed.reshape(num_chunks, chunk_sz, D_seed)
        y_c = y_val.reshape(num_chunks, chunk_sz)
        beta = jnp.clip(inv_t.astype(jnp.bfloat16), 1.0, 60.0)

        def eval_chunk(carry, inputs):
            loss_acc, t1_acc, t5_acc = carry
            zc, yc = inputs
            logits = (jnp.matmul(zc, M_seed.T, preferred_element_type=jnp.bfloat16) * beta).astype(jnp.float32)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(log_probs, yc[:, None], axis=-1).squeeze(-1)
            _, top5_idx = jax.lax.top_k(logits, 5)
            t1 = jnp.sum(top5_idx[:, 0] == yc)
            t5 = jnp.sum(jnp.any(top5_idx == yc[:, None], axis=-1))
            return (loss_acc + jnp.sum(nll), t1_acc + t1, t5_acc + t5), None

        (loss_sum, t1_hits, t5_hits), _ = jax.lax.scan(
            eval_chunk,
            (jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.int32)),
            (phi_c, y_c),
        )
        sub_logits = (jnp.matmul(phi_seed[:1024], M_seed.T, preferred_element_type=jnp.bfloat16) * beta).astype(jnp.float32)
        sub_y = y_val.reshape(-1)[:1024]
        target_scores = jnp.take_along_axis(sub_logits, sub_y[:, None], axis=-1)
        ranks = jnp.sum(sub_logits > target_scores, axis=-1) + 1
        return loss_sum / float(N), t1_hits, t5_hits, ranks

    def compute_isometry_error_tpu(p: dict) -> float:
        errs: list[float] = []
        mats: list[jax.Array] = [p["w_lift"], p["readout_rot"]]
        for i in range(L // 2):
            mats.append(p["unet_gates"][i])
        for sub_k in muon_block_keys:
            for i in range(L):
                mats.append(p["blocks"][sub_k][i])
        for w in mats:
            wf = np.asarray(w, dtype=np.float32)
            r, c = wf.shape
            if r >= c:
                gram = wf.T @ wf
                err = np.linalg.norm(gram - np.eye(c, dtype=np.float32), ord="fro") / math.sqrt(c)
            else:
                gram = wf @ wf.T
                err = np.linalg.norm(gram - np.eye(r, dtype=np.float32), ord="fro") / math.sqrt(r)
            errs.append(float(err))
        return float(np.mean(errs))

    def evaluate_validation(p: dict, num_batches: int = 10) -> dict[str, float]:
        losses: list[float] = []
        t1_total = 0
        t5_total = 0
        tok_total = 0
        all_ranks: list[np.ndarray] = []
        stride = B_micro * T
        for b_idx in range(num_batches):
            base = b_idx * stride
            chunk = val_tokens[base : base + stride + 1].astype(np.int32)
            x_v = jnp.asarray(chunk[:-1].reshape(B_micro, T))
            y_v = jnp.asarray(chunk[1:].reshape(B_micro, T))
            l_b, t1_b, t5_b, ranks_b = eval_batch_step(p, x_v, y_v)
            losses.append(float(l_b))
            t1_total += int(t1_b)
            t5_total += int(t5_b)
            tok_total += stride
            if b_idx < 2:
                all_ranks.append(np.asarray(ranks_b))
        val_ce = float(np.mean(losses))
        val_ppl = float(math.exp(min(val_ce, 20.0)))
        med_rank = float(np.median(np.concatenate(all_ranks))) if all_ranks else -1.0
        iso_err = compute_isometry_error_tpu(p)
        return {
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "top1_acc": 100.0 * t1_total / max(1, tok_total),
            "top5_acc": 100.0 * t5_total / max(1, tok_total),
            "med_rank": med_rank,
            "iso_err": iso_err,
        }

    print("Compiling XLA HLO training & evaluation programs on TPU v5e-1...", flush=True)
    t_comp0 = time.perf_counter()
    x_warm = jnp.zeros((grad_accum, B_micro, T), dtype=jnp.int32)
    y_warm = jnp.zeros((grad_accum, B_micro, T), dtype=jnp.int32)
    p_copy = jax.tree_util.tree_map(lambda x: x.copy(), params)
    s_copy = jax.tree_util.tree_map(lambda x: x.copy(), opt_state)
    p_copy, s_copy, l_warm = train_step(p_copy, s_copy, x_warm, y_warm, jnp.array(0.0, dtype=jnp.float32), key)
    jax.block_until_ready(l_warm)
    del p_copy, s_copy
    gc.collect()
    val0 = evaluate_validation(params)
    print(f"XLA compilation complete in {time.perf_counter() - t_comp0:.2f}s", flush=True)

    history: list[dict[str, float]] = []
    print(
        f"[Epoch 0.00 | Step    0 |   0.00M tok] Val CE: {val0['val_ce']:.4f} | Val PPL: {val0['val_ppl']:8.1f} | "
        f"Top-1: {val0['top1_acc']:5.2f}% | Top-5: {val0['top5_acc']:5.2f}% | MedRank: {val0['med_rank']:5.0f} | IsoErr: {val0['iso_err']:.4f}",
        flush=True,
    )
    history.append({"epoch": 0.0, "step": 0, "tokens_M": 0.0, "train_ce": val0["val_ce"], "tok_per_s": 0.0, **val0})

    warmup_steps = 150
    running_loss_arr = jnp.zeros((), dtype=jnp.float32)
    running_steps = 0
    train_t0 = time.perf_counter()
    window_t0 = time.perf_counter()
    window_tokens = 0

    global_step = 0
    step_rng = jax.random.PRNGKey(100)

    for ep in range(num_epochs):
        rng = np.random.default_rng(42 + ep)
        batch_order = rng.permutation(steps_per_epoch)

        for b_pos in range(steps_per_epoch):
            global_step += 1
            if global_step <= warmup_steps:
                lr_mult = global_step / float(warmup_steps)
            else:
                prog = (global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                lr_mult = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))

            chunk_idx = int(batch_order[b_pos])
            start_tok = chunk_idx * tok_per_step
            buf = train_tokens[start_tok : start_tok + tok_per_step + 1].astype(np.int32)
            x_np = buf[:-1].reshape(grad_accum, B_micro, T)
            y_np = buf[1:].reshape(grad_accum, B_micro, T)
            x_j = jnp.asarray(x_np)
            y_j = jnp.asarray(y_np)

            step_rng, sub_k = jax.random.split(step_rng)
            params, opt_state, loss_j = train_step(
                params, opt_state, x_j, y_j, jnp.array(lr_mult, dtype=jnp.float32), sub_k
            )

            running_loss_arr = running_loss_arr + loss_j
            running_steps += 1
            window_tokens += tok_per_step

            if (global_step == eval_every + (ep * steps_per_epoch)) or (global_step == (ep + 1) * steps_per_epoch):
                train_ce = float(running_loss_arr) / max(1, running_steps)
                dt_w = time.perf_counter() - window_t0
                tps = window_tokens / max(dt_w, 1e-6)
                running_loss_arr = jnp.zeros((), dtype=jnp.float32)
                running_steps = 0

                val_metrics = evaluate_validation(params)
                ep_frac = round((global_step / float(steps_per_epoch)) * 2.0) / 2.0
                tok_m = (global_step * tok_per_step) / 1e6
                print(
                    f"[Epoch {ep_frac:4.2f} | Step {global_step:4d} | {tok_m:6.2f}M tok] "
                    f"Train CE: {train_ce:.4f} | Val CE: {val_metrics['val_ce']:.4f} | Val PPL: {val_metrics['val_ppl']:6.1f} | "
                    f"Top-1: {val_metrics['top1_acc']:5.2f}% | Top-5: {val_metrics['top5_acc']:5.2f}% | "
                    f"MedRank: {val_metrics['med_rank']:4.0f} | IsoErr: {val_metrics['iso_err']:.4f} | "
                    f"{tps:,.0f} tok/s",
                    flush=True,
                )
                history.append(
                    {
                        "epoch": ep_frac,
                        "step": global_step,
                        "tokens_M": tok_m,
                        "train_ce": train_ce,
                        "tok_per_s": tps,
                        **val_metrics,
                    }
                )
                Path("/content/tpu_100M_results.json").write_text(json.dumps(history, indent=2))
                window_t0 = time.perf_counter()
                window_tokens = 0

    total_time = time.perf_counter() - train_t0
    print("=" * 90, flush=True)
    print(f"COMPLETED {num_epochs} FULL 100M-TOKEN EPOCHS ON TPU v5e-1 IN {total_time:.1f}s ({total_time/60:.2f} min)", flush=True)
    print("=" * 90, flush=True)
    return history


def main() -> None:
    args = parse_args()
    backend = args.backend
    if backend == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                backend = "cuda"
        except ImportError:
            pass
    if backend == "auto":
        try:
            import jax

            if jax.default_backend() == "tpu":
                backend = "tpu"
        except ImportError:
            pass

    if backend == "cuda":
        run_cuda_benchmark(num_epochs=args.epochs, data_dir=args.data_dir)
    elif backend == "tpu":
        run_tpu_benchmark(num_epochs=args.epochs, data_dir=args.data_dir)
    else:
        raise RuntimeError(f"No CUDA GPU or Cloud TPU detected for backend={args.backend!r}.")


if __name__ == "__main__":
    main()
