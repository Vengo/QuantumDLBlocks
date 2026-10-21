"""Controlled hypothesis verification for EXPERIMENT_PLAN_005:
Analytic-Gradient L2-Tiled POVM Cross-Entropy for HBM Memory Reduction & Roofline Arithmetic Intensity.

Verifies:
- Hypothesis H1: Exact analytical gradient equivalence (PyTorch & JAX) and O(N / B_tile)
  transient logit working-set reduction + O(N) saved activation footprint.
- Hypothesis H2: Roofline arithmetic intensity transition across the 96 MiB L2 cache threshold
  on FineWeb 100M (N=16384, D_seed=128, V=50304) and monotonic peak buffer scaling.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.kernels.fused_povm_ce import (
    estimate_povm_head_roofline,
    fused_tiled_povm_cross_entropy,
)
from qdlblocks_jax.fused_povm_ce import (
    estimate_povm_head_roofline_jax,
    fused_tiled_povm_cross_entropy_jax,
)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom < 1e-12:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def run_h1_gradient_and_memory() -> dict[str, Any]:
    """Run Hypothesis H1: Exact gradient equivalence & saved/transient memory reduction."""
    torch.manual_seed(42)
    N = 2048
    D_seed = 64
    V = 8192
    tile_size = 256

    z_raw = torch.randn(N, D_seed, dtype=torch.float32)
    z_base = F.normalize(z_raw, p=2.0, dim=-1)
    m_raw = torch.randn(V, D_seed, dtype=torch.float32)
    m_base = F.normalize(m_raw, p=2.0, dim=-1)
    beta_base = torch.tensor(28.5, dtype=torch.float32)
    targets = torch.randint(0, V, (N,), dtype=torch.long)

    # 1. PyTorch Dense Reference
    z_ref = z_base.clone().requires_grad_(True)
    m_ref = m_base.clone().requires_grad_(True)
    beta_ref = beta_base.clone().requires_grad_(True)
    logits_ref = F.linear(z_ref.float(), m_ref.float()) * beta_ref.float()
    loss_ref = F.cross_entropy(logits_ref, targets)
    loss_ref.backward()

    # 2. PyTorch Tiled Analytic Kernel
    z_tiled = z_base.clone().requires_grad_(True)
    m_tiled = m_base.clone().requires_grad_(True)
    beta_tiled = beta_base.clone().requires_grad_(True)
    loss_tiled = fused_tiled_povm_cross_entropy(
        z_tiled, m_tiled, beta_tiled, targets, tile_size=tile_size
    )
    loss_tiled.backward()

    loss_diff_torch = float(torch.abs(loss_tiled - loss_ref).item())
    assert z_ref.grad is not None and z_tiled.grad is not None
    assert m_ref.grad is not None and m_tiled.grad is not None
    assert beta_ref.grad is not None and beta_tiled.grad is not None

    dz_err_torch = float(torch.max(torch.abs(z_tiled.grad - z_ref.grad)).item())
    dm_err_torch = float(torch.max(torch.abs(m_tiled.grad - m_ref.grad)).item())
    dbeta_err_torch = float(torch.abs(beta_tiled.grad - beta_ref.grad).item())
    max_grad_err_torch = max(dz_err_torch, dm_err_torch, dbeta_err_torch)

    dz_rel_torch = _rel_l2(
        z_tiled.grad.detach().cpu().numpy(), z_ref.grad.detach().cpu().numpy()
    )
    dm_rel_torch = _rel_l2(
        m_tiled.grad.detach().cpu().numpy(), m_ref.grad.detach().cpu().numpy()
    )
    dbeta_rel_torch = _rel_l2(
        beta_tiled.grad.detach().cpu().numpy(), beta_ref.grad.detach().cpu().numpy()
    )
    max_rel_l2_torch = max(dz_rel_torch, dm_rel_torch, dbeta_rel_torch)

    # 3. JAX Dense Reference vs. JAX Tiled Custom VJP
    z_j = jnp.asarray(z_base.numpy(), dtype=jnp.float32)
    m_j = jnp.asarray(m_base.numpy(), dtype=jnp.float32)
    beta_j = jnp.asarray(beta_base.numpy(), dtype=jnp.float32)
    y_j = jnp.asarray(targets.numpy(), dtype=jnp.int32)

    def _dense_jax_ce(z_in: jax.Array, m_in: jax.Array, b_in: jax.Array) -> jax.Array:
        logits = (
            jnp.matmul(z_in, m_in.T, precision=jax.lax.Precision.HIGHEST)
            * b_in.reshape(())
        )
        lse = jax.nn.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, y_j[:, None], axis=-1).squeeze(-1)
        return jnp.mean(lse - tgt)

    loss_jax_ref, (dz_j_ref, dm_j_ref, dbeta_j_ref) = jax.jit(
        jax.value_and_grad(_dense_jax_ce, argnums=(0, 1, 2))
    )(z_j, m_j, beta_j)

    loss_jax_tiled, (dz_j_tiled, dm_j_tiled, dbeta_j_tiled) = jax.jit(
        jax.value_and_grad(
            lambda z_in, m_in, b_in: fused_tiled_povm_cross_entropy_jax(
                z_in, m_in, b_in, y_j, tile_size=tile_size
            ),
            argnums=(0, 1, 2),
        )
    )(z_j, m_j, beta_j)

    loss_diff_jax = float(jnp.abs(loss_jax_tiled - loss_jax_ref))
    dz_err_jax = float(jnp.max(jnp.abs(dz_j_tiled - dz_j_ref)))
    dm_err_jax = float(jnp.max(jnp.abs(dm_j_tiled - dm_j_ref)))
    dbeta_err_jax = float(jnp.abs(dbeta_j_tiled - dbeta_j_ref))
    max_grad_err_jax = max(dz_err_jax, dm_err_jax, dbeta_err_jax)

    dz_rel_jax = _rel_l2(np.asarray(dz_j_tiled), np.asarray(dz_j_ref))
    dm_rel_jax = _rel_l2(np.asarray(dm_j_tiled), np.asarray(dm_j_ref))
    dbeta_rel_jax = _rel_l2(np.asarray(dbeta_j_tiled), np.asarray(dbeta_j_ref))
    max_rel_l2_jax = max(dz_rel_jax, dm_rel_jax, dbeta_rel_jax)

    # 4. Memory footprint verification
    roof_tiled = estimate_povm_head_roofline(N, D_seed, V, tile_size=tile_size, dtype_bytes=2)
    roof_dense = estimate_povm_head_roofline(N, D_seed, V, tile_size=N, dtype_bytes=2)

    peak_buffer_mb_tiled = float(roof_tiled["peak_logit_buffer_mb"])
    peak_buffer_mb_dense = float(roof_dense["peak_logit_buffer_mb"])
    saved_activation_kb = float(roof_tiled["saved_activation_kb"])

    assert loss_diff_torch <= 1e-5, f"PyTorch loss diff {loss_diff_torch} > 1e-5"
    assert loss_diff_jax <= 1e-5, f"JAX loss diff {loss_diff_jax} > 1e-5"
    assert max_grad_err_torch <= 1e-4, f"PyTorch max grad err {max_grad_err_torch} > 1e-4"
    assert max_rel_l2_torch <= 1e-4, f"PyTorch max rel L2 err {max_rel_l2_torch} > 1e-4"
    assert max_grad_err_jax <= 1e-4, f"JAX max grad err {max_grad_err_jax} > 1e-4"
    assert max_rel_l2_jax <= 1e-4, f"JAX max rel L2 err {max_rel_l2_jax} > 1e-4"
    assert peak_buffer_mb_tiled == 4.0, f"Expected 4.0 MiB tiled buffer, got {peak_buffer_mb_tiled}"
    assert peak_buffer_mb_dense == 32.0, f"Expected 32.0 MiB dense buffer, got {peak_buffer_mb_dense}"
    assert saved_activation_kb == 8.0, f"Expected 8.0 KiB saved lse, got {saved_activation_kb}"

    return {
        "loss_diff_torch": loss_diff_torch,
        "loss_diff_jax": loss_diff_jax,
        "max_grad_err_torch": max_grad_err_torch,
        "max_rel_l2_torch": max_rel_l2_torch,
        "max_grad_err_jax": max_grad_err_jax,
        "max_rel_l2_jax": max_rel_l2_jax,
        "peak_buffer_mb_tiled": peak_buffer_mb_tiled,
        "peak_buffer_mb_dense": peak_buffer_mb_dense,
        "saved_activation_kb": saved_activation_kb,
        "passed": True,
    }


def run_h2_roofline_scaling() -> dict[str, Any]:
    """Run Hypothesis H2: Roofline arithmetic intensity & tile-size sweep."""
    N_fw = 16384
    D_fw = 128
    V_fw = 50304
    l2_bytes = 96 * 1024 * 1024
    tile_sizes = [256, 512, 1024, 2048, 4096, 16384]

    fineweb_sweep: dict[str, dict[str, float | bool]] = {}
    for ts in tile_sizes:
        res_t = estimate_povm_head_roofline(
            N_fw, D_fw, V_fw, tile_size=ts, dtype_bytes=2, l2_cache_bytes=l2_bytes
        )
        res_j = estimate_povm_head_roofline_jax(
            N_fw, D_fw, V_fw, tile_size=ts, dtype_bytes=2, l2_cache_bytes=l2_bytes
        )
        assert res_t == res_j
        fineweb_sweep[str(ts)] = res_t

    ai_tile512 = float(fineweb_sweep["512"]["arithmetic_intensity"])
    ai_tile16384 = float(fineweb_sweep["16384"]["arithmetic_intensity"])
    ai_ratio = round(ai_tile512 / ai_tile16384, 3)

    assert fineweb_sweep["512"]["fits_in_l2"] is True
    assert fineweb_sweep["16384"]["fits_in_l2"] is False
    assert ai_tile512 >= 2.5 * ai_tile16384, (
        f"Expected AI(512)={ai_tile512} >= 2.5 * AI(16384)={ai_tile16384}"
    )

    # Live execution benchmark & monotonic buffer scaling on N=2048, D_seed=64, V=16384
    N_bench = 2048
    D_bench = 64
    V_bench = 16384
    torch.manual_seed(105)
    z_b = F.normalize(torch.randn(N_bench, D_bench, dtype=torch.float32), p=2.0, dim=-1)
    m_b = F.normalize(torch.randn(V_bench, D_bench, dtype=torch.float32), p=2.0, dim=-1)
    beta_b = torch.tensor(28.5, dtype=torch.float32)
    y_b = torch.randint(0, V_bench, (N_bench,), dtype=torch.long)

    live_benchmark: dict[str, dict[str, float]] = {}
    for ts in [256, 512, 1024, 2048]:
        z_run = z_b.clone().requires_grad_(True)
        m_run = m_b.clone().requires_grad_(True)
        beta_run = beta_b.clone().requires_grad_(True)
        t0 = time.perf_counter()
        loss = fused_tiled_povm_cross_entropy(z_run, m_run, beta_run, y_b, tile_size=ts)
        loss.backward()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        roof = estimate_povm_head_roofline(N_bench, D_bench, V_bench, tile_size=ts, dtype_bytes=2)
        live_benchmark[str(ts)] = {
            "loss": round(float(loss.item()), 6),
            "elapsed_ms": round(elapsed_ms, 2),
            "peak_logit_buffer_mb": float(roof["peak_logit_buffer_mb"]),
        }

    buf_256 = live_benchmark["256"]["peak_logit_buffer_mb"]
    buf_2048 = live_benchmark["2048"]["peak_logit_buffer_mb"]
    assert buf_256 <= 0.125 * buf_2048, (
        f"Expected peak_logit_buffer_mb(256)={buf_256} <= 0.125 * {buf_2048}"
    )

    return {
        "fineweb_sweep": fineweb_sweep,
        "ai_tile512": ai_tile512,
        "ai_tile16384": ai_tile16384,
        "ai_ratio": ai_ratio,
        "live_benchmark": live_benchmark,
        "passed": True,
    }


def main() -> None:
    print("=" * 84)
    print("EXPERIMENT_PLAN_005: Analytic-Gradient L2-Tiled POVM Cross-Entropy Verification")
    print("=" * 84)

    h1 = run_h1_gradient_and_memory()
    print("\n[H1] Exact Gradient Equivalence & Memory Footprint Reduction (N=2048, D=64, V=8192):")
    print(f"  PyTorch Forward Loss Diff : {h1['loss_diff_torch']:.3e} (<= 1e-5)")
    print(f"  PyTorch Max Grad Abs Err  : {h1['max_grad_err_torch']:.3e} (<= 1e-4)")
    print(f"  PyTorch Max Grad Rel L2   : {h1['max_rel_l2_torch']:.3e} (<= 1e-4)")
    print(f"  JAX Forward Loss Diff     : {h1['loss_diff_jax']:.3e} (<= 1e-5)")
    print(f"  JAX Max Grad Abs Err      : {h1['max_grad_err_jax']:.3e} (<= 1e-4)")
    print(f"  JAX Max Grad Rel L2       : {h1['max_rel_l2_jax']:.3e} (<= 1e-4)")
    print(
        f"  Peak Logit Buffer         : {h1['peak_buffer_mb_tiled']:.1f} MiB (tiled=256) "
        f"vs {h1['peak_buffer_mb_dense']:.1f} MiB (dense=2048) [8.0x reduction]"
    )
    print(f"  Saved Activation (LSE)    : {h1['saved_activation_kb']:.1f} KiB")

    h2 = run_h2_roofline_scaling()
    print("\n[H2] FineWeb 100M Roofline Sweep (N=16384, D_seed=128, V=50304, L2=96 MiB):")
    print(f"  {'Tile Size':>10} | {'Peak Buf (MiB)':>14} | {'Fits L2':>8} | {'AI (FLOP/B)':>12}")
    print("  " + "-" * 54)
    for ts, metrics in h2["fineweb_sweep"].items():
        print(
            f"  {ts:>10} | {metrics['peak_logit_buffer_mb']:>14.3f} | "
            f"{str(metrics['fits_in_l2']):>8} | {metrics['arithmetic_intensity']:>12.2f}"
        )
    print(
        f"  AI Jump (tile=512 vs 16384): {h2['ai_tile512']:.2f} vs {h2['ai_tile16384']:.2f} "
        f"({h2['ai_ratio']:.2f}x >= 2.50x)"
    )

    out_path = REPO_ROOT / ".cache" / "hypothesis_experiments_005.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "h1_gradient_and_memory": h1,
        "h2_roofline_scaling": h2,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved verification artifact to {out_path}")


if __name__ == "__main__":
    main()
