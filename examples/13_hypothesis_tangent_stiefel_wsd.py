"""Example 13: Controlled Verification of Observable Hypotheses on Tangent-Space Stiefel Muon
Projection (H1) and Elimination of the High-IsoErr Plateau Under Constant High-LR WSD Training (H2).

Implements Section 3.3 of EXPERIMENT_PLAN_003.md and writes empirical results to
`.cache/hypothesis_experiments_003.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.functional import zeropower_via_newtonschulz5
from qdlblocks.metrics import stiefel_isometry_error
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params
from qdlblocks.optim.tangent_stiefel_muon import (
    TangentStiefelMuon,
    project_stiefel_tangent,
    stiefel_quadratic_retract,
)
from qdlblocks_jax.tangent_stiefel import (
    newton_schulz_5_jax,
    project_stiefel_tangent_jax,
    stiefel_quadratic_retract_jax,
)


def _skew_symmetry_rel_error_torch(w: torch.Tensor, delta: torch.Tensor, g: torch.Tensor) -> float:
    wf = w.float()
    df = delta.float()
    gf_norm = float(torch.norm(g.float(), p="fro").item())
    if wf.size(0) >= wf.size(1):
        sym = wf.t() @ df + df.t() @ wf
    else:
        sym = wf @ df.t() + df @ wf.t()
    return float(torch.norm(sym, p="fro").item() / max(gf_norm, 1e-12))


def _skew_symmetry_rel_error_jax(w: jnp.ndarray, delta: jnp.ndarray, g: jnp.ndarray) -> float:
    wf = w.astype(jnp.float32)
    df = delta.astype(jnp.float32)
    gf_norm = float(jnp.linalg.norm(g.astype(jnp.float32)))
    if wf.shape[0] >= wf.shape[1]:
        sym = wf.T @ df + df.T @ wf
    else:
        sym = wf @ df.T + df @ wf.T
    return float(jnp.linalg.norm(sym) / max(gf_norm, 1e-12))


def run_experiment_h1_tangent_projection() -> dict[str, Any]:
    """Experiment 1 (Hypothesis H1): Tangent Space Orthogonality & Single-Step Drift."""
    print("\n" + "=" * 92)
    print("EXPERIMENT 1 (H1): Tangent Space Skew-Symmetry & First-Order vs. Second-Order Stiefel Drift")
    print("=" * 92)

    eta = 0.05
    shapes = [("tall_128x64", (128, 64)), ("wide_64x128", (64, 128))]
    results: dict[str, Any] = {}

    print(
        f"  {'Shape':<14} | {'SkewErr(PT)':>12} | {'SkewErr(JAX)':>12} | "
        f"{'IsoErr(Raw)':>12} | {'IsoErr(Tan+Ret)':>15} | {'Ratio(Tan/Raw)':>14}"
    )
    print("  " + "-" * 88)

    for key, (r, c) in shapes:
        torch.manual_seed(42)
        w = torch.empty(r, c, dtype=torch.float32)
        nn.init.orthogonal_(w)
        e = torch.randn(r, c, dtype=torch.float32)
        g = e / torch.norm(e, p="fro") + 0.5 * w

        # 1. Skew-symmetry check in PyTorch and JAX
        delta_pt = project_stiefel_tangent(w, g)
        skew_pt = _skew_symmetry_rel_error_torch(w, delta_pt, g)

        w_jax = jnp.asarray(w.numpy())
        g_jax = jnp.asarray(g.numpy())
        delta_jax = project_stiefel_tangent_jax(w_jax, g_jax)
        skew_jax = _skew_symmetry_rel_error_jax(w_jax, delta_jax, g_jax)

        assert skew_pt <= 1e-5, f"H1 Failed ({key} PyTorch): skew_err={skew_pt:.3e} > 1e-5"
        assert skew_jax <= 1e-5, f"H1 Failed ({key} JAX): skew_err={skew_jax:.3e} > 1e-5"

        # 2. Single-step ambient vs. double-tangent-projected + quadratic-retracted update
        aspect_scale = math.sqrt(max(1.0, r / c))
        u_raw = zeropower_via_newtonschulz5(g, steps=5)
        w_raw = w - eta * aspect_scale * u_raw
        iso_raw = stiefel_isometry_error(w_raw)

        g_tan = project_stiefel_tangent(w, g)
        u_tan = zeropower_via_newtonschulz5(g_tan, steps=5) * aspect_scale
        delta_tan = project_stiefel_tangent(w, u_tan)
        w_tan = stiefel_quadratic_retract(w - eta * delta_tan, alpha=0.5)
        iso_tan = stiefel_isometry_error(w_tan)

        # Also verify JAX parity for w_tan
        u_tan_jax = newton_schulz_5_jax(delta_jax, steps=5) * aspect_scale
        delta_tan_jax = project_stiefel_tangent_jax(w_jax, u_tan_jax)
        w_tan_jax = stiefel_quadratic_retract_jax(w_jax - eta * delta_tan_jax, alpha=0.5)
        max_pt_jax_diff = float(np.max(np.abs(w_tan.numpy() - np.asarray(w_tan_jax))))
        assert max_pt_jax_diff <= 1e-5, f"PyTorch-JAX parity diff too large: {max_pt_jax_diff:.3e}"

        ratio = iso_tan / max(iso_raw, 1e-12)
        print(
            f"  {key:<14} | {skew_pt:12.3e} | {skew_jax:12.3e} | "
            f"{iso_raw:12.6f} | {iso_tan:15.6f} | {ratio:14.4f}"
        )

        assert iso_raw >= 0.04, f"H1 Failed ({key}): expected iso_raw >= 0.04, got {iso_raw:.6f}"
        assert iso_tan <= 0.005, f"H1 Failed ({key}): expected iso_tan <= 0.005, got {iso_tan:.6f}"
        assert iso_tan <= 0.10 * iso_raw, (
            f"H1 Failed ({key}): expected iso_tan <= 0.10 * iso_raw, got {iso_tan:.6f} vs {0.10 * iso_raw:.6f}"
        )

        results[key] = {
            "shape": [r, c],
            "eta": eta,
            "skew_symmetry_rel_error_torch": round(skew_pt, 9),
            "skew_symmetry_rel_error_jax": round(skew_jax, 9),
            "iso_err_raw": round(iso_raw, 6),
            "iso_err_tan": round(iso_tan, 6),
            "iso_err_ratio": round(ratio, 6),
            "max_pt_jax_diff": round(max_pt_jax_diff, 9),
        }

    results["passed"] = True
    return results


def _generate_2gram_markov_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic 2-gram context-sensitive Markov sequences per Section 3.3."""
    rng = np.random.default_rng(42)
    next_tok = (np.arange(256) * 5 + 11) % 256
    ctx_offset = (np.arange(256) * 13) % 8

    def _sample_split(num_seqs: int) -> torch.Tensor:
        seqs = np.zeros((num_seqs, 65), dtype=np.int64)
        seqs[:, 0] = rng.integers(0, 256, size=num_seqs)
        seqs[:, 1] = (next_tok[seqs[:, 0]] + rng.integers(0, 4, size=num_seqs)) % 256
        for t in range(1, 64):
            u = rng.random(num_seqs)
            det = (next_tok[seqs[:, t]] + ctx_offset[seqs[:, t - 1]]) % 256
            rand_tok = rng.integers(0, 256, size=num_seqs)
            seqs[:, t + 1] = np.where(u < 0.90, det, rand_tok)
        return torch.from_numpy(seqs)

    train_data = _sample_split(256)
    val_data = _sample_split(64)
    return train_data, val_data


def _mean_muon_iso_err(muon_params: list[nn.Parameter]) -> float:
    return float(np.mean([stiefel_isometry_error(p) for p in muon_params]))


def run_experiment_h2_wsd_plateau(device: torch.device) -> dict[str, Any]:
    """Experiment 2 (Hypothesis H2): Constant-LR WSD Plateau Training on QuantumBornGPT."""
    print("\n" + "=" * 92)
    print("EXPERIMENT 2 (H2): Constant High-LR WSD Plateau Training (100 Steps, eta_muon=0.03)")
    print("=" * 92)

    train_data, val_data = _generate_2gram_markov_dataset()
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    x_train, y_train = train_data[:, :-1], train_data[:, 1:]
    x_val, y_val = val_data[:, :-1], val_data[:, 1:]

    vocab_size = 256
    steps = 100
    batch_size = 16

    def build_model() -> QuantumBornGPT:
        torch.manual_seed(123)
        return QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=64,
            seed_dim=32,
            num_heads=4,
            num_layers=2,
            hidden_dim=128,
            block_size=64,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
        ).to(device)

    adamw_patterns = ("wte", "povm_head", "embed", "unet_scales", "smear")

    def run_arm(arm_name: str, use_tangent: bool) -> dict[str, Any]:
        model = build_model()
        muon_params, adamw_params = split_muon_adamw_params(model, adamw_name_patterns=adamw_patterns)

        if use_tangent:
            opt_muon: torch.optim.Optimizer = TangentStiefelMuon(
                muon_params,
                lr=0.03,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                retract_alpha=0.5,
                retract_every=1,
            )
        else:
            opt_muon = StiefelMuon(
                muon_params,
                lr=0.03,
                momentum=0.95,
                weight_decay=0.01,
                decay_mode="cautious",
                stiefel_penalty=0.005,
                ns_steps=5,
                retract_every=4,
            )
        opt_adamw = torch.optim.AdamW(adamw_params, lr=0.008, weight_decay=0.01)

        initial_iso_err = _mean_muon_iso_err(muon_params)
        history: list[dict[str, float]] = []

        model.train()
        for step in range(1, steps + 1):
            start_idx = ((step - 1) * batch_size) % x_train.size(0)
            xb = x_train[start_idx : start_idx + batch_size]
            yb = y_train[start_idx : start_idx + batch_size]

            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adamw.step()

            if step in (1, 25, 50, 75, steps):
                model.eval()
                with torch.no_grad():
                    val_logits, _ = model(x_val, y_val)
                    val_ce = float(
                        F.cross_entropy(val_logits.reshape(-1, vocab_size), y_val.reshape(-1)).item()
                    )
                iso_now = _mean_muon_iso_err(muon_params)
                history.append(
                    {
                        "step": float(step),
                        "train_ce": round(float(loss.detach().item()), 4),
                        "val_ce": round(val_ce, 4),
                        "iso_err": round(iso_now, 6),
                    }
                )
                model.train()

        final_metrics = history[-1]
        print(
            f"  [{arm_name:<28}] Init IsoErr: {initial_iso_err:.4f} -> "
            f"Final IsoErr: {final_metrics['iso_err']:.6f} | "
            f"Final Train CE: {final_metrics['train_ce']:.4f} | "
            f"Final Val CE: {final_metrics['val_ce']:.4f}"
        )
        return {
            "initial_iso_err": round(initial_iso_err, 6),
            "final_iso_err": final_metrics["iso_err"],
            "final_train_ce": final_metrics["train_ce"],
            "final_val_ce": final_metrics["val_ce"],
            "history": history,
        }

    arm_a = run_arm("Arm A (StiefelMuon)", use_tangent=False)
    arm_b = run_arm("Arm B (TangentStiefelMuon)", use_tangent=True)

    iso_a = arm_a["final_iso_err"]
    iso_b = arm_b["final_iso_err"]
    val_a = arm_a["final_val_ce"]
    val_b = arm_b["final_val_ce"]

    print("  " + "-" * 88)
    print(
        f"  Comparison Summary: IsoErr Reduction = {iso_a / max(iso_b, 1e-9):.1f}x "
        f"({iso_a:.4f} -> {iso_b:.6f}) | Val CE Gain = {val_a - val_b:+.4f} nats ({val_a:.4f} -> {val_b:.4f})"
    )

    assert iso_b <= 0.05, f"H2 Failed: expected final_iso_err(Arm B) <= 0.05, got {iso_b:.6f}"
    assert iso_b <= 0.15 * iso_a, (
        f"H2 Failed: expected final_iso_err(Arm B) <= 0.15 * final_iso_err(Arm A), "
        f"got {iso_b:.6f} vs {0.15 * iso_a:.6f}"
    )
    assert val_b <= val_a - 0.05, (
        f"H2 Failed: expected final_val_ce(Arm B) <= final_val_ce(Arm A) - 0.05, "
        f"got {val_b:.4f} vs {val_a:.4f}"
    )

    return {
        "arm_a_stiefel_muon": arm_a,
        "arm_b_tangent_stiefel_muon": arm_b,
        "iso_err_ratio_b_over_a": round(iso_b / max(iso_a, 1e-12), 6),
        "val_ce_gain_nats": round(val_a - val_b, 4),
        "passed": True,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXPERIMENT_PLAN_003 Controlled Verification on device={device}...")

    h1_res = run_experiment_h1_tangent_projection()
    h2_res = run_experiment_h2_wsd_plateau(device=device)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_003",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "h1_tangent_projection": h1_res,
        "h2_wsd_plateau_training": h2_res,
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_003.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nAll H1 & H2 quantitative assertions PASSED in {elapsed:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
