"""Example 23: Controlled Verification of Contextual Bloch-Purity Dynamic POVM Sharpness,
Karpathy Overfitting Capacity Scaling, & Classical Modded-NanoGPT Baseline (Plan 013).

Tests:
- Hypothesis H1 (Constant-Logit-Norm Theorem & Heteroscedastic Sharpness Recovery via
  `ContextualBlochPurityGate`):
  1. Verifies that under global-beta spherical readout, per-token logit norm coefficient
     of variation satisfies `CV(||ell||_2) < 0.08`.
  2. On a controlled Heteroscedastic Contextual Language Modeling Task (`V=512, d=128, D_seed=64,
     L=2, H=4`, 150 steps) with 50% deterministic contexts (`H* = 0.0` nats) and 50% 16-way
     branching contexts (`H* = ln(16) = 2.7726` nats), `QuantumBornGPT(..., use_dynamic_povm_beta=True)`
     achieves `CE_dyn <= CE_static - 0.15`, `beta_deterministic >= 1.35 * beta_branching`, and
     `CV_dyn(||ell||_2) >= 0.15`.
- Hypothesis H2 (Karpathy Overfitting Capacity Scaling & Classical Modded-NanoGPT Verification):
  On a 32-record sequence memorization benchmark (`N=32, T=64, V=1024`, 120 steps):
  1. Expanding seed readout capacity from `D_seed=16` (static beta) to `D_seed=64` with
     `OvercompleteSeedReadoutAdapter` and `ContextualBlochPurityGate` reduces final memorization
     Cross-Entropy by >= 50% (`CE_{D=64,dyn} <= 0.50 * CE_{D=16,static}`) and achieves
     `Top-1 Memorization Accuracy >= 98.0%`.
  2. `ClassicalModdedGPT(L=2, d=128, H=4)` achieves `Top-1 Memorization Accuracy >= 98.0%`
     and `CE_classical <= 0.10`.

Writes empirical verification telemetry to `.cache/hypothesis_experiments_013.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks import (
    ClassicalModdedGPT,
    QuantumBornGPT,
    compute_logit_norm_variation_metrics,
    l2_normalize,
)


def run_h1_heteroscedastic_sharpness_experiment(device: torch.device) -> dict[str, Any]:
    """Run Hypothesis 1: Constant-Logit-Norm Theorem & Heteroscedastic Sharpness Recovery."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 1 (H1): Constant-Logit-Norm Bottleneck & Heteroscedastic Sharpness Recovery")
    print("=" * 100)

    V = 512
    d = 128
    d_seed = 64
    L = 2
    H = 4
    K_branch = 16
    steps = 150
    batch_size = 32
    seq_len = 32

    torch.manual_seed(42)
    np.random.seed(42)

    # Build a shared QuantumBornGPT backbone + tight-frame seed POVM geometry so both static-beta
    # and dynamic-beta models share identical directional cosine similarities <phi_{b,t} | m_v>
    # and isolate the Constant-Logit-Norm sharpness bottleneck.
    def _build_heteroscedastic_model(use_dynamic_beta: bool) -> QuantumBornGPT:
        torch.manual_seed(101)
        m = QuantumBornGPT(
            vocab_size=V,
            embed_dim=d,
            seed_dim=d_seed,
            num_heads=H,
            num_layers=L,
            hidden_dim=192,
            block_size=seq_len,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            use_two_beam_mlp=True,
            use_overcomplete_seed_adapter=True,
            use_dynamic_povm_beta=use_dynamic_beta,
            povm_temperature=0.035,
        ).to(device)

        with torch.no_grad():
            g_cpu = torch.Generator().manual_seed(202)
            # Isotropic unit vocabulary POVM frame M_seed on S^{d_seed - 1}
            raw_m = torch.randn(V, d_seed, generator=g_cpu).to(device)
            u_m, _, vt_m = torch.linalg.svd(raw_m, full_matrices=False)
            m_vocab = l2_normalize(u_m @ vt_m, dim=-1)

            # Encode context-type regime (deterministic u < V//2 vs branching u >= V//2) in seed states
            reg_vec = l2_normalize(torch.randn(1, d_seed, generator=g_cpu).to(device), dim=-1)
            w_seed = m_vocab.clone()
            w_seed[: V // 2] = l2_normalize(m_vocab[: V // 2] + 0.65 * reg_vec, dim=-1)
            w_seed[V // 2 :] = l2_normalize(m_vocab[V // 2 :] - 0.65 * reg_vec, dim=-1)
            m.wte.weight.copy_(w_seed)
            assert m.povm_head_seed_delta is not None
            m.povm_head_seed_delta.copy_(m_vocab - w_seed)

            # Make causal attention local so each position (b, t) has a well-defined context entropy H*(Y | X_{b,t})
            for blk in m.blocks:
                blk.attn.out_proj.weight.zero_()
        return m

    m_static = _build_heteroscedastic_model(use_dynamic_beta=False)
    m_dyn = _build_heteroscedastic_model(use_dynamic_beta=True)

    # Extract directional cosine similarities <phi_u | m_v> for each context token u in [0, V):
    # - Deterministic contexts u in [0, V//2): single ground-truth target y*(u) = argmax_v <phi_u | m_v> (H* = 0 nats)
    # - Branching contexts u in [V//2, V): uniform over top K=16 continuations of |phi_u> (H* = ln 16 = 2.7726 nats)
    with torch.no_grad():
        m_static.eval()
        probe_idx = torch.arange(V, device=device, dtype=torch.long).unsqueeze(1)
        probe_logits, _ = m_static(probe_idx)
        probe_cos = probe_logits[:, 0, :].float()  # proportional to <phi_u | m_v>
        top_k_indices = torch.topk(probe_cos, k=K_branch, dim=-1).indices  # (V, K_branch)
        det_next = top_k_indices[: V // 2, 0].contiguous()  # (V // 2,)
        branch_next = top_k_indices[V // 2 :, :K_branch].contiguous()  # (V // 2, K_branch)

    def _sample_heteroscedastic_batch(n_seq: int, T_len: int, seed_val: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=device).manual_seed(seed_val)
        # Half deterministic contexts, half branching contexts
        is_branch = torch.randint(0, 2, (n_seq, T_len), generator=gen, device=device, dtype=torch.bool)
        u_det = torch.randint(0, V // 2, (n_seq, T_len), generator=gen, device=device, dtype=torch.long)
        u_br = torch.randint(V // 2, V, (n_seq, T_len), generator=gen, device=device, dtype=torch.long)
        x = torch.where(is_branch, u_br, u_det)

        y_det = det_next[u_det]
        k_choice = torch.randint(0, K_branch, (n_seq, T_len), generator=gen, device=device, dtype=torch.long)
        y_br = branch_next[u_br - (V // 2), k_choice]
        y = torch.where(is_branch, y_br, y_det)
        return x, y, is_branch

    x_val, y_val, mask_branch_val = _sample_heteroscedastic_batch(64, seq_len, seed_val=999)

    # Freeze directional backbone so static global beta vs. ContextualBlochPurityGate are compared
    # on identical directional representations <phi_{b,t} | m_v> for 150 optimization steps
    for m in (m_static, m_dyn):
        for name, p in m.named_parameters():
            if not any(k in name for k in ("tied_inv_temp", "purity_gate")):
                p.requires_grad_(False)

    opt_static = torch.optim.AdamW(
        [p for p in m_static.parameters() if p.requires_grad],
        lr=5e-2,
        weight_decay=0.0,
    )
    opt_dyn = torch.optim.AdamW(
        [p for p in m_dyn.parameters() if p.requires_grad],
        lr=1.5e-2,
        weight_decay=1e-4,
    )

    for step in range(1, steps + 1):
        xb, yb, _ = _sample_heteroscedastic_batch(batch_size, seq_len, seed_val=1000 + step)

        opt_static.zero_grad(set_to_none=True)
        _, loss_s = m_static(xb, yb)
        assert loss_s is not None
        loss_s.backward()
        opt_static.step()

        opt_dyn.zero_grad(set_to_none=True)
        _, loss_d = m_dyn(xb, yb)
        assert loss_d is not None
        loss_d.backward()
        opt_dyn.step()

    @torch.no_grad()
    def _eval_heteroscedastic(m: QuantumBornGPT) -> dict[str, float]:
        m.eval()
        logits, loss, l_states = m(x_val, y_val, return_layer_states=True)
        assert loss is not None
        norm_metrics = compute_logit_norm_variation_metrics(logits)

        flat_logits = logits.reshape(-1, V).float()
        flat_y = y_val.reshape(-1)
        flat_branch = mask_branch_val.reshape(-1)
        flat_det = ~flat_branch

        ce_all = float(F.cross_entropy(flat_logits, flat_y).item())
        ce_det = float(F.cross_entropy(flat_logits[flat_det], flat_y[flat_det]).item())
        ce_branch = float(F.cross_entropy(flat_logits[flat_branch], flat_y[flat_branch]).item())

        psi_last = l_states[-1]
        assert m.tied_inv_temp is not None
        base_beta = torch.clamp(m.tied_inv_temp, min=1.0, max=60.0)
        if m.purity_gate is not None:
            beta_tok = m.purity_gate(psi_last, base_beta).reshape(-1)
            beta_det = float(beta_tok[flat_det].mean().item())
            beta_branch = float(beta_tok[flat_branch].mean().item())
        else:
            b_val = float(base_beta.item())
            beta_det = b_val
            beta_branch = b_val

        return {
            "val_ce_total": round(ce_all, 4),
            "val_ce_deterministic": round(ce_det, 4),
            "val_ce_branching": round(ce_branch, 4),
            "beta_deterministic_mean": round(beta_det, 4),
            "beta_branching_mean": round(beta_branch, 4),
            "beta_ratio_det_over_branch": round(beta_det / max(beta_branch, 1e-6), 4),
            "logit_norm_mean": round(norm_metrics["logit_norm_mean"], 4),
            "logit_norm_std": round(norm_metrics["logit_norm_std"], 4),
            "logit_norm_cv": round(norm_metrics["logit_norm_cv"], 4),
            "mean_entropy_nats": round(norm_metrics["mean_entropy_nats"], 4),
        }

    res_static = _eval_heteroscedastic(m_static)
    res_dyn = _eval_heteroscedastic(m_dyn)
    ce_gain = round(res_static["val_ce_total"] - res_dyn["val_ce_total"], 4)

    print(
        f"  [Static Global Beta ] Val CE: {res_static['val_ce_total']:.4f} "
        f"(Det: {res_static['val_ce_deterministic']:.4f}, Branch: {res_static['val_ce_branching']:.4f}) | "
        f"beta_det={res_static['beta_deterministic_mean']:.2f}, beta_branch={res_static['beta_branching_mean']:.2f} | "
        f"CV(||ell||_2)={res_static['logit_norm_cv']:.4f}"
    )
    print(
        f"  [Dynamic Purity Gate] Val CE: {res_dyn['val_ce_total']:.4f} "
        f"(Det: {res_dyn['val_ce_deterministic']:.4f}, Branch: {res_dyn['val_ce_branching']:.4f}) | "
        f"beta_det={res_dyn['beta_deterministic_mean']:.2f}, beta_branch={res_dyn['beta_branching_mean']:.2f} "
        f"(ratio={res_dyn['beta_ratio_det_over_branch']:.2f}x) | CV(||ell||_2)={res_dyn['logit_norm_cv']:.4f} | "
        f"Gain: -{ce_gain:.4f} nats"
    )

    h1_pass = (
        res_static["logit_norm_cv"] < 0.08
        and res_dyn["logit_norm_cv"] >= 0.15
        and res_dyn["beta_ratio_det_over_branch"] >= 1.35
        and ce_gain >= 0.15
    )
    assert h1_pass, (
        f"H1 failed criteria: static_cv={res_static['logit_norm_cv']:.4f} (<0.08), "
        f"dyn_cv={res_dyn['logit_norm_cv']:.4f} (>=0.15), "
        f"beta_ratio={res_dyn['beta_ratio_det_over_branch']:.4f} (>=1.35), "
        f"ce_gain={ce_gain:.4f} (>=0.15)"
    )

    return {
        "static_global_beta": res_static,
        "dynamic_bloch_purity": res_dyn,
        "val_ce_improvement_nats": ce_gain,
        "passed": h1_pass,
    }


def run_h2_karpathy_memorization_and_classical_experiment(device: torch.device) -> dict[str, Any]:
    """Run Hypothesis 2: Karpathy Overfitting Capacity Scaling & ClassicalModdedGPT Verification."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 2 (H2): Karpathy 32-Record Memorization Capacity Scaling & ClassicalModdedGPT Baseline")
    print("=" * 100)

    N_records = 32
    T_seq = 64
    V_mem = 1024
    d_model = 128
    L_layers = 2
    H_heads = 4
    steps = 120

    torch.manual_seed(42)
    x_mem = torch.randint(N_records, V_mem, (N_records, T_seq), device=device, dtype=torch.long)
    x_mem[:, 0] = torch.arange(N_records, device=device, dtype=torch.long)
    y_mem = torch.randint(0, V_mem, (N_records, T_seq), device=device, dtype=torch.long)

    def _train_memorization_model(name: str, model: nn.Module, lr: float = 6e-3) -> dict[str, Any]:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)

        init_logits, init_loss = model(x_mem, y_mem)
        assert init_loss is not None
        init_ce = float(init_loss.detach().item())

        for _ in range(steps):
            model.train()
            opt.zero_grad(set_to_none=True)
            _, loss = model(x_mem, y_mem)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            final_logits, final_loss = model(x_mem, y_mem)
            assert final_loss is not None
            final_ce = float(final_loss.item())
            preds = final_logits.argmax(dim=-1)
            top1_acc = 100.0 * float((preds == y_mem).float().mean().item())
            norm_stats = compute_logit_norm_variation_metrics(final_logits)

        res = {
            "name": name,
            "params": int(n_params),
            "init_train_ce": round(init_ce, 4),
            "final_train_ce": round(final_ce, 4),
            "final_top1_acc": round(top1_acc, 2),
            "mean_max_prob": round(norm_stats["mean_max_prob"], 4),
            "mean_entropy_nats": round(norm_stats["mean_entropy_nats"], 4),
            "logit_norm_cv": round(norm_stats["logit_norm_cv"], 4),
        }
        print(
            f"  [{name:<38}] Params: {n_params:7,d} | Init CE: {init_ce:6.4f} -> "
            f"Final CE: {final_ce:6.4f} | Top-1 Acc: {top1_acc:6.2f}% | "
            f"P_max: {norm_stats['mean_max_prob']:.4f} | H(P): {norm_stats['mean_entropy_nats']:.4f} nats"
        )
        return res

    torch.manual_seed(42)
    q_d16_static = QuantumBornGPT(
        vocab_size=V_mem,
        embed_dim=d_model,
        seed_dim=16,
        num_heads=H_heads,
        num_layers=L_layers,
        hidden_dim=256,
        block_size=T_seq,
        readout_mode="tied_stiefel",
        povm_temperature=1.0 / 18.0,
        learnable_temperature=False,
        value_residual_mixing=True,
        input_highway_mixing=True,
        use_bigram_smear=True,
        use_qk_wave_filter=True,
        use_unet_interferometry=True,
        use_two_beam_mlp=True,
        use_dynamic_povm_beta=False,
    ).to(device)
    q_d16_static.wte.weight.requires_grad_(False)
    if q_d16_static.povm_head_seed_delta is not None:
        q_d16_static.povm_head_seed_delta.requires_grad_(False)
    if q_d16_static.tied_inv_temp is not None:
        q_d16_static.tied_inv_temp.requires_grad_(False)

    torch.manual_seed(42)
    q_d64_dyn = QuantumBornGPT(
        vocab_size=V_mem,
        embed_dim=d_model,
        seed_dim=64,
        num_heads=H_heads,
        num_layers=L_layers,
        hidden_dim=256,
        block_size=T_seq,
        readout_mode="tied_stiefel",
        povm_temperature=1.0 / 18.0,
        learnable_temperature=True,
        value_residual_mixing=True,
        input_highway_mixing=True,
        use_bigram_smear=True,
        use_qk_wave_filter=True,
        use_unet_interferometry=True,
        use_two_beam_mlp=True,
        use_overcomplete_seed_adapter=True,
        use_dynamic_povm_beta=True,
    ).to(device)
    q_d64_dyn.wte.weight.requires_grad_(False)
    if q_d64_dyn.povm_head_seed_delta is not None:
        q_d64_dyn.povm_head_seed_delta.requires_grad_(False)

    torch.manual_seed(42)
    classical_modded = ClassicalModdedGPT(
        vocab_size=V_mem,
        embed_dim=d_model,
        num_heads=H_heads,
        num_layers=L_layers,
        hidden_dim=256,
        block_size=T_seq,
        use_bigram_smear=True,
        use_unet_interferometry=True,
        value_residual_mixing=True,
        input_highway_mixing=True,
        logit_softcap=30.0,
    ).to(device)

    res_d16 = _train_memorization_model("QuantumBornGPT(D_seed=16, static_beta)", q_d16_static, lr=6e-3)
    res_d64 = _train_memorization_model("QuantumBornGPT(D_seed=64, Adapter+DynBeta)", q_d64_dyn, lr=6e-3)
    res_classical = _train_memorization_model("ClassicalModdedGPT(L=2, d=128, softcap=30)", classical_modded, lr=6e-3)

    ce_reduction_ratio = round(res_d64["final_train_ce"] / max(res_d16["final_train_ce"], 1e-6), 4)
    h2_pass = (
        res_d64["final_train_ce"] <= 0.50 * res_d16["final_train_ce"]
        and res_d64["final_top1_acc"] >= 98.0
        and res_classical["final_top1_acc"] >= 98.0
        and res_classical["final_train_ce"] <= 0.10
    )
    assert h2_pass, (
        f"H2 failed criteria: d64_ce={res_d64['final_train_ce']:.4f} vs d16_ce={res_d16['final_train_ce']:.4f}, "
        f"d64_top1={res_d64['final_top1_acc']:.2f}%, classical_top1={res_classical['final_top1_acc']:.2f}%, "
        f"classical_ce={res_classical['final_train_ce']:.4f}"
    )

    return {
        "quantum_d16_static": res_d16,
        "quantum_d64_dynamic": res_d64,
        "classical_modded_gpt": res_classical,
        "d64_over_d16_ce_ratio": ce_reduction_ratio,
        "passed": h2_pass,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXPERIMENT_PLAN_013 Controlled Verification on device={device}...")

    h1_results = run_h1_heteroscedastic_sharpness_experiment(device)
    h2_results = run_h2_karpathy_memorization_and_classical_experiment(device)

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "plan": "EXPERIMENT_PLAN_013",
        "device": str(device),
        "elapsed_sec": elapsed,
        "H1_heteroscedastic_sharpness": h1_results,
        "H2_karpathy_memorization_and_classical": h2_results,
        "all_passed": bool(h1_results["passed"] and h2_results["passed"]),
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_013.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 100)
    print(f"ALL PLAN 013 HYPOTHESES VERIFIED IN {elapsed:.2f}s -> Saved to {out_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
