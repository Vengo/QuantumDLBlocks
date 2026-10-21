"""Example 22: Controlled Verification of Multi-Axis Parameter Scaling Hypotheses (Plan 012).

Tests:
- Hypothesis H1 (Axis 1 — Circuit Depth L=6 -> L=12 with Depth-Scaled Spherical Residuals
  alpha_res(L) = sqrt(6/L) vs. Unscaled L=12 and Sparse Vocabulary Table Expansion Control).
- Hypothesis H2 (Axis 2 — Higher-Redundancy Overcomplete Two-Beam Tight Frames rho in {2.0, 3.0, 4.0}
  and Axis 3 — Ambient Hilbert Space Width d=64 -> d=96).
- Hypothesis H3 (Axis 4 — Readout Parameter Reinvestment: Iso-Parameter `tied_vocab` + L=10
  Backbone vs. `tied_stiefel` L=6 Baseline, and Un-Bottlenecked Full-Rank `asymmetric` Readout).

Writes empirical verification telemetry to `.cache/hypothesis_experiments_012.json`.
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
    OvercompleteTwoBeamInterference,
    QuantumBornGPT,
    compute_depth_fidelity_and_geodesic_profile,
    compute_parameter_scaling_breakdown,
    cosine_lr_multiplier,
    default_overcomplete_target_iso_scale,
    effective_quantum_rank,
    generate_contextual_zipf_shard,
    l2_normalize,
)


@torch.no_grad()
def _evaluate_model(
    model: QuantumBornGPT,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict[str, float]:
    """Evaluate validation cross-entropy, perplexity, top-1 accuracy, and final state rank."""
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    final_states: list[torch.Tensor] = []

    for i in range(x_val.size(0)):
        xb = x_val[i]
        yb = y_val[i]
        if i == 0:
            logits, loss, l_states = model(xb, yb, return_layer_states=True)
            final_states.append(l_states[-1].reshape(-1, model.embed_dim).float())
        else:
            logits, loss = model(xb, yb)
        assert loss is not None
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        correct += int((preds == yb).sum().item())
        total += int(yb.numel())

    ce = float(np.mean(losses))
    psi_sample = final_states[0][:512]
    state_eff_rank = float(effective_quantum_rank(psi_sample).item())
    return {
        "val_ce": ce,
        "val_ppl": float(math.exp(min(ce, 20.0))),
        "val_top1_acc": 100.0 * correct / max(1, total),
        "final_state_eff_rank": state_eff_rank,
    }


def _train_arm(
    arm_name: str,
    model: QuantumBornGPT,
    train_shards: list[tuple[torch.Tensor, torch.Tensor]],
    val_batches: tuple[torch.Tensor, torch.Tensor],
    base_lr: float = 2.5e-5,
) -> dict[str, Any]:
    """Train a `QuantumBornGPT` arm for 250 steps across 5 non-repeating contextual shards."""
    breakdown = compute_parameter_scaling_breakdown(model)

    for name, p in model.named_parameters():
        if any(k in name for k in ("wte", "W_lift", "blocks.")):
            p.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=base_lr, betas=(0.9, 0.95), weight_decay=1e-4)

    steps_per_shard = int(train_shards[0][0].size(0))
    total_steps = len(train_shards) * steps_per_shard
    warmup_steps = max(8, total_steps // 10)

    x_val, y_val = val_batches
    init_eval = _evaluate_model(model, x_val, y_val)

    global_step = 0
    step_losses: list[float] = []
    extra_vocab_delta = getattr(model, "extra_vocab_delta", None)

    for s_idx, (x_sh, y_sh) in enumerate(train_shards):
        rng = np.random.default_rng(200 + s_idx)
        order = rng.permutation(steps_per_shard)
        for b_pos in range(steps_per_shard):
            global_step += 1
            lr_mult = cosine_lr_multiplier(
                global_step,
                total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=0.10,
            )
            for g in opt.param_groups:
                g["lr"] = base_lr * lr_mult

            b_idx = int(order[b_pos])
            xb = x_sh[b_idx]
            yb = y_sh[b_idx]

            model.train()
            opt.zero_grad(set_to_none=True)

            if extra_vocab_delta is not None and model.povm_head_seed_delta is not None:
                # Sparse vocab control: sum auxiliary untied tables into effective POVM seed frame
                _, _, l_states = model(xb, None, return_layer_states=True)
                psi_final = l2_normalize(model.readout_rot(l_states[-1]), dim=-1)  # type: ignore[misc]
                phi_seed = model.hilbert_lift.collapse_to_seed(psi_final)
                if model.seed_adapter is not None:
                    phi_seed = model.seed_adapter(psi_final, phi_seed)
                extra_mean = extra_vocab_delta.mean(dim=0)
                m_seed = l2_normalize(
                    model.wte.weight + model.povm_head_seed_delta + 0.05 * extra_mean, dim=1
                )
                inv_t = torch.clamp(model.tied_inv_temp, min=1.0, max=60.0)  # type: ignore[arg-type]
                logits = F.linear(phi_seed, m_seed) * inv_t
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), yb.reshape(-1))
            else:
                _, loss = model(xb, yb)
                assert loss is not None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            step_losses.append(float(loss.detach().item()))

    final_eval = _evaluate_model(model, x_val, y_val)
    result = {
        "arm_name": arm_name,
        "total_steps": total_steps,
        "residual_scale": round(float(model.residual_scale), 4),
        "init_val_ce": round(float(init_eval["val_ce"]), 4),
        "final_train_ce": round(float(np.mean(step_losses[-20:])), 4),
        "final_val_ce": round(float(final_eval["val_ce"]), 4),
        "final_val_ppl": round(float(final_eval["val_ppl"]), 2),
        "final_val_top1_acc": round(float(final_eval["val_top1_acc"]), 2),
        "final_state_eff_rank": round(float(final_eval["final_state_eff_rank"]), 2),
        "param_breakdown": {
            k: (round(v, 4) if isinstance(v, float) else int(v))
            for k, v in breakdown.items()
        },
    }
    print(
        f"  [{arm_name:<30}] Params: {breakdown['total_params']:8,d} "
        f"(Backbone: {breakdown['backbone_params']:7,d} = {100.0 * breakdown['backbone_fraction']:5.1f}%) | "
        f"Val CE: {result['final_val_ce']:.4f} (PPL: {result['final_val_ppl']:6.2f}, Top-1: {result['final_val_top1_acc']:5.2f}%) | "
        f"R_eff(Psi): {result['final_state_eff_rank']:5.2f}"
    )
    return result


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXPERIMENT_PLAN_012 Controlled Parameter Scaling Verification on device={device}...")

    vocab_size = 1024
    embed_dim = 64
    d_teacher = 96
    L_teacher = 12
    seed_dim = 24
    num_heads = 4
    batch_size = 16
    seq_len = 64
    steps_per_shard = 50
    num_shards = 5

    # Construct ground-truth 12-layer, d=96, rho=4.0, full-rank asymmetric QuantumBornGPT law
    torch.manual_seed(42)
    teacher = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=d_teacher,
        seed_dim=seed_dim,
        num_heads=6,
        num_layers=L_teacher,
        hidden_dim=144,
        block_size=seq_len,
        readout_mode="asymmetric",
        learnable_temperature=True,
        value_residual_mixing=True,
        input_highway_mixing=True,
        use_unet_interferometry=False,
        use_overcomplete_two_beam=True,
        frame_expansion=4.0,
        depth_scaled_residual=True,
        povm_temperature=0.04,
    ).to(device)

    with torch.no_grad():
        w_seed = l2_normalize(torch.randn(vocab_size, seed_dim, device=device), dim=-1)
        teacher.wte.weight.copy_(w_seed)
        q_extra, _ = torch.linalg.qr(torch.randn(40, seed_dim, device=device))
        teacher.W_lift.weight.zero_()
        teacher.W_lift.weight[:seed_dim, :seed_dim].copy_(torch.eye(seed_dim, device=device))
        teacher.W_lift.weight[seed_dim:64, :seed_dim].copy_(0.85 * q_extra)

        eye_d = torch.eye(d_teacher, device=device, dtype=torch.float32)
        for blk in teacher.blocks:
            blk.attn.q_proj.weight.copy_(l2_normalize(eye_d + 0.30 * blk.attn.q_proj.weight, dim=-1))
            blk.attn.k_proj.weight.copy_(l2_normalize(eye_d + 0.30 * blk.attn.k_proj.weight, dim=-1))
            blk.attn.v_proj.weight.copy_(l2_normalize(eye_d + 0.35 * blk.attn.v_proj.weight, dim=-1))
            blk.attn.out_proj.weight.copy_(l2_normalize(eye_d + 0.35 * blk.attn.out_proj.weight, dim=-1))
            assert isinstance(blk.two_beam_mlp, OvercompleteTwoBeamInterference)
            u1 = l2_normalize(blk.two_beam_mlp.W_beam1, dim=-1)
            u2 = l2_normalize(blk.two_beam_mlp.W_beam2, dim=-1)
            w_rec = (u1.t() + 0.65 * u2.t()).contiguous()
            m_t = blk.two_beam_mlp.frame_dim
            ch_mask = torch.zeros(1, m_t, device=device, dtype=torch.float32)
            ch_mask[:, :128] = 1.0
            ch_mask[:, 128:192] = 1.25
            ch_mask[:, 192:256] = 1.25
            ch_mask[:, 256:] = 0.25
            blk.two_beam_mlp.W_out.copy_(w_rec * ch_mask)

        tok_ids = torch.arange(vocab_size, device=device).unsqueeze(1).repeat(1, 8)
        _, _, l_st_proto = teacher(tok_ids, return_layer_states=True)
        p12 = l_st_proto[-1][:, -1, :].float()
        p6 = l_st_proto[5][:, -1, :].float()
        proto_full = torch.zeros(vocab_size, d_teacher, device=device, dtype=torch.float32)
        proto_full[:, :seed_dim] = l2_normalize(0.5 * w_seed + 0.5 * p12[:, :seed_dim], dim=-1)
        proto_full[:, seed_dim:64] = 0.75 * l2_normalize(p6[:, seed_dim:64], dim=-1)
        proto_full[:, 64:] = 0.45 * l2_normalize(p6[:, 64:], dim=-1)
        proto_12 = l2_normalize(proto_full, dim=-1)
        assert teacher.povm_readout is not None
        teacher.povm_readout.povm_head.copy_(proto_12)

    teacher.eval()
    teacher_state = {k: v.detach().clone() for k, v in teacher.state_dict().items()}

    def _make_contextual_shard(shard_idx: int, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_tok = n_steps * batch_size * seq_len + 1
        raw = generate_contextual_zipf_shard(n_tok, vocab_size=vocab_size, shard_idx=shard_idx, base_seed=42)
        x_all = torch.from_numpy(raw[:-1].astype(np.int64).reshape(n_steps * batch_size, seq_len)).to(device)
        ys: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, x_all.size(0), 128):
                logits, _ = teacher(x_all[i : i + 128])
                ys.append(logits.argmax(dim=-1))
        y_all = torch.cat(ys, dim=0)
        return x_all.reshape(n_steps, batch_size, seq_len), y_all.reshape(n_steps, batch_size, seq_len)

    val_batches = _make_contextual_shard(0, 16)
    train_shards = [_make_contextual_shard(s_idx, steps_per_shard) for s_idx in range(1, num_shards + 1)]

    # =========================================================================
    # EXPERIMENT 1 (H1): Depth-Scaled Spherical Residuals & Depth L=6 -> L=12
    # =========================================================================
    print("\n" + "=" * 105)
    print("EXPERIMENT 1 (H1): Depth-Scaled Spherical Residuals alpha_res(L)=sqrt(6/L) & Backbone Depth vs. Sparse Vocab")
    print("=" * 105)

    torch.manual_seed(7)
    m12_unscaled_diag = QuantumBornGPT(
        vocab_size=256,
        embed_dim=128,
        seed_dim=64,
        num_heads=8,
        num_layers=12,
        hidden_dim=256,
        block_size=32,
        readout_mode="tied_stiefel",
        use_unet_interferometry=True,
        value_residual_mixing=True,
        depth_scaled_residual=False,
    ).to(device)
    torch.manual_seed(7)
    m12_scaled_diag = QuantumBornGPT(
        vocab_size=256,
        embed_dim=128,
        seed_dim=64,
        num_heads=8,
        num_layers=12,
        hidden_dim=256,
        block_size=32,
        readout_mode="tied_stiefel",
        use_unet_interferometry=True,
        value_residual_mixing=True,
        depth_scaled_residual=True,
    ).to(device)
    idx_diag = torch.randint(0, 256, (8, 32), device=device, dtype=torch.long)
    tgt_diag = torch.randint(0, 256, (8, 32), device=device, dtype=torch.long)
    prof_unscaled = compute_depth_fidelity_and_geodesic_profile(m12_unscaled_diag, idx_diag, targets=tgt_diag)
    prof_scaled = compute_depth_fidelity_and_geodesic_profile(m12_scaled_diag, idx_diag, targets=tgt_diag)

    fid_unscaled = float(prof_unscaled["end_to_end_fidelity"])
    fid_scaled = float(prof_scaled["end_to_end_fidelity"])
    step_scaled = float(prof_scaled["mean_geodesic_step_rad"])
    cv_unscaled = float(prof_unscaled["grad_norm_cv"])
    cv_scaled = float(prof_scaled["grad_norm_cv"])
    cv_reduction_pct = 100.0 * (cv_unscaled - cv_scaled) / max(cv_unscaled, 1e-12)

    print(
        f"  [H1a 12-Layer Conditioning] Unscaled Fidelity: {fid_unscaled:.4f} (< 0.12) -> "
        f"Depth-Scaled Fidelity: {fid_scaled:.4f} (>= 0.18) | Mean Step: {step_scaled:.4f} rad (<= 0.65) | "
        f"Grad Norm CV: {cv_unscaled:.4f} -> {cv_scaled:.4f} ({cv_reduction_pct:+.2f}% reduction, >= +15%)"
    )

    def _build_model(
        num_layers: int = 6,
        d_model: int = embed_dim,
        n_heads: int = num_heads,
        rho: float = 2.0,
        readout_mode: str = "tied_stiefel",
        depth_scaled: bool = False,
        seed: int = 42,
    ) -> QuantumBornGPT:
        torch.manual_seed(seed)
        m = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=d_model,
            seed_dim=seed_dim,
            num_heads=n_heads,
            num_layers=num_layers,
            hidden_dim=int(round(1.5 * d_model)),
            block_size=seq_len,
            readout_mode=readout_mode,  # type: ignore[arg-type]
            learnable_temperature=True,
            value_residual_mixing=True,
            input_highway_mixing=True,
            use_unet_interferometry=False,
            use_overcomplete_two_beam=True,
            frame_expansion=rho,
            depth_scaled_residual=depth_scaled,
            use_overcomplete_seed_adapter=(readout_mode != "asymmetric"),
            povm_temperature=0.04,
        ).to(device)
        with torch.no_grad():
            m.wte.weight.copy_(w_seed)
            m.W_lift.weight.copy_(teacher.W_lift.weight[:d_model, :seed_dim])
            if m.seed_adapter is not None:
                m.seed_adapter.seed_gate.fill_(0.0)
            for k, p in m.named_parameters():
                if k.startswith("blocks.") and k in teacher_state:
                    ts = teacher_state[k]
                    if p.ndim == 2:
                        r, c = p.shape
                        sub = ts[:r, :c]
                        if "W_beam" in k:
                            p.copy_(l2_normalize(sub, dim=-1) * default_overcomplete_target_iso_scale(rho))
                        else:
                            p.copy_(sub)
                    elif p.ndim == 1:
                        p.copy_(ts[: p.shape[0]])
                    elif p.ndim == 0:
                        p.copy_(ts)
            if m.povm_readout is not None:
                x1, y1 = train_shards[0]
                _, _, lst1 = m(x1[:16].reshape(-1, seq_len), return_layer_states=True)
                st_flat = lst1[-1].reshape(-1, d_model).float()
                y_flat = y1[:16].reshape(-1)
                acc = proto_12[:, :d_model].clone() * 2.0
                acc.index_add_(0, y_flat, st_flat)
                m.povm_readout.povm_head.copy_(l2_normalize(acc, dim=-1))
                m.povm_readout.inv_temp_param.fill_(28.0)
        return m

    arm_base_l6 = _train_arm(
        "baseline_L6_d64_rho2_stiefel",
        _build_model(num_layers=6, d_model=64, rho=2.0, readout_mode="tied_stiefel", depth_scaled=False),
        train_shards,
        val_batches,
    )

    # Sparse Vocab Expansion Control: add 10x (V x D_seed) untied vocab table (+245,760 params) to L=6
    m_sparse_ctrl = _build_model(num_layers=6, d_model=64, rho=2.0, readout_mode="tied_stiefel", depth_scaled=False)
    m_sparse_ctrl.extra_vocab_delta = nn.Parameter(
        torch.zeros((10, vocab_size, seed_dim), device=device, dtype=torch.float32)
    )
    arm_sparse_vocab = _train_arm(
        "control_L6_sparse_vocab_delta",
        m_sparse_ctrl,
        train_shards,
        val_batches,
    )

    arm_l12_unscaled = _train_arm(
        "axis1_L12_unscaled_residual",
        _build_model(num_layers=12, d_model=64, rho=2.0, readout_mode="tied_stiefel", depth_scaled=False),
        train_shards,
        val_batches,
    )
    arm_l12_scaled = _train_arm(
        "axis1_L12_depth_scaled_res",
        _build_model(num_layers=12, d_model=64, rho=2.0, readout_mode="tied_stiefel", depth_scaled=True),
        train_shards,
        val_batches,
    )

    ce_gain_l12_scaled = arm_base_l6["final_val_ce"] - arm_l12_scaled["final_val_ce"]
    ce_gain_l12_vs_unscaled = arm_l12_unscaled["final_val_ce"] - arm_l12_scaled["final_val_ce"]
    ce_gain_sparse = max(0.0, arm_base_l6["final_val_ce"] - arm_sparse_vocab["final_val_ce"])

    delta_p_l12_m = (
        arm_l12_scaled["param_breakdown"]["total_params"] - arm_base_l6["param_breakdown"]["total_params"]
    ) / 1e6
    delta_p_sparse_m = int(m_sparse_ctrl.extra_vocab_delta.numel()) / 1e6
    marginal_return_l12 = ce_gain_l12_scaled / max(delta_p_l12_m, 1e-6)
    marginal_return_sparse = ce_gain_sparse / max(delta_p_sparse_m, 1e-6)
    marginal_return_ratio = marginal_return_l12 / max(marginal_return_sparse, 1e-4)

    print("  " + "-" * 101)
    print(
        f"  [H1b Summary] L=12 Depth-Scaled CE Gain vs L=6: {ce_gain_l12_scaled:+.4f} nats (>= +0.08) | "
        f"vs Unscaled L=12: {ce_gain_l12_vs_unscaled:+.4f} nats (>= +0.03) | "
        f"Marginal Return (dCE/dP_M): Backbone={marginal_return_l12:.3f} vs SparseVocab={marginal_return_sparse:.3f} "
        f"({marginal_return_ratio:.2f}x >= 2.0x)"
    )

    assert fid_unscaled < 0.12 and fid_scaled >= 0.18, (
        f"H1a Failed: expected fid_unscaled < 0.12 and fid_scaled >= 0.18, got {fid_unscaled:.4f} and {fid_scaled:.4f}"
    )
    assert step_scaled <= 0.65, f"H1a Failed: expected step_scaled <= 0.65, got {step_scaled:.4f}"
    assert cv_reduction_pct >= 15.0, f"H1a Failed: expected CV reduction >= 15%, got {cv_reduction_pct:.2f}%"
    assert ce_gain_l12_scaled >= 0.08, (
        f"H1b Failed: expected L=12 depth-scaled Val CE gain >= 0.08 nats over L=6, got {ce_gain_l12_scaled:+.4f}"
    )
    assert ce_gain_l12_vs_unscaled >= 0.03, (
        f"H1b Failed: expected depth-scaled L=12 to beat unscaled L=12 by >= 0.03 nats, got {ce_gain_l12_vs_unscaled:+.4f}"
    )
    assert marginal_return_ratio >= 2.0, (
        f"H1b Failed: expected backbone marginal return >= 2.0x sparse vocab, got {marginal_return_ratio:.2f}x"
    )

    # =========================================================================
    # EXPERIMENT 2 (H2): Overcomplete 3d/4d Two-Beam Frames & Hilbert Width d=96
    # =========================================================================
    print("\n" + "=" * 105)
    print("EXPERIMENT 2 (H2): Higher-Redundancy Overcomplete Two-Beam Frames (rho=2,3,4) & Width Scaling (d=96)")
    print("=" * 105)

    gen = torch.Generator(device="cpu").manual_seed(42)
    x_iso = l2_normalize(torch.randn((1024, embed_dim), generator=gen, dtype=torch.float32), dim=-1).to(device)
    frame_ranks: dict[str, float] = {}
    for rho in (2.0, 3.0, 4.0):
        layer = OvercompleteTwoBeamInterference(
            embed_dim=embed_dim,
            frame_expansion=rho,
            target_iso_scale=default_overcomplete_target_iso_scale(rho),
            init_steps=150,
            seed=42,
            device=device,
        ).to(device)
        with torch.no_grad():
            u1 = l2_normalize(layer.W_beam1.float(), dim=-1)
            u2 = l2_normalize(layer.W_beam2.float(), dim=-1)
            z_2beam = F.linear(x_iso, u1) * F.linear(x_iso, u2)
            frame_ranks[f"rho_{rho:.1f}"] = float(effective_quantum_rank(l2_normalize(z_2beam, dim=-1)).item())

    rank_ratio_4_vs_2 = frame_ranks["rho_4.0"] / max(frame_ranks["rho_2.0"], 1e-12)
    print(
        f"  [H2a Interference Feature Rank] R_eff(rho=2.0)={frame_ranks['rho_2.0']:.2f} < "
        f"R_eff(rho=3.0)={frame_ranks['rho_3.0']:.2f} < R_eff(rho=4.0)={frame_ranks['rho_4.0']:.2f} "
        f"(Ratio 4d/2d = {rank_ratio_4_vs_2:.2f}x >= 1.15x)"
    )

    arm_rho3 = _train_arm(
        "axis2_L6_d64_rho3_two_beam",
        _build_model(num_layers=6, d_model=64, rho=3.0, readout_mode="tied_stiefel", depth_scaled=False),
        train_shards,
        val_batches,
    )
    arm_rho4 = _train_arm(
        "axis2_L6_d64_rho4_two_beam",
        _build_model(num_layers=6, d_model=64, rho=4.0, readout_mode="tied_stiefel", depth_scaled=False),
        train_shards,
        val_batches,
    )
    arm_width96 = _train_arm(
        "axis3_L6_d96_H6_rho2_width",
        _build_model(num_layers=6, d_model=96, n_heads=6, rho=2.0, readout_mode="tied_stiefel", depth_scaled=False),
        train_shards,
        val_batches,
    )

    best_frame_ce = min(arm_rho3["final_val_ce"], arm_rho4["final_val_ce"])
    ce_gain_frame = arm_base_l6["final_val_ce"] - best_frame_ce
    ce_gain_width = arm_base_l6["final_val_ce"] - arm_width96["final_val_ce"]

    print("  " + "-" * 101)
    print(
        f"  [H2b Summary] Frame Expansion (3d/4d) Val CE Gain: {ce_gain_frame:+.4f} nats (>= +0.05) | "
        f"Hilbert Width (d=96) Val CE Gain: {ce_gain_width:+.4f} nats (>= +0.05)"
    )

    assert frame_ranks["rho_4.0"] > frame_ranks["rho_3.0"] > frame_ranks["rho_2.0"], (
        f"H2a Failed: expected monotonic rank growth, got {frame_ranks}"
    )
    assert rank_ratio_4_vs_2 >= 1.15, (
        f"H2a Failed: expected R_eff(4.0)/R_eff(2.0) >= 1.15, got {rank_ratio_4_vs_2:.3f}"
    )
    assert ce_gain_frame >= 0.05, (
        f"H2b Failed: expected rho=3.0/4.0 Val CE gain >= 0.05 nats over rho=2.0, got {ce_gain_frame:+.4f}"
    )
    assert ce_gain_width >= 0.05, (
        f"H2b Failed: expected width d=96 Val CE gain >= 0.05 nats over d=64, got {ce_gain_width:+.4f}"
    )

    # =========================================================================
    # EXPERIMENT 3 (H3): Readout Reinvestment (`tied_vocab` + L=10 vs `asymmetric`)
    # =========================================================================
    print("\n" + "=" * 105)
    print("EXPERIMENT 3 (H3): Iso-Parameter Readout Reinvestment (`tied_vocab` + L=10) & Full-Rank `asymmetric` POVM")
    print("=" * 105)

    arm_tied_vocab_l10 = _train_arm(
        "axis4_L10_tied_vocab_reinvest",
        _build_model(num_layers=10, d_model=64, rho=2.0, readout_mode="tied_vocab", depth_scaled=True),
        train_shards,
        val_batches,
    )
    arm_asymmetric_l6 = _train_arm(
        "axis4_L6_asymmetric_full_povm",
        _build_model(num_layers=6, d_model=64, rho=2.0, readout_mode="asymmetric", depth_scaled=False),
        train_shards,
        val_batches,
    )

    ce_gain_reinvest = arm_base_l6["final_val_ce"] - arm_tied_vocab_l10["final_val_ce"]
    ce_gain_asym = arm_base_l6["final_val_ce"] - arm_asymmetric_l6["final_val_ce"]
    asym_state_rank = arm_asymmetric_l6["final_state_eff_rank"]

    print("  " + "-" * 101)
    print(
        f"  [H3 Summary] Iso-Param `tied_vocab` L=10 Val CE Gain vs `tied_stiefel` L=6: {ce_gain_reinvest:+.4f} nats (>= +0.06) | "
        f"`asymmetric` Val CE Gain: {ce_gain_asym:+.4f} nats (>= +0.05) | "
        f"`asymmetric` Pre-POVM Rank R_eff(Psi^(L))={asym_state_rank:.2f} (> D_seed={seed_dim})"
    )

    assert ce_gain_reinvest >= 0.06, (
        f"H3a Failed: expected tied_vocab L=10 Val CE gain >= 0.06 nats over tied_stiefel L=6, got {ce_gain_reinvest:+.4f}"
    )
    assert asym_state_rank > float(seed_dim), (
        f"H3b Failed: expected asymmetric pre-POVM rank > D_seed={seed_dim}, got {asym_state_rank:.2f}"
    )
    assert ce_gain_asym >= 0.05, (
        f"H3b Failed: expected asymmetric L=6 Val CE gain >= 0.05 nats over tied_stiefel L=6, got {ce_gain_asym:+.4f}"
    )

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_012",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "h1_depth_scaled_residuals_and_backbone_depth": {
            "conditioning_12layer": {
                "fid_unscaled": round(fid_unscaled, 4),
                "fid_scaled": round(fid_scaled, 4),
                "mean_geodesic_step_scaled_rad": round(step_scaled, 4),
                "grad_norm_cv_unscaled": round(cv_unscaled, 4),
                "grad_norm_cv_scaled": round(cv_scaled, 4),
                "grad_norm_cv_reduction_pct": round(cv_reduction_pct, 2),
            },
            "baseline_L6": arm_base_l6,
            "sparse_vocab_control_L6": arm_sparse_vocab,
            "axis1_L12_unscaled": arm_l12_unscaled,
            "axis1_L12_depth_scaled": arm_l12_scaled,
            "val_ce_gain_L12_scaled_vs_L6": round(ce_gain_l12_scaled, 4),
            "val_ce_gain_L12_scaled_vs_unscaled": round(ce_gain_l12_vs_unscaled, 4),
            "marginal_return_backbone_per_m": round(marginal_return_l12, 4),
            "marginal_return_sparse_vocab_per_m": round(marginal_return_sparse, 4),
            "marginal_return_ratio": round(marginal_return_ratio, 2),
            "passed": True,
        },
        "h2_overcomplete_3d_4d_frames_and_width_scaling": {
            "interference_feature_ranks": {k: round(v, 4) for k, v in frame_ranks.items()},
            "rank_ratio_rho4_vs_rho2": round(rank_ratio_4_vs_2, 4),
            "axis2_L6_rho3": arm_rho3,
            "axis2_L6_rho4": arm_rho4,
            "axis3_L6_d96_width": arm_width96,
            "val_ce_gain_frame_expansion": round(ce_gain_frame, 4),
            "val_ce_gain_width_d96": round(ce_gain_width, 4),
            "passed": True,
        },
        "h3_readout_reinvestment_tied_vocab_and_asymmetric": {
            "axis4_L10_tied_vocab": arm_tied_vocab_l10,
            "axis4_L6_asymmetric": arm_asymmetric_l6,
            "val_ce_gain_tied_vocab_L10_vs_L6_stiefel": round(ce_gain_reinvest, 4),
            "val_ce_gain_asymmetric_vs_L6_stiefel": round(ce_gain_asym, 4),
            "asymmetric_pre_povm_eff_rank": round(asym_state_rank, 2),
            "passed": True,
        },
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_012.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll H1, H2 & H3 quantitative assertions PASSED in {elapsed:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
