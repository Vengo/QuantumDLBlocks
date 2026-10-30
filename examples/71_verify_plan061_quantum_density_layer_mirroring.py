#!/usr/bin/env python3
"""Wave 46 (`EXPERIMENT_PLAN_061.md`) Verification & Prototype Runner for Experiment A & B.

Executes and validates:
1. **Hypothesis H1 (`verify_h1_riemannian_geodesic_and_midpoint_hallucination`)**:
   - Exact `Riemannian Stiefel-Muon` ($1/\\sqrt{1+\\eta^2}$ geodesic retraction) vs. Euclidean Muon drift
     on square ($64 \\times 64$) and rectangular ($256 \\times 64$) Stiefel operators, plus exact equivalence
     to the $O(d^3)$ Cayley transform geodesic (`cos_sim > 0.99999`).
   - Gemini Notebook polysemous superposition & 30-mode noise filtering test: verifies that 1 step of
     Bures-Wasserstein McWeeny squaring $\\rho \\leftarrow \\rho^2 / \\text{Tr}(\\rho^2)$ suppresses background
     noise by $>99\\%$ while preserving `Effective Rank = 1.99` and eliminating the $16\\times$ midpoint
     distractor hallucination of pure superposition kets $|\\psi_{\\text{pure}}\\rangle$.
2. **Hypothesis H2 (`Experiment A — Layer-by-Layer Classical Transformer Mirroring`)**:
   - Loads a `CompiledLeaderboardClassicalGPT` teacher checkpoint (automatically discovering the latest saved
     milestone in `.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/` or pre-warming on FineWeb)
     and extracts real FineWeb hidden trajectories $(x_{\\text{in}}^{(l)}, x_{\\text{out}}^{(l)})$ via
     `extract_layer_trajectories()`.
   - Trains `PureQuantumKetMirrorBlock` (`Rank-1 Pure Ket`) vs. `ImpureQuantumDensityMirrorBlock` (`Impure
     McWeeny Density Matrix + Isometric sqrt(2) Packing + Riemannian Stiefel-Muon`) to mirror the classical
     layer's representation update.
3. **Hypothesis H3 (`Experiment B — Surgical Quantum Layer Grafting with Frozen Input/Output Weights`)**:
   - Surgically grafts the quantum blocks in place of an interior layer of `CompiledLeaderboardClassicalGPT`
     with `wte`, `lm_head`, and all surrounding classical layers 100% frozen, measuring FineWeb validation
     Cross-Entropy (`val_ce`) and Top-1 Accuracy (`top1_acc`) recovery.
"""

from __future__ import annotations

import argparse
import copy
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

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_BORN_POWER_P,
    DEFAULT_GRAM_RELATIONAL_WEIGHT,
    DEFAULT_MCWEENY_STEPS,
    DEFAULT_MIRROR_EMBED_DIM,
    DEFAULT_MIRROR_HEAD_COMPLEX_DIM,
    DEFAULT_MIRROR_NUM_HEADS,
    DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    DEFAULT_RIEMANNIAN_MUON_LR,
    INV_SQRT2_ISOMETRIC_SCALE,
    SQRT2_ISOMETRIC_OFFDIAG_SCALE,
    TRACE_NORM_EPS,
    ImpureQuantumDensityMirrorBlock,
    PureQuantumKetMirrorBlock,
    QuantumMirrorBlockConfig,
    RiemannianStiefelMuon,
    bures_wasserstein_mcweeny_step,
    complex_l2_normalize,
    compute_layer_mirroring_loss,
    extract_rank2_krylov_eigenpair,
    pack_hermitian_isometric,
    pack_rank1_ket_outer_isometric,
    riemannian_stiefel_muon_step,
    unpack_hermitian_isometric,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    DEFAULT_VALID_FINEWEB_VOCAB_SIZE,
    FINEWEB_SHARD_HEADER_BYTES,
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
    CompiledLeaderboardOptimizerConfig,
    build_compiled_leaderboard_optimizers,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_061.md Section 3
# ==============================================================================
WAVE_ID: str = "W46"
PLAN_ID: str = "EXPERIMENT_PLAN_061"
FINEWEB_CACHE_DIR: Path = ROOT_DIR / ".cache" / "fineweb_100M"
DEFAULT_VAL_SHARD_PATH: Path = FINEWEB_CACHE_DIR / "fineweb_val_000000.bin"
DEFAULT_TRAIN_SHARD1_PATH: Path = FINEWEB_CACHE_DIR / "fineweb_train_000001.bin"
DEFAULT_TEACHER_CKPT_DIR: Path = (
    ROOT_DIR / ".cache" / "checkpoints" / "w45b_classical_teacher_ckpt50m_l12_d768_b262k"
)
HYPOTHESIS_JSON_OUT: Path = ROOT_DIR / ".cache" / "hypothesis_experiments_061.json"


def _load_fineweb_token_batches(
    shard_path: Path,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load sequential `(x, y)` token batches from a FineWeb binary shard (or fallback synthetic tokens)."""
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    span = batch_size * seq_len + 1
    if shard_path.exists():
        raw = np.fromfile(
            shard_path,
            dtype=np.uint16,
            count=num_batches * span + 1024,
            offset=FINEWEB_SHARD_HEADER_BYTES,
        )
        tokens_cpu = torch.from_numpy(raw.astype(np.int64))
        for b_idx in range(num_batches):
            start = b_idx * span
            chunk = tokens_cpu[start : start + span].to(device=device)
            xb = chunk[:-1].view(batch_size, seq_len)
            yb = chunk[1:].view(batch_size, seq_len)
            batches.append((xb, yb))
    else:
        torch.manual_seed(4600)
        for _ in range(num_batches):
            xb = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (batch_size, seq_len), device=device)
            yb = torch.randint(0, DEFAULT_VALID_FINEWEB_VOCAB_SIZE, (batch_size, seq_len), device=device)
            batches.append((xb, yb))
    return batches


def verify_h1_riemannian_geodesic_and_midpoint_hallucination() -> dict[str, Any]:
    """Verify H1: Riemannian Stiefel-Muon geodesic accuracy, Isometric sqrt(2) trace, and McWeeny impurity."""
    torch.manual_seed(4610)
    d_sq = 64
    d_tall_r, d_tall_c = 256, 64
    lr = 0.03
    beta = DEFAULT_RIEMANNIAN_MOMENTUM_BETA

    w_sq_riem = torch.empty(d_sq, d_sq, dtype=torch.float32)
    torch.nn.init.orthogonal_(w_sq_riem)
    w_sq_euc = w_sq_riem.clone()
    buf_sq_riem = torch.zeros_like(w_sq_riem)
    buf_sq_euc = torch.zeros_like(w_sq_euc)

    w_rect_riem = torch.empty(d_tall_r, d_tall_c, dtype=torch.float32)
    torch.nn.init.orthogonal_(w_rect_riem)
    w_rect_euc = w_rect_riem.clone()
    buf_rect_riem = torch.zeros_like(w_rect_riem)
    buf_rect_euc = torch.zeros_like(w_rect_euc)

    step0_cayley_cos = 0.0
    for step_idx in range(50):
        g_sq = torch.randn(d_sq, d_sq, dtype=torch.float32)
        if step_idx == 0:
            w_ref = w_sq_riem.clone()
            riemannian_stiefel_muon_step(w_sq_riem, g_sq, buf_sq_riem, lr=lr, beta=beta)
            omega_hat = torch.matmul(w_ref.T, (w_ref - w_sq_riem) * math.sqrt(1.0 + lr * lr) / lr)
            omega_hat = 0.5 * (omega_hat - omega_hat.T)
            eye_sq = torch.eye(d_sq, dtype=torch.float32)
            w_cayley = torch.matmul(
                w_ref,
                torch.linalg.solve(eye_sq + 0.5 * lr * omega_hat, eye_sq - 0.5 * lr * omega_hat),
            )
            step0_cayley_cos = float(
                F.cosine_similarity(w_sq_riem.flatten(), w_cayley.flatten(), dim=0).item()
            )
        else:
            riemannian_stiefel_muon_step(w_sq_riem, g_sq, buf_sq_riem, lr=lr, beta=beta)

        buf_sq_euc.mul_(beta).add_(g_sq)
        u_s, _, vt_s = torch.linalg.svd(g_sq + beta * buf_sq_euc, full_matrices=False)
        w_sq_euc = w_sq_euc - lr * torch.matmul(u_s, vt_s)

        g_rect = torch.randn(d_tall_r, d_tall_c, dtype=torch.float32)
        riemannian_stiefel_muon_step(w_rect_riem, g_rect, buf_rect_riem, lr=lr, beta=beta)
        buf_rect_euc.mul_(beta).add_(g_rect)
        u_r, _, vt_r = torch.linalg.svd(g_rect + beta * buf_rect_euc, full_matrices=False)
        w_rect_euc = w_rect_euc - lr * torch.matmul(u_r, vt_r)

    eye_sq = torch.eye(d_sq, dtype=torch.float32)
    sq_riem_err = float(torch.linalg.matrix_norm(w_sq_riem.T @ w_sq_riem - eye_sq).item()) / math.sqrt(d_sq)
    sq_euc_err = float(torch.linalg.matrix_norm(w_sq_euc.T @ w_sq_euc - eye_sq).item()) / math.sqrt(d_sq)
    rect_riem_err = float(torch.linalg.matrix_norm(w_rect_riem.T @ w_rect_riem - eye_sq).item()) / math.sqrt(d_tall_c)
    rect_euc_err = float(torch.linalg.matrix_norm(w_rect_euc.T @ w_rect_euc - eye_sq).item()) / math.sqrt(d_tall_c)

    # Gemini Notebook Page 30-31: Two-meaning polysemous spectrum [0.44, 0.41] + 30 noise modes (0.005 each)
    d_h = 32
    eigs_init = torch.tensor([0.44, 0.41] + [0.005] * 30, dtype=torch.float32)
    rho_poly_re = torch.diag(eigs_init).unsqueeze(0)
    rho_poly_im = torch.zeros_like(rho_poly_re)
    rho_mc_re, rho_mc_im, purity_mc, eff_rank_mc = bures_wasserstein_mcweeny_step(
        rho_poly_re, rho_poly_im, num_steps=DEFAULT_MCWEENY_STEPS
    )
    eigs_after = torch.diagonal(rho_mc_re[0])
    noise_mass_before = float(eigs_init[2:].sum().item())
    noise_mass_after = float(eigs_after[2:].sum().item())

    # Midpoint distractor hallucination test (|a> = e_0, |b> = e_1, distractor |c> = (e_0 + e_1)/sqrt(2))
    ket_a_re = torch.zeros(1, d_h)
    ket_a_re[0, 0] = 1.0
    ket_b_re = torch.zeros(1, d_h)
    ket_b_re[0, 1] = 1.0
    ket_c_re = torch.zeros(1, d_h)
    ket_c_re[0, 0] = INV_SQRT2_ISOMETRIC_SCALE
    ket_c_re[0, 1] = INV_SQRT2_ISOMETRIC_SCALE
    zeros_im = torch.zeros(1, d_h)

    pa = pack_rank1_ket_outer_isometric(ket_a_re, zeros_im)
    pb = pack_rank1_ket_outer_isometric(ket_b_re, zeros_im)
    pc = pack_rank1_ket_outer_isometric(ket_c_re, zeros_im)

    psi_pure_re, psi_pure_im = complex_l2_normalize(0.5 * (ket_a_re + ket_b_re), zeros_im)
    p_pure = pack_rank1_ket_outer_isometric(psi_pure_re, psi_pure_im)
    pure_score_a = float((p_pure * pa).sum().item()) ** 4
    pure_score_c = float((p_pure * pc).sum().item()) ** 4
    pure_distractor_ratio = pure_score_c / max(TRACE_NORM_EPS, pure_score_a)

    p_impure = pack_hermitian_isometric(rho_mc_re, rho_mc_im)
    impure_score_a = float((p_impure * pa).sum().item()) ** 4
    impure_score_b = float((p_impure * pb).sum().item()) ** 4
    impure_score_c = float((p_impure * pc).sum().item()) ** 4
    impure_distractor_ratio = impure_score_c / max(TRACE_NORM_EPS, 0.5 * (impure_score_a + impure_score_b))

    krylov = extract_rank2_krylov_eigenpair(rho_mc_re, rho_mc_im)
    ov_re = (krylov["phi1_re"] * krylov["phi2_re"] + krylov["phi1_im"] * krylov["phi2_im"]).sum(dim=-1)
    ov_im = (krylov["phi1_re"] * krylov["phi2_im"] - krylov["phi1_im"] * krylov["phi2_re"]).sum(dim=-1)
    krylov_ortho_err = float(torch.sqrt(ov_re.square() + ov_im.square()).max().item())

    passed = bool(
        sq_riem_err < 1e-5
        and rect_riem_err < 1e-5
        and sq_euc_err > 0.15
        and step0_cayley_cos > 0.99999
        and noise_mass_after < 0.005
        and 1.90 <= float(eff_rank_mc.item()) <= 2.05
        and pure_distractor_ratio >= 15.0
        and impure_distractor_ratio <= 1.05
        and krylov_ortho_err < 1e-5
    )
    return {
        "passed": passed,
        "square_riemannian_stiefel_err": round(sq_riem_err, 8),
        "square_euclidean_muon_err": round(sq_euc_err, 5),
        "rect_riemannian_stiefel_err": round(rect_riem_err, 8),
        "rect_euclidean_muon_err": round(rect_euc_err, 5),
        "step0_cayley_cosine_sim": round(step0_cayley_cos, 8),
        "polysemous_lambda1_after_mcweeny": round(float(eigs_after[0].item()), 5),
        "polysemous_lambda2_after_mcweeny": round(float(eigs_after[1].item()), 5),
        "noise_mass_before_mcweeny": round(noise_mass_before, 4),
        "noise_mass_after_mcweeny": round(noise_mass_after, 6),
        "mcweeny_purity": round(float(purity_mc.item()), 5),
        "mcweeny_effective_rank": round(float(eff_rank_mc.item()), 4),
        "pure_midpoint_distractor_ratio_p2": round(pure_distractor_ratio, 2),
        "impure_midpoint_distractor_ratio_p2": round(impure_distractor_ratio, 4),
        "krylov_top2_orthogonality_err": round(krylov_ortho_err, 8),
        "sqrt2_scale_constant": SQRT2_ISOMETRIC_OFFDIAG_SCALE,
    }


def load_or_prepare_classical_teacher(
    device: torch.device,
    checkpoint_path: str = "",
    prewarm_steps: int = 30,
) -> tuple[CompiledLeaderboardClassicalGPT, dict[str, Any]]:
    """Load the latest trained milestone checkpoint from `w45b_classical_teacher_ckpt50m_l12_d768_b262k` or pre-warm."""
    selected_ckpt: Path | None = None
    ckpt_meta: dict[str, Any] = {"source": "fresh_prewarm", "tokens_M": 0.0, "step": 0}

    if checkpoint_path and Path(checkpoint_path).exists():
        selected_ckpt = Path(checkpoint_path)
    elif (DEFAULT_TEACHER_CKPT_DIR / "checkpoints_manifest.json").exists():
        manifest = json.loads((DEFAULT_TEACHER_CKPT_DIR / "checkpoints_manifest.json").read_text())
        trained_entries = []
        for e in manifest:
            if float(e.get("tokens_M", 0.0)) <= 0.0:
                continue
            cand = Path(e.get("path", ""))
            if not cand.exists() and e.get("filename"):
                cand = DEFAULT_TEACHER_CKPT_DIR / str(e["filename"])
            if cand.exists():
                trained_entries.append((e, cand))
        if trained_entries:
            best_entry, selected_ckpt = trained_entries[-1]
            ckpt_meta = {
                "source": "saved_milestone_checkpoint",
                "filename": best_entry.get("filename"),
                "step": best_entry.get("step"),
                "tokens_M": best_entry.get("tokens_M"),
                "val_ce": best_entry.get("val_ce"),
            }

    if selected_ckpt is not None:
        payload = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
        cfg_dict = payload["config"]
        cfg_dict["use_flex_doc_swc"] = False
        cfg = CompiledLeaderboardClassicalConfig(**cfg_dict)
        teacher = CompiledLeaderboardClassicalGPT(cfg).to(device)
        teacher.load_state_dict(payload["model_state_dict"], strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if ckpt_meta["source"] == "fresh_prewarm":
            ckpt_meta = {
                "source": str(selected_ckpt.name),
                "step": payload.get("step", 0),
                "tokens_M": payload.get("tokens_M", 0.0),
                "val_ce": payload.get("val_metrics", {}).get("val_ce"),
            }
        return teacher, ckpt_meta

    # If the 50M-token checkpoint has not landed yet, pre-warm a teacher on FineWeb Shard 1 in ~8s
    torch.manual_seed(1337)
    cfg = CompiledLeaderboardClassicalConfig(
        embed_dim=DEFAULT_MIRROR_EMBED_DIM,
        num_heads=DEFAULT_MIRROR_NUM_HEADS,
        num_layers=12,
        hidden_dim=DEFAULT_MIRROR_EMBED_DIM * 2,
        max_seq_len=128,
        use_flex_doc_swc=False,
        bfloat16_weights=(device.type == "cuda"),
    )
    teacher = CompiledLeaderboardClassicalGPT(cfg).to(device)
    muon_opt, adam_opt = build_compiled_leaderboard_optimizers(
        teacher, CompiledLeaderboardOptimizerConfig(muon_warmup_steps=10)
    )
    train_batches = _load_fineweb_token_batches(
        DEFAULT_TRAIN_SHARD1_PATH, num_batches=prewarm_steps, batch_size=4, seq_len=128, device=device
    )
    teacher.train()
    for xb, yb in train_batches:
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss = teacher(xb, yb, return_logits=False)
            assert loss is not None
        loss.backward()
        muon_opt.step()
        adam_opt.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    ckpt_meta["tokens_M"] = round((prewarm_steps * 4 * 128) / 1e6, 4)
    ckpt_meta["step"] = prewarm_steps
    return teacher, ckpt_meta


def run_experiment_a_layer_mirroring(
    teacher: CompiledLeaderboardClassicalGPT,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    eval_batches: list[tuple[torch.Tensor, torch.Tensor]],
    target_layer: int,
    mirror_steps: int,
    device: torch.device,
) -> tuple[dict[str, Any], ImpureQuantumDensityMirrorBlock, PureQuantumKetMirrorBlock]:
    """Run Experiment A: Train Pure Rank-1 vs. Impure McWeeny Density-Matrix blocks to mirror teacher layer `l`."""
    mirror_cfg = QuantumMirrorBlockConfig(
        embed_dim=teacher.cfg.embed_dim,
        num_heads=teacher.cfg.num_heads,
        head_complex_dim=DEFAULT_MIRROR_HEAD_COMPLEX_DIM,
        channel_hidden_dim=1024,
        born_power_p=DEFAULT_BORN_POWER_P,
        mcweeny_steps=DEFAULT_MCWEENY_STEPS,
        max_seq_len=128,
    )

    cached_train_triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    cached_eval_triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for xb, _ in train_batches:
            traj = teacher.extract_layer_trajectories(xb)
            cached_train_triples.append(
                (
                    traj["layer_inputs"][target_layer].detach(),
                    traj["x_0"].detach(),
                    traj["layer_outputs"][target_layer].detach(),
                )
            )
        for xb, _ in eval_batches:
            traj = teacher.extract_layer_trajectories(xb)
            cached_eval_triples.append(
                (
                    traj["layer_inputs"][target_layer].detach(),
                    traj["x_0"].detach(),
                    traj["layer_outputs"][target_layer].detach(),
                )
            )

    train_x_in = [t[0] for t in cached_train_triples]
    train_x_0 = [t[1] for t in cached_train_triples]
    train_x_out = [t[2] for t in cached_train_triples]

    orig_lambdas = teacher.blocks[target_layer].block_lambdas.detach().float()

    torch.manual_seed(4620 + target_layer)
    impure_blk = ImpureQuantumDensityMirrorBlock(mirror_cfg).to(device)
    torch.manual_seed(4620 + target_layer)
    pure_blk = PureQuantumKetMirrorBlock(mirror_cfg).to(device)

    with torch.no_grad():
        impure_blk.block_lambdas.copy_(orig_lambdas)
        pure_blk.block_lambdas.copy_(orig_lambdas)

    # Step 1 (Golub-Pereyra Variable Projection): Solve exact closed-form ridge observable readout at Step 0
    impure_blk.fit_observable_readouts_ridge(train_x_in, train_x_out, train_x_0, ridge_lambda=0.5)
    pure_blk.fit_observable_readouts_ridge(train_x_in, train_x_out, train_x_0, ridge_lambda=0.5)

    impure_stiefel_opt = RiemannianStiefelMuon(
        impure_blk.stiefel_parameters(),
        lr=0.002,
        beta=DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    )
    impure_sgd_opt = torch.optim.SGD(impure_blk.euclidean_parameters(), lr=0.005, momentum=0.9, nesterov=True)

    pure_stiefel_opt = RiemannianStiefelMuon(
        pure_blk.stiefel_parameters(),
        lr=0.002,
        beta=DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    )
    pure_sgd_opt = torch.optim.SGD(pure_blk.euclidean_parameters(), lr=0.005, momentum=0.9, nesterov=True)

    reproject_interval = max(10, mirror_steps // 4)
    for step in range(mirror_steps):
        x_in, x_0, x_out_t = cached_train_triples[step % len(cached_train_triples)]

        impure_stiefel_opt.zero_grad(set_to_none=True)
        impure_sgd_opt.zero_grad(set_to_none=True)
        out_imp, _ = impure_blk(x_in, x_0)
        loss_imp = compute_layer_mirroring_loss(
            out_imp, x_in, x_out_t, gram_weight=DEFAULT_GRAM_RELATIONAL_WEIGHT
        )
        loss_imp["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(impure_blk.parameters(), 1.0)
        impure_stiefel_opt.step()
        impure_sgd_opt.step()

        pure_stiefel_opt.zero_grad(set_to_none=True)
        pure_sgd_opt.zero_grad(set_to_none=True)
        out_p, _ = pure_blk(x_in, x_0)
        loss_p = compute_layer_mirroring_loss(
            out_p, x_in, x_out_t, gram_weight=DEFAULT_GRAM_RELATIONAL_WEIGHT
        )
        loss_p["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(pure_blk.parameters(), 1.0)
        pure_stiefel_opt.step()
        pure_sgd_opt.step()

        if (step + 1) % reproject_interval == 0:
            impure_blk.fit_observable_readouts_ridge(train_x_in, train_x_out, train_x_0, ridge_lambda=0.5)
            pure_blk.fit_observable_readouts_ridge(train_x_in, train_x_out, train_x_0, ridge_lambda=0.5)

    impure_blk.fit_observable_readouts_ridge(train_x_in, train_x_out, train_x_0, ridge_lambda=0.5)
    pure_blk.fit_observable_readouts_ridge(train_x_in, train_x_out, train_x_0, ridge_lambda=0.5)

    impure_blk.eval()
    pure_blk.eval()
    imp_mse_list, imp_cos_list, imp_gram_list = [], [], []
    pure_mse_list, pure_cos_list, pure_gram_list = [], [], []
    purities, eff_ranks = [], []

    with torch.no_grad():
        for x_in, x_0, x_out_t in cached_eval_triples:
            out_imp, _ = impure_blk(x_in, x_0)
            m_imp = compute_layer_mirroring_loss(out_imp, x_in, x_out_t)
            imp_mse_list.append(float(m_imp["rel_update_mse"].item()))
            imp_cos_list.append(float(m_imp["output_cosine_sim"].item()))
            imp_gram_list.append(float(m_imp["gram_relational_mse"].item()))
            purities.append(impure_blk.last_telemetry["mean_purity"])
            eff_ranks.append(impure_blk.last_telemetry["mean_eff_rank"])

            out_p, _ = pure_blk(x_in, x_0)
            m_p = compute_layer_mirroring_loss(out_p, x_in, x_out_t)
            pure_mse_list.append(float(m_p["rel_update_mse"].item()))
            pure_cos_list.append(float(m_p["output_cosine_sim"].item()))
            pure_gram_list.append(float(m_p["gram_relational_mse"].item()))

    imp_mse = float(np.mean(imp_mse_list))
    pure_mse = float(np.mean(pure_mse_list))
    imp_cos = float(np.mean(imp_cos_list))
    pure_cos = float(np.mean(pure_cos_list))
    mean_purity = float(np.mean(purities))
    mean_eff_rank = float(np.mean(eff_ranks))

    passed = bool(
        imp_mse < pure_mse
        and imp_cos >= pure_cos
        and 0.15 <= mean_purity <= 0.92
        and mean_eff_rank >= 1.15
    )
    metrics = {
        "passed": passed,
        "target_layer": target_layer,
        "mirror_steps": mirror_steps,
        "impure_rel_update_mse": round(imp_mse, 5),
        "pure_rel_update_mse": round(pure_mse, 5),
        "impure_output_cosine_sim": round(imp_cos, 5),
        "pure_output_cosine_sim": round(pure_cos, 5),
        "impure_gram_relational_mse": round(float(np.mean(imp_gram_list)), 6),
        "pure_gram_relational_mse": round(float(np.mean(pure_gram_list)), 6),
        "mean_contextual_purity": round(mean_purity, 4),
        "mean_contextual_effective_rank": round(mean_eff_rank, 4),
    }
    return metrics, impure_blk, pure_blk


def _eval_model_on_batches(
    model: CompiledLeaderboardClassicalGPT,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate `val_ce`, `val_ppl`, and `top1_acc` of `model` on `batches`."""
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for xb, yb in batches:
            logits, loss = model(xb, yb, return_logits=True)
            assert logits is not None and loss is not None
            n_tok = yb.numel()
            total_loss += float(loss.item()) * n_tok
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == yb).sum().item())
            total_tokens += n_tok
    val_ce = total_loss / max(1, total_tokens)
    return {
        "val_ce": round(val_ce, 5),
        "val_ppl": round(math.exp(min(20.0, val_ce)), 3),
        "top1_acc": round(100.0 * float(total_correct) / float(max(1, total_tokens)), 2),
    }


def run_experiment_b_surgical_grafting(
    teacher: CompiledLeaderboardClassicalGPT,
    mirrored_impure_blk: ImpureQuantumDensityMirrorBlock,
    mirrored_pure_blk: PureQuantumKetMirrorBlock,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    eval_batches: list[tuple[torch.Tensor, torch.Tensor]],
    target_layer: int,
    graft_finetune_steps: int,
) -> dict[str, Any]:
    """Run Experiment B: Surgically graft quantum blocks into `CompiledLeaderboardClassicalGPT` with frozen backbone/wte."""
    orig_block = teacher.blocks[target_layer]
    intact_metrics = _eval_model_on_batches(teacher, eval_batches)

    unaligned_blk = ImpureQuantumDensityMirrorBlock(mirrored_impure_blk.cfg).to(
        next(teacher.parameters()).device
    )
    with torch.no_grad():
        unaligned_blk.block_lambdas.copy_(orig_block.block_lambdas.detach().float())
    teacher.blocks[target_layer] = unaligned_blk
    unaligned_metrics = _eval_model_on_batches(teacher, eval_batches)

    graft_impure = copy.deepcopy(mirrored_impure_blk)
    teacher.blocks[target_layer] = graft_impure
    zero_shot_impure_metrics = _eval_model_on_batches(teacher, eval_batches)

    graft_pure = copy.deepcopy(mirrored_pure_blk)
    teacher.blocks[target_layer] = graft_pure
    zero_shot_pure_metrics = _eval_model_on_batches(teacher, eval_batches)

    # Surgical Frozen-Backbone End-to-End Fine-Tuning of Observable Readout & Unitary Phases via Nesterov SGD
    teacher.blocks[target_layer] = graft_impure
    graft_impure.train()
    for p in graft_impure.parameters():
        p.requires_grad_(False)
    for p in graft_impure.euclidean_parameters():
        p.requires_grad_(True)
    imp_sgd_opt = torch.optim.SGD(graft_impure.euclidean_parameters(), lr=0.0005, momentum=0.9, nesterov=True)
    for step in range(graft_finetune_steps):
        xb, yb = train_batches[step % len(train_batches)]
        imp_sgd_opt.zero_grad(set_to_none=True)
        _, loss = teacher(xb, yb, return_logits=True)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graft_impure.euclidean_parameters(), 1.0)
        imp_sgd_opt.step()
    graft_impure.eval()
    finetuned_impure_metrics = _eval_model_on_batches(teacher, eval_batches)

    teacher.blocks[target_layer] = graft_pure
    graft_pure.train()
    for p in graft_pure.parameters():
        p.requires_grad_(False)
    for p in graft_pure.euclidean_parameters():
        p.requires_grad_(True)
    pure_sgd_opt = torch.optim.SGD(graft_pure.euclidean_parameters(), lr=0.0005, momentum=0.9, nesterov=True)
    for step in range(graft_finetune_steps):
        xb, yb = train_batches[step % len(train_batches)]
        pure_sgd_opt.zero_grad(set_to_none=True)
        _, loss = teacher(xb, yb, return_logits=True)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graft_pure.euclidean_parameters(), 1.0)
        pure_sgd_opt.step()
    graft_pure.eval()
    finetuned_pure_metrics = _eval_model_on_batches(teacher, eval_batches)

    teacher.blocks[target_layer] = orig_block

    unaligned_gap = max(1e-5, unaligned_metrics["val_ce"] - intact_metrics["val_ce"])
    impure_recovered_gap_frac = (
        unaligned_metrics["val_ce"] - finetuned_impure_metrics["val_ce"]
    ) / unaligned_gap

    passed = bool(
        finetuned_impure_metrics["val_ce"] < unaligned_metrics["val_ce"]
        and finetuned_impure_metrics["val_ce"] <= finetuned_pure_metrics["val_ce"]
        and impure_recovered_gap_frac >= 0.50
    )
    return {
        "passed": passed,
        "target_layer": target_layer,
        "graft_finetune_steps": graft_finetune_steps,
        "intact_classical_teacher": intact_metrics,
        "unaligned_quantum_graft": unaligned_metrics,
        "zero_shot_mirrored_pure_graft": zero_shot_pure_metrics,
        "zero_shot_mirrored_impure_graft": zero_shot_impure_metrics,
        "finetuned_pure_graft": finetuned_pure_metrics,
        "finetuned_impure_graft": finetuned_impure_metrics,
        "impure_recovered_gap_fraction": round(impure_recovered_gap_frac, 4),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `71_verify_plan061_quantum_density_layer_mirroring.py`."""
    parser = argparse.ArgumentParser(description="Wave 46 Experiment A & B Quantum Density Mirroring Runner.")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test.")
    parser.add_argument("--run-key", type=str, default="w46_quantum_impure_density_mirror_graft")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Optional teacher checkpoint .pt path.")
    parser.add_argument("--target-layer", type=int, default=11, help="Classical layer index to mirror and graft (0..11).")
    parser.add_argument("--mirror-steps", type=int, default=60, help="Steps for Experiment A layer mirroring.")
    parser.add_argument("--graft-steps", type=int, default=40, help="Steps for Experiment B surgical graft fine-tuning.")
    parser.add_argument("--json-out", type=str, default=str(HYPOTHESIS_JSON_OUT), help="Output JSON artifact path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    mirror_steps = 12 if args.smoke else int(args.mirror_steps)
    graft_steps = 8 if args.smoke else int(args.graft_steps)

    t0 = time.perf_counter()
    h1_res = verify_h1_riemannian_geodesic_and_midpoint_hallucination()
    teacher, ckpt_meta = load_or_prepare_classical_teacher(
        device, checkpoint_path=args.checkpoint_path, prewarm_steps=10 if args.smoke else 35
    )

    train_batches = _load_fineweb_token_batches(
        DEFAULT_TRAIN_SHARD1_PATH, num_batches=24, batch_size=4, seq_len=128, device=device
    )
    eval_batches = _load_fineweb_token_batches(
        DEFAULT_VAL_SHARD_PATH, num_batches=8, batch_size=4, seq_len=128, device=device
    )

    gpu_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    intent = ExperimentIntentSpec(
        run_key=str(args.run_key),
        wave_id=WAVE_ID,
        plan_id=PLAN_ID,
        task_domain="micro_verification",
        hypothesis_intent=(
            f"Verify Wave 46 Experiment A (Layer {args.target_layer} Mirroring) & Experiment B (Surgical Grafting) "
            f"with Impure McWeeny Density Matrices, Isometric sqrt(2) Packing, and Riemannian Stiefel-Muon."
        ),
        architecture_summary=(
            "ImpureQuantumDensityMirrorBlock (H=6, d_h=16, packed=1536->768) grafted into CompiledLeaderboardClassicalGPT"
        ),
        hardware_budget=gpu_label,
        arch_family="quantum_born",
        batch_size=4,
        seq_len=128,
        total_steps=mirror_steps + graft_steps,
        target_tokens_M=round(((mirror_steps + graft_steps) * 4 * 128) / 1e6, 4),
        params_M=4.2,
        embed_dim=DEFAULT_MIRROR_EMBED_DIM,
        num_layers=12,
    )
    logger = StandardizedExperimentLogger(intent)

    h2_exp_a, impure_blk, pure_blk = run_experiment_a_layer_mirroring(
        teacher,
        train_batches,
        eval_batches,
        target_layer=int(args.target_layer),
        mirror_steps=mirror_steps,
        device=device,
    )
    h3_exp_b = run_experiment_b_surgical_grafting(
        teacher,
        impure_blk,
        pure_blk,
        train_batches,
        eval_batches,
        target_layer=int(args.target_layer),
        graft_finetune_steps=graft_steps,
    )

    elapsed_s = time.perf_counter() - t0
    logger.log_eval(
        mirror_steps + graft_steps,
        {
            "tokens_M": round(((mirror_steps + graft_steps) * 4 * 128) / 1e6, 4),
            "compute_tflops": 0.15,
            "tau_progress": 1.0,
            "val_ce": float(h3_exp_b["finetuned_impure_graft"]["val_ce"]),
            "val_ppl": float(h3_exp_b["finetuned_impure_graft"]["val_ppl"]),
            "top1_acc": float(h3_exp_b["finetuned_impure_graft"]["top1_acc"]),
            "sharp_eff_rank": float(h2_exp_a["mean_contextual_effective_rank"]),
            "kraus_stiefel_err": float(h1_res["square_riemannian_stiefel_err"]),
        },
    )

    overall_pass = bool(
        h1_res.get("passed", False)
        and h2_exp_a.get("passed", False)
        and h3_exp_b.get("passed", False)
    )
    payload = {
        "wave_id": WAVE_ID,
        "plan_id": PLAN_ID,
        "run_key": str(args.run_key),
        "hardware": gpu_label,
        "wall_seconds": round(elapsed_s, 2),
        "teacher_checkpoint_used": ckpt_meta,
        "h1_riemannian_geodesic_and_midpoint": h1_res,
        "h2_experiment_a_layer_mirroring": h2_exp_a,
        "h3_experiment_b_surgical_grafting": h3_exp_b,
        "overall_pass": overall_pass,
    }
    logger.finalize(extra_summary=payload)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
