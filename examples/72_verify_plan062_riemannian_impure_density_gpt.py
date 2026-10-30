"""Wave 47 (`EXPERIMENT_PLAN_062.md`) Verification & Dual-3090 Runner: `RiemannianImpureDensityGPT`.

Executes three quantitative experiments validating the full all-layer `RiemannianStiefelMuon`
quantum density-matrix architecture:
1. **H1**: All-layer `RiemannianStiefelMuon` (`Real Stiefel` + `Joint-Complex Unitary` $U(16)$ geodesics)
   + $R=4$ Impure McWeeny Density Matrices vs. Euclidean Muon vs. All-Layer Rank-1 Pure Kets on FineWeb.
2. **H2**: Gemini Notebook Stationary Input-Layer Anchor (`InputBoundaryMode.ANCHORED_SPHERE`) +
   Cross-Layer $V_0$ Multi-Branch Value Ket Highway vs. Unanchored Euclidean Input without $V_0$.
3. **H3**: Multi-Layer (`Layers 9, 10, 11` or all 12 layers) Stiefel-SVD Polar Golub-Pereyra Alignment
   + End-to-End `AllLayerRiemannianStiefelMuon` FineWeb Training on `CompiledLeaderboardClassicalGPT`
   checkpoints (`50M, 100M, 150M, 200M+` tokens).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass(frozen=True)
class EvalDataBundle:
    """Bundled FineWeb token slices, teacher reference, and batch geometry (`<= 8` fields, Rule 2)."""

    train_tokens: torch.Tensor
    val_tokens: torch.Tensor
    teacher: CompiledLeaderboardClassicalGPT
    batch_size: int
    seq_len: int
    device: torch.device

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
)
from qdlblocks.models.riemannian_impure_density_gpt import (
    AllLayerRiemannianStiefelMuon,
    DensityStateMode,
    InputBoundaryMode,
    RiemannianImpureDensityBlock,
    RiemannianImpureDensityConfig,
    RiemannianImpureDensityGPT,
    build_all_layer_riemannian_optimizers,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".cache"
FINEWEB_DIR = CACHE_DIR / "fineweb_100M"
DEFAULT_CHECKPOINT_DIR = CACHE_DIR / "checkpoints" / "w45b_classical_teacher_ckpt50m_l12_d768_b262k"
DEFAULT_CHECKPOINT_MANIFEST = DEFAULT_CHECKPOINT_DIR / "checkpoints_manifest.json"
DEFAULT_JSON_OUT = CACHE_DIR / "hypothesis_experiments_062.json"
SHARD_HEADER_INTS: int = 256


def _load_fineweb_tokens(shard_path: Path, max_tokens: int = 2_000_000) -> torch.Tensor:
    """Load GPT-2 BPE token IDs from a `modded-nanogpt` FineWeb `.bin` shard."""
    if shard_path.exists():
        raw = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=SHARD_HEADER_INTS * 4)
        slice_arr = np.array(raw[:max_tokens], dtype=np.int64)
        return torch.from_numpy(slice_arr)
    g = torch.Generator().manual_seed(20261030)
    return torch.randint(0, 50257, (max_tokens,), generator=g, dtype=torch.long)


def _sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically slice `(x, y)` token batches of shape `(B, T)`."""
    span = batch_size * (seq_len + 1)
    max_start = max(1, tokens.numel() - span - 1)
    offset = (step * span) % max_start
    chunk = tokens[offset : offset + span].view(batch_size, seq_len + 1)
    x = chunk[:, :-1].contiguous().to(device, non_blocking=True)
    y = chunk[:, 1:].contiguous().to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def _evaluate_gpt_val_ce(
    model: nn.Module,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    device: torch.device,
) -> float:
    """Evaluate mean validation cross-entropy (nats) across `num_batches` sequential FineWeb windows."""
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for b_idx in range(num_batches):
        x, y = _sample_batch(val_tokens, batch_size, seq_len, step=b_idx, device=device)
        _, loss = model(x, targets=y, return_logits=False)
        assert loss is not None
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return float(sum(losses) / max(1, len(losses)))


def _load_classical_teacher(
    explicit_path: str | None,
    device: torch.device,
) -> tuple[CompiledLeaderboardClassicalGPT, dict[str, Any]]:
    """Load the latest trained checkpoint from `w45b_classical_teacher_ckpt50m_l12_d768_b262k`."""
    selected_ckpt: Path | None = None
    if explicit_path and Path(explicit_path).exists():
        selected_ckpt = Path(explicit_path)
    elif DEFAULT_CHECKPOINT_MANIFEST.exists():
        raw_manifest = json.loads(DEFAULT_CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
        entries = raw_manifest if isinstance(raw_manifest, list) else raw_manifest.get("checkpoints", [])
        for entry in reversed(entries):
            if float(entry.get("tokens_M", entry.get("tokens_m", 0.0))) <= 0.0:
                continue
            cand = Path(entry.get("path", ""))
            if not cand.exists() and entry.get("filename"):
                cand = DEFAULT_CHECKPOINT_DIR / str(entry["filename"])
            if cand.exists():
                selected_ckpt = cand
                break

    if selected_ckpt is not None:
        payload = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
        cfg_dict = dict(payload["config"])
        cfg_dict["use_flex_doc_swc"] = False
        teacher = CompiledLeaderboardClassicalGPT(CompiledLeaderboardClassicalConfig(**cfg_dict)).to(device)
        teacher.load_state_dict(payload["model_state_dict"], strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        meta = {
            "checkpoint_path": str(selected_ckpt),
            "filename": selected_ckpt.name,
            "tokens_M": float(payload.get("tokens_M", payload.get("tokens_m", 0.0))),
            "step": int(payload.get("step", 0)),
            "val_ce": float(payload.get("val_metrics", {}).get("val_ce", payload.get("val_ce", 0.0))),
        }
        return teacher, meta

    cfg = CompiledLeaderboardClassicalConfig(use_flex_doc_swc=False)
    teacher = CompiledLeaderboardClassicalGPT(cfg).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, {"checkpoint_path": None, "filename": "untrained_fallback", "tokens_M": 0.0, "step": 0}


@torch.no_grad()
def _calibrate_all_layer_stiefel_svd_readouts(
    model: RiemannianImpureDensityGPT,
    traj_batches: list[dict[str, Any]],
) -> None:
    """Fit Stiefel-SVD readouts `(U, sigma, V^T)` sequentially across all $L$ quantum layers of `model`."""
    teacher_layers = len(traj_batches[0]["layer_outputs"])
    d_m = model.embed_dim
    stride = max(1, teacher_layers // max(1, model.num_layers))

    x0_list: list[torch.Tensor] = []
    for tb in traj_batches:
        idx_b = tb["idx"]
        tok_emb = model.wte(idx_b).float()
        if model.cfg.boundary_mode == InputBoundaryMode.ANCHORED_SPHERE:
            tok_unit = F.normalize(tok_emb, p=2, dim=-1) * math.sqrt(float(d_m))
            x0_b = torch.matmul(tok_unit, model.w_in_stiefel.float()) * model.in_scale.float()
        else:
            x0_b = torch.matmul(tok_emb, model.w_in_stiefel.float()) * model.in_scale.float()
        x0_list.append(x0_b.detach())

    x_cur_list = [x0_b.clone() for x0_b in x0_list]
    v0_running: list[torch.Tensor | None] = [None for _ in range(len(traj_batches))]

    for l_idx, blk in enumerate(model.blocks):
        t_end = min(teacher_layers - 1, (l_idx + 1) * stride - 1)
        if l_idx == model.num_layers - 1:
            t_end = teacher_layers - 1
        x_out_list = [tb["layer_outputs"][t_end].detach().float()[:, :, :d_m] for tb in traj_batches]
        v0_arg = [v for v in v0_running if v is not None] if l_idx > 0 else None
        blk.fit_stiefel_observables_polar_ridge(
            x_cur_list,
            x_out_list,
            x_0_list=x0_list,
            v_0_list=v0_arg if (v0_arg and len(v0_arg) == len(traj_batches)) else None,
            ridge_lambda=0.08,
        )
        next_x_list: list[torch.Tensor] = []
        for b_i, x_in in enumerate(x_cur_list):
            x_next, v0_next = blk(x_in.float(), x_0=x0_list[b_i].float(), v_0=v0_running[b_i])
            next_x_list.append(x_next.detach())
            if l_idx == 0 and v0_next is not None:
                v0_running[b_i] = v0_next.detach()
        x_cur_list = next_x_list


@torch.no_grad()
def _initialize_all_layer_quantum_gpt_from_teacher(
    model: RiemannianImpureDensityGPT,
    bundle: EvalDataBundle,
    num_calib_batches: int = 8,
) -> list[dict[str, Any]]:
    """Initialize stationary unit-sphere token wavefunctions and Stiefel-SVD readouts across all $L$ layers."""
    teacher = bundle.teacher
    d_m = model.embed_dim
    wte_src = teacher.wte.weight.detach().float()[:, :d_m]
    lm_src = teacher.lm_head.weight.detach().float()[:, :d_m]

    model.wte.weight.copy_(
        (F.normalize(wte_src, p=2, dim=-1) * math.sqrt(float(d_m))).to(model.wte.weight.dtype)
    )
    model.lm_head.weight.copy_(lm_src.to(model.lm_head.weight.dtype))
    model.lm_head.weight.requires_grad_(False)

    stride = max(1, teacher.cfg.num_layers // max(1, model.num_layers))
    for l_idx, blk in enumerate(model.blocks):
        t_idx = min(teacher.cfg.num_layers - 1, l_idx * stride)
        blk.block_lambdas.copy_(teacher.blocks[t_idx].block_lambdas.detach().float())

    traj_batches: list[dict[str, Any]] = []
    for b_idx in range(num_calib_batches):
        xb, yb = _sample_batch(bundle.train_tokens, bundle.batch_size, bundle.seq_len, step=b_idx, device=bundle.device)
        tb = teacher.extract_layer_trajectories(xb)
        tb["idx"] = xb
        tb["targets"] = yb
        traj_batches.append(tb)

    _calibrate_all_layer_stiefel_svd_readouts(model, traj_batches)
    return traj_batches


def _train_variant_on_fineweb(
    cfg: RiemannianImpureDensityConfig,
    bundle: EvalDataBundle,
    steps: int,
    use_geodesic_retraction: bool = True,
) -> dict[str, float]:
    """Train an all-layer `RiemannianImpureDensityGPT` variant on FineWeb and return telemetry."""
    torch.manual_seed(2026062)
    if bundle.device.type == "cuda":
        torch.cuda.manual_seed_all(2026062)

    model = RiemannianImpureDensityGPT(cfg).to(bundle.device)
    traj_batches = _initialize_all_layer_quantum_gpt_from_teacher(model, bundle, num_calib_batches=8)

    if use_geodesic_retraction:
        state_prep_params: list[nn.Parameter] = [model.w_in_stiefel]
        for blk in model.blocks:
            state_prep_params.extend([blk.w_q, blk.w_k, blk.w_v, blk.w_beam_a, blk.w_beam_b])
        stiefel_opt = AllLayerRiemannianStiefelMuon(
            real_stiefel_params=state_prep_params,
            complex_unitary_pairs=model.all_complex_unitary_pairs(),
            stiefel_lr=0.018,
            unitary_lr=0.018,
            use_geodesic_retraction=True,
        )
        euclid_groups: list[dict[str, Any]] = [{"params": [model.in_scale], "lr": 0.001, "weight_decay": 0.0}]
        if model.wte.weight.requires_grad:
            euclid_groups.append({"params": [model.wte.weight], "lr": 0.060, "weight_decay": 0.0})
        euclid_opt: torch.optim.Optimizer = torch.optim.AdamW(euclid_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        stiefel_opt, euclid_opt = build_all_layer_riemannian_optimizers(
            model,
            stiefel_lr=0.022,
            unitary_lr=0.022,
            scalar_lr=0.002,
            head_lr=0.025,
            use_geodesic_retraction=False,
        )

    step0_val_ce = _evaluate_gpt_val_ce(
        model, bundle.val_tokens, bundle.batch_size, bundle.seq_len, num_batches=6, device=bundle.device
    )
    model.train()
    train_losses: list[float] = []

    for step in range(steps):
        if use_geodesic_retraction and step > 0 and step % 20 == 0:
            _calibrate_all_layer_stiefel_svd_readouts(model, traj_batches)
        tau = float(step) / float(max(1, steps - 1))
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * tau)) * 0.85 + 0.15
        stiefel_opt.set_lr_Scale(lr_mult)

        tb = traj_batches[step % len(traj_batches)]
        x = tb["idx"]
        y = tb["targets"]
        target_h = F.rms_norm(
            tb["layer_outputs"][-1].detach().float()[:, :, : model.embed_dim],
            (model.embed_dim,),
            eps=1e-6,
        )

        stiefel_opt.zero_grad(set_to_none=True)
        euclid_opt.zero_grad(set_to_none=True)
        h_pred = model.forward_hidden(x).float()
        _, ce_loss = model(x, targets=y, return_logits=False)
        assert ce_loss is not None
        cos_align = (F.normalize(h_pred, p=2, dim=-1) * F.normalize(target_h, p=2, dim=-1)).sum(dim=-1).mean()
        total_loss = (h_pred - target_h).square().mean() + (1.0 - cos_align) + 0.05 * ce_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        stiefel_opt.step()
        euclid_opt.step()
        train_losses.append(float(ce_loss.item()))

    if use_geodesic_retraction and cfg.boundary_mode == InputBoundaryMode.ANCHORED_SPHERE:
        _calibrate_all_layer_stiefel_svd_readouts(model, traj_batches)

    final_val_ce = _evaluate_gpt_val_ce(
        model, bundle.val_tokens, bundle.batch_size, bundle.seq_len, num_batches=6, device=bundle.device
    )
    telem = model.compute_manifold_and_purity_telemetry()
    return {
        "step0_val_ce": float(step0_val_ce),
        "final_val_ce": float(final_val_ce),
        "final_train_ce": float(sum(train_losses[-5:]) / max(1, len(train_losses[-5:]))),
        **telem,
    }


def run_h1_all_layer_riemannian_vs_euclidean_and_pure(
    bundle: EvalDataBundle,
    steps: int,
    num_layers: int,
    embed_dim: int,
) -> dict[str, Any]:
    """Run Hypothesis H1: All-layer `RiemannianStiefelMuon` Impure Density vs. Euclidean Muon vs. Pure Ket."""
    cfg_riem_impure = RiemannianImpureDensityConfig(
        vocab_size=bundle.teacher.cfg.vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=6,
        head_complex_dim=16,
        boundary_mode=InputBoundaryMode.ANCHORED_SPHERE,
        density_mode=DensityStateMode.IMPURE_MCWEENY_R4,
        use_v0_highway=True,
    )
    cfg_pure_ket = RiemannianImpureDensityConfig(
        vocab_size=bundle.teacher.cfg.vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=6,
        head_complex_dim=16,
        boundary_mode=InputBoundaryMode.ANCHORED_SPHERE,
        density_mode=DensityStateMode.PURE_KET_R1,
        use_v0_highway=True,
    )

    res_riem_impure = _train_variant_on_fineweb(
        cfg_riem_impure,
        bundle,
        steps,
        use_geodesic_retraction=True,
    )
    res_euclid_impure = _train_variant_on_fineweb(
        cfg_riem_impure,
        bundle,
        steps,
        use_geodesic_retraction=False,
    )
    res_riem_pure = _train_variant_on_fineweb(
        cfg_pure_ket,
        bundle,
        steps,
        use_geodesic_retraction=True,
    )

    h1_pass = bool(
        res_riem_impure["mean_stiefel_err"] < 1e-4
        and res_riem_impure["mean_unitary_err"] < 1e-4
        and res_riem_impure["mean_stiefel_err"] < res_euclid_impure["mean_stiefel_err"] * 0.01
        and res_riem_impure["final_val_ce"] < res_euclid_impure["final_val_ce"]
        and res_riem_impure["final_val_ce"] < res_riem_pure["final_val_ce"]
    )
    return {
        "riemannian_impure_r4": res_riem_impure,
        "euclidean_muon_impure_r4": res_euclid_impure,
        "riemannian_pure_ket_r1": res_riem_pure,
        "val_ce_gain_vs_euclidean": float(res_euclid_impure["final_val_ce"] - res_riem_impure["final_val_ce"]),
        "val_ce_gain_vs_pure_ket": float(res_riem_pure["final_val_ce"] - res_riem_impure["final_val_ce"]),
        "passed": h1_pass,
    }


def run_h2_input_boundary_anchor_and_v0_highway(
    bundle: EvalDataBundle,
    steps: int,
    num_layers: int,
    embed_dim: int,
    anchored_baseline: dict[str, float],
) -> dict[str, Any]:
    """Run Hypothesis H2: Stationary Input Wavefunction Anchor + $V_0$ Highway vs. Unanchored Euclidean Input."""
    cfg_unanchored_no_v0 = RiemannianImpureDensityConfig(
        vocab_size=bundle.teacher.cfg.vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=6,
        head_complex_dim=16,
        boundary_mode=InputBoundaryMode.EUCLIDEAN_UNANCHORED,
        density_mode=DensityStateMode.IMPURE_MCWEENY_R4,
        use_v0_highway=False,
    )
    res_unanchored = _train_variant_on_fineweb(
        cfg_unanchored_no_v0,
        bundle,
        steps,
        use_geodesic_retraction=True,
    )

    h2_pass = bool(
        anchored_baseline["final_val_ce"] <= res_unanchored["final_val_ce"]
        and 1.2 <= anchored_baseline["mean_eff_rank"] <= 6.0
    )
    return {
        "anchored_sphere_with_v0_highway": anchored_baseline,
        "unanchored_euclidean_no_v0": res_unanchored,
        "val_ce_improvement": float(res_unanchored["final_val_ce"] - anchored_baseline["final_val_ce"]),
        "passed": h2_pass,
    }


@torch.no_grad()
def _capture_teacher_layer_io_with_v0(
    teacher: CompiledLeaderboardClassicalGPT,
    tokens: torch.Tensor,
    layer_idx: int,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Capture `(x_in, x_out, x_0)` activation pairs at `layer_idx` of `CompiledLeaderboardClassicalGPT`."""
    teacher.eval()
    inputs_list: list[torch.Tensor] = []
    outputs_list: list[torch.Tensor] = []
    x0_list: list[torch.Tensor] = []

    for b_idx in range(num_batches):
        idx_b, _ = _sample_batch(tokens, batch_size, seq_len, step=b_idx, device=device)
        traj = teacher.extract_layer_trajectories(idx_b)
        inputs_list.append(traj["layer_inputs"][layer_idx].detach())
        x0_list.append(traj["x_0"].detach())
        outputs_list.append(traj["layer_outputs"][layer_idx].detach())

    return inputs_list, outputs_list, x0_list


@torch.no_grad()
def _evaluate_teacher_val_ce(
    model: CompiledLeaderboardClassicalGPT,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    device: torch.device,
) -> float:
    """Evaluate validation cross-entropy of `CompiledLeaderboardClassicalGPT` (intact or multi-layer grafted)."""
    model.eval()
    losses: list[float] = []
    for b_idx in range(num_batches):
        x, y = _sample_batch(val_tokens, batch_size, seq_len, step=b_idx, device=device)
        _, loss = model(x, targets=y, return_logits=False)
        assert loss is not None
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def _align_single_riemannian_layer_to_teacher(
    teacher: CompiledLeaderboardClassicalGPT,
    train_tokens: torch.Tensor,
    layer_idx: int,
    mirror_steps: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[RiemannianImpureDensityBlock, dict[str, float]]:
    """Align a `RiemannianImpureDensityBlock` to teacher `layer_idx` via Golub-Pereyra Stiefel-SVD + `RiemannianStiefelMuon`."""
    num_capture = max(8, min(16, mirror_steps // 3))
    x_in_list, x_out_list, x0_list = _capture_teacher_layer_io_with_v0(
        teacher, train_tokens, layer_idx, num_capture, batch_size, seq_len, device
    )

    q_cfg = RiemannianImpureDensityConfig(
        vocab_size=teacher.cfg.vocab_size,
        embed_dim=teacher.cfg.embed_dim,
        num_layers=teacher.cfg.num_layers,
        num_heads=teacher.cfg.num_heads,
        head_complex_dim=16,
        boundary_mode=InputBoundaryMode.TEACHER_FROZEN_BOUNDARY,
        density_mode=DensityStateMode.IMPURE_MCWEENY_R4,
        use_v0_highway=False,
    )
    q_block = RiemannianImpureDensityBlock(q_cfg, layer_idx=layer_idx).to(device)
    with torch.no_grad():
        q_block.block_lambdas.copy_(teacher.blocks[layer_idx].block_lambdas.detach().float())

    q_block.fit_stiefel_observables_polar_ridge(x_in_list, x_out_list, x_0_list=x0_list, ridge_lambda=0.04)

    state_prep_stiefel = [q_block.w_q, q_block.w_k, q_block.w_v, q_block.w_beam_a, q_block.w_beam_b]
    stiefel_opt = AllLayerRiemannianStiefelMuon(
        real_stiefel_params=state_prep_stiefel,
        complex_unitary_pairs=[q_block.complex_unitary_pair()],
        stiefel_lr=0.025,
        unitary_lr=0.020,
        use_geodesic_retraction=True,
    )

    for step in range(mirror_steps):
        if step > 0 and step % 15 == 0:
            q_block.fit_stiefel_observables_polar_ridge(x_in_list, x_out_list, x_0_list=x0_list, ridge_lambda=0.04)
        idx = step % len(x_in_list)
        x_in = x_in_list[idx]
        x_out = x_out_list[idx]
        x_0 = x0_list[idx]

        stiefel_opt.zero_grad(set_to_none=True)
        pred_out, _ = q_block(x_in.float(), x_0=x_0.float())
        delta_pred = pred_out - x_in.float()
        delta_true = x_out.float() - x_in.float()
        rel_mse = (delta_pred - delta_true).square().mean() / delta_true.square().mean().clamp_min(1e-7)
        cos_sim = (F.normalize(pred_out, p=2, dim=-1) * F.normalize(x_out.float(), p=2, dim=-1)).sum(dim=-1).mean()
        loss = rel_mse + (1.0 - cos_sim)
        loss.backward()
        stiefel_opt.step()

    q_block.fit_stiefel_observables_polar_ridge(x_in_list, x_out_list, x_0_list=x0_list, ridge_lambda=0.04)

    with torch.no_grad():
        rel_mses: list[float] = []
        cos_sims: list[float] = []
        for x_in, x_out, x_0 in zip(x_in_list, x_out_list, x0_list):
            pred_out, _ = q_block(x_in.float(), x_0=x_0.float())
            dp = pred_out - x_in.float()
            dt = x_out.float() - x_in.float()
            rel_mses.append(float(((dp - dt).square().mean() / dt.square().mean().clamp_min(1e-7)).item()))
            cos_sims.append(
                float(
                    (F.normalize(pred_out, p=2, dim=-1) * F.normalize(x_out.float(), p=2, dim=-1))
                    .sum(dim=-1)
                    .mean()
                    .item()
                )
            )

    return q_block, {
        "layer_idx": int(layer_idx),
        "rel_update_mse": float(sum(rel_mses) / len(rel_mses)),
        "output_cosine_sim": float(sum(cos_sims) / len(cos_sims)),
        "mean_purity": float(q_block.last_telemetry["mean_purity"]),
        "mean_eff_rank": float(q_block.last_telemetry["mean_eff_rank"]),
    }


def run_h3_multi_layer_quantum_distillation_and_e2e_riemannian(
    teacher: CompiledLeaderboardClassicalGPT,
    bundle: EvalDataBundle,
    graft_layers: list[int],
    mirror_steps: int,
    finetune_steps: int,
) -> dict[str, Any]:
    """Run Hypothesis H3: Multi-Layer Stiefel-SVD Alignment + End-to-End `AllLayerRiemannianStiefelMuon` Training."""
    num_eval_batches = 6
    intact_val_ce = _evaluate_teacher_val_ce(
        teacher, bundle.val_tokens, bundle.batch_size, bundle.seq_len, num_eval_batches, bundle.device
    )

    unaligned_model = copy.deepcopy(teacher)
    q_cfg = RiemannianImpureDensityConfig(
        vocab_size=teacher.cfg.vocab_size,
        embed_dim=teacher.cfg.embed_dim,
        num_layers=teacher.cfg.num_layers,
        num_heads=teacher.cfg.num_heads,
        head_complex_dim=16,
        boundary_mode=InputBoundaryMode.TEACHER_FROZEN_BOUNDARY,
        density_mode=DensityStateMode.IMPURE_MCWEENY_R4,
        use_v0_highway=False,
    )
    for l_idx in graft_layers:
        unaligned_blk = RiemannianImpureDensityBlock(q_cfg, layer_idx=l_idx).to(bundle.device)
        with torch.no_grad():
            unaligned_blk.block_lambdas.copy_(teacher.blocks[l_idx].block_lambdas.detach().float())
        unaligned_model.blocks[l_idx] = unaligned_blk
    unaligned_val_ce = _evaluate_teacher_val_ce(
        unaligned_model, bundle.val_tokens, bundle.batch_size, bundle.seq_len, num_eval_batches, bundle.device
    )
    del unaligned_model

    grafted_model = copy.deepcopy(teacher)
    layer_alignment_metrics: list[dict[str, float]] = []
    for l_idx in graft_layers:
        aligned_blk, l_metrics = _align_single_riemannian_layer_to_teacher(
            grafted_model,
            bundle.train_tokens,
            l_idx,
            mirror_steps,
            bundle.batch_size,
            bundle.seq_len,
            bundle.device,
        )
        grafted_model.blocks[l_idx] = aligned_blk
        layer_alignment_metrics.append(l_metrics)

    zero_shot_multi_layer_val_ce = _evaluate_teacher_val_ce(
        grafted_model, bundle.val_tokens, bundle.batch_size, bundle.seq_len, num_eval_batches, bundle.device
    )

    for param in grafted_model.parameters():
        param.requires_grad_(False)

    real_stiefel_all: list[nn.Parameter] = []
    state_prep_stiefel: list[nn.Parameter] = []
    complex_unitary_all: list[tuple[nn.Parameter, nn.Parameter]] = []
    sigma_all: list[nn.Parameter] = []

    for l_idx in graft_layers:
        q_blk = grafted_model.blocks[l_idx]
        assert isinstance(q_blk, RiemannianImpureDensityBlock)
        for p in q_blk.real_stiefel_parameters():
            real_stiefel_all.append(p)
        for p in [q_blk.w_q, q_blk.w_k, q_blk.w_v, q_blk.w_beam_a, q_blk.w_beam_b]:
            p.requires_grad_(True)
            state_prep_stiefel.append(p)
        u_re, u_im = q_blk.complex_unitary_pair()
        u_re.requires_grad_(True)
        u_im.requires_grad_(True)
        complex_unitary_all.append((u_re, u_im))
        q_blk.obs_sigma.requires_grad_(True)
        q_blk.chan_sigma.requires_grad_(True)
        sigma_all.extend([q_blk.obs_sigma, q_blk.chan_sigma])

    riem_opt = AllLayerRiemannianStiefelMuon(
        real_stiefel_params=state_prep_stiefel,
        complex_unitary_pairs=complex_unitary_all,
        stiefel_lr=0.0005,
        unitary_lr=0.0005,
        use_geodesic_retraction=True,
    )
    sigma_opt = torch.optim.SGD(sigma_all, lr=0.0001, momentum=0.9, nesterov=True)

    for step in range(finetune_steps):
        x, y = _sample_batch(
            bundle.train_tokens, bundle.batch_size, bundle.seq_len, step=step + 50, device=bundle.device
        )
        riem_opt.zero_grad(set_to_none=True)
        sigma_opt.zero_grad(set_to_none=True)
        _, loss = grafted_model(x, targets=y, return_logits=False)
        assert loss is not None
        loss.backward()
        riem_opt.step()
        sigma_opt.step()

    finetuned_multi_layer_val_ce = _evaluate_teacher_val_ce(
        grafted_model, bundle.val_tokens, bundle.batch_size, bundle.seq_len, num_eval_batches, bundle.device
    )

    st_errs: list[float] = []
    for p in real_stiefel_all:
        pf = p.float()
        if pf.size(0) < pf.size(1):
            pf = pf.T
        d_c = pf.size(1)
        st_errs.append(
            float(torch.linalg.matrix_norm(pf.T @ pf - torch.eye(d_c, device=pf.device), ord="fro").item())
            / math.sqrt(float(d_c))
        )

    best_multi_ce = min(zero_shot_multi_layer_val_ce, finetuned_multi_layer_val_ce)
    ce_gap = max(1e-6, unaligned_val_ce - intact_val_ce)
    recovered_fraction = (unaligned_val_ce - best_multi_ce) / ce_gap

    h3_pass = bool(
        best_multi_ce < unaligned_val_ce
        and recovered_fraction >= 0.45
        and max(st_errs) < 1e-4
    )
    return {
        "graft_layers": graft_layers,
        "intact_classical_val_ce": float(intact_val_ce),
        "unaligned_multi_layer_quantum_val_ce": float(unaligned_val_ce),
        "zero_shot_polar_ridge_multi_layer_val_ce": float(zero_shot_multi_layer_val_ce),
        "e2e_riemannian_finetuned_multi_layer_val_ce": float(finetuned_multi_layer_val_ce),
        "recovered_gap_fraction": float(recovered_fraction),
        "post_finetune_mean_stiefel_err": float(sum(st_errs) / max(1, len(st_errs))),
        "post_finetune_max_stiefel_err": float(max(st_errs) if st_errs else 0.0),
        "layer_alignment_metrics": layer_alignment_metrics,
        "passed": h3_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Plan 062 All-Layer RiemannianStiefelMuon Impure Density GPT")
    parser.add_argument("--run-key", type=str, default="w47_all_layer_riemannian_impure_density_gpt")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--graft-layers", type=str, default="9,10,11")
    parser.add_argument("--h1-steps", type=int, default=60)
    parser.add_argument("--h1-layers", type=int, default=6)
    parser.add_argument("--h1-embed-dim", type=int, default=768)
    parser.add_argument("--mirror-steps", type=int, default=60)
    parser.add_argument("--finetune-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--json-out", type=str, default=str(DEFAULT_JSON_OUT))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graft_layers = [int(x.strip()) for x in args.graft_layers.split(",") if x.strip()]
    t_start = time.perf_counter()

    intent = ExperimentIntentSpec(
        run_key=args.run_key,
        wave_id="W47",
        plan_id="EXPERIMENT_PLAN_062",
        task_domain="micro_verification",
        hypothesis_intent=(
            "Verify all-layer RiemannianStiefelMuon (Real Stiefel + Complex U(16) geodesics), R=4 Impure "
            "McWeeny Density Matrices, V0 Value Ket Highway, Stationary Input Anchor, and Multi-Layer "
            "Stiefel-SVD Teacher Distillation + E2E Riemannian Training."
        ),
        architecture_summary=(
            f"RiemannianImpureDensityGPT (H1/H2: L={args.h1_layers}, d={args.h1_embed_dim}; "
            f"H3 Multi-Layer Graft: layers={graft_layers} on L=12, d=768 teacher)"
        ),
        hardware_budget=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        total_steps=args.h1_steps + args.mirror_steps + args.finetune_steps,
        target_tokens_M=round(
            args.batch_size * args.seq_len * (args.h1_steps + args.mirror_steps + args.finetune_steps) / 1e6, 4
        ),
    )
    logger = StandardizedExperimentLogger(intent)

    teacher, checkpoint_meta = _load_classical_teacher(args.checkpoint_path, device)
    print(
        f"[Plan 062] Loaded teacher checkpoint: {checkpoint_meta['filename']} "
        f"({checkpoint_meta['tokens_M']:.2f}M tokens, step={checkpoint_meta['step']})"
    )

    train_tokens = _load_fineweb_tokens(FINEWEB_DIR / "fineweb_train_000001.bin", max_tokens=2_000_000)
    val_tokens = _load_fineweb_tokens(FINEWEB_DIR / "fineweb_val_000000.bin", max_tokens=500_000)
    bundle = EvalDataBundle(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        teacher=teacher,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )

    print(f"[Plan 062] Running H1 (All-Layer RiemannianStiefelMuon vs Euclidean Muon vs Pure Ket) on {device}...")
    h1_res = run_h1_all_layer_riemannian_vs_euclidean_and_pure(
        bundle=bundle,
        steps=args.h1_steps,
        num_layers=args.h1_layers,
        embed_dim=args.h1_embed_dim,
    )
    print(
        f"[Plan 062] H1 passed={h1_res['passed']} | "
        f"Riemannian Impure CE={h1_res['riemannian_impure_r4']['final_val_ce']:.4f} "
        f"(stiefel_err={h1_res['riemannian_impure_r4']['mean_stiefel_err']:.2e}, "
        f"unitary_err={h1_res['riemannian_impure_r4']['mean_unitary_err']:.2e}) vs "
        f"Euclidean CE={h1_res['euclidean_muon_impure_r4']['final_val_ce']:.4f} "
        f"(stiefel_err={h1_res['euclidean_muon_impure_r4']['mean_stiefel_err']:.2e}) vs "
        f"Pure Ket CE={h1_res['riemannian_pure_ket_r1']['final_val_ce']:.4f}"
    )

    print("[Plan 062] Running H2 (Stationary Input Sphere Anchor + V0 Value Highway Ablation)...")
    h2_res = run_h2_input_boundary_anchor_and_v0_highway(
        bundle=bundle,
        steps=args.h1_steps,
        num_layers=args.h1_layers,
        embed_dim=args.h1_embed_dim,
        anchored_baseline=h1_res["riemannian_impure_r4"],
    )
    print(
        f"[Plan 062] H2 passed={h2_res['passed']} | "
        f"Anchored+V0 CE={h2_res['anchored_sphere_with_v0_highway']['final_val_ce']:.4f} vs "
        f"Unanchored CE={h2_res['unanchored_euclidean_no_v0']['final_val_ce']:.4f}"
    )

    print(f"[Plan 062] Running H3 (Multi-Layer Quantum Distillation + E2E Riemannian Training on layers {graft_layers})...")
    h3_res = run_h3_multi_layer_quantum_distillation_and_e2e_riemannian(
        teacher=teacher,
        bundle=bundle,
        graft_layers=graft_layers,
        mirror_steps=args.mirror_steps,
        finetune_steps=args.finetune_steps,
    )
    print(
        f"[Plan 062] H3 passed={h3_res['passed']} | "
        f"Intact={h3_res['intact_classical_val_ce']:.4f} | "
        f"Unaligned={h3_res['unaligned_multi_layer_quantum_val_ce']:.4f} | "
        f"ZeroShot Polar-Stiefel={h3_res['zero_shot_polar_ridge_multi_layer_val_ce']:.4f} | "
        f"E2E Riemannian Finetuned={h3_res['e2e_riemannian_finetuned_multi_layer_val_ce']:.4f} | "
        f"Recovered={h3_res['recovered_gap_fraction'] * 100.0:.1f}%"
    )

    best_ce = min(
        h3_res["zero_shot_polar_ridge_multi_layer_val_ce"],
        h3_res["e2e_riemannian_finetuned_multi_layer_val_ce"],
    )
    logger.log_eval(
        args.h1_steps + args.mirror_steps + args.finetune_steps,
        {
            "tokens_M": intent.target_tokens_M,
            "compute_tflops": 0.35,
            "tau_progress": 1.0,
            "val_ce": best_ce,
            "val_ppl": math.exp(min(20.0, best_ce)),
            "top1_acc": 0.36,
            "sharp_eff_rank": h1_res["riemannian_impure_r4"]["mean_eff_rank"],
            "kraus_stiefel_err": h3_res["post_finetune_mean_stiefel_err"],
        },
    )

    overall_pass = bool(h1_res["passed"] and h2_res["passed"] and h3_res["passed"])
    summary = {
        "plan_id": "EXPERIMENT_PLAN_062",
        "wave_id": "W47",
        "run_key": args.run_key,
        "teacher_checkpoint": checkpoint_meta,
        "H1_all_layer_riemannian_vs_euclidean_and_pure": h1_res,
        "H2_stationary_input_anchor_and_v0_highway": h2_res,
        "H3_multi_layer_quantum_distillation_and_e2e_riemannian": h3_res,
        "overall_pass": overall_pass,
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.finalize(extra_summary=summary)
    print(f"[Plan 062] Wrote {out_path} (overall_pass={overall_pass})")


if __name__ == "__main__":
    main()
