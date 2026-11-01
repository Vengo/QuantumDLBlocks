"""Wave 50 Diagnostic CLI: Per-Matrix 5-Subspace Stiefel & Unitary Normal/Tangent Gradient Decomposer.

Decomposes backpropagation gradients across every matrix role in `ModularQuantumStatMechGPT`
(`w_in_stiefel`, `w_q`, `w_k`, `w_v`, `w_obs`, `w_beam_a`, `w_beam_b`, `w_chan`, `head_unitary`,
and vocabulary spheres `wte`, `lm_head`) into five mutually orthogonal Frobenius subspaces:
1. Horizontal Tangent: $G_{\\text{horiz}} = (I_D - W W^\\top) G$ ($\\dim = (D - d)d$)
2. Vertical Skew-Symmetric Tangent: $G_{\\text{skew}} = W\\,\\mathrm{skew}(W^\\top G)$ ($\\dim = d(d-1)/2$)
3. 1D Scalar Trace Normal: $G_{\\text{trace}} = W\\,(\\frac{\\operatorname{Tr}(W^\\top G)}{d} I_d)$ ($\\dim = 1$)
4. $(d-1)$-D Traceless Diagonal Normal: $G_{\\text{diag\\_tl}} = W\\,\\operatorname{diag}(\\operatorname{diag}(W^\\top G) - \\frac{\\operatorname{Tr}(W^\\top G)}{d}\\mathbf{1}_d)$ ($\\dim = d-1$)
5. $d(d-1)/2$-D Off-Diagonal Symmetric Normal: $G_{\\text{off\\_sym}} = W\\,\\mathrm{sym}_{\\text{off}}(W^\\top G)$ ($\\dim = d(d-1)/2$)

Reports both Frobenius norm fractions ($\\|G_{\\text{sub}}\\|_F / \\|G\\|_F$) and Pythagorean squared-energy
fractions ($\\|G_{\\text{sub}}\\|_F^2 / \\|G\\|_F^2$), alongside the exact dimensional equipartition baseline
and per-DOF enrichment ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from qdlblocks.layers.isometric_born_trace_readout import VocabularyReadoutMode
from qdlblocks.layers.quantum_statmech_channels import (
    DEFAULT_KRAUS_CHANNEL_RANK,
    EntropyGateMode,
    PurificationMode,
    QuantumChannelMode,
)
from qdlblocks.models.modular_quantum_statmech_gpt import (
    DEFAULT_STATMECH_EMBED_DIM,
    DEFAULT_STATMECH_NUM_HEADS,
    DEFAULT_STATMECH_NUM_LAYERS,
    DEFAULT_STATMECH_VOCAB_SIZE,
    ModularQuantumStatMechConfig,
    ModularQuantumStatMechGPT,
)

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7)
# ---------------------------------------------------------------------------
CACHE_DIR = REPO_ROOT / ".cache"
FINEWEB_DIR = CACHE_DIR / "fineweb_100M"
DEFAULT_JSON_OUT = CACHE_DIR / "w50_stiefel_normal_gradient_decomposition.json"
SHARD_HEADER_INTS: int = 256
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_SEQ_LEN: int = 1024
DIAG_NORM_EPS: float = 1e-30


def decompose_real_stiefel_gradient_5subspace(
    weight: torch.Tensor,
    grad: torch.Tensor,
) -> dict[str, float]:
    """Decompose `grad` on $\\mathrm{St}(D, d)$ into 5 mutually orthogonal subspaces (`horiz`, `skew`, `trace`, `diag_tl`, `off_sym`)."""
    w = weight.detach().float()
    g = torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if w.size(-2) < w.size(-1):
        w = w.transpose(-2, -1)
        g = g.transpose(-2, -1)

    rows_d = int(w.size(-2))
    cols_d = int(w.size(-1))
    total_dofs = float(rows_d * cols_d)
    horiz_dofs = float((rows_d - cols_d) * cols_d)
    skew_dofs = float(cols_d * (cols_d - 1) // 2)
    trace_dofs = 1.0
    diag_tl_dofs = float(cols_d - 1)
    off_sym_dofs = float(cols_d * (cols_d - 1) // 2)

    p_mat = torch.matmul(w.transpose(-2, -1), g)
    p_skew = 0.5 * (p_mat - p_mat.transpose(-2, -1))
    p_sym = 0.5 * (p_mat + p_mat.transpose(-2, -1))

    diag_vec = torch.diagonal(p_sym, dim1=-2, dim2=-1)
    trace_val = diag_vec.sum(dim=-1, keepdim=True)
    mean_diag = trace_val / float(cols_d)
    eye_d = torch.eye(cols_d, device=w.device, dtype=w.dtype)

    p_trace = mean_diag.unsqueeze(-1) * eye_d
    p_diag_tl = torch.diag_embed(diag_vec - mean_diag)
    p_off_sym = p_sym - torch.diag_embed(diag_vec)

    g_proj_span = torch.matmul(w, p_mat)
    g_horiz = g - g_proj_span
    g_skew = torch.matmul(w, p_skew)
    g_trace = torch.matmul(w, p_trace)
    g_diag_tl = torch.matmul(w, p_diag_tl)
    g_off_sym = torch.matmul(w, p_off_sym)

    g_tan = g_horiz + g_skew
    g_norm = g_trace + g_diag_tl + g_off_sym

    tot_sq = g.square().sum().clamp_min(DIAG_NORM_EPS)
    tot_fro = tot_sq.sqrt()

    def _nf(t: torch.Tensor) -> float:
        return float((t.square().sum().sqrt() / tot_fro).item())

    def _ef(t: torch.Tensor) -> float:
        return float((t.square().sum() / tot_sq).item())

    e_horiz = _ef(g_horiz)
    e_skew = _ef(g_skew)
    e_trace = _ef(g_trace)
    e_diag_tl = _ef(g_diag_tl)
    e_off_sym = _ef(g_off_sym)

    iso_horiz = horiz_dofs / total_dofs
    iso_skew = skew_dofs / total_dofs
    iso_trace = trace_dofs / total_dofs
    iso_diag_tl = diag_tl_dofs / total_dofs
    iso_off_sym = off_sym_dofs / total_dofs

    return {
        "rows_D": float(rows_d),
        "cols_d": float(cols_d),
        "grad_fro_norm": float(tot_fro.item()),
        "norm_frac_tan_total": _nf(g_tan),
        "norm_frac_horiz_tan": _nf(g_horiz),
        "norm_frac_skew_tan": _nf(g_skew),
        "norm_frac_normal_total": _nf(g_norm),
        "norm_frac_trace_normal": _nf(g_trace),
        "norm_frac_diag_tl_normal": _nf(g_diag_tl),
        "norm_frac_off_sym_normal": _nf(g_off_sym),
        "energy_frac_tan_total": _ef(g_tan),
        "energy_frac_horiz_tan": e_horiz,
        "energy_frac_skew_tan": e_skew,
        "energy_frac_normal_total": _ef(g_norm),
        "energy_frac_trace_normal": e_trace,
        "energy_frac_diag_tl_normal": e_diag_tl,
        "energy_frac_off_sym_normal": e_off_sym,
        "iso_energy_horiz_tan": iso_horiz,
        "iso_energy_skew_tan": iso_skew,
        "iso_energy_normal_total": (trace_dofs + diag_tl_dofs + off_sym_dofs) / total_dofs,
        "iso_energy_diag_tl_normal": iso_diag_tl,
        "iso_energy_off_sym_normal": iso_off_sym,
        "dof_enrichment_horiz_tan": e_horiz / max(1e-12, iso_horiz) if iso_horiz > 0 else 0.0,
        "dof_enrichment_skew_tan": e_skew / max(1e-12, iso_skew),
        "dof_enrichment_trace_normal": e_trace / max(1e-12, iso_trace),
        "dof_enrichment_diag_tl_normal": e_diag_tl / max(1e-12, iso_diag_tl),
        "dof_enrichment_off_sym_normal": e_off_sym / max(1e-12, iso_off_sym),
    }


def decompose_complex_unitary_gradient_subspaces(
    u_re: torch.Tensor,
    u_im: torch.Tensor,
    g_re: torch.Tensor,
    g_im: torch.Tensor,
) -> dict[str, float]:
    """Decompose complex `head_unitary` `(H, D, d)` gradient into skew-Hermitian tangent and Hermitian normal parts."""
    ar = u_re.detach().float()
    ai = u_im.detach().float()
    gr = torch.nan_to_num(g_re.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    gi = torch.nan_to_num(g_im.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)

    d_col = int(ar.size(-1))
    at_re = ar.transpose(-2, -1)
    at_im = -ai.transpose(-2, -1)

    p_re = torch.matmul(at_re, gr) - torch.matmul(at_im, gi)
    p_im = torch.matmul(at_re, gi) + torch.matmul(at_im, gr)

    herm_re = 0.5 * (p_re + p_re.transpose(-2, -1))
    herm_im = 0.5 * (p_im - p_im.transpose(-2, -1))
    skew_re = 0.5 * (p_re - p_re.transpose(-2, -1))
    skew_im = 0.5 * (p_im + p_im.transpose(-2, -1))

    diag_vec = torch.diagonal(herm_re, dim1=-2, dim2=-1)
    mean_diag = diag_vec.sum(dim=-1, keepdim=True) / float(d_col)
    eye_d = torch.eye(d_col, device=ar.device, dtype=ar.dtype).unsqueeze(0)

    p_trace_re = mean_diag.unsqueeze(-1) * eye_d
    p_diag_tl_re = torch.diag_embed(diag_vec - mean_diag)
    p_off_herm_re = herm_re - torch.diag_embed(diag_vec)
    p_off_herm_im = herm_im

    tot_sq = (gr.square().sum(dim=(-2, -1)) + gi.square().sum(dim=(-2, -1))).clamp_min(DIAG_NORM_EPS)
    skew_sq = skew_re.square().sum(dim=(-2, -1)) + skew_im.square().sum(dim=(-2, -1))
    herm_sq = herm_re.square().sum(dim=(-2, -1)) + herm_im.square().sum(dim=(-2, -1))
    trace_sq = p_trace_re.square().sum(dim=(-2, -1))
    diag_tl_sq = p_diag_tl_re.square().sum(dim=(-2, -1))
    off_herm_sq = p_off_herm_re.square().sum(dim=(-2, -1)) + p_off_herm_im.square().sum(dim=(-2, -1))
    horiz_sq = (tot_sq - skew_sq - herm_sq).clamp_min(0.0)

    return {
        "grad_fro_norm": float(tot_sq.sqrt().mean().item()),
        "norm_frac_tan_total": float(((horiz_sq + skew_sq).sqrt() / tot_sq.sqrt()).mean().item()),
        "norm_frac_skew_tan": float((skew_sq.sqrt() / tot_sq.sqrt()).mean().item()),
        "norm_frac_normal_total": float((herm_sq.sqrt() / tot_sq.sqrt()).mean().item()),
        "norm_frac_trace_normal": float((trace_sq.sqrt() / tot_sq.sqrt()).mean().item()),
        "norm_frac_diag_tl_normal": float((diag_tl_sq.sqrt() / tot_sq.sqrt()).mean().item()),
        "norm_frac_off_sym_normal": float((off_herm_sq.sqrt() / tot_sq.sqrt()).mean().item()),
        "energy_frac_tan_total": float(((horiz_sq + skew_sq) / tot_sq).mean().item()),
        "energy_frac_skew_tan": float((skew_sq / tot_sq).mean().item()),
        "energy_frac_normal_total": float((herm_sq / tot_sq).mean().item()),
        "energy_frac_trace_normal": float((trace_sq / tot_sq).mean().item()),
        "energy_frac_diag_tl_normal": float((diag_tl_sq / tot_sq).mean().item()),
        "energy_frac_off_sym_normal": float((off_herm_sq / tot_sq).mean().item()),
    }


def load_batch_at_step(shard_path: Path, step_offset: int, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a deterministic `(B, T)` FineWeb batch at `step_offset`."""
    raw = np.memmap(shard_path, dtype=np.uint16, mode="r")
    header = np.fromfile(shard_path, dtype=np.int32, count=SHARD_HEADER_INTS)
    tokens = raw[SHARD_HEADER_INTS * 2 :] if (len(header) >= 3 and int(header[0]) == 20240520) else raw
    span = batch_size * (seq_len + 1)
    offset = ((step_offset + 1) * span) % max(span, len(tokens) - span)
    chunk = np.asarray(tokens[offset : offset + span], dtype=np.int64)
    t_cpu = torch.from_numpy(chunk).clamp_(0, DEFAULT_STATMECH_VOCAB_SIZE - 1).view(batch_size, seq_len + 1)
    return t_cpu[:, :-1].contiguous().to(device), t_cpu[:, 1:].contiguous().to(device)


def profile_model_gradients(
    model: ModularQuantumStatMechGPT,
    shard_path: Path,
    step_offset: int,
    num_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    """Accumulate gradients over `num_batches` and return per-role and per-layer 5-subspace breakdown."""
    model.zero_grad(set_to_none=True)
    losses: list[float] = []
    for b_idx in range(num_batches):
        idx, targets = load_batch_at_step(shard_path, step_offset + b_idx, DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, device)
        _, loss = model(idx, targets=targets, return_logits=False)
        assert loss is not None
        (loss / float(num_batches)).backward()
        losses.append(float(loss.detach().item()))

    role_accum: dict[str, list[dict[str, float]]] = {
        "w_in_stiefel": [decompose_real_stiefel_gradient_5subspace(model.w_in_stiefel, model.w_in_stiefel.grad)],  # type: ignore[arg-type]
        "w_q": [],
        "w_k": [],
        "w_v": [],
        "w_obs": [],
        "w_beam_a": [],
        "w_beam_b": [],
        "w_chan": [],
        "head_unitary": [],
    }
    per_layer_summary: list[dict[str, Any]] = []

    for l_idx, blk in enumerate(model.blocks):
        dq = decompose_real_stiefel_gradient_5subspace(blk.w_q, blk.w_q.grad)  # type: ignore[arg-type]
        dk = decompose_real_stiefel_gradient_5subspace(blk.w_k, blk.w_k.grad)  # type: ignore[arg-type]
        dv = decompose_real_stiefel_gradient_5subspace(blk.w_v, blk.w_v.grad)  # type: ignore[arg-type]
        dobs = decompose_real_stiefel_gradient_5subspace(blk.w_obs, blk.w_obs.grad)  # type: ignore[arg-type]
        dba = decompose_real_stiefel_gradient_5subspace(blk.w_beam_a, blk.w_beam_a.grad)  # type: ignore[arg-type]
        dbb = decompose_real_stiefel_gradient_5subspace(blk.w_beam_b, blk.w_beam_b.grad)  # type: ignore[arg-type]
        dch = decompose_real_stiefel_gradient_5subspace(blk.w_chan, blk.w_chan.grad)  # type: ignore[arg-type]
        du = decompose_complex_unitary_gradient_subspaces(
            blk.head_unitary_re, blk.head_unitary_im, blk.head_unitary_re.grad, blk.head_unitary_im.grad  # type: ignore[arg-type]
        )

        role_accum["w_q"].append(dq)
        role_accum["w_k"].append(dk)
        role_accum["w_v"].append(dv)
        role_accum["w_obs"].append(dobs)
        role_accum["w_beam_a"].append(dba)
        role_accum["w_beam_b"].append(dbb)
        role_accum["w_chan"].append(dch)
        role_accum["head_unitary"].append(du)

        per_layer_summary.append(
            {
                "layer": l_idx,
                "w_q_normal_frac": dq["norm_frac_normal_total"],
                "w_k_normal_frac": dk["norm_frac_normal_total"],
                "w_v_normal_frac": dv["norm_frac_normal_total"],
                "w_obs_normal_frac": dobs["norm_frac_normal_total"],
                "w_beam_a_normal_frac": dba["norm_frac_normal_total"],
                "w_beam_b_normal_frac": dbb["norm_frac_normal_total"],
                "w_chan_normal_frac": dch["norm_frac_normal_total"],
                "unitary_normal_frac": du["norm_frac_normal_total"],
            }
        )

    role_means: dict[str, dict[str, float]] = {}
    for role, dlist in role_accum.items():
        keys = dlist[0].keys()
        role_means[role] = {k: float(sum(d[k] for d in dlist) / len(dlist)) for k in keys}

    return {
        "mean_train_loss": float(sum(losses) / len(losses)),
        "role_summary": role_means,
        "per_layer_normal_frac": per_layer_summary,
    }


def _capture_block0_context_and_obs_grad(
    model: ModularQuantumStatMechGPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run 1 forward-backward pass and capture `packed_ctx_all` and `obs_320.grad` on `model.blocks[0]`."""
    import torch.nn.functional as F
    from qdlblocks.layers.impure_density_mirror_block import (
        TRACE_NORM_EPS,
        complex_l2_normalize,
        fast_rank_r_hermitian_and_ket_pack,
    )
    from qdlblocks.layers.quantum_statmech_channels import (
        causal_born_p3_weights,
        compiled_statmech_purify_evolve_uhlmann_krylov_step,
    )

    blk0 = model.blocks[0]
    captured: dict[str, torch.Tensor] = {}
    orig_fn = blk0.compute_quantum_attention_features

    def _hook_attn(x_base: torch.Tensor, v_0: torch.Tensor | None = None) -> Any:
        bsz, seq_len, _ = x_base.shape
        work_dtype = x_base.dtype
        h_norm = F.rms_norm(x_base, (blk0.embed_dim,), eps=TRACE_NORM_EPS)
        q_raw = torch.matmul(h_norm, blk0.w_q.to(work_dtype)).view(bsz, seq_len, blk0.num_heads, 2, blk0.qk_complex_dim)
        k_raw = torch.matmul(h_norm, blk0.w_k.to(work_dtype)).view(bsz, seq_len, blk0.num_heads, 2, blk0.qk_complex_dim)
        v_proj = torch.matmul(h_norm, blk0.w_v.to(work_dtype))
        q_re, q_im = blk0._apply_complex_rope(
            *complex_l2_normalize(q_raw[..., 0, :].transpose(1, 2), q_raw[..., 1, :].transpose(1, 2))
        )
        k_re, k_im = blk0._apply_complex_rope(
            *complex_l2_normalize(k_raw[..., 0, :].transpose(1, 2), k_raw[..., 1, :].transpose(1, 2))
        )
        q_cat, k_cat, k_skew = torch.cat([q_re, q_im], -1), torch.cat([k_re, k_im], -1), torch.cat([-k_im, k_re], -1)
        attn_w = causal_born_p3_weights(torch.matmul(q_cat, k_cat.transpose(-2, -1)), torch.matmul(q_cat, k_skew.transpose(-2, -1)))
        v_raw = v_proj.view(bsz, seq_len, blk0.num_heads, blk0.v_rank, 2, blk0.d_h)
        v_re, v_im = v_raw[..., 0, :].permute(0, 2, 1, 3, 4), v_raw[..., 1, :].permute(0, 2, 1, 3, 4)
        v_flat = torch.cat([v_re, v_im], dim=-1).reshape(bsz, blk0.num_heads, seq_len, -1)
        p_288 = fast_rank_r_hermitian_and_ket_pack(v_re, v_im, blk0.re_pack_mask, blk0.im_pack_mask)
        p_ctx = torch.matmul(attn_w, torch.cat([p_288, v_flat], dim=-1))
        captured["packed_ctx_all"] = p_ctx.detach().float()
        obs_320, _, p_rho_t1, lam1, lam2, _, _ = compiled_statmech_purify_evolve_uhlmann_krylov_step(
            p_ctx, blk0.head_unitary_re, blk0.head_unitary_im, blk0.re_pack_mask, blk0.im_pack_mask
        )
        obs_320.retain_grad()
        captured["obs_320"] = obs_320
        l1_b, l2_b = lam1.mean(dim=1), lam2.mean(dim=1)
        l_sum = (l1_b + l2_b).clamp_min(1e-6)
        kw = torch.stack([l1_b / l_sum, l2_b / l_sum], dim=-1)
        return (
            obs_320.to(work_dtype).transpose(1, 2).reshape(bsz, seq_len, blk0.total_obs_dim),
            v_proj,
            kw,
            p_rho_t1.to(work_dtype).transpose(1, 2).reshape(bsz, seq_len, -1),
        )

    blk0.compute_quantum_attention_features = _hook_attn  # type: ignore[method-assign]
    _, loss_full = model(idx, targets=targets, return_logits=False)
    assert loss_full is not None
    loss_full.backward()
    blk0.compute_quantum_attention_features = orig_fn  # type: ignore[method-assign]
    return captured["packed_ctx_all"], captured["obs_320"].grad.detach().float()


def run_pathway_isolation_audit(
    shard_path: Path,
    ckpt_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Isolate the exact forward-pass sources of normal gradients on `head_unitary`, `w_v`, and `w_beam_a/b`."""
    import torch.nn.functional as F
    from qdlblocks.layers.impure_density_mirror_block import (
        DEFAULT_UHLMANN_PHASE_EPS,
        INV_SQRT2_ISOMETRIC_SCALE,
        TRACE_NORM_EPS,
        extract_rank2_krylov_eigenpair_uhlmann,
        unpack_hermitian_isometric,
    )
    from qdlblocks.layers.quantum_statmech_channels import (
        evolve_head_unitary_functional,
        purify_density_statmech,
    )

    idx, targets = load_batch_at_step(shard_path, 0, DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, device)
    results: dict[str, Any] = {}

    torch.manual_seed(2026063)
    cfg = ModularQuantumStatMechConfig()
    model = ModularQuantumStatMechGPT(cfg, kraus_rank=DEFAULT_KRAUS_CHANNEL_RANK).to(device)
    blk0 = model.blocks[0]
    packed_all, grad_probe = _capture_block0_context_and_obs_grad(model, idx, targets)

    d_h = blk0.d_h
    d_sq = d_h * d_h
    base_pack_dim = d_sq + 2 * d_h
    packed_rho0 = packed_all[..., :d_sq]
    v_bar_re = packed_all[..., d_sq : d_sq + d_h]
    v_bar_im = packed_all[..., d_sq + d_h : base_pack_dim]
    psi_bar = packed_all[..., base_pack_dim:].view(DEFAULT_BATCH_SIZE, blk0.num_heads, DEFAULT_SEQ_LEN, blk0.v_rank, 2, d_h)
    rho0_re, rho0_im = unpack_hermitian_isometric(packed_rho0, d_h)
    rho1_re, rho1_im, _, _ = purify_density_statmech(rho0_re, rho0_im)

    for sub_mode in ["full_obs320", "only_rho_raw", "only_rho_traceless", "only_krylov", "only_psi_u"]:
        u_re = blk0.head_unitary_re.detach().float().clone().requires_grad_(True)
        u_im = blk0.head_unitary_im.detach().float().clone().requires_grad_(True)
        if sub_mode == "only_rho_traceless":
            eye_h = torch.eye(d_h, device=device, dtype=torch.float32) / float(d_h)
            r_in_re = rho1_re - eye_h
            ro_re, ro_im = evolve_head_unitary_functional(r_in_re, rho1_im, u_re, u_im)
            ro_re = ro_re + eye_h
        else:
            ro_re, ro_im = evolve_head_unitary_functional(rho1_re, rho1_im, u_re, u_im)

        u0_re = u_re.unsqueeze(0).unsqueeze(2)
        u0_im = u_im.unsqueeze(0).unsqueeze(2)
        ut_re = u0_re.transpose(-2, -1)
        ut_im = u0_im.transpose(-2, -1)
        v_u_re = torch.matmul(v_bar_re.unsqueeze(-2), ut_re).squeeze(-2) - torch.matmul(v_bar_im.unsqueeze(-2), ut_im).squeeze(-2)
        v_u_im = torch.matmul(v_bar_re.unsqueeze(-2), ut_im).squeeze(-2) + torch.matmul(v_bar_im.unsqueeze(-2), ut_re).squeeze(-2)
        packed_rho_t1 = (ro_re * blk0.re_pack_mask + ro_im * blk0.im_pack_mask).flatten(start_dim=-2, end_dim=-1)
        packed_rho_out = F.rms_norm(packed_rho_t1, (d_sq,), eps=TRACE_NORM_EPS)
        lam1, lam2, phi1_re, phi1_im, phi2_re, phi2_im = extract_rank2_krylov_eigenpair_uhlmann(
            ro_re, ro_im, v_u_re, v_u_im, eps=DEFAULT_UHLMANN_PHASE_EPS
        )
        s1 = torch.sqrt(lam1.clamp_min(DEFAULT_UHLMANN_PHASE_EPS)).unsqueeze(-1)
        s2 = torch.sqrt(lam2.clamp_min(DEFAULT_UHLMANN_PHASE_EPS)).unsqueeze(-1)
        krylov_64 = F.rms_norm(
            torch.cat([s1 * phi1_re, s1 * phi1_im, s2 * phi2_re, s2 * phi2_im], dim=-1), (4 * d_h,), eps=TRACE_NORM_EPS
        )
        psi_u_re = torch.matmul(psi_bar[..., 0, :], ut_re) - torch.matmul(psi_bar[..., 1, :], ut_im)
        psi_u_im = torch.matmul(psi_bar[..., 0, :], ut_im) + torch.matmul(psi_bar[..., 1, :], ut_re)
        psi_u_flat = F.rms_norm(torch.cat([psi_u_re, psi_u_im], dim=-1).flatten(-2, -1), (128,), eps=TRACE_NORM_EPS)

        if sub_mode in ("only_rho_raw", "only_rho_traceless"):
            loss_sub = (packed_rho_out * grad_probe[..., :256]).sum()
        elif sub_mode == "only_krylov":
            loss_sub = (krylov_64 * grad_probe[..., 256:]).sum()
        elif sub_mode == "only_psi_u":
            loss_sub = (psi_u_flat * grad_probe[..., :128]).sum()
        else:
            rho_plus_psi = torch.cat(
                [INV_SQRT2_ISOMETRIC_SCALE * (packed_rho_out[..., :128] + psi_u_flat), packed_rho_out[..., 128:]], dim=-1
            )
            obs_320 = torch.cat([rho_plus_psi, krylov_64], dim=-1)
            loss_sub = (obs_320 * grad_probe).sum()
        loss_sub.backward()
        assert u_re.grad is not None and u_im.grad is not None
        results[sub_mode] = decompose_complex_unitary_gradient_subspaces(u_re, u_im, u_re.grad, u_im.grad)
        print(
            f"[Unitary Pathway {sub_mode:<20}] NormF={results[sub_mode]['norm_frac_normal_total']:.4f} "
            f"SkewE%={results[sub_mode]['energy_frac_skew_tan']*100:>6.2f}% "
            f"DiagTLE%={results[sub_mode]['energy_frac_diag_tl_normal']*100:>6.2f}% "
            f"OffHermE%={results[sub_mode]['energy_frac_off_sym_normal']*100:>6.2f}% "
            f"||G||_F={results[sub_mode]['grad_fro_norm']:.4e}"
        )

    del model
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Subspace Stiefel & Unitary Normal Gradient Decomposer")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--json-out", type=str, default=str(DEFAULT_JSON_OUT))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    shard_path = sorted(FINEWEB_DIR.glob("fineweb_train_*.bin"))[0]

    checkpoints_to_analyze: list[tuple[str, Path | None, int, EntropyGateMode]] = [
        ("step0_fresh_init", None, 0, EntropyGateMode.NONE),
        ("step0_isometric_clean", None, 0, EntropyGateMode.ISOMETRIC_NORMALIZED_BRANCHES),
        (
            "ckpt_150m_riemannian_tangent",
            CACHE_DIR / "checkpoints" / "w49_v3_riemannian_tangent_150m.pt",
            18311,
            EntropyGateMode.NONE,
        ),
        (
            "ckpt_150m_isometric_clean",
            CACHE_DIR / "checkpoints" / "w49_v3_riemannian_tangent_150m.pt",
            18311,
            EntropyGateMode.ISOMETRIC_NORMALIZED_BRANCHES,
        ),
        (
            "ckpt_200m_euclidean_rebuilt",
            CACHE_DIR / "checkpoints" / "w49_v3_rebuilt_200m_step24414.pt",
            24414,
            EntropyGateMode.NONE,
        ),
    ]

    report: dict[str, Any] = {}
    for label, ckpt_path, step_offset, gate_mode in checkpoints_to_analyze:
        torch.manual_seed(2026063)
        cfg = ModularQuantumStatMechConfig(
            vocab_size=DEFAULT_STATMECH_VOCAB_SIZE,
            n_layer=DEFAULT_STATMECH_NUM_LAYERS,
            n_head=DEFAULT_STATMECH_NUM_HEADS,
            n_embd=DEFAULT_STATMECH_EMBED_DIM,
            purification_mode=PurificationMode.QUADRATIC_MCWEENY,
            channel_mode=QuantumChannelMode.UNITARY_U16,
            entropy_gate_mode=gate_mode,
            readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
        )
        model = ModularQuantumStatMechGPT(cfg, kraus_rank=DEFAULT_KRAUS_CHANNEL_RANK).to(device)
        model.set_compiled_krylov(True)
        if ckpt_path is not None and ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            del payload
        res = profile_model_gradients(model, shard_path, step_offset, int(args.num_batches), device)
        del model
        torch.cuda.empty_cache()
        report[label] = res

        print(f"\n=== {label} (loss={res['mean_train_loss']:.4f}) ===")
        header = (
            f"{'Role':<15} | {'Shape':<11} | {'NormF':>6} {'IsoNF':>6} | "
            f"{'HorizE%':>7} {'SkewE%':>7} | {'TraceE%':>7} {'DiagTLE%':>8} {'OffSymE%':>8} | {'DiagEnrich':>10}"
        )
        print(header)
        print("-" * len(header))
        for role, m in res["role_summary"].items():
            if role == "head_unitary":
                print(
                    f"{role:<15} | {'(6,16,16)':<11} | {m['norm_frac_normal_total']:>6.4f} {0.7071:>6.4f} | "
                    f"{0.0:>6.2f}% {m['energy_frac_skew_tan']*100:>6.2f}% | "
                    f"{m['energy_frac_trace_normal']*100:>6.2f}% {m['energy_frac_diag_tl_normal']*100:>7.2f}% "
                    f"{m['energy_frac_off_sym_normal']*100:>7.2f}% | {'N/A':>10}"
                )
            else:
                sh = f"({int(m['rows_D'])},{int(m['cols_d'])})"
                iso_nf = m["iso_energy_normal_total"] ** 0.5
                print(
                    f"{role:<15} | {sh:<11} | {m['norm_frac_normal_total']:>6.4f} {iso_nf:>6.4f} | "
                    f"{m['energy_frac_horiz_tan']*100:>6.2f}% {m['energy_frac_skew_tan']*100:>6.2f}% | "
                    f"{m['energy_frac_trace_normal']*100:>6.2f}% {m['energy_frac_diag_tl_normal']*100:>7.2f}% "
                    f"{m['energy_frac_off_sym_normal']*100:>7.2f}% | {m['dof_enrichment_diag_tl_normal']:>9.2f}x"
                )

    out_p = Path(args.json_out)
    report["unitary_pathway_isolation"] = run_pathway_isolation_audit(
        shard_path, CACHE_DIR / "checkpoints" / "w49_v3_riemannian_tangent_150m.pt", device
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
