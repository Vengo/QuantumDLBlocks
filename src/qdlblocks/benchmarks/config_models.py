"""Typed Dataclass Configurations & Shared Builders for QuantumBornGPT Experiments (Plan 033).

Replaces 90+ argument function signatures and duplicated per-script optimizer/initialization
routines with composable `@dataclass` configs (`BackboneConfig`, `QubitHierarchyConfig`,
`KBosonReadoutConfig`, `OptimizerConfig`, `QuantumExperimentSpec`) and canonical builders
where every function accepts <= 8 arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.hierarchical_qubit import initialize_hierarchical_16qubit_with_ppmi_svd
from ..models.born_gpt import QuantumBornGPT
from ..optim.muon import StiefelMuon, polar_snap_weights, split_muon_adamw_params


@dataclass(frozen=True)
class BackboneConfig:
    """Core geometric Transformer backbone dimensions and attention topology."""

    num_layers: int = 4
    embed_dim: int = 256
    seed_dim: int = 256
    num_heads: int = 4
    frame_expansion: float = 2.0
    block_size: int = 512
    window_size: int = 384
    num_value_embeddings: int = 2
    ngram_vocab_size: int = 65536
    ngram_inject_scale: float = 0.12
    use_mach_zehnder_beamsplitter: bool = False
    use_wavelet_cholesky_branch: bool = False


@dataclass(frozen=True)
class QubitHierarchyConfig:
    """16-Qubit Product-Register Hierarchy and Step-0 PPMI-SVD initialization config."""

    qubit_registers: int = 16
    qubit_amplitudes: int = 16
    num_higher_states: int = 128
    use_ortho_decouple: bool = False
    decouple_self_in_full_ambient: bool = True
    directed_predecessor_mix: float = 0.35
    ppmi_alpha: float = 0.75
    use_wavelet_cholesky_basis: bool = False


@dataclass(frozen=True)
class KBosonReadoutConfig:
    """Two-Beam k-Boson Bloch fidelity readout and Scheduled Stiefel Kraus bath config."""

    diffuse_rank: int = 16
    diffuse_schedule: str = "warmup_hold_anneal"
    lambda_diff_start: float = 0.22
    lambda_diff_end: float = 0.12
    lambda_diff_min: float = 0.08
    lambda_diff_max: float = 0.35
    contextual_purity_gain: float = 0.15
    quadratic_diffuse_fidelity: float = 0.50
    kraus_spectral_floor: float = 0.0
    use_stereographic_lift: bool = True
    use_cosine_phase_beam_gate: bool = True
    povm_temperature: float = 0.030


@dataclass(frozen=True)
class OptimizerConfig:
    """StiefelMuon + Multi-Group AdamW learning rates and Stiefel retraction settings."""

    muon_lr: float = 0.018
    body_adam_lr: float = 0.003
    qubit_embed_lr: float = 0.006
    head_lr: float = 0.012
    ngram_lr: float = 0.025
    stiefel_penalty: float = 0.04
    retract_every: int = 1


@dataclass(frozen=True)
class QuantumExperimentSpec:
    """Complete declarative specification for a QuantumBornGPT training run."""

    key: str
    label: str
    short_label: str
    steps: int = 900
    color: str = "#00f5d4"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    hierarchy: QubitHierarchyConfig = field(default_factory=QubitHierarchyConfig)
    readout: KBosonReadoutConfig = field(default_factory=KBosonReadoutConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@torch.no_grad()
def initialize_povm_log_prior(
    model: nn.Module,
    token_ids: torch.Tensor,
    alpha: float = 0.62,
    bose_saturate_top_k: int = 16,
) -> None:
    """Initialize `model.povm_log_prior` with Bose-saturated empirical log-unigram frequencies."""
    prior_param = getattr(model, "povm_log_prior", None)
    if prior_param is None or not isinstance(prior_param, nn.Parameter):
        return
    v_size = int(prior_param.numel())
    flat = token_ids.reshape(-1).to(torch.int64).clamp(0, v_size - 1)
    counts = torch.bincount(flat, minlength=v_size).float() + 1.0
    probs = counts / counts.sum()
    log_p = torch.log(probs)
    if bose_saturate_top_k > 0 and bool(getattr(model, "use_two_beam_kboson_readout", False)):
        k_top = max(4, min(int(bose_saturate_top_k), v_size // 16))
        knee = torch.topk(log_p, k=k_top, largest=True).values[-1]
        log_p = knee - 1.25 * F.softplus((knee - log_p) / 1.25) + 0.25 * (log_p - knee).clamp_min(0.0)
    log_p = (log_p - log_p.mean()) * float(alpha)
    prior_param.copy_(log_p.to(device=prior_param.device, dtype=prior_param.dtype))


def build_quantum_model_from_spec(
    spec: QuantumExperimentSpec,
    ppmi_embeddings: torch.Tensor,
    init_tokens: np.ndarray,
    prior_tokens: torch.Tensor,
    device: torch.device,
) -> tuple[QuantumBornGPT, dict[str, float]]:
    """Instantiate and Step-0 initialize a `QuantumBornGPT` model from a typed `QuantumExperimentSpec`."""
    b, h, r = spec.backbone, spec.hierarchy, spec.readout
    model = QuantumBornGPT(
        vocab_size=50304, embed_dim=b.embed_dim, seed_dim=b.seed_dim, num_heads=b.num_heads,
        num_layers=b.num_layers, block_size=b.block_size, readout_mode="asymmetric",
        povm_temperature=r.povm_temperature, use_overcomplete_two_beam=True,
        frame_expansion=b.frame_expansion, use_purified_attn=True, true_layer0_v_residual=True,
        use_gated_spherical_residual=True, use_dynamic_povm_beta=True, use_povm_log_prior=True,
        use_ortho_decouple=h.use_ortho_decouple, use_hierarchical_16qubit=True,
        qubit_registers=h.qubit_registers, qubit_amplitudes=h.qubit_amplitudes,
        num_higher_states=h.num_higher_states, use_block_hierarchy_bridge=True,
        bridge_causal_memory=False, use_last_layer_softmax=False, logit_softcap=15.0,
        use_two_beam_kboson_readout=True, diffuse_rank=r.diffuse_rank,
        diffuse_schedule=r.diffuse_schedule, lambda_diff_min=r.lambda_diff_min,
        lambda_diff_max=r.lambda_diff_max, lambda_diff_start=r.lambda_diff_start,
        lambda_diff_end=r.lambda_diff_end, contextual_purity_gain=r.contextual_purity_gain,
        use_stereographic_lift=r.use_stereographic_lift,
        use_cosine_phase_beam_gate=r.use_cosine_phase_beam_gate,
        quadratic_diffuse_fidelity=r.quadratic_diffuse_fidelity,
        kraus_spectral_floor=r.kraus_spectral_floor,
        use_mach_zehnder_beamsplitter=b.use_mach_zehnder_beamsplitter,
        use_wavelet_cholesky_branch=b.use_wavelet_cholesky_branch,
        use_wavelet_cholesky_basis=h.use_wavelet_cholesky_basis,
        use_bipolar_kboson=False, bipolar_hole_weight=0.0, use_unet_interferometry=False,
        unet_skip_mode="scalar", num_value_embeddings=b.num_value_embeddings,
        zero_half_rope=True, window_size=b.window_size, use_ngram_hash=b.ngram_vocab_size > 0,
        ngram_vocab_size=max(1024, b.ngram_vocab_size), ngram_inject_scale=b.ngram_inject_scale,
        use_mudd_and_gates=True, use_smear_gate=True, use_depth_scaled_residual=True,
    ).to(device)

    initialize_povm_log_prior(model, prior_tokens, alpha=0.62, bose_saturate_top_k=16)
    ppmi_stats = initialize_hierarchical_16qubit_with_ppmi_svd(
        model, ppmi_embeddings=ppmi_embeddings, token_shards=[init_tokens], alpha=h.ppmi_alpha,
        kmeans_iters=8, seed=42, device=device,
        decouple_self_in_full_ambient=h.decouple_self_in_full_ambient,
        directed_predecessor_mix=h.directed_predecessor_mix,
    )
    with torch.no_grad():
        if model.hierarchical_povm_readout is not None:
            model.hierarchical_povm_readout.kboson_copies.fill_(38.0)
    return model, ppmi_stats


def build_quantum_born_optimizers(
    model: nn.Module,
    opt_cfg: OptimizerConfig | None = None,
) -> tuple[StiefelMuon | None, torch.optim.AdamW, float, float]:
    """Build StiefelMuon + Multi-Group AdamW with Gentle 2x Zero-WD Qubit/Value-Embedding routing."""
    cfg = opt_cfg or OptimizerConfig()
    muon_params, adam_params = split_muon_adamw_params(model)
    filtered_muon: list[nn.Parameter] = []
    for p in muon_params:
        if p.ndim == 3 or (p.ndim == 2 and min(p.shape) >= 32 and max(p.shape) < 40000):
            filtered_muon.append(p)
        else:
            adam_params.append(p)

    muon_opt = (
        StiefelMuon(
            filtered_muon, lr=cfg.muon_lr, momentum=0.95, nesterov=True, weight_decay=0.01,
            decay_mode="cautious", stiefel_penalty=cfg.stiefel_penalty,
            retract_every=cfg.retract_every, polar_method="polar_express",
            use_normuon=True, use_twin_rail=True,
        )
        if filtered_muon
        else None
    )

    gain_ids: set[int] = set()
    embed_ids: set[int] = set()
    ngram_ids: set[int] = set()
    hr = getattr(model, "hierarchical_povm_readout", None)
    if hr is not None:
        pr_mod = getattr(model, "povm_readout", None)
        base_h = getattr(pr_mod, "povm_head", None) if pr_mod is not None else None
        for gp in (
            hr.token_radial_scale, hr.readout_rms_gain, hr.qubit_readout_gain,
            hr.hier_readout_gain, hr.povm_odd_qubits, hr.hier_even_proj,
            hr.vocab_higher_affinity, hr.kraus_u_diff, hr.kraus_v_diff,
            hr.kboson_copies, hr.diffuse_mix_logit,
            getattr(hr, "wavelet_packets_256", None),
            getattr(hr, "wavelet_packets_1024", None),
            getattr(hr, "wavelet_packets_4096", None),
            getattr(hr, "basis_chol_tril", None),
            getattr(hr, "basis_chol_diag", None),
            getattr(hr, "wavelet_scale_gains", None),
            getattr(hr, "readout_stereo_pole", None),
            getattr(hr, "readout_stereo_tau", None),
            getattr(hr, "readout_stereo_gain", None),
            base_h, getattr(model, "povm_log_prior", None),
        ):
            if gp is not None and gp.requires_grad:
                gain_ids.add(id(gp))

    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in gain_ids:
            continue
        if "ngram_embed" in n:
            ngram_ids.add(id(p))
        elif any(k in n for k in ("unigram_qubits", "higher_states", "value_embeds")):
            embed_ids.add(id(p))
        elif any(k in n for k in ("stereo", "phase", "smear_gate", "lm_head", "beamsplitter")):
            gain_ids.add(id(p))

    body_group = [
        p
        for p in adam_params
        if id(p) not in gain_ids and id(p) not in embed_ids and id(p) not in ngram_ids
    ]
    embed_group = [p for p in adam_params if id(p) in embed_ids]
    gain_group = [p for p in adam_params if id(p) in gain_ids]
    ngram_group = [p for p in adam_params if id(p) in ngram_ids]

    param_groups: list[dict[str, Any]] = []
    if body_group:
        param_groups.append(
            {
                "params": body_group,
                "lr": cfg.body_adam_lr,
                "base_lr": cfg.body_adam_lr,
                "weight_decay": 0.01,
                "group_name": "body",
            }
        )
    if embed_group:
        param_groups.append(
            {
                "params": embed_group,
                "lr": cfg.qubit_embed_lr,
                "base_lr": cfg.qubit_embed_lr,
                "weight_decay": 0.0,
                "group_name": "qubit_embed",
            }
        )
    if gain_group:
        param_groups.append(
            {
                "params": gain_group,
                "lr": cfg.head_lr,
                "base_lr": cfg.head_lr,
                "weight_decay": 0.0,
                "group_name": "readout_head",
            }
        )
    if ngram_group:
        param_groups.append(
            {
                "params": ngram_group,
                "lr": cfg.ngram_lr,
                "base_lr": cfg.ngram_lr,
                "weight_decay": 0.0,
                "group_name": "ngram",
            }
        )

    adam_opt = torch.optim.AdamW(param_groups, lr=cfg.body_adam_lr, betas=(0.9, 0.95), weight_decay=0.0)
    return muon_opt, adam_opt, cfg.muon_lr, cfg.body_adam_lr


def compute_wsd_lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int = 20,
    stable_frac: float = 0.60,
    min_ratio: float = 0.10,
) -> float:
    """Compute Warmup-Stable-Decay (WSD) learning-rate multiplier in `[min_ratio, 1.0]`."""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    stable_end = int(total_steps * stable_frac)
    if step <= stable_end:
        return 1.0
    decay_prog = float(step - stable_end) / float(max(1, total_steps - stable_end))
    return max(min_ratio, 1.0 - (1.0 - min_ratio) * decay_prog)


def build_classical_modded_l4_baseline(
    seq_len: int, ppmi_emb_256: torch.Tensor, device: torch.device
) -> tuple[nn.Module, StiefelMuon | None, torch.optim.AdamW, float, float]:
    """Build matched leaderboard-exact `ClassicalModdedGPT(L=4, d=256, H=4)` baseline and optimizers."""
    from qdlblocks.layers.cooccurrence_init import initialize_classical_with_ppmi_svd
    from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT

    model = ClassicalModdedGPT(
        vocab_size=50304, embed_dim=256, num_heads=4, num_layers=4, block_size=seq_len,
        hidden_dim=512, logit_softcap=15.0, tie_embeddings=False,
        num_value_embeddings=2, zero_half_rope=True, window_size=384, use_smear_gate=True, leaderboard_exact=True,
    ).to(device)
    initialize_classical_with_ppmi_svd(model, ppmi_embeddings=ppmi_emb_256, alpha=0.55, seed=42)
    with torch.no_grad():
        model.lm_head.weight.copy_((0.01 * ppmi_emb_256.to(device)).to(model.lm_head.weight.dtype))
    muon_params, adam_params = split_muon_adamw_params(model)
    filt_muon = [p for p in muon_params if p.ndim == 2 and min(p.shape) >= 32 and max(p.shape) < 40000]
    rem_adam = [p for p in muon_params if p not in set(filt_muon)] + list(adam_params)
    base_muon_lr, base_adam_lr = 0.008, 0.0012
    muon_opt = StiefelMuon(
        filt_muon, lr=base_muon_lr, momentum=0.95, nesterov=True, weight_decay=0.01,
        decay_mode="cautious", stiefel_penalty=0.02, retract_every=2, polar_method="polar_express",
        use_normuon=True, use_twin_rail=True,
    )
    grp = {"params": rem_adam, "lr": base_adam_lr, "base_lr": base_adam_lr, "weight_decay": 0.01, "group_name": "body"}
    adam_opt = torch.optim.AdamW([grp], lr=base_adam_lr, betas=(0.9, 0.95))
    return model, muon_opt, adam_opt, base_muon_lr, base_adam_lr


__all__ = [
    "BackboneConfig",
    "QubitHierarchyConfig",
    "KBosonReadoutConfig",
    "OptimizerConfig",
    "QuantumExperimentSpec",
    "initialize_povm_log_prior",
    "build_quantum_model_from_spec",
    "build_quantum_born_optimizers",
    "build_classical_modded_l4_baseline",
    "compute_wsd_lr_multiplier",
]
