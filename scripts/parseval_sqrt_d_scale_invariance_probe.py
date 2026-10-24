#!/usr/bin/env python3
"""Analytical Parseval sqrt(d) Scale Invariance & Algebraic Unification Probe (Plan 043 / Wave 28).

Implements the Build-Before-Launch diagnostic tool for Hypothesis H1 of EXPERIMENT_PLAN_043.md:
1. Part 1 (Unit-Sphere Welch/Parseval Variance & Logit Scale Invariance Across Widths):
   Constructs unit-normalized Parseval detector frames M in R^{V x d} (V = 50,257) and query states
   psi in S^{d-1} with controlled target cosine alignment rho_target in {0.35, 0.50, 0.65} across
   d in {128, 256, 384, 512, 768, 1024} and audits `unscaled_beta_1`, `fixed_beta_24`, and
   `analytical_parseval_sqrt_d` (beta_0(d) = c * sqrt(d), c = 1.48).
2. Part 2 (Algebraic Identity & Parameter/Clamp Audit of Removing `kboson_k` + `PurityGate`):
   Verifies numerically in float64/float32 via `compute_two_beam_kboson_logits` and
   `direct_chunked_two_beam_kboson_ce` that legacy `k_copies * ((beta_t / k_copies) * O_hat + lambda_diff * f_diff)`
   is algebraically identical (`max_err < 1e-6`) to unified `beta_0(d) * (O_hat + lambda_tilde_diff * f_diff)`
   with `k_copies = 1.0` and `lambda_tilde_diff = (k_copies / beta_0(d)) * lambda_diff`, and tabulates
   exact parameter savings from removing `ContextualBlochPurityGate` + `kboson_raw_k`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.layers.direct_chunked_ce import (  # noqa: E402
    compute_two_beam_kboson_logits,
    direct_chunked_two_beam_kboson_ce,
)
from qdlblocks.layers.dynamic_povm_purity import ContextualBlochPurityGate  # noqa: E402

SCALING_LAWS: tuple[str, ...] = ("unscaled_beta_1", "fixed_beta_24", "analytical_parseval_sqrt_d")


@dataclass(frozen=True)
class ParsevalProbeConfig:
    """Configuration for the Parseval sqrt(d) scale invariance & algebraic unification probe."""

    dims: tuple[int, ...] = (128, 256, 384, 512, 768, 1024)
    audit_dims: tuple[int, ...] = (256, 384, 512)
    vocab_size: int = 50257
    parseval_c: float = 1.48
    fixed_beta: float = 24.0
    rho_targets: tuple[float, ...] = (0.35, 0.50, 0.65)
    num_queries: int = 64
    traceless_gamma: float = 0.55
    quad_diffuse_fidelity: float = 0.50
    legacy_k_copies: float = 34.0
    legacy_lambda_diff: float = 0.10656
    seed: int = 2043
    device: str = "cpu"
    json_out: str = ".cache/parseval_sqrt_d_scale_invariance_probe.json"


@dataclass(frozen=True)
class FrameQueryBundle:
    """Precomputed Parseval frame cosine components for a single embedding width `d`."""

    embed_dim: int
    vocab_size: int
    labels: torch.Tensor
    cos_target: torch.Tensor
    cos_orth: torch.Tensor
    frame_gram_rel_err: float


def compute_cv(values: list[float]) -> float:
    """Compute population coefficient of variation `std / mean` across positive floats."""
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.std(unbiased=False).item()) / max(abs(float(t.mean().item())), 1e-12)


@torch.no_grad()
def build_parseval_frame_bundle(embed_dim: int, cfg: ParsevalProbeConfig, device: torch.device) -> FrameQueryBundle:
    """Construct a unit-sphere Parseval frame `M in R^{V x d}` and orthogonal query components."""
    v_sz, d, n_q = int(cfg.vocab_size), int(embed_dim), int(cfg.num_queries)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + d * 17)

    raw = torch.randn((v_sz, d), generator=gen, dtype=torch.float32).to(device)
    raw = raw - raw.mean(dim=0, keepdim=True)
    evals, evecs = torch.linalg.eigh(raw.t() @ raw)
    inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(1e-8)) * math.sqrt(v_sz / float(d))) @ evecs.t()
    m_frame = F.normalize(raw @ inv_sqrt, p=2.0, dim=-1)

    gram_post = (float(d) / float(v_sz)) * (m_frame.t() @ m_frame)
    eye_d = torch.eye(d, device=device, dtype=torch.float32)
    gram_err = float(torch.linalg.matrix_norm(gram_post - eye_d, ord="fro").item() / math.sqrt(float(d)))

    q_idx = torch.arange(n_q, device=device, dtype=torch.long)
    labels = (q_idx * 787 + 19) % v_sz
    m_y = m_frame.index_select(0, labels)
    noise_raw = torch.randn((n_q, d), generator=gen, dtype=torch.float32).to(device)
    u_orth = F.normalize(noise_raw - (noise_raw * m_y).sum(dim=-1, keepdim=True) * m_y, p=2.0, dim=-1)
    return FrameQueryBundle(
        embed_dim=d, vocab_size=v_sz, labels=labels,
        cos_target=m_y @ m_frame.t(), cos_orth=u_orth @ m_frame.t(), frame_gram_rel_err=gram_err,
    )


@torch.no_grad()
def compute_convex_traceless_observable(cos_scores: torch.Tensor, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(c_hat, o_hat)` where `o_hat = c_hat + gamma * (relu(c_hat)^2 - mean(relu(c_hat)^2))`."""
    c_hat = cos_scores - cos_scores.mean(dim=-1, keepdim=True)
    pos_sq = F.relu(c_hat).square()
    return c_hat, c_hat + float(gamma) * (pos_sq - pos_sq.mean(dim=-1, keepdim=True))


def resolve_scaling_beta(law: str, embed_dim: int, cfg: ParsevalProbeConfig) -> float:
    """Resolve inverse temperature `beta(d)` for the specified scaling law."""
    if law == "unscaled_beta_1":
        return 1.0
    if law == "fixed_beta_24":
        return float(cfg.fixed_beta)
    if law == "analytical_parseval_sqrt_d":
        return float(cfg.parseval_c) * math.sqrt(float(embed_dim))
    raise ValueError(f"Unknown scaling law: {law}")


@torch.no_grad()
def evaluate_law_at_width(law: str, bundle: FrameQueryBundle, cfg: ParsevalProbeConfig) -> dict[str, Any]:
    """Evaluate background logit std, target peak margin, target probability, and CE at width `d`."""
    d, v_sz, n_q = bundle.embed_dim, bundle.vocab_size, bundle.labels.numel()
    beta = resolve_scaling_beta(law, d, cfg)
    row_idx = torch.arange(n_q, device=bundle.labels.device)
    non_target_mask = torch.ones((n_q, v_sz), dtype=torch.bool, device=bundle.labels.device)
    non_target_mask[row_idx, bundle.labels] = False

    rho_results: dict[str, dict[str, float]] = {}
    bg_cos_stds: list[float] = []
    bg_obs_stds: list[float] = []

    for rho in (*cfg.rho_targets, 1.0):
        orth_scale = math.sqrt(max(0.0, 1.0 - float(rho) * float(rho)))
        cos_scores = float(rho) * bundle.cos_target + orth_scale * bundle.cos_orth
        c_hat, o_hat = compute_convex_traceless_observable(cos_scores, cfg.traceless_gamma)
        cos_logits, obs_logits = beta * c_hat, beta * o_hat

        bg_cos = cos_logits.masked_select(non_target_mask).view(n_q, v_sz - 1)
        bg_obs = obs_logits.masked_select(non_target_mask).view(n_q, v_sz - 1)
        bg_cos_std = float(bg_cos.std(dim=-1, unbiased=False).mean().item())
        bg_obs_std = float(bg_obs.std(dim=-1, unbiased=False).mean().item())
        if rho < 0.999:
            bg_cos_stds.append(bg_cos_std)
            bg_obs_stds.append(bg_obs_std)

        target_obs_logit = obs_logits[row_idx, bundle.labels]
        peak_margin = float((target_obs_logit - bg_obs.mean(dim=-1)).mean().item())
        runnerup_margin = float((target_obs_logit - bg_obs.max(dim=-1).values).mean().item())
        obs_ce_vec = F.cross_entropy(obs_logits.double(), bundle.labels, reduction="none")
        obs_ce = float(obs_ce_vec.mean().item())
        cos_ce = float(F.cross_entropy(cos_logits.double(), bundle.labels).item())
        target_prob = float(torch.exp(-obs_ce_vec).mean().item())

        rho_results[f"rho_{rho:.2f}"] = {
            "rho_target": float(rho), "bg_cos_logit_std": bg_cos_std, "bg_obs_logit_std": bg_obs_std,
            "target_peak_margin": peak_margin, "target_runnerup_margin": runnerup_margin,
            "peak_snr": peak_margin / max(bg_obs_std, 1e-12),
            "mean_ce_nats": obs_ce, "mean_cos_ce_nats": cos_ce, "mean_target_prob": target_prob,
        }

    welch_sigma = math.sqrt(float(v_sz - d) / float(d * (v_sz - 1)))
    return {
        "embed_dim": d, "beta": beta, "welch_cos_std_analytical": welch_sigma,
        "analytical_bg_cos_logit_std": beta * welch_sigma,
        "bg_cos_logit_std": sum(bg_cos_stds) / len(bg_cos_stds),
        "bg_obs_logit_std": sum(bg_obs_stds) / len(bg_obs_stds),
        "frame_gram_rel_err": bundle.frame_gram_rel_err, "by_rho": rho_results,
    }


def evaluate_algebraic_unification_for_dim(
    embed_dim: int, cfg: ParsevalProbeConfig, dtype: torch.dtype,
) -> dict[str, float]:
    """Verify legacy `k * ((beta / k) * O_hat + lam * f_diff)` == unified `beta_0 * (O_hat + lam_tilde * f_diff)`."""
    d, v_sz, n_tok = int(embed_dim), 1024, 32
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + d * 31 + (64 if dtype == torch.float64 else 32))

    psi_s_0 = F.normalize(torch.randn((n_tok, d), generator=gen, dtype=dtype), p=2.0, dim=-1)
    psi_d_0 = F.normalize(torch.randn((n_tok, d), generator=gen, dtype=dtype), p=2.0, dim=-1)
    m_s = F.normalize(torch.randn((v_sz, d), generator=gen, dtype=dtype), p=2.0, dim=-1)
    m_d = F.normalize(torch.randn((v_sz, d), generator=gen, dtype=dtype), p=2.0, dim=-1)
    log_prior = 0.25 * torch.randn((v_sz,), generator=gen, dtype=dtype)
    targets = torch.arange(n_tok, dtype=torch.long) % v_sz

    beta_0 = float(cfg.parseval_c) * math.sqrt(float(d))
    k_leg, lam_leg = float(cfg.legacy_k_copies), float(cfg.legacy_lambda_diff)
    lam_tilde = (k_leg / beta_0) * lam_leg
    kw = dict(m_sharp=m_s, m_diff=m_d, quadratic_diffuse_fidelity=cfg.quad_diffuse_fidelity, traceless_gamma=cfg.traceless_gamma, convex_traceless=True)

    logits_legacy = compute_two_beam_kboson_logits(psi_sharp=psi_s_0, psi_diff=psi_d_0, beta_t=beta_0, k_copies=k_leg, lambda_diff=lam_leg, log_prior=log_prior, **kw)
    unit_logits = compute_two_beam_kboson_logits(psi_sharp=psi_s_0, psi_diff=psi_d_0, beta_t=1.0, k_copies=1.0, lambda_diff=lam_tilde, log_prior=None, **kw)
    logit_max_err = float((logits_legacy - (beta_0 * unit_logits + log_prior.unsqueeze(0))).abs().max().item())

    psi_s_leg, psi_d_leg = psi_s_0.clone().detach().requires_grad_(True), psi_d_0.clone().detach().requires_grad_(True)
    ce_legacy = direct_chunked_two_beam_kboson_ce(psi_sharp=psi_s_leg, psi_diff=psi_d_leg, beta_t=beta_0, k_copies=k_leg, lambda_diff=lam_leg, targets=targets, log_prior=log_prior, **kw)
    ce_legacy.backward()

    psi_s_uni, psi_d_uni = psi_s_0.clone().detach().requires_grad_(True), psi_d_0.clone().detach().requires_grad_(True)
    ce_unified = direct_chunked_two_beam_kboson_ce(psi_sharp=psi_s_uni, psi_diff=psi_d_uni, beta_t=beta_0, k_copies=beta_0, lambda_diff=lam_tilde, targets=targets, log_prior=log_prior, **kw)
    ce_unified.backward()

    ce_err = float((ce_legacy.detach() - ce_unified.detach()).abs().item())
    assert psi_s_leg.grad is not None and psi_s_uni.grad is not None and psi_d_leg.grad is not None and psi_d_uni.grad is not None
    grad_s_err = float((psi_s_leg.grad - psi_s_uni.grad).abs().max().item())
    grad_d_err = float((psi_d_leg.grad - psi_d_uni.grad).abs().max().item())

    return {
        "beta_0": beta_0, "legacy_k_copies": k_leg, "legacy_lambda_diff": lam_leg,
        "unified_lambda_tilde_diff": lam_tilde, "logit_max_abs_err": logit_max_err,
        "ce_abs_err": ce_err, "grad_psi_sharp_max_abs_err": grad_s_err,
        "grad_psi_diff_max_abs_err": grad_d_err,
        "max_err": max(logit_max_err, ce_err, grad_s_err, grad_d_err),
    }


def run_algebraic_and_param_audit(cfg: ParsevalProbeConfig) -> dict[str, Any]:
    """Run Part 2 algebraic identity verification and parameter/clamp savings audit."""
    by_dim: dict[str, Any] = {}
    max_err_fp64, max_err_fp32 = 0.0, 0.0
    for d in cfg.audit_dims:
        res_fp64 = evaluate_algebraic_unification_for_dim(d, cfg, torch.float64)
        res_fp32 = evaluate_algebraic_unification_for_dim(d, cfg, torch.float32)
        max_err_fp64 = max(max_err_fp64, res_fp64["max_err"])
        max_err_fp32 = max(max_err_fp32, res_fp32["max_err"])
        gate = ContextualBlochPurityGate(embed_dim=int(d), max_beta=64.0)
        purity_params = sum(p.numel() for p in gate.parameters())
        by_dim[f"d_{d}"] = {
            "embed_dim": int(d), "beta_0": float(cfg.parseval_c) * math.sqrt(float(d)),
            "purity_gate_params": int(purity_params), "kboson_raw_k_params": 1,
            "total_params_saved": int(purity_params + 1),
            "legacy_manual_beta_clamp": 44.0 if d > 256 else 26.8, "parseval_manual_beta_clamps": 0,
            "float64": res_fp64, "float32": res_fp32,
        }
    return {
        "by_dim": by_dim,
        "algebraic_unification_max_err_fp64": max_err_fp64,
        "algebraic_unification_max_err_fp32": max_err_fp32,
        "algebraic_unification_max_err": max(max_err_fp64, max_err_fp32),
        "min_params_saved_across_audit_dims": min(v["total_params_saved"] for v in by_dim.values()),
        "max_params_saved_across_audit_dims": max(v["total_params_saved"] for v in by_dim.values()),
    }


def run_probe(cfg: ParsevalProbeConfig) -> dict[str, Any]:
    """Execute Part 1 (Parseval scale invariance) and Part 2 (algebraic unification & param audit)."""
    device = torch.device(cfg.device)
    bundles = [build_parseval_frame_bundle(d, cfg, device) for d in cfg.dims]
    laws_report: dict[str, Any] = {}
    for law in SCALING_LAWS:
        per_dim = [evaluate_law_at_width(law, b, cfg) for b in bundles]
        bg_cos_stds = [row["bg_cos_logit_std"] for row in per_dim]
        bg_obs_stds = [row["bg_obs_logit_std"] for row in per_dim]
        mono_margin, mono_prob, mono_ce_dec = True, True, True
        for rho in cfg.rho_targets:
            rk = f"rho_{rho:.2f}"
            margins = [row["by_rho"][rk]["target_peak_margin"] for row in per_dim]
            probs = [row["by_rho"][rk]["mean_target_prob"] for row in per_dim]
            ces = [row["by_rho"][rk]["mean_ce_nats"] for row in per_dim]
            mono_margin = mono_margin and all(margins[i] < margins[i + 1] for i in range(len(margins) - 1))
            mono_prob = mono_prob and all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))
            mono_ce_dec = mono_ce_dec and all(ces[i] > ces[i + 1] for i in range(len(ces) - 1))
        laws_report[law] = {
            "per_dim": per_dim,
            "bg_cos_logit_std_cv": compute_cv(bg_cos_stds),
            "bg_obs_logit_std_cv": compute_cv(bg_obs_stds),
            "bg_cos_logit_std_max_min_ratio": max(bg_cos_stds) / max(min(bg_cos_stds), 1e-12),
            "target_margin_monotonic_increasing": mono_margin,
            "target_prob_monotonic_increasing": mono_prob,
            "ce_monotonic_decreasing": mono_ce_dec,
        }

    min_unscaled_cos_ce_rho1 = min(
        row["by_rho"]["rho_1.00"]["mean_cos_ce_nats"] for row in laws_report["unscaled_beta_1"]["per_dim"]
    )
    part2 = run_algebraic_and_param_audit(cfg)
    bg_cv_parseval = float(laws_report["analytical_parseval_sqrt_d"]["bg_cos_logit_std_cv"])
    bg_cv_fixed = float(laws_report["fixed_beta_24"]["bg_cos_logit_std_cv"])
    fixed_drift_ratio = float(laws_report["fixed_beta_24"]["bg_cos_logit_std_max_min_ratio"])
    alg_max_err = float(part2["algebraic_unification_max_err"])
    parseval_mono = bool(
        laws_report["analytical_parseval_sqrt_d"]["target_margin_monotonic_increasing"]
        and laws_report["analytical_parseval_sqrt_d"]["target_prob_monotonic_increasing"]
        and laws_report["analytical_parseval_sqrt_d"]["ce_monotonic_decreasing"]
    )
    gates = {
        "bg_logit_std_cv_parseval_lt_0p015": bg_cv_parseval < 0.015,
        "bg_logit_std_cv_fixed_gt_0p30": bg_cv_fixed > 0.30,
        "unscaled_beta_1_min_ce_gt_9p4_nats": min_unscaled_cos_ce_rho1 > 9.4,
        "parseval_monotonic_improvement_with_sqrt_d": parseval_mono,
        "algebraic_unification_max_err_lt_1e6": alg_max_err < 1e-6,
    }
    return {
        "plan": "EXPERIMENT_PLAN_043", "wave": "Wave 28", "hypothesis": "H1",
        "config": asdict(cfg),
        "summary_metrics": {
            "bg_logit_std_cv_parseval": bg_cv_parseval,
            "bg_obs_logit_std_cv_parseval": float(laws_report["analytical_parseval_sqrt_d"]["bg_obs_logit_std_cv"]),
            "bg_logit_std_cv_fixed": bg_cv_fixed,
            "bg_logit_std_cv_unscaled": float(laws_report["unscaled_beta_1"]["bg_cos_logit_std_cv"]),
            "fixed_beta_24_bg_std_max_min_ratio": fixed_drift_ratio,
            "unscaled_beta_1_min_cos_ce_rho1_nats": min_unscaled_cos_ce_rho1,
            "algebraic_unification_max_err": alg_max_err,
            "algebraic_unification_max_err_fp64": float(part2["algebraic_unification_max_err_fp64"]),
            "algebraic_unification_max_err_fp32": float(part2["algebraic_unification_max_err_fp32"]),
            "params_saved_d256": int(part2["by_dim"]["d_256"]["total_params_saved"]),
            "params_saved_d384": int(part2["by_dim"]["d_384"]["total_params_saved"]),
            "params_saved_d512": int(part2["by_dim"]["d_512"]["total_params_saved"]),
        },
        "part1_scaling_laws": laws_report,
        "part2_algebraic_and_param_audit": part2,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    """Format a concise Markdown table summarizing Part 1 and Part 2 findings."""
    sm = report["summary_metrics"]
    lines = [
        "# Analytical Parseval `sqrt(d)` Scale Invariance & Algebraic Unification Probe (`Wave 28 / H1`)", "",
        "## 1. Background Logit Scale Invariance & Target SNR (`rho_target = 0.50`, `V = 50,257`)", "",
        "| Scaling Law | `d=128` (`bg_std` / `CE`) | `d=256` (`bg_std` / `CE`) | `d=384` (`bg_std` / `CE`) | `d=512` (`bg_std` / `CE`) | `d=768` (`bg_std` / `CE`) | `d=1024` (`bg_std` / `CE`) | `bg_std CV` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for law in SCALING_LAWS:
        law_data = report["part1_scaling_laws"][law]
        cells = [
            f"`{row['bg_cos_logit_std']:.4f}` / `{row['by_rho']['rho_0.50']['mean_ce_nats']:.2f}`"
            for row in law_data["per_dim"]
        ]
        lines.append(f"| `{law}` | " + " | ".join(cells) + f" | `{law_data['bg_cos_logit_std_cv']:.4f}` |")

    lines.extend([
        "", "## 2. Algebraic Unification (`k_copies = 1.0`) & Parameter Savings Audit", "",
        "| Width `d` | `beta_0(d) = c * sqrt(d)` | `PurityGate` Params | `kboson_raw_k` | Total Saved | Legacy Manual Clamp | `fp64` Max Err | `fp32` Max Err |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for d_info in report["part2_algebraic_and_param_audit"]["by_dim"].values():
        lines.append(
            f"| `{d_info['embed_dim']}` | `{d_info['beta_0']:.2f}` | `{d_info['purity_gate_params']:,}` | "
            f"`{d_info['kboson_raw_k_params']}` | `{d_info['total_params_saved']:,}` | "
            f"`<= {d_info['legacy_manual_beta_clamp']:.1f}` -> `None` | "
            f"`{d_info['float64']['max_err']:.2e}` | `{d_info['float32']['max_err']:.2e}` |"
        )
    lines.extend([
        "", "## 3. Falsifiable Acceptance Gates (`H1`)",
        f"- `bg_logit_std_cv_parseval = {sm['bg_logit_std_cv_parseval']:.5f} < 0.015`: `{report['gates']['bg_logit_std_cv_parseval_lt_0p015']}`",
        f"- `bg_logit_std_cv_fixed = {sm['bg_logit_std_cv_fixed']:.5f} > 0.30` (`{sm['fixed_beta_24_bg_std_max_min_ratio']:.2f}x` drift): `{report['gates']['bg_logit_std_cv_fixed_gt_0p30']}`",
        f"- `unscaled_beta_1_min_cos_ce_rho1_nats = {sm['unscaled_beta_1_min_cos_ce_rho1_nats']:.4f} > 9.4`: `{report['gates']['unscaled_beta_1_min_ce_gt_9p4_nats']}`",
        f"- `algebraic_unification_max_err = {sm['algebraic_unification_max_err']:.2e} < 1e-6`: `{report['gates']['algebraic_unification_max_err_lt_1e6']}`",
        f"- `overall_pass`: `{report['overall_pass']}`",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> ParsevalProbeConfig:
    """Parse CLI arguments into a frozen `ParsevalProbeConfig`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=str, default="128,256,384,512,768,1024")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--parseval-c", type=float, default=1.48)
    parser.add_argument("--fixed-beta", type=float, default=24.0)
    parser.add_argument("--rho-targets", type=str, default="0.35,0.50,0.65")
    parser.add_argument("--num-queries", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=".cache/parseval_sqrt_d_scale_invariance_probe.json")
    args = parser.parse_args(argv)
    dims = tuple(int(x.strip()) for x in args.dims.split(",") if x.strip())
    rhos = tuple(float(x.strip()) for x in args.rho_targets.split(",") if x.strip())
    return ParsevalProbeConfig(
        dims=dims, vocab_size=int(args.vocab_size), parseval_c=float(args.parseval_c),
        fixed_beta=float(args.fixed_beta), rho_targets=rhos, num_queries=int(args.num_queries),
        device=str(args.device), json_out=str(args.json_out),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    cfg = parse_args(argv)
    report = run_probe(cfg)
    out_path = Path(cfg.json_out) if Path(cfg.json_out).is_absolute() else (ROOT / cfg.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_markdown_report(report))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
