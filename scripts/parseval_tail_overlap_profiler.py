#!/usr/bin/env python3
"""Parseval Frame Conservation, Incomplete Block-QR Packing, & Traceless Readout Profiler (Plan 038).

Promoted from `scripts/BACKLOG.md` (identified in Plan 037, implemented in Plan 038).
Profiles V = 50,304 unit vectors across dimensions d in {16, 32, 64, 128, 256, 384, 512, 1024, 2048}
under four geometric packing regimes: `zipf_cone`, `random_sphere`, `incomplete_block_qr`, and `welch_frame_landing`.
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

DEFAULT_DIMS: tuple[int, ...] = (16, 32, 64, 128, 256, 384, 512, 1024, 2048)
PACKING_REGIMES: tuple[str, ...] = ("zipf_cone", "random_sphere", "incomplete_block_qr", "welch_frame_landing")


@dataclass(frozen=True)
class ParsevalProfilerConfig:
    """Configuration for Parseval / Incomplete Block-QR vector packing and CE floor profiling."""

    vocab_size: int = 50304
    dims: tuple[int, ...] = DEFAULT_DIMS
    probe_queries: int = 256
    num_landing_steps: int = 6
    landing_lr: float = 0.28
    beta: float = 32.0
    k_copies: float = 34.0
    traceless_gamma: float = 0.55
    alpha_target_328: float = 0.55
    alpha_target_250: float = 0.68
    seed: int = 2038
    device: str = "cpu"
    out_json: str = ""


@torch.no_grad()
def _orthonormalize_row_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """Orthonormalize `(n_blocks, b, d)` row blocks (`b <= d`) via QR / Cholesky-QR."""
    _, b_sz, d = blocks.shape
    if d <= 256:
        q_t, r_mat = torch.linalg.qr(blocks.transpose(-1, -2))
        r_diag = torch.diagonal(r_mat, dim1=-2, dim2=-1)
        sign_d = torch.where(r_diag >= 0.0, torch.ones_like(r_diag), -torch.ones_like(r_diag))
        return F.normalize((q_t * sign_d.unsqueeze(1)).transpose(-1, -2), p=2.0, dim=-1)
    gram = torch.bmm(blocks, blocks.transpose(-1, -2))
    eye_b = torch.eye(b_sz, device=blocks.device, dtype=blocks.dtype).unsqueeze(0)
    chol = torch.linalg.cholesky(gram + 1e-6 * eye_b)
    return F.normalize(torch.linalg.solve_triangular(chol, blocks, upper=False), p=2.0, dim=-1)


@torch.no_grad()
def build_regime_vectors(
    z_base: torch.Tensor, regime: str, cfg: ParsevalProfilerConfig,
) -> tuple[torch.Tensor, int]:
    """Construct `(V, d)` unit vectors on `S^{d-1}` for the requested packing regime."""
    v_size, d = z_base.shape
    block_sz = min(d, 128)
    if regime == "zipf_cone":
        env = torch.exp(-torch.arange(d, device=z_base.device, dtype=torch.float32) / 15.5)
        cone_raw = z_base * env.unsqueeze(0)
        cone_raw[:, 0] = cone_raw[:, 0] + 0.42
        return F.normalize(cone_raw, p=2.0, dim=-1), block_sz

    m_iso = F.normalize(z_base, p=2.0, dim=-1)
    if regime == "random_sphere":
        return m_iso, block_sz

    if regime == "incomplete_block_qr":
        n_full, rem = v_size // block_sz, v_size % block_sz
        blocks = m_iso[: n_full * block_sz].view(n_full, block_sz, d)
        q_prev_rows = _orthonormalize_row_blocks(blocks)[:-1]
        proj = torch.bmm(torch.bmm(blocks[1:], q_prev_rows.transpose(-1, -2)), q_prev_rows)
        damped = blocks.clone()
        damped[1:] = blocks[1:] - 0.5 * proj
        out_blocks = _orthonormalize_row_blocks(damped)
        out_full = out_blocks.reshape(n_full * block_sz, d)
        if rem == 0:
            return out_full, block_sz
        tail = m_iso[n_full * block_sz :]
        q_last_rows = out_blocks[-1]
        tail_damped = tail - 0.5 * ((tail @ q_last_rows.t()) @ q_last_rows)
        q_tail, _ = torch.linalg.qr(tail_damped.t())
        return torch.cat([out_full, F.normalize(q_tail.t(), p=2.0, dim=-1)], dim=0), block_sz

    if regime == "welch_frame_landing":
        m_cur = m_iso.clone()
        eye_d = torch.eye(d, device=z_base.device, dtype=torch.float32)
        scale_dv = float(d) / float(v_size)
        for _ in range(cfg.num_landing_steps):
            diff = scale_dv * (m_cur.t() @ m_cur) - eye_d
            denom = 1.0 + (torch.linalg.norm(diff) / math.sqrt(float(d)))
            m_cur = F.normalize(m_cur - cfg.landing_lr * (m_cur @ (diff / denom)), p=2.0, dim=-1)
        return m_cur, block_sz

    raise ValueError(f"Unknown packing regime: {regime}")


@torch.no_grad()
def compute_intra_block_rms_cos(m_vecs: torch.Tensor, block_sz: int) -> float:
    """Compute RMS off-diagonal cosine similarity within local blocks of size `block_sz <= d`."""
    if block_sz <= 1 or m_vecs.size(0) < block_sz:
        return 0.0
    n_probe = min(16, m_vecs.size(0) // block_sz)
    sub = m_vecs[: n_probe * block_sz].view(n_probe, block_sz, -1)
    gram_b = torch.bmm(sub, sub.transpose(-1, -2))
    eye_b = torch.eye(block_sz, device=m_vecs.device, dtype=torch.float32).unsqueeze(0)
    off_sq = ((gram_b - eye_b) ** 2).sum(dim=(-2, -1))
    return float(torch.sqrt((off_sq / float(block_sz * (block_sz - 1))).mean()).item())


@torch.no_grad()
def evaluate_ce_floors_for_alpha(
    cos_mat: torch.Tensor, q_idx: torch.Tensor, d: int, alpha_target: float, cfg: ParsevalProfilerConfig,
) -> dict[str, float]:
    """Evaluate achievable Cross-Entropy (nats) and tail log-Z inflation for a given `alpha_target`."""
    n_q, v_size = cos_mat.shape
    row_arange = torch.arange(n_q, device=cos_mat.device)
    r_tilde = float(cfg.beta) / float(cfg.k_copies)
    k_val = float(cfg.k_copies)

    c_off = cos_mat.clone()
    c_off[row_arange, q_idx] = 0.0
    alpha_sq = float(alpha_target) * float(alpha_target)
    rem_scale = math.sqrt(max(0.0, 1.0 - alpha_sq))
    target_ov_uncentered = 0.149 + 0.258 * float(alpha_target)
    target_ov_measured = 0.165 + 0.295 * float(alpha_target)

    c_sq_scaled = (rem_scale * rem_scale) * (0.48 * k_val) * (c_off * c_off)
    c_sq_scaled[row_arange, q_idx] = target_ov_uncentered
    e_raw_quad = k_val * torch.log1p(r_tilde * c_sq_scaled)
    ce_raw_quad = float(F.cross_entropy(e_raw_quad, q_idx).item())

    c_sq_sub = torch.relu(c_sq_scaled - (1.25 * (0.48 * k_val) / float(d)))
    c_sq_sub[row_arange, q_idx] = target_ov_measured
    e_trace_quad = k_val * torch.log1p(r_tilde * c_sq_sub)
    ce_trace_quad = float(F.cross_entropy(e_trace_quad, q_idx).item())

    c_lin = rem_scale * cos_mat
    c_lin[row_arange, q_idx] = target_ov_uncentered
    e_uncentered = k_val * torch.log(torch.clamp_min(1.0 + r_tilde * c_lin, 1e-4))
    ce_uncentered = float(F.cross_entropy(e_uncentered, q_idx).item())

    c_meas = rem_scale * cos_mat
    c_meas[row_arange, q_idx] = target_ov_measured
    c_hat = c_meas - c_meas.mean(dim=-1, keepdim=True)
    c_hat_sq = torch.relu(c_hat) ** 2
    tau_pos = c_hat_sq.mean(dim=-1, keepdim=True)
    f_sharp = (
        c_hat
        + float(cfg.traceless_gamma) * (c_hat_sq - tau_pos)
        - (115.0 / float(v_size)) * c_hat_sq.sum(dim=-1, keepdim=True) / k_val
    )
    f_sharp[row_arange, q_idx] = c_hat[row_arange, q_idx] + float(cfg.traceless_gamma) * (
        c_hat_sq[row_arange, q_idx] - tau_pos.squeeze(-1)
    )
    e_traceless = k_val * torch.log(torch.clamp_min(1.0 + r_tilde * f_sharp, 1e-4))
    ce_traceless = float(F.cross_entropy(e_traceless, q_idx).item())

    uniform_tail_logz = math.log(float(max(1, v_size - 1)))
    mask_non_target = torch.ones_like(e_uncentered, dtype=torch.bool)
    mask_non_target[row_arange, q_idx] = False
    tail_unc = e_uncentered[mask_non_target].view(n_q, v_size - 1)
    tail_trc = e_traceless[mask_non_target].view(n_q, v_size - 1)
    delta_logz_unc = float((torch.logsumexp(tail_unc, dim=-1) - uniform_tail_logz).mean().item())
    delta_logz_trc = float((torch.logsumexp(tail_trc, dim=-1) - uniform_tail_logz).mean().item())

    return {
        "alpha_target": float(alpha_target),
        "raw_quadratic_born_ce": round(ce_raw_quad, 4),
        "traceless_quadratic_born_ce": round(ce_trace_quad, 4),
        "uncentered_kboson_ce": round(ce_uncentered, 4),
        "traceless_observable_kboson_ce": round(ce_traceless, 4),
        "delta_logz_tail_uncentered": round(delta_logz_unc, 4),
        "delta_logz_tail_traceless": round(delta_logz_trc, 4),
    }


@torch.no_grad()
def profile_single_dimension(
    d: int, cfg: ParsevalProfilerConfig, device: torch.device,
) -> dict[str, Any]:
    """Profile all four vector-packing regimes at embedding dimension `d`."""
    v_size = int(cfg.vocab_size)
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed) + int(d) * 17)
    z_base = torch.randn((v_size, d), generator=gen, device=device, dtype=torch.float32)

    max_ortho = min(d, v_size)
    welch_lb = math.sqrt(float(v_size - d) / float(d * (v_size - 1))) if v_size > d else 0.0
    n_q = min(int(cfg.probe_queries), v_size)
    q_idx = (torch.arange(n_q, device=device, dtype=torch.long) * max(1, v_size // n_q))[:n_q]
    row_arange = torch.arange(n_q, device=device)

    regimes_out: dict[str, Any] = {}
    for regime in PACKING_REGIMES:
        m_vecs, block_sz = build_regime_vectors(z_base, regime, cfg)
        gram = m_vecs.t() @ m_vecs
        tr_g = float(torch.diagonal(gram).sum().item())
        frob_sq = float((gram * gram).sum().clamp_min(1e-12).item())
        r_sharp = (tr_g * tr_g) / frob_sq
        parseval_sum_sq = float(v_size) / max(r_sharp, 1e-8)

        cos_mat = m_vecs.index_select(0, q_idx) @ m_vecs.t()
        cos_off = cos_mat.clone()
        cos_off[row_arange, q_idx] = 0.0
        emp_rms_off = float(torch.sqrt((cos_off * cos_off).sum() / float(n_q * max(1, v_size - 1))).item())
        max_abs_off = float(cos_off.abs().max().item())
        intra_rms = compute_intra_block_rms_cos(m_vecs, block_sz)

        regimes_out[regime] = {
            "dim": d,
            "vocab_size": v_size,
            "block_qr_size": block_sz,
            "max_orthogonal_subset_size": max_ortho,
            "welch_rms_lower_bound": round(welch_lb, 6),
            "empirical_rms_off_diag_cos": round(emp_rms_off, 6),
            "intra_block_rms_off_diag_cos": round(intra_rms, 6),
            "max_abs_off_diag_cos": round(max_abs_off, 6),
            "sharp_eff_rank": round(r_sharp, 2),
            "parseval_total_sq_overlap": round(parseval_sum_sq, 2),
            "probe_328": evaluate_ce_floors_for_alpha(cos_mat, q_idx, d, cfg.alpha_target_328, cfg),
            "probe_250": evaluate_ce_floors_for_alpha(cos_mat, q_idx, d, cfg.alpha_target_250, cfg),
        }

    return {
        "dim": d,
        "vocab_size": v_size,
        "max_orthogonal_subset_size": max_ortho,
        "welch_rms_lower_bound": round(welch_lb, 6),
        "regimes": regimes_out,
    }


def run_parseval_profiler(cfg: ParsevalProfilerConfig) -> dict[str, Any]:
    """Run the full dimensional and regime Parseval overlap sweep and print Markdown tables."""
    device = torch.device(cfg.device)
    dim_reports = {str(d): profile_single_dimension(int(d), cfg, device) for d in cfg.dims}

    print(
        f"\n### Parseval / Incomplete Block-QR Packing Sweep (`V={cfg.vocab_size}`, "
        f"`beta={cfg.beta}`, `k={cfg.k_copies}`, `gamma={cfg.traceless_gamma}`)"
    )
    print(
        "| Dim `d` | Welch RMS Bound | QR Intra-Block RMS | QR Global RMS | "
        "`zipf_cone` `R_sharp` | `welch_landing` `R_sharp` | Parseval `sum c^2` (`V/R`) | "
        "Uncentered CE (`a=0.55`) | Traceless CE (`a=0.55`) | Uncentered CE (`a=0.68`) | Traceless CE (`a=0.68`) |"
    )
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for d in cfg.dims:
        dr = dim_reports[str(d)]
        r_cone, r_qr, r_welch = (
            dr["regimes"]["zipf_cone"],
            dr["regimes"]["incomplete_block_qr"],
            dr["regimes"]["welch_frame_landing"],
        )
        print(
            f"| `{d}` | `{dr['welch_rms_lower_bound']:.5f}` | "
            f"`{r_qr['intra_block_rms_off_diag_cos']:.5f}` | `{r_qr['empirical_rms_off_diag_cos']:.5f}` | "
            f"`{r_cone['sharp_eff_rank']:.2f}` | `{r_welch['sharp_eff_rank']:.2f}` | "
            f"`{r_welch['parseval_total_sq_overlap']:.2f}` | "
            f"`{r_welch['probe_328']['uncentered_kboson_ce']:.4f}` | "
            f"**`{r_welch['probe_328']['traceless_observable_kboson_ce']:.4f}`** | "
            f"`{r_welch['probe_250']['uncentered_kboson_ce']:.4f}` | "
            f"**`{r_welch['probe_250']['traceless_observable_kboson_ce']:.4f}`** |"
        )

    report = {"config": asdict(cfg), "dimensions": dim_reports}
    if cfg.out_json:
        out_p = Path(cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> ParsevalProfilerConfig:
    """Parse CLI flags into a frozen `ParsevalProfilerConfig`."""
    p = argparse.ArgumentParser(description="Profile Parseval bounds, Incomplete Block-QR, and Traceless CE floors.")
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--dims", type=str, default="16,32,64,128,256,384,512,1024,2048")
    p.add_argument("--probe-queries", type=int, default=256)
    p.add_argument("--num-landing-steps", type=int, default=6)
    p.add_argument("--landing-lr", type=float, default=0.28)
    p.add_argument("--beta", type=float, default=32.0)
    p.add_argument("--k-copies", type=float, default=34.0)
    p.add_argument("--traceless-gamma", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=2038)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    dims_tuple = tuple(int(x.strip()) for x in ns.dims.split(",") if x.strip())
    return ParsevalProfilerConfig(
        vocab_size=ns.vocab_size, dims=dims_tuple, probe_queries=ns.probe_queries,
        num_landing_steps=ns.num_landing_steps, landing_lr=ns.landing_lr,
        beta=ns.beta, k_copies=ns.k_copies, traceless_gamma=ns.traceless_gamma,
        seed=ns.seed, device=ns.device, out_json=ns.out_json,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    run_parseval_profiler(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
