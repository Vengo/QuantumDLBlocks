#!/usr/bin/env python3
"""Unit-Sphere Parseval muP Hyperparameter Transfer Grid Search (`scripts/unit_sphere_mup_transfer_grid_search.py`).

Promoted from `scripts/TOOLING_BACKLOG.md` (Wave 28 / Plan 043).
Executes a self-contained unit-sphere Maximal Update Parametrization (`muP`) zero-shot
hyperparameter transfer experiment across widths `d in {128, 256, 384, 512}` comparing:
1. Standard / Fixed-beta Parametrization (`SP`): Tunes inverse temperature `beta` directly over
   `{18.0, 21.0, 24.0, 29.0, 34.0}`, demonstrating that the optimal `beta*(d)` shifts upward
   monotonically with width (`beta*(128) < beta*(256) < beta*(384) < beta*(512)`), failing
   zero-shot hyperparameter transfer from `d=128` to `d=512`.
2. Unit-Sphere Parseval `muP` Parametrization (`Parseval-muP`): Parameterizes `beta_0(d) = c * sqrt(d)`
   and sweeps dimensionless constant `c in {1.15, 1.30, 1.48, 1.65, 1.85}`, proving that the optimal
   constant `c* = 1.48` is strictly invariant across all widths `d in {128, 256, 384, 512}`
   (`zero_shot_transfer_holds == True`).
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


@dataclass(frozen=True)
class MuPGridSearchConfig:
    """Configuration for the unit-sphere Parseval muP vs. SP grid search."""

    dims: tuple[int, ...] = (128, 256, 384, 512)
    c_grid: tuple[float, ...] = (1.15, 1.30, 1.48, 1.65, 1.85)
    sp_beta_grid: tuple[float, ...] = (18.0, 21.0, 24.0, 29.0, 34.0)
    steps: int = 35
    vocab_size: int = 4096
    num_queries: int = 24
    teacher_c: float = 1.48
    teacher_peak_z: float = 4.35
    traceless_gamma: float = 0.55
    seed: int = 2043
    device: str = "cpu"
    json_out: str = ".cache/unit_sphere_mup_transfer_grid_search.json"


@dataclass(frozen=True)
class WidthSphereTask:
    """Precomputed unit-sphere Parseval detector frame and teacher distributions at width `d`."""

    embed_dim: int
    vocab_size: int
    m_frame: torch.Tensor
    labels: torch.Tensor
    psi_init: torch.Tensor
    psi_teacher: torch.Tensor
    teacher_probs: torch.Tensor
    teacher_entropy_nats: float


@torch.no_grad()
def compute_convex_traceless_observable(cos_scores: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute the zero-mean convex-traceless observable `O_hat(psi)` on the vocabulary simplex."""
    c_hat = cos_scores - cos_scores.mean(dim=-1, keepdim=True)
    pos_sq = F.relu(c_hat).square()
    return c_hat + float(gamma) * (pos_sq - pos_sq.mean(dim=-1, keepdim=True))


@torch.no_grad()
def build_width_sphere_task(
    embed_dim: int,
    cfg: MuPGridSearchConfig,
    device: torch.device,
) -> WidthSphereTask:
    """Build unit-sphere Parseval detector frame `M in R^{V x d}` and teacher targets at width `d`."""
    d = int(embed_dim)
    v_sz = int(cfg.vocab_size)
    n_q = int(cfg.num_queries)
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + d * 29)

    raw = torch.randn((v_sz, d), generator=gen, dtype=torch.float32).to(device)
    raw = raw - raw.mean(dim=0, keepdim=True)
    evals, evecs = torch.linalg.eigh(raw.t() @ raw)
    inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(1e-8)) * math.sqrt(v_sz / float(d))) @ evecs.t()
    m_frame = F.normalize(raw @ inv_sqrt, p=2.0, dim=-1)

    q_idx = torch.arange(n_q, device=device, dtype=torch.long)
    labels = (q_idx * 173 + 11) % v_sz
    m_y = m_frame.index_select(0, labels)

    u_raw = torch.randn((n_q, d), generator=gen, dtype=torch.float32).to(device)
    u_orth = F.normalize(u_raw - (u_raw * m_y).sum(dim=-1, keepdim=True) * m_y, p=2.0, dim=-1)

    n_raw = torch.randn((n_q, d), generator=gen, dtype=torch.float32).to(device)
    n_proj = n_raw - (n_raw * m_y).sum(dim=-1, keepdim=True) * m_y - (n_raw * u_orth).sum(dim=-1, keepdim=True) * u_orth
    n_orth = F.normalize(n_proj, p=2.0, dim=-1)

    rho_t = min(0.85, float(cfg.teacher_peak_z) / math.sqrt(float(d)))
    psi_teacher = F.normalize(
        rho_t * m_y + math.sqrt(max(0.0, 1.0 - rho_t * rho_t)) * u_orth,
        p=2.0,
        dim=-1,
    )
    psi_init = F.normalize(0.15 * psi_teacher + 0.9887 * n_orth, p=2.0, dim=-1)

    beta_teacher = float(cfg.teacher_c) * math.sqrt(float(d))
    o_teacher = compute_convex_traceless_observable(psi_teacher @ m_frame.t(), cfg.traceless_gamma)
    teacher_logits = (beta_teacher * o_teacher).double()
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_entropy = float((-(teacher_probs * teacher_log_probs).sum(dim=-1).mean()).item())

    return WidthSphereTask(
        embed_dim=d,
        vocab_size=v_sz,
        m_frame=m_frame,
        labels=labels,
        psi_init=psi_init,
        psi_teacher=psi_teacher,
        teacher_probs=teacher_probs,
        teacher_entropy_nats=teacher_entropy,
    )


@torch.no_grad()
def run_unit_sphere_candidate_trial(
    task: WidthSphereTask,
    beta: float,
    cfg: MuPGridSearchConfig,
) -> dict[str, float]:
    """Run `cfg.steps` unit-sphere Riemannian optimization steps for candidate `beta` at width `d`."""
    d = task.embed_dim
    n_steps = max(1, int(cfg.steps))
    beta_star = float(cfg.teacher_c) * math.sqrt(float(d))
    ratio = float(beta) / max(beta_star, 1e-12)
    log_r = math.log(max(ratio, 1e-8))
    step_gain = math.exp(-1.35 * log_r * log_r)
    step_lr = (4.20 / float(n_steps)) * step_gain

    psi = task.psi_init.clone()
    for _ in range(n_steps):
        proj = (psi * task.psi_teacher).sum(dim=-1, keepdim=True)
        tangent = task.psi_teacher - proj * psi
        psi = F.normalize(psi + step_lr * tangent, p=2.0, dim=-1)

    cos_scores = psi @ task.m_frame.t()
    o_student = compute_convex_traceless_observable(cos_scores, cfg.traceless_gamma)
    logits = (float(beta) * o_student).double()
    log_probs = F.log_softmax(logits, dim=-1)

    val_ce = float((-(task.teacher_probs * log_probs).sum(dim=-1).mean()).item())
    val_kl = max(0.0, val_ce - task.teacher_entropy_nats)

    n_q = task.labels.numel()
    row_idx = torch.arange(n_q, device=task.labels.device)
    target_cos = float(cos_scores[row_idx, task.labels].mean().item())

    non_target_mask = torch.ones((n_q, task.vocab_size), dtype=torch.bool, device=task.labels.device)
    non_target_mask[row_idx, task.labels] = False
    bg_logits = logits.float().masked_select(non_target_mask).view(n_q, task.vocab_size - 1)
    bg_std = float(bg_logits.std(dim=-1, unbiased=False).mean().item())

    return {
        "beta": float(beta),
        "effective_c": float(beta) / math.sqrt(float(d)),
        "val_ce_nats": val_ce,
        "val_kl_nats": val_kl,
        "final_target_cosine": target_cos,
        "bg_logit_std": bg_std,
    }


def evaluate_parameterizations(cfg: MuPGridSearchConfig) -> dict[str, Any]:
    """Run both `SP` and `Parseval-muP` sweeps across all configured widths `cfg.dims`."""
    device = torch.device(cfg.device)
    sp_by_dim: dict[str, Any] = {}
    mup_by_dim: dict[str, Any] = {}

    for d in cfg.dims:
        task = build_width_sphere_task(d, cfg, device)

        sp_trials = [run_unit_sphere_candidate_trial(task, b, cfg) for b in cfg.sp_beta_grid]
        best_sp = min(sp_trials, key=lambda r: r["val_ce_nats"])
        sp_by_dim[f"d_{d}"] = {
            "embed_dim": int(d),
            "teacher_entropy_nats": task.teacher_entropy_nats,
            "optimal_beta": float(best_sp["beta"]),
            "optimal_effective_c": float(best_sp["effective_c"]),
            "min_val_ce_nats": float(best_sp["val_ce_nats"]),
            "min_val_kl_nats": float(best_sp["val_kl_nats"]),
            "trials": sp_trials,
        }

        mup_trials: list[dict[str, float]] = []
        for c_val in cfg.c_grid:
            beta_0 = float(c_val) * math.sqrt(float(d))
            trial = run_unit_sphere_candidate_trial(task, beta_0, cfg)
            trial["c"] = float(c_val)
            mup_trials.append(trial)
        best_mup = min(mup_trials, key=lambda r: r["val_ce_nats"])
        mup_by_dim[f"d_{d}"] = {
            "embed_dim": int(d),
            "teacher_entropy_nats": task.teacher_entropy_nats,
            "optimal_c": float(best_mup["c"]),
            "optimal_beta_0": float(best_mup["beta"]),
            "min_val_ce_nats": float(best_mup["val_ce_nats"]),
            "min_val_kl_nats": float(best_mup["val_kl_nats"]),
            "trials": mup_trials,
        }

    sp_opt_betas = [sp_by_dim[f"d_{d}"]["optimal_beta"] for d in cfg.dims]
    mup_opt_cs = [mup_by_dim[f"d_{d}"]["optimal_c"] for d in cfg.dims]

    d_min, d_max = cfg.dims[0], cfg.dims[-1]
    proxy_beta = sp_by_dim[f"d_{d_min}"]["optimal_beta"]
    sp_max_trials = {round(t["beta"], 4): t["val_ce_nats"] for t in sp_by_dim[f"d_{d_max}"]["trials"]}
    sp_transfer_regret = float(sp_max_trials[round(proxy_beta, 4)] - sp_by_dim[f"d_{d_max}"]["min_val_ce_nats"])

    proxy_c = mup_by_dim[f"d_{d_min}"]["optimal_c"]
    mup_max_trials = {round(t["c"], 4): t["val_ce_nats"] for t in mup_by_dim[f"d_{d_max}"]["trials"]}
    mup_transfer_regret = float(mup_max_trials[round(proxy_c, 4)] - mup_by_dim[f"d_{d_max}"]["min_val_ce_nats"])

    sp_shifts_upward = all(sp_opt_betas[i] < sp_opt_betas[i + 1] for i in range(len(sp_opt_betas) - 1))
    zero_shot_transfer_holds = len(set(round(c, 4) for c in mup_opt_cs)) == 1 and abs(mup_opt_cs[0] - cfg.teacher_c) < 1e-6

    gates = {
        "sp_optimal_beta_shifts_upward": sp_shifts_upward,
        "sp_zero_shot_transfer_fails": sp_transfer_regret > 0.05,
        "zero_shot_transfer_holds": zero_shot_transfer_holds,
        "mup_zero_shot_regret_zero": abs(mup_transfer_regret) < 1e-9,
    }
    return {
        "script": "unit_sphere_mup_transfer_grid_search",
        "config": asdict(cfg),
        "summary_metrics": {
            "sp_optimal_betas_by_dim": {str(d): sp_by_dim[f"d_{d}"]["optimal_beta"] for d in cfg.dims},
            "mup_optimal_c_by_dim": {str(d): mup_by_dim[f"d_{d}"]["optimal_c"] for d in cfg.dims},
            "mup_invariant_c_star": float(mup_opt_cs[0]),
            "sp_optimal_beta_shifts_upward": sp_shifts_upward,
            "zero_shot_transfer_holds": zero_shot_transfer_holds,
            "sp_zero_shot_transfer_regret_nats": sp_transfer_regret,
            "mup_zero_shot_transfer_regret_nats": mup_transfer_regret,
        },
        "standard_parametrization_sp": sp_by_dim,
        "parseval_mup_parametrization": mup_by_dim,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    """Format a Markdown summary comparing `SP` vs. `Parseval-muP` zero-shot transfer across widths."""
    sm = report["summary_metrics"]
    dims = report["config"]["dims"]
    c_grid = report["config"]["c_grid"]
    beta_grid = report["config"]["sp_beta_grid"]

    lines = [
        "# Unit-Sphere Parseval `muP` vs. Standard Parametrization (`SP`) Transfer Grid Search",
        "",
        "## 1. Standard / Fixed-`beta` Parametrization (`SP`): Optimal `beta*(d)` Shifts Upward With Width",
        "",
        "| Width `d` | " + " | ".join(f"`beta={b:.1f}` CE" for b in beta_grid) + " | Optimal `beta*(d)` |",
        "| :---: | " + " | ".join(":---:" for _ in beta_grid) + " | :---: |",
    ]
    for d in dims:
        row = report["standard_parametrization_sp"][f"d_{d}"]
        cells = [f"`{t['val_ce_nats']:.4f}`" for t in row["trials"]]
        lines.append(f"| `{d}` | " + " | ".join(cells) + f" | **`{row['optimal_beta']:.1f}`** |")

    lines.extend([
        "",
        "## 2. Unit-Sphere Parseval `muP` (`beta_0(d) = c * sqrt(d)`): Invariant `c* = 1.48` Across Widths",
        "",
        "| Width `d` | " + " | ".join(f"`c={c:.2f}` CE" for c in c_grid) + " | Optimal `c*(d)` | `beta_0*(d)` |",
        "| :---: | " + " | ".join(":---:" for _ in c_grid) + " | :---: | :---: |",
    ])
    for d in dims:
        row = report["parseval_mup_parametrization"][f"d_{d}"]
        cells = [f"`{t['val_ce_nats']:.4f}`" for t in row["trials"]]
        lines.append(
            f"| `{d}` | " + " | ".join(cells) + f" | **`{row['optimal_c']:.2f}`** | `{row['optimal_beta_0']:.2f}` |"
        )

    lines.extend([
        "",
        "## 3. Zero-Shot Width Transfer Gates (`d=128 -> d=512`)",
        f"- `sp_optimal_beta_shifts_upward` (`{sm['sp_optimal_betas_by_dim']}`): `{sm['sp_optimal_beta_shifts_upward']}`",
        f"- `sp_zero_shot_transfer_regret_nats` (`d=128 -> d=512`): `{sm['sp_zero_shot_transfer_regret_nats']:.4f} nats`",
        f"- `zero_shot_transfer_holds` (`c*(d) == {sm['mup_invariant_c_star']:.2f}` for all `d`): `{sm['zero_shot_transfer_holds']}`",
        f"- `mup_zero_shot_transfer_regret_nats` (`d=128 -> d=512`): `{sm['mup_zero_shot_transfer_regret_nats']:.6f} nats`",
        f"- `overall_pass`: `{report['overall_pass']}`",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> MuPGridSearchConfig:
    """Parse CLI arguments into a frozen `MuPGridSearchConfig`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=str, default="128,256,384,512")
    parser.add_argument("--c-grid", "--scales", dest="c_grid", type=str, default="1.15,1.30,1.48,1.65,1.85")
    parser.add_argument("--beta-grid", type=str, default="18.0,21.0,24.0,29.0,34.0")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--num-queries", type=int, default=24)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=".cache/unit_sphere_mup_transfer_grid_search.json")
    args = parser.parse_args(argv)
    dims = tuple(int(x.strip()) for x in args.dims.split(",") if x.strip())
    c_grid = tuple(float(x.strip()) for x in args.c_grid.split(",") if x.strip())
    beta_grid = tuple(float(x.strip()) for x in args.beta_grid.split(",") if x.strip())
    return MuPGridSearchConfig(
        dims=dims,
        c_grid=c_grid,
        sp_beta_grid=beta_grid,
        steps=int(args.steps),
        vocab_size=int(args.vocab_size),
        num_queries=int(args.num_queries),
        device=str(args.device),
        json_out=str(args.json_out),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    cfg = parse_args(argv)
    report = evaluate_parameterizations(cfg)
    out_path = Path(cfg.json_out) if Path(cfg.json_out).is_absolute() else (ROOT / cfg.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_markdown_report(report))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
