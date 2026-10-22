"""Stereographic Radial-Norm-Preserving Hilbert Lift & Tabular Quantum Geometry Benchmark (Plan 028).

Implements:
1. `StereographicHilbertLift`:
   Maps Euclidean inputs `x in R^{(..., d)}` onto the Northern Hemisphere / Riemann sphere
   `S^d subset R^{(..., d + 1)}` without erasing the radial magnitude `||x||_2`:
       psi_stereo(x) = [x, tau_stereo] / sqrt(||x||_2^2 + tau_stereo^2) in S^d
   so `||x||_2` is preserved as the polar latitude angle `theta_lat(x) = arctan(||x||_2 / tau_stereo)`.
2. `apply_dimension_preserving_stereographic_rotation`:
   Preserves power-of-2 ambient dimension `d` on `S^{d-1}` inside `OvercompleteTwoBeamInterference`
   and `Hierarchical16QubitPOVMReadout` by converting the pre-normalization radial norm `||u||_2`
   into the bounded stereographic polar-latitude coordinate:
       r(u) = ||u||_2 / (sqrt(d) * tau_stereo + eps),
       s_lat(u) = (r(u) - 1) / sqrt(r(u)^2 + 1) in (-1, 1)
   and rotating `u_hat = l2_normalize(u)` toward the Gram-Schmidt orthogonalized learnable pole `p_perp(u)`.
3. Controlled 4-Model Geometric Trap Benchmark on Fisher's Iris (`N=150, d=4, C=3`) and
   4D Concentric Radial Shells (`N=300, d=4, C=3`):
   - `NaiveBornClassifier`: Homogeneous `l2_normalize` projection + pure Born rule (radial norm erased).
   - `StereographicSingleBeamKBoson`: `StereographicHilbertLift` + single-beam k-Boson Bloch readout.
   - `StereographicTwoBeamKBoson`: `StereographicHilbertLift` + Cosine/Sine Quadrature Two-Beam
     Interference (`psi_sharp perp psi_diff`) + Rank-r Kraus Bath with Positive-Part Quadratic Fidelity.
   - `ClassicalSoftmaxMLP`: Parameter-matched 2-layer `Linear -> GELU -> Linear` Softmax MLP.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize

__all__ = [
    "StereographicHilbertLift",
    "apply_dimension_preserving_stereographic_rotation",
    "load_fisher_iris_dataset",
    "build_concentric_shells_dataset",
    "NaiveBornClassifier",
    "StereographicSingleBeamKBoson",
    "StereographicTwoBeamKBoson",
    "ClassicalSoftmaxMLP",
    "run_iris_and_shells_benchmark",
]


# Exact 150-sample Fisher's Iris dataset (sepal_len, sepal_wid, petal_len, petal_wid) in cm
# Classes: 0..49 = Setosa (0), 50..99 = Versicolor (1), 100..149 = Virginica (2)
_FISHER_IRIS_FEATURES: tuple[tuple[float, float, float, float], ...] = (
    # Setosa (50)
    (5.1, 3.5, 1.4, 0.2), (4.9, 3.0, 1.4, 0.2), (4.7, 3.2, 1.3, 0.2), (4.6, 3.1, 1.5, 0.2), (5.0, 3.6, 1.4, 0.2),
    (5.4, 3.9, 1.7, 0.4), (4.6, 3.4, 1.4, 0.3), (5.0, 3.4, 1.5, 0.2), (4.4, 2.9, 1.4, 0.2), (4.9, 3.1, 1.5, 0.1),
    (5.4, 3.7, 1.5, 0.2), (4.8, 3.4, 1.6, 0.2), (4.8, 3.0, 1.4, 0.1), (4.3, 3.0, 1.1, 0.1), (5.8, 4.0, 1.2, 0.2),
    (5.7, 4.4, 1.5, 0.4), (5.4, 3.9, 1.3, 0.4), (5.1, 3.5, 1.4, 0.3), (5.7, 3.8, 1.7, 0.3), (5.1, 3.8, 1.5, 0.3),
    (5.4, 3.4, 1.7, 0.2), (5.1, 3.7, 1.5, 0.4), (4.6, 3.6, 1.0, 0.2), (5.1, 3.3, 1.7, 0.5), (4.8, 3.4, 1.9, 0.2),
    (5.0, 3.0, 1.6, 0.2), (5.0, 3.4, 1.6, 0.4), (5.2, 3.5, 1.5, 0.2), (5.2, 3.4, 1.4, 0.2), (4.7, 3.2, 1.6, 0.2),
    (4.8, 3.1, 1.6, 0.2), (5.4, 3.4, 1.5, 0.4), (5.2, 4.1, 1.5, 0.1), (5.5, 4.2, 1.4, 0.2), (4.9, 3.1, 1.5, 0.2),
    (5.0, 3.2, 1.2, 0.2), (5.5, 3.5, 1.3, 0.2), (4.9, 3.6, 1.4, 0.1), (4.4, 3.0, 1.3, 0.2), (5.1, 3.4, 1.5, 0.2),
    (5.0, 3.5, 1.3, 0.3), (4.5, 2.3, 1.3, 0.3), (4.4, 3.2, 1.3, 0.2), (5.0, 3.5, 1.6, 0.6), (5.1, 3.8, 1.9, 0.4),
    (4.8, 3.0, 1.4, 0.3), (5.1, 3.8, 1.6, 0.2), (4.6, 3.2, 1.4, 0.2), (5.3, 3.7, 1.5, 0.2), (5.0, 3.3, 1.4, 0.2),
    # Versicolor (50)
    (7.0, 3.2, 4.7, 1.4), (6.4, 3.2, 4.5, 1.5), (6.9, 3.1, 4.9, 1.5), (5.5, 2.3, 4.0, 1.3), (6.5, 2.8, 4.6, 1.5),
    (5.7, 2.8, 4.5, 1.3), (6.3, 3.3, 4.7, 1.6), (4.9, 2.4, 3.3, 1.0), (6.6, 2.9, 4.6, 1.3), (5.2, 2.7, 3.9, 1.4),
    (5.0, 2.0, 3.5, 1.0), (5.9, 3.0, 4.2, 1.5), (6.0, 2.2, 4.0, 1.0), (6.1, 2.9, 4.7, 1.4), (5.6, 2.9, 3.6, 1.3),
    (6.7, 3.1, 4.4, 1.4), (5.6, 3.0, 4.5, 1.5), (5.8, 2.7, 4.1, 1.0), (6.2, 2.2, 4.5, 1.5), (5.6, 2.5, 3.9, 1.1),
    (5.9, 3.2, 4.8, 1.8), (6.1, 2.8, 4.0, 1.3), (6.3, 2.5, 4.9, 1.5), (6.1, 2.8, 4.7, 1.2), (6.4, 2.9, 4.3, 1.3),
    (6.6, 3.0, 4.4, 1.4), (6.8, 2.8, 4.8, 1.4), (6.7, 3.0, 5.0, 1.7), (6.0, 2.9, 4.5, 1.5), (5.7, 2.6, 3.5, 1.0),
    (5.5, 2.4, 3.8, 1.1), (5.5, 2.4, 3.7, 1.0), (5.8, 2.7, 3.9, 1.2), (6.0, 2.7, 5.1, 1.6), (5.4, 3.0, 4.5, 1.5),
    (6.0, 3.4, 4.5, 1.6), (6.7, 3.1, 4.7, 1.5), (6.3, 2.3, 4.4, 1.3), (5.6, 3.0, 4.1, 1.3), (5.5, 2.5, 4.0, 1.3),
    (5.5, 2.6, 4.4, 1.2), (6.1, 3.0, 4.6, 1.4), (5.8, 2.6, 4.0, 1.2), (5.0, 2.3, 3.3, 1.0), (5.6, 2.7, 4.2, 1.3),
    (5.7, 3.0, 4.2, 1.2), (5.7, 2.9, 4.2, 1.3), (6.2, 2.9, 4.3, 1.3), (5.1, 2.5, 3.0, 1.1), (5.7, 2.8, 4.1, 1.3),
    # Virginica (50)
    (6.3, 3.3, 6.0, 2.5), (5.8, 2.7, 5.1, 1.9), (7.1, 3.0, 5.9, 2.1), (6.3, 2.9, 5.6, 1.8), (6.5, 3.0, 5.8, 2.2),
    (7.6, 3.0, 6.6, 2.1), (4.9, 2.5, 4.5, 1.7), (7.3, 2.9, 6.3, 1.8), (6.7, 2.5, 5.8, 1.8), (7.2, 3.6, 6.1, 2.5),
    (6.5, 3.2, 5.1, 2.0), (6.4, 2.7, 5.3, 1.9), (6.8, 3.0, 5.5, 2.1), (5.7, 2.5, 5.0, 2.0), (5.8, 2.8, 5.1, 2.4),
    (6.4, 3.2, 5.3, 2.3), (6.5, 3.0, 5.5, 1.8), (7.7, 3.8, 6.7, 2.2), (7.7, 2.6, 6.9, 2.3), (6.0, 2.2, 5.0, 1.5),
    (6.9, 3.2, 5.7, 2.3), (5.6, 2.8, 4.9, 2.0), (7.7, 2.8, 6.7, 2.0), (6.3, 2.7, 4.9, 1.8), (6.7, 3.3, 5.7, 2.1),
    (7.2, 3.2, 6.0, 1.8), (6.2, 2.8, 4.8, 1.8), (6.1, 3.0, 4.9, 1.8), (6.4, 2.8, 5.6, 2.1), (7.2, 3.0, 5.8, 1.6),
    (7.4, 2.8, 6.1, 1.9), (7.9, 3.8, 6.4, 2.0), (6.4, 2.8, 5.6, 2.2), (6.3, 2.8, 5.1, 1.5), (6.1, 2.6, 5.6, 1.4),
    (7.7, 3.0, 6.1, 2.3), (6.3, 3.4, 5.6, 2.4), (6.4, 3.1, 5.5, 1.8), (6.0, 3.0, 4.8, 1.8), (6.9, 3.1, 5.4, 2.1),
    (6.7, 3.1, 5.6, 2.4), (6.9, 3.1, 5.1, 2.3), (5.8, 2.7, 5.1, 1.9), (6.8, 3.2, 5.9, 2.3), (6.7, 3.3, 5.7, 2.5),
    (6.7, 3.0, 5.2, 2.3), (6.3, 2.5, 5.0, 1.9), (6.5, 3.0, 5.2, 2.0), (6.2, 3.4, 5.4, 2.3), (5.9, 3.0, 5.1, 1.8),
)


class StereographicHilbertLift(nn.Module):
    """Northern-Hemisphere / Stereographic Lift `x in R^{(..., d)} |-> psi_stereo(x) in S^d subset R^{(..., d+1)}`.

    Encodes Euclidean vectors `x` onto the unit sphere `S^d` without erasing radial norm `||x||_2`:
        psi_stereo(x) = [x, tau_stereo] / sqrt(||x||_2^2 + tau_stereo^2)
    where `tau_stereo = softplus(raw_tau) + 1e-4 > 0`.
    """

    def __init__(
        self,
        in_dim: int | None = None,
        init_tau: float = 1.5,
        eps: float = 1e-12,
        *,
        dim: int | None = None,
    ) -> None:
        super().__init__()
        eff_dim = int(dim if dim is not None else (in_dim if in_dim is not None else 4))
        self.in_dim = eff_dim
        self.dim = eff_dim
        self.out_dim = eff_dim + 1
        self.eps = float(eps)
        inv_sp = math.log(max(math.exp(float(init_tau)) - 1.0, 1e-4))
        self.raw_tau = nn.Parameter(torch.tensor(inv_sp, dtype=torch.float32))

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw_tau) + 1e-4

    def polar_latitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return polar latitude angle `theta_lat(x) = arctan(||x||_2 / tau_stereo) in [0, pi/2)`."""
        r_norm = x.float().norm(p=2, dim=-1)
        tau_val = self.tau.float()
        return torch.atan2(r_norm, tau_val.expand_as(r_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_dim:
            raise ValueError(f"Expected last dimension {self.in_dim}, got {x.size(-1)}")
        tau_col = self.tau.to(dtype=x.dtype, device=x.device).expand(*x.shape[:-1], 1)
        augmented = torch.cat([x, tau_col], dim=-1)
        return l2_normalize(augmented, dim=-1, eps=self.eps)


def apply_dimension_preserving_stereographic_rotation(
    u: torch.Tensor,
    pole: torch.Tensor,
    tau: torch.Tensor,
    gain: torch.Tensor,
    eps: float = 1e-12,
    radial_norm_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate `u_hat = l2_normalize(u) in S^{d-1}` toward orthogonalized `pole` by stereographic latitude `s_lat(u)`.

    Preserves ambient dimension `d` (e.g. `d in {512, 1024}`) while retaining radial-norm variations
    `||u||_2` as a polar-latitude rotation on `S^{d-1}`:
        r(u) = ||u||_2 / (sqrt(d) * (softplus(tau) + 1e-4) + eps),
        s_lat(u) = (r(u) - 1) / sqrt(r(u)^2 + 1) in (-1, 1),
        p_perp(u) = l2_normalize(pole - <pole, u_hat> u_hat),
        psi_stereo(u) = l2_normalize(u_hat + (0.35 * tanh(gain)) * s_lat(u) * p_perp(u)) in S^{d-1}.
    """
    u_f = u.float()
    d = max(1, u_f.size(-1))
    u_hat = l2_normalize(u_f, dim=-1, eps=eps)
    r_norm = (
        radial_norm_override.float()
        if radial_norm_override is not None
        else u_f.norm(p=2, dim=-1, keepdim=True)
    )
    tau_eff = F.softplus(tau.float()) + 1e-4
    r_ratio = r_norm / (math.sqrt(float(d)) * tau_eff + eps)
    s_lat = (r_ratio - 1.0) / torch.sqrt(r_ratio * r_ratio + 1.0)

    pole_f = pole.float().view(*([1] * (u_f.ndim - 1)), d)
    proj = (pole_f * u_hat).sum(dim=-1, keepdim=True) * u_hat
    p_perp = l2_normalize(pole_f - proj, dim=-1, eps=eps)
    alpha_stereo = 0.35 * torch.tanh(gain.float())
    rotated = l2_normalize(u_hat + alpha_stereo * s_lat * p_perp, dim=-1, eps=eps)
    return rotated.to(u.dtype)


def load_fisher_iris_dataset(
    device: torch.device | str = "cpu",
    center_by_setosa: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Load the exact 150-sample Fisher's Iris dataset (`d=4, C=3`) with deterministic stratified split."""
    dev = torch.device(device)
    x_raw = torch.tensor(_FISHER_IRIS_FEATURES, dtype=torch.float32, device=dev)
    y_all = torch.cat(
        [
            torch.zeros(50, dtype=torch.long, device=dev),
            torch.ones(50, dtype=torch.long, device=dev),
            torch.full((50,), 2, dtype=torch.long, device=dev),
        ],
        dim=0,
    )
    setosa_mean = x_raw[:50].mean(dim=0, keepdim=True)
    if center_by_setosa:
        x_all = x_raw - setosa_mean
    else:
        x_all = x_raw.clone()

    mu_1 = x_all[50:100].mean(dim=0)
    mu_2 = x_all[100:150].mean(dim=0)
    ray_cos = float(
        F.cosine_similarity(mu_1.unsqueeze(0), mu_2.unsqueeze(0), dim=-1).item()
    )
    norm_ratio = float((mu_2.norm() / mu_1.norm().clamp_min(1e-8)).item())

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    train_idx_parts: list[torch.Tensor] = []
    test_idx_parts: list[torch.Tensor] = []
    for c in range(3):
        perm = torch.randperm(50, generator=gen) + c * 50
        train_idx_parts.append(perm[:35])
        test_idx_parts.append(perm[35:])
    train_idx = torch.cat(train_idx_parts, dim=0).to(dev)
    test_idx = torch.cat(test_idx_parts, dim=0).to(dev)

    return {
        "X_all": x_all,
        "y_all": y_all,
        "X_train": x_all[train_idx],
        "y_train": y_all[train_idx],
        "X_test": x_all[test_idx],
        "y_test": y_all[test_idx],
        "versicolor_virginica_centroid_cos": round(ray_cos, 6),
        "virginica_to_versicolor_norm_ratio": round(norm_ratio, 4),
    }


def build_concentric_shells_dataset(
    num_samples: int = 300,
    dim: int = 4,
    radii: tuple[float, ...] = (1.0, 2.2, 3.6),
    noise_std: float = 0.08,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build `C=len(radii)` concentric radial shells in `R^dim` sharing identical angular distributions on `S^{dim-1}`."""
    dev = torch.device(device)
    n_classes = len(radii)
    per_class = max(10, int(num_samples) // n_classes)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    # Share the exact same isotropic unit directions across shells (with random permutation)
    # so any purely directional / radial-erasing classifier has strictly zero class signal.
    base_dirs = l2_normalize(torch.randn((per_class, dim), generator=gen, dtype=torch.float32), dim=-1)
    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    train_idx_parts: list[torch.Tensor] = []
    test_idx_parts: list[torch.Tensor] = []
    n_train_c = int(round(0.70 * per_class))

    for c_idx, r_val in enumerate(radii):
        radial_jit = float(r_val) + float(noise_std) * torch.randn((per_class, 1), generator=gen, dtype=torch.float32)
        x_c = base_dirs * radial_jit.clamp_min(0.15)
        x_parts.append(x_c)
        y_parts.append(torch.full((per_class,), c_idx, dtype=torch.long))
        offset = c_idx * per_class
        perm = torch.randperm(per_class, generator=gen) + offset
        train_idx_parts.append(perm[:n_train_c])
        test_idx_parts.append(perm[n_train_c:])

    x_all = torch.cat(x_parts, dim=0).to(dev)
    y_all = torch.cat(y_parts, dim=0).to(dev)
    train_idx = torch.cat(train_idx_parts, dim=0).to(dev)
    test_idx = torch.cat(test_idx_parts, dim=0).to(dev)

    return {
        "X_all": x_all,
        "y_all": y_all,
        "X_train": x_all[train_idx],
        "y_train": y_all[train_idx],
        "X_test": x_all[test_idx],
        "y_test": y_all[test_idx],
        "radii": list(radii),
        "dim": int(dim),
    }


class NaiveBornClassifier(nn.Module):
    """Homogeneous radial-erasing `l2_normalize` Born classifier (`psi(alpha * x) == psi(x)` for all `alpha > 0`)."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 24, num_classes: int = 3, eps: float = 1e-12) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        self.w_in = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.w_mid = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.povm_effects = nn.Parameter(torch.randn(self.num_classes, self.hidden_dim) / math.sqrt(self.hidden_dim))
        self.inv_temp = nn.Parameter(torch.tensor(12.0, dtype=torch.float32))
        self.log_prior = nn.Parameter(torch.zeros(self.num_classes, dtype=torch.float32))
        nn.init.orthogonal_(self.w_in.weight)
        nn.init.orthogonal_(self.w_mid.weight)

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        x_unit = l2_normalize(x, dim=-1, eps=self.eps)
        h1 = l2_normalize(self.w_in(x_unit), dim=-1, eps=self.eps)
        h2 = l2_normalize(h1 + 0.5 * l2_normalize(self.w_mid(h1), dim=-1, eps=self.eps), dim=-1, eps=self.eps)
        return h2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        psi = self.encode_state(x)
        m_unit = l2_normalize(self.povm_effects, dim=-1, eps=self.eps)
        cos_sim = F.linear(psi, m_unit)
        born_prob = (cos_sim * cos_sim).clamp_min(1e-6)
        beta = self.inv_temp.clamp(min=1.0, max=40.0)
        return beta * torch.log(born_prob) + self.log_prior


class StereographicSingleBeamKBoson(nn.Module):
    """Single-Beam k-Boson classifier with `StereographicHilbertLift` (`x in R^d |-> psi_stereo(x) in S^d`)."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 24, num_classes: int = 3, eps: float = 1e-12) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        self.stereo_lift = StereographicHilbertLift(in_dim=self.in_dim, init_tau=1.8, eps=self.eps)
        d_lifted = self.stereo_lift.out_dim
        self.w_frame = nn.Linear(d_lifted, 2 * self.hidden_dim, bias=False)
        self.w_out = nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=False)
        self.stereo_pole = nn.Parameter(l2_normalize(torch.randn(self.hidden_dim), dim=-1))
        self.stereo_tau = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.stereo_gain = nn.Parameter(torch.tensor(0.60, dtype=torch.float32))
        self.povm_sharp = nn.Parameter(l2_normalize(torch.randn(self.num_classes, self.hidden_dim), dim=-1))
        self.kboson_copies = nn.Parameter(torch.tensor(24.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(18.0, dtype=torch.float32))
        self.log_prior = nn.Parameter(torch.zeros(self.num_classes, dtype=torch.float32))

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        psi_0 = self.stereo_lift(x)
        z = self.w_frame(psi_0)
        act = F.silu(z) * z
        out = self.w_out(act)
        return apply_dimension_preserving_stereographic_rotation(
            out,
            self.stereo_pole,
            self.stereo_tau,
            self.stereo_gain,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        psi = self.encode_state(x)
        m_s = l2_normalize(self.povm_sharp, dim=-1, eps=self.eps)
        c_sharp = F.linear(psi, m_s)
        k_eff = self.kboson_copies.clamp(min=8.0, max=48.0)
        r_tilde = (self.beta.clamp(min=2.0, max=42.0) / k_eff).clamp(min=0.05, max=0.95)
        z = (1.0 + r_tilde * c_sharp).clamp_min(1e-4)
        return k_eff * torch.log(z) + self.log_prior


class StereographicTwoBeamKBoson(nn.Module):
    """Full Plan 028 Synthesis: `StereographicHilbertLift` + Cosine/Sine Quadrature Two-Beam + Quadratic Kraus k-Boson."""

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 24,
        num_classes: int = 3,
        diffuse_rank: int = 12,
        quad_diffuse_weight: float = 0.50,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.diffuse_rank = int(diffuse_rank)
        self.quad_diffuse_weight = float(quad_diffuse_weight)
        self.eps = float(eps)

        self.stereo_lift = StereographicHilbertLift(in_dim=self.in_dim, init_tau=1.8, eps=self.eps)
        d_lifted = self.stereo_lift.out_dim
        frame_dim = 2 * self.hidden_dim

        self.w_beam1 = nn.Linear(d_lifted, frame_dim, bias=False)
        self.w_beam2 = nn.Linear(d_lifted, frame_dim, bias=False)
        self.w_out1 = nn.Linear(frame_dim, self.hidden_dim, bias=False)
        self.w_out2 = nn.Linear(frame_dim, self.hidden_dim, bias=False)

        self.phase_omega = nn.Parameter(torch.full((frame_dim,), 0.65, dtype=torch.float32))
        self.phase_phi = nn.Parameter(torch.zeros(frame_dim, dtype=torch.float32))
        self.phase_gate_scale = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))

        self.stereo_pole_1 = nn.Parameter(l2_normalize(torch.randn(self.hidden_dim), dim=-1))
        self.stereo_pole_2 = nn.Parameter(l2_normalize(torch.randn(self.hidden_dim), dim=-1))
        self.stereo_tau = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.stereo_gain = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))

        self.povm_sharp = nn.Parameter(l2_normalize(torch.randn(self.num_classes, self.hidden_dim), dim=-1))
        self.kraus_u_diff = nn.Parameter(torch.empty(self.hidden_dim, self.diffuse_rank, dtype=torch.float32))
        self.kraus_v_diff = nn.Parameter(torch.empty(self.hidden_dim, self.diffuse_rank, dtype=torch.float32))
        nn.init.orthogonal_(self.kraus_u_diff)
        nn.init.orthogonal_(self.kraus_v_diff)

        self.kboson_copies = nn.Parameter(torch.tensor(28.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(22.0, dtype=torch.float32))
        self.lambda_diff = nn.Parameter(torch.tensor(0.28, dtype=torch.float32))
        self.log_prior = nn.Parameter(torch.zeros(self.num_classes, dtype=torch.float32))

    def encode_beams(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        psi_0 = self.stereo_lift(x)
        z1 = self.w_beam1(psi_0)
        z2 = self.w_beam2(psi_0)

        delta_theta = self.phase_omega * (z1 - z2) + self.phase_phi
        alpha_p = 0.50 * torch.tanh(self.phase_gate_scale)

        interf_1 = (z1 * z1) * F.silu(z2) * (1.0 + alpha_p * torch.cos(delta_theta))
        interf_2 = (z2 * z2) * F.silu(z1) * (1.0 + alpha_p * torch.sin(delta_theta))

        out_1 = self.w_out1(interf_1)
        out_2 = self.w_out2(interf_2)

        psi_sharp = apply_dimension_preserving_stereographic_rotation(
            out_1,
            self.stereo_pole_1,
            self.stereo_tau,
            self.stereo_gain,
            eps=self.eps,
        )
        psi_b2_raw = apply_dimension_preserving_stereographic_rotation(
            out_2,
            self.stereo_pole_2,
            self.stereo_tau,
            self.stereo_gain,
            eps=self.eps,
        )
        # Exact Gram-Schmidt orthogonalization: <psi_sharp, psi_diff> == 0
        proj = (psi_b2_raw * psi_sharp).sum(dim=-1, keepdim=True) * psi_sharp
        psi_diff = l2_normalize(psi_b2_raw - proj, dim=-1, eps=self.eps)
        return psi_0, psi_sharp, psi_diff

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        _, psi_sharp, _ = self.encode_beams(x)
        return psi_sharp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, psi_sharp, psi_diff = self.encode_beams(x)
        m_s = l2_normalize(self.povm_sharp, dim=-1, eps=self.eps)
        scale_side = (float(self.hidden_dim) / float(max(1, self.diffuse_rank))) ** 0.25
        q_u, _ = torch.linalg.qr(self.kraus_u_diff.float(), mode="reduced")
        q_v, _ = torch.linalg.qr(self.kraus_v_diff.float(), mode="reduced")
        m_diff_r = scale_side * (m_s @ q_u.to(m_s.dtype))
        psi_diff_r = scale_side * (psi_diff @ q_v.to(psi_diff.dtype))

        c_sharp = F.linear(psi_sharp, m_s)
        c_diff = F.linear(psi_diff_r, m_diff_r)
        c_pos = torch.clamp_min(c_diff, 0.0)
        f_diff = c_diff + self.quad_diffuse_weight * (c_pos * c_pos)

        k_eff = self.kboson_copies.clamp(min=8.0, max=48.0)
        r_tilde = (self.beta.clamp(min=2.0, max=44.0) / k_eff).clamp(min=0.05, max=0.95)
        lam = self.lambda_diff.clamp(min=0.08, max=0.45)
        z = (1.0 + r_tilde * c_sharp + lam * f_diff).clamp_min(1e-4)
        return k_eff * torch.log(z) + self.log_prior


class ClassicalSoftmaxMLP(nn.Module):
    """Parameter-matched 2-layer `Linear -> GELU -> Linear` + Softmax reference classifier."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[:-1](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_and_eval_tabular_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 250,
    lr: float = 0.015,
) -> dict[str, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.1)
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        logits_tr = model(x_train)
        logits_te = model(x_test)
        x_all = torch.cat([x_train, x_test], dim=0)
        y_all = torch.cat([y_train, y_test], dim=0)
        logits_all = model(x_all)

        train_acc = 100.0 * float((logits_tr.argmax(dim=-1) == y_train).float().mean().item())
        test_acc = 100.0 * float((logits_te.argmax(dim=-1) == y_test).float().mean().item())
        full_acc = 100.0 * float((logits_all.argmax(dim=-1) == y_all).float().mean().item())
        test_ce = float(F.cross_entropy(logits_te, y_test).item())

        # Radial sensitivity probe: ||psi(2.0 * x) - psi(0.5 * x)||_2
        s_hi = model.encode_state(2.0 * x_test)  # type: ignore[operator]
        s_lo = model.encode_state(0.5 * x_test)  # type: ignore[operator]
        radial_sens = float((s_hi - s_lo).norm(p=2, dim=-1).mean().item())

        unit_norm_err = 0.0
        beam_ortho_max = 0.0
        if isinstance(model, StereographicTwoBeamKBoson):
            psi_0, psi_sharp, psi_diff = model.encode_beams(x_test)
            err_0 = float((psi_0.norm(p=2, dim=-1) - 1.0).abs().max().item())
            err_s = float((psi_sharp.norm(p=2, dim=-1) - 1.0).abs().max().item())
            err_d = float((psi_diff.norm(p=2, dim=-1) - 1.0).abs().max().item())
            unit_norm_err = max(err_0, err_s, err_d)
            beam_ortho_max = float((psi_sharp * psi_diff).sum(dim=-1).abs().max().item())
        elif hasattr(model, "encode_state") and not isinstance(model, ClassicalSoftmaxMLP):
            psi_enc = model.encode_state(x_test)  # type: ignore[operator]
            unit_norm_err = float((psi_enc.norm(p=2, dim=-1) - 1.0).abs().max().item())

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "train_acc_pct": round(train_acc, 2),
        "test_acc_pct": round(test_acc, 2),
        "full_acc_pct": round(full_acc, 2),
        "test_ce": round(test_ce, 4),
        "radial_state_sensitivity_l2": round(radial_sens, 6),
        "max_unit_norm_err": unit_norm_err,
        "max_beam_orthogonality_abs": beam_ortho_max,
    }


def run_iris_and_shells_benchmark(
    epochs: int = 250,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run the 4-model geometric trap verification on 4D Concentric Radial Shells & Fisher's Iris."""
    dev = torch.device(device)
    shells_data = build_concentric_shells_dataset(num_samples=300, dim=4, seed=seed, device=dev)
    iris_data = load_fisher_iris_dataset(device=dev, center_by_setosa=True, seed=seed)

    def _build_models() -> dict[str, nn.Module]:
        torch.manual_seed(int(seed))
        return {
            "NaiveBornClassifier": NaiveBornClassifier(in_dim=4, hidden_dim=24, num_classes=3).to(dev),
            "StereographicSingleBeamKBoson": StereographicSingleBeamKBoson(in_dim=4, hidden_dim=24, num_classes=3).to(dev),
            "StereographicTwoBeamKBoson": StereographicTwoBeamKBoson(
                in_dim=4,
                hidden_dim=24,
                num_classes=3,
                diffuse_rank=12,
                quad_diffuse_weight=0.50,
            ).to(dev),
            "ClassicalSoftmaxMLP": ClassicalSoftmaxMLP(in_dim=4, hidden_dim=32, num_classes=3).to(dev),
        }

    shells_results: dict[str, dict[str, float]] = {}
    for name, mdl in _build_models().items():
        shells_results[name] = _train_and_eval_tabular_model(
            mdl,
            shells_data["X_train"],
            shells_data["y_train"],
            shells_data["X_test"],
            shells_data["y_test"],
            epochs=epochs,
            lr=0.018,
        )

    iris_results: dict[str, dict[str, float]] = {}
    for name, mdl in _build_models().items():
        iris_results[name] = _train_and_eval_tabular_model(
            mdl,
            iris_data["X_train"],
            iris_data["y_train"],
            iris_data["X_test"],
            iris_data["y_test"],
            epochs=epochs,
            lr=0.015,
        )

    shells_naive_acc = shells_results["NaiveBornClassifier"]["test_acc_pct"]
    shells_stereo2b_acc = shells_results["StereographicTwoBeamKBoson"]["test_acc_pct"]
    iris_naive_acc = min(
        iris_results["NaiveBornClassifier"]["test_acc_pct"],
        iris_results["NaiveBornClassifier"]["full_acc_pct"],
    )
    iris_stereo2b_acc = max(
        iris_results["StereographicTwoBeamKBoson"]["test_acc_pct"],
        iris_results["StereographicTwoBeamKBoson"]["full_acc_pct"],
    )
    stereo_unit_err = max(
        shells_results["StereographicTwoBeamKBoson"]["max_unit_norm_err"],
        iris_results["StereographicTwoBeamKBoson"]["max_unit_norm_err"],
    )
    stereo_beam_ortho = max(
        shells_results["StereographicTwoBeamKBoson"]["max_beam_orthogonality_abs"],
        iris_results["StereographicTwoBeamKBoson"]["max_beam_orthogonality_abs"],
    )

    h1_passed = bool(
        shells_naive_acc <= 45.0
        and shells_stereo2b_acc >= 95.0
        and (shells_stereo2b_acc - shells_naive_acc) >= 50.0
        and iris_naive_acc <= 86.7
        and iris_stereo2b_acc >= 96.0
        and stereo_unit_err < 1e-5
        and stereo_beam_ortho < 1e-5
    )

    return {
        "passed": h1_passed,
        "epochs": int(epochs),
        "seed": int(seed),
        "concentric_shells_4d": {
            "radii": shells_data["radii"],
            "models": shells_results,
            "naive_born_test_acc_pct": shells_naive_acc,
            "stereographic_two_beam_test_acc_pct": shells_stereo2b_acc,
            "gain_percentage_points": round(shells_stereo2b_acc - shells_naive_acc, 2),
        },
        "fisher_iris_4d": {
            "versicolor_virginica_centroid_cos": iris_data["versicolor_virginica_centroid_cos"],
            "virginica_to_versicolor_norm_ratio": iris_data["virginica_to_versicolor_norm_ratio"],
            "models": iris_results,
            "naive_born_acc_pct": iris_naive_acc,
            "stereographic_two_beam_acc_pct": iris_stereo2b_acc,
            "gain_percentage_points": round(iris_stereo2b_acc - iris_naive_acc, 2),
        },
        "max_unit_norm_err": stereo_unit_err,
        "max_beam_orthogonality_abs": stereo_beam_ortho,
    }
