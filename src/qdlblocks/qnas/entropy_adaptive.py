"""Method 2: Thermodynamic / Entropy-Driven Adaptive Topology Search."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ..functional import kraus_transform, power_purify, state_to_density
from ..layers.complex_linear import ComplexLinear
from ..layers.measurement import POVMMeasurement
from ..metrics import purity, von_neumann_entropy


class EntropyAdaptiveNetwork(nn.Module):
    """Open Quantum Kraus Network that dynamically adapts Hilbert dimensions via von Neumann Entropy."""

    def __init__(
        self,
        in_features: int = 64,
        stem_dim: int = 8,
        init_dims: tuple[int, ...] = (8, 8, 8),
        num_classes: int = 10,
        num_kraus: int = 3,
        min_dim: int = 6,
        max_dim: int = 20,
        delta_dim: int = 2,
        expand_entropy_ratio: float = 0.65,
        purify_purity_threshold: float = 0.45,
        compress_tail_mass: float = 0.02,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if len(init_dims) < 2:
            raise ValueError("init_dims must have at least 2 stages.")
        self.in_features = in_features
        self.stem_dim = stem_dim
        self.layer_dims: list[int] = list(init_dims)
        self.num_classes = num_classes
        self.num_kraus = num_kraus
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.delta_dim = delta_dim
        self.expand_entropy_ratio = expand_entropy_ratio
        self.purify_purity_threshold = purify_purity_threshold
        self.compress_tail_mass = compress_tail_mass
        self.eps = eps

        self.stem = ComplexLinear(in_features, stem_dim, bias=False)

        self.kraus_real = nn.ParameterList()
        self.kraus_imag = nn.ParameterList()
        for l_idx, n_out in enumerate(self.layer_dims):
            n_in = stem_dim if l_idx == 0 else self.layer_dims[l_idx - 1]
            scale = 1.0 / math.sqrt(2.0 * num_kraus * n_in)
            self.kraus_real.append(nn.Parameter(torch.randn(num_kraus, n_out, n_in) * scale))
            self.kraus_imag.append(nn.Parameter(torch.randn(num_kraus, n_out, n_in) * scale))

        self.purification_powers: list[int] = [1 for _ in self.layer_dims]
        self.adaptation_history: list[dict[str, Any]] = []

        self.povm = POVMMeasurement(hilbert_dim=self.layer_dims[-1], num_classes=num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(15.0))

    def forward(
        self, x: torch.Tensor, return_densities: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        psi_stem = self.stem(x)
        rho = state_to_density(psi_stem, eps=self.eps)
        densities: list[torch.Tensor] = []

        for l_idx in range(len(self.layer_dims)):
            a_l = torch.complex(self.kraus_real[l_idx], self.kraus_imag[l_idx])
            rho = kraus_transform(rho, a_l, eps=self.eps)
            p_l = self.purification_powers[l_idx]
            if p_l > 1:
                rho = power_purify(rho, power=p_l, eps=self.eps)
            densities.append(rho)

        logits = self.logit_scale * self.povm(rho)
        if return_densities:
            return logits, densities
        return logits

    @torch.no_grad()
    def step_topology_adaptation(self, x_val: torch.Tensor) -> dict[str, Any]:
        """Adapt internal Hilbert dimensions and purification powers from validation batch telemetry."""
        was_training = self.training
        self.eval()
        _, densities = self.forward(x_val, return_densities=True)
        if was_training:
            self.train()

        num_stages = len(self.layer_dims)
        entropies: list[float] = []
        purities: list[float] = []
        events: list[str] = []

        for l_idx in range(num_stages):
            rho_bar = densities[l_idx].mean(dim=0)
            s_val = float(von_neumann_entropy(rho_bar, eps=self.eps).item())
            p_val = float(purity(rho_bar).item())
            entropies.append(s_val)
            purities.append(p_val)

        # Adapt internal stages 0 .. num_stages - 2
        for l_idx in range(num_stages - 1):
            k_ops, n_l, n_in_l = self.kraus_real[l_idx].shape
            n_next = self.kraus_real[l_idx + 1].size(1)
            device = self.kraus_real[l_idx].device
            dtype = self.kraus_real[l_idx].dtype

            s_l = entropies[l_idx]
            s_ratio = s_l / max(math.log(max(n_l, 2)), 1e-6)
            gam_l = purities[l_idx]

            # Rule 1: Purification Activation
            if gam_l < self.purify_purity_threshold and self.purification_powers[l_idx] < 3:
                self.purification_powers[l_idx] += 1
                events.append(f"stage_{l_idx}:purify_p->{self.purification_powers[l_idx]}")

            # Rule 2: Hilbert Subspace Expansion
            if s_ratio > self.expand_entropy_ratio and (n_l + self.delta_dim) <= self.max_dim:
                new_n = n_l + self.delta_dim
                pad_out_r = 0.01 * torch.randn(k_ops, self.delta_dim, n_in_l, device=device, dtype=dtype)
                pad_out_i = 0.01 * torch.randn(k_ops, self.delta_dim, n_in_l, device=device, dtype=dtype)
                self.kraus_real[l_idx] = nn.Parameter(
                    torch.cat([self.kraus_real[l_idx].data, pad_out_r], dim=1)
                )
                self.kraus_imag[l_idx] = nn.Parameter(
                    torch.cat([self.kraus_imag[l_idx].data, pad_out_i], dim=1)
                )

                pad_in_r = 0.01 * torch.randn(k_ops, n_next, self.delta_dim, device=device, dtype=dtype)
                pad_in_i = 0.01 * torch.randn(k_ops, n_next, self.delta_dim, device=device, dtype=dtype)
                self.kraus_real[l_idx + 1] = nn.Parameter(
                    torch.cat([self.kraus_real[l_idx + 1].data, pad_in_r], dim=2)
                )
                self.kraus_imag[l_idx + 1] = nn.Parameter(
                    torch.cat([self.kraus_imag[l_idx + 1].data, pad_in_i], dim=2)
                )

                self.layer_dims[l_idx] = new_n
                events.append(f"stage_{l_idx}:expand_{n_l}->{new_n}")

            # Rule 3: Hilbert Subspace Compression (only if Rule 2 did not trigger)
            elif (n_l - self.delta_dim) >= self.min_dim:
                rho_bar = densities[l_idx].mean(dim=0)
                rho_bar = 0.5 * (rho_bar + rho_bar.conj().transpose(-1, -2))
                eigvals, eigvecs = torch.linalg.eigh(rho_bar)
                tail_mass = float(eigvals[: self.delta_dim].real.clamp(min=0.0).sum().item())
                if tail_mass < self.compress_tail_mass:
                    new_n = n_l - self.delta_dim
                    u_top = eigvecs[:, self.delta_dim :]  # (n_l, new_n)
                    a_curr = torch.complex(self.kraus_real[l_idx].data, self.kraus_imag[l_idx].data)
                    a_next = torch.complex(
                        self.kraus_real[l_idx + 1].data, self.kraus_imag[l_idx + 1].data
                    )

                    a_curr_proj = torch.einsum("ij,kjm->kim", u_top.conj().T, a_curr)
                    a_next_proj = torch.einsum("kij,jm->kim", a_next, u_top)

                    self.kraus_real[l_idx] = nn.Parameter(a_curr_proj.real.contiguous())
                    self.kraus_imag[l_idx] = nn.Parameter(a_curr_proj.imag.contiguous())
                    self.kraus_real[l_idx + 1] = nn.Parameter(a_next_proj.real.contiguous())
                    self.kraus_imag[l_idx + 1] = nn.Parameter(a_next_proj.imag.contiguous())

                    self.layer_dims[l_idx] = new_n
                    events.append(f"stage_{l_idx}:compress_{n_l}->{new_n}")

        record: dict[str, Any] = {
            "layer_dims": list(self.layer_dims),
            "purification_powers": list(self.purification_powers),
            "entropies": entropies,
            "purities": purities,
            "events": events,
        }
        self.adaptation_history.append(record)
        return record


AdaptiveKrausBottleneckNetwork = EntropyAdaptiveNetwork
