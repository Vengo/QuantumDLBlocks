r"""Verification Script for `EXPERIMENT_PLAN_083.md` (`Wave 63`): Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density-Matrix Highway & `512D` Tomographic Readout."""

from __future__ import annotations

import json
import torch

from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (
    ExplicitDensityTomographicGPT,
    ExplicitDensityTomographicSpec,
    ExplicitDensityVariantMode,
    initialize_explicit_density_tomographic_model_,
)
from scripts.explicit_density_tomography_and_eval_speed_profiler import (
    audit_isometric_and_tomographic_trace_identities,
    audit_rule9_single_gemm_tomographic_autograd,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identities = audit_isometric_and_tomographic_trace_identities(device)
    assert identities["passed"], f"Tomographic trace identity check failed: {identities}"

    autograd_rep = audit_rule9_single_gemm_tomographic_autograd(device)
    assert autograd_rep["passed"], f"Rule-9 autograd efficiency audit failed: {autograd_rep}"

    idx = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
    targets = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)

    variant_checks: dict[str, dict[str, float | int]] = {}
    for mode, use_rho, use_tomo, use_vac, use_reg in [
        (ExplicitDensityVariantMode.T0_PURE_WAVEPACKET_RANK1_BASELINE, False, False, False, False),
        (ExplicitDensityVariantMode.T1_EXPLICIT_OUTER_DENSITY_TOMOGRAPHIC_512D, True, True, False, False),
        (ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM, True, True, True, False),
        (ExplicitDensityVariantMode.T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE, True, True, True, True),
    ]:
        spec = ExplicitDensityTomographicSpec(
            variant_mode=mode,
            enable_density_highway=use_rho,
            enable_tomographic_readout=use_tomo,
            enable_noclick_and_vacuum=use_vac,
            enable_two_register_bilinear=use_reg,
            num_layers=2,
        )
        model = ExplicitDensityTomographicGPT(spec).to(device=device)
        init_ms, meta = initialize_explicit_density_tomographic_model_(model)
        assert meta["init_tokens_scanned"] == 0

        loss = model.forward_loss(idx, targets)
        assert torch.isfinite(loss), f"Non-finite loss for {mode.value}"
        loss.backward()
        assert model.wte_512.grad is not None and torch.isfinite(model.wte_512.grad).all()
        assert model.w_det_512.grad is not None and torch.isfinite(model.w_det_512.grad).all()

        telem = model.get_noclick_and_density_telemetry()
        variant_checks[mode.value] = {
            "loss": round(float(loss.detach().item()), 4),
            "init_ms": round(init_ms, 2),
            "init_tokens_scanned": int(meta["init_tokens_scanned"]),
            **telem,
        }

    summary = {
        "plan": "EXPERIMENT_PLAN_083",
        "wave": "W63",
        "identities": identities,
        "rule9_autograd": autograd_rep,
        "variants": variant_checks,
        "status": "PASSED",
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
