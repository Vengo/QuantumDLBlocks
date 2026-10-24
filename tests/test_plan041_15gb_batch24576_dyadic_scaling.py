"""Unit tests for Experiment Plan 041 (`Wave 26`): 15 GB RTX 5060 Ti `bfloat16`/`fp8` optimization,
Solution A + B (`Convex-Peak Traceless` + `Dyadic Coarse-to-Fine Born`), and `24,576`-token batch scaling.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.direct_chunked_ce import direct_chunked_two_beam_kboson_ce
from qdlblocks.layers.wavelet_cholesky import BlockCholeskyLinear
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    initialize_radical_2pow16_with_ppmi_svd,
)

ROOT = Path(__file__).resolve().parents[1]


def test_l2_normalize_memory_efficient_bfloat16_backward() -> None:
    """Verify `_L2NormalizeFn` matches float32 reference values and gradients for `bfloat16`."""
    torch.manual_seed(101)
    x_bf16 = torch.randn(4, 16, 64, dtype=torch.bfloat16, requires_grad=True)
    x_fp32 = x_bf16.detach().float().requires_grad_(True)
    grad_out = torch.randn(4, 16, 64, dtype=torch.bfloat16)

    y_bf16 = l2_normalize(x_bf16, dim=-1)
    assert y_bf16.dtype == torch.bfloat16
    y_bf16.backward(grad_out)

    y_ref = (x_fp32 * torch.rsqrt(torch.sum(x_fp32 * x_fp32, dim=-1, keepdim=True) + 1e-12))
    y_ref.backward(grad_out.float())

    assert x_bf16.grad is not None and x_fp32.grad is not None
    assert torch.allclose(y_bf16.float(), y_ref, atol=2e-2, rtol=2e-2)
    assert torch.allclose(x_bf16.grad.float(), x_fp32.grad, atol=3e-2, rtol=3e-2)


def test_block_cholesky_linear_supports_non_power_of_two_dim_384() -> None:
    """Verify `BlockCholeskyLinear` supports `d = 384` and `coarse_dim = 48` via GCD block sizing."""
    chol_48 = BlockCholeskyLinear(48, max_block_size=32, fast_spd_bmm=True)
    assert chol_48.block_size == 16
    assert chol_48.num_blocks == 3
    x = torch.randn(2, 8, 48, requires_grad=True)
    y = chol_48(x)
    assert y.shape == (2, 8, 48)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dyadic_hierarchical_born_forward_and_backward_d384() -> None:
    """Verify `Radical2Pow16WaveletCholeskyGPT` at `embed_dim = 384` with `use_dyadic_hierarchical_born=True`."""
    torch.manual_seed(2041)
    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        vocab_size=512,
        valid_vocab_size=500,
        ambient_n=1024,
        num_layers=2,
        embed_dim=384,
        num_heads=6,
        block_size=32,
        window_size=32,
        fine_detail_dim=32,
        mid_detail_dim=48,
        diffuse_rank=4,
        interferometer_expansion=1.5,
        use_fast_cholesky_and_compile=True,
        use_convex_traceless_born=True,
        traceless_gamma=0.25,
        use_dyadic_hierarchical_born=True,
        dyadic_coarse_alpha=0.22,
        dyadic_coarse_ce_weight=0.08,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg)
    with torch.no_grad():
        torch.nn.init.normal_(model.pyramid.wav_256, std=0.15)
        torch.nn.init.normal_(model.pyramid.wav_2048, std=0.15)
        torch.nn.init.normal_(model.pyramid.wav_8192, std=0.15)
        torch.nn.init.normal_(model.pyramid.wav_65536, std=0.15)

    x = torch.randint(0, 500, (2, 16))
    y = torch.randint(0, 500, (2, 16))
    _, loss = model(x, targets=y, return_logits=False)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.pyramid.wav_256.grad is not None and torch.isfinite(model.pyramid.wav_256.grad).all()
    assert model.pyramid.wav_65536.grad is not None and torch.isfinite(model.pyramid.wav_65536.grad).all()


def test_direct_chunked_ce_gemm_precision_modes() -> None:
    """Verify `direct_chunked_two_beam_kboson_ce` runs cleanly with `bf16`, `fp8`, and `int8` modes."""
    torch.manual_seed(42)
    psi_s = l2_normalize(torch.randn(32, 32, requires_grad=True), dim=-1)
    psi_d = l2_normalize(torch.randn(32, 32, requires_grad=True), dim=-1)
    m_s = l2_normalize(torch.randn(64, 32, requires_grad=True), dim=-1)
    m_d = l2_normalize(torch.randn(64, 32, requires_grad=True), dim=-1)
    beta_t = torch.full((32, 1), 12.0, requires_grad=True)
    k_val = torch.tensor(16.0, requires_grad=True)
    lam_val = torch.tensor(0.12, requires_grad=True)
    targets = torch.randint(0, 60, (32,))

    for prec in ("bf16", "fp8", "int8"):
        loss = direct_chunked_two_beam_kboson_ce(
            psi_sharp=psi_s,
            psi_diff=psi_d,
            m_sharp=m_s,
            m_diff=m_d,
            beta_t=beta_t,
            k_copies=k_val,
            lambda_diff=lam_val,
            targets=targets,
            chunk_size=16,
            valid_vocab_size=60,
            quadratic_diffuse_fidelity=0.5,
            traceless_gamma=-0.25,
            gemm_precision=prec,
        )
        assert torch.isfinite(loss)


def test_rtx5060ti_15gb_batch24576_sizer_artifact_schema() -> None:
    """Verify `.cache/rtx5060ti_15gb_batch24576_network_sizer.json` exists and passes all gates."""
    sizer_path = ROOT / ".cache" / "rtx5060ti_15gb_batch24576_network_sizer.json"
    assert sizer_path.exists(), f"Missing {sizer_path}"
    data = json.loads(sizer_path.read_text(encoding="utf-8"))
    dtypes = data["activation_dtypes"]
    for key in ("psi_in", "bigram_psi", "v_0", "psi_blk", "diff_blk", "m_sharp", "m_diff"):
        assert dtypes[key] == "torch.bfloat16", f"{key} had dtype {dtypes[key]}"

    candidates = {c["name"]: c for c in data["candidates"]}
    assert len(candidates) >= 10
    for name, c in candidates.items():
        assert c["status"] == "OK", f"{name} failed with {c['status']}"
        assert c["peak_vram_gb"] <= 15.0, f"{name} exceeded 15.0 GB VRAM: {c['peak_vram_gb']}"
    assert candidates["w26_b48_d256_L6_dyadic_fp8"]["tok_per_s"] > candidates["w26_b48_d256_L6_dyadic_bf16"]["tok_per_s"]
    assert candidates["w26_b48_d256_L6_dyadic_bf16"]["tok_per_s"] > candidates["w26_b48_d256_L6_dyadic_int8"]["tok_per_s"]
