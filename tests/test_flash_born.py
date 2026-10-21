"""Unit tests for the custom Triton Flash-Born Attention forward and backward kernels."""

from __future__ import annotations

import copy

import pytest
import torch

from qdlblocks import HAS_TRITON, SubspaceProjectorBornAttention, l2_normalize


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton and CUDA required for Flash-Born kernel tests",
)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("learnable_temp", [False, True])
def test_flash_born_matches_reference_forward_and_backward(
    is_causal: bool, learnable_temp: bool
) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, D, H = 2, 64, 128, 4  # head_dim = 32 (power of 2 >= 16)
    attn_ref = SubspaceProjectorBornAttention(
        embed_dim=D,
        num_heads=H,
        block_size=T,
        is_causal=is_causal,
        learnable_temperature=learnable_temp,
        value_residual_mixing=True,
        use_flash_born=False,
    ).to(device)

    attn_flash = copy.deepcopy(attn_ref)
    attn_flash.use_flash_born = True

    x_ref = l2_normalize(torch.randn(B, T, D, device=device), dim=-1).detach().requires_grad_(True)
    psi0_ref = l2_normalize(torch.randn(B, T, D, device=device), dim=-1).detach()

    x_flash = x_ref.detach().clone().requires_grad_(True)
    psi0_flash = psi0_ref.detach().clone()

    out_ref = attn_ref(x_ref, psi_0=psi0_ref)
    out_flash = attn_flash(x_flash, psi_0=psi0_flash)

    assert torch.allclose(out_flash, out_ref, atol=1e-4, rtol=1e-3), (
        f"Max forward diff: {(out_flash - out_ref).abs().max().item()}"
    )

    grad_out = torch.randn_like(out_ref)
    (out_ref * grad_out).sum().backward()
    (out_flash * grad_out).sum().backward()

    assert x_ref.grad is not None and x_flash.grad is not None
    assert torch.allclose(x_flash.grad, x_ref.grad, atol=2e-3, rtol=1e-2), (
        f"Max dX diff: {(x_flash.grad - x_ref.grad).abs().max().item()}"
    )

    for (n1, p1), (n2, p2) in zip(
        attn_ref.named_parameters(), attn_flash.named_parameters(), strict=True
    ):
        assert p1.grad is not None and p2.grad is not None, f"Missing grad for {n1}"
        assert torch.allclose(p2.grad, p1.grad, atol=3e-3, rtol=2e-2), (
            f"Max grad diff for {n1}: {(p2.grad - p1.grad).abs().max().item()}"
        )


def test_flash_born_rdna4_and_cuda_launch_metadata() -> None:
    from qdlblocks.kernels import get_flash_born_launch_meta, get_rdna4_autotune_configs

    hip_meta = get_flash_born_launch_meta(64, is_hip=True)
    assert hip_meta["num_warps"] == 4
    assert hip_meta["waves_per_eu"] == 2

    cuda_meta = get_flash_born_launch_meta(64, is_hip=False)
    assert cuda_meta["num_warps"] == 4
    assert "waves_per_eu" not in cuda_meta

    cfgs = get_rdna4_autotune_configs()
    assert len(cfgs) == 4
    assert all(c["matrix_instr_nonkdim"] == 16 and c["waves_per_eu"] in (2, 4) for c in cfgs)


