"""Cross-Layer First-Layer Value Residual ($V^{(0)}$) & Cubic ReLU^2 Norm Control Diagnostics (Plan 015).

Provides:
1. `compute_cross_layer_v_orthogonality_metrics`: Measures the orthogonal subspace energy
   of Layer-l value states $V^{(l)}$ relative to the same-layer column span
   $\\text{span}(W_V^{(l)} x_{\\text{in}}^{(l)}, W_V^{(l)} x_0)$.
2. `compute_mlp_attn_homogeneity_metrics`: Verifies degree-2 quadratic scaling of QK-RMSNorm
   attention ($\\propto c^2$) vs. degree-3 cubic scaling of $\\text{ReLU}^2$ MLP ($\\propto c^3$)
   under weight norm inflation $W \\mapsto c W$.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import apply_unitary_rope

__all__ = [
    "compute_cross_layer_v_orthogonality_metrics",
    "compute_mlp_attn_homogeneity_metrics",
]


@torch.no_grad()
def compute_cross_layer_v_orthogonality_metrics(
    v_mixed: torch.Tensor,
    x_in: torch.Tensor,
    x_0: torch.Tensor,
    w_v_layer: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute the orthogonal residual ratio of `v_mixed` relative to Layer $l$'s same-layer span.

    Projects `v_mixed` (flattened to `(B * T, D)`) onto the subspace spanned for each token by
    $U_1 = x_{\\text{in}} W_V^{(l)\\top}$ and $U_2 = x_0 W_V^{(l)\\top}$.

    Under same-layer value residual mixing ($V_{\\text{same}}^{(l)} = (1 - \\alpha) W_V^{(l)} x_{\\text{in}}^{(l)} + \\alpha W_V^{(l)} x_0$),
    $V_{\\text{same}}^{(l)}$ lies strictly inside $\\text{span}(U_1, U_2)$ (`orth_residual_ratio < 1e-5`).
    Under true cross-layer first-layer value residual mixing ($V_{\\text{cross}}^{(l)} = (1 - \\alpha) W_V^{(l)} x_{\\text{in}}^{(l)} + \\alpha V^{(0)}$),
    $V^{(0)} = W_V^{(0)} x_{\\text{in}}^{(0)}$ introduces genuine orthogonal subspace energy outside
    $\\text{span}(U_1, U_2)$ (`orth_residual_ratio >= 0.15`).

    Args:
        v_mixed: Mixed value tensor of shape `(B, H, T, D_h)`, `(B, T, D)`, or `(B * T, D)`.
        x_in: Layer-$l$ input hidden states of shape `(B, T, D)` or `(B * T, D)`.
        x_0: Initial Layer-0 token states of shape `(B, T, D)` or `(B * T, D)`.
        w_v_layer: Layer-$l$ value projection matrix `v_proj.weight` of shape `(D, D)`.
        eps: Numerical floor for safe normalization.

    Returns:
        Dictionary with `orth_residual_ratio`, `parallel_energy_ratio`, and `total_v_rms`.
    """
    w_f = w_v_layer.detach().float()
    d_model = int(w_f.shape[0])
    x_in_2d = x_in.detach().float().reshape(-1, d_model)
    x_0_2d = x_0.detach().float().reshape(-1, d_model)
    n_tokens = int(x_in_2d.shape[0])

    v_f = v_mixed.detach().float()
    if v_f.ndim == 4:
        # Standard attention layout is (B, H, T, D_h); transpose to (B, T, H, D_h) before flattening
        if x_in.ndim == 3 and v_f.shape[1] == x_in.shape[1] and v_f.shape[2] != x_in.shape[1]:
            v_2d = v_f.contiguous().reshape(n_tokens, d_model)
        else:
            v_2d = v_f.transpose(1, 2).contiguous().reshape(n_tokens, d_model)
    else:
        v_2d = v_f.reshape(n_tokens, d_model)

    u1 = F.linear(x_in_2d, w_f)  # (N, D)
    u2 = F.linear(x_0_2d, w_f)  # (N, D)

    # Orthonormalize {u1, u2} per token via Gram-Schmidt in R^D
    u1_norm = torch.norm(u1, p=2, dim=-1, keepdim=True)
    e1 = u1 / (u1_norm + eps)

    proj_u2_on_e1 = torch.sum(u2 * e1, dim=-1, keepdim=True) * e1
    u2_orth = u2 - proj_u2_on_e1
    u2_orth_norm = torch.norm(u2_orth, p=2, dim=-1, keepdim=True)
    e2 = torch.where(u2_orth_norm > eps, u2_orth / (u2_orth_norm + eps), torch.zeros_like(u2_orth))

    v_par = torch.sum(v_2d * e1, dim=-1, keepdim=True) * e1 + torch.sum(v_2d * e2, dim=-1, keepdim=True) * e2
    v_orth = v_2d - v_par

    v_fro = float(torch.norm(v_2d, p="fro").item())
    v_orth_fro = float(torch.norm(v_orth, p="fro").item())
    v_par_fro = float(torch.norm(v_par, p="fro").item())

    orth_residual_ratio = v_orth_fro / (v_fro + eps)
    parallel_energy_ratio = (v_par_fro * v_par_fro) / (v_fro * v_fro + eps)
    total_v_rms = float(torch.sqrt(torch.mean(v_2d * v_2d) + eps).item())

    return {
        "orth_residual_ratio": float(orth_residual_ratio),
        "parallel_energy_ratio": float(parallel_energy_ratio),
        "total_v_rms": float(total_v_rms),
    }


@torch.no_grad()
def compute_mlp_attn_homogeneity_metrics(
    block: nn.Module,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale_factor: float = 4.0,
) -> dict[str, float]:
    """Evaluate attention ($c^2$) vs. $\\text{ReLU}^2$ MLP ($c^3$) homogeneity under weight scaling $c$.

    In `ClassicalModdedBlock`:
    - Attention (`SDPA(RMSNorm(Q), RMSNorm(K), V) @ W_O^T`) is homogeneous of degree 2 in
      $(W_Q, W_K, W_V, W_O)$ because QK-RMSNorm is scale-invariant ($c^0$) while $W_V$ and $W_O$
      are each linear ($c^1 \\cdot c^1 = c^2$).
    - $\\text{ReLU}^2$ MLP ($\\text{ReLU}(h W_1^T)^2 W_2^T$) is homogeneous of degree 3 in
      $(W_1, W_2)$ ($c^2 \\cdot c^1 = c^3$).

    Args:
        block: `ClassicalModdedBlock` instance containing `q_proj`, `k_proj`, `v_proj`, `out_proj`,
            `fc1`, and `fc2`.
        x: Input hidden state tensor of shape `(B, T, D)`.
        cos: Unitary RoPE cosine table `(1, 1, T, D_h // 2)`.
        sin: Unitary RoPE sine table `(1, 1, T, D_h // 2)`.
        scale_factor: Scalar multiplier $c > 0$ applied to all 2D block weights (default: `4.0`).

    Returns:
        Dictionary with `attn_scale_ratio`, `mlp_scale_ratio`, `dominance_growth_ratio`,
        `rho_mlp_attn_base`, and `rho_mlp_attn_scaled`.
    """
    c = float(scale_factor)
    x_f = x.detach().float()
    cos_f = cos.detach().float()
    sin_f = sin.detach().float()
    b_sz, t_sz, d_model = x_f.shape
    num_heads = int(getattr(block, "num_heads", 4))
    head_dim = d_model // num_heads
    eps = float(getattr(block, "eps", 1e-6))

    w_q = block.q_proj.weight.detach().float()  # type: ignore[union-attr]
    w_k = block.k_proj.weight.detach().float()  # type: ignore[union-attr]
    w_v = block.v_proj.weight.detach().float()  # type: ignore[union-attr]
    w_o = block.out_proj.weight.detach().float()  # type: ignore[union-attr]
    w_1 = block.fc1.weight.detach().float()  # type: ignore[union-attr]
    w_2 = block.fc2.weight.detach().float()  # type: ignore[union-attr]

    # If block was initialized with zero_init_residual=True, use deterministic orthogonal probe weights
    # for w_o / w_2 so homogeneity ratios are well-defined.
    if float(torch.norm(w_o).item()) < 1e-12:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            w_o = torch.empty_like(w_o, device="cpu")
            nn.init.orthogonal_(w_o, gain=0.5)
            w_o = w_o.to(x_f.device)
    if float(torch.norm(w_2).item()) < 1e-12:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            w_2 = torch.empty_like(w_2, device="cpu")
            nn.init.orthogonal_(w_2, gain=0.5)
            w_2 = w_2.to(x_f.device)

    h = F.rms_norm(x_f, (d_model,), eps=eps)

    def _eval_at_scale(scale: float) -> tuple[float, float, float]:
        q = F.linear(h, scale * w_q).view(b_sz, t_sz, num_heads, head_dim).transpose(1, 2)
        k = F.linear(h, scale * w_k).view(b_sz, t_sz, num_heads, head_dim).transpose(1, 2)
        v = F.linear(h, scale * w_v).view(b_sz, t_sz, num_heads, head_dim).transpose(1, 2)

        q = apply_unitary_rope(q, cos_f, sin_f)
        k = apply_unitary_rope(k, cos_f, sin_f)
        # Scale-invariant unit RMSNorm on Q and K
        q = F.rms_norm(q, (head_dim,), eps=1e-12)
        k = F.rms_norm(k, (head_dim,), eps=1e-12)

        sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        sdpa_flat = sdpa.transpose(1, 2).contiguous().view(b_sz, t_sz, d_model)
        attn_out = F.linear(sdpa_flat, scale * w_o)

        r_mlp = F.relu(F.linear(h, scale * w_1))
        mlp_out = F.linear(r_mlp.square(), scale * w_2)

        attn_norm = float(torch.norm(attn_out, p="fro").item())
        mlp_norm = float(torch.norm(mlp_out, p="fro").item())
        rho = mlp_norm / max(attn_norm, 1e-12)
        return attn_norm, mlp_norm, rho

    attn_1, mlp_1, rho_1 = _eval_at_scale(1.0)
    attn_c, mlp_c, rho_c = _eval_at_scale(c)

    attn_scale_ratio = attn_c / max(attn_1, 1e-12)
    mlp_scale_ratio = mlp_c / max(mlp_1, 1e-12)
    dominance_growth_ratio = rho_c / max(rho_1, 1e-12)

    return {
        "attn_scale_ratio": float(attn_scale_ratio),
        "mlp_scale_ratio": float(mlp_scale_ratio),
        "dominance_growth_ratio": float(dominance_growth_ratio),
        "rho_mlp_attn_base": float(rho_1),
        "rho_mlp_attn_scaled": float(rho_c),
    }
