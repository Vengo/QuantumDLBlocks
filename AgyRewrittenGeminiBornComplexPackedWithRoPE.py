import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PART 1: NATIVE COMPLEX ROTARY POSITION EMBEDDINGS (ComplexRoPE)
# ============================================================================
class ComplexRoPE(nn.Module):
    """
    Native Complex Rotary Position Embeddings for complex registers z = z_re + i * z_im.
    
    Unlike standard real RoPE (which must pair up adjacent real numbers (x_2m, x_2m+1)),
    every coordinate a in {0, ..., d_h - 1} of our register is ALREADY a complex number!
    So each coordinate a gets its own rotary frequency omega_a = base^(-a / d_h) and
    rotates in the complex plane by e^(i * t * omega_a) = cos(t * omega_a) + i * sin(t * omega_a).
    """
    def __init__(self, d_register: int = 32, base: float = 10000.0):
        super().__init__()
        # Each of the 32 complex coordinates gets its own frequency omega_a
        inv_freq = 1.0 / (base ** (torch.arange(0, d_register, dtype=torch.float32) / d_register))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, z_re: torch.Tensor, z_im: torch.Tensor):
        """
        Args:
            z_re, z_im: Real and Imaginary register tensors of shape (B, H, T, d_h)
        Returns:
            rot_re, rot_im: Phase-rotated tensors z * e^(i * t * omega_a) of shape (B, H, T, d_h)
        """
        T = z_re.size(2)
        positions = torch.arange(T, device=z_re.device, dtype=self.inv_freq.dtype)
        # Outer product of token position t and frequency omega_a -> angles (T, d_h)
        angles = torch.outer(positions, self.inv_freq)  # (T, d_h)

        # Broadcast across (B, H, T, d_h)
        cos_theta = torch.cos(angles).unsqueeze(0).unsqueeze(0).to(z_re.dtype)
        sin_theta = torch.sin(angles).unsqueeze(0).unsqueeze(0).to(z_re.dtype)

        # Complex multiplication: (z_re + i z_im) * (cos + i sin)
        #   Real part      = z_re * cos - z_im * sin
        #   Imaginary part = z_re * sin + z_im * cos
        rot_re = z_re * cos_theta - z_im * sin_theta
        rot_im = z_re * sin_theta + z_im * cos_theta
        return rot_re, rot_im


# ============================================================================
# PART 2: FUSED HERMITIAN TRIANGLE PACKING & UNPACKING (REFINEMENTS 3 & 4)
# ============================================================================
def pack_hermitian_outer_product(v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
    """
    Forms the rank-1 Hermitian density matrix rho_s = |v_s><v_s| = rho_re + i * rho_im
    for each token and packs BOTH parts into a SINGLE real (d_h x d_h) matrix P_s
    using a single `torch.where` selection (Refinement 3):

      - Upper Triangle + Diagonal (i <= j): stores Symmetric rho_re[i, j]
      - Strict Lower Triangle     (i >  j): stores Skew-Symmetric rho_im[i, j]

    Why zero information is lost:
      1. rho_re is symmetric (rho_re[i, j] == +rho_re[j, i]), so its lower triangle is redundant.
      2. rho_im is skew-symmetric (rho_im[i, j] == -rho_im[j, i]) and its diagonal is 0
         (since v_im[i]*v_re[i] - v_re[i]*v_im[i] == 0), so its diagonal and upper triangle
         are redundant.
      3. Together, 528 upper-real + 496 lower-imag entries = 1,024 = 32 x 32 entries!

    Args:
        v_re, v_im: Real and Imaginary Value kets of shape (B, H, T, d_h)
    Returns:
        packed_flat: Packed real tensor of shape (B, H, T, d_h * d_h) ready for 1-GEMM mixing.
    """
    B, H, T, dh = v_re.shape

    # 1. Compute real symmetric and imaginary skew-symmetric outer products:
    #    rho_re[i, j] = v_re[i]*v_re[j] + v_im[i]*v_im[j]  (Symmetric)
    #    rho_im[i, j] = v_im[i]*v_re[j] - v_re[i]*v_im[j]  (Skew-Symmetric, zero diagonal)
    rho_re = v_re.unsqueeze(-1) * v_re.unsqueeze(-2) + v_im.unsqueeze(-1) * v_im.unsqueeze(-2)
    rho_im = v_im.unsqueeze(-1) * v_re.unsqueeze(-2) - v_re.unsqueeze(-1) * v_im.unsqueeze(-2)

    # 2. Single-pass triangle selection (fuses into 1 pointwise kernel under @torch.compile)
    idx = torch.arange(dh, device=v_re.device)
    upper_mask = (idx[:, None] <= idx[None, :]).view(1, 1, 1, dh, dh)
    packed = torch.where(upper_mask, rho_re, rho_im)
    return packed.reshape(B, H, T, dh * dh)


def pack_hermitian_outer_product_and_ket(v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
    """
    Refinements 3 & 4 Combined:
    Packs the (dh * dh = 1024) Hermitian outer product |v_s><v_s| AND the (2 * dh = 64)
    first-moment Value ket [v_re, v_im] into a single real tensor of width 1088 = 17 * 64
    so ONE Tensor-Core GEMM computes both rho_t and |v_bar_t>!
    """
    return torch.cat([pack_hermitian_outer_product(v_re, v_im), v_re, v_im], dim=-1)


def unpack_hermitian_matrix(packed_flat: torch.Tensor, dh: int):
    """
    Unpacks a mixed packed tensor P_t of shape (B, H, T, dh * dh) back into
    the full Symmetric Real matrix rho_re and Skew-Symmetric Imaginary matrix rho_im.

    CRITICAL BUG-PREVENTION NOTE ON THE DIAGONAL:
      Using `torch.where(upper_mask, P, P^T)` takes P[i, j] on the upper triangle AND diagonal (i <= j)
      and P^T[i, j] = P[j, i] strictly below the diagonal (i > j), so the diagonal R[i, i] is NEVER doubled,
      and `rho_im` has an exact 0.0 diagonal!
    """
    B, H, T, _ = packed_flat.shape
    P = packed_flat.view(B, H, T, dh, dh)
    P_t = P.transpose(-2, -1)

    idx = torch.arange(dh, device=packed_flat.device)
    upper_mask = (idx[:, None] <= idx[None, :]).view(1, 1, 1, dh, dh)
    strict_lower_mask = (idx[:, None] > idx[None, :]).view(1, 1, 1, dh, dh)
    strict_upper_mask = (idx[:, None] < idx[None, :]).view(1, 1, 1, dh, dh)
    zero = torch.zeros(1, device=packed_flat.device, dtype=packed_flat.dtype)

    # 1. Reconstruct Symmetric Real Matrix (i <= j from P, i > j from P^T)
    rho_re = torch.where(upper_mask, P, P_t)

    # 2. Reconstruct Skew-Symmetric Imaginary Matrix (i > j from P, i < j from -P^T, i == j is 0)
    rho_im = torch.where(strict_lower_mask, P, torch.where(strict_upper_mask, -P_t, zero))

    return rho_re, rho_im


# ============================================================================
# PART 3: FULL COMPLEX HERMITIAN ATTENTION WITH ROPE + 1-GEMM PACKING + V-SPACE KRYLOV PROBE
# ============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        z = torch.complex(torch.randn(out_features, in_features), torch.randn(out_features, in_features))
        q, r = torch.linalg.qr(z)
        q = q * torch.sgn(torch.diagonal(r, 0)).unsqueeze(0)
        with torch.no_grad():
            self.weight_re.copy_(q.real)
            self.weight_im.copy_(q.imag)

    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor):
        out_re = F.linear(x_re, self.weight_re) - F.linear(x_im, self.weight_im)
        out_im = F.linear(x_re, self.weight_im) + F.linear(x_im, self.weight_re)
        return out_re, out_im


def complex_l2_normalize(x_re: torch.Tensor, x_im: torch.Tensor, dim: int = -1, eps: float = 1e-7):
    inv_norm = torch.rsqrt(x_re.square().sum(dim=dim, keepdim=True) + x_im.square().sum(dim=dim, keepdim=True) + eps)
    return x_re * inv_norm, x_im * inv_norm


class PackedComplexRegisterDensityAttention(nn.Module):
    """
    Complex Hermitian (512, 32, 32, 32) Register Density Attention with Refinements 1, 3, and 4 (Zero gamma_coh):
      1. Refinement 1: Native ComplexRoPE applied strictly to Q and K for relative-position shift-invariant
         Born weights w_ts, leaving V un-rotated in position-invariant content V-space.
      2. Refinement 3: Single-allocation Hermitian Triangle Packing + 1-GEMM mixing of width 1088 = 1024 + 64.
      3. Refinement 4: V-Space Krylov Power Iteration probing of rho_t^purified = rho_t^2 / Tr(rho_t^2)
         using the normalized first-moment V-space anchor ket |v_bar_t> = normalize(sum_{s<=t} w_ts |v_s>).
    """
    def __init__(self, d_model: int = 1024, n_registers: int = 32, d_register: int = 32, p_boson: int = 2):
        super().__init__()
        self.H = n_registers
        self.dh = d_register
        self.p_boson = p_boson

        self.rope = ComplexRoPE(d_register=d_register)
        self.W_q = ComplexLinear(d_model, d_model)
        self.W_k = ComplexLinear(d_model, d_model)
        self.W_v = ComplexLinear(d_model, d_model)
        self.W_out = ComplexLinear(d_model, d_model)

    def forward(self, psi_re: torch.Tensor, psi_im: torch.Tensor, eps: float = 1e-7):
        B, T, D = psi_re.shape
        H, dh = self.H, self.dh

        # 1. Project to complex Q, K, V registers and normalize each register to unit complex norm
        q_re, q_im = self.W_q(psi_re, psi_im)
        k_re, k_im = self.W_k(psi_re, psi_im)
        v_re, v_im = self.W_v(psi_re, psi_im)

        q_re, q_im = complex_l2_normalize(q_re.view(B, T, H, dh).transpose(1, 2),
                                          q_im.view(B, T, H, dh).transpose(1, 2), dim=-1, eps=eps)
        k_re, k_im = complex_l2_normalize(k_re.view(B, T, H, dh).transpose(1, 2),
                                          k_im.view(B, T, H, dh).transpose(1, 2), dim=-1, eps=eps)
        v_re, v_im = complex_l2_normalize(v_re.view(B, T, H, dh).transpose(1, 2),
                                          v_im.view(B, T, H, dh).transpose(1, 2), dim=-1, eps=eps)

        # 2. Refinement 1: Apply Native Complex RoPE strictly to Q and K for the Step 3 inner product.
        #    V is NOT rotated by RoPE so Value states remain position-invariant content kets in V-space.
        q_rot_re, q_rot_im = self.rope(q_re, q_im)
        k_rot_re, k_rot_im = self.rope(k_re, k_im)

        # 3. Hermitian Inner Product <q_t^rot | k_s^rot>_C (depends strictly on relative distance t - s)
        c_re = torch.matmul(q_rot_re, k_rot_re.transpose(-2, -1)) + torch.matmul(q_rot_im, k_rot_im.transpose(-2, -1))
        c_im = torch.matmul(q_rot_re, k_rot_im.transpose(-2, -1)) - torch.matmul(q_rot_im, k_rot_re.transpose(-2, -1))
        mag_sq = c_re.square() + c_im.square()

        # 4. Softmax-Free p-Boson Probability Weights w_ts >= 0, sum_{s<=t} w_ts = 1 (Zero gamma_coh)
        causal_mask = torch.tril(torch.ones(T, T, device=psi_re.device, dtype=torch.bool))
        intensity = mag_sq.pow(self.p_boson).masked_fill(~causal_mask, 0.0)
        w_born = intensity / (intensity.sum(dim=-1, keepdim=True) + eps)

        # 5. Refinements 3 & 4: Pack |v_s><v_s| (1024) + [v_re, v_im] (64) into width 1088 = 17 * 64
        v_packed_and_ket = pack_hermitian_outer_product_and_ket(v_re, v_im)

        #    ONE real Tensor-Core GEMM computes BOTH rho_t AND the first-moment V-space anchor ket |v_bar_t>!
        mixed_all = torch.matmul(w_born, v_packed_and_ket)
        d_sq = dh * dh
        rho_packed_mixed = mixed_all[..., :d_sq]
        v_bar_re, v_bar_im = complex_l2_normalize(
            mixed_all[..., d_sq : d_sq + dh],
            mixed_all[..., d_sq + dh :],
            dim=-1,
            eps=eps,
        )

        #    Unpack back into full (B, 32, 512, 32, 32) Symmetric Real & Skew-Symmetric Imaginary matrices
        rho_re, rho_im = unpack_hermitian_matrix(rho_packed_mixed, dh=dh)

        # 6. Complex McWeeny Matrix-Squaring Purification: rho_purified = (rho @ rho) / Tr(rho^2)
        sq_re = torch.matmul(rho_re, rho_re) - torch.matmul(rho_im, rho_im)
        sq_im = torch.matmul(rho_re, rho_im) + torch.matmul(rho_im, rho_re)
        tr_sq = sq_re.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        rho_re = sq_re / (tr_sq + eps)
        rho_im = sq_im / (tr_sq + eps)

        # 7. Refinement 4: Probe rho_t^purified in V-Space with |v_bar_t> (Krylov Power Iteration)
        v_bar_re_col, v_bar_im_col = v_bar_re.unsqueeze(-1), v_bar_im.unsqueeze(-1)
        u_re = (torch.matmul(rho_re, v_bar_re_col) - torch.matmul(rho_im, v_bar_im_col)).squeeze(-1)
        u_im = (torch.matmul(rho_re, v_bar_im_col) + torch.matmul(rho_im, v_bar_re_col)).squeeze(-1)

        out_re, out_im = self.W_out(
            u_re.transpose(1, 2).contiguous().view(B, T, D),
            u_im.transpose(1, 2).contiguous().view(B, T, D),
        )
        return complex_l2_normalize(psi_re + out_re, psi_im + out_im, dim=-1, eps=eps)


# ============================================================================
# SELF-CONTAINED DEBUGGER & VERIFICATION SUITE
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(7)
    B, H, T, dh = 1, 32, 64, 32

    # 1. Verify Hermitian Triangle Packing + Unpacking has 0.00e+00 error against 2-GEMM Reference
    v_re, v_im = complex_l2_normalize(torch.randn(B, H, T, dh), torch.randn(B, H, T, dh), dim=-1)
    w_born = torch.softmax(torch.randn(B, H, T, T), dim=-1).tril()
    w_born = w_born / (w_born.sum(dim=-1, keepdim=True) + 1e-8)

    # Reference 2-GEMM Unpacked Path
    ref_outer_re = (v_re.unsqueeze(-1) * v_re.unsqueeze(-2) + v_im.unsqueeze(-1) * v_im.unsqueeze(-2)).view(B, H, T, dh * dh)
    ref_outer_im = (v_im.unsqueeze(-1) * v_re.unsqueeze(-2) - v_re.unsqueeze(-1) * v_im.unsqueeze(-2)).view(B, H, T, dh * dh)
    ref_rho_re = torch.matmul(w_born, ref_outer_re).view(B, H, T, dh, dh)
    ref_rho_im = torch.matmul(w_born, ref_outer_im).view(B, H, T, dh, dh)

    # Fast 1-GEMM Packed Path (with fused V-bar ket in width 1088)
    packed_and_ket = pack_hermitian_outer_product_and_ket(v_re, v_im)
    mixed_all = torch.matmul(w_born, packed_and_ket)
    unpacked_re, unpacked_im = unpack_hermitian_matrix(mixed_all[..., : dh * dh], dh=dh)

    err_re = (ref_rho_re - unpacked_re).abs().max().item()
    err_im = (ref_rho_im - unpacked_im).abs().max().item()
    print(f"[Check 1] Packing/Unpacking Max Real Error : {err_re:.2e} (Expected: 0.00e+00)")
    print(f"[Check 2] Packing/Unpacking Max Imag Error : {err_im:.2e} (Expected: 0.00e+00)")

    # 2. Verify ComplexRoPE Relative-Distance Shift Invariance: <q_{t+delta} | k_{s+delta}> == <q_t | k_s>
    rope = ComplexRoPE(d_register=dh)
    q_single_re, q_single_im = complex_l2_normalize(torch.randn(1, 1, 1, dh), torch.randn(1, 1, 1, dh))
    k_single_re, k_single_im = complex_l2_normalize(torch.randn(1, 1, 1, dh), torch.randn(1, 1, 1, dh))

    q_seq_re, q_seq_im = rope(q_single_re.expand(1, 1, 20, dh), q_single_im.expand(1, 1, 20, dh))
    k_seq_re, k_seq_im = rope(k_single_re.expand(1, 1, 20, dh), k_single_im.expand(1, 1, 20, dh))

    c_re = torch.matmul(q_seq_re, k_seq_re.transpose(-2, -1)) + torch.matmul(q_seq_im, k_seq_im.transpose(-2, -1))
    c_im = torch.matmul(q_seq_re, k_seq_im.transpose(-2, -1)) - torch.matmul(q_seq_im, k_seq_re.transpose(-2, -1))

    shift_err_re = (c_re[0, 0, 7, 2] - c_re[0, 0, 15, 10]).abs().item()
    shift_err_im = (c_im[0, 0, 7, 2] - c_im[0, 0, 15, 10]).abs().item()
    print(f"[Check 3] ComplexRoPE Shift-Invariance Err : Re={shift_err_re:.2e}, Im={shift_err_im:.2e}")

    # 3. Verify Refinement 4 (V-Space Krylov Probe vs Q-Space Probe at Step 0 when 1 token dominates)
    q_rand_re, q_rand_im = complex_l2_normalize(torch.randn(B, H, 1, dh), torch.randn(B, H, 1, dh), dim=-1)
    vs_re, vs_im = v_re[:, :, :1, :], v_im[:, :, :1, :]
    rho1_re, rho1_im = unpack_hermitian_matrix(pack_hermitian_outer_product(vs_re, vs_im), dh=dh)

    # Probe with |q_t> (random Q-space ket): norm shrinks by ~1/sqrt(32) = 0.177 and phase scrambles
    uq_re = (torch.matmul(rho1_re, q_rand_re.unsqueeze(-1)) - torch.matmul(rho1_im, q_rand_im.unsqueeze(-1))).squeeze(-1)
    uq_im = (torch.matmul(rho1_re, q_rand_im.unsqueeze(-1)) + torch.matmul(rho1_im, q_rand_re.unsqueeze(-1))).squeeze(-1)
    q_probe_norm = (uq_re.square().sum(dim=-1) + uq_im.square().sum(dim=-1)).sqrt().mean().item()

    # Probe with |v_bar_t> = |v_s> (V-space Krylov anchor ket): exact overlap 1.0 + 0.0i and unit norm 1.0000
    uv_re = (torch.matmul(rho1_re, vs_re.unsqueeze(-1)) - torch.matmul(rho1_im, vs_im.unsqueeze(-1))).squeeze(-1)
    uv_im = (torch.matmul(rho1_re, vs_im.unsqueeze(-1)) + torch.matmul(rho1_im, vs_re.unsqueeze(-1))).squeeze(-1)
    v_probe_norm = (uv_re.square().sum(dim=-1) + uv_im.square().sum(dim=-1)).sqrt().mean().item()
    v_overlap_re = (vs_re * uv_re + vs_im * uv_im).sum(dim=-1).mean().item()
    v_overlap_im = (vs_re * uv_im - vs_im * uv_re).sum(dim=-1).abs().max().item()
    print(f"[Check 4] Single-Token Step-0 Probe Norm   : Q-Probe={q_probe_norm:.4f} (~1/sqrt(32)) vs V-Krylov={v_probe_norm:.4f} (Overlap={v_overlap_re:.4f} + {v_overlap_im:.2e}i)")
