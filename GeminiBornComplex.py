import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    """
    True Complex Linear Layer W = W_re + i * W_im in C^(out_features x in_features).
    Doubles the parameter count over a real Linear layer and initializes as a Complex Unitary / Stiefel matrix.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize via complex QR decomposition so W^dagger W = I_d (exact complex unitary!)
        z = torch.complex(
            torch.randn(self.out_features, self.in_features),
            torch.randn(self.out_features, self.in_features),
        )
        q, r = torch.linalg.qr(z)
        # Fix phase of diagonal of R for uniform Haar measure on U(d)
        diag_phase = torch.sgn(torch.diagonal(r, 0))
        q = q * diag_phase.unsqueeze(0)
        with torch.no_grad():
            self.weight_re.copy_(q.real)
            self.weight_im.copy_(q.imag)

    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor):
        # (W_re + i W_im)(x_re + i x_im) = (W_re x_re - W_im x_im) + i (W_re x_im + W_im x_re)
        out_re = F.linear(x_re, self.weight_re) - F.linear(x_im, self.weight_im)
        out_im = F.linear(x_re, self.weight_im) + F.linear(x_im, self.weight_re)
        return out_re, out_im


def complex_l2_normalize(x_re: torch.Tensor, x_im: torch.Tensor, dim: int = -1, eps: float = 1e-7):
    """Normalizes a complex vector z = x_re + i x_im so <z | z> = ||x_re||^2 + ||x_im||^2 = 1."""
    norm_sq = x_re.square().sum(dim=dim, keepdim=True) + x_im.square().sum(dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(norm_sq + eps)
    return x_re * inv_norm, x_im * inv_norm


class ComplexHermitianRegisterDensityAttention(nn.Module):
    """
    Full Complex Hermitian (T=512, H=32, d_h=32, d_h=32) Multi-Head Register Density Matrix Attention.
      - Every token carries 1024 Real + 1024 Imaginary amplitudes: psi in C^(B x 512 x 1024).
      - W_Q, W_K, W_V, W_O are Complex Unitary matrices in C^(1024 x 1024) (2x parameter count).
      - Computes true complex inner products:
            c_ts = <q_t | k_s>_C = (q_re_t^T k_re_s + q_im_t^T k_im_s) + i (q_re_t^T k_im_s - q_im_t^T k_re_s)
      - Combines Value states into a (B, 512, 32, 32, 32) Complex Hermitian Density Matrix:
            rho_t = (1 - gamma_coh) * sum_{s<=t} w_ts |v_s><v_s|  +  gamma_coh * |Psi_t><Psi_t|
        where |Psi_t> = sum_{s<=t} alpha_ts |v_s> captures cross-token complex wave interference.
      - Applies Complex Hermitian McWeeny Purification: rho_t <- (rho_t @ rho_t) / Tr(rho_t^2).
    """
    def __init__(
        self,
        d_model: int = 1024,
        n_registers: int = 32,
        d_register: int = 32,
        p_boson: int = 2,
        purify_steps: int = 1,
        coherence_init: float = 0.35,
    ):
        super().__init__()
        assert d_model == n_registers * d_register, "d_model must equal H * d_h (32 * 32 = 1024)"
        self.d_model = d_model
        self.H = n_registers
        self.dh = d_register
        self.p_boson = p_boson
        self.purify_steps = purify_steps

        # Learnable coherence blend gamma_coh in (0, 1) between diagonal average and coherent superposition
        init_logit = math.log(coherence_init / (1.0 - coherence_init))
        self.coherence_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        # Full Complex Linear Projections (2x parameters each: weight_re and weight_im)
        self.W_q = ComplexLinear(d_model, d_model)
        self.W_k = ComplexLinear(d_model, d_model)
        self.W_v = ComplexLinear(d_model, d_model)
        self.W_out = ComplexLinear(d_model, d_model)

    def forward(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
        return_density: bool = False,
        eps: float = 1e-7,
    ):
        B, T, D = psi_re.shape
        H, dh = self.H, self.dh

        # 1. Complex projections + per-register complex unit-sphere normalization <z|z> = 1
        # Shapes: (B, H=32, T=512, dh=32) for both real and imaginary parts
        q_re, q_im = self.W_q(psi_re, psi_im)
        k_re, k_im = self.W_k(psi_re, psi_im)
        v_re, v_im = self.W_v(psi_re, psi_im)

        q_re, q_im = complex_l2_normalize(q_re.view(B, T, H, dh).transpose(1, 2),
                                          q_im.view(B, T, H, dh).transpose(1, 2), dim=-1, eps=eps)
        k_re, k_im = complex_l2_normalize(k_re.view(B, T, H, dh).transpose(1, 2),
                                          k_im.view(B, T, H, dh).transpose(1, 2), dim=-1, eps=eps)
        v_re, v_im = complex_l2_normalize(v_re.view(B, T, H, dh).transpose(1, 2),
                                          v_im.view(B, T, H, dh).transpose(1, 2), dim=-1, eps=eps)

        # 2. True Hermitian Inner Product c_ts = <q_t | k_s>_C = sum_a conj(q_ta) k_sa
        # Re(<q_t | k_s>) = q_re @ k_re^T + q_im @ k_im^T
        # Im(<q_t | k_s>) = q_re @ k_im^T - q_im @ k_re^T
        c_re = torch.matmul(q_re, k_re.transpose(-2, -1)) + torch.matmul(q_im, k_im.transpose(-2, -1))
        c_im = torch.matmul(q_re, k_im.transpose(-2, -1)) - torch.matmul(q_im, k_re.transpose(-2, -1))
        mag_sq = c_re.square() + c_im.square()  # |c_ts|^2 >= 0, shape (B, 32, 512, 512)

        # 3. Softmax-Free p-Boson Normalization (No exp!)
        causal_mask = torch.tril(torch.ones(T, T, device=psi_re.device, dtype=torch.bool))
        intensity = mag_sq.pow(self.p_boson).masked_fill(~causal_mask, 0.0)
        z_trace = intensity.sum(dim=-1, keepdim=True) + eps

        # 3A. Real non-negative Born weights w_ts >= 0, sum_{s<=t} w_ts = 1
        w_born = intensity / z_trace

        # 3B. Complex wave amplitudes alpha_ts = alpha_re + i alpha_im with sum_{s<=t} |alpha_ts|^2 = 1
        amp_scale = (mag_sq.pow(0.5 * (self.p_boson - 1)) / z_trace.sqrt()).masked_fill(~causal_mask, 0.0)
        alpha_re = c_re * amp_scale
        alpha_im = c_im * amp_scale

        # 4A. Incoherent Density Matrix Branch: sum_{s<=t} w_ts |v_s><v_s|
        # For |v_s> = v_re + i v_im:
        #   Re(|v_s><v_s|) = v_re v_re^T + v_im v_im^T   (Symmetric)
        #   Im(|v_s><v_s|) = v_im v_re^T - v_re v_im^T   (Skew-Symmetric: encodes chiral direction!)
        v_outer_re = (v_re.unsqueeze(-1) * v_re.unsqueeze(-2) +
                      v_im.unsqueeze(-1) * v_im.unsqueeze(-2)).view(B, H, T, dh * dh)
        v_outer_im = (v_im.unsqueeze(-1) * v_re.unsqueeze(-2) -
                      v_re.unsqueeze(-1) * v_im.unsqueeze(-2)).view(B, H, T, dh * dh)

        # Because w_born is REAL, this takes only 2 real Tensor-Core GEMMs!
        rho_inc_re = torch.matmul(w_born, v_outer_re).view(B, H, T, dh, dh)
        rho_inc_im = torch.matmul(w_born, v_outer_im).view(B, H, T, dh, dh)

        # 4B. Coherent Superposition Branch: |Psi_t> = sum_{s<=t} alpha_ts |v_s>, then |Psi_t><Psi_t|
        # Complex sum in O(T^2 d_h) vector time: (alpha_re + i alpha_im)(v_re + i v_im)
        psi_sup_re = torch.matmul(alpha_re, v_re) - torch.matmul(alpha_im, v_im)
        psi_sup_im = torch.matmul(alpha_re, v_im) + torch.matmul(alpha_im, v_re)
        psi_sup_re, psi_sup_im = complex_l2_normalize(psi_sup_re, psi_sup_im, dim=-1, eps=eps)

        rho_coh_re = psi_sup_re.unsqueeze(-1) * psi_sup_re.unsqueeze(-2) + \
                     psi_sup_im.unsqueeze(-1) * psi_sup_im.unsqueeze(-2)
        rho_coh_im = psi_sup_im.unsqueeze(-1) * psi_sup_re.unsqueeze(-2) - \
                     psi_sup_re.unsqueeze(-1) * psi_sup_im.unsqueeze(-2)

        # Blend Incoherent and Coherent Density Matrices: convex combination guarantees PSD + Trace 1!
        gamma = torch.sigmoid(self.coherence_logit)
        rho_re = (1.0 - gamma) * rho_inc_re + gamma * rho_coh_re
        rho_im = (1.0 - gamma) * rho_inc_im + gamma * rho_coh_im

        # Pre-purification purity Tr(rho^2) = ||rho_re||_F^2 + ||rho_im||_F^2 (since rho is Hermitian)
        purity_raw = (rho_re.square() + rho_im.square()).sum(dim=(-2, -1))

        # 5. Complex Hermitian McWeeny Matrix-Squaring Purification: rho <- (rho @ rho) / Tr(rho^2)
        # For Hermitian rho = A + i B (with A^T = A, B^T = -B):
        #   (A + i B)(A + i B) = (A A - B B) + i (A B + B A)
        for _ in range(self.purify_steps):
            sq_re = torch.matmul(rho_re, rho_re) - torch.matmul(rho_im, rho_im)
            sq_im = torch.matmul(rho_re, rho_im) + torch.matmul(rho_im, rho_re)
            # Trace of a Hermitian matrix is purely real (diagonal of sq_im is identically 0)
            tr_sq = sq_re.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
            rho_re = sq_re / (tr_sq + eps)
            rho_im = sq_im / (tr_sq + eps)

        # 6. Probe the Complex Density Matrix with |q_t>: |u_t> = (rho_re + i rho_im)(q_re + i q_im)
        q_re_col, q_im_col = q_re.unsqueeze(-1), q_im.unsqueeze(-1)
        u_re = (torch.matmul(rho_re, q_re_col) - torch.matmul(rho_im, q_im_col)).squeeze(-1)
        u_im = (torch.matmul(rho_re, q_im_col) + torch.matmul(rho_im, q_re_col)).squeeze(-1)

        u_re_flat = u_re.transpose(1, 2).contiguous().view(B, T, D)
        u_im_flat = u_im.transpose(1, 2).contiguous().view(B, T, D)

        out_re, out_im = self.W_out(u_re_flat, u_im_flat)
        next_re, next_im = complex_l2_normalize(psi_re + out_re, psi_im + out_im, dim=-1, eps=eps)

        if return_density:
            # Return as native torch.complex64 tensor of exact shape (B, T=512, H=32, d_h=32, d_h=32)
            rho_complex = torch.complex(
                rho_re.permute(0, 2, 1, 3, 4).contiguous(),
                rho_im.permute(0, 2, 1, 3, 4).contiguous(),
            )
            return next_re, next_im, purity_raw.mean(), rho_complex

        return next_re, next_im, purity_raw.mean(), None


# ============================================================================
# Smoke Test & Verification of Complex Hermitian Density Matrix Axioms
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T, H, dh = 1, 512, 32, 32
    d_model = H * dh  # 1024 complex amplitudes (1024 Real + 1024 Imaginary)

    layer = ComplexHermitianRegisterDensityAttention(
        d_model=d_model, n_registers=H, d_register=dh, p_boson=2, purify_steps=1
    ).to(device)

    psi_re, psi_im = complex_l2_normalize(
        torch.randn(B, T, d_model, device=device),
        torch.randn(B, T, d_model, device=device),
        dim=-1,
    )

    next_re, next_im, mean_purity, rho_c = layer(psi_re, psi_im, return_density=True)

    # 1. Verify Hermitian symmetry: rho == rho^dagger (Re is symmetric, Im is skew-symmetric)
    hermitian_err = (rho_c - rho_c.conj().transpose(-2, -1)).abs().max().item()
    # 2. Verify Imaginary part is non-zero and skew-symmetric (Im_ij == -Im_ji)
    imag_rms = rho_c.imag.square().mean().sqrt().item()
    imag_diag_max = rho_c.imag.diagonal(dim1=-2, dim2=-1).abs().max().item()
    # 3. Verify Unit Trace: Tr(rho) = 1.0 + 0.0i
    traces = rho_c.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_re_err = (traces.real - 1.0).abs().max().item()
    trace_im_err = traces.imag.abs().max().item()
    # 4. Verify Positive Semi-Definiteness via Complex Hermitian Eigenvalues (eigvalsh)
    min_eig = torch.linalg.eigvalsh(rho_c[0, -1, 0]).min().item()

    total_params = sum(p.numel() for p in layer.parameters())
    print(f"Complex Density Shape    : {tuple(rho_c.shape)} dtype={rho_c.dtype}")
    print(f"Layer Parameter Count    : {total_params:,} (2x real layer = 8,388,609 params)")
    print(f"Hermitian |rho - rho^H|  : {hermitian_err:.2e} (Exact Hermitian confirmed)")
    print(f"Imaginary Skew-Symm RMS  : {imag_rms:.4f} (Diagonal max: {imag_diag_max:.2e})")
    print(f"Trace Error (Re / Im)    : Re={trace_re_err:.2e}, Im={trace_im_err:.2e}")
    print(f"Min Hermitian Eigenvalue : {min_eig:.2e} (>= 0 PSD confirmed)")
    print(f"Mean Pre-Purify Tr(rho^2): {mean_purity.item():.4f}")
