import torch
import torch.nn as nn
import torch.nn.functional as F

def j_pair(x: torch.Tensor) -> torch.Tensor:
    """Exact imaginary unit i on real tensors in R^D: (x_2m, x_2m+1) -> (-x_2m+1, x_2m)."""
    x_2 = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    return torch.stack([-x_2[..., 1], x_2[..., 0]], dim=-1).reshape_as(x)

class SoftmaxFreeComplexBornAttention(nn.Module):
    """
    100% Softmax-Free Quantum Attention in C^(D/2) (implemented via real J_pair tensors):
      1. Lifts token wavefunctions |psi_t> into higher-D complex Q, K, V kets.
      2. Computes complex inner products c_ts = <q_t | k_s>_C = Re_ts + i * Im_ts.
      3. Normalizes WITHOUT Softmax/exp via p-boson algebraic trace normalization:
         - Incoherent Born weights: w_ts = |c_ts|^(2p) / sum_j |c_tj|^(2p)  (sum_s w_ts = 1)
         - Coherent complex amplitudes: alpha_ts = c_ts * |c_ts|^(p-1) / sqrt(Z)  (sum_s |alpha_ts|^2 = 1)
      4. Outputs the Rank-2 Density Matrix factor (psi_incoherent, psi_coherent) and purity Tr(rho_t^2).
    """
    def __init__(self, d_model: int = 64, lift_mult: int = 2, p_boson: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_lift = d_model * lift_mult
        self.p_boson = p_boson

        # Stiefel isometric projections into higher-dimensional space R^(D_lift) ~= C^(D_lift/2)
        self.U_q = nn.Parameter(torch.linalg.qr(torch.randn(self.d_lift, d_model))[0])
        self.U_k = nn.Parameter(torch.linalg.qr(torch.randn(self.d_lift, d_model))[0])
        self.U_v = nn.Parameter(torch.linalg.qr(torch.randn(self.d_lift, d_model))[0])
        self.W_out = nn.Parameter(torch.linalg.qr(torch.randn(self.d_lift, d_model))[0].T)

    def forward(self, psi: torch.Tensor, eps: float = 1e-8):
        # 1. Lift token wavefunctions to unit complex kets in C^(D_lift / 2)
        q = F.normalize(psi @ self.U_q.T, dim=-1)      # (B, T, D_lift)
        k = F.normalize(psi @ self.U_k.T, dim=-1)      # (B, T, D_lift)
        v = F.normalize(psi @ self.U_v.T, dim=-1)      # (B, T, D_lift)

        # 2. Complex inner product <q_t | k_s>_C = Re_ts + i * Im_ts via J_pair
        c_re = q @ k.transpose(-2, -1)                 # (B, T, T) Real part
        c_im = q @ j_pair(k).transpose(-2, -1)         # (B, T, T) Imaginary part
        mag_sq = c_re.pow(2) + c_im.pow(2)             # |<q_t | k_s>_C|^2 >= 0

        # Causal mask (s <= t)
        T = psi.size(1)
        mask = torch.tril(torch.ones(T, T, device=psi.device, dtype=torch.bool))
        mag_sq = mag_sq.masked_fill(~mask, 0.0)

        # 3A. Softmax-Free Born Probability Weights w_ts (sum_s w_ts = 1, w_ts >= 0)
        # For p_boson = 2: I_ts = |c_ts|^4 (2-boson Hong-Ou-Mandel intensity, zero exp!)
        intensity = mag_sq.pow(self.p_boson)
        z_trace = intensity.sum(dim=-1, keepdim=True) + eps
        w_born = intensity / z_trace                   # Exact probability simplex: sum_s w_ts = 1!

        # 3B. Softmax-Free Complex Wave Amplitudes alpha_ts (sum_s |alpha_ts|^2 = 1)
        scale_amp = mag_sq.pow(0.5 * (self.p_boson - 1)) / z_trace.sqrt()
        scale_amp = scale_amp.masked_fill(~mask, 0.0)
        alpha_re = c_re * scale_amp                    # Real amplitude component
        alpha_im = c_im * scale_amp                    # Imaginary amplitude component

        # 4. Combine V as a Rank-2 Density Matrix rho_t = (1-gamma) rho_diag + gamma |Psi><Psi|
        # Branch 1 (Diagonal Density Matrix): Weighted average with w_ts >= 0, sum_s w_ts = 1
        v_incoherent = F.normalize((w_born @ v) @ self.W_out.T, dim=-1)

        # Branch 2 (Coherent Superposition |Psi_t> = sum_s (alpha_re + i alpha_im) |v_s>)
        v_superpos = (alpha_re @ v) + (alpha_im @ j_pair(v))
        # Norm of |Psi_t> before renormalization measures constructive vs destructive interference!
        coherence_norm = v_superpos.norm(dim=-1, keepdim=True)
        v_coherent = F.normalize(v_superpos @ self.W_out.T, dim=-1)

        # Exact Tsallis-2 / Rényi-2 Purity of the attention distribution: sum_s w_ts^2 in [1/t, 1]
        purity = (w_born.pow(2)).sum(dim=-1, keepdim=True)
        return v_incoherent, v_coherent, w_born, purity, coherence_norm
