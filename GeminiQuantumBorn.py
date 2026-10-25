import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumBornWaveAttention(nn.Module):
    """
    Projects unit token wavefunctions |psi_t> in S^(d-1) into a higher-dimensional
    Hilbert space R^D (D = expansion * d) to form Q, K, V kets, then combines:
      - Beam 1: Incoherent k-Boson Bloch Fidelity Attention (positive probability routing)
      - Beam 2: Coherent Amplitude Superposition Attention (constructive/destructive interference)
    """
    def __init__(self, d_model: int = 64, lift_expansion: int = 2, k_boson: float = 8.0):
        super().__init__()
        self.d_model = d_model
        self.d_lift = d_model * lift_expansion
        self.k_boson = nn.Parameter(torch.tensor(k_boson))

        # Stiefel isometric projections from R^d into higher-dimensional R^D
        self.U_q = nn.Parameter(torch.linalg.qr(torch.randn(self.d_lift, d_model))[0])
        self.U_k = nn.Parameter(torch.linalg.qr(torch.randn(self.d_lift, d_model))[0])
        self.U_v = nn.Parameter(torch.linalg.qr(torch.randn(self.d_lift, d_model))[0])
        self.W_out = nn.Parameter(torch.linalg.qr(torch.randn(self.d_lift, d_model))[0].T)

    def forward(self, psi: torch.Tensor):
        # psi: (B, T, d_model) unit wavefunctions on S^(d-1)
        B, T, _ = psi.shape
        q = F.normalize(psi @ self.U_q.T, dim=-1)  # (B, T, D_lift) unit Query kets
        k = F.normalize(psi @ self.U_k.T, dim=-1)  # (B, T, D_lift) unit Key kets
        v = F.normalize(psi @ self.U_v.T, dim=-1)  # (B, T, D_lift) unit Value kets

        # Quantum overlap amplitude c_{t,s} = <q_t | k_s> in [-1, 1]
        c = q @ k.transpose(-2, -1)                # (B, T, T)
        causal_mask = torch.tril(torch.ones(T, T, device=psi.device, dtype=torch.bool))

        # Branch 1: Incoherent k-Boson Bloch Fidelity P(s|t) ~ ((1 + c)/2)^k
        bloch_fid = torch.clamp(0.5 * (1.0 + c), min=1e-5)
        log_energy = self.k_boson * torch.log(2.0 * bloch_fid)
        log_energy = log_energy.masked_fill(~causal_mask, float("-inf"))
        w_incoherent = F.softmax(log_energy, dim=-1)
        v_incoherent = F.normalize(w_incoherent @ v, dim=-1)

        # Branch 2: Coherent Amplitude Superposition (allows destructive interference c < 0)
        amp_coherent = c.masked_fill(~causal_mask, 0.0)
        v_superpos = amp_coherent @ v
        # Pre-normalization norm measures contextual wave purity r_tilde!
        purity = v_superpos.norm(dim=-1, keepdim=True) / (causal_mask.sum(-1, keepdim=True).sqrt())
        v_coherent = F.normalize(v_superpos, dim=-1)

        # Project interfered higher-dimensional state back to S^(d-1)
        psi_mixed = F.normalize((v_incoherent + 0.25 * v_coherent) @ self.W_out.T, dim=-1)
        return psi_mixed, w_incoherent, purity
