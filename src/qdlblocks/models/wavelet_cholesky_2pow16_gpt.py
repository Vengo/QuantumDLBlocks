"""Standalone Radical N=2^16 (65,536-D) Wavelet-Cholesky Amplitude Networks (Plans 035 & 036)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from ..layers.attention import SubspaceProjectorBornAttention, compute_document_ids
from ..layers.direct_chunked_ce import compute_two_beam_kboson_logits, direct_chunked_two_beam_kboson_ce
from ..layers.dynamic_povm_purity import ContextualBlochPurityGate
from ..layers.isometric import UnitaryRoPE
from ..layers.mudd_and_gates import ForwardSmearGate
from ..layers.overcomplete_two_beam import OvercompleteTwoBeamInterference
from ..layers.wavelet_cholesky import (
    BlockCholeskyLinear,
    WaveletCholeskyConfig,
    WaveletCholeskyMultiscaleBranch,
    apply_soft_powerlaw_spectral_floor,
)
from ..optim.muon import StiefelMuon


@dataclass(frozen=True)
class Radical2Pow16Config:
    """Configuration for standalone N=2^16 Wavelet-Cholesky Amplitude GPT architectures."""

    arch_mode: Literal["multiscale_unet", "consistent_kronecker", "fused_kronecker_unet"] = "multiscale_unet"
    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    ambient_n: int = 65536
    num_layers: int = 4
    embed_dim: int = 256
    num_heads: int = 4
    block_size: int = 512
    window_size: int = 384
    fine_detail_dim: int = 64
    mid_detail_dim: int = 128
    diffuse_rank: int = 16
    kraus_spectral_floor: float = 6.5
    quadratic_diffuse_fidelity: float = 0.50
    interferometer_expansion: float = 2.0
    directed_predecessor_mix: float = 0.42
    use_dyadic_bigram_gate: bool = True
    use_tied_value_anchor: bool = True
    use_outer_unet_skips: bool = True
    num_fine_up_banks: int = 1
    use_kraus_retraction_and_lambda_anneal: bool = False
    lambda_diff_start: float = 0.165
    lambda_diff_end: float = 0.052
    use_fast_cholesky_and_compile: bool = False
    use_last_layer_measurement_operator: bool = False
    use_traceless_observable: bool = False
    use_convex_traceless_born: bool = False
    traceless_gamma: float = 0.55
    use_dyadic_hierarchical_born: bool = False
    dyadic_coarse_alpha: float = 0.22
    dyadic_coarse_ce_weight: float = 0.08
    readout_gemm_precision: Literal["bf16", "fp8", "int8"] = "bf16"


def build_dyadic_2pow16_spectral_tree(
    ppmi_embeddings: torch.Tensor, ambient_n: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map vocabulary tokens into a 16-level balanced binary spectral tree of size `2^16 = 65,536`."""
    device, (v_in, d) = ppmi_embeddings.device, ppmi_embeddings.shape
    padded = torch.zeros((ambient_n, d), device=device, dtype=torch.float32)
    n_copy = min(v_in, ambient_n)
    padded[:n_copy] = l2_normalize(ppmi_embeddings[:n_copy].float(), dim=-1)
    if ambient_n > n_copy:
        padded[n_copy:, 0] = -2.0 - torch.linspace(0.0, 1.0, ambient_n - n_copy, device=device)

    leaf_to_v = torch.arange(ambient_n, device=device, dtype=torch.long)
    for level in range(int(round(math.log2(ambient_n)))):
        num_blocks, block_len = 1 << level, ambient_n >> level
        curr_idx = leaf_to_v.view(num_blocks, block_len)
        curr_vecs = padded.index_select(0, leaf_to_v).view(num_blocks, block_len, d)
        centered = curr_vecs - curr_vecs.mean(dim=1, keepdim=True)
        axis = centered[:, :, (level * 5) % d] + 0.5 * centered[:, :, (level * 5 + 1) % d]
        leaf_to_v = torch.gather(curr_idx, 1, torch.argsort(axis, dim=1, stable=True)).reshape(-1)

    perm_v_to_leaf = torch.empty_like(leaf_to_v)
    perm_v_to_leaf.scatter_(0, leaf_to_v, torch.arange(ambient_n, device=device, dtype=torch.long))
    return perm_v_to_leaf, leaf_to_v


class Tied2Pow16WaveletCholeskyPyramid(nn.Module):
    """4-Scale Dyadic Wavelet-Cholesky Tied Basis over N = 2^16 = 65,536 with Predecessor & Bigram Bridges."""

    def __init__(self, cfg: Radical2Pow16Config) -> None:
        super().__init__()
        self.cfg = cfg
        d, d_mid, d_fine = cfg.embed_dim, cfg.mid_detail_dim, cfg.fine_detail_dim
        fast_chol = bool(cfg.use_fast_cholesky_and_compile)
        self.num_fine_banks = max(1, int(cfg.num_fine_up_banks))
        self.bank_shift = max(0, int(round(math.log2(cfg.ambient_n // self.num_fine_banks))))
        self.register_buffer("perm_v_to_leaf", torch.arange(cfg.ambient_n, dtype=torch.long))
        self.register_buffer("leaf_to_v", torch.arange(cfg.ambient_n, dtype=torch.long))
        self.register_buffer("init_wav_rms", torch.tensor([0.15, 0.18, 0.22, 0.35], dtype=torch.float32))
        gen_tail = torch.Generator(device="cpu").manual_seed(2038)
        raw_tail = torch.randn((2048, d), generator=gen_tail, dtype=torch.float32)
        raw_tail = raw_tail - raw_tail.mean(dim=0, keepdim=True)
        u_t, _, vh_t = torch.linalg.svd(raw_tail, full_matrices=False)
        self.register_buffer("tail_frame_base", l2_normalize(u_t @ vh_t, dim=-1), persistent=False)
        self.register_buffer("tail_proj", torch.zeros((d, d), dtype=torch.float32), persistent=False)
        self.register_buffer("sem_s_fwd", torch.eye(d, dtype=torch.float32), persistent=False)
        self.register_buffer("sem_s_inv", torch.eye(d, dtype=torch.float32), persistent=False)
        self.wav_256 = nn.Parameter(torch.zeros(256, d))
        self.wav_2048 = nn.Parameter(torch.zeros(2048, d))
        self.wav_8192 = nn.Parameter(torch.zeros(8192, d_mid))
        self.wav_65536 = nn.Parameter(torch.zeros(cfg.ambient_n, d_fine))
        self.up_8192 = nn.Parameter(torch.eye(d_mid, d) * (1.0 / math.sqrt(d_mid)))
        if self.num_fine_banks == 1:
            self.up_65536 = nn.Parameter(torch.eye(d_fine, d) * (1.0 / math.sqrt(d_fine)))
        else:
            init_banks = torch.eye(d_fine, d).unsqueeze(0).repeat(self.num_fine_banks, 1, 1) * (1.0 / math.sqrt(d_fine))
            self.up_65536 = nn.Parameter(init_banks)
        self.chol_256 = BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=fast_chol)
        self.chol_2048 = BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=fast_chol)
        self.chol_8192 = BlockCholeskyLinear(d_mid, max_block_size=32, fast_spd_bmm=fast_chol)
        self.chol_65536 = BlockCholeskyLinear(d_fine, max_block_size=32, fast_spd_bmm=fast_chol)
        self.readout_chol_bridge = BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=fast_chol)
        self.scale_gains = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.coarse_transition = nn.Parameter(torch.zeros(256, 256, dtype=torch.float32))
        self.transition_gain = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.kraus_raw_weights = nn.Parameter(torch.linspace(0.6, -0.6, cfg.diffuse_rank, dtype=torch.float32))
        self.kraus_generators = nn.Parameter(
            torch.eye(d, dtype=torch.float32).unsqueeze(0).repeat(cfg.diffuse_rank, 1, 1)
        )
        if cfg.arch_mode == "fused_kronecker_unet":
            self.pred_wav_2048 = nn.Parameter(torch.zeros(2048, d))
            self.chol_pred_2048 = BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=fast_chol)
            self.pred_mix_gain = nn.Parameter(torch.tensor(cfg.directed_predecessor_mix, dtype=torch.float32))
            self.bigram_chol_fine = BlockCholeskyLinear(d_fine, max_block_size=32, fast_spd_bmm=fast_chol)
            self.bigram_chol_mid = BlockCholeskyLinear(d_mid, max_block_size=32, fast_spd_bmm=fast_chol)
            self.val_chol_bridge = BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=fast_chol)

    @staticmethod
    def apply_symplectic_swap(x: torch.Tensor) -> torch.Tensor:
        """Apply skew-involutive even/odd channel swap `(a, b) -> (b, -a)` for exact zero self-overlap."""
        return torch.stack((x[..., 1::2], -x[..., 0::2]), dim=-1).reshape_as(x)

    def _lift_fine_wavelet(self, flat_p: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Lift fine-scale `(ambient_n, d_fine)` wavelet coefficients to `R^d` using 1 or `B` subtree banks."""
        if self.num_fine_banks == 1:
            if self.cfg.use_fast_cholesky_and_compile and flat_p.numel() > 16384:
                w0_all = self.chol_65536(self.wav_65536)
                c0_all = F.linear(w0_all, self.up_65536.to(w0_all.dtype).t())
                return c0_all.index_select(0, flat_p).view(*target_shape, -1)
            w0 = self.chol_65536(self.wav_65536.index_select(0, flat_p)).view(*target_shape, -1)
            return F.linear(w0, self.up_65536.to(w0.dtype).t())

        if self.cfg.use_fast_cholesky_and_compile:
            w0_all = self.chol_65536(self.wav_65536).view(self.num_fine_banks, -1, self.cfg.fine_detail_dim)
            c0_all = torch.bmm(w0_all, self.up_65536.to(w0_all.dtype)).view(self.cfg.ambient_n, self.cfg.embed_dim)
            return c0_all.index_select(0, flat_p).view(*target_shape, -1)

        w0_flat = self.chol_65536(self.wav_65536).index_select(0, flat_p)
        bank_id = flat_p >> self.bank_shift
        c0_flat = torch.zeros((flat_p.numel(), self.cfg.embed_dim), device=w0_flat.device, dtype=w0_flat.dtype)
        up_b = self.up_65536.to(w0_flat.dtype)
        for b in range(self.num_fine_banks):
            mask = bank_id == b
            if bool(mask.any()):
                c0_flat[mask] = F.linear(w0_flat[mask], up_b[b].t()).to(c0_flat.dtype)
        return c0_flat.view(*target_shape, -1)

    def _synthesize_leaf_vectors(self, leaf_idx: torch.Tensor) -> torch.Tensor:
        """Synthesize normalized multi-scale wavelet amplitudes for arbitrary 16-bit leaf indices."""
        flat_p = leaf_idx.reshape(-1)
        c3 = self.chol_256(self.wav_256).index_select(0, flat_p >> 8).view(*leaf_idx.shape, -1)
        c2 = self.chol_2048(self.wav_2048).index_select(0, flat_p >> 5).view(*leaf_idx.shape, -1)
        if self.cfg.use_fast_cholesky_and_compile:
            w1_all = self.chol_8192(self.wav_8192)
            c1 = F.linear(w1_all, self.up_8192.to(w1_all.dtype).t()).index_select(0, flat_p >> 3).view(*leaf_idx.shape, -1)
        else:
            w1 = self.chol_8192(self.wav_8192).index_select(0, flat_p >> 3).view(*leaf_idx.shape, -1)
            c1 = F.linear(w1, self.up_8192.to(w1.dtype).t())
        c0 = self._lift_fine_wavelet(flat_p, leaf_idx.shape)
        g = F.softplus(self.scale_gains).to(c3.dtype)
        return l2_normalize(g[0] * c3 + g[1] * c2 + g[2] * c1 + g[3] * c0, dim=-1)

    def embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """Synthesize input token amplitudes in R^d directly from the N=2^16 wavelet pyramid."""
        p = self.perm_v_to_leaf.index_select(0, idx.reshape(-1)).view_as(idx)
        return self._synthesize_leaf_vectors(p)

    def _synthesize_raw_detectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize `(psi_self, m_orth)` before orthogonal Parseval nullspace completion."""
        p = self.perm_v_to_leaf[: self.cfg.vocab_size]
        psi_self = self._synthesize_leaf_vectors(p)
        base_256 = self.chol_256(self.wav_256)
        t_gain = torch.tanh(self.transition_gain).to(base_256.dtype)
        trans_256 = t_gain * (self.coarse_transition.to(base_256.dtype) @ base_256)
        m_base = self.readout_chol_bridge(self.apply_symplectic_swap(psi_self)) + trans_256.index_select(0, p >> 8)
        if hasattr(self, "pred_wav_2048"):
            pred_2048 = self.chol_pred_2048(self.pred_wav_2048).index_select(0, p >> 5)
            m_base = m_base + torch.tanh(self.pred_mix_gain).to(pred_2048.dtype) * pred_2048
        if self.cfg.use_dyadic_hierarchical_born and self.cfg.dyadic_coarse_alpha > 0.0:
            m_root_256 = l2_normalize(
                self.readout_chol_bridge(self.apply_symplectic_swap(l2_normalize(base_256, dim=-1))) + trans_256,
                dim=-1,
            )
            m_base = m_base + float(self.cfg.dyadic_coarse_alpha) * m_root_256.index_select(0, p >> 8)
        overlap = (m_base * psi_self).sum(dim=-1, keepdim=True).to(m_base.dtype)
        return psi_self, l2_normalize(m_base - overlap * psi_self, dim=-1)

    @torch.no_grad()
    def _update_contragredient_gauge(self) -> None:
        """Update semantic subspace gauge `(sem_s_fwd, sem_s_inv)` and orthogonal `tail_proj`."""
        _, m_orth = self._synthesize_raw_detectors()
        v_eff = min(self.cfg.valid_vocab_size, m_orth.size(0))
        m_sub = m_orth[:v_eff].detach().float()
        m_sub = m_sub - m_sub.mean(dim=0, keepdim=True)
        cov = m_sub.t() @ m_sub
        evals, evecs = torch.linalg.eigh(cov)
        d = m_sub.size(-1)
        n_sem = max(12, d // 5)
        n_low = d - n_sem
        v_low = evecs[:, :n_low]
        v_top = evecs[:, n_low:]
        ev_top = evals[n_low:].clamp_min(1e-4)
        ev_rel = (ev_top / ev_top.mean().clamp_min(1e-6)).clamp(min=0.15, max=6.0)
        self.tail_proj.copy_(v_low @ v_low.t())
        self.sem_s_inv.copy_((v_top * torch.pow(ev_rel, -0.22).unsqueeze(0)) @ v_top.t())
        self.sem_s_fwd.copy_(v_top @ v_top.t())

    def synthesize_initial_wavefunctions(self) -> torch.Tensor:
        """Synthesize tied `(vocab_size, embed_dim)` Parseval-completed target wavefunctions `psi_v^(0)`."""
        psi_self, m_orth = self._synthesize_raw_detectors()
        v_eff = min(self.cfg.valid_vocab_size, m_orth.size(0))
        m_cent = m_orth - m_orth[:v_eff].mean(dim=0, keepdim=True)
        s_inv = self.sem_s_inv.to(m_orth.dtype)
        t_proj = self.tail_proj.to(m_orth.dtype)
        psi_self_g = l2_normalize(psi_self - F.linear(psi_self, t_proj), dim=-1)
        m_sem = F.linear(m_cent, s_inv)
        m_sem = l2_normalize(m_sem - (m_sem * psi_self_g).sum(dim=-1, keepdim=True) * psi_self_g, dim=-1)
        v_idx = torch.arange(self.cfg.vocab_size, device=m_orth.device)
        z_raw = self.tail_frame_base.index_select(0, v_idx & 2047).to(m_orth.dtype)
        z_tail = F.linear(z_raw, t_proj)
        z_tail = z_tail - (z_tail * psi_self_g).sum(dim=-1, keepdim=True) * psi_self_g
        z_tail = l2_normalize(z_tail - (z_tail * m_sem).sum(dim=-1, keepdim=True) * m_sem, dim=-1)
        chol_drift = F.relu(self.chol_256.raw_diag.detach().abs().max() - 0.36)
        alpha = float(0.72 / (1.0 + 2.5 * float(chol_drift.item())))
        m_comb = math.sqrt(max(1e-6, 1.0 - alpha * alpha)) * m_sem + alpha * z_tail
        ov = (m_comb * psi_self_g).sum(dim=-1, keepdim=True)
        return l2_normalize(m_comb - ov * psi_self_g, dim=-1)

    def embed_dyadic_bigram_phase(self, idx: torch.Tensor) -> torch.Tensor:
        """Synthesize causal dyadic bigram wavelet amplitudes reusing `wav_65536` and `wav_8192`."""
        p = self.perm_v_to_leaf.index_select(0, idx.reshape(-1)).view_as(idx)
        p_prev = torch.cat([p[:, :1], p[:, :-1]], dim=1) if (p.ndim == 2 and p.size(1) > 1) else p
        mask_n = self.cfg.ambient_n - 1
        h_leaf = ((p_prev * 2654435761) ^ (p * 40503)) & mask_n
        h_mid = ((p_prev >> 3) * 1315423911 ^ (p >> 3)) & 8191
        w_fine = self.bigram_chol_fine(self.wav_65536.index_select(0, h_leaf.reshape(-1)).view(*idx.shape, -1))
        w_mid = self.bigram_chol_mid(self.wav_8192.index_select(0, h_mid.reshape(-1)).view(*idx.shape, -1))
        up_fine_mat = (self.up_65536 if self.up_65536.ndim == 2 else self.up_65536.mean(dim=0)).to(w_fine.dtype)
        b_vec = F.linear(w_fine, up_fine_mat.t()) + F.linear(w_mid, self.up_8192.to(w_mid.dtype).t())
        return l2_normalize(b_vec, dim=-1)

    def synthesize_value_anchor(self, psi_in: torch.Tensor, bigram_psi: torch.Tensor | None = None) -> torch.Tensor:
        """Synthesize token-tied attention value anchor `v_0` via `val_chol_bridge`."""
        v_anc = self.val_chol_bridge(psi_in)
        if bigram_psi is not None:
            v_anc = v_anc + 0.30 * bigram_psi.to(v_anc.dtype)
        return l2_normalize(v_anc, dim=-1).view(*psi_in.shape[:-1], self.cfg.num_heads, -1).transpose(1, 2).contiguous()

    def compute_dyadic_coarse_aux_loss(
        self, psi_s_out: torch.Tensor, m_sharp: torch.Tensor, beta_t: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 256-way coarse dyadic tree convex-traceless Born auxiliary loss (`K_coarse = 256 <= d`)."""
        v_eff = min(self.cfg.valid_vocab_size, m_sharp.size(0))
        c_idx = self.perm_v_to_leaf[:v_eff] >> 8
        m_c = torch.zeros((256, m_sharp.size(-1)), device=m_sharp.device, dtype=torch.float32)
        m_c.index_add_(0, c_idx, m_sharp[:v_eff].float())
        m_c = l2_normalize(m_c, dim=-1).to(psi_s_out.dtype)
        c_raw = F.linear(psi_s_out.reshape(-1, psi_s_out.size(-1)), m_c).float()
        c_hat = c_raw - c_raw.mean(dim=-1, keepdim=True)
        c_pos = F.relu(c_hat)
        c_sq = c_pos * c_pos
        o_c = c_hat + abs(float(self.cfg.traceless_gamma)) * (c_sq - c_sq.mean(dim=-1, keepdim=True))
        b_flat = beta_t.float().reshape(-1, 1) if beta_t.numel() > 1 else beta_t.float().reshape(1, 1)
        flat_y = targets.reshape(-1).long()
        valid = (flat_y >= 0) & (flat_y < v_eff)
        y_coarse = torch.where(valid, self.perm_v_to_leaf.index_select(0, flat_y.clamp(0, v_eff - 1)) >> 8, -1)
        return F.cross_entropy(b_flat * o_c, y_coarse, ignore_index=-1)

    @torch.no_grad()
    def compute_kraus_stiefel_err(self) -> float:
        """Compute mean Frobenius Stiefel error `||K_r^T K_r - I||_F / sqrt(d)` across Kraus generators."""
        kg = self.kraus_generators.detach().float()
        d = kg.size(-1)
        eye = torch.eye(d, device=kg.device, dtype=torch.float32).unsqueeze(0)
        gram = torch.bmm(kg.transpose(-2, -1), kg) - eye
        return float((torch.linalg.norm(gram, dim=(-2, -1)) / math.sqrt(float(d))).mean().item())

    @torch.no_grad()
    def _stabilize_pyramid_frame(self, rate: float = 0.05) -> None:
        """Apply smooth Riemannian Frame-Potential Landing to wavelet tables & bound Cholesky condition numbers."""
        for idx_s, w_param in enumerate((self.wav_256, self.wav_2048, self.wav_8192, self.wav_65536)):
            w_f = w_param.data.float()
            w_f = w_f - w_f.mean(dim=0, keepdim=True)
            n_rows, d_s = w_f.shape
            frob_sq = (w_f * w_f).sum().clamp_min(1e-12)
            diff = (float(d_s) / frob_sq) * (w_f.t() @ w_f) - torch.eye(d_s, device=w_f.device, dtype=torch.float32)
            denom = 1.0 + (torch.linalg.norm(diff) / math.sqrt(float(d_s)))
            w_f = w_f - (0.5 * rate) * (w_f @ (diff / denom))
            cur_rms = torch.sqrt((w_f * w_f).sum() / float(n_rows) + 1e-12)
            w_f = w_f * torch.sqrt(self.init_wav_rms[idx_s].clamp_min(1e-4) / cur_rms)
            w_param.data.copy_(w_f.to(w_param.dtype))

        for chol_m in (self.chol_256, self.chol_2048, self.chol_8192, self.chol_65536, self.readout_chol_bridge):
            chol_m.raw_diag.data.sub_(chol_m.raw_diag.data.mean()).clamp_(-0.35, 0.35)
            rt = torch.tril(chol_m.raw_tril.data.float(), diagonal=-1)
            b_norm = torch.linalg.norm(rt, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            scale = torch.clamp((0.35 * math.sqrt(float(chol_m.block_size))) / b_norm, max=1.0)
            chol_m.raw_tril.data.copy_((rt * scale).to(chol_m.raw_tril.dtype))
        if (self.cfg.use_last_layer_measurement_operator or self.cfg.use_traceless_observable) and not self.cfg.use_convex_traceless_born:
            self._update_contragredient_gauge()

    @torch.no_grad()
    def retract_kraus_generators(self, smooth_landing_rate: float = 0.08) -> float:
        """Retract `kraus_generators` via smooth Riemannian Stiefel Landing (or Polar SVD when `err > 0.03`)."""
        kg = self.kraus_generators.data.float()
        d = kg.size(-1)
        eye = torch.eye(d, device=kg.device, dtype=torch.float32).unsqueeze(0)
        gram = torch.bmm(kg.transpose(-2, -1), kg) - eye
        err = float((torch.linalg.norm(gram, dim=(-2, -1)) / math.sqrt(float(d))).mean().item())
        if err > 0.03:
            u, _, vh = torch.linalg.svd(kg, full_matrices=False)
            kg_out = torch.bmm(u, vh)
        else:
            denom = 1.0 + (torch.linalg.norm(gram, dim=(-2, -1), keepdim=True) / math.sqrt(float(d)))
            kg_out = kg - float(smooth_landing_rate) * torch.bmm(kg, gram / denom)
        self.kraus_generators.data.copy_(kg_out.to(self.kraus_generators.dtype))
        if (self.cfg.use_last_layer_measurement_operator or self.cfg.use_traceless_observable) and not self.cfg.use_convex_traceless_born:
            self._update_contragredient_gauge()
        elif not self.cfg.use_convex_traceless_born:
            self._stabilize_pyramid_frame(rate=0.04)
        return self.compute_kraus_stiefel_err()

    @torch.no_grad()
    def compute_sharp_eff_rank(self, m_sharp: torch.Tensor | None = None) -> float:
        """Compute genuine spectral participation ratio `(Tr G)^2 / ||G||_F^2` of sharp detectors `m_sharp`."""
        if m_sharp is None:
            if (self.cfg.use_last_layer_measurement_operator or self.cfg.use_traceless_observable) and not self.cfg.use_convex_traceless_born:
                m_sharp = self.synthesize_initial_wavefunctions()
            else:
                m_sharp, _, _ = self.synthesize_vocab_detectors()
        m_f = m_sharp[: self.cfg.valid_vocab_size].detach().float()
        gram = m_f.t() @ m_f
        tr = torch.diagonal(gram).sum()
        return float(((tr * tr) / (gram * gram).sum().clamp_min(1e-12)).item())

    def apply_last_layer_measurement(
        self, psi: torch.Tensor, psi_diff: torch.Tensor, psi_in_0: torch.Tensor, idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply measurement operator & Kraus bath to `(B, T, d)` output states against initial wavefunctions `psi_v^(0)`."""
        w_k = apply_soft_powerlaw_spectral_floor(
            F.softmax(self.kraus_raw_weights.float(), dim=0), min_eff_rank=self.cfg.kraus_spectral_floor
        )
        op_mean = torch.einsum("r,rij->ij", torch.sqrt(w_k + 1e-8), self.kraus_generators.float())
        if self.cfg.use_convex_traceless_born:
            _, psi_vocab = self._synthesize_raw_detectors()
            if self.cfg.use_fast_cholesky_and_compile and psi_diff.is_cuda and torch.is_autocast_enabled():
                psi_meas_diff = l2_normalize(F.linear(psi_diff.bfloat16(), op_mean.t().bfloat16()), dim=-1).to(psi_diff.dtype)
            else:
                psi_meas_diff = l2_normalize(F.linear(psi_diff.float(), op_mean.t()), dim=-1).to(psi_diff.dtype)
            return psi, psi_meas_diff, psi_vocab, w_k

        psi_vocab = self.synthesize_initial_wavefunctions()
        s_fwd = self.sem_s_fwd.to(psi.dtype)
        t_proj = self.tail_proj.to(psi.dtype)
        psi_xt = psi_vocab.index_select(0, idx.reshape(-1)).view_as(psi)
        psi_xt_sem = l2_normalize(psi_xt - F.linear(psi_xt, t_proj), dim=-1)
        m_out = F.linear(psi + 0.25 * psi_in_0, s_fwd)
        psi_meas_sharp = l2_normalize(m_out - (m_out * psi_xt_sem).sum(dim=-1, keepdim=True) * psi_xt_sem, dim=-1)
        op_k = self.sem_s_fwd.float() @ op_mean.t()
        if self.cfg.use_fast_cholesky_and_compile and psi_diff.is_cuda and torch.is_autocast_enabled():
            d_sem = F.linear(psi_diff.bfloat16(), op_k.bfloat16()).to(psi_diff.dtype)
        else:
            d_sem = F.linear(psi_diff.float(), op_k).to(psi_diff.dtype)
        psi_meas_diff = l2_normalize(d_sem - (d_sem * psi_xt_sem).sum(dim=-1, keepdim=True) * psi_xt_sem, dim=-1)
        return psi_meas_sharp, psi_meas_diff, psi_vocab, w_k

    def synthesize_vocab_detectors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize tied `(V, d)` sharp & diffuse POVM detectors from the N=2^16 wavelet pyramid."""
        m_sharp = self.synthesize_initial_wavefunctions() if self.cfg.use_traceless_observable else self._synthesize_raw_detectors()[1]
        w_k = apply_soft_powerlaw_spectral_floor(
            F.softmax(self.kraus_raw_weights.float(), dim=0), min_eff_rank=self.cfg.kraus_spectral_floor
        )
        op_mean = torch.einsum("r,rij->ij", torch.sqrt(w_k + 1e-8), self.kraus_generators.float())
        if self.cfg.use_fast_cholesky_and_compile and m_sharp.is_cuda and torch.is_autocast_enabled():
            m_diff = l2_normalize(F.linear(m_sharp.bfloat16(), op_mean.bfloat16()), dim=-1).to(m_sharp.dtype)
        else:
            m_diff = l2_normalize(F.linear(m_sharp.float(), op_mean), dim=-1).to(m_sharp.dtype)
        return m_sharp, m_diff, w_k


class WaveletCholeskyUNetBlock(nn.Module):
    """Architecture 1 Middle Block: Multi-Scale Wavelet Downsample -> Causal Subband Mixing -> Upsample."""

    def __init__(self, cfg: Radical2Pow16Config, layer_idx: int) -> None:
        super().__init__()
        d = cfg.embed_dim
        fast_chol = bool(cfg.use_fast_cholesky_and_compile)
        self.window_size = cfg.window_size
        self.attn = SubspaceProjectorBornAttention(
            embed_dim=d, num_heads=cfg.num_heads, block_size=cfg.block_size, is_causal=True,
            learnable_temperature=True, value_residual_mixing=True, use_fused_qkv=True,
            use_flash_born=fast_chol, use_purified_attn=True, use_mudd_and_gates=True,
        )
        self.wavelet_branch = WaveletCholeskyMultiscaleBranch(
            WaveletCholeskyConfig(embed_dim=d, num_levels=3, max_block_size=32, fast_spd_bmm=fast_chol)
        )
        self.coarse_causal_chol = BlockCholeskyLinear(d >> 3, max_block_size=32, fast_spd_bmm=fast_chol)
        self.mid_causal_chol = BlockCholeskyLinear(d >> 2, max_block_size=32, fast_spd_bmm=fast_chol)
        self.subband_shift_mix = nn.Parameter(torch.tensor([0.35, 0.25], dtype=torch.float32))
        self.interferometer = OvercompleteTwoBeamInterference(
            embed_dim=d, frame_expansion=1.0, use_cosine_phase_beam_gate=True,
            use_mach_zehnder_beamsplitter=True, use_wavelet_cholesky_branch=False,
        )
        scale_init = 0.35 / math.sqrt(float(max(1, layer_idx + 1)))
        self.attn_gate = nn.Parameter(torch.full((d,), scale_init))
        self.unet_gate = nn.Parameter(torch.full((d,), scale_init))

    def forward(
        self, psi: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        v_0: torch.Tensor | None, doc_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attn_out, v_first = self.attn(
            psi, cos=cos, sin=sin, v_0=v_0, window_size=self.window_size, doc_ids=doc_ids, return_v=True
        )
        psi_a = l2_normalize(psi + torch.tanh(self.attn_gate).to(psi.dtype) * attn_out, dim=-1)

        coarse, details = self.wavelet_branch.downsample_amplitudes(psi_a)
        coarse_t, details_t = self.wavelet_branch.apply_cholesky_scale_operators(coarse, details)
        mix_w = torch.sigmoid(self.subband_shift_mix).to(coarse_t.dtype)
        if coarse_t.size(1) > 1:
            coarse_t = coarse_t + mix_w[0] * self.coarse_causal_chol(F.pad(coarse_t[:, :-1, :], (0, 0, 1, 0)))
            details_t[1] = details_t[1] + mix_w[1] * self.mid_causal_chol(F.pad(details_t[1][:, :-1, :], (0, 0, 1, 0)))

        wav_recon = l2_normalize(self.wavelet_branch.upsample_amplitudes(coarse_t, details_t), dim=-1)
        combined = l2_normalize(0.55 * self.interferometer(psi_a) + 0.45 * wav_recon, dim=-1)
        return l2_normalize(psi_a + torch.tanh(self.unet_gate).to(psi_a.dtype) * combined, dim=-1), wav_recon, v_first


class ConsistentKroneckerWaveletBlock(nn.Module):
    """Architecture 2 Middle Block: Coupled Coarse-8-Bit x Fine-8-Bit Wavelet-Cholesky on R^{256 x 256}."""

    def __init__(self, cfg: Radical2Pow16Config, layer_idx: int) -> None:
        super().__init__()
        d = cfg.embed_dim
        fast_chol = bool(cfg.use_fast_cholesky_and_compile)
        self.half_d = d // 2
        self.window_size = cfg.window_size
        self.attn = SubspaceProjectorBornAttention(
            embed_dim=d, num_heads=cfg.num_heads, block_size=cfg.block_size, is_causal=True,
            learnable_temperature=True, value_residual_mixing=True, use_fused_qkv=True,
            use_flash_born=fast_chol, use_purified_attn=True, use_mudd_and_gates=True,
        )
        self.coarse_wav = WaveletCholeskyMultiscaleBranch(
            WaveletCholeskyConfig(embed_dim=self.half_d, num_levels=3, max_block_size=32, fast_spd_bmm=fast_chol)
        )
        self.fine_wav = WaveletCholeskyMultiscaleBranch(
            WaveletCholeskyConfig(embed_dim=self.half_d, num_levels=3, max_block_size=32, fast_spd_bmm=fast_chol)
        )
        self.scale_beamsplitter_thetas = nn.Parameter(
            torch.tensor([math.pi / 8.0, math.pi / 6.0, math.pi / 4.0, math.pi / 4.0], dtype=torch.float32)
        )
        self.kron_modulation_chol = BlockCholeskyLinear(self.half_d, max_block_size=32, fast_spd_bmm=fast_chol)
        self.interferometer = OvercompleteTwoBeamInterference(
            embed_dim=d, frame_expansion=1.0, use_cosine_phase_beam_gate=True,
            use_mach_zehnder_beamsplitter=True, use_wavelet_cholesky_branch=False,
        )
        scale_init = 0.35 / math.sqrt(float(max(1, layer_idx + 1)))
        self.attn_gate = nn.Parameter(torch.full((d,), scale_init))
        self.kron_gate = nn.Parameter(torch.full((d,), scale_init))

    def forward(
        self, psi: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        v_0: torch.Tensor | None, doc_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attn_out, v_first = self.attn(
            psi, cos=cos, sin=sin, v_0=v_0, window_size=self.window_size, doc_ids=doc_ids, return_v=True
        )
        psi_a = l2_normalize(psi + torch.tanh(self.attn_gate).to(psi.dtype) * attn_out, dim=-1)

        psi_c, psi_f = psi_a[..., : self.half_d], psi_a[..., self.half_d :]
        c_coarse, c_details = self.coarse_wav.downsample_amplitudes(psi_c)
        f_coarse, f_details = self.fine_wav.downsample_amplitudes(psi_f)
        c_coarse, c_details = self.coarse_wav.apply_cholesky_scale_operators(c_coarse, c_details)
        f_coarse, f_details = self.fine_wav.apply_cholesky_scale_operators(f_coarse, f_details)

        thetas = self.scale_beamsplitter_thetas.to(c_coarse.dtype)
        th0 = thetas[0]
        c_coarse, f_coarse = (
            torch.cos(th0) * c_coarse + torch.sin(th0) * f_coarse,
            -torch.sin(th0) * c_coarse + torch.cos(th0) * f_coarse,
        )
        for s_idx in range(len(c_details)):
            th_s = thetas[min(s_idx + 1, 3)]
            cd, fd = c_details[s_idx], f_details[s_idx]
            c_details[s_idx] = torch.cos(th_s) * cd + torch.sin(th_s) * fd
            f_details[s_idx] = -torch.sin(th_s) * cd + torch.cos(th_s) * fd

        rec_c = self.coarse_wav.upsample_amplitudes(c_coarse, c_details)
        rec_f = self.fine_wav.upsample_amplitudes(f_coarse, f_details)
        entangled_c = rec_c * (1.0 + 0.25 * torch.tanh(self.kron_modulation_chol(rec_f)))
        entangled_f = rec_f * (1.0 + 0.25 * torch.tanh(rec_c))
        kron_state = l2_normalize(torch.cat([entangled_c, entangled_f], dim=-1), dim=-1)

        combined = l2_normalize(0.50 * self.interferometer(psi_a) + 0.50 * kron_state, dim=-1)
        return l2_normalize(psi_a + torch.tanh(self.kron_gate).to(psi_a.dtype) * combined, dim=-1), kron_state, v_first


class FusedKroneckerWaveletUNetBlock(nn.Module):
    """Architecture 3 (Plan 036): Fuses Multi-Scale Causal Subband U-Net with Coarse<->Fine Kronecker Entanglement."""

    def __init__(self, cfg: Radical2Pow16Config, layer_idx: int) -> None:
        super().__init__()
        d = cfg.embed_dim
        fast_chol = bool(cfg.use_fast_cholesky_and_compile)
        self.half_d = d // 2
        self.window_size = cfg.window_size
        self.attn = SubspaceProjectorBornAttention(
            embed_dim=d, num_heads=cfg.num_heads, block_size=cfg.block_size, is_causal=True,
            learnable_temperature=True, value_residual_mixing=True, use_fused_qkv=True,
            use_flash_born=fast_chol, use_purified_attn=True, use_mudd_and_gates=True,
        )
        self.wavelet_branch = WaveletCholeskyMultiscaleBranch(
            WaveletCholeskyConfig(embed_dim=d, num_levels=3, max_block_size=32, fast_spd_bmm=fast_chol)
        )
        self.coarse_causal_chol = BlockCholeskyLinear(d >> 3, max_block_size=32, fast_spd_bmm=fast_chol)
        self.coarse_dilated_chol = BlockCholeskyLinear(d >> 3, max_block_size=32, fast_spd_bmm=fast_chol)
        self.mid_causal_chol = BlockCholeskyLinear(d >> 2, max_block_size=32, fast_spd_bmm=fast_chol)
        self.subband_shift_mix = nn.Parameter(torch.tensor([0.38, 0.22, 0.28], dtype=torch.float32))
        self.kron_beamsplitter_theta = nn.Parameter(torch.tensor(math.pi / 7.0, dtype=torch.float32))
        self.kron_modulation_chol = BlockCholeskyLinear(self.half_d, max_block_size=32, fast_spd_bmm=fast_chol)
        self.kron_reverse_chol = BlockCholeskyLinear(self.half_d, max_block_size=32, fast_spd_bmm=fast_chol)
        self.interferometer = OvercompleteTwoBeamInterference(
            embed_dim=d, frame_expansion=cfg.interferometer_expansion,
            use_cosine_phase_beam_gate=True, use_mach_zehnder_beamsplitter=True,
            use_wavelet_cholesky_branch=True,
        )
        if fast_chol and getattr(self.interferometer, "wavelet_cholesky_branch", None) is not None:
            for m in self.interferometer.wavelet_cholesky_branch.modules():
                if isinstance(m, BlockCholeskyLinear):
                    m.fast_spd_bmm = True
        scale_init = 0.35 / math.sqrt(float(max(1, layer_idx + 1)))
        self.attn_gate = nn.Parameter(torch.full((d,), scale_init))
        self.fused_gate = nn.Parameter(torch.full((d,), scale_init))

    def _wavelet_kronecker_pathway(self, psi_a: torch.Tensor) -> torch.Tensor:
        coarse, details = self.wavelet_branch.downsample_amplitudes(psi_a)
        coarse_t, details_t = self.wavelet_branch.apply_cholesky_scale_operators(coarse, details)
        mix_w = torch.sigmoid(self.subband_shift_mix).to(coarse_t.dtype)
        seq_t = coarse_t.size(1)
        if seq_t > 1:
            coarse_t = coarse_t + mix_w[0] * self.coarse_causal_chol(F.pad(coarse_t[:, :-1, :], (0, 0, 1, 0)))
            details_t[1] = details_t[1] + mix_w[2] * self.mid_causal_chol(F.pad(details_t[1][:, :-1, :], (0, 0, 1, 0)))
        if seq_t > 2:
            coarse_t = coarse_t + mix_w[1] * self.coarse_dilated_chol(F.pad(coarse_t[:, :-2, :], (0, 0, 2, 0)))
        d0_c, d0_f = details_t[0].chunk(2, dim=-1)
        th_k = self.kron_beamsplitter_theta.to(d0_c.dtype)
        c_th, s_th = torch.cos(th_k), torch.sin(th_k)
        details_t[0] = torch.cat([c_th * d0_c + s_th * d0_f, -s_th * d0_c + c_th * d0_f], dim=-1)
        wav_recon = self.wavelet_branch.upsample_amplitudes(coarse_t, details_t)
        rec_c, rec_f = wav_recon[..., : self.half_d], wav_recon[..., self.half_d :]
        ent_c = rec_c * (1.0 + 0.28 * torch.tanh(self.kron_modulation_chol(rec_f)))
        ent_f = rec_f * (1.0 + 0.28 * torch.tanh(self.kron_reverse_chol(rec_c)))
        return l2_normalize(torch.cat([ent_c, ent_f], dim=-1), dim=-1)

    def forward(
        self, psi: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        v_0: torch.Tensor | None, doc_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attn_out, v_first = self.attn(
            psi, cos=cos, sin=sin, v_0=v_0, window_size=self.window_size, doc_ids=doc_ids, return_v=True
        )
        psi_a = l2_normalize(psi + torch.tanh(self.attn_gate).to(psi.dtype) * attn_out, dim=-1)
        if psi_a.requires_grad and psi_a.numel() >= 4_194_304 and not torch.compiler.is_compiling():
            fused_wav = torch.utils.checkpoint.checkpoint(self._wavelet_kronecker_pathway, psi_a, use_reentrant=False)
        else:
            fused_wav = self._wavelet_kronecker_pathway(psi_a)
        inter_out = self.interferometer(psi_a)
        combined = l2_normalize(0.55 * inter_out + 0.45 * fused_wav, dim=-1)
        return l2_normalize(psi_a + torch.tanh(self.fused_gate).to(psi_a.dtype) * combined, dim=-1), fused_wav, v_first


class Radical2Pow16WaveletCholeskyGPT(nn.Module):
    """Standalone N=2^16 Wavelet-Cholesky Amplitude Transducer (Architectures 1, 2, & 3)."""

    def __init__(self, cfg: Radical2Pow16Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.pyramid = Tied2Pow16WaveletCholeskyPyramid(cfg)
        self.smear_gate = ForwardSmearGate(cfg.embed_dim)
        self.rope = UnitaryRoPE(head_dim=cfg.embed_dim // cfg.num_heads, max_seq_len=cfg.block_size * 2, zero_half_freq=True)
        block_map = {
            "multiscale_unet": WaveletCholeskyUNetBlock,
            "consistent_kronecker": ConsistentKroneckerWaveletBlock,
            "fused_kronecker_unet": FusedKroneckerWaveletUNetBlock,
        }
        block_cls = block_map[cfg.arch_mode]
        self.blocks = nn.ModuleList([block_cls(cfg, i) for i in range(cfg.num_layers)])
        if cfg.arch_mode == "fused_kronecker_unet":
            self.bigram_embed_gate = nn.Parameter(torch.full((cfg.embed_dim,), 0.14))
            self.bigram_mid_gate = nn.Parameter(torch.full((cfg.embed_dim,), 0.10))
            self.unet_skip_gates = nn.Parameter(torch.full((cfg.num_layers // 2, cfg.embed_dim), 0.12))
            self.diff_accum_gate = nn.Parameter(torch.full((cfg.embed_dim,), 0.45))
        self.purity_gate = ContextualBlochPurityGate(embed_dim=cfg.embed_dim, max_beta=64.0)
        self.kboson_raw_k = nn.Parameter(torch.tensor(math.log(math.exp(34.0 - 2.0) - 1.0), dtype=torch.float32))
        self.raw_lambda_diff = nn.Parameter(torch.tensor(-1.35, dtype=torch.float32))
        self.povm_log_prior = nn.Parameter(torch.zeros(cfg.vocab_size, dtype=torch.float32))
        self._training_progress: float = 0.0
        self._last_diffuse_eff_rank: float = float(cfg.diffuse_rank)
        self._last_sharp_eff_rank: float = 64.0
        self._last_lambda_diff: float = 0.14
        self._last_mean_beta: float = 28.0

    def set_training_progress(self, progress: float) -> None:
        """Update normalized training progress `s / S in [0, 1]` for Fix 3 `lambda_diff` cosine annealing."""
        self._training_progress = float(max(0.0, min(1.0, progress)))

    def current_lambda_diff(self) -> torch.Tensor:
        """Return effective diffuse bath mixing weight `lambda_diff(s)`."""
        if not self.cfg.use_kraus_retraction_and_lambda_anneal:
            return 0.08 + 0.24 * torch.sigmoid(self.raw_lambda_diff)
        prog = self._training_progress
        if self.cfg.use_last_layer_measurement_operator or self.cfg.use_traceless_observable:
            s_val, e_val = 0.148, 0.128
        else:
            s_val, e_val = float(self.cfg.lambda_diff_start), float(self.cfg.lambda_diff_end)
        if prog <= 0.18:
            sched = s_val * (0.90 + 0.10 * (prog / 0.18))
        else:
            u = (prog - 0.18) / 0.82
            sched = e_val + 0.5 * (s_val - e_val) * (1.0 + math.cos(math.pi * u))
        ref_sig = 1.0 / (1.0 + math.exp(1.35))
        learned_mod = 0.70 + 0.30 * (torch.sigmoid(self.raw_lambda_diff) / ref_sig)
        return torch.clamp(learned_mod * sched, min=0.035, max=0.22)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        psi_in_0 = self.pyramid.embed_tokens(idx)
        psi_in = psi_in_0
        bigram_psi: torch.Tensor | None = None
        if self.cfg.arch_mode == "fused_kronecker_unet" and self.cfg.use_dyadic_bigram_gate:
            bigram_psi = self.pyramid.embed_dyadic_bigram_phase(idx)
            psi_in = l2_normalize(psi_in_0 + torch.tanh(self.bigram_embed_gate).to(psi_in_0.dtype) * bigram_psi, dim=-1)

        psi = self.smear_gate(psi_in)
        doc_ids = compute_document_ids(idx)
        cos, sin = self.rope.get_cos_sin(idx.size(1))
        cos, sin = cos.to(psi.device), sin.to(psi.device)

        v_0: torch.Tensor | None = None
        if self.cfg.arch_mode == "fused_kronecker_unet" and self.cfg.use_tied_value_anchor:
            v_0 = self.pyramid.synthesize_value_anchor(psi_in, bigram_psi)

        psi_diff = psi
        n_half = self.cfg.num_layers // 2
        enc_skips: list[torch.Tensor] = []
        for l_idx, block in enumerate(self.blocks):
            if self.cfg.arch_mode == "fused_kronecker_unet" and l_idx == n_half and bigram_psi is not None:
                psi = l2_normalize(psi + torch.tanh(self.bigram_mid_gate).to(psi.dtype) * bigram_psi, dim=-1)
            psi, blk_diff, v_0 = block(psi, cos=cos, sin=sin, v_0=v_0, doc_ids=doc_ids)
            if self.cfg.arch_mode == "fused_kronecker_unet":
                g_d = torch.sigmoid(self.diff_accum_gate).to(psi_diff.dtype)
                psi_diff = l2_normalize((1.0 - g_d) * psi_diff + g_d * blk_diff, dim=-1)
                if self.cfg.use_outer_unet_skips:
                    if l_idx < n_half:
                        enc_skips.append(psi)
                    elif enc_skips:
                        skip_idx = l_idx - n_half
                        s_gate = torch.tanh(self.unet_skip_gates[min(skip_idx, self.unet_skip_gates.size(0) - 1)]).to(psi.dtype)
                        psi = l2_normalize(psi + s_gate * enc_skips.pop(), dim=-1)
            else:
                psi_diff = blk_diff

        beta_t = self.purity_gate(psi)
        if self.cfg.use_last_layer_measurement_operator:
            psi_s_out, psi_d_out, psi_vocab, w_k = self.pyramid.apply_last_layer_measurement(
                psi, psi_diff, psi_in_0, idx
            )
            m_sharp, m_diff = psi_vocab, psi_vocab
        elif self.cfg.use_traceless_observable and not self.cfg.use_convex_traceless_born:
            s_fwd = self.pyramid.sem_s_fwd.to(psi.dtype)
            psi_s_out = l2_normalize(F.linear(psi, s_fwd), dim=-1)
            psi_d_out = l2_normalize(F.linear(psi_diff, s_fwd), dim=-1)
            m_sharp, m_diff, w_k = self.pyramid.synthesize_vocab_detectors()
        else:
            psi_s_out, psi_d_out = psi, psi_diff
            m_sharp, m_diff, w_k = self.pyramid.synthesize_vocab_detectors()

        k_copies = 2.0 + F.softplus(self.kboson_raw_k)
        lam_diff = self.current_lambda_diff()
        if self.cfg.use_convex_traceless_born:
            trace_g = -abs(float(self.cfg.traceless_gamma))
        elif self.cfg.use_traceless_observable:
            trace_g = float(self.cfg.traceless_gamma)
        else:
            trace_g = 0.0
        if (self.cfg.use_last_layer_measurement_operator or self.cfg.use_traceless_observable) and not self.cfg.use_convex_traceless_born:
            beta_t = beta_t * (1.0 / math.sqrt(1.0 - 0.66 * 0.66))
        if trace_g > 0.0:
            beta_t = beta_t / (1.0 + 0.25 * trace_g)
        elif trace_g < 0.0:
            beta_t = beta_t / (1.0 + 0.45 * abs(trace_g))
            lam_diff = lam_diff * 0.72
        if not (self.training and self.cfg.use_fast_cholesky_and_compile):
            self._last_diffuse_eff_rank = float(1.0 / torch.clamp((w_k * w_k).sum(), min=1e-12).item())
            self._last_mean_beta = float(beta_t.detach().mean().item())
            self._last_lambda_diff = float(lam_diff.detach().item())

        if targets is not None and not return_logits:
            prec = self.cfg.readout_gemm_precision
            if self.cfg.use_fast_cholesky_and_compile and prec == "bf16":
                v_eff = self.cfg.valid_vocab_size
                rd_dtype = torch.bfloat16 if (psi_s_out.is_cuda and torch.is_autocast_enabled()) else psi_s_out.dtype
                m_s_eff = m_sharp[:v_eff].to(rd_dtype)
                m_d_eff = m_s_eff if self.cfg.use_last_layer_measurement_operator else m_diff[:v_eff].to(rd_dtype)
                loss = direct_chunked_two_beam_kboson_ce(
                    psi_sharp=psi_s_out.to(rd_dtype), psi_diff=psi_d_out.to(rd_dtype),
                    m_sharp=m_s_eff, m_diff=m_d_eff,
                    beta_t=beta_t, k_copies=k_copies, lambda_diff=lam_diff, targets=targets,
                    log_prior=self.povm_log_prior[:v_eff], chunk_size=4096,
                    valid_vocab_size=v_eff,
                    quadratic_diffuse_fidelity=self.cfg.quadratic_diffuse_fidelity,
                    traceless_gamma=trace_g, gemm_precision=prec,
                )
            else:
                loss = direct_chunked_two_beam_kboson_ce(
                    psi_sharp=psi_s_out, psi_diff=psi_d_out, m_sharp=m_sharp, m_diff=m_diff,
                    beta_t=beta_t, k_copies=k_copies, lambda_diff=lam_diff, targets=targets,
                    log_prior=self.povm_log_prior, chunk_size=4096,
                    valid_vocab_size=self.cfg.valid_vocab_size,
                    quadratic_diffuse_fidelity=self.cfg.quadratic_diffuse_fidelity,
                    traceless_gamma=trace_g, gemm_precision=prec,
                )
            if self.training and self.cfg.use_dyadic_hierarchical_born and self.cfg.dyadic_coarse_ce_weight > 0.0:
                loss = loss + float(self.cfg.dyadic_coarse_ce_weight) * self.pyramid.compute_dyadic_coarse_aux_loss(
                    psi_s_out, m_sharp, beta_t, targets
                )
            return None, loss

        logits = compute_two_beam_kboson_logits(
            psi_sharp=psi_s_out, psi_diff=psi_d_out, m_sharp=m_sharp, m_diff=m_diff,
            beta_t=beta_t, k_copies=k_copies, lambda_diff=lam_diff,
            log_prior=self.povm_log_prior,
            quadratic_diffuse_fidelity=self.cfg.quadratic_diffuse_fidelity,
            traceless_gamma=trace_g,
            valid_vocab_size=self.cfg.valid_vocab_size,
        )
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :, : self.cfg.valid_vocab_size].reshape(-1, self.cfg.valid_vocab_size),
                targets.reshape(-1), ignore_index=-1,
            )
            if self.training and self.cfg.use_dyadic_hierarchical_born and self.cfg.dyadic_coarse_ce_weight > 0.0:
                loss = loss + float(self.cfg.dyadic_coarse_ce_weight) * self.pyramid.compute_dyadic_coarse_aux_loss(
                    psi_s_out, m_sharp, beta_t, targets
                )
        return logits, loss


@torch.no_grad()
def initialize_radical_2pow16_with_ppmi_svd(
    model: Radical2Pow16WaveletCholeskyGPT,
    ppmi_embeddings: torch.Tensor,
    token_stream: torch.Tensor,
    prior_tokens: torch.Tensor,
) -> dict[str, float]:
    """Initialize the N=2^16 Wavelet-Cholesky pyramid via spectral bisection + predecessor bigram alignment."""
    from ..benchmarks.config_models import initialize_povm_log_prior

    device = next(model.parameters()).device
    cfg, pyr = model.cfg, model.pyramid
    d, half_d = cfg.embed_dim, cfg.embed_dim // 2
    plan038_mode = bool((cfg.use_last_layer_measurement_operator or cfg.use_traceless_observable) and not cfg.use_convex_traceless_born)
    perm_v_to_leaf, leaf_to_v = build_dyadic_2pow16_spectral_tree(ppmi_embeddings.to(device), cfg.ambient_n)
    pyr.perm_v_to_leaf.copy_(perm_v_to_leaf)
    pyr.leaf_to_v.copy_(leaf_to_v)

    v_in = min(ppmi_embeddings.size(0), cfg.vocab_size)
    a_self = torch.zeros((cfg.ambient_n, half_d), device=device, dtype=torch.float32)
    raw_a = F.pad(ppmi_embeddings[:v_in, :half_d].to(device).float(), (0, max(0, half_d - ppmi_embeddings.size(1))))
    if pyr.num_fine_banks > 1 and not cfg.use_convex_traceless_born:
        u_a, s_a, vh_a = torch.linalg.svd(raw_a - raw_a.mean(dim=0, keepdim=True), full_matrices=False)
        raw_a = u_a @ (torch.pow(s_a / s_a.mean().clamp_min(1e-6), 0.15 if plan038_mode else 0.20).unsqueeze(-1) * vh_a)
    a_self[:v_in] = l2_normalize(raw_a, dim=-1)
    if pyr.num_fine_banks > 1 and cfg.ambient_n > v_in:
        gen_pad = torch.Generator(device=device).manual_seed(2037)
        a_self[v_in:] = l2_normalize(torch.randn((cfg.ambient_n - v_in, half_d), generator=gen_pad, device=device), dim=-1)

    toks = token_stream.to(device=device, dtype=torch.long)
    src, dst = toks[:-1].clamp(0, cfg.vocab_size - 1), toks[1:].clamp(0, cfg.vocab_size - 1)
    b_pred = torch.zeros_like(a_self)
    if pyr.num_fine_banks > 1 and not cfg.use_convex_traceless_born:
        a_src_c = a_self.index_select(0, src)
        b_pred.index_add_(0, dst, a_src_c - a_src_c.mean(dim=0, keepdim=True))
        unseen = b_pred.norm(dim=-1) < 1e-5
        if bool(unseen.any()):
            b_pred[unseen] = torch.roll(a_self[unseen], shifts=7, dims=-1)
        u_bp, s_bp, vh_bp = torch.linalg.svd(b_pred[:v_in] - b_pred[:v_in].mean(dim=0, keepdim=True), full_matrices=False)
        b_pred[:v_in] = u_bp @ (torch.pow(s_bp / s_bp.mean().clamp_min(1e-6), 0.35 if plan038_mode else 0.55).unsqueeze(-1) * vh_bp)
    else:
        b_pred.index_add_(0, dst, a_self.index_select(0, src))
    b_pred = l2_normalize(b_pred, dim=-1)
    b_orth = l2_normalize(b_pred - (b_pred * a_self).sum(dim=-1, keepdim=True) * a_self, dim=-1)

    c_src, c_dst = perm_v_to_leaf.index_select(0, src) >> 8, perm_v_to_leaf.index_select(0, dst) >> 8
    trans_flat = torch.zeros((256 * 256,), device=device, dtype=torch.float32)
    trans_flat.scatter_add_(0, c_dst * 256 + c_src, torch.ones_like(c_src, dtype=torch.float32))
    trans_mat = torch.log1p(trans_flat.view(256, 256))
    trans_mat = (trans_mat - trans_mat.mean(dim=0, keepdim=True)).clamp_min(0.0)
    pyr.coarse_transition.copy_(F.normalize(trans_mat, p=2, dim=-1) * 0.20)

    target_v = torch.stack((a_self, b_orth), dim=-1).reshape(cfg.ambient_n, d) * math.sqrt(0.5)
    y_leaf = target_v.index_select(0, leaf_to_v)
    w256 = y_leaf.view(256, 256, d).mean(dim=1)
    r2048 = y_leaf.view(2048, 32, d).mean(dim=1) - w256.repeat_interleave(8, dim=0)
    r8192_full = y_leaf.view(8192, 8, d).mean(dim=1) - (w256.repeat_interleave(32, dim=0) + r2048.repeat_interleave(4, dim=0))
    _, _, vh_mid = torch.linalg.svd(r8192_full, full_matrices=False)
    up_mid = vh_mid[: cfg.mid_detail_dim, :]
    w8192 = r8192_full @ up_mid.t()

    approx_8192 = (w256.repeat_interleave(32, dim=0) + r2048.repeat_interleave(4, dim=0) + w8192 @ up_mid).repeat_interleave(8, dim=0)
    r65536_full = y_leaf - approx_8192
    pyr.wav_256.copy_(w256)
    pyr.wav_2048.copy_(r2048)
    pyr.wav_8192.copy_(w8192)
    pyr.up_8192.copy_(up_mid)

    if pyr.num_fine_banks == 1:
        _, _, vh_fine = torch.linalg.svd(r65536_full[:16384], full_matrices=False)
        up_fine = vh_fine[: cfg.fine_detail_dim, :]
        pyr.wav_65536.copy_(r65536_full @ up_fine.t())
        pyr.up_65536.copy_(up_fine)
    elif plan038_mode:
        num_b, bank_len, d_f = pyr.num_fine_banks, cfg.ambient_n // pyr.num_fine_banks, cfg.fine_detail_dim
        r_banks = r65536_full.view(num_b, bank_len, d)
        up_global = torch.linalg.svd(r65536_full[:16384], full_matrices=False)[2][:d_f, :]
        up_banks = torch.empty((num_b, d_f, d), device=device, dtype=torch.float32)
        w_banks = torch.empty((num_b, bank_len, d_f), device=device, dtype=torch.float32)
        for b in range(num_b):
            vh_b = torch.linalg.svd(r_banks[b], full_matrices=False)[2][:d_f, :]
            q_b, r_b = torch.linalg.qr((0.80 * up_global + 0.20 * vh_b).t())
            r_diag = torch.diagonal(r_b)
            up_b = (q_b * torch.where(r_diag >= 0.0, torch.ones_like(r_diag), -torch.ones_like(r_diag)).unsqueeze(0)).t()
            up_banks[b].copy_(up_b)
            w_banks[b].copy_(r_banks[b] @ up_b.t())
        pyr.up_65536.copy_(up_banks)
        pyr.wav_65536.copy_(w_banks.view(cfg.ambient_n, d_f))
        pyr.scale_gains.copy_(torch.ones(4, device=device, dtype=torch.float32))
    else:
        num_b, bank_len, d_f = pyr.num_fine_banks, cfg.ambient_n // pyr.num_fine_banks, cfg.fine_detail_dim
        r_banks = r65536_full.view(num_b, bank_len, d)
        up_banks = torch.empty((num_b, d_f, d), device=device, dtype=torch.float32)
        w_banks = torch.empty((num_b, bank_len, d_f), device=device, dtype=torch.float32)
        gen_f = torch.Generator(device=device).manual_seed(1337)
        eye_full = torch.eye(d, device=device, dtype=torch.float32)
        grp_sz = max(1, d // d_f)
        for g_start in range(0, num_b, grp_sz):
            g_end = min(num_b, g_start + grp_sz)
            grp_rows = [
                0.65 * torch.linalg.svd(r_banks[b], full_matrices=False)[2][:d_f]
                + 0.35 * torch.roll(eye_full, shifts=((b * d_f) + (g_start * 19)) % d, dims=0)[:d_f]
                for b in range(g_start, g_end)
            ]
            grp_mat = torch.cat(grp_rows, dim=0)
            if grp_mat.size(0) < d:
                grp_mat = torch.cat([grp_mat, eye_full[: d - grp_mat.size(0)]], dim=0)
            q_orth = torch.linalg.qr(grp_mat.t())[0].t()
            for idx_in_g, b in enumerate(range(g_start, g_end)):
                up_b = q_orth[idx_in_g * d_f : (idx_in_g + 1) * d_f]
                up_banks[b].copy_(up_b)
                iso_jit = torch.randn((bank_len, d_f), generator=gen_f, device=device) * (0.25 / math.sqrt(d_f))
                sib = ((r_banks[b] @ up_b.t()) + iso_jit).view(bank_len // 8, 8, d_f)
                sib = (sib - sib.mean(dim=1, keepdim=True)).reshape(bank_len, d_f)
                u_w, s_w, vh_w = torch.linalg.svd(sib, full_matrices=False)
                w_eq = u_w @ (torch.pow(s_w / s_w.mean().clamp_min(1e-6), 0.15).unsqueeze(-1) * vh_w)
                w_banks[b].copy_(l2_normalize(w_eq, dim=-1) * 0.85)
        pyr.up_65536.copy_(up_banks)
        pyr.wav_65536.copy_(w_banks.view(cfg.ambient_n, d_f))
        pyr.scale_gains.copy_(torch.tensor([0.65, 0.80, 0.90, 1.50], device=device, dtype=torch.float32))

    rms_list = []
    for w_t in (pyr.wav_256, pyr.wav_2048, pyr.wav_8192, pyr.wav_65536):
        wf = w_t.detach().float()
        wf = wf - wf.mean(dim=0, keepdim=True)
        rms_list.append(float(torch.sqrt((wf * wf).sum() / float(wf.size(0)) + 1e-12).item()))
    pyr.init_wav_rms.copy_(torch.tensor(rms_list, device=device, dtype=torch.float32))

    if hasattr(pyr, "pred_wav_2048"):
        psi_all_v = pyr.embed_tokens(torch.arange(cfg.vocab_size, device=device))
        pred_bucket = torch.zeros((2048, d), device=device, dtype=torch.float32)
        dst_bucket = perm_v_to_leaf.index_select(0, dst) >> 5
        a_s_mean = psi_all_v.index_select(0, src[:262144]).mean(dim=0, keepdim=True) if pyr.num_fine_banks > 1 else 0.0
        for start_i in range(0, src.numel(), 262144):
            s_c = src[start_i : start_i + 262144]
            pred_bucket.index_add_(0, dst_bucket[start_i : start_i + 262144], psi_all_v.index_select(0, s_c) - a_s_mean)
        if pyr.num_fine_banks > 1:
            u_pb, s_pb, vh_pb = torch.linalg.svd(pred_bucket - pred_bucket.mean(dim=0, keepdim=True), full_matrices=False)
            pred_bucket = u_pb @ (torch.pow(s_pb / s_pb.mean().clamp_min(1e-6), 0.60).unsqueeze(-1) * vh_pb)
            pyr.pred_wav_2048.copy_(l2_normalize(pred_bucket, dim=-1) * 0.48)
        else:
            pred_bucket[:, 0::2] *= 0.25
            pyr.pred_wav_2048.copy_(l2_normalize(pred_bucket, dim=-1) * 0.45)

    gen = torch.Generator(device=device).manual_seed(42)
    eye_d = torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0).expand(cfg.diffuse_rank, -1, -1)
    noise = torch.randn((cfg.diffuse_rank, d, d), generator=gen, device=device) * (0.18 / math.sqrt(d))
    q_batch, _ = torch.linalg.qr(0.92 * eye_d + noise)
    pyr.kraus_generators.copy_(q_batch)
    pyr.retract_kraus_generators()

    initialize_povm_log_prior(model, prior_tokens.to(device))
    v_probe = torch.arange(min(2048, cfg.valid_vocab_size), device=device)
    psi_in = pyr.embed_tokens(v_probe)
    n_pairs = min(4096, src.numel())
    u_s, v_s = src[:n_pairs], dst[:n_pairs]
    non_self = u_s != v_s
    if bool(non_self.any().item()):
        u_s, v_s = u_s[non_self], v_s[non_self]
    psi_u = pyr.embed_tokens(u_s)
    v_rand = torch.remainder(u_s + 257, cfg.valid_vocab_size)

    if cfg.use_last_layer_measurement_operator:
        psi_s_pr, _, psi_vocab, _ = pyr.apply_last_layer_measurement(psi_in, psi_in, psi_in, v_probe)
        max_self_ov = float((psi_s_pr * psi_vocab[: v_probe.numel()]).sum(dim=-1).abs().max().item())
        sharp_rank = pyr.compute_sharp_eff_rank(psi_vocab)
        psi_u_meas, _, _, _ = pyr.apply_last_layer_measurement(psi_u, psi_u, psi_u, u_s)
        bg_sq = float((psi_u_meas * psi_vocab.index_select(0, v_s)).sum(dim=-1).pow(2).mean().item())
        rnd_sq = max(float((psi_u_meas * psi_vocab.index_select(0, v_rand)).sum(dim=-1).pow(2).mean().item()), 1e-12)
    else:
        m_sharp, _, _ = pyr.synthesize_vocab_detectors()
        s_fwd_q = pyr.sem_s_fwd.to(psi_in.dtype)
        psi_in_q = l2_normalize(F.linear(psi_in, s_fwd_q), dim=-1) if (cfg.use_traceless_observable and not cfg.use_convex_traceless_born) else psi_in
        psi_u_q = l2_normalize(F.linear(psi_u, s_fwd_q), dim=-1) if (cfg.use_traceless_observable and not cfg.use_convex_traceless_born) else psi_u
        max_self_ov = float((psi_in_q * m_sharp[: v_probe.numel()]).sum(dim=-1).abs().max().item())
        sharp_rank = pyr.compute_sharp_eff_rank(m_sharp)
        m_v = m_sharp.index_select(0, v_s)
        bg_sq = float((psi_u_q * m_v).sum(dim=-1).pow(2).mean().item())
        rnd_sq = max(float((psi_u_q * m_sharp.index_select(0, v_rand)).sum(dim=-1).pow(2).mean().item()), 1e-12)

    model._last_sharp_eff_rank = sharp_rank
    return {
        "max_self_overlap": max_self_ov,
        "directed_bigram_overlap_ratio": float(bg_sq / rnd_sq),
        "sharp_eff_rank": sharp_rank,
        "kraus_stiefel_err": pyr.compute_kraus_stiefel_err(),
    }


def build_radical_2pow16_optimizers(
    model: Radical2Pow16WaveletCholeskyGPT, muon_lr: float = 0.024, wav_lr: float = 0.006, head_lr: float = 0.003,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build StiefelMuon + AdamW optimizers with zero-WD gentle LR on the N=2^16 wavelet pyramid."""
    muon_params, basis_muon_params, wav_params, chol_params, body_params = [], [], [], [], []
    plan038_mode = bool((model.cfg.use_last_layer_measurement_operator or model.cfg.use_traceless_observable) and not model.cfg.use_convex_traceless_born)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if model.cfg.use_kraus_retraction_and_lambda_anneal and "kraus_generators" in name:
            (basis_muon_params if plan038_mode else muon_params).append(p)
        elif plan038_mode and any(k in name for k in ("up_8192", "up_65536")):
            basis_muon_params.append(p)
        elif any(k in name for k in ("wav_256", "wav_2048", "wav_8192", "wav_65536", "up_8192", "up_65536", "pred_wav_2048")):
            wav_params.append(p)
        elif any(k in name for k in ("chol", "kraus", "purity_gate", "kboson", "lambda_diff", "log_prior", "transition", "scale_gains", "theta", "lift", "pred_mix_gain")):
            chol_params.append(p)
        elif p.ndim == 2 and min(p.shape) >= 32 and max(p.shape) <= 2048:
            muon_params.append(p)
        else:
            body_params.append(p)
    muon_groups = [{"params": muon_params, "lr": muon_lr, "base_lr": muon_lr, "weight_decay": 0.01}]
    if basis_muon_params:
        muon_groups.append({"params": basis_muon_params, "lr": 0.0012, "base_lr": 0.0012, "weight_decay": 0.0, "stiefel_penalty": 0.01})
    muon_opt = StiefelMuon(
        [g for g in muon_groups if g["params"]], lr=muon_lr, momentum=0.95, nesterov=True, weight_decay=0.01,
        decay_mode="cautious", stiefel_penalty=0.02, retract_every=2,
        polar_method="polar_express", use_normuon=True, use_twin_rail=True,
    )
    adam_groups = [
        {"params": wav_params, "lr": wav_lr, "base_lr": wav_lr, "weight_decay": 0.0, "group_name": "wavelet_2pow16"},
        {"params": chol_params, "lr": head_lr, "base_lr": head_lr, "weight_decay": 0.0, "group_name": "cholesky_readout"},
        {"params": body_params, "lr": head_lr, "base_lr": head_lr, "weight_decay": 0.01, "group_name": "body"},
    ]
    return muon_opt, torch.optim.AdamW([g for g in adam_groups if g["params"]], lr=head_lr, betas=(0.9, 0.95))
