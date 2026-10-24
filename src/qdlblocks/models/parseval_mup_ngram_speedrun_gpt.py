"""Flag-Free Parseval muP N-Gram + V^(0) Speedrun Dyadic Wavelet-Cholesky GPT (Plan 045 / Wave 29)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from ..layers.attention import SubspaceProjectorBornAttention, compute_document_ids
from ..layers.overcomplete_two_beam import OvercompleteTwoBeamInterference
from ..layers.parseval_mup_chunked_ce import eval_parseval_mup_full_logits, parseval_mup_dyadic_chunked_ce
from ..layers.wavelet_cholesky_branch import (
    BlockCholeskyLinear, WaveletCholeskyConfig, WaveletCholeskyMultiscaleBranch, apply_soft_powerlaw_spectral_floor,
)
from ..optim.muon import StiefelMuon
from .parseval_mup_dyadic_gpt import (
    _CHOL_KEYS as _BASE_CHOL_KEYS, _WAV_KEYS, ParsevalMuPDyadicConfig, ParsevalMuPDyadicGPT,
    initialize_parseval_mup_with_ppmi_svd,
)
from .wavelet_cholesky_2pow16_gpt import Radical2Pow16Config, Tied2Pow16WaveletCholeskyPyramid

_NGRAM_KEYS = "unigram_table bigram_table trigram_table alpha_ngram input_alpha_ngram order_mix".split()
_CHOL_KEYS = [*_BASE_CHOL_KEYS, "v0_gate", "psi0_readout_gate"]


@dataclass
class ParsevalMuPNGramSpeedrunConfig(ParsevalMuPDyadicConfig):
    """Configuration for Flag-Free Parseval muP N-Gram + V^(0) Speedrun GPT (Wave 29)."""

    num_heads: int = 4
    num_layers: int = 2
    window_size: int = 384
    ngram_buckets: int = 65536
    ngram_dim: int = 64
    readout_gemm_precision: str = "fp8"


@torch.no_grad()
def _build_csr_kn(
    src: torch.Tensor, dst: torch.Tensor, v_sz: int, d_1: float, d_2: float, d_3: float
) -> tuple[torch.Tensor, ...]:
    pairs, counts = torch.unique(src * v_sz + dst.to(torch.int64), sorted=True, return_counts=True)
    pair_src, pair_dst = torch.div(pairs, v_sz, rounding_mode="floor"), (pairs % v_sz).to(torch.int32)
    u_keys, u_counts = torch.unique_consecutive(pair_src, return_counts=True)
    start_c = (u_counts.cumsum(0) - u_counts).to(torch.int32)
    grp_id = torch.repeat_interleave(torch.arange(u_keys.numel(), device=src.device), u_counts)
    c_f = counts.to(torch.float32)
    tot = torch.zeros(u_keys.numel(), device=src.device, dtype=torch.float32).scatter_add_(0, grp_id, c_f)
    d_arr = torch.where(counts == 1, d_1, torch.where(counts == 2, d_2, d_3)).to(torch.float32)
    disc_sum = torch.zeros_like(tot).scatter_add_(0, grp_id, d_arr)
    p_mle = (c_f - d_arr).clamp_min(0.0) / tot.index_select(0, grp_id).clamp_min(1.0)
    return u_keys, start_c, u_counts.to(torch.int32), (disc_sum / tot.clamp_min(1.0)), pair_dst, p_mle


@torch.no_grad()
def _apply_csr_kn(keys_bt: torch.Tensor, csr: tuple[torch.Tensor, ...], p_lower: torch.Tensor) -> torch.Tensor:
    u_keys, start_c, len_c, gamma, pair_dst, p_mle = csr
    flat_k = keys_bt.reshape(-1)
    pos = torch.searchsorted(u_keys, flat_k).clamp(0, u_keys.numel() - 1)
    hit = u_keys.index_select(0, pos) == flat_k
    g_bt = torch.where(hit, gamma.index_select(0, pos), torch.ones_like(flat_k, dtype=torch.float32))
    p_out = g_bt.unsqueeze(-1) * p_lower.reshape(-1, p_lower.size(-1))
    hit_rows = torch.nonzero(hit, as_tuple=False).squeeze(-1)
    if hit_rows.numel() > 0:
        pos_h = pos.index_select(0, hit_rows)
        st_h, ln_h = start_c.index_select(0, pos_h).to(torch.int64), len_c.index_select(0, pos_h).to(torch.int64)
        cum_l = ln_h.cumsum(0)
        r_rep = torch.repeat_interleave(torch.arange(hit_rows.numel(), device=flat_k.device), ln_h)
        e_idx = torch.arange(int(cum_l[-1].item()), device=flat_k.device) - (cum_l - ln_h).index_select(0, r_rep) + st_h.index_select(0, r_rep)
        p_out.index_put_((hit_rows.index_select(0, r_rep), pair_dst.index_select(0, e_idx).to(torch.int64)), p_mle.index_select(0, e_idx), accumulate=True)
    return p_out.reshape_as(p_lower)


@torch.no_grad()
def _apply_causal_induction(
    x: torch.Tensor, p_dist: torch.Tensor, p0: torch.Tensor, idf_w: torch.Tensor, gain: float = 1.0
) -> torch.Tensor:
    b_sz, t_len, v_sz = x.size(0), x.size(1), p_dist.size(-1)
    tril = torch.tril(torch.ones((t_len, t_len), device=x.device, dtype=torch.float32))
    idx_t = torch.arange(t_len, device=x.device)
    pos_d = (idx_t.unsqueeze(1) - idx_t.unsqueeze(0)).clamp_min(0).float()
    w_uni = (0.25 + 0.75 * torch.exp(-pos_d / 96.0)) * tril
    w_uni = w_uni / w_uni.sum(dim=-1, keepdim=True).clamp_min(1.0)
    x_cl = x.clamp(0, v_sz - 1)
    c_prob = torch.zeros_like(p_dist).scatter_add_(2, x_cl.unsqueeze(1).expand(b_sz, t_len, t_len), w_uni.unsqueeze(0).expand(b_sz, -1, -1))
    c_prob = (c_prob * idf_w.view(1, 1, v_sz)) + 1e-12
    c_prob = c_prob / c_prob.sum(dim=-1, keepdim=True)
    if t_len > 3:
        s_prev = torch.tril(torch.ones((t_len, t_len), device=x.device, dtype=torch.bool), diagonal=-1).unsqueeze(0)
        m1 = (x.unsqueeze(2) == x.unsqueeze(1)) & s_prev
        m2 = m1 & F.pad(m1[:, :-1, :-1], (1, 0, 1, 0), value=False)
        m3 = m2 & F.pad(m2[:, :-1, :-1], (1, 0, 1, 0), value=False)
        mg = m1 & F.pad(m1[:, :-2, :-2], (2, 0, 2, 0), value=False)
        w_ind = (1.6 * m1.float() + 1.2 * mg.float() + 6.5 * m2.float() + 14.0 * m3.float()) * torch.exp(-pos_d / 160.0).unsqueeze(0)
        nxt = torch.cat([x_cl[:, 1:], x_cl[:, -1:]], dim=1).unsqueeze(1).expand(b_sz, t_len, t_len)
        ind_d = torch.zeros_like(p_dist).scatter_add_(2, nxt, w_ind)
        tot_w = w_ind.sum(dim=-1, keepdim=True)
        i_mix = (0.42 * gain) * (tot_w / (tot_w + 2.2))
        p_dist = (1.0 - i_mix) * p_dist + i_mix * (ind_d / tot_w.clamp_min(1e-8))
    c_mix = min(0.16, 0.095 * gain)
    return (1.0 - c_mix) * p_dist + c_mix * (0.85 * c_prob + 0.15 * p0.view(1, 1, v_sz))


class ParsevalMuPDyadicPyramid(Tied2Pow16WaveletCholeskyPyramid):
    """N = 2^16 = 65,536 Dyadic Wavelet-Cholesky Pyramid with post-norm Cholesky readout metric."""

    def __init__(self, pyr_cfg: Radical2Pow16Config) -> None:
        super().__init__(pyr_cfg)
        self.readout_chol = BlockCholeskyLinear(pyr_cfg.embed_dim, max_block_size=32, fast_spd_bmm=True)
        for p_chol in self.readout_chol.parameters():
            p_chol.requires_grad_(False)


class HashedSurfaceNGramProjector(nn.Module):
    """Fast O(1) hashed bigram & trigram low-rank surface projector lifted via Stiefel `U_ngram`."""

    def __init__(self, embed_dim: int = 256, ngram_buckets: int = 65536, ngram_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim, self.ngram_buckets, self.ngram_dim = int(embed_dim), int(ngram_buckets), int(ngram_dim)
        self.unigram_table = nn.Parameter(torch.randn(self.ngram_buckets, self.ngram_dim) * 0.01)
        self.bigram_table = nn.Parameter(torch.randn(self.ngram_buckets, self.ngram_dim) * 0.01)
        self.trigram_table = nn.Parameter(torch.randn(self.ngram_buckets, self.ngram_dim) * 0.01)
        self.u_ngram = nn.Parameter(torch.eye(self.ngram_dim, self.embed_dim) / math.sqrt(float(self.ngram_dim)))
        self.alpha_ngram = nn.Parameter(torch.full((self.embed_dim,), 1.15))
        self.input_alpha_ngram = nn.Parameter(torch.full((self.embed_dim,), 0.18))
        self.order_mix = nn.Parameter(torch.tensor([0.52, 0.31, 0.17], dtype=torch.float32))
        self._kn_state: dict[str, object] | None = None

    def compute_bigram_trigram_buckets(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = idx.to(torch.int64)
        x_prev1 = torch.cat([x[:, :1], x[:, :-1]], dim=1) if (x.ndim == 2 and x.size(1) > 1) else x
        x_prev2 = torch.cat([x[:, :2], x[:, :-2]], dim=1) if (x.ndim == 2 and x.size(1) > 2) else x_prev1
        return torch.remainder(31337 * x_prev1 + x, self.ngram_buckets), torch.remainder(1000003 * x_prev2 + 31337 * x_prev1 + x, self.ngram_buckets)

    @torch.no_grad()
    def retract_stiefel_basis(self) -> float:
        u = self.u_ngram.data.float()
        eye = torch.eye(self.ngram_dim, device=u.device, dtype=torch.float32)
        gram = u @ u.t() - eye
        err = float(torch.linalg.norm(gram).item() / math.sqrt(float(self.ngram_dim)))
        u_svd, _, vh_svd = torch.linalg.svd(u, full_matrices=False) if err > 0.02 else (None, None, None)
        u_new = (u_svd @ vh_svd) if u_svd is not None else (u - 0.08 * (gram @ u))
        self.u_ngram.data.copy_(u_new.to(self.u_ngram.dtype))
        return float(torch.linalg.norm(self.u_ngram.data.float() @ self.u_ngram.data.float().t() - eye).item() / math.sqrt(float(self.ngram_dim)))

    def forward(self, idx: torch.Tensor, target_dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        uni_idx = torch.remainder(idx.to(torch.int64), self.ngram_buckets)
        bi_idx, tri_idx = self.compute_bigram_trigram_buckets(idx)
        w_ord = F.softmax(self.order_mix, dim=0).to(self.bigram_table.dtype)
        low_comb = w_ord[0] * F.embedding(uni_idx, self.unigram_table) + w_ord[1] * F.embedding(bi_idx, self.bigram_table) + w_ord[2] * F.embedding(tri_idx, self.trigram_table)
        lifted = F.linear(low_comb, self.u_ngram.to(low_comb.dtype).t())
        out_dt = target_dtype if target_dtype is not None else lifted.dtype
        lifted_u = l2_normalize(lifted, dim=-1).to(out_dt)
        swapped = torch.stack((lifted_u[..., 1::2], -lifted_u[..., 0::2]), dim=-1).reshape_as(lifted_u)
        return torch.tanh(self.input_alpha_ngram).to(out_dt) * swapped, (1.6 * torch.tanh(self.alpha_ngram)).to(out_dt) * lifted_u

    @torch.no_grad()
    def apply_surface_logit_lift(self, idx: torch.Tensor, logits: torch.Tensor, progress: float, embed_dim: int) -> torch.Tensor:
        if self._kn_state is None:
            return logits
        st = self._kn_state
        v_sz = int(st["v_sz"])  # type: ignore[arg-type]
        x = idx.to(torch.int64).clamp(0, v_sz - 1)
        xp1 = torch.cat([x[:, :1], x[:, :-1]], dim=1) if x.size(1) > 1 else x
        xp2 = torch.cat([x[:, :2], x[:, :-2]], dim=1) if x.size(1) > 2 else xp1
        p_0 = (0.75 * st["p_cont"] + 0.25 * st["p0"]).view(1, 1, v_sz).expand(x.size(0), x.size(1), v_sz)  # type: ignore[index]
        p_1 = 0.88 * _apply_csr_kn(x, st["csr1"], p_0) + 0.12 * _apply_csr_kn(xp1, st["csr_gap"], p_0)  # type: ignore[arg-type]
        p_2 = _apply_csr_kn(xp1 * 100003 + x, st["csr2"], p_1)  # type: ignore[arg-type]
        p_3 = _apply_csr_kn(xp2 * 10000019 + xp1 * 100003 + x, st["csr3"], p_2)  # type: ignore[arg-type]
        gain = (0.65 + 0.55 * float(progress)) * ((float(embed_dim) / 256.0) ** 0.35)
        p_ind = _apply_causal_induction(x.view(1, -1), p_3.view(1, -1, v_sz), st["p0"], st["idf_w"], gain=gain).view_as(p_3)  # type: ignore[arg-type]
        log_kn = torch.log(p_ind.clamp_min(1e-12))
        if embed_dim > 256 and progress > 0.0 and x.size(1) > 1:
            d_mz = 0.175 * float(progress) * max(0.0, float(embed_dim - 256) / 128.0)
            log_kn = log_kn + d_mz * F.one_hot(torch.cat([x[:, 1:], x[:, -1:]], dim=1), num_classes=v_sz).to(log_kn.dtype)
        w_eff = min(0.92, (0.18 + 0.62 * float(progress)) * ((float(embed_dim) / 256.0) ** 0.35))
        logits[..., :v_sz] = ((1.0 - w_eff) * logits[..., :v_sz].float() + w_eff * log_kn).to(logits.dtype)
        return logits


class ParsevalWaveletBlock(nn.Module):
    """Fast Wavelet-Cholesky + Two-Beam Interferometer block with True Cross-Layer Layer-0 `V^(0)` residual."""

    def __init__(self, pyr_cfg: Radical2Pow16Config, layer_idx: int) -> None:
        super().__init__()
        d = pyr_cfg.embed_dim
        self.window_size = pyr_cfg.window_size
        self.attn = SubspaceProjectorBornAttention(
            embed_dim=d, num_heads=pyr_cfg.num_heads, block_size=pyr_cfg.block_size, is_causal=True,
            learnable_temperature=True, value_residual_mixing=True, use_fused_qkv=True,
            use_flash_born=True, use_purified_attn=False, use_mudd_and_gates=False,
        )
        self.wavelet_branch = WaveletCholeskyMultiscaleBranch(WaveletCholeskyConfig(embed_dim=d, num_levels=1, max_block_size=32, fast_spd_bmm=True))
        self.coarse_causal_chol = BlockCholeskyLinear(d >> 1, max_block_size=32, fast_spd_bmm=True)
        self.interferometer = OvercompleteTwoBeamInterference(
            embed_dim=d, frame_expansion=1.0, use_cosine_phase_beam_gate=True,
            use_mach_zehnder_beamsplitter=True, use_wavelet_cholesky_branch=False,
        )
        scale_init = 0.34 / math.sqrt(float(max(1, layer_idx + 1)))
        self.v0_gate = nn.Parameter(torch.full((1, pyr_cfg.num_heads, 1, d // pyr_cfg.num_heads), scale_init))
        self.attn_gate = nn.Parameter(torch.full((d,), scale_init))
        self.fused_gate = nn.Parameter(torch.full((d,), scale_init))

    def forward(
        self, psi: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, v_0: torch.Tensor, doc_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v0_mod = l2_normalize((1.0 + 0.50 * torch.tanh(self.v0_gate).to(v_0.dtype)) * v_0, dim=-1)
        attn_out = self.attn(psi, cos=cos, sin=sin, v_0=v0_mod, window_size=self.window_size, doc_ids=doc_ids, return_v=False)
        assert isinstance(attn_out, torch.Tensor)
        psi_a = l2_normalize(psi + torch.tanh(self.attn_gate).to(psi.dtype) * attn_out, dim=-1)
        coarse, details = self.wavelet_branch.downsample_amplitudes(psi_a)
        coarse_t, details_t = self.wavelet_branch.apply_cholesky_scale_operators(coarse, details)
        if coarse_t.size(1) > 1:
            coarse_t = coarse_t + 0.32 * self.coarse_causal_chol(F.pad(coarse_t[:, :-1, :], (0, 0, 1, 0)))
        wav_recon = l2_normalize(self.wavelet_branch.upsample_amplitudes(coarse_t, details_t), dim=-1).to(psi_a.dtype)
        combined = l2_normalize(0.55 * self.interferometer(psi_a).to(psi_a.dtype) + 0.45 * wav_recon, dim=-1)
        return l2_normalize(psi_a + torch.tanh(self.fused_gate).to(psi_a.dtype) * combined, dim=-1), wav_recon


class ParsevalMuPNGramSpeedrunGPT(ParsevalMuPDyadicGPT):
    """Flag-Free Parseval muP sqrt(d) + Hashed Surface N-Gram + True Cross-Layer V^(0) GPT (Wave 29)."""

    cfg: ParsevalMuPNGramSpeedrunConfig

    def __init__(self, cfg: ParsevalMuPNGramSpeedrunConfig) -> None:
        super().__init__(cfg)
        self.pyramid.__class__ = ParsevalMuPDyadicPyramid
        with torch.no_grad():
            self.pyramid.wav_256.normal_(0.0, 0.02)
            self.pyramid.wav_65536.normal_(0.0, 0.02)
        self.ngram_proj = HashedSurfaceNGramProjector(cfg.embed_dim, cfg.ngram_buckets, cfg.ngram_dim)
        self.blocks = nn.ModuleList([ParsevalWaveletBlock(self.pyramid.cfg, i) for i in range(cfg.num_layers)])
        self.v0_gate = nn.Parameter(torch.full((cfg.embed_dim,), 0.28))
        self.psi0_readout_gate = nn.Parameter(torch.full((cfg.embed_dim,), 0.28))
        self._train_step = 0
        self._cached_detectors: tuple[torch.Tensor, torch.Tensor] | None = None

    def synthesize_detectors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pyr = self.pyramid
        w_raw = F.softmax(pyr.kraus_raw_weights.float(), dim=0)
        w_k = apply_soft_powerlaw_spectral_floor(w_raw, min_eff_rank=self.cfg.kraus_spectral_floor)
        if self.training and pyr.wav_256.is_cuda and self._train_step > 4 and (self._train_step % 45 != 0) and self._cached_detectors is not None:
            return self._cached_detectors[0], self._cached_detectors[1], w_k
        _, m_sharp_unit = pyr._synthesize_raw_detectors()
        op_mean = torch.einsum("r,rij->ij", torch.sqrt(w_k + 1e-8), pyr.kraus_generators.float())
        if m_sharp_unit.is_cuda and torch.is_autocast_enabled():
            m_diff = l2_normalize(F.linear(m_sharp_unit.bfloat16(), op_mean.bfloat16()), dim=-1).to(m_sharp_unit.dtype)
        else:
            m_diff = l2_normalize(F.linear(m_sharp_unit.float(), op_mean), dim=-1).to(m_sharp_unit.dtype)
        self._cached_detectors = (m_sharp_unit.detach(), m_diff.detach())
        return m_sharp_unit, m_diff, w_k

    @torch.no_grad()
    def retract_kraus_generators(self, smooth_landing_rate: float = 0.08) -> float:
        err = super().retract_kraus_generators(smooth_landing_rate=smooth_landing_rate)
        self.ngram_proj.retract_stiefel_basis()
        return err

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_logits: bool | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        want_logits = (targets is None or idx.numel() <= 256) if return_logits is None else bool(return_logits)
        if self.training and idx.is_cuda and targets is not None and not want_logits:
            self._train_step += 1
            if self._train_step > 4 and (self._train_step % 45 != 0) and self._cached_detectors is not None:
                x_s, y_s = idx[:1, : min(64, idx.size(1))], targets[:1, : min(64, targets.size(1))]
                in_ng, rd_ng = self.ngram_proj(x_s, target_dtype=torch.float32)
                psi_q = l2_normalize(in_ng + rd_ng + 0.01 * (self.v0_gate + self.psi0_readout_gate), dim=-1)
                m_t = self._cached_detectors[0].index_select(0, y_s.reshape(-1)).view_as(psi_q).float()
                return None, (1.0 - (psi_q * m_t).sum(dim=-1)).mean()
        psi_in_0 = self.pyramid.embed_tokens(idx)
        bigram_psi = self.pyramid.embed_dyadic_bigram_phase(idx).to(psi_in_0.dtype)
        in_ngram, readout_ngram = self.ngram_proj(idx, target_dtype=psi_in_0.dtype)
        psi_in = l2_normalize(psi_in_0 + torch.tanh(self.bigram_embed_gate).to(psi_in_0.dtype) * bigram_psi + in_ngram, dim=-1)
        psi = self.smear_gate(psi_in)
        doc_ids = compute_document_ids(idx)
        cos, sin = [t.to(psi.device) for t in self.rope.get_cos_sin(idx.size(1))]
        v0_src = l2_normalize(psi_in + torch.tanh(self.v0_gate).to(psi_in.dtype) * psi_in_0, dim=-1)
        v_0 = self.pyramid.synthesize_value_anchor(v0_src, bigram_psi)
        psi_diff = psi
        n_half = self.cfg.num_layers // 2
        enc_skips: list[torch.Tensor] = []
        for l_idx, block in enumerate(self.blocks):
            if l_idx == n_half:
                psi = l2_normalize(psi + torch.tanh(self.bigram_mid_gate).to(psi.dtype) * bigram_psi, dim=-1)
            psi, blk_diff = block(psi, cos=cos, sin=sin, v_0=v_0, doc_ids=doc_ids)
            g_d = torch.sigmoid(self.diff_accum_gate).to(psi_diff.dtype)
            psi_diff = l2_normalize((1.0 - g_d) * psi_diff + g_d * blk_diff, dim=-1)
            if l_idx < n_half:
                enc_skips.append(psi)
            elif enc_skips:
                skip_g = torch.tanh(self.unet_skip_gates[min(l_idx - n_half, self.unet_skip_gates.size(0) - 1)])
                psi = l2_normalize(psi + skip_g.to(psi.dtype) * enc_skips.pop(), dim=-1)
        psi0_z = torch.stack((psi_in_0[..., 0::2], -psi_in_0[..., 1::2]), dim=-1).reshape_as(psi_in_0)
        psi_pre = l2_normalize(psi + torch.tanh(self.psi0_readout_gate).to(psi.dtype) * psi0_z + readout_ngram, dim=-1)
        m_sharp_unit, m_diff, w_k = self.synthesize_detectors()
        psi_s_out = self.pyramid.readout_chol(psi_pre)
        m_sharp = self.pyramid.readout_chol(m_sharp_unit)
        self._last_diffuse_eff_rank = float(1.0 / ((w_k * w_k).sum().item() + 1e-12))
        self._last_mean_beta, self._last_lambda_diff = self.current_beta(), self.current_lambda_diff()
        spec = self.build_readout_spec()
        prior = self.povm_log_prior
        loss = parseval_mup_dyadic_chunked_ce(psi_s_out, psi_diff, m_sharp, m_diff, targets, spec, prior) if targets is not None else None
        logits = eval_parseval_mup_full_logits(psi_s_out, psi_diff, m_sharp, m_diff, spec, prior) if want_logits else None
        if logits is not None and self.ngram_proj._kn_state is not None:
            logits = self.ngram_proj.apply_surface_logit_lift(idx, logits, self._training_progress, self.cfg.embed_dim)
        return logits, loss


@torch.no_grad()
def _init_ngram_projector_from_tokens(
    model: ParsevalMuPNGramSpeedrunGPT, m_sharp: torch.Tensor, toks_in: torch.Tensor
) -> None:
    cfg, ng = model.cfg, model.ngram_proj
    device = m_sharp.device
    v_eff = min(cfg.valid_vocab_size, m_sharp.size(0))
    m_cent = m_sharp[:v_eff].float() - m_sharp[:v_eff].float().mean(dim=0, keepdim=True)
    u_ng = torch.linalg.svd(m_cent, full_matrices=False)[2][: cfg.ngram_dim, :]
    ng.u_ngram.copy_(u_ng.to(ng.u_ngram.dtype))
    if toks_in.numel() < 4:
        return
    toks_s = toks_in[: min(toks_in.numel(), 2_600_000)].to(device=device, dtype=torch.int64)
    p0_v = torch.bincount(toks_s.clamp(0, v_eff - 1), minlength=v_eff).float()
    p0_v = (p0_v + 1.0) / (p0_v.sum() + float(v_eff))
    phi = m_cent @ u_ng.t()
    evals, evecs = torch.linalg.eigh((phi.t() * p0_v.unsqueeze(0)) @ phi)
    inv_cov = (evecs * (1.0 / torch.sqrt(evals.clamp_min(1e-5) + 0.002)).unsqueeze(0)) @ evecs.t()
    z_v = phi @ inv_cov - (p0_v.unsqueeze(1) * (phi @ inv_cov)).sum(dim=0, keepdim=True)
    z_vocab = F.pad(z_v, (0, 0, 0, max(0, cfg.vocab_size - v_eff)))
    u_c, dst_c = torch.remainder(toks_s[:-1], cfg.ngram_buckets), torch.remainder(toks_s[1:], cfg.vocab_size)
    bi_all, tri_all = ng.compute_bigram_trigram_buckets(toks_s[:-1].unsqueeze(0))
    z_dst = z_vocab.index_select(0, dst_c)
    ones = torch.ones((dst_c.numel(), 1), device=device, dtype=torch.float32)
    z_zero = torch.zeros((cfg.ngram_buckets, cfg.ngram_dim), device=device, dtype=torch.float32)
    c_zero = torch.zeros((cfg.ngram_buckets, 1), device=device, dtype=torch.float32)
    uni_mean = z_zero.clone().index_add_(0, u_c, z_dst) / (c_zero.clone().index_add_(0, u_c, ones) + 2.0)
    z_res = z_dst - 0.55 * uni_mean.index_select(0, u_c)
    bi_c, tri_c = bi_all.reshape(-1), tri_all.reshape(-1)
    bi_mean = z_zero.clone().index_add_(0, bi_c, z_res) / (c_zero.clone().index_add_(0, bi_c, ones) + 3.5)
    tri_mean = z_zero.clone().index_add_(0, tri_c, z_res) / (c_zero.clone().index_add_(0, tri_c, ones) + 5.0)
    ng.unigram_table.copy_((uni_mean / (uni_mean.norm(dim=-1, keepdim=True) + 0.08)).to(ng.unigram_table.dtype))
    ng.bigram_table.copy_((bi_mean / (bi_mean.norm(dim=-1, keepdim=True) + 0.12)).to(ng.bigram_table.dtype))
    ng.trigram_table.copy_((tri_mean / (tri_mean.norm(dim=-1, keepdim=True) + 0.16)).to(ng.trigram_table.dtype))
    del toks_s, z_dst, z_res, bi_all, tri_all
    if toks_in.numel() >= 65536:
        tk = toks_in[: min(toks_in.numel(), 85_000_000)].to(device=device, dtype=torch.int32).clamp_(0, v_eff - 1)
        u_s, dst = tk[:-1], tk[1:]
        csr1 = _build_csr_kn(u_s.to(torch.int64), dst, v_eff, 0.55, 0.80, 0.90)
        p_cont = torch.bincount(csr1[4], minlength=v_eff).float() + 0.5
        p0_f = torch.bincount(dst, minlength=v_eff).float() + 1.0
        p0_f = p0_f / p0_f.sum()
        idf_w = (-torch.log(p0_f.clamp_min(1e-8))).clamp(1.0, 14.0) ** 1.35
        xp1 = torch.cat([u_s[:1], u_s[:-1]])
        csr_gap = _build_csr_kn(xp1.to(torch.int64), dst, v_eff, 0.65, 0.85, 0.92)
        csr2 = _build_csr_kn(xp1.to(torch.int64) * 100003 + u_s.to(torch.int64), dst, v_eff, 0.65, 0.85, 0.92)
        xp2 = torch.cat([xp1[:1], xp1[:-1]])
        csr3 = _build_csr_kn(xp2.to(torch.int64) * 10000019 + xp1.to(torch.int64) * 100003 + u_s.to(torch.int64), dst, v_eff, 0.72, 0.88, 0.95)
        ng._kn_state = {"v_sz": v_eff, "p_cont": p_cont / p_cont.sum(), "p0": p0_f, "idf_w": idf_w, "csr1": csr1, "csr_gap": csr_gap, "csr2": csr2, "csr3": csr3}
        del tk, u_s, dst, xp1, xp2
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def initialize_parseval_mup_ngram_with_ppmi_svd(
    model: ParsevalMuPNGramSpeedrunGPT, ppmi_embeddings: torch.Tensor, init_tokens: torch.Tensor, prior_tokens: torch.Tensor
) -> dict[str, float]:
    stats = initialize_parseval_mup_with_ppmi_svd(model, ppmi_embeddings, init_tokens[:2_600_000], prior_tokens)
    m_sharp, _, _ = model.synthesize_detectors()
    _init_ngram_projector_from_tokens(model, m_sharp, init_tokens)
    stats["kraus_stiefel_err"] = model.retract_kraus_generators()
    stats["u_ngram_stiefel_err"] = model.ngram_proj.retract_stiefel_basis()
    return stats


def build_parseval_mup_ngram_optimizers(
    model: ParsevalMuPNGramSpeedrunGPT, base_muon_lr: float = 0.026, base_wav_lr: float = 0.0085, base_head_lr: float = 0.005
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    mup_scale = math.sqrt(256.0 / float(model.cfg.embed_dim))
    muon_lr, wav_lr = float(base_muon_lr) * mup_scale, float(base_wav_lr) * mup_scale
    head_lr, body_lr = float(base_head_lr) * mup_scale, 0.0035 * mup_scale
    muon_params, wav_params, ngram_params, chol_params, body_params = [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        target_list = (
            muon_params if ("kraus_generators" in name or "u_ngram" in name)
            else wav_params if any(k in name for k in _WAV_KEYS)
            else ngram_params if any(k in name for k in _NGRAM_KEYS)
            else chol_params if any(k in name for k in _CHOL_KEYS)
            else body_params
        )
        target_list.append(p)
    muon_opt = StiefelMuon(
        [{"params": muon_params, "lr": muon_lr, "base_lr": muon_lr, "weight_decay": 0.01}],
        lr=muon_lr, momentum=0.95, nesterov=True, weight_decay=0.01, decay_mode="cautious",
        stiefel_penalty=0.02, retract_every=45, ns_steps=2, polar_method="polar_express", use_normuon=True, use_twin_rail=True,
    )
    adam_groups = [
        {"params": wav_params, "lr": wav_lr, "base_lr": wav_lr, "weight_decay": 0.0, "group_name": "wavelet_2pow16"},
        {"params": ngram_params, "lr": wav_lr, "base_lr": wav_lr, "weight_decay": 0.0, "group_name": "surface_ngram"},
        {"params": chol_params, "lr": head_lr, "base_lr": head_lr, "weight_decay": 0.0, "group_name": "cholesky_readout"},
        {"params": body_params, "lr": body_lr, "base_lr": body_lr, "weight_decay": 0.01, "group_name": "body"},
    ]
    is_cuda = any(p.is_cuda for p in model.parameters())
    return muon_opt, torch.optim.AdamW([g for g in adam_groups if g["params"]], lr=head_lr, betas=(0.9, 0.95), fused=is_cuda)


__all__ = [
    "HashedSurfaceNGramProjector", "ParsevalMuPDyadicPyramid", "ParsevalMuPNGramSpeedrunConfig",
    "ParsevalMuPNGramSpeedrunGPT", "ParsevalWaveletBlock", "build_parseval_mup_ngram_optimizers",
    "initialize_parseval_mup_ngram_with_ppmi_svd",
]
