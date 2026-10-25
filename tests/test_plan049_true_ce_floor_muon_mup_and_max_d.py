"""Unit and hypothesis tests for Experiment Plan 049 (Wave 34):
True Dyadic Shell CE Floor, Symplectic-Preserving `StiefelMuon` + Parseval `muP`
Restoration, Pre-`readout_chol` Subspace Normalization, Compiled 50,257-Way `M=4`
Coincidence Kernel, and Max-Width `d in [256, 1024]` Scaling.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qdlblocks.functional import l2_normalize
from qdlblocks.kernels.flash_born import get_flash_born_launch_meta
from qdlblocks.kernels.fused_parseval_ce import (
    M4_SUBSPACE_SCALE,
    compiled_m4_coincidence_chunk_step,
)
from qdlblocks.layers.multirate_wavefunction_conv import WaveletScaleState
from qdlblocks.layers.parseval_mup_chunked_ce import ParsevalChunkedCEReadoutSpec
from qdlblocks.layers.subspace_coincidence_born import (
    SubspaceCoincidenceSpec,
    coarse_coincidence_ce_from_centers,
    compute_coarse_cluster_centers,
    eval_subspace_coincidence_full_logits,
    subspace_coincidence_dyadic_chunked_ce,
    subspace_l2_normalize,
)
from qdlblocks.models.level_separated_wavelet_conv_gpt import (
     DEFAULT_FLASH_BORN_HEAD_DIM,
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    _fold_to_level_dim,
    build_level_separated_wavelet_conv_optimizers,
    initialize_level_separated_wavelet_conv_with_ppmi_svd,
    update_parseval_mup_ngram_learning_rates,
)
from qdlblocks.models.parseval_mup_ngram_speedrun_gpt import (
    DEFAULT_NGRAM_MUON_LR,
    FINE_WAVELET_SPARSE_LR,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
TEST_EMBED_DIM: int = 128
TEST_REF_EMBED_DIM: int = 256
TEST_NUM_SUBSPACES: int = 4
TEST_BATCH_SIZE: int = 2
TEST_SEQ_LEN: int = 16
TEST_VOCAB_SIZE: int = 512
TEST_VALID_VOCAB_SIZE: int = 500
MIN_STEP0_SHARP_EFF_RANK: float = 12.5
MAX_COARSE_CLUSTER_CE_NATS: float = 0.05
FINITE_DIFF_EPS: float = 2e-4
GRAD_ATOL: float = 5e-3
PARITY_ATOL: float = 1e-4


def test_symplectic_preserving_fold_and_step0_rank() -> None:
    """Verify _fold_to_level_dim and WaveletScaleState preserve symplectic orthogonality and step-0 rank >= 12.5."""
    torch.manual_seed(49)
    cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=TEST_VOCAB_SIZE,
        valid_vocab_size=TEST_VALID_VOCAB_SIZE,
        embed_dim=TEST_REF_EMBED_DIM,
        num_layers=2,
        block_size=TEST_SEQ_LEN,
        num_subspaces=TEST_NUM_SUBSPACES,
    )
    model = LevelSeparatedWaveletConvGPT(cfg)
    pyr = model.pyramid

    x = l2_normalize(torch.randn(32, TEST_REF_EMBED_DIM), dim=-1)
    x_swap = pyr.apply_symplectic_swap(x)
    level_dim = TEST_REF_EMBED_DIM // TEST_NUM_SUBSPACES

    f_x = _fold_to_level_dim(x, level_dim)
    f_swap = _fold_to_level_dim(x_swap, level_dim)
    assert torch.allclose(pyr.apply_symplectic_swap(f_x), f_swap, atol=1e-6)
    max_self_ov = float((f_x * f_swap).sum(dim=-1).abs().max().item())
    assert max_self_ov < 1e-6, f"Symplectic fold broke orthogonality: max_self_ov={max_self_ov}"

    levels_psi = [l2_normalize(torch.randn(16, level_dim), dim=-1) for _ in range(4)]
    levels_det = [pyr.apply_symplectic_swap(t) for t in levels_psi]
    state_psi = WaveletScaleState(*levels_psi).normalize_levels()
    state_det = WaveletScaleState(*levels_det).normalize_levels()

    psi_inter = state_psi.to_interleaved_product_of_spheres()
    det_inter = state_det.to_interleaved_product_of_spheres()
    assert torch.allclose(pyr.apply_symplectic_swap(psi_inter), det_inter, atol=1e-6)
    assert float((psi_inter * det_inter).sum(dim=-1).abs().max().item()) < 1e-6

    ppmi_emb = l2_normalize(torch.randn(TEST_VOCAB_SIZE, TEST_REF_EMBED_DIM), dim=-1)
    init_toks = torch.randint(0, TEST_VALID_VOCAB_SIZE, (4096,))
    prior_toks = torch.randint(0, TEST_VALID_VOCAB_SIZE, (2048,))
    stats = initialize_level_separated_wavelet_conv_with_ppmi_svd(model, ppmi_emb, init_toks, prior_toks)
    assert stats["sharp_eff_rank"] >= MIN_STEP0_SHARP_EFF_RANK, (
        f"Expected step-0 sharp_eff_rank >= {MIN_STEP0_SHARP_EFF_RANK}, got {stats['sharp_eff_rank']}"
    )


def test_compiled_m4_coincidence_chunk_step_forward_backward() -> None:
    """Verify compiled_m4_coincidence_chunk_step matches finite-difference grads, eval logits, and M=4 margin."""
    torch.manual_seed(149)
    c_tok, v_act = 6, 32
    w_sharp_m = (torch.randn(4, c_tok, v_act, dtype=torch.float32) * 0.15).requires_grad_(True)
    w_diff = (torch.randn(c_tok, v_act, dtype=torch.float32) * 0.25).requires_grad_(True)
    yc_safe = torch.randint(0, v_act, (c_tok, 1), dtype=torch.long)
    valid_f = torch.ones((c_tok, 1), dtype=torch.float32)
    row_scale = torch.full((c_tok, 1), 1.0 / float(c_tok), dtype=torch.float32)
    scalars = (16.0, 0.15, 0.25, 0.22, 0.14, M4_SUBSPACE_SCALE)
    log_prior = (torch.randn(1, v_act, dtype=torch.float32) * 0.2).requires_grad_(True)
    v_idx = torch.arange(c_tok, dtype=torch.long).unsqueeze(1)

    loss_sum, grad_s_m, grad_d, dprior = compiled_m4_coincidence_chunk_step(
        w_sharp_m.detach(),
        w_diff.detach(),
        yc_safe,
        (valid_f, row_scale),
        scalars,
        log_prior.detach(),
        v_idx,
    )
    assert torch.isfinite(loss_sum)

    # Check finite-difference gradient on a sample of w_sharp_m entries across all 4 subspaces
    for m_idx in range(4):
        for i_tok in range(2):
            for j_voc in range(3):
                w_pos = w_sharp_m.detach().clone()
                w_neg = w_sharp_m.detach().clone()
                w_pos[m_idx, i_tok, j_voc] += FINITE_DIFF_EPS
                w_neg[m_idx, i_tok, j_voc] -= FINITE_DIFF_EPS
                l_pos, _, _, _ = compiled_m4_coincidence_chunk_step(
                    w_pos, w_diff.detach(), yc_safe, (valid_f, row_scale), scalars, log_prior.detach(), v_idx
                )
                l_neg, _, _, _ = compiled_m4_coincidence_chunk_step(
                    w_neg, w_diff.detach(), yc_safe, (valid_f, row_scale), scalars, log_prior.detach(), v_idx
                )
                num_g = float((l_pos - l_neg).item()) / (2.0 * FINITE_DIFF_EPS * float(c_tok))
                ana_g = float(grad_s_m[m_idx, i_tok, j_voc].item())
                assert abs(num_g - ana_g) < GRAD_ATOL, f"Grad mismatch at m={m_idx}: {num_g} vs {ana_g}"

    # Verify train/eval parity between subspace_coincidence_dyadic_chunked_ce and eval_subspace_coincidence_full_logits
    psi_s = subspace_l2_normalize(torch.randn(2, 5, TEST_EMBED_DIM), num_subspaces=4, partition_mode="strided")
    psi_d = l2_normalize(torch.randn(2, 5, TEST_EMBED_DIM), dim=-1)
    m_s = subspace_l2_normalize(torch.randn(64, TEST_EMBED_DIM), num_subspaces=4, partition_mode="strided")
    m_d = l2_normalize(torch.randn(64, TEST_EMBED_DIM), dim=-1)
    targets = torch.randint(0, 60, (2, 5))
    r_spec = ParsevalChunkedCEReadoutSpec(
        beta_0=18.0,
        lambda_diff_scaled=0.15,
        traceless_gamma=0.25,
        dyadic_coarse_weight=0.0,
        valid_vocab_size=60,
        chunk_size=4,
    )
    c_spec_m4 = SubspaceCoincidenceSpec(
        num_subspaces=4,
        coincidence_gamma=0.22,
        subspace_var_penalty=0.0,
        coarse_coincidence_weight=0.0,
        partition_mode="strided",
        use_fine_m4_chunked_coincidence=True,
        pre_normalized_inputs=True,
    )
    loss_chunked = subspace_coincidence_dyadic_chunked_ce(psi_s, psi_d, m_s, m_d, targets, r_spec, c_spec_m4)
    full_logits_m4 = eval_subspace_coincidence_full_logits(psi_s, psi_d, m_s, m_d, r_spec, c_spec_m4)
    loss_eval = F.cross_entropy(full_logits_m4.reshape(-1, 60), targets.reshape(-1))
    assert abs(float(loss_chunked.item()) - float(loss_eval.item())) < PARITY_ATOL

    # Verify M=4 coincidence increases target-to-single-subspace-impostor margin over M=1
    m_s_planted = m_s.clone()
    target_vec = psi_s[0, 0].clone()
    m_s_planted[0] = target_vec
    impostor_vec = torch.zeros_like(target_vec)
    impostor_vec.view(TEST_EMBED_DIM // 8, 4, 2)[:, 0, :] = target_vec.view(TEST_EMBED_DIM // 8, 4, 2)[:, 0, :]
    m_s_planted[1] = subspace_l2_normalize(impostor_vec.unsqueeze(0), num_subspaces=4).squeeze(0)

    c_spec_m1 = SubspaceCoincidenceSpec(
        num_subspaces=4,
        coincidence_gamma=0.0,
        subspace_var_penalty=0.0,
        coarse_coincidence_weight=0.0,
        partition_mode="strided",
        use_fine_m4_chunked_coincidence=True,
        pre_normalized_inputs=True,
    )
    logits_m4 = eval_subspace_coincidence_full_logits(
        psi_s[:1, :1], psi_d[:1, :1], m_s_planted, m_d, r_spec, c_spec_m4
    )
    logits_m1 = eval_subspace_coincidence_full_logits(
        psi_s[:1, :1], psi_d[:1, :1], m_s_planted, m_d, r_spec, c_spec_m1
    )
    margin_m4 = float((logits_m4[0, 0, 0] - logits_m4[0, 0, 1]).item())
    margin_m1 = float((logits_m1[0, 0, 0] - logits_m1[0, 0, 1]).item())
    assert margin_m4 > margin_m1, f"Expected M=4 margin ({margin_m4}) > M=1 margin ({margin_m1})"


def test_stiefel_muon_2d_and_chol_group_routing() -> None:
    """Verify StiefelMuon routes 2D+3D matrices (retract_every=2) and AdamW routes conv_bank and wav_65536."""
    cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=TEST_VOCAB_SIZE,
        valid_vocab_size=TEST_VALID_VOCAB_SIZE,
        embed_dim=TEST_REF_EMBED_DIM,
        num_layers=2,
        block_size=TEST_SEQ_LEN,
    )
    model = LevelSeparatedWaveletConvGPT(cfg)
    stiefel_opt, adamw_opt = build_level_separated_wavelet_conv_optimizers(model)

    assert stiefel_opt.defaults["retract_every"] == 2
    muon_param_ids = {id(p) for g in stiefel_opt.param_groups for p in g["params"]}

    # Verify 2D attention & interferometer weights and 3D kraus_generators are in StiefelMuon
    assert id(model.pyramid.kraus_generators) in muon_param_ids
    for blk in model.blocks:
        assert id(blk.attn.qkv_proj.weight) in muon_param_ids
        assert id(blk.attn.out_proj.weight) in muon_param_ids
        assert id(blk.interf.W_beam1) in muon_param_ids
        assert id(blk.interf.W_beam2) in muon_param_ids
        assert id(blk.interf.W_out) in muon_param_ids

    groups_by_name = {g["group_name"]: g for g in adamw_opt.param_groups}
    assert "cholesky_and_gates" in groups_by_name
    assert "wavelet_fine_sparse" in groups_by_name
    assert "wavelet_2pow16" in groups_by_name

    chol_ids = {id(p) for p in groups_by_name["cholesky_and_gates"]["params"]}
    fine_wav_ids = {id(p) for p in groups_by_name["wavelet_fine_sparse"]["params"]}

    assert groups_by_name["cholesky_and_gates"]["weight_decay"] == 0.0
    for name, p in model.named_parameters():
        if "conv_bank" in name and "diag_raw" in name:
            assert id(p) in chol_ids, f"{name} must be routed to cholesky_and_gates (wd=0.0)"

    assert id(model.pyramid.wav_65536) in fine_wav_ids
    assert id(model.pyramid.wav_8192) in fine_wav_ids
    expected_fine_lr = FINE_WAVELET_SPARSE_LR * math.sqrt(256.0 / float(TEST_REF_EMBED_DIM))
    assert abs(groups_by_name["wavelet_fine_sparse"]["base_lr"] - expected_fine_lr) < 1e-6

    update_parseval_mup_ngram_learning_rates(
        stiefel_opt, adamw_opt, lr_256=0.5 * DEFAULT_NGRAM_MUON_LR, embed_dim=TEST_REF_EMBED_DIM
    )
    assert abs(groups_by_name["wavelet_fine_sparse"]["lr"] - 0.5 * expected_fine_lr) < 1e-6


def test_pre_readout_chol_subspace_normalization_preserves_beta() -> None:
    """Verify pre-readout_chol subspace normalization preserves beta_0(d) and coarse cluster CE < 0.05."""
    torch.manual_seed(349)
    cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=TEST_VOCAB_SIZE,
        valid_vocab_size=TEST_VALID_VOCAB_SIZE,
        embed_dim=TEST_EMBED_DIM,
        num_layers=2,
        block_size=TEST_SEQ_LEN,
        use_coincidence_warmup=True,
    )
    model = LevelSeparatedWaveletConvGPT(cfg)
    model.eval()
    model.set_training_progress(0.5)

    idx = torch.randint(0, TEST_VALID_VOCAB_SIZE, (TEST_BATCH_SIZE, TEST_SEQ_LEN))
    targets = torch.randint(0, TEST_VALID_VOCAB_SIZE, (TEST_BATCH_SIZE, TEST_SEQ_LEN))

    with torch.no_grad():
        logits_before, loss_before = model(idx, targets=targets, return_logits=True)
        scale_absorbed = model.absorb_beta_into_readout_cholesky()
        spec_after = model.build_coincidence_spec()
        logits_after, loss_after = model(idx, targets=targets, return_logits=True)

    assert isinstance(scale_absorbed, dict) and model._beta_absorbed_scale > 1.0
    assert abs(spec_after.coincidence_gamma - model.current_coincidence_gamma()) < 1e-6
    assert logits_before is not None and logits_after is not None
    assert loss_before is not None and loss_after is not None
    std_before = float(logits_before.std().item())
    std_after = float(logits_after.std().item())
    assert abs(std_before - std_after) / max(1e-6, std_before) < 0.02, (
        f"Logit scale collapsed after readout_chol absorption: std_before={std_before:.4f}, std_after={std_after:.4f}"
    )
    assert torch.allclose(logits_before, logits_after, atol=5e-2, rtol=5e-2)
    assert abs(float(loss_before.item()) - float(loss_after.item())) < 2e-2

    # Verify compute_coarse_cluster_centers achieves < 0.05 nats coarse CE at alpha = 1.0
    with torch.no_grad():
        m_sep = model.synthesize_level_separated_detectors().to_interleaved_product_of_spheres()
        r_spec_c = model.build_readout_spec()
        c_spec_c = model.build_coincidence_spec()
        centers, v_map = compute_coarse_cluster_centers(m_sep, r_spec_c, c_spec_c)
        assert centers.shape == (r_spec_c.num_coarse_buckets, TEST_EMBED_DIM)
        assert v_map.numel() == TEST_VALID_VOCAB_SIZE
        v_targets = torch.arange(min(64, TEST_VALID_VOCAB_SIZE), dtype=torch.long)
        psi_ideal = m_sep.index_select(0, v_targets)
        coarse_ce = coarse_coincidence_ce_from_centers(
            psi_ideal,
            m_sep,
            v_targets,
            r_spec_c,
            c_spec_c,
        )
    assert float(coarse_ce.item()) < MAX_COARSE_CLUSTER_CE_NATS, (
        f"Expected ideal coarse cluster CE < {MAX_COARSE_CLUSTER_CE_NATS}, got {float(coarse_ce.item())}"
    )


def test_checkpointed_detector_synthesis_and_head_dim_power_of_2() -> None:
    """Verify head_dim == 64 across d in {256..1024} and checkpointed synthesize_detectors parity."""
    for d in (256, 384, 512, 640, 768, 1024):
        cfg_d = LevelSeparatedWaveletConvConfig(embed_dim=d)
        assert cfg_d.head_dim == DEFAULT_FLASH_BORN_HEAD_DIM, f"Width d={d} produced head_dim={cfg_d.head_dim}"
        meta = get_flash_born_launch_meta(512, head_dim=cfg_d.head_dim)
        assert meta["BLOCK_M"] == 64 and meta["BLOCK_N"] == 64

    for non_pow2_hd in (48, 80, 96, 112):
        meta_np2 = get_flash_born_launch_meta(512, head_dim=non_pow2_hd)
        assert meta_np2["BLOCK_M"] in (32, 64)
        assert (non_pow2_hd >= 16) and (non_pow2_hd <= 256) and (non_pow2_hd % 16 == 0)

    torch.manual_seed(449)
    cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=TEST_VOCAB_SIZE,
        valid_vocab_size=TEST_VALID_VOCAB_SIZE,
        embed_dim=TEST_EMBED_DIM,
        num_layers=2,
        block_size=TEST_SEQ_LEN,
    )
    model = LevelSeparatedWaveletConvGPT(cfg)
    model.train()

    probe = torch.randn(TEST_VOCAB_SIZE, TEST_EMBED_DIM)
    m_s_un, m_d_un, wk_un = model.synthesize_detectors(use_checkpoint=False)
    loss_un = (m_s_un * probe).sum() + 0.3 * (m_d_un * probe).sum() + wk_un.sum()
    loss_un.backward()
    grad_65536_un = model.pyramid.wav_65536.grad.detach().clone()
    grad_256_un = model.pyramid.wav_256.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    m_s_ck, m_d_ck, wk_ck = model.synthesize_detectors(use_checkpoint=True)
    assert torch.allclose(m_s_un, m_s_ck, atol=1e-6)
    assert torch.allclose(m_d_un, m_d_ck, atol=1e-6)
    loss_ck = (m_s_ck * probe).sum() + 0.3 * (m_d_ck * probe).sum() + wk_ck.sum()
    loss_ck.backward()
    assert torch.allclose(grad_65536_un, model.pyramid.wav_65536.grad, atol=1e-5)
    assert torch.allclose(grad_256_un, model.pyramid.wav_256.grad, atol=1e-5)


if __name__ == "__main__":
    for fn in (
        test_symplectic_preserving_fold_and_step0_rank,
        test_compiled_m4_coincidence_chunk_step_forward_backward,
        test_stiefel_muon_2d_and_chol_group_routing,
        test_pre_readout_chol_subspace_normalization_preserves_beta,
        test_checkpointed_detector_synthesis_and_head_dim_power_of_2,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    print("All 5 Plan 049 unit & hypothesis tests PASSED.")
