"""Unit tests for Wave 48 (`EXPERIMENT_PLAN_063.md`): `StandaloneQuantumBornGPT` & Active-Token Vocabulary Manifold Optimizers."""

from __future__ import annotations

import ast
import inspect
import math
import textwrap

import torch

from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_TIKHONOV_GRAM_LAMBDA,
    complex_l2_normalize,
    extract_rank2_krylov_eigenpair_uhlmann,
    fast_rank_r_hermitian_and_ket_pack,
    initialize_stiefel_parameter,
    riemannian_complex_unitary_muon_step,
    riemannian_stiefel_muon_step,
    unpack_hermitian_isometric,
)
from qdlblocks.models.riemannian_impure_density_gpt import fast_rank_r_token_density_pack
from qdlblocks.models.standalone_quantum_born_gpt import (
    QuantumAttentionMode,
    Rank2KrylovBornChunkedCEFn,
    StandaloneQuantumBornBlock,
    StandaloneQuantumBornConfig,
    StandaloneQuantumBornGPT,
    VocabOptimizerMode,
    VocabularyReadoutMode,
    build_standalone_quantum_born_optimizers,
    eval_rank2_krylov_born_logits,
)
from qdlblocks.optim.vocab_manifold_muon import (
    BatchSubspaceTikhonovGramMuon,
    RiemannianSphereTangentSNROptimizer,
    _newton_schulz_coupled_inv_sqrt,
    project_rows_to_rms_sphere_,
)


def test_fused_288_hermitian_and_ket_pack_isometry_and_phase() -> None:
    """Verify fused 288-width packing isometry and Uhlmann Rank-2 Krylov phase covariance (`H1`)."""
    torch.manual_seed(42)
    bsz, heads, seq_len, r_branches, d_h = 2, 6, 8, 4, 16
    v_re = torch.randn(bsz, heads, seq_len, r_branches, d_h)
    v_im = torch.randn(bsz, heads, seq_len, r_branches, d_h)

    cfg = StandaloneQuantumBornConfig(n_layer=2, n_head=heads, n_embd=192, v_rank=r_branches)
    blk = StandaloneQuantumBornBlock(cfg, layer_idx=0)

    packed_288 = fast_rank_r_hermitian_and_ket_pack(v_re, v_im, blk.re_pack_mask, blk.im_pack_mask)
    assert packed_288.shape == (bsz, heads, seq_len, 288)

    ref_256 = fast_rank_r_token_density_pack(v_re, v_im, blk.re_pack_mask, blk.im_pack_mask)
    diff_256 = float((packed_288[..., :256] - ref_256).abs().max().item())
    assert diff_256 < 1e-6, f"Expected first 256 dims to match fast_rank_r_token_density_pack < 1e-6, got {diff_256}"

    v_coh_re = packed_288[..., 256:272]
    v_coh_im = packed_288[..., 272:288]
    coh_norm_err = float(((v_coh_re.square().sum(dim=-1) + v_coh_im.square().sum(dim=-1)).sqrt() - 1.0).abs().max().item())
    assert coh_norm_err < 1e-5, f"Expected unit-norm coherent ket < 1e-5, got {coh_norm_err}"

    rho_re, rho_im = unpack_hermitian_isometric(packed_288[..., :256], d_h)
    lam1, lam2, phi1_re, phi1_im, phi2_re, phi2_im = extract_rank2_krylov_eigenpair_uhlmann(
        rho_re, rho_im, v_coh_re, v_coh_im
    )

    # Check Uhlmann gauge: <v_bar | phi1> is real and non-negative
    anchor_re, anchor_im = complex_l2_normalize(v_coh_re, v_coh_im)
    ov_re = (anchor_re * phi1_re + anchor_im * phi1_im).sum(dim=-1)
    ov_im = (anchor_re * phi1_im - anchor_im * phi1_re).sum(dim=-1)
    assert float(ov_im.abs().max().item()) < 1e-5, f"Expected Im<v_bar|phi1> == 0, got {ov_im.abs().max().item()}"
    assert float(ov_re.min().item()) >= -1e-6

    # Check Gram-Schmidt orthogonality |<phi1 | phi2>| < 1e-5
    p12_re = (phi1_re * phi2_re + phi1_im * phi2_im).sum(dim=-1)
    p12_im = (phi1_re * phi2_im - phi1_im * phi2_re).sum(dim=-1)
    ortho_err = float((p12_re.square() + p12_im.square()).sqrt().max().item())
    assert ortho_err < 1e-5, f"Expected |<phi1|phi2>| < 1e-5, got {ortho_err}"

    # Check global phase rotation |v> -> e^{i theta} |v>: density is invariant, Krylov kets rotate by e^{i theta}
    theta = torch.tensor(0.73)
    c_th, s_th = torch.cos(theta), torch.sin(theta)
    v_re_rot = v_re * c_th - v_im * s_th
    v_im_rot = v_re * s_th + v_im * c_th
    packed_rot = fast_rank_r_hermitian_and_ket_pack(v_re_rot, v_im_rot, blk.re_pack_mask, blk.im_pack_mask)

    density_invariance_err = float((packed_rot[..., :256] - packed_288[..., :256]).abs().max().item())
    assert density_invariance_err < 1e-5

    rho_rot_re, rho_rot_im = unpack_hermitian_isometric(packed_rot[..., :256], d_h)
    _, _, phi1_rot_re, phi1_rot_im, _, _ = extract_rank2_krylov_eigenpair_uhlmann(
        rho_rot_re, rho_rot_im, packed_rot[..., 256:272], packed_rot[..., 272:288]
    )
    expected_phi1_re = phi1_re * c_th - phi1_im * s_th
    expected_phi1_im = phi1_re * s_th + phi1_im * c_th
    cov_err = float(
        ((phi1_rot_re - expected_phi1_re).square() + (phi1_rot_im - expected_phi1_im).square()).sqrt().max().item()
    )
    assert cov_err < 1e-4, f"Expected Uhlmann phase covariance error < 1e-4, got {cov_err}"


def test_tikhonov_riemannian_stiefel_and_unitary_muon_stability() -> None:
    """Verify Tikhonov-damped Stiefel & $U(16)$ Muon stability under rank-deficient (`rank=2`) gradients (`H2`)."""
    torch.manual_seed(17)
    d_rows, d_cols = 128, 64
    w_stiefel = torch.empty(d_rows, d_cols)
    initialize_stiefel_parameter(w_stiefel)
    mom_w = torch.zeros_like(w_stiefel)

    heads, d_h = 6, 16
    u_re = torch.eye(d_h).unsqueeze(0).expand(heads, -1, -1).clone()
    u_im = torch.zeros(heads, d_h, d_h)
    mom_re = torch.zeros_like(u_re)
    mom_im = torch.zeros_like(u_im)

    for _ in range(50):
        # Rank-2 ill-conditioned gradient for real Stiefel matrix
        u_low = torch.randn(d_rows, 2)
        v_low = torch.randn(2, d_cols)
        g_rank2 = u_low @ v_low
        riemannian_stiefel_muon_step(
            w_stiefel,
            g_rank2,
            mom_w,
            lr=0.03,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        )

        # Rank-2 ill-conditioned gradient for batched complex U(16) unitary
        g_re = torch.randn(heads, d_h, 2) @ torch.randn(heads, 2, d_h)
        g_im = torch.randn(heads, d_h, 2) @ torch.randn(heads, 2, d_h)
        riemannian_complex_unitary_muon_step(
            u_re,
            u_im,
            g_re,
            g_im,
            mom_re,
            mom_im,
            lr=0.025,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        )

    assert torch.isfinite(w_stiefel).all()
    assert torch.isfinite(u_re).all() and torch.isfinite(u_im).all()

    st_err = float(torch.linalg.matrix_norm(w_stiefel.T @ w_stiefel - torch.eye(d_cols), ord="fro").item()) / math.sqrt(
        d_cols
    )
    assert st_err < 1e-5, f"Expected Stiefel error < 1e-5 under rank-2 gradients, got {st_err}"

    eye_h = torch.eye(d_h).unsqueeze(0)
    gram_re = torch.matmul(u_re.transpose(-2, -1), u_re) + torch.matmul(u_im.transpose(-2, -1), u_im)
    gram_im = torch.matmul(u_re.transpose(-2, -1), u_im) - torch.matmul(u_im.transpose(-2, -1), u_re)
    u_err = float(
        ((gram_re - eye_h).square().sum(dim=(-2, -1)) + gram_im.square().sum(dim=(-2, -1))).sqrt().max().item()
    ) / math.sqrt(d_h)
    assert u_err < 1e-5, f"Expected U(16) unitary error < 1e-5 under rank-2 gradients, got {u_err}"


def test_batch_subspace_gram_muon_and_sphere_tangent_snr_active_isolation() -> None:
    """Verify coupled Newton-Schulz $A^{-1/2}$ accuracy, sphere RMS invariance, and inactive-row isolation (`H3`)."""
    torch.manual_seed(29)
    d_dim = 64
    mat = torch.randn(128, d_dim)
    mat = mat / torch.linalg.matrix_norm(mat, ord="fro")
    gram_a = mat.T @ mat + 1e-2 * torch.eye(d_dim)
    inv_sqrt_a = _newton_schulz_coupled_inv_sqrt(gram_a, steps=6)
    res_err = float(torch.linalg.matrix_norm(inv_sqrt_a @ inv_sqrt_a @ gram_a - torch.eye(d_dim), ord="fro").item()) / math.sqrt(
        d_dim
    )
    assert res_err < 1e-3, f"Expected coupled Newton-Schulz A^{{-1/2}} residual < 1e-3, got {res_err}"

    vocab_sz = 200
    active_ids = torch.tensor([3, 7, 19, 42, 105], dtype=torch.long)
    inactive_mask = torch.ones(vocab_sz, dtype=torch.bool)
    inactive_mask[active_ids] = False

    for opt_cls in (BatchSubspaceTikhonovGramMuon, RiemannianSphereTangentSNROptimizer):
        weight = torch.nn.Parameter(torch.randn(vocab_sz, d_dim))
        project_rows_to_rms_sphere_(weight)
        w_init_inactive = weight.data[inactive_mask].clone()

        opt = opt_cls([weight], lr=0.06)
        for _ in range(10):
            weight.grad = torch.zeros_like(weight)
            weight.grad[active_ids] = torch.randn(active_ids.numel(), d_dim)
            opt.step_active(active_ids)

        inactive_diff = float((weight.data[inactive_mask] - w_init_inactive).abs().max().item())
        assert inactive_diff == 0.0, f"{opt_cls.__name__} modified inactive rows by {inactive_diff}"

        rms_err = float((weight.data.norm(dim=-1) / math.sqrt(d_dim) - 1.0).abs().max().item())
        assert rms_err < 1e-5, f"{opt_cls.__name__} sphere RMS error {rms_err} >= 1e-5"


def test_standalone_quantum_born_gpt_end_to_end_and_rule9_autograd() -> None:
    """Verify `StandaloneQuantumBornGPT` forward/backward, `Rank2KrylovBornChunkedCEFn` gradients, and Rule 9 AST."""
    # 1. Rule-9 AST check on Rank2KrylovBornChunkedCEFn.backward: 0 matmuls and 0 power calls
    bwd_src = textwrap.dedent(inspect.getsource(Rank2KrylovBornChunkedCEFn.backward))
    bwd_tree = ast.parse(bwd_src)
    for node in ast.walk(bwd_tree):
        assert not isinstance(node, ast.MatMult), "Rule 9 violation: @ operator found in backward()"
        if isinstance(node, ast.BinOp):
            assert not isinstance(node.op, ast.Pow), "Rule 9 violation: ** operator found in backward()"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"matmul", "mm", "bmm", "pow"}, (
                f"Rule 9 violation: {node.func.attr}() called in backward()"
            )

    # 2. Verify Rank2KrylovBornChunkedCEFn analytical gradients against dense PyTorch autograd
    torch.manual_seed(37)
    n_tok, d_model, vocab_sz = 24, 64, 96
    h_a = torch.randn(n_tok, d_model, dtype=torch.float32, requires_grad=True)
    raw_kw = torch.rand(n_tok, 2, dtype=torch.float32) + 0.1
    kw_a = (raw_kw / raw_kw.sum(dim=-1, keepdim=True)).detach().requires_grad_(True)
    w_a = torch.randn(vocab_sz, d_model, dtype=torch.float32, requires_grad=True)
    tau_a = torch.tensor(8.0, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, vocab_sz, (n_tok,), dtype=torch.long)

    loss_chunked = Rank2KrylovBornChunkedCEFn.apply(h_a, kw_a, w_a, targets, tau_a, 15.0, 8)
    loss_chunked.backward()

    h_b = h_a.detach().clone().requires_grad_(True)
    kw_b = kw_a.detach().clone().requires_grad_(True)
    w_b = w_a.detach().clone().requires_grad_(True)
    tau_b = tau_a.detach().clone().requires_grad_(True)
    logits_dense = eval_rank2_krylov_born_logits(
        h_b.unsqueeze(0), kw_b.unsqueeze(0), w_b, tau_b, softcap=15.0
    ).squeeze(0)
    loss_dense = torch.nn.functional.cross_entropy(logits_dense, targets)
    loss_dense.backward()

    assert abs(float(loss_chunked.item() - loss_dense.item())) < 1e-5
    assert float((h_a.grad - h_b.grad).abs().max().item()) < 1e-4  # type: ignore[operator]
    assert float((kw_a.grad - kw_b.grad).abs().max().item()) < 1e-4  # type: ignore[operator]
    assert float((w_a.grad - w_b.grad).abs().max().item()) < 1e-4  # type: ignore[operator]
    assert float((tau_a.grad - tau_b.grad).abs().max().item()) < 1e-4  # type: ignore[operator]

    # 3. Verify StandaloneQuantumBornGPT end-to-end training step across both readout modes
    for readout_mode in (VocabularyReadoutMode.RANK2_KRYLOV_BORN, VocabularyReadoutMode.SOFTCAP_LINEAR_CE):
        cfg = StandaloneQuantumBornConfig(
            vocab_size=128,
            n_layer=2,
            n_head=4,
            n_embd=128,
            v_rank=2,
            attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320,
            readout_mode=readout_mode,
            vocab_optim_mode=VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON,
        )
        model = StandaloneQuantumBornGPT(cfg)
        opt_bundle = build_standalone_quantum_born_optimizers(model)
        idx = torch.randint(0, 128, (2, 16))
        tgt = torch.randint(0, 128, (2, 16))

        for _ in range(3):
            opt_bundle.zero_grad()
            _, loss = model(idx, targets=tgt)
            assert loss is not None and torch.isfinite(loss)
            loss.backward()
            opt_bundle.step(idx=idx, targets=tgt)

        telem = model.compute_manifold_and_purity_telemetry()
        assert telem["max_stiefel_err"] < 1e-4
        assert telem["mean_unitary_err"] < 1e-4
        assert telem["max_wte_sphere_rms_err"] < 1e-4
        assert telem["max_head_sphere_rms_err"] < 1e-4
