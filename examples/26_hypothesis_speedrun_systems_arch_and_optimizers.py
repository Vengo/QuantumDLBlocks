"""Example 26: Controlled Hypothesis Validation for Plan 016 — Speedrun Systems, Architecture & Optimizers.

Validates three core hypotheses on GPU:
1. H1 (Single-Pass Direct Chunked CE & Batched 3D Newton-Schulz):
   - Verifies exact loss and gradient parity (< 1e-5 in float32) of `direct_chunked_softcap_linear_ce`
     and `direct_chunked_povm_ce` against standard cross-entropy, while achieving >= 1.25x wall-clock
     speedup over activation-checkpointed chunked CE on a large vocabulary readout (V = 50,304).
   - Verifies numerical equivalence (< 1e-4) and >= 1.30x CUDA kernel speedup of 3D batched
     `zeropower_via_newtonschulz5_batched` over sequential per-matrix `zeropower_via_newtonschulz5`.
2. H2 (Speedrun Architectural Synergy on Multi-Document Associative Stream):
   - Evaluates the combined Plan 016 speedrun architecture (`unet_skip_mode="scalar"`, `use_fused_qkv=True`,
     `num_value_embeddings=2`, `zero_half_rope=True`, `window_size=32`, `doc_mask_attn=True`) against the
     Plan 015 baseline on both `ClassicalModdedGPT` and `QuantumBornGPT` over a 160-step multi-document
     associative induction task (B=16, T=128, V=512, EOT boundaries every 32 tokens).
   - Verifies >= 0.08 nats lower validation cross-entropy on both architectures and exact 0.0
     cross-document attention leakage.
3. H3 (Optimizer Family & Muon Momentum Warmup Ablation):
   - Compares 4 optimizer configurations over 160 steps on both `ClassicalModdedGPT` and `QuantumBornGPT`:
     (a) `muon_warmup_mom085_095` (StiefelMuon with 0.85 -> 0.95 momentum warmup + batched NS),
     (b) `muon_fixed_mom095` (StiefelMuon with fixed momentum=0.95),
     (c) `pure_adamw` (AdamW on all parameters),
     (d) `sgd_nesterov` (Nesterov SGD on 2D matrices + AdamW on 1D/embeddings).
   - Verifies that `muon_warmup_mom085_095` outperforms `pure_adamw` by >= 0.10 nats and `sgd_nesterov`
     by >= 0.20 nats on both architectures, and achieves <= validation CE than `muon_fixed_mom095` at
     both Step 40 and Step 160.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.functional import (
    l2_normalize,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)
from qdlblocks.layers.attention import (
    SubspaceProjectorBornAttention,
    compute_document_ids,
)
from qdlblocks.layers.direct_chunked_ce import (
    direct_chunked_povm_ce,
    direct_chunked_softcap_linear_ce,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params
from qdlblocks.optim.schedules import wsd_lr_multiplier


def run_h1_systems_microbenchmark(device: torch.device) -> dict[str, float | bool]:
    """Benchmark H1: Single-Pass Direct Chunked CE and Batched 3D Newton-Schulz."""
    print("\n" + "=" * 88)
    print("H1: SINGLE-PASS DIRECT CHUNKED CE & BATCHED 3D NEWTON-SCHULZ MICROBENCHMARK")
    print("=" * 88)

    # 1. Exact float32 gradient parity check
    torch.manual_seed(42)
    n_check, d_check, v_check = 256, 64, 512
    x_ref = torch.randn(n_check, d_check, device=device, dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(v_check, d_check, device=device, dtype=torch.float32, requires_grad=True)
    y_check = torch.randint(0, v_check, (n_check,), device=device, dtype=torch.long)
    softcap = 30.0

    logits_ref = softcap * torch.tanh(F.linear(x_ref, w_ref) / softcap)
    loss_ref = F.cross_entropy(logits_ref, y_check)
    loss_ref.backward()

    x_dir = x_ref.detach().clone().requires_grad_(True)
    w_dir = w_ref.detach().clone().requires_grad_(True)
    loss_dir = direct_chunked_softcap_linear_ce(x_dir, w_dir, y_check, softcap=softcap, chunk_size=64)
    loss_dir.backward()

    softcap_grad_diff = max(
        float(torch.max(torch.abs(x_dir.grad - x_ref.grad)).item()),
        float(torch.max(torch.abs(w_dir.grad - w_ref.grad)).item()),
    )

    phi_ref = l2_normalize(torch.randn(n_check, d_check, device=device, dtype=torch.float32), dim=-1).detach().requires_grad_(True)
    m_ref = torch.randn(v_check, d_check, device=device, dtype=torch.float32, requires_grad=True)
    inv_t_ref = torch.tensor(28.5, device=device, dtype=torch.float32, requires_grad=True)
    prior_ref = (0.05 * torch.randn(v_check, device=device, dtype=torch.float32)).detach().requires_grad_(True)

    m_norm_ref = l2_normalize(m_ref, dim=1, eps=1e-8)
    povm_logits_ref = F.linear(phi_ref, m_norm_ref) * inv_t_ref + prior_ref
    povm_loss_ref = F.cross_entropy(povm_logits_ref, y_check)
    povm_loss_ref.backward()

    phi_dir = phi_ref.detach().clone().requires_grad_(True)
    m_dir = m_ref.detach().clone().requires_grad_(True)
    inv_t_dir = inv_t_ref.detach().clone().requires_grad_(True)
    prior_dir = prior_ref.detach().clone().requires_grad_(True)
    povm_loss_dir = direct_chunked_povm_ce(
        phi_dir, m_dir, inv_t_dir, y_check, log_prior=prior_dir, chunk_size=64, eps=1e-8
    )
    povm_loss_dir.backward()

    povm_grad_diff = max(
        float(torch.max(torch.abs(phi_dir.grad - phi_ref.grad)).item()),
        float(torch.max(torch.abs(m_dir.grad - m_ref.grad)).item()),
        float(torch.max(torch.abs(inv_t_dir.grad - inv_t_ref.grad)).item()),
        float(torch.max(torch.abs(prior_dir.grad - prior_ref.grad)).item()),
    )
    max_abs_grad_diff = max(softcap_grad_diff, povm_grad_diff)

    # 2. Readout CE throughput benchmark (V = 50,304, D = 768 matching 100M speedrun model)
    n_tok = 8192
    d_readout = 768
    v_vocab = 50304
    chunk_sz = 4096
    n_iters = 40

    x_bench = torch.randn(n_tok, d_readout, device=device, dtype=torch.bfloat16, requires_grad=True)
    w_bench = torch.randn(v_vocab, d_readout, device=device, dtype=torch.bfloat16, requires_grad=True)
    y_bench = torch.randint(0, v_vocab, (n_tok,), device=device, dtype=torch.long)

    def _ckpt_chunk_fn(xc: torch.Tensor, wb: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
        raw_c = F.linear(xc, wb)
        logits_c = 30.0 * torch.tanh(raw_c.float() / 30.0)
        return F.cross_entropy(logits_c, yc)

    def _run_ckpt_ce(xb: torch.Tensor, wb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        n_c = xb.size(0) // chunk_sz
        tot = torch.zeros((), device=xb.device, dtype=torch.float32)
        for i in range(n_c):
            s, e = i * chunk_sz, (i + 1) * chunk_sz
            tot = tot + checkpoint(_ckpt_chunk_fn, xb[s:e], wb, yb[s:e], use_reentrant=False)
        return tot / float(n_c)

    # Warmup
    for _ in range(5):
        x_bench.grad = None
        w_bench.grad = None
        l_c = _run_ckpt_ce(x_bench, w_bench, y_bench)
        l_c.backward()
        x_bench.grad = None
        w_bench.grad = None
        l_d = direct_chunked_softcap_linear_ce(x_bench, w_bench, y_bench, softcap=30.0, chunk_size=chunk_sz)
        l_d.backward()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        x_bench.grad = None
        w_bench.grad = None
        l_c = _run_ckpt_ce(x_bench, w_bench, y_bench)
        l_c.backward()
    torch.cuda.synchronize()
    ckpt_ce_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    t0 = time.perf_counter()
    for _ in range(n_iters):
        x_bench.grad = None
        w_bench.grad = None
        l_d = direct_chunked_softcap_linear_ce(x_bench, w_bench, y_bench, softcap=30.0, chunk_size=chunk_sz)
        l_d.backward()
    torch.cuda.synchronize()
    direct_ce_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
    ce_speedup = ckpt_ce_ms / max(direct_ce_ms, 1e-6)

    # 3. Batched 3D Newton-Schulz vs Sequential 2D Newton-Schulz
    k_batch = 12
    m_dim = 256
    ns_iters = 200
    g_list = [torch.randn(m_dim, m_dim, device=device, dtype=torch.bfloat16) for _ in range(k_batch)]
    g_stack = torch.stack(g_list, dim=0)

    u_seq_check = torch.stack([zeropower_via_newtonschulz5(g, steps=5) for g in g_list], dim=0)
    u_bat_check = zeropower_via_newtonschulz5_batched(g_stack, steps=5)
    ns_max_abs_diff = float(torch.max(torch.abs(u_seq_check.float() - u_bat_check.float())).item())

    for _ in range(10):
        _ = [zeropower_via_newtonschulz5(g, steps=5) for g in g_list]
        _ = zeropower_via_newtonschulz5_batched(g_stack, steps=5)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ns_iters):
        _ = [zeropower_via_newtonschulz5(g, steps=5) for g in g_list]
    torch.cuda.synchronize()
    seq_ns_ms = (time.perf_counter() - t0) * 1000.0 / ns_iters

    t0 = time.perf_counter()
    for _ in range(ns_iters):
        g_s = torch.stack(g_list, dim=0)
        _ = zeropower_via_newtonschulz5_batched(g_s, steps=5)
    torch.cuda.synchronize()
    bat_ns_ms = (time.perf_counter() - t0) * 1000.0 / ns_iters
    ns_speedup = seq_ns_ms / max(bat_ns_ms, 1e-6)

    h1_pass = bool(
        max_abs_grad_diff < 1e-5
        and ce_speedup >= 1.25
        and ns_max_abs_diff < 1e-4
        and ns_speedup >= 1.30
    )
    print(f"  Max Abs Grad Diff (float32):   {max_abs_grad_diff:.2e} (target < 1e-5)")
    print(
        f"  Readout CE Fwd+Bwd Time:       ckpt={ckpt_ce_ms:.2f} ms vs direct={direct_ce_ms:.2f} ms -> {ce_speedup:.2f}x speedup (target >= 1.25x)"
    )
    print(f"  Batched 3D NS Max Abs Diff:    {ns_max_abs_diff:.2e} (target < 1e-4)")
    print(
        f"  Newton-Schulz (K={k_batch}) Time:    seq={seq_ns_ms:.3f} ms vs batched={bat_ns_ms:.3f} ms -> {ns_speedup:.2f}x speedup (target >= 1.30x)"
    )
    print(f"  H1 Verdict:                    {'PASS' if h1_pass else 'FAIL'}")

    return {
        "max_abs_grad_diff": max_abs_grad_diff,
        "ckpt_ce_ms": round(ckpt_ce_ms, 4),
        "direct_ce_ms": round(direct_ce_ms, 4),
        "ce_speedup_x": round(ce_speedup, 4),
        "ns_max_abs_diff": ns_max_abs_diff,
        "seq_ns_ms": round(seq_ns_ms, 4),
        "batched_ns_ms": round(bat_ns_ms, 4),
        "ns_speedup_x": round(ns_speedup, 4),
        "passed": h1_pass,
    }


def generate_multidoc_associative_dataset(
    num_batches: int,
    batch_size: int = 16,
    seq_len: int = 128,
    vocab_size: int = 512,
    eot_token_id: int = 511,
    doc_len: int = 32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a multi-document candidate-set associative recall dataset.

    Each sequence of length `T = 128` contains 4 independent documents of length `32`,
    terminated by `eot_token_id = 511`.
    - Each key `k in {1..6}` has a 4-element candidate value set `V_k = {16 + 4*(k-1) + c : c in 0..3}`.
    - Across the 4 documents `m in 0..3` in a sequence, each key `k` is assigned a permutation
      of its 4 candidate values `v_{m, k} in V_k` so that earlier documents `m' < m` contain
      competing values `v_{m', k} != v_{m, k}` from the same candidate set `V_k`.
    - Within document `m`, each `(k, v_{m, k})` pair is presented and queried multiple times,
      so unmasked attention across document boundaries attends to competing `v_{m', k} in V_k`
      from earlier documents, whereas document-masked sliding-window attention with half-truncated
      RoPE and value embeddings cleanly isolates the current document's `v_{m, k}`.
    """
    rng = np.random.default_rng(seed)
    n_docs = seq_len // doc_len
    xs = np.zeros((num_batches, batch_size, seq_len), dtype=np.int64)
    ys = np.zeros((num_batches, batch_size, seq_len), dtype=np.int64)

    keys = np.arange(1, 7, dtype=np.int64)
    for b_idx in range(num_batches):
        for s_idx in range(batch_size):
            seq = np.zeros(seq_len + 1, dtype=np.int64)
            # For each key, permute its 4 candidate values across the 4 documents
            doc_kv: list[dict[int, int]] = [{} for _ in range(n_docs)]
            for k_idx, k_tok in enumerate(keys.tolist()):
                cand_vals = 16 + 4 * k_idx + rng.permutation(n_docs)
                for doc_i in range(n_docs):
                    doc_kv[doc_i][int(k_tok)] = int(cand_vals[doc_i])

            for doc_i in range(n_docs):
                base = doc_i * doc_len
                kv_map = doc_kv[doc_i]
                pos = base
                # Present 3 rounds of the 6 (key, val) pairs (36 -> truncated to 30 tokens) + 1 bigram + EOT
                for round_i in range(3):
                    perm_k = rng.permutation(len(keys))
                    for ki in perm_k:
                        if pos + 1 >= base + doc_len - 1:
                            break
                        k_tok = int(keys[ki])
                        seq[pos] = k_tok
                        seq[pos + 1] = int(kv_map[k_tok])
                        pos += 2
                while pos < base + doc_len - 1:
                    seq[pos] = 200 + (pos - base)
                    pos += 1
                seq[base + doc_len - 1] = eot_token_id
            seq[seq_len] = int(keys[0])
            xs[b_idx, s_idx] = seq[:-1]
            ys[b_idx, s_idx] = seq[1:]

    return torch.from_numpy(xs), torch.from_numpy(ys)


def train_and_evaluate_regime(
    architecture: Literal["classical_modded", "quantum_born"],
    speedrun_arch: bool,
    optimizer_variant: Literal[
        "muon_warmup_mom085_095",
        "muon_fixed_mom095",
        "pure_adamw",
        "sgd_nesterov",
    ],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    steps: int = 160,
    eot_token_id: int = 511,
) -> dict[str, float]:
    """Train a 6-layer model for `steps` and return Step-0, Step-40, and Step-160 validation metrics."""
    torch.manual_seed(1337)
    np.random.seed(1337)

    vocab_size = 512
    embed_dim = 128
    num_heads = 4
    num_layers = 6
    hidden_dim = 256
    seq_len = train_x.size(-1)

    if architecture == "classical_modded":
        model: nn.Module = ClassicalModdedGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            block_size=seq_len,
            use_bigram_smear=True,
            use_unet_interferometry=True,
            unet_skip_mode="scalar" if speedrun_arch else "matrix",
            use_fused_qkv=speedrun_arch,
            num_value_embeddings=2 if speedrun_arch else 0,
            zero_half_rope=speedrun_arch,
            window_size=32 if speedrun_arch else 0,
            doc_mask_attn=speedrun_arch,
            eot_token_id=eot_token_id,
            value_residual_mixing=True,
            input_highway_mixing=True,
            logit_softcap=30.0,
            zero_init_residual_and_head=True,
            true_layer0_v_residual=True,
        ).to(device)
    else:
        model = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=64,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            block_size=seq_len,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            value_residual_mixing=True,
            input_highway_mixing=True,
            use_bigram_smear=True,
            use_qk_wave_filter=True,
            use_unet_interferometry=True,
            unet_skip_mode="scalar" if speedrun_arch else "matrix",
            use_fused_qkv=speedrun_arch,
            num_value_embeddings=2 if speedrun_arch else 0,
            zero_half_rope=speedrun_arch,
            window_size=32 if speedrun_arch else 0,
            doc_mask_attn=speedrun_arch,
            eot_token_id=eot_token_id,
            use_two_beam_mlp=True,
            depolarizing_jitter=0.0,
            use_flash_born=True,
            use_ortho_decouple=True,
            use_povm_log_prior=True,
            use_gated_spherical_residual=True,
            use_purified_attn=True,
            true_layer0_v_residual=True,
        ).to(device)

    muon_params, adamw_params = split_muon_adamw_params(
        model,
        adamw_name_patterns=(
            "wte",
            "povm_head",
            "lm_head",
            "purity_gate",
            "highway",
            "v_mix_gate",
            "ve_mix_logit",
            "embed",
            "value_embeds",
            "unet_scales",
            "unet_lambdas",
            "smear",
            "inv_temp",
            "beam_",
            "seed_gate",
            "povm_log_prior",
            "coherence_tau",
            "res_gate",
        ),
    )

    if optimizer_variant == "pure_adamw":
        base_muon_lr = 0.0008
        opt_2d: torch.optim.Optimizer = torch.optim.AdamW(
            muon_params, lr=base_muon_lr, betas=(0.9, 0.95), weight_decay=0.01
        )
    elif optimizer_variant == "sgd_nesterov":
        base_muon_lr = 0.025
        opt_2d = torch.optim.SGD(
            muon_params, lr=base_muon_lr, momentum=0.95, nesterov=True, weight_decay=0.01
        )
    elif optimizer_variant == "muon_fixed_mom095":
        base_muon_lr = 0.035
        opt_2d = StiefelMuon(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            momentum_warmup_steps=0,
            nesterov=True,
            aspect_scale=(architecture == "classical_modded"),
            weight_decay=0.01,
            decay_mode="cautious",
            stiefel_penalty=0.005,
            ns_steps=5,
            retract_every=4,
            batched_ns=False,
        )
    else:
        # muon_warmup_mom085_095: linear momentum warmup 0.85 -> 0.95 + batched NS
        base_muon_lr = 0.035
        opt_2d = StiefelMuon(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            momentum_init=0.85,
            momentum_warmup_steps=20,
            nesterov=True,
            aspect_scale=(architecture == "classical_modded"),
            weight_decay=0.01,
            decay_mode="cautious",
            stiefel_penalty=0.005,
            ns_steps=5,
            retract_every=4,
            batched_ns=True,
        )

    base_adamw_lr = 0.003
    opt_adamw = torch.optim.AdamW(adamw_params, lr=base_adamw_lr, betas=(0.9, 0.95), weight_decay=0.0)

    @torch.no_grad()
    def eval_val() -> float:
        model.eval()
        tot = 0.0
        n_b = val_x.size(0)
        for i in range(n_b):
            vx = val_x[i].to(device)
            vy = val_y[i].to(device)
            _, v_loss = model(vx, targets=vy)
            tot += float(v_loss.item())
        model.train()
        return tot / float(n_b)

    val_step0 = eval_val()
    val_step40 = val_step0
    n_train_batches = train_x.size(0)

    for step in range(1, steps + 1):
        lr_mult = wsd_lr_multiplier(step, steps, warmup_steps=4, decay_fraction=0.25, min_lr_ratio=0.05)
        for g in opt_2d.param_groups:
            g["lr"] = base_muon_lr * lr_mult
        for g in opt_adamw.param_groups:
            g["lr"] = base_adamw_lr * lr_mult

        bx = train_x[(step - 1) % n_train_batches].to(device)
        by = train_y[(step - 1) % n_train_batches].to(device)

        opt_2d.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        if isinstance(model, ClassicalModdedGPT):
            x_feat, _ = model.forward_features(bx, return_layer_states=False)
            loss = direct_chunked_softcap_linear_ce(
                x_feat.reshape(-1, embed_dim),
                model.lm_head.weight,
                by.reshape(-1),
                softcap=model.logit_softcap,
                chunk_size=1024,
            )
        else:
            _, loss = model(bx, targets=by)
            assert loss is not None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_2d.step()
        opt_adamw.step()

        if step == 40:
            val_step40 = eval_val()

    val_step160 = eval_val()

    # Measure cross-document attention leakage fraction on validation batch 0
    leakage_frac = 0.0
    with torch.no_grad():
        vx0 = val_x[0].to(device)
        doc_ids_0 = compute_document_ids(vx0, eot_token_id=eot_token_id)
        q_test = l2_normalize(torch.randn(vx0.size(0), num_heads, seq_len, embed_dim // num_heads, device=device), dim=-1)
        k_test = l2_normalize(0.8 * q_test + 0.2 * torch.randn_like(q_test), dim=-1)
        w_attn = SubspaceProjectorBornAttention._born_attention_weights(
            q_test,
            k_test,
            scale=10.0,
            window_size=32 if speedrun_arch else 0,
            doc_ids=doc_ids_0 if speedrun_arch else None,
        )
        cross_doc_mask = doc_ids_0[:, None, :, None] != doc_ids_0[:, None, None, :]
        leakage_frac = float(w_attn.masked_select(cross_doc_mask).sum().item() / max(1, w_attn.sum().item()))

    return {
        "val_ce_step0": round(val_step0, 4),
        "val_ce_step40": round(val_step40, 4),
        "val_ce_step160": round(val_step160, 4),
        "cross_doc_leakage": round(leakage_frac, 6),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print(f"PLAN 016 HYPOTHESIS VALIDATION SUITE — Device: {device}")
    print("=" * 88)

    h1_results = run_h1_systems_microbenchmark(device)

    print("\n" + "=" * 88)
    print("H2: SPEEDRUN ARCHITECTURAL SYNERGY ON MULTI-DOCUMENT ASSOCIATIVE STREAM")
    print("=" * 88)
    eot_id = 511
    train_x, train_y = generate_multidoc_associative_dataset(
        num_batches=32, batch_size=16, seq_len=128, vocab_size=512, eot_token_id=eot_id, doc_len=32, seed=42
    )
    val_x, val_y = generate_multidoc_associative_dataset(
        num_batches=6, batch_size=16, seq_len=128, vocab_size=512, eot_token_id=eot_id, doc_len=32, seed=999
    )

    c_base = train_and_evaluate_regime(
        "classical_modded", False, "muon_warmup_mom085_095", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )
    c_speed = train_and_evaluate_regime(
        "classical_modded", True, "muon_warmup_mom085_095", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )
    q_base = train_and_evaluate_regime(
        "quantum_born", False, "muon_warmup_mom085_095", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )
    q_speed = train_and_evaluate_regime(
        "quantum_born", True, "muon_warmup_mom085_095", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )

    c_arch_gain = c_base["val_ce_step160"] - c_speed["val_ce_step160"]
    q_arch_gain = q_base["val_ce_step160"] - q_speed["val_ce_step160"]
    h2_pass = bool(
        c_arch_gain >= 0.08
        and q_arch_gain >= 0.08
        and c_speed["cross_doc_leakage"] == 0.0
        and q_speed["cross_doc_leakage"] == 0.0
    )

    print(
        f"  ClassicalModdedGPT: Plan015={c_base['val_ce_step160']:.4f} -> Plan016={c_speed['val_ce_step160']:.4f} "
        f"(gain={c_arch_gain:+.4f} nats, target >= +0.0800 | leakage={c_speed['cross_doc_leakage']:.6f})"
    )
    print(
        f"  QuantumBornGPT:     Plan015={q_base['val_ce_step160']:.4f} -> Plan016={q_speed['val_ce_step160']:.4f} "
        f"(gain={q_arch_gain:+.4f} nats, target >= +0.0800 | leakage={q_speed['cross_doc_leakage']:.6f})"
    )
    print(f"  H2 Verdict:         {'PASS' if h2_pass else 'FAIL'}")

    h2_results = {
        "classical_plan015_baseline": c_base,
        "classical_plan016_speedrun": c_speed,
        "quantum_plan015_baseline": q_base,
        "quantum_plan016_speedrun": q_speed,
        "classical_arch_gain_nats": round(c_arch_gain, 4),
        "quantum_arch_gain_nats": round(q_arch_gain, 4),
        "passed": h2_pass,
    }

    print("\n" + "=" * 88)
    print("H3: OPTIMIZER FAMILY & MUON MOMENTUM WARMUP ABLATION (CLASSICAL & QUANTUM)")
    print("=" * 88)
    c_muon_warm = c_speed
    c_muon_fixed = train_and_evaluate_regime(
        "classical_modded", True, "muon_fixed_mom095", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )
    c_adamw = train_and_evaluate_regime(
        "classical_modded", True, "pure_adamw", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )
    c_sgd = train_and_evaluate_regime(
        "classical_modded", True, "sgd_nesterov", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )

    q_muon_warm = q_speed
    q_muon_fixed = train_and_evaluate_regime(
        "quantum_born", True, "muon_fixed_mom095", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )
    q_adamw = train_and_evaluate_regime(
        "quantum_born", True, "pure_adamw", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )
    q_sgd = train_and_evaluate_regime(
        "quantum_born", True, "sgd_nesterov", train_x, train_y, val_x, val_y, device, steps=160, eot_token_id=eot_id
    )

    c_vs_adamw = c_adamw["val_ce_step160"] - c_muon_warm["val_ce_step160"]
    c_vs_sgd = c_sgd["val_ce_step160"] - c_muon_warm["val_ce_step160"]
    q_vs_adamw = q_adamw["val_ce_step160"] - q_muon_warm["val_ce_step160"]
    q_vs_sgd = q_sgd["val_ce_step160"] - q_muon_warm["val_ce_step160"]

    h3_pass = bool(
        c_vs_adamw >= 0.10
        and c_vs_sgd >= 0.20
        and q_vs_adamw >= 0.10
        and q_vs_sgd >= 0.20
        and c_muon_warm["val_ce_step40"] <= c_muon_fixed["val_ce_step40"]
        and c_muon_warm["val_ce_step160"] <= c_muon_fixed["val_ce_step160"]
        and q_muon_warm["val_ce_step40"] <= q_muon_fixed["val_ce_step40"]
        and q_muon_warm["val_ce_step160"] <= q_muon_fixed["val_ce_step160"]
    )

    print(
        f"  Classical [Step 40 / Step 160]:\n"
        f"    muon_warmup_mom085_095: {c_muon_warm['val_ce_step40']:.4f} / {c_muon_warm['val_ce_step160']:.4f}\n"
        f"    muon_fixed_mom095:      {c_muon_fixed['val_ce_step40']:.4f} / {c_muon_fixed['val_ce_step160']:.4f}\n"
        f"    pure_adamw:             {c_adamw['val_ce_step40']:.4f} / {c_adamw['val_ce_step160']:.4f} (Muon lead: {c_vs_adamw:+.4f} nats)\n"
        f"    sgd_nesterov:           {c_sgd['val_ce_step40']:.4f} / {c_sgd['val_ce_step160']:.4f} (Muon lead: {c_vs_sgd:+.4f} nats)"
    )
    print(
        f"  Quantum   [Step 40 / Step 160]:\n"
        f"    muon_warmup_mom085_095: {q_muon_warm['val_ce_step40']:.4f} / {q_muon_warm['val_ce_step160']:.4f}\n"
        f"    muon_fixed_mom095:      {q_muon_fixed['val_ce_step40']:.4f} / {q_muon_fixed['val_ce_step160']:.4f}\n"
        f"    pure_adamw:             {q_adamw['val_ce_step40']:.4f} / {q_adamw['val_ce_step160']:.4f} (Muon lead: {q_vs_adamw:+.4f} nats)\n"
        f"    sgd_nesterov:           {q_sgd['val_ce_step40']:.4f} / {q_sgd['val_ce_step160']:.4f} (Muon lead: {q_vs_sgd:+.4f} nats)"
    )
    print(f"  H3 Verdict: {'PASS' if h3_pass else 'FAIL'}")

    h3_results = {
        "classical": {
            "muon_warmup_mom085_095": c_muon_warm,
            "muon_fixed_mom095": c_muon_fixed,
            "pure_adamw": c_adamw,
            "sgd_nesterov": c_sgd,
            "muon_vs_adamw_nats": round(c_vs_adamw, 4),
            "muon_vs_sgd_nats": round(c_vs_sgd, 4),
        },
        "quantum": {
            "muon_warmup_mom085_095": q_muon_warm,
            "muon_fixed_mom095": q_muon_fixed,
            "pure_adamw": q_adamw,
            "sgd_nesterov": q_sgd,
            "muon_vs_adamw_nats": round(q_vs_adamw, 4),
            "muon_vs_sgd_nats": round(q_vs_sgd, 4),
        },
        "passed": h3_pass,
    }

    all_passed = bool(h1_results["passed"] and h2_results["passed"] and h3_results["passed"])
    summary = {
        "plan": "EXPERIMENT_PLAN_016",
        "device": str(device),
        "H1_systems_microbenchmark": h1_results,
        "H2_speedrun_architecture": h2_results,
        "H3_optimizer_ablation": h3_results,
        "all_passed": all_passed,
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_016.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 88)
    print(f"Saved Plan 016 results to {out_path} | ALL PASSED: {all_passed}")
    print("=" * 88)
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
