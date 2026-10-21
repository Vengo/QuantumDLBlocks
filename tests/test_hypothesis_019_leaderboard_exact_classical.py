"""Tests for Experiment Plan 019: Exact KellerJordan/modded-nanogpt Leaderboard Parity, Stale RUNNING Fix, and Wave-11 Fleet."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.model_card import (
    build_model_card_from_manifest_entry,
    validate_model_card,
)
from qdlblocks.models.classical_modded_gpt import (
    ClassicalModdedGPT,
    LeaderboardRotary,
    apply_leaderboard_rotary,
)
from qdlblocks.optim.muon import (
    StiefelMuon,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
MODEL_CARDS_DIR = REPO_ROOT / "results" / "model_cards"
LIVE_DIR = REPO_ROOT / ".cache" / "live_logs"


def test_stiefel_muon_ema_momentum_parity() -> None:
    """Verify StiefelMuon(ema_momentum=True, weight_decay=0.0) matches exact modded-nanogpt Muon update (<1e-6 diff)."""
    torch.manual_seed(2025)
    d = 64
    d_ff = 256
    lr = 0.05
    mom_target = 0.95
    mom_init = 0.85
    warmup_steps = 300

    # Include square (d, d), rectangular (d_ff, d) and (d, d_ff), and fused QKV (3*d, d)
    shapes = [(d, d), (d_ff, d), (d, d_ff), (3 * d, d)]
    params_sm = [torch.nn.Parameter(torch.randn(s, dtype=torch.float32) * 0.05) for s in shapes]
    params_ref = [torch.nn.Parameter(p.detach().clone()) for p in params_sm]
    bufs_ref = [torch.zeros_like(p) for p in params_ref]

    opt = StiefelMuon(
        params_sm,
        lr=lr,
        momentum=mom_target,
        momentum_warmup_steps=warmup_steps,
        momentum_init=mom_init,
        nesterov=True,
        aspect_scale=True,
        ema_momentum=True,
        weight_decay=0.0,
        decay_mode="none",
        stiefel_penalty=0.0,
        ns_steps=5,
        retract_every=1_000_000,
        batched_ns=True,
    )

    max_diff = 0.0
    for step_idx in range(10):
        eff_m = opt.get_effective_momentum(step_idx + 1)

        grads = [torch.randn_like(p) * 0.02 for p in params_sm]
        for p_s, g in zip(params_sm, grads):
            p_s.grad = g.clone()
        opt.step()

        with torch.no_grad():
            g_in_list = []
            for p_r, buf_r, g in zip(params_ref, bufs_ref, grads):
                # Official modded-nanogpt lines 209-214:
                buf_r.lerp_(g, 1.0 - eff_m)
                g_in = g.lerp(buf_r, eff_m)
                g_in_list.append(g_in)

            # params_ref[0] is (d, d) and params_ref[3] is (3*d, d), which bucket together into (4, d, d)
            g_sq_stack = torch.cat([g_in_list[0].unsqueeze(0), g_in_list[3].view(3, d, d)], dim=0)
            u_sq_stack = zeropower_via_newtonschulz5_batched(g_sq_stack, steps=5)
            params_ref[0].add_(u_sq_stack[0].to(params_ref[0].dtype), alpha=-lr)
            params_ref[3].add_(u_sq_stack[1:].reshape_as(params_ref[3]).to(params_ref[3].dtype), alpha=-lr)

            for idx_rect in (1, 2):
                p_r = params_ref[idx_rect]
                g_in = g_in_list[idx_rect]
                u = zeropower_via_newtonschulz5(g_in, steps=5)
                scale = max(1.0, float(p_r.size(-2)) / float(p_r.size(-1))) ** 0.5
                p_r.add_(u.to(p_r.dtype), alpha=-lr * scale)

        for p_s, p_r in zip(params_sm, params_ref):
            diff = float((p_s - p_r).abs().max().item())
            max_diff = max(max_diff, diff)

    assert max_diff < 1e-6, f"Expected max_diff < 1e-6 vs reference modded-nanogpt Muon, got {max_diff:.3e}"


def test_classical_modded_gpt_leaderboard_exact_architecture() -> None:
    """Verify ClassicalModdedGPT(leaderboard_exact=True) matches all 9 modded-nanogpt architectural invariants."""
    torch.manual_seed(42)

    # 1. Verify full 12-layer leaderboard configuration metadata & structure
    model_12 = ClassicalModdedGPT(
        vocab_size=50304,
        embed_dim=768,
        num_heads=6,
        num_layers=12,
        hidden_dim=3072,
        block_size=128,
        leaderboard_exact=True,
    )
    assert model_12.leaderboard_exact is True
    assert model_12.num_heads == 6
    assert model_12.embed_dim // model_12.num_heads == 128
    assert model_12.skip_attn_layers == (7,)
    assert model_12.logit_softcap == 15.0
    assert model_12.use_bigram_smear is False
    assert model_12.smear_embed_gate is None
    assert model_12.residual_scale == 1.0
    assert isinstance(model_12.rope, LeaderboardRotary)
    assert model_12.rope.head_dim == 128

    # Check half-chunk rotary frequencies: second quarter of angular_freq is strictly zero (cos=1, sin=0)
    cos_16, sin_16 = model_12.rope.get_cos_sin(16)
    assert cos_16.shape == (16, 64)
    assert sin_16.shape == (16, 64)
    assert torch.allclose(cos_16[:, 32:], torch.ones_like(cos_16[:, 32:]))
    assert torch.allclose(sin_16[:, 32:], torch.zeros_like(sin_16[:, 32:]))
    x_rot = torch.randn(2, 6, 16, 128)
    y_rot = apply_leaderboard_rotary(x_rot, cos_16, sin_16)
    assert y_rot.shape == x_rot.shape
    # Because second quarter of frequencies is 0, features [32:64] and [96:128] are unchanged
    assert torch.allclose(y_rot[..., 32:64], x_rot[..., 32:64], atol=1e-6)
    assert torch.allclose(y_rot[..., 96:128], x_rot[..., 96:128], atol=1e-6)

    # Check Value Embedding layer mapping on L=12: [0, 1, 2, None, None, None, None, None, None, 0, 1, 2]
    ve_map_12 = [model_12._ve_bank_index(i) for i in range(12)]
    assert ve_map_12 == [0, 1, 2, None, None, None, None, None, None, 0, 1, 2]

    # Check pre-block skip_weights = ones(6), block_lambdas = [1.0, 0.0], sa_lambdas = [0.5, 0.5]
    assert model_12.skip_weights is not None
    assert torch.allclose(model_12.skip_weights, torch.ones(6))
    for l_idx, blk in enumerate(model_12.blocks):
        assert blk.block_lambdas is not None
        assert torch.allclose(blk.block_lambdas, torch.tensor([1.0, 0.0]))
        assert blk.highway_gate is None
        assert blk.attn_scale == 0.12
        if l_idx == 7:
            assert blk.skip_attention is True
            assert blk.sa_lambdas is None
        else:
            assert blk.skip_attention is False
            assert blk.sa_lambdas is not None
            assert torch.allclose(blk.sa_lambdas, torch.tensor([0.5, 0.5]))

    # 2. Verify Step-0 uniform zero logits, ln(50304) initial loss, and 100% finite non-zero gradients by Step 2
    model_small = ClassicalModdedGPT(
        vocab_size=50304,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        hidden_dim=512,
        block_size=32,
        leaderboard_exact=True,
    )
    assert [model_small._ve_bank_index(i) for i in range(4)] == [0, 1, 0, 1]

    idx = torch.randint(0, 50257, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 50257, (2, 16), dtype=torch.long)

    with torch.no_grad():
        logits_0, _ = model_small(idx)
        assert float(logits_0.abs().max().item()) == 0.0, "Expected exact 0.0 initial logits via 15*tanh(raw/15)"

    opt = torch.optim.Adam(model_small.parameters(), lr=1e-2)
    step0_loss = 0.0
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        _, loss = model_small(idx, targets)
        assert loss is not None
        if step == 0:
            step0_loss = float(loss.item())
        loss.backward()
        if step < 2:
            opt.step()

    expected_loss0 = math.log(50304.0)
    assert abs(step0_loss - expected_loss0) < 1e-4, f"Expected ln(50304)={expected_loss0:.5f}, got {step0_loss:.5f}"

    # At step=2 (3rd backward pass), 100% of parameters have finite non-zero gradients
    total_params = 0
    nonzero_grad_params = 0
    for name, p in model_small.named_parameters():
        if p.requires_grad:
            total_params += 1
            assert p.grad is not None, f"Missing gradient on {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient on {name}"
            if float(p.grad.abs().max().item()) > 0.0:
                nonzero_grad_params += 1

    assert nonzero_grad_params == total_params, (
        f"Expected 100% ({total_params}/{total_params}) parameters to have non-zero gradients by step 2, "
        f"got {nonzero_grad_params}/{total_params}"
    )


def test_progressive_window_zero_recompiles() -> None:
    """Verify torch.compile-wrapped ClassicalModdedGPT(leaderboard_exact=True) executes set_window_size with 0 recompiles."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ClassicalModdedGPT(
        vocab_size=50304,
        embed_dim=128,
        num_heads=4,
        num_layers=6,
        hidden_dim=512,
        block_size=64,
        window_size=128,
        doc_mask_attn=True,
        leaderboard_exact=True,
    ).to(device)

    idx = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
    idx[0, 15] = 50256  # <|endoftext|> boundary

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    compiled_fn = torch.compile(lambda x: model.forward_features(x)[0], dynamic=False)

    # Step 1: initial compilation at window_size=128
    model.set_window_size(128)
    out_128 = compiled_fn(idx)
    assert out_128.shape == (2, 32, 128)
    frames_after_first = int(torch._dynamo.utils.counters["frames"]["ok"])
    assert frames_after_first >= 1

    # Steps 2..4: progressive window size updates (256 -> 384 -> 512) must trigger 0 Dynamo recompilations
    for w in (256, 384, 512):
        model.set_window_size(w)
        assert model.window_size == w
        assert int(model._window_size_buf.item()) == w
        out_w = compiled_fn(idx)
        assert out_w.shape == (2, 32, 128)

    frames_after_all = int(torch._dynamo.utils.counters["frames"]["ok"])
    recompiles = frames_after_all - frames_after_first
    assert recompiles == 0, f"Expected 0 Dynamo recompilations across set_window_size updates, got {recompiles}"


def test_monitor_stale_running_fix_and_wave11_cards() -> None:
    """Verify collect_snapshot marks stale incomplete runs as STOPPED (0 false RUNNING) and validates Wave-11 cards."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.monitor_colab_runs import DEFAULT_OBSOLETE_KEYS, collect_snapshot

    snap = collect_snapshot(
        configs_dir=CONFIGS_DIR,
        model_cards_dir=MODEL_CARDS_DIR,
        live_logs_dir=LIVE_DIR,
        include_historical_specs=True,
        skip_remote_poll=True,
    )
    catalog = snap["runs_catalog"]
    by_key = {r["key"]: r for r in catalog}

    # 1. Verify w7_900m_qborn_earlyfix_frame4d_synergy512 is STOPPED (not RUNNING) and obsolete
    assert "w7_900m_qborn_earlyfix_frame4d_synergy512" in by_key
    w7_run = by_key["w7_900m_qborn_earlyfix_frame4d_synergy512"]
    assert w7_run["status"] == "STOPPED", f"Expected STOPPED for w7_900m_qborn_earlyfix_frame4d_synergy512, got {w7_run['status']}"
    assert w7_run["obsolete"] is True

    # 2. When skip_remote_poll=True and no remote process is live, 0 runs should report status == 'RUNNING'
    running_runs = [r["key"] for r in catalog if r.get("status") == "RUNNING" or r.get("run_state") == "RUNNING"]
    assert running_runs == [], f"Expected 0 stale RUNNING runs when skip_remote_poll=True, got {running_runs}"

    # 3. Verify all superseded incomplete runs in DEFAULT_OBSOLETE_KEYS are marked obsolete
    for obs_k in (
        "w7_900m_qborn_earlyfix_frame4d_synergy512",
        "w6_900m_classical_modded_gpt_baseline",
        "w8_900m_classical_truev0_zeroinit_wsd0",
        "w9_900m_classical_speedrun_v2_t512_b524k",
        "w10_900m_classical_speedrun_v3_fastsdpa_b131k",
        "w10_900m_classical_speedrun_v3_b65k_13725steps",
    ):
        assert obs_k in DEFAULT_OBSOLETE_KEYS
        if obs_k in by_key:
            assert by_key[obs_k]["obsolete"] is True

    # 4. Validate configs/wave11_fleet.json and its 5 Model Cards
    w11_path = CONFIGS_DIR / "wave11_fleet.json"
    assert w11_path.exists(), "configs/wave11_fleet.json must exist"
    w11_data = json.loads(w11_path.read_text())
    assert w11_data["wave_id"] == "W11"
    assert w11_data["plan_id"] == "EXPERIMENT_PLAN_019"
    runs = w11_data["runs"]
    assert len(runs) == 5

    classical_exact_keys = {
        "w11_900m_classical_leaderboard_exact_b524k_1716st",
        "w11_900m_classical_leaderboard_exact_b262k_3429st",
        "w11_900m_classical_leaderboard_exact_progwin_docmask",
    }
    for r_spec in runs:
        card = build_model_card_from_manifest_entry(
            r_spec,
            wave_id=w11_data["wave_id"],
            wave_description=w11_data["wave"],
            plan_id=w11_data["plan_id"],
        )
        errors = validate_model_card(card)
        assert errors == [], f"Model card validation errors for {r_spec['key']}: {errors}"
        if r_spec["key"] in classical_exact_keys:
            assert card["architecture"]["leaderboard_exact_classical"] is True
            assert card["architecture"]["num_heads"] == 6
            assert card["architecture"]["head_dim"] == 128
            assert card["architecture"]["logit_softcap"] == 15.0
            assert card["optimization"]["ema_momentum"] is True
            assert card["optimization"]["weight_decay"] == 0.0
            assert card["optimization"]["adamw_lr_wte"] == 0.60
            assert card["optimization"]["adamw_lr_scalar"] == 0.04
            assert card["optimization"]["warmup_steps"] == 0
        if r_spec["key"] == "w11_900m_classical_leaderboard_exact_progwin_docmask":
            assert card["architecture"]["progressive_window"] is True
            assert card["architecture"]["doc_mask_attn"] is True
            assert card["architecture"]["sliding_window"] == 128
