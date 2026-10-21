#!/usr/bin/env python3
"""Controlled empirical verification for Experiment Plan 019 (Hypotheses H1, H2, H3).

Writes structured results to `.cache/hypothesis_experiments_019.json`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.model_card import (
    build_model_card_from_manifest_entry,
    validate_model_card,
)
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks.optim.muon import (
    StiefelMuon,
    split_muon_adamw_params,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)
from qdlblocks.optim.schedules import wsd_lr_multiplier
from scripts.monitor_colab_runs import collect_snapshot


def run_h1_verification() -> dict:
    """Verify H1: 0 stale RUNNING badges and valid Wave-11 Model Cards."""
    snap = collect_snapshot(
        configs_dir=REPO_ROOT / "configs",
        model_cards_dir=REPO_ROOT / "results" / "model_cards",
        live_logs_dir=REPO_ROOT / ".cache" / "live_logs",
        include_historical_specs=True,
        skip_remote_poll=True,
    )
    catalog = snap["runs_catalog"]
    by_key = {r["key"]: r for r in catalog}
    running_keys = [r["key"] for r in catalog if r.get("status") == "RUNNING" or r.get("run_state") == "RUNNING"]
    w7_entry = by_key["w7_900m_qborn_earlyfix_frame4d_synergy512"]

    w11_data = json.loads((REPO_ROOT / "configs" / "wave11_fleet.json").read_text())
    card_errors: dict[str, list[str]] = {}
    for r_spec in w11_data["runs"]:
        card = build_model_card_from_manifest_entry(
            r_spec,
            wave_id=w11_data["wave_id"],
            wave_description=w11_data["wave"],
            plan_id=w11_data["plan_id"],
        )
        errs = validate_model_card(card)
        if errs:
            card_errors[r_spec["key"]] = errs

    passed = (
        len(running_keys) == 0
        and w7_entry["status"] == "STOPPED"
        and w7_entry["obsolete"] is True
        and len(card_errors) == 0
    )
    return {
        "passed": passed,
        "running_count": len(running_keys),
        "running_keys": running_keys,
        "w7_status": w7_entry["status"],
        "w7_obsolete": w7_entry["obsolete"],
        "w7_tokens_M": w7_entry.get("live_tokens_M"),
        "w7_best_val_ce": w7_entry.get("best_val_ce"),
        "wave11_run_count": len(w11_data["runs"]),
        "wave11_card_validation_errors": card_errors,
    }


def run_h2_verification(device: torch.device) -> dict:
    """Verify H2: StiefelMuon EMA parity, Step-0 ln(50304) loss, and 60-step convergence advantage."""
    torch.manual_seed(2025)

    # 1. StiefelMuon(ema_momentum=True) exact numerical parity
    shapes = [(64, 64), (256, 64), (64, 256), (192, 64)]
    params_sm = [torch.nn.Parameter(torch.randn(s, device=device, dtype=torch.float32) * 0.05) for s in shapes]
    params_ref = [torch.nn.Parameter(p.detach().clone()) for p in params_sm]
    bufs_ref = [torch.zeros_like(p) for p in params_ref]
    opt_test = StiefelMuon(
        params_sm,
        lr=0.05,
        momentum=0.95,
        momentum_warmup_steps=300,
        momentum_init=0.85,
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
    muon_max_diff = 0.0
    for step_idx in range(10):
        eff_m = opt_test.get_effective_momentum(step_idx + 1)
        grads = [torch.randn_like(p) * 0.02 for p in params_sm]
        for p_s, g in zip(params_sm, grads):
            p_s.grad = g.clone()
        opt_test.step()
        with torch.no_grad():
            g_in_list = []
            for p_r, buf_r, g in zip(params_ref, bufs_ref, grads):
                buf_r.lerp_(g, 1.0 - eff_m)
                g_in = g.lerp(buf_r, eff_m)
                g_in_list.append(g_in)
            g_sq_stack = torch.cat([g_in_list[0].unsqueeze(0), g_in_list[3].view(3, 64, 64)], dim=0)
            u_sq_stack = zeropower_via_newtonschulz5_batched(g_sq_stack, steps=5)
            params_ref[0].add_(u_sq_stack[0].to(params_ref[0].dtype), alpha=-0.05)
            params_ref[3].add_(u_sq_stack[1:].reshape_as(params_ref[3]).to(params_ref[3].dtype), alpha=-0.05)
            for idx_rect in (1, 2):
                p_r = params_ref[idx_rect]
                g_in = g_in_list[idx_rect]
                u = zeropower_via_newtonschulz5(g_in, steps=5)
                scale = max(1.0, float(p_r.size(-2)) / float(p_r.size(-1))) ** 0.5
                p_r.add_(u.to(p_r.dtype), alpha=-0.05 * scale)
        for p_s, p_r in zip(params_sm, params_ref):
            muon_max_diff = max(muon_max_diff, float((p_s - p_r).abs().max().item()))

    # 2. Step-0 exact uniform zero logits and ln(50304) initial loss on 12-layer leaderboard model
    torch.manual_seed(42)
    model_12 = ClassicalModdedGPT(
        vocab_size=50304,
        embed_dim=768,
        num_heads=6,
        num_layers=12,
        hidden_dim=3072,
        block_size=128,
        leaderboard_exact=True,
    ).to(device)
    x0 = torch.randint(0, 50257, (2, 64), device=device, dtype=torch.long)
    y0 = torch.randint(0, 50257, (2, 64), device=device, dtype=torch.long)
    with torch.no_grad():
        logits0, loss0 = model_12(x0, y0)
        assert loss0 is not None
        max_abs_logit0 = float(logits0.abs().max().item())
        step0_loss_50304 = float(loss0.item())
    del model_12
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Controlled 60-step convergence benchmark on structured Zipfian + Markov token stream
    rng = np.random.default_rng(1337)
    vocab_sz = 4096
    ranks = np.arange(1, 513, dtype=np.float64)
    probs = (1.0 / (ranks ** 1.15))
    probs /= probs.sum()
    n_tok = 60 * 16 * 128 + 4096
    base_toks = rng.choice(512, size=n_tok, p=probs).astype(np.int64)
    # Inject deterministic Markov bigram structure so attention + embeddings learn rapidly
    for i in range(1, n_tok):
        if i % 3 == 1:
            base_toks[i] = (base_toks[i - 1] * 7 + 13) % 512
        elif i % 64 == 0:
            base_toks[i] = 511  # document boundary

    train_toks = torch.from_numpy(base_toks[:-2049]).to(device=device, dtype=torch.long)
    val_toks = torch.from_numpy(base_toks[-2049:]).to(device=device, dtype=torch.long)

    def run_60step_trial(leaderboard_exact: bool) -> dict[str, float]:
        torch.manual_seed(99)
        B, T = 16, 128
        total_steps = 60
        if leaderboard_exact:
            m = ClassicalModdedGPT(
                vocab_size=vocab_sz,
                embed_dim=256,
                num_heads=4,  # head_dim=64
                num_layers=6,
                hidden_dim=1024,
                block_size=T,
                leaderboard_exact=True,
            ).to(device)
            m.wte.bfloat16()
            if m.value_embeds is not None:
                m.value_embeds.bfloat16()
        else:
            m = ClassicalModdedGPT(
                vocab_size=vocab_sz,
                embed_dim=256,
                num_heads=8,  # head_dim=32
                num_layers=6,
                hidden_dim=1024,
                block_size=T,
                use_bigram_smear=True,
                use_unet_interferometry=True,
                unet_skip_mode="scalar",
                use_fused_qkv=True,
                num_value_embeddings=3,
                skip_attn_layers=(3,),
                zero_half_rope=True,
                depth_scaled_residual=True,
                logit_softcap=30.0,
                zero_init_residual_and_head=True,
                true_layer0_v_residual=True,
                leaderboard_exact=False,
            ).to(device)

        muon_p, _ = split_muon_adamw_params(
            m,
            adamw_name_patterns=(
                "wte",
                "lm_head",
                "highway_gate",
                "v_mix_gate",
                "value_embeds",
                "unet_scales",
                "unet_lambdas",
                "skip_weights",
                "block_lambdas",
                "sa_lambdas",
                "smear",
            ),
        )
        wte_g = [p for n, p in m.named_parameters() if p.requires_grad and ("wte" in n or "value_embeds" in n)]
        head_g = [p for n, p in m.named_parameters() if p.requires_grad and "lm_head" in n]
        muon_ids = {id(p) for p in muon_p}
        wh_ids = {id(p) for p in wte_g} | {id(p) for p in head_g}
        scalar_g = [p for _, p in m.named_parameters() if p.requires_grad and id(p) not in muon_ids and id(p) not in wh_ids]

        if leaderboard_exact:
            base_muon_lr = 0.05
            opt_m = StiefelMuon(
                muon_p,
                lr=base_muon_lr,
                momentum=0.95,
                momentum_warmup_steps=300,
                momentum_init=0.85,
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
            opt_a: torch.optim.Optimizer = torch.optim.Adam(
                [
                    {"params": head_g, "lr": 0.008, "base_lr": 0.008, "weight_decay": 0.0},
                    {"params": wte_g, "lr": 0.60, "base_lr": 0.60, "weight_decay": 0.0},
                    {"params": scalar_g, "lr": 0.04, "base_lr": 0.04, "weight_decay": 0.0},
                ],
                betas=(0.8, 0.95),
                eps=1e-10,
                fused=(device.type == "cuda"),
            )
            warmup_st = 0
            decay_frac = 0.40
            min_lr = 0.10
        else:
            base_muon_lr = 0.025
            opt_m = StiefelMuon(
                muon_p,
                lr=base_muon_lr,
                momentum=0.95,
                momentum_warmup_steps=300,
                momentum_init=0.85,
                nesterov=True,
                aspect_scale=True,
                ema_momentum=False,
                weight_decay=0.01,
                decay_mode="cautious",
                stiefel_penalty=0.0,
                ns_steps=5,
                retract_every=1_000_000,
                batched_ns=True,
            )
            opt_a = torch.optim.AdamW(
                [
                    {"params": wte_g, "lr": 0.20, "base_lr": 0.20, "weight_decay": 0.0},
                    {"params": head_g, "lr": 0.008, "base_lr": 0.008, "weight_decay": 0.0},
                    {"params": scalar_g, "lr": 0.02, "base_lr": 0.02, "weight_decay": 0.0},
                ],
                betas=(0.9, 0.95),
                fused=(device.type == "cuda"),
            )
            warmup_st = 15
            decay_frac = 0.25
            min_lr = 0.0

        def eval_val() -> float:
            m.eval()
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                xv = val_toks[:-1][: B * T].view(B, T)
                yv = val_toks[1:][: B * T].view(B, T)
                _, l_v = m(xv, yv)
                assert l_v is not None
            m.train()
            return float(l_v.item())

        val_0 = eval_val()
        tok_per_step = B * T
        for st in range(1, total_steps + 1):
            lr_mult = wsd_lr_multiplier(
                st,
                total_steps,
                warmup_steps=warmup_st,
                decay_fraction=decay_frac,
                min_lr_ratio=min_lr,
            )
            for g in opt_m.param_groups:
                g["lr"] = base_muon_lr * lr_mult
            for g in opt_a.param_groups:
                g["lr"] = float(g["base_lr"]) * lr_mult

            start = (st - 1) * tok_per_step
            xb = train_toks[start : start + tok_per_step].view(B, T)
            yb = train_toks[start + 1 : start + tok_per_step + 1].view(B, T)
            opt_m.zero_grad(set_to_none=True)
            opt_a.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, loss = m(xb, yb)
                assert loss is not None
            loss.backward()
            if not leaderboard_exact:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt_m.step()
            opt_a.step()

        val_final = eval_val()
        return {"val_0": round(val_0, 5), "val_final": round(val_final, 5)}

    res_v2 = run_60step_trial(leaderboard_exact=False)
    res_exact = run_60step_trial(leaderboard_exact=True)
    delta_ce = round(res_v2["val_final"] - res_exact["val_final"], 5)

    passed = (
        muon_max_diff < 1e-6
        and max_abs_logit0 == 0.0
        and abs(step0_loss_50304 - math.log(50304.0)) < 1e-4
        and delta_ce >= 0.15
    )
    return {
        "passed": passed,
        "muon_ema_max_diff": muon_max_diff,
        "step0_max_abs_logit": max_abs_logit0,
        "step0_loss_50304": round(step0_loss_50304, 6),
        "expected_ln_50304": round(math.log(50304.0), 6),
        "wave9_v2_60step": res_v2,
        "leaderboard_exact_60step": res_exact,
        "delta_val_ce_improvement_nats": delta_ce,
    }


def run_h3_verification(device: torch.device) -> dict:
    """Verify H3: 100% finite non-zero gradients by Step 2 and 0 Dynamo recompilations across progressive windows."""
    torch.manual_seed(17)
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
    idx[0, 15] = 50256
    targets = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    grad_counts_by_step: dict[str, str] = {}
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    step2_nonzero = 0
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        assert loss is not None
        loss.backward()
        nz = sum(
            1
            for p in model.parameters()
            if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max().item()) > 0.0
        )
        grad_counts_by_step[f"step_{step}"] = f"{nz}/{total_params}"
        if step == 2:
            step2_nonzero = nz
        if step < 2:
            opt.step()

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled_fn = torch.compile(lambda x: model.forward_features(x)[0], dynamic=False)

    model.set_window_size(128)
    _ = compiled_fn(idx)
    frames_after_first = int(torch._dynamo.utils.counters["frames"]["ok"])

    for w in (256, 384, 512):
        model.set_window_size(w)
        out_w = compiled_fn(idx)
        assert torch.isfinite(out_w).all()

    frames_after_all = int(torch._dynamo.utils.counters["frames"]["ok"])
    recompiles = frames_after_all - frames_after_first

    passed = (step2_nonzero == total_params) and (recompiles == 0)
    return {
        "passed": passed,
        "total_trainable_tensors": total_params,
        "nonzero_finite_grads_by_step": grad_counts_by_step,
        "step2_Coverage_pct": round(100.0 * step2_nonzero / max(1, total_params), 2),
        "dynamo_frames_first_step": frames_after_first,
        "dynamo_frames_after_4_windows": frames_after_all,
        "dynamo_recompilations": recompiles,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment Plan 019 Verification on {device}...", flush=True)

    h1 = run_h1_verification()
    print(f"  [H1] Stale RUNNING Fix & Wave-11 Cards: passed={h1['passed']} | running_count={h1['running_count']} | w7_status={h1['w7_status']}", flush=True)

    h2 = run_h2_verification(device)
    print(
        f"  [H2] Leaderboard-Exact Parity & 60-Step Convergence: passed={h2['passed']} | "
        f"muon_diff={h2['muon_ema_max_diff']:.2e} | step0_loss={h2['step0_loss_50304']:.5f} | "
        f"V2={h2['wave9_v2_60step']['val_final']:.4f} vs Exact={h2['leaderboard_exact_60step']['val_final']:.4f} "
        f"(delta={h2['delta_val_ce_improvement_nats']:+.4f} nats)",
        flush=True,
    )

    h3 = run_h3_verification(device)
    print(
        f"  [H3] Progressive Window 0-Recompile & Step-2 Grad Coverage: passed={h3['passed']} | "
        f"grads={h3['nonzero_finite_grads_by_step']} | recompiles={h3['dynamo_recompilations']}",
        flush=True,
    )

    out_path = REPO_ROOT / ".cache" / "hypothesis_experiments_019.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_id": "EXPERIMENT_PLAN_019",
        "device": str(device),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "all_passed": bool(h1["passed"] and h2["passed"] and h3["passed"]),
        "H1_stale_running_and_wave11_cards": h1,
        "H2_leaderboard_exact_parity_and_convergence": h2,
        "H3_progressive_window_zero_recompiles": h3,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved verification report to {out_path} (all_passed={payload['all_passed']})", flush=True)
    if not payload["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
