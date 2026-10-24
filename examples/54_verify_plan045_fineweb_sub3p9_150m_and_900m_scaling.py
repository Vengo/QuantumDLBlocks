#!/usr/bin/env python3
"""Wave 29 Verification Runner: Breaking <3.90 Val CE Under 150M Tokens & 900M Scaling (Plan 045).

Verifies Hypotheses H1, H2, and H3 from `EXPERIMENT_PLAN_045.md` using the dedicated
Rule-6 flag-free model `ParsevalMuPNGramSpeedrunGPT` (`src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py`)
and the multi-horizon Chinchilla scaling projector (`scripts/fineweb_150m_300m_900m_scaling_projector.py`):
- H1: Parameter budget (`d=256` and `d=384`), exact post-norm Cholesky temperature absorption
  (`max_logit_abs_err < 1e-4`), finite non-zero gradients across `ngram_proj`, `v0_gate`,
  `kraus_generators`, and `wav_65536`, and GPU training throughput (`>= 24,000 tok/s`).
- H2: Controlled head-to-head FineWeb comparison (`train_full_100M.bin` -> `val_slice.bin`):
  * Arm 1 (`w28_parseval_mup_baseline_d256_900steps`, `7.37M` tokens): `5.4372` Val CE.
  * Arm 2 (`w29_parseval_mup_ngram_v0_b16_d256_900steps`, `7.37M` tokens at `d=256`): `<= 5.32` Val CE.
  * Arm 3 (`w29_parseval_mup_ngram_v0_b16_d384_1800steps`, `14.75M` tokens at `d=384`): `< 4.98` Val CE in `< 75s`.
- H3: Multi-horizon FineWeb scaling projection up to `150M`, `300M`, and `900M` tokens (`< 3.90` Val CE
  under `150M` tokens, `~3.55-3.63` at `300M` tokens, `~3.26-3.34` at `900M` tokens).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(ROOT / "src"), str(ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger  # noqa: E402
from qdlblocks.models.parseval_mup_ngram_speedrun_gpt import (  # noqa: E402
    ParsevalMuPNGramSpeedrunConfig,
    ParsevalMuPNGramSpeedrunGPT,
    build_parseval_mup_ngram_optimizers,
    initialize_parseval_mup_ngram_with_ppmi_svd,
)
from scripts.fineweb_150m_300m_900m_scaling_projector import ScalingProjectorConfig, run_scaling_projector  # noqa: E402

_EX48_SPEC = importlib.util.spec_from_file_location(
    "plan038_runner", ROOT / "examples" / "48_verify_plan038_last_layer_measurement_and_traceless.py"
)
assert _EX48_SPEC is not None and _EX48_SPEC.loader is not None
_EX48 = importlib.util.module_from_spec(_EX48_SPEC)
sys.modules["plan038_runner"] = _EX48
_EX48_SPEC.loader.exec_module(_EX48)

_EX53_SPEC = importlib.util.spec_from_file_location(
    "plan043_runner", ROOT / "examples" / "53_verify_plan043_parseval_sqrt_d_noclamp_readout.py"
)
assert _EX53_SPEC is not None and _EX53_SPEC.loader is not None
_EX53 = importlib.util.module_from_spec(_EX53_SPEC)
sys.modules["plan043_runner"] = _EX53
_EX53_SPEC.loader.exec_module(_EX53)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass(frozen=True)
class W29ArmSpec:
    """Configuration for a Wave 29 FineWeb speedrun scaling arm."""

    name: str
    embed_dim: int
    num_heads: int
    num_layers: int = 2
    batch_size: int = 16
    block_size: int = 512
    steps: int = 900
    eta_base: float = 0.0065
    eta_max: float = 0.026
    hypothesis_intent: str = ""


def verify_h1_architecture_and_throughput(device: torch.device) -> dict[str, Any]:
    """Verify H1: parameter budget, Cholesky absorption, finite gradients, and >=24k tok/s throughput."""
    torch.manual_seed(42)
    cfg_256 = ParsevalMuPNGramSpeedrunConfig(block_size=512, embed_dim=256, num_heads=4, num_layers=2)
    cfg_384 = ParsevalMuPNGramSpeedrunConfig(block_size=512, embed_dim=384, num_heads=6, num_layers=2)
    model_256 = ParsevalMuPNGramSpeedrunGPT(cfg_256).to(device)
    model_384 = ParsevalMuPNGramSpeedrunGPT(cfg_384).to(device)
    params_256 = sum(p.numel() for p in model_256.parameters())
    params_384 = sum(p.numel() for p in model_384.parameters())
    ngram_params_256 = sum(p.numel() for p in model_256.ngram_proj.parameters())
    del model_384

    model_abs = ParsevalMuPNGramSpeedrunGPT(cfg_256).eval()
    x_probe = torch.randint(0, cfg_256.valid_vocab_size, (2, 16), dtype=torch.long)
    with torch.no_grad():
        logits_pre, _ = model_abs(x_probe, targets=None, return_logits=True)
        abs_info = model_abs.absorb_beta_into_readout_cholesky()
        logits_post, _ = model_abs(x_probe, targets=None, return_logits=True)
    assert logits_pre is not None and logits_post is not None
    v_eff = cfg_256.valid_vocab_size
    max_logit_err = float((logits_pre[..., :v_eff].float() - logits_post[..., :v_eff].float()).abs().max().item())
    del model_abs

    model_g = ParsevalMuPNGramSpeedrunGPT(cfg_256).to(device)
    model_g.train()
    x_g = torch.randint(0, v_eff, (2, 64), device=device, dtype=torch.long)
    y_g = torch.randint(0, v_eff, (2, 64), device=device, dtype=torch.long)
    _, loss_g = model_g(x_g, y_g, return_logits=False)
    assert loss_g is not None
    loss_g.backward()
    grad_norms = {
        "ngram_unigram": float(model_g.ngram_proj.unigram_table.grad.norm().item()),  # type: ignore[union-attr]
        "ngram_bigram": float(model_g.ngram_proj.bigram_table.grad.norm().item()),  # type: ignore[union-attr]
        "ngram_trigram": float(model_g.ngram_proj.trigram_table.grad.norm().item()),  # type: ignore[union-attr]
        "ngram_u_stiefel": float(model_g.ngram_proj.u_ngram.grad.norm().item()),  # type: ignore[union-attr]
        "v0_gate": float(model_g.v0_gate.grad.norm().item()),  # type: ignore[union-attr]
        "block0_v0_gate": float(model_g.blocks[0].v0_gate.grad.norm().item()),  # type: ignore[union-attr]
        "kraus_generators": float(model_g.pyramid.kraus_generators.grad.norm().item()),  # type: ignore[union-attr]
        "wav_65536": float(model_g.pyramid.wav_65536.grad.norm().item()),  # type: ignore[union-attr]
    }
    grads_finite_nonzero = all(math.isfinite(v) and v > 0.0 for v in grad_norms.values())

    muon_opt, adam_opt = build_parseval_mup_ngram_optimizers(model_g)
    x_tp = torch.randint(0, v_eff, (16, 512), device=device, dtype=torch.long)
    y_tp = torch.randint(0, v_eff, (16, 512), device=device, dtype=torch.long)
    for _ in range(3):
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, l_w = model_g(x_tp, y_tp, return_logits=False)
        assert l_w is not None
        l_w.backward()
        muon_opt.step()
        adam_opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    bench_steps = 8
    for _ in range(bench_steps):
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, l_b = model_g(x_tp, y_tp, return_logits=False)
        assert l_b is not None
        l_b.backward()
        muon_opt.step()
        adam_opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tok_per_s = round((bench_steps * 16 * 512) / max(1e-6, time.perf_counter() - t0), 1)
    del model_256, model_g, muon_opt, adam_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    h1_pass = bool(max_logit_err < 1e-4 and grads_finite_nonzero and tok_per_s >= 24000.0)
    print(
        f"  [H1] params(d=256)={params_256:,} | params(d=384)={params_384:,} | "
        f"chol_err={max_logit_err:.2e} | grads_ok={grads_finite_nonzero} | tok/s={tok_per_s:,.1f}",
        flush=True,
    )
    return {
        "pass": h1_pass,
        "params_d256": params_256,
        "params_d384": params_384,
        "ngram_proj_params_d256": ngram_params_256,
        "cholesky_absorption_max_logit_err": max_logit_err,
        "cholesky_absorption_info": abs_info,
        "grad_norms": grad_norms,
        "grads_finite_nonzero": grads_finite_nonzero,
        "gpu_throughput_tok_per_s": tok_per_s,
    }


def emit_w28_baseline_arm() -> dict[str, Any]:
    """Emit standardized Arm 1 (`w28_parseval_mup_baseline_d256_900steps`) from verified Wave 28 run."""
    w28_src = ROOT / ".cache" / "live_logs" / "w28_flagfree_parseval_mup_b16_d256_900steps_results.json"
    src_data = json.loads(w28_src.read_text(encoding="utf-8"))
    rkey = "w28_parseval_mup_baseline_d256_900steps"
    intent = ExperimentIntentSpec(
        run_key=rkey, wave_id="W29", plan_id="EXPERIMENT_PLAN_045", task_domain="fineweb_edu_local",
        hypothesis_intent="Baseline Wave 28 Flag-Free Parseval muP sqrt(d) readout at d=256 (7.37M tokens, 5.4372 Val CE).",
        architecture_summary="ParsevalMuPDyadicGPT (L=4, d=256, H=4, 256-root dyadic pyramid)",
        batch_size=16, seq_len=512, total_steps=900, target_tokens_M=7.37,
        params_M=float(src_data.get("params_M", 11.79)), embed_dim=256, num_layers=4,
    )
    logger = StandardizedExperimentLogger(intent)
    for ev in src_data.get("evals", []):
        logger.log_eval(int(ev["step"]), ev)
    res = logger.finalize(extra_summary={"val_ce_at_tau_0p50": src_data.get("val_ce_at_tau_0p50", 5.5625)})
    print(f"  [{rkey}] (Baseline) final_val_ce={res['final_val_ce']:.4f} | top1={res['final_top1_acc']:.2f}%", flush=True)
    return res


def run_w29_arm(
    spec: W29ArmSpec, device: torch.device, shared_data: dict[str, Any], force_rerun: bool = False
) -> dict[str, Any]:
    """Train and evaluate a Wave 29 `ParsevalMuPNGramSpeedrunGPT` arm with standardized live logging."""
    cache_path = ROOT / ".cache" / "live_logs" / f"{spec.name}_results.json"
    if cache_path.exists() and not force_rerun:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("completed") and cached.get("final_val_ce") is not None:
            print(
                f"  [{spec.name}] (Cached) d={spec.embed_dim} | final_val_ce={cached['final_val_ce']:.4f} | "
                f"top1={cached.get('final_top1_acc', 0.0):.2f}% | wall_s={cached.get('wall_s', 0.0):.1f}s",
                flush=True,
            )
            return cached

    tok_per_step = spec.batch_size * spec.block_size
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()

    cfg = ParsevalMuPNGramSpeedrunConfig(
        block_size=spec.block_size, embed_dim=spec.embed_dim, num_heads=spec.num_heads,
        num_layers=spec.num_layers, window_size=min(384, spec.block_size), readout_gemm_precision="fp8",
    )
    model = ParsevalMuPNGramSpeedrunGPT(cfg).to(device)
    init_stats = initialize_parseval_mup_ngram_with_ppmi_svd(
        model, shared_data["ppmi_emb_256"], shared_data["init_tokens"], shared_data["prior_tokens"]
    )
    if shared_data.get("kn_state") is not None:
        model.ngram_proj._kn_state = shared_data["kn_state"]
    else:
        shared_data["kn_state"] = model.ngram_proj._kn_state

    muon_opt, adam_opt = build_parseval_mup_ngram_optimizers(
        model, base_muon_lr=spec.eta_max, base_wav_lr=0.0085, base_head_lr=0.005
    )
    n_params = sum(p.numel() for p in model.parameters())
    params_m = round(n_params / 1e6, 4)
    intent = ExperimentIntentSpec(
        run_key=spec.name, wave_id="W29", plan_id="EXPERIMENT_PLAN_045", task_domain="fineweb_edu_local",
        hypothesis_intent=spec.hypothesis_intent,
        architecture_summary=f"ParsevalMuPNGramSpeedrunGPT (L={spec.num_layers}, d={spec.embed_dim}, H={spec.num_heads}, HashedNGram+V0)",
        batch_size=spec.batch_size, seq_len=spec.block_size, total_steps=spec.steps,
        target_tokens_M=round((spec.steps * tok_per_step) / 1e6, 2), params_M=params_m,
        embed_dim=spec.embed_dim, num_layers=spec.num_layers,
    )
    logger = StandardizedExperimentLogger(intent)
    work_gpu: torch.Tensor = shared_data["work_gpu"]
    val_batches = shared_data["val_batches"]
    n_work = int(work_gpu.numel())
    mid_step = spec.steps // 2
    eval_steps_set = {0, spec.steps // 4, mid_step, (3 * spec.steps) // 4, spec.steps}
    val_ce_mid = 0.0
    max_stiefel_err = float(init_stats["kraus_stiefel_err"])
    t0 = time.perf_counter()

    for step in range(spec.steps + 1):
        if step > 0:
            model.train()
            model.set_training_progress(float(step - 1) / float(spec.steps))
            lr_256, mom = _EX53.compute_step_lr_and_momentum(
                "modded_nanogpt_wsd", step - 1, spec.steps, spec.eta_base, spec.eta_max
            )
            _EX53.apply_scaled_group_lrs(muon_opt, adam_opt, lr_256, spec.embed_dim, mom)
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            base = ((step - 1) * tok_per_step) % max(1, n_work - tok_per_step - 2)
            c_i64 = work_gpu[base : base + tok_per_step + 1].to(torch.int64) & 0xFFFF
            x_in = c_i64[:-1].view(spec.batch_size, spec.block_size)
            y_tg = c_i64[1:].view(spec.batch_size, spec.block_size)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x_in, y_tg, return_logits=False)
            assert loss is not None
            loss.backward()
            muon_opt.step()
            adam_opt.step()
            if step % 90 == 0 or step == spec.steps:
                max_stiefel_err = max(max_stiefel_err, model.retract_kraus_generators())
            if step % 150 == 0 or step == spec.steps:
                el_s = max(1e-3, time.perf_counter() - t0)
                logger.log_step(step, float(loss.item()), (step * tok_per_step) / el_s, lr_256 / spec.eta_max, el_s / 60.0)

        if step in eval_steps_set:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model.set_training_progress(float(step) / float(spec.steps))
            ev = _EX53.eval_parseval_mup_quantum(model, val_batches)
            el_s = max(1e-3, time.perf_counter() - t0)
            tok_seen = step * tok_per_step
            tau = float(step) / float(max(1, spec.steps))
            ev.update({
                "step": step, "tokens_M": round(tok_seen / 1e6, 2),
                "compute_tflops": round(6.0 * params_m * (tok_seen / 1e6), 4),
                "tau_progress": round(tau, 4), "elapsed_min": round(el_s / 60.0, 4),
                "tok_per_s": round(tok_seen / el_s, 1) if step > 0 else 0.0,
            })
            logger.log_eval(step, ev)
            if step == mid_step:
                val_ce_mid = float(ev["val_ce"])
            print(
                f"  [{spec.name}] step={step:4d}/{spec.steps} ({ev['tokens_M']:5.2f}M tok, tau={tau:.2f}) | "
                f"val_ce={ev['val_ce']:.4f} | top1={ev['val_top1']:.2f}% | elapsed={el_s:.1f}s | tok/s={ev['tok_per_s']:.0f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_s = round(time.perf_counter() - t0, 2)
    extra = {
        "params": n_params, "params_M": params_m, "val_ce_at_tau_0p50": round(val_ce_mid, 4),
        "max_kraus_stiefel_err": round(max_stiefel_err, 6),
        "u_ngram_stiefel_err": round(model.ngram_proj.retract_stiefel_basis(), 6), "elapsed_sec": wall_s,
    }
    res = logger.finalize(extra_summary=extra)
    del model, muon_opt, adam_opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def compute_early_log_token_slope(evals: list[dict[str, Any]]) -> float:
    """Compute log-token scaling slope `(CE(tok_last) - CE(tok_first_nonzero)) / log10(tok_last / tok_first)`."""
    pts = [(float(e["tokens_M"]), float(e["val_ce"])) for e in evals if float(e.get("tokens_M", 0.0)) >= 1.5]
    if len(pts) < 2:
        return 0.0
    t_first, ce_first = pts[0]
    t_last, ce_last = pts[-1]
    return round((ce_last - ce_first) / max(1e-6, math.log10(t_last / t_first)), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / ".cache" / "hypothesis_experiments_045.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    h1_res = verify_h1_architecture_and_throughput(device)

    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=85_000_000, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt", map_location="cpu", weights_only=True)
    ppmi_emb_256 = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(val_arr, batch_size=4, seq_len=512, num_batches=6, device=device)
    shared_data: dict[str, Any] = {
        "ppmi_emb_256": ppmi_emb_256,
        "init_tokens": torch.from_numpy(train_arr[:85_000_000].astype("int32")),
        "prior_tokens": torch.from_numpy(train_arr[:2_000_000].astype("int64")),
        "work_gpu": torch.from_numpy(train_arr[:18_000_000].view("int16")).to(device),
        "val_batches": val_batches,
        "kn_state": None,
    }

    arm1_res = emit_w28_baseline_arm()
    arm2_spec = W29ArmSpec(
        name="w29_parseval_mup_ngram_v0_b16_d256_900steps", embed_dim=256, num_heads=4, steps=900,
        hypothesis_intent="Verify HashedSurfaceNGramProjector + Cross-Layer V^(0) residual achieves <=5.32 Val CE at 7.37M tokens (d=256).",
    )
    arm3_spec = W29ArmSpec(
        name="w29_parseval_mup_ngram_v0_b16_d384_1800steps", embed_dim=384, num_heads=6, steps=1800,
        hypothesis_intent="Verify d=384 ParsevalMuPNGramSpeedrunGPT achieves <4.98 Val CE at 14.75M tokens in <75s and projects <3.90 under 150M tokens.",
    )
    arm2_res = run_w29_arm(arm2_spec, device, shared_data, force_rerun=args.force_rerun)
    arm3_res = run_w29_arm(arm3_spec, device, shared_data, force_rerun=args.force_rerun)

    scaling_proj = run_scaling_projector(ScalingProjectorConfig())
    w29_traj = scaling_proj["w29_flagship_trajectory"]
    post_cd = w29_traj["post_wsd_cooldown_by_horizon"]
    pre_cd = w29_traj["pre_cooldown_by_horizon"]

    ce_w28 = float(arm1_res["final_val_ce"])
    ce_d256 = float(arm2_res["final_val_ce"])
    ce_d384 = float(arm3_res["final_val_ce"])
    wall_d384 = float(arm3_res.get("elapsed_sec", arm3_res.get("wall_s", 0.0)))
    slope_w28 = compute_early_log_token_slope(arm1_res["evals"])
    slope_w29 = compute_early_log_token_slope(arm2_res["evals"])

    h2_pass = bool(ce_d256 <= 5.32 and ce_d256 < ce_w28 and ce_d384 < 4.98 and wall_d384 < 75.0)
    ce_150m = float(post_cd["150M"])
    ce_300m = float(post_cd["300M"])
    ce_900m = float(post_cd["900M"])
    h3_pass = bool(ce_150m < 3.90 and 3.45 <= ce_300m <= 3.68 and 3.20 <= ce_900m <= 3.38)

    payload = {
        "plan": "EXPERIMENT_PLAN_045",
        "wave": "Wave 29",
        "H1": h1_res,
        "H2": {
            "pass": h2_pass,
            "w28_baseline_d256_ce": ce_w28,
            "w29_ngram_v0_d256_ce": ce_d256,
            "d256_delta_ce_vs_w28": round(ce_d256 - ce_w28, 4),
            "w28_log_token_slope": slope_w28,
            "w29_log_token_slope": slope_w29,
            "w29_ngram_v0_d384_1800steps_ce": ce_d384,
            "w29_d384_elapsed_sec": wall_d384,
        },
        "H3": {
            "pass": h3_pass,
            "projected_ce_50M": float(post_cd["50M"]),
            "projected_ce_100M": float(post_cd["100M"]),
            "projected_ce_150M_post_cooldown": ce_150m,
            "projected_ce_150M_pre_cooldown": float(pre_cd["150M"]),
            "tokens_M_to_sub_3p90": float(w29_traj["tokens_m_to_break_3p90_post_cooldown"]),
            "projected_ce_300M": ce_300m,
            "projected_ce_450M": float(post_cd["450M"]),
            "projected_ce_900M": ce_900m,
        },
        "variants": {
            arm1_res["run_key"]: arm1_res,
            arm2_res["run_key"]: arm2_res,
            arm3_res["run_key"]: arm3_res,
        },
        "overall_pass": bool(h1_res["pass"] and h2_pass and h3_pass),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} (overall_pass={payload['overall_pass']})", flush=True)


if __name__ == "__main__":
    main()
