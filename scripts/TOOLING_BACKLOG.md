# QuantumDLBlocks Operational Scripts & Tooling Backlog (`scripts/TOOLING_BACKLOG.md`)

This document tracks:
1. **Implemented (`Live`) Scripts**: First-class CLI tools in `scripts/` that agents MUST call instead of crafting ad-hoc bash/python one-liners.
2. **Candidate (`Nice-to-Have`) Scripts**: High-leverage tooling identified during experiment planning and wave launches, surfaced to the user so we can decide when to promote them into full implementations.

---

## 1. Implemented (`Live`) Operational Scripts

| Script | Purpose & Replaced Ad-Hoc Pattern | Example Invocation |
| :--- | :--- | :--- |
| [`scripts/monitor_colab_runs.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py) | Single source of truth polling Colab GPU/TPU sessions every 20s, syncing `.cache/live_logs/`, and serving the React dashboard + JSON API on `:8765`. | `python scripts/monitor_colab_runs.py --once` or `curl -s http://127.0.0.1:8765/api/report` |
| [`scripts/colab_fleet.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py) | Declarative Colab fleet manager (`status`, `push-code`, `launch`, `harvest`, `kill`) driven by JSON wave manifests (`configs/wave6_fleet.json`). Replaces ad-hoc `tar + base64` bundles, manual `kill -9 <kernel_pid>` launch strings, and manual log harvesting. | `python scripts/colab_fleet.py launch --manifest configs/wave6_fleet.json` |
| [`scripts/inspect_model.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py) | Parameter budget breakdown (`seed_and_embed`, `attention`, `interference_frames`, `mlp`, `readout_and_adapters`, `purity_and_gates`) + optional N-step CUDA/CPU synthetic smoke test (`tok/s`, peak VRAM, logit L2 norm CV). Replaces ad-hoc `python -c` parameter counting snippets. | `python scripts/inspect_model.py --seed-dim 512 --overcomplete-two-beam --frame-expansion 4.0 --two-beam-seed-adapter --dynamic-povm-beta --smoke-steps 2` |
| [`scripts/overfit_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/overfit_probe.py) | Karpathy memorization capacity & logit-sharpness probe across $N \in \{1, 8, 32, \dots\}$ real FineWeb sequences. Measures `final_ce`, `top1_acc`, `mean_max_prob`, `mean_output_entropy_nats`, and `logit_norm_cv`. | `python scripts/overfit_probe.py --variants q_adapter384_2beam,q_adapter384_2beam_dynbeta,classical_modded --records 1,8,32` |
| [`scripts/calibration_rank_histogram_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/calibration_rank_histogram_probe.py) | Target-rank Cross-Entropy decomposition (`Rank 1`, `Ranks 2–5`, `Ranks 6–50`, `Ranks 51–1000`, `Ranks > 1000`), top-$k$ probability mass allocation, Expected Calibration Error (`ECE`), and layerwise angular velocity & spectral rank $R_\sigma$ across Classical, Quantum POVM, and Quantum Calibrated Softmax models. | `python scripts/calibration_rank_histogram_probe.py --steps 15 --max-vram-gb 8.0` |

---

## 2. Candidate (`Nice-to-Have`) Scripts for User Consideration

Whenever a new `EXPERIMENT_PLAN_0XX.md` is drafted, Section 5 of the plan identifies candidate scripts in two buckets: **`Build-Before-Launch`** (implemented immediately as part of the plan) and **`Nice-to-Have`** (surfaced here and in the plan summary so we can decide whether to implement them).

| Priority | Proposed Script | Problem Solved / Ad-Hoc Work Eliminated | Proposed CLI Interface | Est. Agent Time Saved |
| :---: | :--- | :--- | :--- | :--- |
| **P1** | `scripts/fleet_registry.py` + `configs/fleet_runs.json` | Currently, adding a new wave (`w6_*`, `w7_*`) or updating all-time record KPI cards requires editing 3 Python/React sections inside `scripts/monitor_colab_runs.py` and restarting the daemon. Extracting run specs into a hot-reloaded `configs/fleet_runs.json` managed by `scripts/fleet_registry.py` eliminates all manual edits and daemon restarts. | `python scripts/fleet_registry.py register --manifest configs/wave6_fleet.json --sync-records` | ~5–8 min per wave |
| **P1** | `scripts/verify_plan.py` | Pre-audits an `EXPERIMENT_PLAN_0XX.md` before spawning `blind-plan-reviewer` (checking required sections, explicit quantitative thresholds, optimizer parameter group coverage, and CLI flag parity), and post-audits the implementation (`pytest` + `examples/XX_hypothesis_*.py` + JSON `"overall_pass": true` + 2-step `11_fineweb` smoke test). | `python scripts/verify_plan.py --plan 013 --stage pre-review` / `--stage post-implement` | ~6–10 min per plan |
| **P2** | `scripts/colab_session_pool.py` | Automates connecting/reconnecting a pool of named Colab GPU/TPU sessions (`colab connect -s ... --gpu --variant g4/a100`), verifying assigned GPU tier via `nvidia-smi` (catching silent downgrades to T4), and pre-warming `/content/data/fineweb10B` shards in parallel. | `python scripts/colab_session_pool.py ensure --manifest configs/wave6_fleet.json --prewarm-shards 2` | ~5–10 min per wave |
| **P2** | `scripts/wave_report.py` | Reads all `.cache/live_logs/*_results.json` and `.cache/hypothesis_experiments_*.json` files to generate a unified Markdown + JSON Pareto leaderboard (Val CE vs. Params vs. Tok/s vs. Quantum Telemetry $R_{\text{eff}}, S_{\text{vN}}, \text{IsoErr}$) and automatically updates `PROJECT_AND_SESSION_SUMMARY.md`. | `python scripts/wave_report.py --wave W6 --update-summary` | ~5 min per wave |
| **P3** | `scripts/checkpoint_eval.py` | Runs post-training qualitative generation, per-token entropy vs. dynamic-$\beta_{b,t}$ correlation analysis, and calibration plots on a saved model checkpoint. | `python scripts/checkpoint_eval.py --checkpoint .cache/checkpoints/w6_best.pt --prompts configs/eval_prompts.json` | ~5 min per analysis |
| **P1** | `scripts/colab_fleet.py --compare-waves w6,w7,w8` | Prints an aligned 50M-token trajectory comparison table across waves (`w6,w7,w8`) directly from `.cache/live_logs/` without ad-hoc parsing scripts. | `python scripts/colab_fleet.py compare --waves w6,w7,w8` / `python scripts/colab_fleet.py --compare-waves w6,w7,w8` | ~5 min per wave |
| **P2** | `scripts/colab_fleet.py compare --optimizer-filter` | Groups and filters optimizer ablation runs (`muon_adamw`, `pure_adamw`, `sgd_nesterov`) side-by-side in CLI trajectory tables. | `python scripts/colab_fleet.py compare --waves w8,w9 --optimizer-filter muon_adamw,pure_adamw,sgd_nesterov` | ~3 min per ablation |
| **P2** | `scripts/vram_envelope_autotuner.py` | Binary-searches `(micro_batch_size, grad_accum, checkpoint_blocks, chunk_tokens)` under `torch.cuda.set_per_process_memory_fraction` to find the highest-throughput zero-OOM configuration within a target VRAM cap (e.g., `8.0 GB`). | `python scripts/vram_envelope_autotuner.py --max-vram-gb 8.0 --embed-dim 1024 --num-layers 8` | ~5 min per hardware target |
