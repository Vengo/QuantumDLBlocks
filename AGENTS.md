# QuantumDLBlocks Agent Guidelines (`AGENTS.md`)

## 1. Live Colab GPU/TPU Experiment Monitoring (`monitor_colab_runs.py`)

- The background daemon `scripts/monitor_colab_runs.py --serve --host 0.0.0.0 --port 8765 --interval 20` is the **single source of truth** that automatically polls remote Colab sessions every 20 seconds and syncs their logs into `.cache/live_logs/`.
- **Do not spawn competing `colab exec` commands** just to check training progress while `--serve` is running. Instead:
  - Run `curl -s http://127.0.0.1:8765/api/report` or `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once` (which reads directly from `http://127.0.0.1:8765/api/snapshot` in `<10ms`).
  - Or inspect the synced log files directly in `/home/festivity/agy/QuantumDLBlocks/.cache/live_logs/`.
- See `.agents/skills/colab-experiment-monitor/SKILL.md` for full details on launching detached runs and comparing against the FineWeb 100M Slowrun (`3.183` Val CE) and Modded-NanoGPT Speedrun (`3.280` Val CE in `0.665` min) records on the genuine `4,999,488`-token validation shard.

## 2. First-Class Operational Scripts (Use Scripts Instead of Ad-Hoc Commands)

Agents MUST use the repository's reusable CLI tools in `scripts/` instead of writing ad-hoc `python -c` or multi-line `colab exec` bash snippets:

1. **Colab Fleet Code Sync, Launch, Status, & Harvesting (`scripts/colab_fleet.py`)**:
   - Push `src/` + `examples/11_*.py` and launch a wave from a declarative manifest (`configs/wave6_fleet.json`):
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py launch --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave6_fleet.json
     ```
   - Check live GPU & PID health or harvest completed `.log` / `_results.json` artifacts:
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py status --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave6_fleet.json
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py harvest --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave6_fleet.json
     ```
2. **Model Parameter Breakdown & Synthetic Smoke Testing (`scripts/inspect_model.py`)**:
   - Inspect module-group parameter budgets and run a 2-step CUDA/CPU forward+backward smoke test:
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py --seed-dim 512 --overcomplete-two-beam --frame-expansion 4.0 --two-beam-seed-adapter --dynamic-povm-beta --smoke-steps 2
     ```
3. **Karpathy Memorization Capacity & Logit-Sharpness Probe (`scripts/overfit_probe.py`)**:
   - Measure overfitting accuracy (`top1_acc`), output entropy (`mean_output_entropy_nats`), and logit L2 norm variation (`logit_norm_cv`) across $N \in \{1, 8, 32\}$ FineWeb records:
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/overfit_probe.py --variants q_adapter384_2beam,q_adapter384_2beam_dynbeta,classical_modded --records 1,8,32
     ```

## 3. Standard Planning & Execution Workflow: Mandatory Tooling & "Nice-to-Have" Script Gate

Every new `EXPERIMENT_PLAN_0XX.md` and subagent workflow MUST include an explicit **Operational Tooling & Script Identification** gate:

1. **In Every `EXPERIMENT_PLAN_0XX.md` (Required Section 5: Operational Tooling & Script Opportunities)**:
   - **Existing Scripts Used**: Specify exact `scripts/inspect_model.py`, `scripts/overfit_probe.py`, `scripts/colab_fleet.py`, and `scripts/monitor_colab_runs.py` invocations that will validate, launch, and monitor the experiment.
   - **`Build-Before-Launch` Scripts**: Any repetitive command needed to execute or launch the plan that is not yet scripted must be added to `scripts/` as part of the plan implementation before fleet launch.
   - **`Nice-to-Have` Scripts (Surfaced to User at Launch)**: Identify 1–3 candidate scripts that would make launching, monitoring, or analyzing this class of task even easier in the future. Log them in `scripts/TOOLING_BACKLOG.md` and highlight them to the user when presenting the plan/launch so we can decide together whether to build them immediately.
2. **In Blind Plan Reviews (`.agents/agents/blind-plan-reviewer/agent.md`)**:
   - The blind reviewer audits Section 5 alongside Hypotheses, Planned Changes, and Testing to verify that launch/verification steps rely on deterministic scripts rather than fragile ad-hoc commands, and that `Nice-to-Have` script opportunities are clearly articulated.
3. **In Implementer Subagent Handoffs (Friction & Ad-Hoc Command Audit)**:
   - Every implementer subagent must include a short **Tooling Friction Audit** in its completion report listing any ad-hoc one-liners it ran more than once so they can be promoted into `scripts/`.

## 4. Script & Module Engineering Best Practices (`docs/SCRIPT_BEST_PRACTICES.md`)

All agents and subagents writing or modifying Python files in `examples/`, `scripts/`, and `src/qdlblocks/` MUST enforce the eight rules in [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md):

1. **Rule 1 — Comment-Free Code-Length Budget (`<= 1,000` SLOC Soft Warning Limit, `<= 2,500` SLOC Hard Limit, Comments & Docstrings Are Free)**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) counts **only executable code lines (SLOC)**—comments (`# ...`), module/class/function docstrings (`"""..."""`), and blank lines do **not** count against the `<= 1,000` soft limit or `<= 2,500` hard limit (or `<= 40` nested-function limit). Never strip comments/docstrings or pack statements with semicolons (`a; b`).
2. **Rule 2 — Function Signature Ceiling (`<= 8` Arguments)**:
   - No new function or method may accept more than `8` positional/keyword arguments (excluding `self`/`cls`). Group domain hyperparameters into frozen `@dataclass(frozen=True)` objects (`BackboneConfig`, `QubitHierarchyConfig`, `KBosonReadoutConfig`, `OptimizerConfig`, `QuantumExperimentSpec` in [`src/qdlblocks/benchmarks/config_models.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py)).
3. **Rule 3 — Strategy / Polymorphic Dispatch Over Boolean Flag Jungles**:
   - Replace mutually exclusive boolean flags selecting `if / elif / else` branches with explicit strategy classes, enums, or composable sub-configs resolved once at construction time.
4. **Rule 4 — Single Shared Builders Between Local `8 GB` Scripts & Cloud Runners**:
   - Never copy-paste `initialize_povm_log_prior`, `build_quantum_born_optimizers`, `build_quantum_model_from_spec`, or validation loops into per-plan scripts; always import them from [`src/qdlblocks/benchmarks/config_models.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py) and [`src/qdlblocks/benchmarks/calibration_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/calibration_probe.py).
5. **Rule 5 — No Giant Closure-Capturing Nested Functions**:
   - Keep helper functions top-level, pure, and independently unit-testable via `pytest`.
6. **Rule 6 — Clean, Flag-Free Per-Experiment Modules Over Boolean-Flag Accumulation in Legacy Models**:
   - Never thread new `use_*` or `is_*` boolean flags into legacy model or loss files (`wavelet_cholesky_2pow16_gpt.py`, `direct_chunked_ce.py`) when testing a new mathematical formulation. Instead, import shared building-block primitives (`BlockCholeskyLinear`, `WaveletCholeskyMultiscaleBranch`, `OvercompleteTwoBeamInterference`, `StiefelMuon`) into a clean, dedicated, flag-free module and verification script that directly express the experiment's equations with zero legacy branches, leaving previous wave models frozen for reproducible comparison.
7. **Rule 7 — Top-of-File Named Constants & Mandatory `EXPERIMENT_PLAN_0XX.md` Synchronization**:
   - All numerical constants, scaling coefficients, exponents, thresholds, and empirical calibration factors must be defined as documented `UPPER_SNAKE_CASE` constants at the start of the file (never as unexplained inline magic numbers inside functions). Introducing or modifying any constant during implementation requires updating the corresponding `EXPERIMENT_PLAN_0XX.md` (and [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md) if in a shared module) to document its definition and rationale.
8. **Rule 8 — Thorough Building-Block Commenting & Experiment Provenance**:
   - Because comments and docstrings do not count against line limits, every reusable building block in `src/qdlblocks/` must be commented thoroughly enough for a fresh reviewer to understand *when* that piece is useful, *why* it was built mathematically, and *which experiment/wave* (`EXPERIMENT_PLAN_0XX.md` / `Wave XX`) generated and validated it.

## 5. Reusable Building-Blocks Catalog (`docs/BUILDING_BLOCKS.md`) & Mandatory Contribute-Back Protocol

Before drafting any new `EXPERIMENT_PLAN_0XX.md`, writing a new model/layer/kernel, or launching a subagent, agents MUST read [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md) (and the top-level summary in [`README.md`](file:///home/festivity/agy/QuantumDLBlocks/README.md)) to understand the current state of our reusable components:

1. **Know & Compose the Canonical Building Blocks at Launch**:
   - **Fused GPU Kernels (`src/qdlblocks/kernels/`)**:
     - `compiled_parseval_convex_traceless_chunk_step`, `fused_fp8_or_bf16_chunk_gemm`, `quantize_fp8_vocab_transposed` ([`fused_parseval_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_parseval_ce.py)): Reusable `@torch.compile` + FP8/BF16 Tensor-Core readout kernel (`24.7k–25.3k tok/s` at $d=256$).
     - `flash_born_attention`, `flash_purified_born_attention` ([`flash_born_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_born_triton.py)): $O(T)$ SRAM-tiled Triton Born attention (`64x64` tiles, automatic for $d \ge 384$).
   - **Canonical Layers & Flag-Free Readouts (`src/qdlblocks/layers/`)**:
     - `parseval_mup_dyadic_chunked_ce`, `eval_parseval_mup_full_logits`, `ParsevalChunkedCEReadoutSpec` ([`parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py)): Flag-free analytical Parseval $\mu\text{P}$ scale $\beta_0(d) = c^*\sqrt{d}$ ($c^* = 1.48$) + Convex-Peak Traceless Born + 256-root dyadic conditional readout (replaces legacy `PurityGate`, `kboson_raw_k`, and manual `26.8` / `44.0` temperature clamps).
     - `subspace_l2_normalize`, `subspace_coincidence_dyadic_chunked_ce`, `eval_subspace_coincidence_full_logits`, `SubspaceCoincidenceSpec` ([`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py)): Product-of-Spheres $(\frac{1}{\sqrt{M}}\mathbb{S}^{d/M - 1})^M$ normalization and $M$-subspace coincidence gating (`Wave 31` / `EXPERIMENT_PLAN_046.md`).
     - `BlockCholeskyLinear(d, max_block_size=32, fast_spd_bmm=True)` & `WaveletCholeskyMultiscaleBranch` ([`wavelet_cholesky_branch.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky_branch.py)): Batched `torch.bmm` SPD operators (`2.55x` faster) and Sweldens lifting wavelets.
     - `OvercompleteTwoBeamInterference` ([`overcomplete_two_beam.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_two_beam.py)): Mach-Zehnder two-beam phase interferometer.
   - **Canonical Models & Pyramids (`src/qdlblocks/models/`)**:
     - `ParsevalMuPDyadicGPT` ([`parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py)), `ParsevalMuPNGramSpeedrunGPT` ([`parseval_mup_ngram_speedrun_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py)), and `ParsevalMuPSubspaceCoincidenceGPT` ([`parseval_mup_subspace_coincidence_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py)): Current canonical flag-free language models with proportional wavelet detail ranks ($d_{\text{mid}} = d // 2, d_{\text{fine}} = d // 4$) and post-normalization Cholesky metric absorption (`absorb_beta_into_readout_cholesky()`).
     - `WaveletCholeskyVisionClassifier` & `TiedDyadicWaveletClassPyramid` ([`wavelet_cholesky_vision.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_vision.py)): `~11s`/run `CIFAR-100` & `ILSVRC ImageNet-1K` ($K=1,000$) fast-loop ablation laboratory.
   - **Riemannian Stiefel Optimization (`src/qdlblocks/optim/muon.py`)**:
     - `StiefelMuon` (`polar_express`, `stiefel_penalty=0.02`, `use_normuon=True`, `use_twin_rail=True`): Always route `kraus_generators` (`(R, d, d)`) and 2D orthogonal bases to `StiefelMuon` + `retract_kraus_generators()`.
2. **Contribute Back When Validating New Components**:
   - Whenever an experiment wave proves out a reusable kernel, layer, optimizer schedule, or diagnostic probe, extract it into `src/qdlblocks/{kernels,layers,optim,benchmarks}/` or `scripts/`, export it in `__init__.py`, comment it thoroughly with its `Wave` / `EXPERIMENT_PLAN_0XX` provenance (Rule 8), and update [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md) and [`README.md`](file:///home/festivity/agy/QuantumDLBlocks/README.md) (marking any superseded component as `FROZEN` or `LEGACY`) before closing out the wave.

## 6. Mandatory Standardized Experiment Intent & Progress Logging (`StandardizedExperimentLogger`)

Every training experiment—whether executed on remote Colab (`100M–900M` FineWeb speedruns), on the local workstation (`8 GB` / `15 GB` `FineWeb-Edu` scaling runs), or on vision/synthetic ablation tasks (`CIFAR-100`, `ILSVRC ImageNet-1K`)—MUST record its experimental intent and live progress using [`StandardizedExperimentLogger`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/experiment_logger.py) (`from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger`):

1. **Declare Explicit Intent Before Step 0 (`ExperimentIntentSpec`)**:
   - Every run must emit an `[EXPERIMENT_INTENT]` header block to `.cache/live_logs/{run_key}.log` and `.cache/model_cards/{run_key}.json` specifying:
     - `run_key`, `wave_id` (e.g. `W28`, `W29`), `plan_id` (e.g. `EXPERIMENT_PLAN_045`)
     - `task_domain`: `"fineweb_speedrun"` (full 100M–900M FineWeb), `"fineweb_edu_local"` (local FineWeb-Edu Wavelet-Cholesky scaling), `"vision_ablation"` (`CIFAR-100` / `ImageNet-1K`), or `"micro_verification"`
     - `hypothesis_intent`: 1–2 sentence statement of what hypothesis the run tests and its quantitative target
     - `architecture_summary`, `hardware_budget`, `batch_size`, `seq_len`, `total_steps`, `target_tokens_M`
2. **Stream Standardized Step & Eval Progress Live (`logger.log_step(...)` & `logger.log_eval(...)`)**:
   - Emit standardized `[STEP ...]` and `[EVAL ...]` lines to `.cache/live_logs/{run_key}.log` and atomically update `.cache/live_logs/{run_key}_results.json`, `.cache/model_cards/{run_key}.json` (`ModelCard v1.0`), and `.cache/benchmarks.db` (`BenchmarkSQLiteDB`) at every evaluation checkpoint (`tokens_M`, `compute_tflops`, `tau_progress`, `val_ce`, `val_ppl`, `top1_acc`, `sharp_eff_rank`, `kraus_stiefel_err`).
3. **Sync & Verify in WebUI (`scripts/sync_waves_to_webui.py`)**:
   - After or during any experiment wave, run:
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload
     ```
     so all runs appear immediately in the appropriate WebUI tab (`🏆 Full FineWeb Speedrun`, `⚡ Local FineWeb-Edu Waves 18–29`, or `🧪 Micro-Verifications & Vision Ablations`).

## 7. Implementer Context-Budget & Graceful Experiment Wrap-Up Protocol

When implementing and running an `EXPERIMENT_PLAN_0XX.md`, implementer agents and subagents MUST follow this discipline:

1. **Successful Code Implementation Does Not Require Full Run Evaluation Success at All Costs**:
   - Delivering clean, well-commented code that passes `pytest` unit tests and `scripts/ast_signature_and_length_linter.py` is a successful code implementation even if a full training evaluation does not pass every empirical target on the first try.
2. **Proactive Error Fixing vs. Context-Limit Discipline**:
   - Agents should proactively fix code/runtime errors without stopping prematurely for instructions. However, if diagnosing or tuning a training evaluation approaches the limit of the agent's context window (or exceeds `2–3` training retries), **stop retrying** rather than exhausting context.
3. **Wrap Up with a Structured Results & Post-Mortem Document**:
   - When wrapping up an experiment whose full training evaluation partially succeeded or revealed unexpected dynamics, update `EXPERIMENT_PLAN_0XX.md` (Section 6) with:
     - All constants introduced/modified during implementation (Rule 7 Constants Ledger),
     - The exact empirical results and telemetry observed, and
     - Concrete mathematical/architectural ideas and hypotheses about what happened and what the next wave should test.

