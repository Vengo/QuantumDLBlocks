# Experiment Plan 044: Standardized Experiment Progress Logging, Dedicated `FineWeb-Edu (Waves 18–29)` WebUI Tab, Interactive Box-Zoom Charts, & Single-Screen Scrollable Leaderboards

## 1. Executive Summary & User-Aligned Design Goals

Following user review of our initial WebUI plan, `EXPERIMENT_PLAN_044.md` implements four interconnected upgrades across `src/qdlblocks/benchmarks/`, `scripts/monitor_colab_runs.py`, and `scripts/`:

1. **Change 1 — Standardized Experiment Intent & Live Progress Logger (`StandardizedExperimentLogger`)**:
   - Recent waves (`Waves 18–28`) ran locally on the user's RTX 5060 Ti across distinct task domains (`FineWeb-Edu` local scaling, `CIFAR-100`, `1,000-Class`, `ILSVRC ImageNet-1K`) rather than remote Colab 900M FineWeb speedruns.
   - We introduce [`src/qdlblocks/benchmarks/experiment_logger.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/experiment_logger.py) (`ExperimentIntentSpec` & `StandardizedExperimentLogger`) and **Section 6 of [`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md#L86-L104)** so every training run emits:
     - An explicit `[EXPERIMENT_INTENT]` header block in `.cache/live_logs/{run_key}.log` recording `run_key`, `wave_id`, `plan_id`, `task_domain` (`"fineweb_speedrun"` | `"fineweb_edu_local"` | `"vision_ablation"` | `"micro_verification"`), `hypothesis_intent`, `architecture_summary`, and `hardware_budget`.
     - Standardized `[STEP ...]` and `[EVAL ...]` progress lines plus atomic live updates to `.cache/live_logs/{run_key}_results.json`, `.cache/model_cards/{run_key}.json` (`ModelCard v1.0`), and `.cache/benchmarks.db` (`BenchmarkSQLiteDB`).
2. **Change 2 — Dedicated `FineWeb-Edu (Waves 18–29)` Tab, Single-Screen Scrollable Tables (`max-height: 62vh`), & Click-and-Drag Chart Box Zoom**:
   - **Dedicated `⚡ FineWeb-Edu & Wavelet-Cholesky (Waves 18–29)` Tab**: Rather than mixing `2.46M–150M` local `FineWeb-Edu` experiments into the 900M Colab FineWeb speedrun table, `BenchmarkSQLiteDB` and `/api/snapshot` expose a dedicated `fineweb_edu_catalog` rendered on its own top-level WebUI tab (`activeTab === "fineweb_edu"`), while `CIFAR-100` / `ImageNet-1K` / `Plans 013–043` micro-verifications live on the `🧪 Micro-Verifications & Vision Ablations` tab.
   - **Single-Screen Vertical Scrollbar on All Results Tables**: `.table-container` is styled with `max-height: 62vh; overflow-y: auto; overflow-x: auto;` and sticky `<thead> th { position: sticky; top: 0; z-index: 5; }` so every leaderboard table fits comfortably within one vertical screen and its scrollbar is always immediately accessible.
   - **Interactive Click-and-Drag Rectangular Box Zoom on `SvgTrajectoryChart`**: Clicking and dragging on any trajectory chart draws a selection box (`[x1, x2] × [y1, y2]`) and zooms the chart to that region, with a `"🔍 Reset Zoom"` button (and double-click reset) in the top-right corner of the chart.
3. **Change 3 — Task-Aware `parse_log_metrics()`, Standardized X-Axis Toggle (`Tokens`, `TFLOPs`, `τ`, `Wall Time`), & Decluttered Chart Layout**:
   - **Task-Aware `parse_log_metrics()`**: Distinguishes `fineweb_speedrun` vs. `fineweb_edu_local` vs. `vision_ablation`, parses `[EXPERIMENT_INTENT]` blocks, and supports both `list`-formatted and `dict`-formatted `*_results.json` files without clobbering `best_val_ce` or `evals`.
   - **Chart Decluttering on the Main FineWeb Tab**:
     - **Removed**: `"Quantum State Capacity Law"` (`SvgRankScalingChart`) is removed completely.
     - **Moved to Bottom (`Secondary & Historical Diagnostics`)**: `"Best Val CE vs Seed Dimension D_seed"` (`SvgSeedScalingChart`) and `"Seed Quantum State Effective Rank R_eff(rho_seed)"` are moved below the main Results Leaderboard Table at the bottom of the page.
     - **Stiefel Manifold Drift vs. Modern Readout Observables**: Keeps `"Riemannian Stiefel Manifold Drift ||W^T W - I||_F / sqrt(d)"` (`iso_err` / `kraus_stiefel_err`) and pairs it on the `FineWeb-Edu` tab with `Sharp Detector Effective Rank R_sharp` and an explanation of how `StiefelMuon` landing (`kraus_stiefel_err <= 0.025`) prevents detector rank collapse (`R_sharp -> 26.8–33.6`).
4. **Change 4 — `Build-Before-Launch` & `Nice-to-Have` Operational Scripts**:
   - [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py): Normalizes and ingests all `Waves 18–29` and `Plans 013–045` logs/JSONs into `.cache/benchmarks.db` and hot-reloads `:8765`.
   - [`scripts/sqlite_benchmark_query_cli.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sqlite_benchmark_query_cli.py): Fast terminal query CLI over `.cache/benchmarks.db`.
   - [`scripts/webui_static_html_exporter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/webui_static_html_exporter.py): Self-contained static HTML dashboard exporter.
