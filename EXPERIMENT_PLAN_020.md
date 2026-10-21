# Experiment Plan 020 — SQLite Telemetry & Model Card Database, Automatic Pareto-Frontier Obsolete Lifecycle, Multi-Tab WebUI (Main Leaderboard / Micro-Verifications / Live Colab Queue), & Wave-12 900M-Token Unification Fleet

**Date**: 2026-10-19  
**Repository**: `/home/festivity/agy/QuantumDLBlocks`  
**Status**: Revised — Addressing Blind Review Clarifications  

---

## 1. Executive Project Summary: Problems Solved, Outcomes Targeted, Steps Taken (Waves 1–11), & Next Steps

### 1.1 Core Scientific & Engineering Problems We Have Been Solving
1. **Bridging the Quantum-Born vs. Classical Language Modeling Gap on Real Pretraining Scale (`900M` FineWeb-Edu Tokens)**:
   - Throughout Waves 1–11, we have developed **`QuantumBornGPT`**, a quantum-geometric causal language model where token embeddings live on a unit complex/real Hilbert sphere, sequence mixing operates via unitary Rotary Position Embeddings (`UnitaryRoPE`) and `SubspaceProjectorBornAttention` (Born-rule transition probabilities $|\langle q_t | k_s \rangle|^2$ and purified rank-2 density-matrix traces implemented in a custom Triton kernel `flash_purified_born_attention`), feed-forward blocks use overcomplete frame expansions (`2d` and `4d` two-beam bilinear interference), optimization runs on the Riemannian Stiefel manifold (`StiefelMuon` with 5-step Newton-Schulz polar retraction), and token probabilities are read out via Positive Operator-Valued Measures (`POVM`).
   - All main benchmark runs are evaluated on a strictly held-out **5,000,000-token FineWeb-Edu validation split** (`vocab_size = 50,304`, evaluated at a fixed context length `T_val = 512` so validation cross-entropy `Val CE` and perplexity `Val PPL = exp(Val CE)` are 100% comparable across every wave).
   - Alongside `QuantumBornGPT`, we maintain **`ClassicalModdedGPT`**, our reproduction of the `KellerJordan/modded-nanogpt` 12-layer (`d = 768`) speedrun architecture (`ReLU^2` MLP, U-Net encoder-to-decoder skip connections, value embeddings, parameterless RMSNorm, and `Muon + AdamW` optimization).
2. **Closing the Classical Speedrun Gap (`3.4280` in Wave 9 $\to$ `3.3688` in Wave 11 toward `3.280` Val CE)**:
   - In Wave 9 (Plan 016), our `ClassicalModdedGPT` reached `3.4280` Val CE (`30.8` PPL). In Plan 019 (Wave 11), we audited the exact `KellerJordan/modded-nanogpt` record (`records/track_1_short/2025-01-16_Sub3Min.txt`) and implemented `--leaderboard-exact-classical` (fixing 9 subtle discrepancies: `H=6, head_dim=128` with `attn_scale=0.12`, `LeaderboardRotary` with `base=1024`, unit-gain `ema_momentum=True` in `StiefelMuon` with `lr=0.04..0.05` and `weight_decay=0.0`, untied `AdamW` with `lr_wte=0.6, lr_scalar=0.04, lr_head=0.008, betas=(0.8, 0.95)`, pre-block scalar U-Net skip connections, `skip_attn_layers=(7,)`, `logit_softcap=15.0`, and progressive long/short sliding-window schedules `128 -> 1792` with `<|endoftext|>` document masking).
   - **Wave-11 Results (All 5 `900M`-Token Runs Completed 100%)**:
     - **`w11_900m_classical_leaderboard_exact_b262k_3429st` (`gpu_g4_w6_2`)**: **`3.3688` Val CE (`29.0` PPL, `38.40%` Top-1, `60.47%` Top-5, `157,506 tok/s`, `95.36 min`)** — **New #1 All-Time Classical Record** (`-0.0592` nats better than Wave 9's `3.4280` at the exact same `3,429` steps, and `15.2%` faster!).
     - **`w11_900m_classical_leaderboard_exact_progwin_docmask` (`gpu_a100_w6_1`)**: **`3.3755` Val CE (`29.2` PPL, `38.38%` Top-1, `117,249 tok/s` on A100, `127.75 min`)** — **New #2 All-Time Classical Record** (`1,710` steps at $B_{\text{tok}}=524\text{k}$ with `--progressive-window --doc-mask-attn`).
     - **`w11_900m_classical_leaderboard_exact_b524k_1716st` (`gpu_g4_w6_1`)**: **`3.3779` Val CE (`29.3` PPL, `38.30%` Top-1, `157,862 tok/s`, `94.89 min`)** — **New #3 All-Time Classical Record** (`-0.0721` nats better than Wave 9's `524k` run at `3.4500`).
     - **`w11_900m_qborn_speedrun_v4_L8_leaderboard_hybrids` (`gpu_a100_w6_2`)**: **`3.6787` Val CE (`39.6` PPL, `35.38%` Top-1, `56.89%` Top-5, `IsoErr = 1.0297`, `81,270 tok/s`, `184.89 min`)** — **New #1 All-Time QuantumBornGPT Record** (first Quantum model to break `< 3.68` Val CE and `< 40.0` Val PPL, `-0.0139` nats better than Wave 10's `3.6926`).
     - **`w11_900m_qborn_speedrun_v4_deep_L12_asym1024_frame4d` (`gpu_g4_w6_3`)**: **`3.7005` Val CE (`40.5` PPL, `35.13%` Top-1, `56.55%` Top-5, `IsoErr = 0.9626`, `85,328 tok/s`, `176.04 min`)** — **New #3 All-Time QuantumBornGPT Record** (improving deep `L=12` QuantumBornGPT from `3.7077` in Wave 10 to `3.7005`).
3. **Diagnosing & Fixing the Dashboard Process-Detection Bug (`w8_...` Showing `COMPILING` & 5 Idle Colab Sessions Showing Active)**:
   - The user noticed that even after Wave 11 finished, the WebUI showed 5 running Colab sessions and `w8_900m_qborn_earlyfix_flash_asym1024_L8_layer0v` as `"COMPILING"`.
   - We traced this to two root causes and fixed both:
     1. **Fallback to Superseded Wave in `select_session_poll_targets`**: When the Wave-11 run on `gpu_a100_w6_1` completed (`"COMPLETED"` in its local log), `DynamicWaveRegistry.select_session_poll_targets` skipped the completed W11 run and fell back to the older incomplete W8 run (`w8_900m_qborn_earlyfix_flash_asym1024_L8_layer0v`) on that same session.
     2. **`pgrep -af` Self-Matching in Remote Colab Probe**: Inside `fetch_colab_fleet_member`, the remote probe executed `subprocess.getoutput("pgrep -af '11_fineweb|run_hyp|run_w'")`, which matched its own `/bin/sh -c` parent shell command string, causing `ps_alive` to return `True` even when all 5 Colab GPUs were completely idle (`0%` SM util, `0.0 GiB` VRAM used)! We fixed `select_session_poll_targets` to never fall back to superseded waves and updated the remote probe to use bracketed regex `[1]1_fineweb|[r]un_hyp|[r]un_w` plus explicit PID/parent-PID filtering.

---

## 2. Long-Term Architectural Requirements (Plan 020)

Based on our design alignment with the user, Plan 020 upgrades the QuantumDLBlocks benchmarking platform and launches Wave 12 across four components:

1. **Component 1 — SQLite Telemetry, Model Card, & Session Queue Database (`src/qdlblocks/benchmarks/db.py` & `.cache/benchmarks.db`)**:
   - Store all data displayed in the WebUI in a **WAL-mode SQLite database** (`/home/festivity/agy/QuantumDLBlocks/.cache/benchmarks.db` by default, configurable via `db_path`).
   - The WebUI (`GET /api/snapshot`, `GET /api/model-cards`, `GET /api/micro-verifications`, `GET /api/colab-sessions`, `GET /api/queue`, and initial SSR HTML) reads and displays its data from this SQLite database.
   - Support **hybrid dual ingestion**:
     - **Homelab Polling Mode (Active Now)**: The background daemon (`monitor_colab_runs.py --serve --interval 20`) polls the 5 remote Colab sessions every 20 seconds, parses log/JSON/hardware telemetry, and upserts all runs, evaluation checkpoints, step metrics, and Colab session statuses directly into SQLite.
     - **Cloud Push API Mode (Ready for GCP Hosting)**: Remote Colab runners (or local scripts) can push live status updates directly into the SQLite database via `POST /api/ingest` (run logs, metrics, and Model Cards) and `POST /api/colab-heartbeat` (session GPU telemetry and active PID command), and manage queued jobs via `POST /api/queue`.
2. **Component 2 — Automatic Pareto-Frontier Obsolete Lifecycle by Model Family**:
   - Within each model family (`arch_family` $\in \{\text{"classical\_modded"}, \text{"quantum\_born"}\}$) across completed `main_benchmark` runs (`run_tier == "main_benchmark"`), we define two distinct, complementary concepts:
     1. **Global Family 2D Pareto Frontier (`pareto_frontier = True`)**: Evaluated across **all** completed `main_benchmark` runs in the family (regardless of wave). A completed run $A$ has `pareto_frontier = True` iff no completed run $C \ne A$ of the same `arch_family` has $\text{best\_val\_ce}(C) \le \text{best\_val\_ce}(A)$ and $\text{wall\_min}(C) \le \text{wall\_min}(A)$ with at least one strict inequality ($<$).
     2. **Wave-Superseded Automatic Obsolete Marking (`auto_pareto_dominated = True`)**: Evaluated strictly across **newer waves** ($\text{wave\_num}(B) > \text{wave\_num}(A)$) so intra-wave ablations do not auto-obsolete each other. A completed run $A$ is superseded by a newer wave iff there exists a completed run $B$ of the **same `arch_family`** in a **strictly newer wave** ($\text{wave\_num}(B) > \text{wave\_num}(A)$) that achieves **strictly lower validation loss (`best_val_ce`) in less or equal wall-clock time (`wall_min`)**:
        $$\text{WaveSuperseded}(A) \iff \exists B \in \mathcal{C}_{\text{family}(A)}:\; \text{wave\_num}(B) > \text{wave\_num}(A) \;\land\; \text{best\_val\_ce}(B) < \text{best\_val\_ce}(A) \;\land\; \text{wall\_min}(B) \le \text{wall\_min}(A)$$
   - Any run $A$ satisfying $\text{WaveSuperseded}(A)$ is automatically marked `obsolete = True` (unless the user has explicitly overridden it via `explicit_user_override = 1` in SQLite / `POST /api/obsolete`), with the canonical `obsolete_reason` string:
     ```
     "Not on {fam_disp} Pareto frontier: superseded by {B_star.key} ({B_star.wave_id}) with lower Val CE ({B_star.best_val_ce:.4f} vs {A.best_val_ce:.4f}) in less time ({B_star.wall_min:.1f}m vs {A.wall_min:.1f}m)"
     ```
     where `fam_disp = "ClassicalModdedGPT" if arch_family == "classical_modded" else "QuantumBornGPT"` and $B^*$ is the dominating run minimizing `(best_val_ce, wall_min, key)`.
3. **Component 3 — Multi-Tab WebUI: (1) Main FineWeb-Edu Leaderboard, (2) Pre-Launch Micro-Verifications Tab, & (3) Live Colab Fleet & Job Queue Inspector**:
   - **Tab 1 (`🏆 Main FineWeb-Edu Leaderboard & Scaling Explorer`)**: Displays `run_tier == "main_benchmark"` runs (`100M–900M` FineWeb-Edu runs) from SQLite, with `⚡ PARETO FRONTIER` badges, automatic Pareto-obsolete filtering, interactive SVG trajectories, and Model Cards.
   - **Tab 2 (`🧪 Pre-Launch Micro-Verifications & Hypothesis Tests`)**: Displays `run_tier == "micro_verification"` runs stored in SQLite — including the 60-step controlled runs the user asked about (`verify019_60step_leaderboard_exact` reaching **`2.9254` Val CE** vs `verify019_60step_wave9_v2` at **`3.2974` Val CE** from `examples/29_verify_plan019_leaderboard_exact_classical.py`) as well as all historical pre-launch hypothesis verification runs from Plans 013–019 (`.cache/hypothesis_experiments_013.json` through `019.json`). Includes its own sortable table, trajectory comparison chart, and clickable Model Cards!
   - **Tab 3 & Persistent Header Banner (`🖥️ Live Colab Fleet & Session Job Queue`)**: Displays real-time GPU hardware & process status for all 5 Colab sessions (`gpu_g4_w6_1..3`, `gpu_a100_w6_1..2`) plus the SQLite `session_queue` table showing what is currently running (`🟢 PROCESS RUNNING` vs `⚪ IDLE`), what finished most recently, and what jobs are queued next per session.
4. **Component 4 — Wave-12 `900M`-Token Unification Fleet (`configs/wave12_fleet.json`)**:
   - Synthesizes our Wave-11 discoveries into 5 next-generation `900M`-token runs (3 Classical Leaderboard-Exact scaling runs targeting `< 3.34` Val CE and 2 QuantumBornGPT Speedrun V5 runs combining `L=12` + `4d-Frame` with `H=8, head_dim=128` + `muon_lr=0.035` + `wsd 0.40->0.10`), registered into SQLite and `session_queue`.

---

## 3. Observable Hypotheses & Quantitative Pass/Fail Criteria

- **Hypothesis H1 (SQLite WAL Database Completeness, Dual Poll/Push Ingestion, & Micro-Verification Backfill)**:
  - *Prediction*: Migrating and syncing all existing runs, `.cache/model_cards/*.json`, `.cache/live_logs/*`, and `.cache/hypothesis_experiments_013..019.json` into `/home/festivity/agy/QuantumDLBlocks/.cache/benchmarks.db` via `BenchmarkSQLiteDB` will populate:
    1. All $\ge 55$ `main_benchmark` runs (`run_tier = "main_benchmark"`) with `0` duplicate `run_key` rows and 100% matching `best_val_ce`, `evals`, and `steps`.
    2. At least $\ge 15$ `micro_verification` runs (`run_tier = "micro_verification"`), explicitly including `verify019_60step_leaderboard_exact` (`best_val_ce = 2.9254`) and `verify019_60step_wave9_v2` (`best_val_ce = 3.2974`) with their 5-point step `0..60` evaluation curves.
    3. All 5 Colab sessions (`gpu_g4_w6_1..3`, `gpu_a100_w6_1..2`) in `colab_sessions` and the 5 Wave-12 jobs in `session_queue`.
    4. Pushing a live update via `POST /api/ingest` or `POST /api/colab-heartbeat` persists immediately to SQLite and is reflected in `GET /api/snapshot` in `< 25 ms` with `0` server restarts.
- **Hypothesis H2 (Automatic Pareto-Frontier Obsolete Marking by Model Family)**:
  - *Prediction*: Running `BenchmarkSQLiteDB.recompute_pareto_and_obsolete()` over all completed `main_benchmark` runs independently for `classical_modded` and `quantum_born` will:
    1. Identify the exact non-dominated `(wall_min, best_val_ce)` Pareto frontier for `classical_modded` (including `w11_900m_classical_leaderboard_exact_b262k_3429st` at `3.3688` CE in `95.36m` and `w11_900m_classical_leaderboard_exact_b524k_1716st` at `3.3779` CE in `94.89m`) and for `quantum_born` (including `w11_900m_qborn_speedrun_v4_L8_leaderboard_hybrids` at `3.6787` CE in `184.89m` and `w11_900m_qborn_speedrun_v4_deep_L12_asym1024_frame4d` at `3.7005` CE in `176.04m`).
    2. Automatically mark older-wave runs that achieved strictly higher `best_val_ce` in equal or greater `wall_min` within the same `arch_family` as `obsolete = True` with the exact canonical string `"Not on {fam_disp} Pareto frontier: superseded by {B_star.key} ({B_star.wave_id}) with lower Val CE ({B_star.best_val_ce:.4f} vs {A.best_val_ce:.4f}) in less time ({B_star.wall_min:.1f}m vs {A.wall_min:.1f}m)"` (for example, `w9_900m_classical_speedrun_v2_t1024_b262k` at `3.4280` CE in `110.22m` is superseded by `w11_900m_classical_leaderboard_exact_b262k_3429st` at `3.3688` CE in `95.36m`).
    3. Preserve any manual user override (`explicit_user_override = 1`) set via `POST /api/obsolete` so a user-restored or user-obsoleted run is never overwritten by automatic Pareto recomputation.
- **Hypothesis H3 (Wave-12 Quantum V5 Head-Dimension & EMA-Muon Pre-Launch Verification)**:
  - *Prediction*: In a controlled 40-step pre-launch micro-verification (`examples/30_verify_plan020_sqlite_pareto_and_wave12.py`) on a deterministic synthetic Zipfian+Markov token stream (`vocab_size = 4096`, `batch_size = 16`, `seq_len = 128`, evaluated every `10` steps on a held-out 16-batch validation split), we compare two matched `QuantumBornGPT` (`num_layers = 6`, `embed_dim = 256`, `seed_dim = 256`, `readout_mode = "asymmetric"`, `overcomplete_two_beam = True`, `frame_expansion = 4.0`, `use_purified_attn = True`, `use_dynamic_povm_beta = True`, `use_ortho_decouple = True`, `use_povm_log_prior = True`, `use_gated_spherical_residual = True`, `true_layer0_v_residual = True`, `unet_skip_mode = "scalar"`, `use_fused_qkv = True`, `num_value_embeddings = 2`, `zero_half_rope = True`) models initialized from the exact same random seed (`seed = 1337`):
    - **Control Arm (`verify020_40step_qborn_v4_H8_d32_lr025`)**: `num_heads = 8` (`head_dim = 32`), `StiefelMuon(lr = 0.025, ema_momentum = False)`.
    - **Treatment Arm (`verify020_40step_qborn_v5_H4_d64_lr035`)**: `num_heads = 4` (`head_dim = 64`, doubling per-head Hilbert subspace dimension just as `H=8, head_dim=128` doubles `H=16, head_dim=64` at `d=1024`), `StiefelMuon(lr = 0.035, ema_momentum = True, weight_decay = 0.0)`.
    - Both runs will be ingested into SQLite with `run_tier = "micro_verification"`, and `verify020_40step_qborn_v5_H4_d64_lr035` will achieve a final validation cross-entropy at step 40 at least `>= 0.08` nats lower than `verify020_40step_qborn_v4_H8_d32_lr025`.

---

## 4. Detailed Technical Specification

### 4.1 Operational Scripting Classification (`Build-Before-Launch` vs. `Nice-to-Have`)
- **`Build-Before-Launch`**:
  - `src/qdlblocks/benchmarks/db.py`: SQLite WAL database engine (`BenchmarkSQLiteDB`), schema creation, Pareto-frontier & automatic obsolete computation, micro-verification backfill, and snapshot materialization.
  - `scripts/monitor_colab_runs.py` & `scripts/colab_fleet.py`: Wiring `BenchmarkSQLiteDB` into `collect_snapshot`, `create_benchmark_http_server` (`GET /api/snapshot`, `GET /api/micro-verifications`, `GET /api/queue`, `POST /api/ingest`, `POST /api/colab-heartbeat`, `POST /api/queue`, `POST /api/obsolete`), and the 3-Tab React WebUI.
- **`Nice-to-Have` (Logged to `/home/festivity/agy/QuantumDLBlocks/scripts/BACKLOG.md`, Not Built Before Launch)**:
  - Cloud SQL / PostgreSQL replica adapter for multi-region GCP Cloud Run deployments (`--postgres-url`).

### 4.2 Component 1: SQLite Database Schema & Engine (`src/qdlblocks/benchmarks/db.py`)

We create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/db.py` implementing `BenchmarkSQLiteDB` and export it in `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/__init__.py`:

1. **Database File & Concurrency**:
   - Default path: `DEFAULT_DB_PATH = REPO_ROOT / ".cache" / "benchmarks.db"` (and when a custom `live_logs_dir` is passed in tests without an explicit `db_path`, defaults to `live_logs_dir / "benchmarks.db"` so unit tests are 100% isolated).
   - Connections use `sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)` with:
     - `PRAGMA journal_mode=WAL;`
     - `PRAGMA synchronous=NORMAL;`
     - `PRAGMA foreign_keys=ON;`
2. **Relational Tables**:
   - **`runs`**:
     ```sql
     CREATE TABLE IF NOT EXISTS runs (
         run_key TEXT PRIMARY KEY,
         wave_id TEXT NOT NULL,
         wave_num REAL NOT NULL DEFAULT 0.0,
         wave_label TEXT NOT NULL DEFAULT '',
         arch_family TEXT NOT NULL DEFAULT 'quantum_born',
         run_tier TEXT NOT NULL DEFAULT 'main_benchmark', -- 'main_benchmark' | 'micro_verification'
         session TEXT NOT NULL DEFAULT '',
         accel TEXT NOT NULL DEFAULT '',
         kind TEXT NOT NULL DEFAULT 'gpu',
         label TEXT NOT NULL DEFAULT '',
         short_label TEXT NOT NULL DEFAULT '',
         color TEXT NOT NULL DEFAULT '#58a6ff',
         status TEXT NOT NULL DEFAULT 'IDLE',
         boot_status TEXT NOT NULL DEFAULT 'Idle',
         params_M REAL NOT NULL DEFAULT 0.0,
         seed_dim INTEGER NOT NULL DEFAULT 128,
         embed_dim INTEGER NOT NULL DEFAULT 1024,
         num_layers INTEGER NOT NULL DEFAULT 6,
         d_ff TEXT NOT NULL DEFAULT '',
         readout_type TEXT NOT NULL DEFAULT '',
         optimizer_mode TEXT NOT NULL DEFAULT 'muon_adamw',
         schedule TEXT NOT NULL DEFAULT 'COSINE',
         target_tokens_M REAL NOT NULL DEFAULT 900.0,
         live_tokens_M REAL NOT NULL DEFAULT 0.0,
         last_eval_tokens_M REAL NOT NULL DEFAULT 0.0,
         current_step INTEGER NOT NULL DEFAULT 0,
         total_steps INTEGER NOT NULL DEFAULT 0,
         progress_pct REAL NOT NULL DEFAULT 0.0,
         live_train_ce REAL,
         ep05_ce REAL,
         ep10_ce REAL,
         ep20_ce REAL,
         ep30_ce REAL,
         best_val_ce REAL,
         best_val_ppl REAL,
         best_top1 REAL,
         best_top5 REAL,
         final_r_eff REAL,
         final_iso_err REAL,
         avg_tok_per_s REAL NOT NULL DEFAULT 0.0,
         wall_min REAL NOT NULL DEFAULT 0.0,
         eta_min REAL NOT NULL DEFAULT 0.0,
         pareto_frontier INTEGER NOT NULL DEFAULT 0,
         obsolete INTEGER NOT NULL DEFAULT 0,
         obsolete_reason TEXT NOT NULL DEFAULT '',
         obsolete_at TEXT NOT NULL DEFAULT '',
         explicit_user_override INTEGER NOT NULL DEFAULT 0,
         default_visible INTEGER NOT NULL DEFAULT 1,
         latest_log_line TEXT NOT NULL DEFAULT '',
         catalog_entry_json TEXT NOT NULL DEFAULT '{}',
         model_card_json TEXT NOT NULL DEFAULT '{}',
         updated_at REAL NOT NULL DEFAULT 0.0
     );
     ```
   - **`run_evals`**:
     ```sql
     CREATE TABLE IF NOT EXISTS run_evals (
         run_key TEXT NOT NULL,
         eval_idx INTEGER NOT NULL,
         step INTEGER NOT NULL DEFAULT 0,
         epoch REAL NOT NULL DEFAULT 0.0,
         tokens_M REAL NOT NULL DEFAULT 0.0,
         elapsed_min REAL NOT NULL DEFAULT 0.0,
         val_ce REAL NOT NULL,
         val_ppl REAL NOT NULL,
         top1_acc REAL NOT NULL DEFAULT 0.0,
         top5_acc REAL NOT NULL DEFAULT 0.0,
         seed_eff_rank REAL,
         iso_err REAL NOT NULL DEFAULT 0.0,
         PRIMARY KEY (run_key, eval_idx),
         FOREIGN KEY (run_key) REFERENCES runs(run_key) ON DELETE CASCADE
     );
     ```
   - **`run_steps`**:
     ```sql
     CREATE TABLE IF NOT EXISTS run_steps (
         run_key TEXT NOT NULL,
         step INTEGER NOT NULL,
         tokens_M REAL NOT NULL DEFAULT 0.0,
         elapsed_min REAL NOT NULL DEFAULT 0.0,
         train_ce REAL NOT NULL,
         tok_per_s REAL NOT NULL DEFAULT 0.0,
         PRIMARY KEY (run_key, step),
         FOREIGN KEY (run_key) REFERENCES runs(run_key) ON DELETE CASCADE
     );
     ```
   - **`colab_sessions`**:
     ```sql
     CREATE TABLE IF NOT EXISTS colab_sessions (
         session TEXT PRIMARY KEY,
         device TEXT NOT NULL DEFAULT '',
         gpu_util_pct REAL NOT NULL DEFAULT 0.0,
         mem_bw_util_pct REAL NOT NULL DEFAULT 0.0,
         vram_used_gib REAL NOT NULL DEFAULT 0.0,
         vram_total_gib REAL NOT NULL DEFAULT 0.0,
         power_w REAL NOT NULL DEFAULT 0.0,
         temp_c REAL NOT NULL DEFAULT 0.0,
         ps_alive INTEGER NOT NULL DEFAULT 0,
         ps_cmd TEXT NOT NULL DEFAULT '',
         active_run_key TEXT NOT NULL DEFAULT '',
         wave_id TEXT NOT NULL DEFAULT '',
         run_state TEXT NOT NULL DEFAULT 'IDLE',
         session_json TEXT NOT NULL DEFAULT '{}',
         updated_at REAL NOT NULL DEFAULT 0.0
     );
     ```
   - **`session_queue`**:
     ```sql
     CREATE TABLE IF NOT EXISTS session_queue (
         queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
         session TEXT NOT NULL,
         run_key TEXT NOT NULL UNIQUE,
         wave_id TEXT NOT NULL DEFAULT '',
         arch_family TEXT NOT NULL DEFAULT 'quantum_born',
         hardware TEXT NOT NULL DEFAULT '',
         short_label TEXT NOT NULL DEFAULT '',
         cli_args TEXT NOT NULL DEFAULT '',
         status TEXT NOT NULL DEFAULT 'QUEUED', -- 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'CANCELLED'
         priority INTEGER NOT NULL DEFAULT 100,
         created_at TEXT NOT NULL DEFAULT '',
         updated_at TEXT NOT NULL DEFAULT ''
     );
     ```
   - **`ui_state`**:
     ```sql
     CREATE TABLE IF NOT EXISTS ui_state (
         state_key TEXT PRIMARY KEY,
         state_json TEXT NOT NULL DEFAULT '{}'
     );
     ```

3. **Run Tier Classification (`main_benchmark` vs. `micro_verification`)**:
   - Function `classify_run_tier(run_key: str, target_tokens_M: float, wave_id: str = "", model_card: dict[str, Any] | None = None) -> str`:
     - Returns `"micro_verification"` if:
       - `(model_card or {}).get("run_tier") == "micro_verification"`, OR
       - `run_key.lower().startswith(("verify", "micro_", "plan0", "h1_", "h2_", "h3_"))`, OR
       - `wave_id.strip().upper() in ("MICRO", "VERIFY")`, OR
       - `0.0 < float(target_tokens_M) < 50.0` (since all main FineWeb-Edu benchmarks are $\ge 100\text{M}$ tokens, whereas pre-launch micro-verifications are `0.1M–25M` tokens).
     - Otherwise returns `"main_benchmark"`.

4. **Public Method Signatures on `BenchmarkSQLiteDB` & Queue Lifecycle Rules**:
   - `BenchmarkSQLiteDB.__init__(self, db_path: Path = DEFAULT_DB_PATH) -> None`: Initializes the SQLite connection lock, creates parent directories, applies WAL PRAGMAs, and executes schema DDL.
   - `BenchmarkSQLiteDB.upsert_run(self, cat_entry: dict[str, Any], model_card: dict[str, Any] | None = None, explicit_user_override: bool | None = None) -> None`: Upserts a single run into `runs`, replaces its `run_evals` and `run_steps` rows inside a transaction, and if `run_key` exists in `session_queue` with `status != 'CANCELLED'`, automatically transitions `session_queue.status` to:
     - `'COMPLETED'` when `cat_entry.get("run_state") == "COMPLETED"` (or `cat_entry.get("completed") is True`),
     - `'RUNNING'` when `cat_entry.get("run_state") in ("RUNNING", "COMPILING")`,
     - leaving `'QUEUED'` unchanged when `cat_entry.get("run_state") in ("IDLE", "")`.
   - `BenchmarkSQLiteDB.upsert_colab_session(self, session_obj: dict[str, Any]) -> None`: Upserts a Colab session status row into `colab_sessions`, and if `session_obj.get("ps_alive")` is `True` and `session_obj.get("active_run_key")` matches a `'QUEUED'` row in `session_queue`, transitions that row's `status` to `'RUNNING'`.
   - `BenchmarkSQLiteDB.enqueue_job(self, session: str, run_key: str, wave_id: str, arch_family: str = "quantum_born", hardware: str = "", short_label: str = "", cli_args: str = "", status: str = "QUEUED", priority: int = 100) -> dict[str, Any]`: Upserts a job into `session_queue` (if a run with `run_key` is already `COMPLETED` in `runs`, sets `status = 'COMPLETED'`; if `RUNNING`/`COMPILING`, sets `status = 'RUNNING'`).
   - `BenchmarkSQLiteDB.sync_queue_manifest(self, manifest_path: Path) -> list[dict[str, Any]]`: Reads a `configs/wave*_fleet.json` manifest and calls `enqueue_job` for each run entry in `manifest["runs"]`.
   - `BenchmarkSQLiteDB.set_obsolete_override(self, run_keys: list[str], obsolete: bool, reason: str = "") -> dict[str, Any]`: Sets `obsolete = int(obsolete)`, `obsolete_reason = reason`, and `explicit_user_override = 1` on each specified `run_key` in `runs` (and updates `catalog_entry_json` and `model_card_json`), then calls `recompute_pareto_and_obsolete()`.
   - `BenchmarkSQLiteDB.recompute_pareto_and_obsolete(self, default_obsolete_reasons: dict[str, str] | None = None) -> None`: Loads all `main_benchmark` runs from `runs`, computes `pareto_frontier` and automatic wave-superseded `obsolete` via `compute_family_pareto_and_obsolete`, and writes updated flags back to `runs`.
   - `BenchmarkSQLiteDB.sync_snapshot(self, catalog: list[dict[str, Any]], colab_sessions: list[dict[str, Any]], ui_prefs: dict[str, Any], default_obsolete_reasons: dict[str, str] | None = None) -> dict[str, Any]`: Batch-upserts `catalog`, `colab_sessions`, and `ui_prefs` in a single WAL transaction, runs `recompute_pareto_and_obsolete`, and returns the materialized snapshot dictionary (`runs_catalog`, `micro_runs_catalog`, `colab_sessions`, `session_queue`, `db_stats`).
   - `BenchmarkSQLiteDB.load_snapshot_payload(self) -> dict[str, Any]`: Reads `runs` (split into `runs_catalog` for `run_tier == 'main_benchmark'` and `micro_runs_catalog` for `run_tier == 'micro_verification'`), `colab_sessions`, `session_queue`, `ui_state`, and `db_stats` directly from SQLite.

### 4.3 Component 2: Automatic Pareto-Frontier & Obsolete Recomputation (`compute_family_pareto_and_obsolete`)

Inside `src/qdlblocks/benchmarks/db.py`, we provide `compute_family_pareto_and_obsolete(catalog: list[dict[str, Any]], explicit_overrides: set[str], default_obsolete_reasons: dict[str, str]) -> list[dict[str, Any]]`:

1. **Scope & Completion Predicate**:
   - Applied to `run_tier == "main_benchmark"` runs. (`micro_verification` runs are never auto-obsoleted by main-benchmark runs).
   - A run $r$ is considered **eligible/completed** for Pareto comparison iff:
     ```python
     is_completed = (
         (str(r.get("run_state") or r.get("status") or "").upper() == "COMPLETED" or bool(r.get("completed")))
         and r.get("best_val_ce") is not None
         and float(r.get("wall_min") or 0.0) > 0.0
     )
     ```
2. **Global Family 2D Pareto Frontier (`pareto_frontier = True`)**:
   - Partition all eligible completed `main_benchmark` runs by `arch_family` (`"classical_modded"` vs. `"quantum_born"`).
   - Within family $F$, a completed run $A \in F$ has `A["pareto_frontier"] = True` iff there is **no** completed run $C \in F$ ($C \ne A$) such that:
     $$\text{best\_val\_ce}(C) \le \text{best\_val\_ce}(A) \quad\land\quad \text{wall\_min}(C) \le \text{wall\_min}(A) \quad\land\quad \big(\text{best\_val\_ce}(C) < \text{best\_val\_ce}(A) \;\lor\; \text{wall\_min}(C) < \text{wall\_min}(A)\big)$$
   - Non-completed runs have `pareto_frontier = False`.
3. **Wave-Superseded Automatic Obsolete Marking (`WaveSuperseded(A)`)**:
   - Within family $F$, a completed run $A \in F$ is **superseded by a newer wave** iff the set of dominating newer-wave runs:
     $$\mathcal{D}(A) = \Big\{\, B \in F \;\Big|\; \text{wave\_num}(B) > \text{wave\_num}(A) \;\land\; \text{best\_val\_ce}(B) < \text{best\_val\_ce}(A) \;\land\; \text{wall\_min}(B) \le \text{wall\_min}(A) \,\Big\}$$
     is non-empty ($\mathcal{D}(A) \ne \emptyset$).
   - When $\mathcal{D}(A) \ne \emptyset$, select the canonical dominating run $B^* = \arg\min_{B \in \mathcal{D}(A)} \big(\text{ float}(B[\text{"best\_val\_ce"}]),\; \text{float}(B[\text{"wall\_min"}]),\; \text{str}(B[\text{"key"}]) \big)$ and format the exact canonical `obsolete_reason`:
     ```python
     fam_disp = "ClassicalModdedGPT" if arch_family == "classical_modded" else "QuantumBornGPT"
     auto_reason = (
         f"Not on {fam_disp} Pareto frontier: superseded by {b_star['key']} ({b_star.get('wave_id', '')}) "
         f"with lower Val CE ({float(b_star['best_val_ce']):.4f} vs {float(a['best_val_ce']):.4f}) "
         f"in less time ({float(b_star['wall_min']):.1f}m vs {float(a['wall_min']):.1f}m)"
     )
     ```
   - **4-Level Precedence Order for `obsolete`**:
     1. If `run_key in explicit_overrides` (or `r.get("explicit_user_override")`): preserve `bool(r.get("obsolete"))` and `str(r.get("obsolete_reason") or "")`.
     2. Else if $\mathcal{D}(A) \ne \emptyset$: set `obsolete = True`, `obsolete_reason = auto_reason`.
     3. Else if `run_key in default_obsolete_reasons` or `run_state == "OOM_CRASH"`: set `obsolete = True`, `obsolete_reason = default_obsolete_reasons.get(run_key, "OOM crash")`.
     4. Else: `obsolete = False`, `obsolete_reason = ""`.

### 4.4 Exact JSON Extraction Mapping for `backfill_micro_verifications` Across Plans 013–019

In `src/qdlblocks/benchmarks/db.py`, `backfill_micro_verifications(db: BenchmarkSQLiteDB, cache_dir: Path = REPO_ROOT / ".cache", model_cards_dir: Path = DEFAULT_MODEL_CARDS_DIR) -> list[dict[str, Any]]` idempotently extracts and upserts `run_tier = "micro_verification"` runs from `.cache/hypothesis_experiments_013..019.json` (gracefully skipping any file that does not exist in isolated test directories):

1. **From `.cache/hypothesis_experiments_019.json` (`EXPERIMENT_PLAN_019`)**:
   - Path `data["H2_leaderboard_exact_parity_and_convergence"]["leaderboard_exact_60step"]`:
     - `run_key = "verify019_60step_leaderboard_exact"`, `arch_family = "classical_modded"`, `wave_id = "VERIFY"`, `wave_num = 19.0`, `target_tokens_M = 1.97` (`60` steps $\times 32,768$ tok/step), `best_val_ce = round(float(arm["val_final"]), 4)` (`2.9254`). If `.cache/live_logs/verify019_60step_leaderboard_exact.log` exists, parses its 5-point step `0, 15, 30, 45, 60` trajectory; otherwise constructs 2-point eval trajectory at `step=0` (`val_0 = 8.3125`) and `step=60` (`val_final = 2.9254`).
   - Path `data["H2_leaderboard_exact_parity_and_convergence"]["wave9_v2_60step"]`:
     - `run_key = "verify019_60step_wave9_v2"`, `arch_family = "classical_modded"`, `wave_id = "VERIFY"`, `wave_num = 19.0`, `target_tokens_M = 1.97`, `best_val_ce = round(float(arm["val_final"]), 4)` (`3.2974`).
2. **From `.cache/hypothesis_experiments_016.json` (`EXPERIMENT_PLAN_016`)**:
   - Path `data["H2_speedrun_architecture"]`:
     - `verify016_160step_classical_speedrun` $\leftarrow$ `arms["classical_plan016_speedrun"]` (`arch_family = "classical_modded"`, 3 evals at steps `0, 40, 160` from `val_ce_step0`, `val_ce_step40`, `val_ce_step160 = 0.8422`).
     - `verify016_160step_classical_baseline` $\leftarrow$ `arms["classical_plan015_baseline"]` (`arch_family = "classical_modded"`, `val_ce_step160 = 1.1175`).
     - `verify016_160step_qborn_speedrun` $\leftarrow$ `arms["quantum_plan016_speedrun"]` (`arch_family = "quantum_born"`, `val_ce_step160 = 0.8620`).
     - `verify016_160step_qborn_baseline` $\leftarrow$ `arms["quantum_plan015_baseline"]` (`arch_family = "quantum_born"`, `val_ce_step160 = 0.7677`).
   - Path `data["H3_optimizer_ablation"]`:
     - `verify016_160step_classical_pure_adamw` $\leftarrow$ `H3["classical"]["pure_adamw"]` (`val_ce_step160 = 1.1141`).
     - `verify016_160step_classical_sgd_nesterov` $\leftarrow$ `H3["classical"]["sgd_nesterov"]` (`val_ce_step160 = 1.3710`).
     - `verify016_160step_qborn_pure_adamw` $\leftarrow$ `H3["quantum"]["pure_adamw"]` (`val_ce_step160 = 0.8869`).
     - `verify016_160step_qborn_sgd_nesterov` $\leftarrow$ `H3["quantum"]["sgd_nesterov"]` (`val_ce_step160 = 0.8899`).
3. **From `.cache/hypothesis_experiments_015.json` (`EXPERIMENT_PLAN_015`)**:
   - Path `data["H1_true_cross_layer_v0"]["multihop_120step_training"]`:
     - `verify015_120step_qborn_cross_layer_v0` (`arch_family = "quantum_born"`, `best_val_ce = 4.0861`).
     - `verify015_120step_qborn_same_layer_v` (`arch_family = "quantum_born"`, `best_val_ce = 4.7660`).
     - `verify015_120step_classical_cross_layer_v0` (`arch_family = "classical_modded"`, `best_val_ce = 5.5976`).
     - `verify015_120step_classical_same_layer_v` (`arch_family = "classical_modded"`, `best_val_ce = 5.7771`).
4. **From `.cache/hypothesis_experiments_014.json` (`EXPERIMENT_PLAN_014`)**:
   - Path `data["H1_early_epoch_quantum"]["arms"]`:
     - `verify014_200step_qborn_earlyfix` $\leftarrow$ `arms["earlyfix_quantum"]` (`arch_family = "quantum_born"`, 5-point `history` at steps `0, 50, 100, 150, 200`, `best_val_ce = 6.4141`).
     - `verify014_200step_qborn_baseline` $\leftarrow$ `arms["baseline_quantum"]` (`arch_family = "quantum_born"`, 5-point `history`, `best_val_ce = 6.6410`).
   - Path `data["H3_classical_speedrun"]["arms"]`:
     - `verify014_200step_classical_speedrun` $\leftarrow$ `arms["speedrun_classical_upgraded"]` (`arch_family = "classical_modded"`, 5-point `history`, `best_val_ce = 6.3930`).
     - `verify014_200step_classical_baseline` $\leftarrow$ `arms["wave6_classical_baseline"]` (`arch_family = "classical_modded"`, 5-point `history`, `best_val_ce = 6.5247`).
5. **From `.cache/hypothesis_experiments_013.json` (`EXPERIMENT_PLAN_013`)**:
   - Path `data["H1_heteroscedastic_sharpness"]`:
     - `verify013_h1_qborn_dynamic_bloch_beta` $\leftarrow$ `H1["dynamic_bloch_purity"]` (`arch_family = "quantum_born"`, `best_val_ce = 1.9685`).
     - `verify013_h1_qborn_static_beta` $\leftarrow$ `H1["static_global_beta"]` (`arch_family = "quantum_born"`, `best_val_ce = 2.5689`).

### 4.5 Component 3: Multi-Tab WebUI & API Endpoints (`scripts/monitor_colab_runs.py`)

1. **HTTP API Endpoints Backed by `BenchmarkSQLiteDB`**:
   - `GET /api/snapshot` (and `GET /api/status`): Reads the latest snapshot from `BenchmarkSQLiteDB.load_snapshot_payload()`, returning:
     - `runs_catalog`: All `main_benchmark` runs (`run_tier == "main_benchmark"`), sorted with active non-obsolete runs first by ascending `best_val_ce`, including `pareto_frontier: bool`, `obsolete: bool`, and `obsolete_reason: str`.
     - `micro_runs_catalog`: All `micro_verification` runs (`run_tier == "micro_verification"`), sorted by `wave_num` descending then `best_val_ce` ascending.
     - `colab_sessions`: All 5 Colab GPU sessions from the `colab_sessions` table.
     - `session_queue`: All queued/active/completed jobs from the `session_queue` table.
     - `db_stats`: `{"db_path": str(db_path), "main_runs_count": ..., "micro_runs_count": ..., "pareto_classical_count": ..., "pareto_quantum_count": ..., "queued_jobs_count": ...}`.
   - `GET /api/micro-verifications`: Returns `{"ok": True, "micro_runs": micro_runs_catalog, "count": len(micro_runs_catalog)}`.
   - `GET /api/queue` & `POST /api/queue`:
     - `GET /api/queue`: Returns `{"ok": True, "queue": session_queue, "count": len(session_queue)}`.
     - `POST /api/queue`: Accepts `{"action": "enqueue" | "cancel" | "sync_manifest", "manifest": "configs/wave12_fleet.json", "item": {...}}` to enqueue or update jobs in SQLite `session_queue` with zero downtime.
   - `POST /api/colab-heartbeat`:
     - Allows a remote Colab session (when hosted on GCP or LAN) to push its live hardware & process status (`session`, `device`, `gpu_util_pct`, `vram_used_gib`, `ps_alive`, `ps_cmd`, `active_run_key`) directly into the `colab_sessions` SQLite table via `db.upsert_colab_session(...)`.
   - `POST /api/colab-stop` & Automatic Idle-VM Shutdown on Run Completion:
     - `query_colab_account_status()` runs `colab sessions` and `colab usage` (non-VM control-plane calls) to record which sessions are currently assigned (`vm_assigned: bool`), `compute_units_balance: float`, `usage_rate_per_hr: float`, and `active_assignments: int`.
     - Whenever the background poller in `collect_snapshot` harvests a newly completed run (`"COMPLETED"` in `polled["log"]` and `not is_alive`) on session `sess`, it automatically invokes `/home/festivity/.local/bin/colab stop -s <sess>` after saving `local_log` and `local_json` so the Colab VM is immediately unassigned (`0.00 CU/hr` burn rate).
     - `POST /api/colab-stop` accepts `{"session": "gpu_g4_w6_1"}` or `{"all_idle": true}` (or `{"all": true}`) to terminate one or all Colab sessions on demand from the WebUI (`🛑 Stop Idle Colab VMs` button) and updates `colab_sessions` in SQLite immediately.
   - `POST /api/ingest`:
     - Upserts the pushed run (`run_tier` auto-detected via `classify_run_tier` or explicitly passed in payload as `"main_benchmark"` or `"micro_verification"`) directly into the SQLite `runs`, `run_evals`, and `run_steps` tables via `db.upsert_run(...)`, recomputes Pareto-frontier / obsolete flags via `db.recompute_pareto_and_obsolete()`, and refreshes the in-memory snapshot cache in `< 25 ms`.
2. **3-Tab Interactive React Frontend in `render_html_dashboard(snap)`**:
   - **Top Navigation Tab Bar**:
     - **Tab 1: `🏆 Main FineWeb-Edu Leaderboard (${mainCatalog.length})`** — The primary `100M–900M` token FineWeb-Edu benchmark explorer, enhanced with:
       - **`⚡ PARETO FRONTIER` Badge** on every run with `r.pareto_frontier === true`, plus a quick filter pill **`⚡ Pareto Frontier Only`**.
       - Automatic Pareto-dominated obsolete hiding (`🗄️ Show Obsolete` toggle shows the exact `"Not on ClassicalModdedGPT/QuantumBornGPT Pareto frontier: superseded by ..."` tooltip and badge).
     - **Tab 2: `🧪 Pre-Launch Micro-Verifications & Hypothesis Tests (${microCatalog.length})`** — Dedicated tab displaying all controlled pre-launch micro-verifications (`verify019_60step_leaderboard_exact` at **`2.9254` Val CE** vs `verify019_60step_wave9_v2` at **`3.2974` Val CE`, plus Plans 013–016 & 020 micro-verifications):
       - Includes an explanatory banner detailing how pre-launch micro-verifications validate mathematical/systems hypotheses on controlled token streams (`0.5M–15M` tokens, `40–200` steps) before committing `900M`-token Colab GPU hours.
       - Interactive SVG Trajectory Chart plotting Validation Cross-Entropy vs. Steps/Tokens for the visible micro-verification runs.
       - Filterable, sortable Micro-Verifications Leaderboard Table with `🃏 Card` buttons opening the full `ModelCardModal` for any micro-verification run!
     - **Tab 3: `🖥️ Live Colab Fleet & Job Queue (${colabSessions.length} GPUs • ${queuedCount} Queued)`** — Full-view operational control center (in addition to the compact live summary bar at the top of Tab 1) showing:
       - Detailed per-session cards for `gpu_g4_w6_1`, `gpu_g4_w6_2`, `gpu_g4_w6_3`, `gpu_a100_w6_1`, and `gpu_a100_w6_2` (live `SM %`, `MemBW %`, `VRAM GiB`, `Power W`, `Temp °C`, `PID cmd`, latest completed run, and next queued run).
       - The **SQLite Session Job Queue Table** (`session_queue`), showing every queued/running/completed wave job, its target Colab session, architecture family, CLI flags, and status badge (`⏳ QUEUED`, `🟢 RUNNING`, `✅ COMPLETED`).

### 4.6 Component 4: Wave-12 `900M`-Token Unification Fleet (`configs/wave12_fleet.json`)

Note on `ClassicalModdedGPT` Deep `L=16` Compatibility: `ClassicalModdedGPT` (`src/qdlblocks/models/classical_modded_gpt.py` lines 412–428, 515–523, 568–577, 642–656) and `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` natively support arbitrary even `num_layers` (`L=16` allocates `n_half = 8` encoder-to-decoder `skip_weights`, maps layers `0, 1, 2` and `13, 14, 15` to the 3 value-embedding tables via `_ve_bank_index`, and parses comma-separated `--skip-attn-layers 7,11` into `(7, 11)`). No additional model or CLI flag changes are required for `L=16`.

We create `/home/festivity/agy/QuantumDLBlocks/configs/wave12_fleet.json` (and add `"W12"` to `WAVE_DEFAULT_LABELS` and `WAVE_DEFAULT_PLANS` in `src/qdlblocks/benchmarks/model_card.py`) with 5 next-generation `900M`-token runs:

| Session | Hardware | Run Key | Architecture & Hypothesis |
| :--- | :--- | :--- | :--- |
| `gpu_g4_w6_1` | RTX PRO 6000 Blackwell (96GB) | `w12_900m_classical_leaderboard_exact_b262k_progwin_docmask` | **Classical Leaderboard-Exact Unification ($B_{\text{tok}}=262\text{k}$, `3,429` Steps + Progressive Window `128->1792` + DocMask)**: Combines our #1 Wave-11 winner (`b262k_3429st` at `3.3688` CE) with our #2 Wave-11 winner (`progwin_docmask` at `3.3755` CE), `H=6, head_dim=128`, `muon_lr=0.045`, `wsd 0.40->0.05`. |
| `gpu_g4_w6_2` | RTX PRO 6000 Blackwell (96GB) | `w12_900m_classical_leaderboard_exact_b131k_6858st` | **Classical Leaderboard-Exact High-Step Scaling ($B_{\text{tok}}=131\text{k}$, `6,858` Steps)**: Tests `2x` optimizer updates (`batch_size=64, grad_accum=2, seq_len=1024`) under unit-gain EMA Muon (`ema_momentum=True, wd=0, muon_lr=0.035, wsd 0.40->0.05`) + `lr_wte=0.6` to push toward `< 3.33` Val CE. |
| `gpu_a100_w6_1` | NVIDIA A100-SXM4-80GB | `w12_900m_classical_leaderboard_exact_deep_L16_b262k` | **Deep `L=16` Classical Leaderboard-Exact ($B_{\text{tok}}=262\text{k}$, `3,429` Steps)**: Scales `ClassicalModdedGPT` from `L=12` to `L=16` (`d=768, H=6, head_dim=128, skip_attn_layers=7,11, num_value_embeddings=3, muon_lr=0.04`) on `900M` FineWeb-Edu tokens. |
| `gpu_g4_w6_3` | RTX PRO 6000 Blackwell (96GB) | `w12_900m_qborn_speedrun_v5_deep_L12_H8_d128_lr035` | **QuantumBornGPT Speedrun V5 Deep `L=12` + `H=8 (head_dim=128)` Unification**: Combines deep `L=12, d=1024, 4d-Frame (4096x2), Asym-1024, SkipAttn(7)` with the `H=8 (head_dim=128)` + `muon_lr=0.035` + `wsd 0.40->0.10` recipe that powered Wave 11's #1 Quantum record (`3.6787` CE). |
| `gpu_a100_w6_2` | NVIDIA A100-SXM4-80GB | `w12_900m_qborn_speedrun_v5_L8_H8_d128_b32k_27450st` | **QuantumBornGPT Speedrun V5 High-Frequency Step Scaling (`L=8, H=8, head_dim=128`, `batch_size=64, grad_accum=1, seq_len=512` = `27,450` Steps)**: Doubles the optimizer step count of our #1 Quantum record (`w11_900m_qborn_speedrun_v4_L8_leaderboard_hybrids`) with `muon_lr=0.03` and `wsd 0.40->0.05`. |

---

## 5. Verification & Unit Testing Plan

1. **Unit Test Suite (`tests/test_hypothesis_020_sqlite_pareto_and_tabs.py`)**:
   - `test_sqlite_db_schema_and_sync`: Verifies that `BenchmarkSQLiteDB` creates all 6 tables in WAL mode, syncs `main_benchmark` and `micro_verification` runs, and round-trips `runs_catalog`, `micro_runs_catalog`, `colab_sessions`, and `session_queue` losslessly.
   - `test_pareto_frontier_and_auto_obsolete_by_family`: Verifies that `compute_family_pareto_and_obsolete` computes independent 2D `(wall_min, best_val_ce)` Pareto frontiers for `classical_modded` and `quantum_born`, automatically marks wave-superseded dominated runs as `obsolete = True` with the exact canonical string `"Not on {fam_disp} Pareto frontier: superseded by ..."`, and respects `explicit_user_override`.
   - `test_micro_verification_separation_and_verify019_presence`: Verifies that `verify019_60step_leaderboard_exact` (`2.9254` CE) and `verify019_60step_wave9_v2` (`3.2974` CE) are classified as `run_tier == "micro_verification"`, appear in `micro_runs_catalog` (and `GET /api/micro-verifications`), and do not pollute the `main_benchmark` leaderboard.
   - `test_http_api_dual_ingest_heartbeat_and_queue`: Boots `create_benchmark_http_server` on an ephemeral port (`port=0`) backed by a temporary SQLite DB and tests `GET /api/snapshot`, `GET /api/micro-verifications`, `POST /api/ingest`, `POST /api/colab-heartbeat`, `POST /api/queue`, and `POST /api/obsolete`.
2. **Pre-Launch Verification Script (`examples/30_verify_plan020_sqlite_pareto_and_wave12.py`)**:
   - Executes `H1`, `H2`, and `H3` end-to-end, syncs `/home/festivity/agy/QuantumDLBlocks/.cache/benchmarks.db`, runs and ingests the 40-step Quantum V5 micro-verification arms (`verify020_40step_qborn_v5_H4_d64_lr035` vs `verify020_40step_qborn_v4_H8_d32_lr025`), enqueues `configs/wave12_fleet.json` into `session_queue`, and writes `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_020.json`.
