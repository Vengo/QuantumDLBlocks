# QuantumDLBlocks Operational Scripts & Tooling Backlog (`scripts/BACKLOG.md`)

See also [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md) for the full table of implemented and candidate operational scripts.

## Plan 015 Candidate (`Nice-to-Have`) Tooling
- **`--compare-waves w6,w7,w8` in `scripts/colab_fleet.py`** (Implemented in `scripts/colab_fleet.py` as `compare --waves w6,w7,w8` and `--compare-waves w6,w7,w8`):
  - Prints the aligned 50M-token trajectory comparison table across waves (`w6,w7,w8`) directly from `.cache/live_logs/`.

## Plan 016 Candidate (`Nice-to-Have`) Tooling
- **`--optimizer-filter` in `scripts/colab_fleet.py compare`**:
  - Allows filtering and grouping optimizer ablation runs (`muon_adamw`, `pure_adamw`, `sgd_nesterov`) side-by-side in CLI comparison tables.

## Plan 018 Candidate (`Nice-to-Have`) Tooling
- **`scripts/colab_fleet.py obsolete --waves w1,w2` CLI subcommand**:
  - Allows batch-marking entire waves obsolete from the terminal (the web UI and HTTP API `POST /api/obsolete` already support bulk marking).

## Plan 019 Candidate (`Nice-to-Have`) Tooling
- **Multi-GPU DDP `NorMuon` + `PolarExpress` variance reduction (`apply_normuon_variance_reduction`)**:
  - Second-moment variance reduction across orthogonalized polar updates for multi-node H100 clusters.

## Plan 020 Candidate (`Nice-to-Have`) Tooling
- **Cloud SQL / PostgreSQL replica adapter for multi-region GCP Cloud Run deployments (`--postgres-url`)**:
  - Optional PostgreSQL backend adapter alongside local WAL-mode SQLite (`BenchmarkSQLiteDB`) for multi-instance Cloud Run hosting.

## Plan 021 Candidate (`Nice-to-Have`) Tooling
- **`scripts/precompute_ppmi_svd_cache.py --shards 1..9 --embed-dims 768,1024 --buckets 2048`**:
  - Standalone offline CLI tool to prebuild and upload `.pt` PPMI-SVD spectral caches to HuggingFace/GCS (`compute_ppmi_svd_embeddings` already computes and caches the 9-shard sketch on GPU in `< 15s` at startup).
- **Multi-node `8xH100` NCCL all-to-all distributed `NgramTable` sharding (`--distributed-ngram-alltoall`)**:
  - Distributed 84.6M-row (`130 GB`) n-gram table sharding across 8 ranks via NCCL all-to-all (our single-GPU Colab fleet uses a compact GPU-resident `HashedNgramEmbedding`).
- **`scripts/colab_fleet.py auto-drain-queue --poll-seconds 60`**:
  - Autonomous queue dispatcher that launches the next `QUEUED` wave when a Colab VM finishes and stops.

## Plan 022 Candidate (`Nice-to-Have`) Tooling
- **`scripts/qubit_fock_spectrum_probe.py`**:
  - A diagnostic CLI tool that loads a trained `Sparse64QubitDensityLayer` checkpoint and plots the $64 \times 64$ two-qubit pair-excitation correlation heatmap $C_{ij} = \mathbb{E}[|\Psi_{\text{pair}}(i, j)|^2]$ and the rank-$R$ density-matrix purity histogram $\text{Tr}(\rho_t^2)$ across FineWeb-Edu validation documents.

## Plan 023 Candidate (`Nice-to-Have`) Tooling
- **`scripts/inspect_hierarchy_clusters.py`**:
  - Post-training CLI diagnostic that prints the top-10 vocabulary tokens most strongly coupled to each of the `256` higher-level quantum states (`256 × 16 × 32`) and the strongest off-diagonal concept-to-concept transitions in $H_{\text{high}} \in \mathbb{R}^{16 \times 256 \times 256}$.

## Plan 024 Candidate (`Nice-to-Have`) Tooling
- **`scripts/calibration_rank_histogram_probe.py`** `[PROMOTED TO EXPERIMENT_PLAN_025]`:
  - A standalone diagnostic CLI that evaluates a checkpoint or live model on `fineweb_val_000000.bin` and outputs a full rank-bucketed calibration table (`Rank 1`, `Ranks 2–5`, `Ranks 6–50`, `Ranks 51–1000`, `Ranks > 1000`, Expected Calibration Error `ECE`, and mean $-\log p(y_t)$ contribution per rank bucket) to visualize how `--last-layer-softmax` shifts probability mass into Ranks 2–50.

## Plan 025 Candidate (`Nice-to-Have`) Tooling
- **`scripts/vram_envelope_autotuner.py`**:
  - Given a hardware VRAM cap (e.g., `--max-vram-gb 8.0` or `16.0`) and a `QuantumBornGPT` or `ClassicalModdedGPT` architecture config, binary-searches `(micro_batch_size, grad_accum, checkpoint_blocks, chunk_tokens)` under `torch.cuda.set_per_process_memory_fraction` to find the highest-throughput configuration that guarantees zero OOMs.

## Plan 026 Candidate (`Nice-to-Have`) Tooling
- **`scripts/kboson_fidelity_landscape_visualizer.py`**:
  - Plots the empirical cosine similarity histogram $p(c_{\text{sharp}})$ and $p(c_{\text{diff}})$ and gradient gain $k / (2 F_{t,v})$ across target rank buckets `1`, `2–5`, `6–50`, `51–1000`, and `>1000` for trained Two-Beam $k$-Boson Mixed-State Readout checkpoints.

## Plan 027 Candidate (`Nice-to-Have`) Tooling
- **`scripts/kraus_schedule_visualizer.py`**:
  - A standalone CLI tool that plots $\lambda_{\text{diff}}(s)$, $k_{\text{eff}}(s)$, $\beta_t(s)$, and the 5-bucket Target-Rank NLL trajectories across schedule arms from `.cache/hypothesis_experiments_027.json` into an interactive SVG/HTML comparison card.

## Plan 028 Candidate (`Nice-to-Have`) Tooling
- **`scripts/stereographic_latitude_probe.py`**:
  - A diagnostic CLI that inspects a trained `--stereographic-lift` checkpoint on `fineweb_val_000000.bin` and plots the layerwise distribution of stereographic polar latitudes $\theta_{\text{lat}}(h_\ell)$ and phase-interference gates $\cos(\omega(z_1 - z_2) + \phi)$ across token frequency and target-rank buckets.

## Plan 029 Candidate (`Nice-to-Have`) Tooling
- **`scripts/synthetic_walk_attention_heatmap_exporter.py`**:
  - A diagnostic CLI that exports per-head Purified Born attention matrices $A_{ij}^{(h)}$ and stereographic polar-latitude trajectories $\theta_{\text{lat}}(t)$ on synthetic FineWeb random-walk sequences into an interactive SVG/HTML visual card.

## Plan 030 Candidate (`Nice-to-Have`) Tooling
- **`scripts/iso_wallclock_pareto_profiler.py`**:
  - A CLI benchmark utility that sweeps `(num_layers, embed_dim, frame_expansion)` under a fixed wall-clock budget (e.g., `--wall-seconds 300 --max-vram-gb 8.0`) on FineWeb-Edu to plot the Iso-Wall-Clock Validation Cross-Entropy vs. Token-Throughput Pareto curve on a single consumer GPU.

## Plan 031 Candidate (`Nice-to-Have`) Tooling
- **`scripts/bipolar_vs_concave_kboson_curve_plotter.py`**:
  - A standalone CLI diagnostic that plots the exact log-fidelity curve $\mathcal{E}(x)$, slope $\partial \mathcal{E}/\partial x$, and empirical per-bucket cosine similarity distribution $p(x_{t, y_t} \mid \text{Rank Bucket})$ comparing legacy concave $k \ln(1+x)$ vs. Bipolar Particle-Hole $k$-Boson across validation checkpoints.

## Plan 032 Candidate (`Nice-to-Have`) Tooling
- **`scripts/full_ambient_step0_calibration_audit.py`**:
  - A lightweight CLI utility that inspects Step-0 inner-product distributions (`c_sharp`, `c_diff`, `max_self_overlap`, `self_register_overlap_max`, `directed_bigram_overlap_ratio`, and `Step 0 ECE`) across `(Q, A, use_ortho_decouple, decouple_self_in_full_ambient)` configurations before launching training runs.
