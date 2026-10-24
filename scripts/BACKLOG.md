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
- **`scripts/vram_envelope_autotuner.py`** `[PROMOTED TO EXPERIMENT_PLAN_033]`:
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
- **`scripts/full_ambient_step0_calibration_audit.py`** `[PROMOTED TO EXPERIMENT_PLAN_033]`:
  - A lightweight CLI utility that inspects Step-0 inner-product distributions (`c_sharp`, `c_diff`, `max_self_overlap`, `self_register_overlap_max`, `directed_bigram_overlap_ratio`, and `Step 0 ECE`) across `(Q, A, use_ortho_decouple, decouple_self_in_full_ambient)` configurations before launching training runs.

## Plan 033 Candidate (`Nice-to-Have`) Tooling
- **`scripts/ast_signature_and_length_linter.py`** `[PROMOTED TO EXPERIMENT_PLAN_034]`:
  - Fast AST pre-commit check that flags any new Python file exceeding `600` lines (`400` lines target) or any new function exceeding `8` arguments without a `@dataclass` config (`docs/SCRIPT_BEST_PRACTICES.md`).
- **`scripts/wavelet_cholesky_spectrum_probe.py`** (extended from `scripts/equidimensional_on_spectrum_probe.py`) `[PROMOTED TO EXPERIMENT_PLAN_034]`:
  - Diagnostic CLI that audits multi-scale Sweldens lifting wavelet reconstruction error, per-scale energy fractions ($E_{\text{coarse}}, E_{\text{detail}, 1..S}$), `BlockCholeskyLinear` SPD condition numbers $\kappa(L_s L_s^\top)$, and Tied-$2^{16}$ basis parameter savings and Step-0 $J_{\text{anti}}$ self-overlap.

## Plan 034 Candidate (`Nice-to-Have`) Tooling
- **`scripts/mera_entanglement_cone_visualizer.py`** `[PROMOTED TO EXPERIMENT_PLAN_035]`:
  - Computes the causal entanglement light-cone and mutual information across dyadic wavelet scales $s \in \{1, \dots, S\}$ for any trained `WaveletCholeskyMultiscaleBranch` checkpoint and exports an interactive SVG/HTML entanglement-cone diagram.
- **`scripts/vocab_dark_ancilla_utilization_audit.py`** `[PROMOTED TO EXPERIMENT_PLAN_035]`:
  - Measures how much probability mass or virtual interference flows through the `15,279` dark ancilla basis states ($50257 \dots 65535$) in the $N = 2^{16} = 65,536$-dimensional ambient Hilbert space during multi-scale Wavelet-Cholesky readout.

## Plan 035 Candidate (`Nice-to-Have`) Tooling
- **`scripts/dyadic_spectral_tree_inspector.py`** `[PROMOTED TO EXPERIMENT_PLAN_036]`:
  - Prints human-readable GPT-2 token clusters at each node of the 16-bit Dyadic Spectral Tree ($2^8 = 256$ macro clusters down to $2$-token sibling leaves) to inspect semantic renormalization structure.
- **`scripts/cholesky_band_sparsity_profiler.py`** `[PROMOTED TO EXPERIMENT_PLAN_036]`:
  - Benchmarks wall-clock speed and condition number $\kappa(L_s L_s^\top)$ across block-diagonal vs. banded lower-triangular Cholesky factorizations on consumer GPUs.

## Plan 036 Candidate (`Nice-to-Have`) Tooling
- **`scripts/dyadic_bigram_transition_heatmap.py`** `[PROMOTED TO EXPERIMENT_PLAN_037]`:
  - Exports the $256 \times 256$ coarse dyadic cluster transition matrix $T_{\text{coarse}}$ and top predecessor-to-successor semantic cluster flows from a trained `Tied2Pow16WaveletCholeskyPyramid`.
- **`scripts/wavelet_scale_ablation_probe.py`** `[PROMOTED TO EXPERIMENT_PLAN_037]`:
  - Evaluates zero-shot validation CE sensitivity when selectively zeroing individual wavelet scales (`256`, `2048`, `8192`, `65536`) at inference time to quantify how much NLL reduction comes from each dyadic resolution level.

## Plan 037 Candidate (`Nice-to-Have`) Tooling
- **`scripts/parseval_tail_overlap_profiler.py`** `[PROMOTED TO EXPERIMENT_PLAN_038]`:
  - Measures the empirical Parseval frame spectrum $\lambda_1 \ge \dots \ge \lambda_d$ of $G = \sum_{v=1}^V m_v m_v^\top$ and the tail partition-function fraction $Z_{\text{tail}} / Z_{\text{total}}$ across rank buckets (`1`, `2–5`, `6–50`, `51–1000`, `>1000`).
- **`scripts/block_cholesky_kernel_benchmark.py`** `[PROMOTED TO EXPERIMENT_PLAN_038]`:
  - Benchmarks two-pass `x @ L @ L^T` vs. single-pass fused `x @ (L L^T)` across batch sizes, sequence lengths, and `bfloat16`/`float32` dtypes.

## Plan 038 Candidate (`Nice-to-Have`) Tooling
- **`scripts/last_layer_measurement_adjoint_bench.py`** `[PROMOTED TO EXPERIMENT_PLAN_039]`:
  - Benchmarks FLOPs, memory traffic, and self-overlap suppression of the $(B \cdot T, d)$ Last-Layer Measurement Operator vs. $(V, d)$ Vocabulary Detector synthesis across batch/sequence shapes.
- **`scripts/traceless_observable_rank_bucket_probe.py`** `[PROMOTED TO EXPERIMENT_PLAN_039]`:
  - Profiles logit margins and probability mass across rank buckets (`1`, `2–5`, `6–20`, `21–100`, `>100`) across `naive_kboson_born`, `traceless_kboson_born`, `convex_traceless_born`, and `dyadic_hierarchical_born` as a function of `traceless_gamma`.

## Plan 039 Candidate (`Nice-to-Have`) Tooling
- **`scripts/multiclass_ce_margin_decomposer.py`** `[PROMOTED TO EXPERIMENT_PLAN_040]`:
  - Decomposes multi-class Cross-Entropy into target-peak deficit vs. positive-cosine Parseval runner-up tail vs. negative-cosine tail across $K \in \{100, 1000, 50257\}$ classes.
- **`scripts/dyadic_tree_conditional_calibration_probe.py`** `[PROMOTED TO EXPERIMENT_PLAN_040]`:
  - Audits coarse-cluster accuracy vs. conditional intra-cluster accuracy and Expected Calibration Error (`ECE`) across multi-scale Wavelet-Cholesky trees.

## Plan 040 Candidate (`Nice-to-Have`) Tooling
- **`scripts/imagenet1k_quantum_block_ablation_profiler.py`** `[PROMOTED TO EXPERIMENT_PLAN_041]`:
  - Standalone CLI profiler that loads trained `WaveletCholeskyVisionClassifier` checkpoints on ImageNet-1K and CIFAR-100 and exports a component-level ablation scorecard (parameter cost, throughput impact, and NLL delta per quantum block).
- **`scripts/fineweb_convex_vs_concave_slope_tracker.py`** `[PROMOTED TO EXPERIMENT_PLAN_041]`:
  - Tracks empirical positive-peak slope $\partial \ell_{t,y_t}/\partial c_{t,y_t}$ vs. negative-tail slope across training checkpoints (`0..3600`) on FineWeb-Edu.

## Plan 041 Candidate (`Nice-to-Have`) Tooling
- **`scripts/dyadic_coarse_to_fine_entropy_profiler.py`**:
  - Decomposes `FineWeb-Edu` token Cross-Entropy (`50,257` classes) into 8-bit coarse dyadic subtree entropy (`256` roots) + 8-bit conditional intra-subtree leaf entropy (`256` leaves/root).
- **`scripts/fp8_vs_bf16_born_logit_drift_auditor.py`**:
  - Measures cosine similarity, max logit error, and KL divergence between `fp8` (`torch._scaled_mm`) and `bf16` (`torch.mm`) Born-Boltzmann readouts across training checkpoints.
- **`scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`** `[PROMOTED TO EXPERIMENT_PLAN_042]`:
  - Audits $N=2^{16}$ Wavelet-Cholesky vocabulary reconstruction $R^2$, detector effective rank $R_{\text{sharp}}$, and empirical multi-batch `FineWeb-Edu` optimization capacity ceiling (`< 3.28` / `< 1.80` nats) across $d \in \{256, 384, 512\}$ and proportional wavelet detail ranks `(128, 64)`, `(192, 96)`, `(256, 128)`.

## Plan 042 Candidate (`Nice-to-Have`) Tooling
- **`scripts/critical_batch_size_gradient_noise_profiler.py`**:
  - Measures empirical gradient noise scale $B_{\text{simple}} = \operatorname{Tr}(\Sigma) / \|G\|_2^2$ across `B in {8, 16, 24, 32, 48}` on `FineWeb-Edu` to predict the exact critical batch size $B_{\text{crit}}(d, \text{CE})$ at each loss level.
- **`scripts/isotoken_wsd_cooldown_branch_evaluator.py`**:
  - Evaluates short 20% WSD cooldown branches (`1.0x -> 0.1x` LR) from any intermediate checkpoint of a long run so intermediate token milestones can be compared on an exact $\tau = 1.0$ cooled-down basis without re-training from Step 0.

