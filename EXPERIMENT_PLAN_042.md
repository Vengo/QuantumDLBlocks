# Experiment Plan 042 (`Wave 27`): Standardized Iso-Token & Iso-FLOP Scaling Tracker, Empirical `< 3.28` Ceiling & Wavelet-Rank Probe, and Smaller-Batch Scaled `d = 512` `FineWeb-Edu` Study

**Date**: October 24, 2026  
**Wave**: `Wave 27` (`EXPERIMENT_PLAN_042.md`)  
**Hardware Target**: Local `NVIDIA GeForce RTX 5060 Ti` (`15.0 GB` VRAM ceiling via `torch.cuda.set_per_process_memory_fraction(0.95, device=0)`, native `bfloat16` & `float8_e4m3fn` tensor-core GEMMs)

---

## 1. Executive Summary & Motivation

In `Wave 26` (`EXPERIMENT_PLAN_041.md`), expanding to `15.0 GB` VRAM and a `24,576`-token batch size (`B = 48, T = 512`) with `Solution A + B` (`Convex-Peak Traceless + Dyadic Coarse-to-Fine Born`) and `FP8` readout GEMMs broke our all-time `FineWeb-Edu` validation Cross-Entropy record (`4.8199` Val CE at `1,800` steps / `44.24M` tokens vs. `Wave 25`'s `5.0162` at `3,600` steps / `29.49M` tokens). However, comparing runs across different batch sizes (`B = 16` vs. `B = 48`) and network widths (`d = 256` vs. `d = 384`) exposed three critical questions that `Wave 27` resolves:

1. **Standardizing Training Progress When Varying Batch Size (`Iso-Token` & `Iso-FLOP` Standardization)**:
   - Step count $S$ means different physical token budgets under different batch sizes ($B_{\text{tok}} = B \times T$): `900` steps is `7.37M` tokens at `B = 16` (`8,192` tok/step), `11.06M` tokens at `B = 24` (`12,288` tok/step), and `22.12M` tokens at `B = 48` (`24,576` tok/step).
   - Moreover, Warmup-Stable-Decay (`WSD`) learning-rate cooldown ($\tau \in [0.80, 1.00]$) reduces validation Cross-Entropy by `-0.18` to `-0.22` nats at the end of a run, so comparing an intermediate step of a long run ($\tau < 0.80$) to the final step of a short run ($\tau = 1.00$) confounds token horizon with schedule phase.
   - We therefore upgrade [`scripts/fineweb_convex_vs_concave_slope_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_convex_vs_concave_slope_tracker.py) and our verification runner to standardize all training budgets and evaluation checkpoints on a universal **Token Horizon Grid**:
     $$N_{\text{tok}} \in \{2.46\text{M},\; 7.37\text{M},\; 14.75\text{M},\; 29.49\text{M},\; 44.24\text{M},\; 100.0\text{M}\}$$
     alongside **Total Training Compute FLOPs** ($C = 6 \, N_{\text{params}} \, N_{\text{tok}}$) and **Optimizer Update Count** ($S_{\text{opt}} = N_{\text{tok}} / B_{\text{tok}}$).
2. **Proving Empirically That `Radical2Pow16WaveletCholeskyGPT` Still Achieves `< 3.28` Val CE (`Empirical Ceiling & Wavelet-Rank Probe`)**:
   - In `Waves 22–26`, our $N = 2^{16}$ Wavelet-Cholesky pyramid kept `(mid_detail_dim, fine_detail_dim) = (128, 64)` fixed even when `embed_dim` increased from `256` to `384`, and `Wave 26` added an `L = 4` inverse-temperature clamp `beta_t <= 26.8`.
   - We build [`scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py) to test both multi-scale wavelet reconstruction fidelity across $d \in \{256, 384, 512\}$ with proportional wavelet detail ranks `(mid_detail_dim, fine_detail_dim)` $\in \{(128, 64), (192, 96), (256, 128)\}$ (`50%` mid, `25%` fine) and multi-batch `FineWeb-Edu` optimization capacity under both `BF16` and `FP8` readout GEMMs, verifying that the full model drives `FineWeb-Edu` Cross-Entropy **well below `3.28` nats (`< 1.50` nats)**.
3. **Bigger Network + Smaller Batch (`B = 16` / `B = 24`) Iso-Token Scaling Study on `FineWeb-Edu`**:
   - At `B = 48` (`24,576` tok/step), a `7.37M`-token run receives only `300` optimizer updates (`5.5536` Val CE at `d = 256`; `5.5461` at `d = 384`), whereas `B = 16` (`8,192` tok/step) receives `900` optimizer updates (`5.4447` Val CE at `d = 256`).
   - Dropping batch size to `B = 16` (`8,192` tok/step) or `B = 24` (`12,288` tok/step) frees activation VRAM to scale the network to **`d = 384, L = 6` (`mid=192, fine=96`, `17.34M` params)** and **`d = 512, L = 6` (`mid=256, fine=128`, `26.88M` params)** while saturating the RTX 5060 Ti tensor cores via wider $(8192 \times 512) \times (512 \times 1024)$ per-layer matrix multiplications and giving the network $2\times$–$3\times$ more `StiefelMuon` updates per million tokens.

---

## 2. Falsifiable Hypotheses & Acceptance Gates

### Hypothesis `H1`: Standardized Token-Horizon (`tokens_M`) & Compute-FLOPs (`FLOPs`) Scaling Tracker
- **Claim**: Upgrading [`scripts/fineweb_convex_vs_concave_slope_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_convex_vs_concave_slope_tracker.py) to index all `Waves 19–27` runs by true token milestones $N_{\text{tok}} \in \{2.46\text{M}, 7.37\text{M}, 14.75\text{M}, 29.49\text{M}, 44.24\text{M}\}$ and compute FLOPs $C = 6 N_{\text{params}} N_{\text{tok}}$ quantifies both the batch-size step-efficiency effect (showing why `B = 16` beats `B = 48` at `7.37M` tokens when $\tau = 1.0$) and the model-width scaling slope acceleration ($\Delta \text{CE} / \Delta \ln N_{\text{tok}}$ steepening as $d$ increases from `256` $\to$ `384` $\to$ `512`).
- **Falsifiable Gate**:
  1. `.cache/fineweb_convex_vs_concave_slope_tracker.json` records standardized `iso_token_table` milestones (`ce_at_2p46m`, `ce_at_7p37m`, `ce_at_14p75m`, `ce_at_29p49m`, `ce_at_44p24m`), `tokens_per_step`, `total_compute_tflops`, `optimizer_updates_per_10m_tokens`, and joint Chinchilla $(N_{\text{params}}, N_{\text{tok}})$ projections.
  2. All unit tests in [`tests/test_fineweb_slope_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_fineweb_slope_tracker.py) pass.

### Hypothesis `H2`: Empirical `< 3.28` Capacity Ceiling & Proportional Wavelet-Rank Probe
- **Claim**: Scaling the $N = 2^{16}$ Wavelet-Cholesky detail ranks proportionally with hidden dimension $d$ (`mid_detail_dim = d // 2`, `fine_detail_dim = d // 4`: `(128, 64)` at $d=256$, `(192, 96)` at $d=384$, `(256, 128)` at $d=512$) strictly increases vocabulary wavelet reconstruction fidelity ($R^2_{\text{recon}}$) and sharp detector effective rank ($R_{\text{sharp}}$) compared to fixed `(128, 64)`, and `Radical2Pow16WaveletCholeskyGPT` with `Solution A + B` and `FP8` readout GEMMs drives empirical Cross-Entropy on real `FineWeb-Edu` sequences **below `3.28` nats (`< 1.80` nats)** across all three widths $d \in \{256, 384, 512\}$.
- **Falsifiable Gate**:
  1. In [`.cache/empirical_3p28_ceiling_and_wavelet_rank_probe.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/empirical_3p28_ceiling_and_wavelet_rank_probe.json):
     - Proportional wavelet ranks `(192, 96)` at $d=384$ and `(256, 128)` at $d=512$ achieve higher reconstruction $R^2$ and higher $R_{\text{sharp}}$ than fixed `(128, 64)`.
     - Empirical `FineWeb-Edu` multi-batch optimization achieves $\text{CE}_{\text{min}} < 3.28\text{ nats}$ (and $< 1.80\text{ nats}$) for all $d \in \{256, 384, 512\}$ under `readout_gemm_precision="fp8"`, with $\text{CE}_{\text{min}}(d=512) < \text{CE}_{\text{min}}(d=384) < \text{CE}_{\text{min}}(d=256)$.

### Hypothesis `H3`: Standardized Iso-Token Scaling Study (`Bigger Network + Smaller Batch` on `FineWeb-Edu`)
- **Claim**: When standardized to identical token horizons ($\tau = 1.0$ WSD cooldown at target token budget $N_{\text{tok}}$) under `15.0 GB` VRAM:
  1. **At the `7.37M`-Token Horizon (`Iso-Token` Width & Batch Study)**:
     - Reducing batch size from `B = 48` (`24,576` tok/step, `300` steps) to `B = 16` (`8,192` tok/step, `900` steps) or `B = 24` (`12,288` tok/step, `600` steps) while scaling the network from `d = 256` (`10.40M` params) $\to$ **`d = 384, L = 6` (`mid=192, fine=96`, `17.34M` params)** $\to$ **`d = 512, L = 6` (`mid=256, fine=128`, `26.88M` params)** produces a strict monotonic improvement in `7.37M`-token Validation Cross-Entropy:
       $$\text{CE}_{7.37\text{M}}(d=512, B=16) < \text{CE}_{7.37\text{M}}(d=384, B=16) < \text{CE}_{7.37\text{M}}(d=256, B=16) = 5.4447 < \text{CE}_{7.37\text{M}}(d=384, B=48) = 5.5461$$
     - Furthermore, at `d = 512`, comparing `B = 16` (`8,192` tok/step, `900` steps = `7.37M` tok) vs. `B = 24` (`12,288` tok/step, `600` steps = `7.37M` tok) identifies the optimal throughput-vs-convergence operating point on the RTX 5060 Ti.
  2. **At the `14.75M`-Token & `29.49M`-Token Horizons (Breaking the `14.75M` and `29.49M` Records)**:
     - Scaling the bigger network (`d = 512, L = 6, mid=256, fine=128`, `26.88M` params) at smaller batch size (`B = 24`, `12,288` tok/step) to **`14.75M` tokens** (`1,200` steps) and **`29.49M` tokens** (`2,400` steps) breaks both:
       - `Wave 26`'s `14.75M`-token record (`5.0658` Val CE at `B = 48, d = 384`), and
       - `Wave 25`'s `29.49M`-token record (`5.0162` Val CE at `B = 16, d = 256`) **and** `Wave 26`'s `44.24M`-token all-time project record (`4.8199` Val CE at `B = 48, d = 256`) in only `29.49M` tokens!
- **Falsifiable Gate**:
  1. $\text{CE}_{7.37\text{M}}(d=512, B=16) < \text{CE}_{7.37\text{M}}(d=384, B=16) < 5.4447$.
  2. $\text{CE}_{14.75\text{M}}(d=512) < 5.0658$ (beats `Wave 26`'s `14.75M`-token `d=384, B=48` record).
  3. $\text{CE}_{29.49\text{M}}(d=512) < 4.8199$ (beats `Wave 25`'s `29.49M`-token `5.0162` record AND `Wave 26`'s `44.24M`-token `4.8199` all-time record in `33%` fewer tokens!).
  4. Peak GPU memory across all runs stays $\le 15.0\text{ GB}$ VRAM.

---

## 3. Planned Code & Architectural Changes

1. **Proportional Multi-Scale Wavelet Synthesis & Width-Aware PPMI-SVD Initialization ([`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py))**:
   - When `cfg.embed_dim > 256` (`d = 384` or `512`) and `ppmi_embeddings` has `256` columns (`half_d = d // 2 > 128`), equalize the variance of the higher SVD components (`128..half_d`) in `initialize_radical_2pow16_with_ppmi_svd` so all $d$ dimensions carry balanced initial signal energy.
   - Allow `beta_t` in `Radical2Pow16WaveletCholeskyGPT.forward` to scale up to `max = 44.0` when `embed_dim > 256` or `num_layers > 4` (while preserving `26.8` for the `d=256, L=4` `Plan 041` `H2` configuration).
2. **Token-Horizon & Compute-FLOPs Standardization in [`scripts/fineweb_convex_vs_concave_slope_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_convex_vs_concave_slope_tracker.py)**:
   - Compute true token coordinates $N_{\text{tok}}(s) = s \times B_{\text{tok}}$ for every eval checkpoint in `.cache/live_logs/*_results.json`, align runs onto standardized token milestones (`2.46M, 7.37M, 14.75M, 29.49M, 44.24M`), compute total training FLOPs $C = 6 N_{\text{params}} N_{\text{tok}}$, and fit both single-run and joint Chinchilla $(N_{\text{params}}, N_{\text{tok}})$ scaling curves.
3. **Empirical `< 3.28` Ceiling & Wavelet-Rank Probe ([`scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py))**:
   - Standalone CLI diagnostic testing wavelet pyramid reconstruction $R^2$, detector effective rank $R_{\text{sharp}}$, and multi-batch `FineWeb-Edu` optimization ceiling across $d \in \{256, 384, 512\}$ and wavelet detail ranks `(128, 64)`, `(192, 96)`, `(256, 128)`.
4. **Iso-Token Scaling Verification Runner ([`examples/52_verify_plan042_isotoken_scaling_and_3p28_ceiling.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/52_verify_plan042_isotoken_scaling_and_3p28_ceiling.py))**:
   - Standardizes all runs by `target_tokens` and `eval_tokens` (automatically converting token milestones into step indices for any `batch_size * seq_len`), executes the `H1`, `H2`, and `H3` experiments, and writes [`.cache/hypothesis_experiments_042.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_042.json).

---

## 4. Verification & Testing Plan

- **Unit & Schema Tests ([`tests/test_plan042_isotoken_scaling_and_3p28_ceiling.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan042_isotoken_scaling_and_3p28_ceiling.py))**:
  - Tests token-to-step standardization across `B in {16, 24, 48}`, proportional wavelet detail ranks `(192, 96)` at $d=384$ and `(256, 128)` at $d=512$, and the schema of `.cache/empirical_3p28_ceiling_and_wavelet_rank_probe.json` and `.cache/fineweb_convex_vs_concave_slope_tracker.json`.
- **Hypothesis Gate Tests ([`tests/test_hypothesis_042_isotoken_scaling_and_3p28_ceiling.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_042_isotoken_scaling_and_3p28_ceiling.py))**:
  - Asserts `overall_pass is True` and all numerical gates for `H1`, `H2`, and `H3` in `.cache/hypothesis_experiments_042.json`.

---

## 5. Operational Tooling & Script Opportunities

### 5.1 Existing Scripts Used
- [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py): Enforces `<= 1,000` lines target, `<= 8` args per function, and zero semicolons across all modified/new files.
- [`scripts/rtx5060ti_15gb_batch24576_network_sizer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/rtx5060ti_15gb_batch24576_network_sizer.py): Hardware throughput and VRAM sizer for `d = 256, 384, 512`.
- [`scripts/fineweb_convex_vs_concave_slope_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_convex_vs_concave_slope_tracker.py): Upgraded in `Wave 27` to standardize by token horizon (`tokens_M`) and compute FLOPs (`6 * N_params * N_tokens`).

### 5.2 `Build-Before-Launch` Scripts
- [`scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py):
  ```bash
  /home/festivity/agy/QuantumDLBlocks/.venv/bin/python scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py --dims 256,384,512 --steps 160 --json-out .cache/empirical_3p28_ceiling_and_wavelet_rank_probe.json
  ```

### 5.3 `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/critical_batch_size_gradient_noise_profiler.py`: Measures empirical gradient noise scale $B_{\text{simple}} = \operatorname{Tr}(\Sigma) / \|G\|_2^2$ across `B in {8, 16, 24, 32, 48}` on `FineWeb-Edu` to predict the exact critical batch size $B_{\text{crit}}(d, \text{CE})$ at each loss level.
2. `scripts/isotoken_wsd_cooldown_branch_evaluator.py`: Evaluates short 20% WSD cooldown branches (`1.0x -> 0.1x` LR) from any intermediate checkpoint of a long run so intermediate token milestones can be compared on an exact $\tau = 1.0$ cooled-down basis without re-training from Step 0.
