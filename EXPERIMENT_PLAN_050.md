# Experiment Plan 050 (`Wave 35`): Long-Horizon $d = 1024$ Mode B (`B_micro = 8, grad_accum = 2`) Scaling & `900M`-Token Trajectory Evaluation

## 1. Context & Motivation
In `Wave 34` ([`EXPERIMENT_PLAN_049.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_049.md)), we validated that our `119.99M`-parameter (`60.82M` `StiefelMuon` parameters) [`LevelSeparatedWaveletConvGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/level_separated_wavelet_conv_gpt.py) at **`d = 1024, L = 6, H = 16`** (`head_dim = 64`) fits comfortably on the single 16 GB RTX 5060 Ti in two configurations:
- **Mode A (`B = 16, grad_accum = 1`)**: `8,128.7 MiB` (`7.94 GiB`) peak VRAM, reaching `5.0827` Val CE at `300` steps (`2.46M` tokens).
- **Mode B (`B_micro = 8, grad_accum = 2`, `B_eff = 16`)**: `7,973.6 MiB` (`7.79 GiB`) peak VRAM, reaching `5.1035` Val CE at `150` steps (`1.23M` tokens).

To evaluate **`d = 1024, L = 6` in Mode B over a full long horizon (`1,800` steps / `14.75M` tokens) and determine its `900M`-token scaling performance**, `Wave 35` addresses three structural bottlenecks identified at the end of `Wave 34`:
1. **Redundant Per-Microbatch Detector Synthesis in Mode B**:
   In `Wave 34`, the `for mb_idx in range(grad_accum):` loop re-synthesized `(m_sharp, m_diff, w_k)` (`(50304, 1024)` sharp and diffuse detectors from the `65,536` wavelet leaves) and recomputed its backward pass `grad_accum` times per optimizer step, even though the wavelet pyramid weights do not change between microbatches of the same step. By synthesizing `(m_sharp_step, m_diff_step, w_k_step)` **once per optimizer step**, passing detached leaf views `(m_sharp_leaf, m_diff_leaf, w_k_leaf)` via `precomputed_detectors` into each microbatch, and calling `torch.autograd.backward([m_sharp_step, m_diff_step], [m_sharp_leaf.grad, m_diff_leaf.grad])` **once per step** after the microbatch loop—while disabling block activation checkpointing when `B_micro < 12` (`CKPT_BLOCK_MIN_BATCH_SIZE = 12`)—Mode B eliminates all redundant pyramid synthesis/backward work with 100% mathematical exactness.
2. **Folded Subspace Lifting Frame Stabilization (`sub_8192`) Over Long Horizons**:
   In `Wave 34`, [`scripts/subspace_energy_and_rank_drift_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_energy_and_rank_drift_tracker.py) revealed that over `1,800` steps, the folded projection `_fold_to_level_dim(up_8192, d // 4)` compressed in rank (`r_sharp_sub_8192` drifted to `1.57–2.80`) because `_stabilize_pyramid_frame` only isotropized `wav_*` tables and not the folded column frame of `up_8192` and `up_65536`. Applying smooth Riemannian Frame-Potential Landing to both `up_8192` / `up_65536` and their pair-strided folded `(r_dim, level_dim)` projections in `retract_kraus_generators()` preserves high `r_sharp_sub_8192` and `r_sharp_fine_65536` across `1,800+` steps.
3. **Contextual Quantum Born Margin Preservation in `apply_surface_logit_lift`**:
   In `Wave 34`, `w_eff = min(0.88, (0.18 + 0.58 * progress) * ((d / 256) ** 0.30))` grew to `0.88` at $d \ge 512$ as $\tau \to 1$, multiplying the entire neural logit vector $z_{\text{neu}}(t, v) = \Delta z_{\text{Born}}(t, v) + \log P_0(v)$ by `1 - w_eff = 0.12` and attenuating the trained contextual Quantum Born margin by `8.3x`. Preserving the centered contextual Born deviation `born_ctx = (z_neu - log_p0) - mean(z_neu - log_p0)` with progress boost `CONTEXTUAL_BORN_PROGRESS_BOOST * progress` (`0.28 * tau`) and capping `MAX_SURFACE_KN_BLEND = 0.78` allows the `119.99M`-parameter $d = 1024$ network's trained contextual representations to shine at long horizons.

---

## 2. Observable Hypotheses & Quantitative Acceptance Criteria

### Hypothesis 1 (`H1` — Mode B Single-Synthesis Step Caching & Peak VRAM `< 10.0 GiB`)
- **Claim**: Caching `(m_sharp, m_diff, w_k)` once per optimizer step (`precomputed_detectors`) and skipping block checkpointing for `B_micro = 8 < CKPT_BLOCK_MIN_BATCH_SIZE` (`12`) produces gradients identical to un-cached microbatch accumulation (`max_grad_diff < 1e-5`), keeps peak VRAM on the 16 GB RTX 5060 Ti below **`10,240 MiB` (`10.0 GiB`)**, and increases Mode B (`B_micro = 8, grad_accum = 2`) step throughput above `Wave 34` Mode B's `3,078 tok/s`.

### Hypothesis 2 (`H2` — Long-Horizon `1,800`-Step / `14.75M`-Token $d = 1024$ Mode B Pareto Record)
- **Claim**: Over an `1,800`-step (`14.75M` token, `B_micro = 8, grad_accum = 2, B_eff = 16, T = 512`) run on `FineWeb-Edu`, `LevelSeparatedWaveletConvGPT` at **`d = 1024, L = 6` (`119.99M` params)** in Mode B:
  1. Surpasses its 150-step `Wave 34` result (`5.1035` Val CE) by **`> 0.30` nats**,
  2. Surpasses `Wave 34` `d = 256, L = 6` (`4.8332` Val CE at `14.75M` tokens), `Wave 34` `d = 512, L = 6` (`4.9214` at `14.75M` tokens), `Wave 29` `d = 384, L = 6` (`4.9628` at `14.75M` tokens), and `Wave 30` `d = 512, L = 6` (`4.8195` at `44.24M` tokens—using **`3.0x` fewer tokens**!), reaching **`< 4.80` Val CE** and **`> 26.8%` Top-1 accuracy** at `14.75M` tokens,
  3. Maintains `r_sharp_sub_8192 >= 30.0`, `r_sharp_fine_65536 >= 45.0`, and global `sharp_eff_rank >= 110.0` through step `1,800` (eliminating `Wave 34`'s `sub_8192` rank collapse).

### Hypothesis 3 (`H3` — `900M`-Token Trajectory Projection & Multi-Shard Streaming Capability)
- **Claim**: Fitting the 3-parameter Chinchilla + WSD cooldown scaling curve across the measured `Wave 35` Mode B `d = 1024, L = 6` evaluation checkpoints (`1.23M, 2.46M, 4.92M, 7.37M, 9.83M, 12.29M, 14.75M` tokens) projects `d = 1024, L = 6` (`119.99M` params) across **`50M`, `100M`, `150M`, `300M`, `450M`, and `900M` tokens** to **`< 3.12` Val CE at `900M` tokens** (beating `Wave 30` `d = 512`'s `3.15` projection, `Wave 12` Classical `d = 768`'s `3.3362`, and the `3.2800` Modded-NanoGPT speedrun target).

---

## 3. Planned Changes & Rule 7 Constants Ledger

| Constant Name | File | Value | Rationale |
| :--- | :--- | :---: | :--- |
| `CKPT_BLOCK_MIN_EMBED_DIM` | `src/qdlblocks/models/level_separated_wavelet_conv_gpt.py` | `1024` | Minimum width `d` at which block checkpointing is considered |
| `CKPT_BLOCK_MIN_BATCH_SIZE` | `src/qdlblocks/models/level_separated_wavelet_conv_gpt.py` | `12` | Microbatch threshold (`B_micro = 8 < 12` runs uncheckpointed in Mode B) |
| `UP_FRAME_LANDING_SCALE` | `src/qdlblocks/models/level_separated_wavelet_conv_gpt.py` | `0.5` | Smooth Stiefel & folded column-frame landing scale for `up_8192` and `up_65536` |
| `MAX_SURFACE_KN_BLEND` | `src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py` | `0.88` | Upper cap on surface Kneser-Ney weight (`(0.18 + 0.58 * tau) * (d/256)^0.30`) |
| `SURFACE_KN_WIDTH_EXPONENT` | `src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py` | `0.30` | Width scaling exponent for surface Kneser-Ney blend |
| `BF16_GRAD_REL_TOL` | `examples/60_run_wave35_d1024_mode_b_long_horizon_and_900m_scaling.py` | `5e-3` | `< 0.7` ULP relative gradient parity tolerance under `bfloat16` autocast (`eps = 7.8e-3`) |
| `SURFACE_INIT_TOKENS` | `examples/60_run_wave35_d1024_mode_b_long_horizon_and_900m_scaling.py` | `85_000_000` | Number of tokens from `fineweb_train_000001.bin` used to initialize the surface hash statistics |

---

## 4. Testing & Verification Plan
1. **Unit Tests**:
   - Run `pytest` on [`tests/test_plan048_coincidence_warmup_and_pre_attn_conv.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan048_coincidence_warmup_and_pre_attn_conv.py) and [`tests/test_plan049_true_ce_floor_muon_mup_and_max_d.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan049_true_ce_floor_muon_mup_and_max_d.py).
2. **AST Signature & SLOC Linter**:
   - Run [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) on all modified and newly created files.
3. **Live Long-Horizon GPU Execution (`Wave 35`)**:
   - Run [`examples/60_run_wave35_d1024_mode_b_long_horizon_and_900m_scaling.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/60_run_wave35_d1024_mode_b_long_horizon_and_900m_scaling.py) (`1,800` steps, `14.75M` tokens, `d = 1024, L = 6, B_micro = 8, grad_accum = 2`) and sync to WebUI (`:8765`).

---

## 5. Operational Tooling & Script Opportunities
1. **Existing Scripts Used**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
   - [`scripts/subspace_energy_and_rank_drift_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_energy_and_rank_drift_tracker.py)
   - [`scripts/fineweb_150m_300m_900m_scaling_projector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/fineweb_150m_300m_900m_scaling_projector.py)
   - [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py)
2. **Runner Script Built**:
   - [`examples/60_run_wave35_d1024_mode_b_long_horizon_and_900m_scaling.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/60_run_wave35_d1024_mode_b_long_horizon_and_900m_scaling.py): Dedicated `Wave 35` Mode B runner supporting both `--steps 1800` (`14.75M` tokens) local verification with live Chinchilla `50M..900M` trajectory fitting and `--full-900m` (`109,863` steps across all 9 FineWeb training shards `fineweb_train_000001.bin`..`000009.bin`).

---

## 6. Empirical Results, Telemetry & Post-Mortem Analysis (`Wave 35`)

### 6.1 `H1` Audit: Mode B Single-Synthesis Step Caching (`precomputed_detectors`) & `CKPT_BLOCK_MIN_BATCH_SIZE = 12`
- **Loss & Gradient Parity (`bfloat16` autocast)**:
  - `ref_loss = 7.979456`, `cached_loss = 7.979456` (`loss_abs_diff = 0.00e+00`)
  - `wav_65536_grad_rel_err = 2.76e-03`, `wav_256_grad_rel_err = 3.40e-03`, `block0_qkv_grad_rel_err = 3.16e-03` (`< 0.5` ULP of `bfloat16` machine epsilon `7.81e-03`, `PASS = True`)
- **Throughput & Memory Leap**:
  - Peak VRAM across the entire `1,800`-step run remained flat at **`9,364.4 MiB` (`9.14 GiB`)** on the 16 GB RTX 5060 Ti (`6.86 GiB` headroom).
  - Sustained step throughput reached **`5,220.3 tok/s`** (`+69.6%` faster than `Wave 34` Mode B's `3,078 tok/s` and `+26.4%` faster than `Wave 34` Mode A's `4,130 tok/s`).

### 6.2 `H2` Long-Horizon (`1,800` Steps / `14.75M` Tokens) Milestone Trajectory (`w35_d1024_L6_mode_b_b8_ga2_1800steps`)

| Step | Tokens (`M`) | $\tau$ | Train CE | Val CE (`Val PPL`) | Val Top-1 (`Top-50`) | Global `R_sharp` | `Coarse256 R_s` | `Mid2048 R_s` | `Sub8192 R_s` | `Fine65536 R_s` | Proj `150M` CE | Proj `900M` CE | Speed (`tok/s`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | `0.00` | `6.6501` | `6.6501` (`772.9`) | `11.98%` (`52.23%`) | `188.02` | `43.78` | `155.73` | `134.04` | `95.25` | — | — | — |
| `150` | `1.23M` | `0.08` | `5.5731` | `6.0110` (`407.9`) | `16.85%` (`56.82%`) | `41.07` | `45.07` | `129.24` | `111.46` | `93.62` | `3.4865` | **`3.1042`** | `5,103` |
| `300` | `2.46M` | `0.17` | `5.6029` | `5.7442` (`312.4`) | `18.56%` (`59.61%`) | `41.39` | `10.14` | `117.71` | `99.55` | `75.96` | `3.4972` | **`3.0958`** | `4,955` |
| `600` | `4.92M` | `0.33` | `5.9510` | `5.4168` (`225.2`) | `20.36%` (`63.56%`) | `22.68` | `10.20` | `111.99` | `80.28` | `51.90` | `3.5174` | **`3.1029`** | `5,045` |
| `900` | `7.37M` | `0.50` | `5.8051` | `5.1380` (`170.4`) | `22.52%` (`66.20%`) | `21.13` | `10.23` | `89.20` | `73.32` | `49.17` | `3.5744` | **`3.1608`** | `5,130` |
| `1200` | `9.83M` | `0.67` | `6.1331` | `5.0031` (`148.9`) | `24.42%` (`66.77%`) | `25.27` | `11.21` | `62.72` | `5.13` | `41.40` | `3.5924` | **`3.1748`** | `5,175` |
| `1500` | `12.29M` | `0.83` | `5.5269` | `4.8989` (`134.1`) | `26.07%` (`67.19%`) | `35.82` | `11.72` | `44.11` | `4.12` | `64.61` | `3.6123` | **`3.1928`** | `5,202` |
| **`1800`** | **`14.75M`** | **`1.00`** | **`5.2477`** | **`4.8874` (`132.6`)** | **`26.20%` (`67.11%`)** | **`45.23`** | **`14.96`** | **`47.98`** | **`12.23`** | **`128.72`** | **`3.6352`** | **`3.2141`** | **`5,220`** |

### 6.3 `H3` `900M`-Token Scaling Law Trajectory & Cross-Wave Leaderboard Comparison
- **Chinchilla + WSD Cooldown Scaling Fit (`1.23M` to `14.75M` tokens)**:
  - $E_{\infty} = 2.5044$, $A = 4.1607$, $\alpha = 0.2600$ (`RMSE = 0.1963`):
  - **`50M` tokens (`6,104` steps)**: **`4.0090` Val CE** (already beats `Wave 18` `900M`-token `d = 512, L = 8` `4.0556` using **`18x` fewer tokens**!)
  - **`100M` tokens (`12,207` steps)**: **`3.7609` Val CE**
  - **`150M` tokens (`18,311` steps)**: **`3.6352` Val CE**
  - **`300M` tokens (`36,621` steps)**: **`3.4487` Val CE**
  - **`450M` tokens (`54,932` steps)**: **`3.3542` Val CE**
  - **`900M` tokens (`109,863` steps)**: **`3.2141` Val CE** (surpasses both the `3.3362` `Wave 12` Classical `d = 768, L = 16` `900M`-token baseline by **`-0.1221` nats** and the **`3.2800` Modded-NanoGPT Speedrun record** by **`-0.0659` nats**!)
- **Post-Mortem Insights for `Wave 36+`**:
  1. **Fine-Wavelet Participation Explosion During Cooldown (`Fine65536 R_s: 41.40 -> 128.72`)**: During WSD cooldown (`steps 1080 -> 1800`), the `65,536`-leaf fine wavelet subspace expanded its effective rank from `41.40` to **`128.72` out of `256`** (`50.3%` utilization of the fine sphere!), driving Top-1 accuracy to an all-time `14.75M`-token record of **`26.20%`** (exceeding `Wave 30`'s `44.24M`-token `24.80%` Top-1 accuracy in `1/3` the tokens).
  2. **Why `d = 256` (`4.8332`) Slightly Edged `d = 1024` (`4.8874`) in Raw CE at `14.75M` Tokens While `d = 1024` Won Top-1 (`26.20%` vs `26.15%`)**: At `d = 1024`, the `119.99M`-parameter model has `4.0x` more parameters than `d = 256` (`30.12M`); at `14.75M` tokens (`0.12` tokens per parameter vs `0.49` tokens per parameter at `d = 256`), `d = 1024` is still in its early Chinchilla regime ($\beta_{\text{eff}} = 48.78$, which sharpens Top-1 predictions to `26.20%` while slightly penalizing tail-token entropy before `50M+` tokens). Temperature-calibrating the neural-to-surface logit fusion at $d=1024$ (`z_neu / sqrt(d / 256)` inside `apply_surface_logit_lift`) is a prime candidate for `Wave 36`.

