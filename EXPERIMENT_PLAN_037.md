# Experiment Plan 037: Banked $N = 2^{16}$ Subtree Isometric Lifts (Fix 2), Kraus Stiefel Retraction + $\lambda_{\text{diff}}$ Annealing (Fix 3), & Fast $C = L L^\top$ / TensorCore Readout Acceleration (Fix 4)

## 1. Context & Mathematical Motivation

### 1.1 Why Training Cannot Push $V = 50,257$ Unrelated Words to $c_{t,v} = 0$ When `sharp_eff_rank` Is Low (Parseval's Identity on $\mathbb{RP}^{d-1}$)
When token probabilities depend on squared inner products $c_{t,v}^2 = |\langle \psi_t \mid m_v \rangle|^2$, each detector $m_v \in \mathbb{S}^{d-1}$ defines a point in Real Projective Space $\mathbb{RP}^{d-1}$ where $+m_v$ and $-m_v$ are identical.
By **Parseval's Identity / Tight Frame Conservation**, for any set of $V = 50,257$ unit vectors $\{m_v\}_{v=1}^V$ spanning an effective dimension $d_{\text{eff}} \le d$, the trace of the frame operator $G = \sum_{v=1}^V m_v m_v^\top$ is strictly $\operatorname{Tr}(G) = \sum_{v=1}^V \|m_v\|_2^2 = V = 50,257$. Consequently, for a typical unit state $\psi_t \in \mathbb{S}^{d-1}$, the sum of squared overlaps across the vocabulary is conserved around:
$$\sum_{v=1}^{V} c_{t,v}^2 \approx \frac{V}{d_{\text{eff}}}$$
- In Plan 036 (`w21`), because `wav_65536` (`65,536 x 64`) was lifted to $\mathbb{R}^{256}$ by a **single** $64 \times 256$ matrix `up_65536`, all fine-scale leaf distinctions were confined to a `64`-D subspace (`sharp_eff_rank = 64.0`), forcing $\sum_{v=1}^V c_{t,v}^2 \approx 50,257 / 64 = \mathbf{785.3}$!
- Furthermore, because $E(F) = k \ln(1 + \beta F)$ is concave in $F = c_{t,v}^2$ and its derivative $\partial E / \partial c_{t,v} = 2 k \beta c_{t,v} / (1 + \beta c_{t,v}^2)$ vanishes linearly as $c_{t,v} \to 0$, the mandatory Parseval background budget $\sum_{v \ne y_t} c_{t,v}^2 \approx V / d_{\text{eff}}$ is spread across thousands of unrelated tokens, inflating the partition function $Z_t$.

### 1.2 The Three Targeted Fixes (`Fix 2`, `Fix 3`, and `Fix 4`)
1. **Fix 2 — Banked `16 x 64 x 256` Dyadic Subtree Isometric Lifts (`up_65536_banks`)**:
   - Partition the $N = 2^{16} = 65,536$ dyadic leaves into `16` top-level 4-bit dyadic subtrees (`bank_id = p >> 12`, `4,096` leaves per subtree) and equip each subtree with its own $64 \times 256$ Stiefel-initialized projection `up_65536_banks[b]` (initialized via SVD of each `4,096`-leaf subtree's residual `r65536_full[b*4096:(b+1)*4096]`).
   - This expands the effective rank of the sharp vocabulary detectors from `sharp_eff_rank = 64.0` to **`>= 190.0`** (spanning the full `256`-D ambient space and cutting the Parseval tail budget $V / d_{\text{eff}}$ by **`> 3x`**) while adding only `+0.25M` parameters (`10.40M -> 10.64M`).
2. **Fix 3 — Periodic Stiefel Retraction on `kraus_generators` + Cosine-Annealed $\lambda_{\text{diff}}(s)$ Schedule (`0.17 -> 0.055`) + Higher `logit_softcap` (`24.0`)**:
   - In `w21`, `pyr.kraus_generators` drifted off the Stiefel manifold (`kraus_stiefel_err = 20.70`) and `lambda_diff` rose from `0.129` to `0.189` at Step 3,600, pouring `19%` of late-training energy into an un-retracted diffuse bath.
   - Fix 3 adds `pyramid.retract_kraus_generators()` (batched QR retraction with positive-diagonal sign alignment every `retract_every` steps, keeping `kraus_stiefel_err <= 0.05`) and a **Warmup-Hold-Cosine-Anneal $\lambda_{\text{diff}}(s)$ schedule** (`0.15 -> 0.17 -> 0.055`) so the diffuse bath accelerates early learning (`Steps 0–1200`) and then sharpens tail probabilities in the second half of training (`Steps 1200–3600`).
3. **Fix 4 — Fast $C = L L^\top$ Block-Cholesky Contraction + Pre-Gather Subtree-Batched Detector Synthesis + `bfloat16` TensorCore Readout (`use_fast_cholesky_and_compile`)**:
   - **Why `w21` ran at `11.5k tok/s`**:
     1. Each of the `61` `BlockCholeskyLinear` modules multiplied the large `(B, T, num_blocks, 32)` activation tensor twice (`(x @ L) @ L^T`). Pre-forming the tiny `(num_blocks, 32, 32)` SPD block matrix $C = L L^\top$ (`< 1 us`) and executing a single `bfloat16` contraction `x @ C` cuts activation GEMMs and backward saved tensors in half while computing the exact same mathematical operator.
     2. In `synthesize_vocab_detectors()`, projecting `wav_8192` (`8,192` rows instead of `50,304` rows) before `index_select` and projecting `wav_65536` (`16 x 4096 x 64`) via a single batched `bfloat16` `torch.bmm` against `up_65536_banks` (`16 x 64 x 256`) eliminates `84%` of the redundant vocabulary-projection FLOPs.
     3. Performing the `direct_chunked_two_beam_kboson_ce` chunk inner products `(B*T, d) @ (d, chunk)` in `bfloat16` TensorCores before casting cosines to `float32` for log-sum-exp, combined with `torch.compile` on the middle block stack, eliminates the FP32 GEMM bottleneck.

---

## 2. Falsifiable Hypotheses (`H1`, `H2`, `H3`)

### Hypothesis 1 (`H1` — Fix 2 + Fix 3 + Fix 4 Mathematical Equivalence, `sharp_eff_rank >= 190`, `kraus_stiefel_err <= 0.05`, & `< 11.0M` Parameter Budget)
- **Claim**: Upgrading [`Radical2Pow16WaveletCholeskyGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py) with **Fix 2** (`num_fine_up_banks = 16`), **Fix 3** (`retract_kraus_generators` + cosine-annealed $\lambda_{\text{diff}}(s)$), and toggleable **Fix 4** (`use_fast_cholesky_and_compile: bool`) will:
  1. Keep total trainable parameters strictly **`< 11,000,000` (`10.64M` params, `3.83x` smaller than `40.78M` `QuantumBornGPT`)**,
  2. Raise Step-0 `sharp_eff_rank` of the synthesized `(V, 256)` vocabulary detectors from `64.0` (Plan 036) to **`>= 190.0`** (`> 3.0x` increase in effective detector rank, reducing Parseval tail overlap $\mathbb{E}_{u \ne v}[c_{u,v}^2]$ by `> 2.5x`),
  3. Keep `kraus_stiefel_err <= 0.05` after optimizer steps + `retract_kraus_generators()`, and
  4. Produce matching forward logits/losses between `use_fast_cholesky_and_compile=False` (without Fix 4) and `use_fast_cholesky_and_compile=True` (with Fix 4) within `bfloat16` numerical tolerance (`max_rel_loss_diff < 2e-3`).

### Hypothesis 2 (`H2` — 900-Step / `7.37M`-Token Head-to-Head Comparison: Fixes 2+3 *Without* Fix 4 vs. Fixes 2+3+4 *With* Fix 4)
- **Claim**: Over `900` steps (`7.37M` tokens, $B=16, T=512$, `<= 8.0 GB` VRAM) on genuine FineWeb-Edu (`fineweb_train_000001.bin` $\to$ `fineweb_val_000000.bin`):
  1. **Both** **`w22_8gb_fixes23_no_fix4_900steps` (Fixes 2+3 without Fix 4)** and **`w22_8gb_fixes234_with_fix4_900steps` (Fixes 2+3+4 with Fix 4)** will achieve lower (better) validation cross-entropy than Plan 036's `w21` (`5.4734`), Plan 035's `w20` (`5.5253`), `40.78M` `QuantumBornGPT` (`5.6092`), and `53.54M` `ClassicalModdedGPT` (`5.9152`), while maintaining `sharp_eff_rank >= 180.0` and `kraus_stiefel_err <= 0.05`.
  2. **`w22_8gb_fixes234_with_fix4_900steps` (With Fix 4)** will achieve **`>= 1.60x` higher training throughput (`tok/s`)** than **`w22_8gb_fixes23_no_fix4_900steps` (Without Fix 4)** while matching its final validation cross-entropy within `<= 0.02 nats`.

### Hypothesis 3 (`H3` — 3,600-Step / `29.49M`-Token Extended Scaling Run With Fixes 2+3+4 Breaking `< 5.00` Val CE)
- **Claim**: Over `3,600` steps (`29.49M` tokens, evaluated at steps `0, 300, 900, 1800, 2700, 3600`) on genuine FineWeb-Edu under the `<= 8.0 GB` VRAM cap:
  - **`w22_8gb_fixes234_with_fix4_3600steps` (`10.64M` params)** will break the **`5.00` Val CE barrier (`< 5.00` Val CE at Step 3600)** — beating Plan 036's `w21` (`5.0211`), Plan 035's `w20` (`5.0858`), `w19` WavChol+MZ (`4.9873`), `38.28M` Parity-Split `QuantumBornGPT` (`5.1157`), and `30.95M` `ClassicalModdedGPT` (`5.5202`), while achieving higher `Top-50` accuracy (`> 65.2%` vs. `w21`'s `64.70%`), `sharp_eff_rank >= 180.0`, and `kraus_stiefel_err <= 0.05`.

---

## 3. Planned Architectural & Code Changes

1. **[`src/qdlblocks/layers/wavelet_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky.py)**:
   - Add `fast_spd_bmm: bool = False` option to `BlockCholeskyLinear`: when `True`, computes `c_mat = torch.bmm(l_mat, l_mat.transpose(-1, -2)).to(x.dtype)` on the small `(num_blocks, block_size, block_size)` Cholesky factor and applies a single contraction `torch.einsum("...gb,gbi->...gi", x_blk, c_mat)` instead of two sequential FP32 contractions.
2. **[`src/qdlblocks/layers/direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py)**:
   - Add `use_bf16_gemm: bool = False` and `logit_softcap: float = 15.0` support to `direct_chunked_two_beam_kboson_ce` and `compute_two_beam_kboson_logits` so chunk inner products `(psi_sharp @ m_s_chunk.t())` execute on `bfloat16` TensorCores when enabled while keeping log-sum-exp in `float32`.
3. **[`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py)**:
   - Add **Fix 2** (`num_fine_up_banks: int = 16`, `up_65536_banks` of shape `(16, 64, 256)`, per-subtree SVD initialization in `initialize_radical_2pow16_with_ppmi_svd`).
   - Add **Fix 3** (`retract_kraus_generators()`, `set_training_progress(progress)`, cosine-annealed $\lambda_{\text{diff}}(s)$ schedule from `lambda_diff_start=0.16` to `lambda_diff_end=0.055`, and `logit_softcap=24.0`).
   - Add **Fix 4** (`use_fast_cholesky_and_compile: bool = False` toggle enabling `fast_spd_bmm=True`, pre-gather `wav_8192` projection, batched `16 x 4096` `bmm` for `wav_65536`, and `use_bf16_gemm=True`).
4. **Promoted CLI Diagnostic Scripts (`scripts/`)**:
   - [`scripts/dyadic_bigram_transition_heatmap.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/dyadic_bigram_transition_heatmap.py): Audits the $256 \times 256$ coarse dyadic transition matrix $T_{\text{coarse}}$ and `2,048`-bucket predecessor wavelet states `pred_wav_2048`, comparing Zipfian-centered vs. uncentered predecessor bigram specificity.
   - [`scripts/wavelet_scale_ablation_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wavelet_scale_ablation_probe.py): Profiles scale-by-scale contribution (`256`, `2048`, `8192`, `65536` with `1` vs. `16` fine banks) to `sharp_eff_rank`, Parseval background overlap $\mathbb{E}_{u \ne v}[c_{u,v}^2]$, and detector synthesis throughput (`ms / step` with and without Fix 4).
5. **Verification Script & Tests**:
   - [`examples/47_verify_plan037_tail_fixes_and_fast_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/47_verify_plan037_tail_fixes_and_fast_cholesky.py)
   - [`tests/test_plan037_tail_fixes_and_tooling.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan037_tail_fixes_and_tooling.py) & [`tests/test_hypothesis_037_tail_fixes_and_fast_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_037_tail_fixes_and_fast_cholesky.py).

---

## 4. Verification & Testing Plan

1. **AST & Best-Practices Gate**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/layers/wavelet_cholesky.py src/qdlblocks/layers/direct_chunked_ce.py src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py scripts/dyadic_bigram_transition_heatmap.py scripts/wavelet_scale_ablation_probe.py examples/47_verify_plan037_tail_fixes_and_fast_cholesky.py
   ```
2. **Unit & Hypothesis Test Suite**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/pytest tests/test_plan037_tail_fixes_and_tooling.py tests/test_hypothesis_037_tail_fixes_and_fast_cholesky.py -v
   ```

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

1. **Existing Scripts Used**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
   - [`scripts/dyadic_spectral_tree_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/dyadic_spectral_tree_inspector.py)
   - [`scripts/cholesky_band_sparsity_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/cholesky_band_sparsity_profiler.py)
2. **`Build-Before-Launch` Scripts (Promoted from `scripts/BACKLOG.md`)**:
   - [`scripts/dyadic_bigram_transition_heatmap.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/dyadic_bigram_transition_heatmap.py)
   - [`scripts/wavelet_scale_ablation_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wavelet_scale_ablation_probe.py)
3. **`Nice-to-Have` Scripts (Logged in `scripts/BACKLOG.md`)**:
   - `scripts/parseval_tail_overlap_profiler.py`: Measures the empirical Parseval frame spectrum $\lambda_1 \ge \dots \ge \lambda_d$ of $G = \sum_{v=1}^V m_v m_v^\top$ and the tail partition-function fraction $Z_{\text{tail}} / Z_{\text{total}}$ across rank buckets (`1`, `2–5`, `6–50`, `51–1000`, `>1000`).
   - `scripts/block_cholesky_kernel_benchmark.py`: Benchmarks two-pass `x @ L @ L^T` vs. single-pass fused `x @ (L L^T)` across batch sizes, sequence lengths, and `bfloat16`/`float32` dtypes.
