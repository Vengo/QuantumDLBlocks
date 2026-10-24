# Experiment Plan 041 (`Wave 26`): 15 GB RTX 5060 Ti `bfloat16`/`fp8` Optimization, Solution A + B (`Convex-Peak Traceless` + `Dyadic Coarse-to-Fine Born`), and `24,576`-Token Batch Network Scaling on `FineWeb-Edu`

## 1. Context & Motivation

Following `Experiment Plan 040` (`Wave 25`), where `Convex-Peak Traceless Born` (`Solution A`) achieved a new `3,600`-step (`29.49M`-token) `FineWeb-Edu` project record of **`5.0162` Validation Cross-Entropy** (`25.25%` Top-1, `25,616 tok/s`, `19.19 min`) under an `8.0 GB` VRAM cap at `B_tok = 8,192` (`B = 16, T = 512`), we identified three high-leverage opportunities to unlock the full compute and memory capacity of the local `NVIDIA GeForce RTX 5060 Ti` (`15.48 GiB` total VRAM):

1. **Consistent `bfloat16` Execution, SRAM-Tiled Flash-Born Attention, & Activation Memory Optimization (`<= 15.0 GB` VRAM)**:
   - Auditing `Radical2Pow16WaveletCholeskyGPT` under `torch.amp.autocast("cuda", dtype=torch.bfloat16)` revealed several silent `float32` promotions and activation memory bottlenecks:
     - PyTorch's autocast `.sum()` rule promoted `overlap = (m_base * psi_self).sum(dim=-1, keepdim=True)` in `Tied2Pow16WaveletCholeskyPyramid._synthesize_raw_detectors` to `float32`, causing `m_sharp` and `m_diff` (`(50304, d)`) to be materialized in `float32`.
     - 1D `float32` gate parameters (`attn_gate`, `fused_gate`, `bigram_embed_gate`, `bigram_mid_gate`, `unet_skip_gates`, `diff_accum_gate`) promoted residual wavefunctions `psi` from `bfloat16` to `float32`.
     - `SubspaceProjectorBornAttention` inside `Radical2Pow16WaveletCholeskyGPT` blocks had `use_flash_born = False` (default), falling back to `_memory_efficient_purified_born_attn` instead of the SRAM-tiled Triton kernel `flash_purified_born_attention`.
     - `OvercompleteTwoBeamInterference` unconditionally computed the second-beam branch (`b2`, `out_2`, `psi_beam2`) even when `return_second_beam=False` and had a typo (`wavelet_branch` vs. `wavelet_cholesky_branch`) in `FusedKroneckerWaveletUNetBlock.__init__`.
     - `l2_normalize` saved duplicate `float32` copies (`x_f` and `x_f * rsqrt`) across 15+ normalization calls per layer.
     - `BlockCholeskyLinear.__init__` required `32 % block_size == 0`, preventing `embed_dim = 384` (`coarse_dim = 48`). Using `math.gcd(self.dim, min(int(max_block_size), self.dim))` supports `embed_dim = 384` (`block_size = 16`) cleanly.
2. **Combining Solution A (`convex_traceless_born`) + Solution B (`dyadic_hierarchical_born`) in `Radical2Pow16WaveletCholeskyGPT`**:
   - In `Radical2Pow16WaveletCholeskyGPT`, the $N = 2^{16} = 65,536$ vocabulary tree already partitions all $50,257$ tokens into $256$ spectral coarse roots (`wav_256`, indexed by $\ell \gg 8$ via `perm_v_to_leaf >> 8`).
   - Reinforcing each fine token's sharp detector with its normalized coarse superclass root (`dyadic_coarse_alpha = 0.22`) before self-loop orthogonalization and adding a `256`-way coarse convex-traceless Born auxiliary loss (`compute_dyadic_coarse_aux_loss`, `dyadic_coarse_ce_weight = 0.08`) combines `Solution A` and `Solution B` directly on `FineWeb-Edu`.
3. **Expanding to `15.0 GB` VRAM (`vram_fraction = 0.95`), `24,576`-Token Batch Size (`B = 48, T = 512`), `bf16` vs. `fp8` vs. `int8` Readout GEMMs, and Scaling the Network (`d = 384, L = 8`)**:
   - Increasing the token batch size $3\times$ from `B_tok = 8,192` (`B = 16, T = 512`) to **`B_tok = 24,576` (`B = 48, T = 512`)** increases Tensor Core arithmetic intensity across the vocabulary readout and backbone projections.
   - Benchmarking `bf16` (`torch.mm`), `fp8` (`torch._scaled_mm` with `torch.float8_e4m3fn`), and `int8` (`torch._int_mm`) in `_DirectChunkedTwoBeamKBosonCEFn.forward` determines the fastest hardware execution path on Blackwell (`RTX 5060 Ti`, compute capability `12.0`).

---

## 2. Falsifiable Hypotheses (`H1`, `H2`, `H3`)

- **`H1` (`bfloat16` Consistency, `fp8` vs. `int8` vs. `bf16` Readout GEMM Benchmarking, & `24,576`-Token Batch Sizing on `15.0 GB` RTX 5060 Ti)**:
  1. Under `torch.amp.autocast("cuda", dtype=torch.bfloat16)`, all 7 core activation tensors (`psi_in`, `bigram_psi`, `v_0`, `psi_blk`, `diff_blk`, `m_sharp`, `m_diff`) remain strictly in `torch.bfloat16`.
  2. At `B_tok = 24,576` (`B = 48, T = 512`), all candidate scales from `d = 256, L = 6` (`10.64M` params, `5.25 GB` VRAM) through `d = 384, L = 8` (`18.49M` params, `7.79 GB`–`8.21 GB` VRAM), `d = 512, L = 8` (`28.27M` params, `9.61 GB`–`10.04 GB` VRAM), and `d = 512, L = 12` (`37.44M` params, `12.66 GB` VRAM) execute forward + backward + `StiefelMuon`/`AdamW` steps with `status == "OK"` and `peak_vram_gb <= 15.0`.
  3. At `B_tok = 24,576` (`d = 256, L = 6`), `bf16` (`40,923 tok/s`) and `fp8` (`43,069 tok/s`) achieve `>= 1.55x` higher throughput than `B_tok = 8,192` (`25,203 tok/s`), and `fp8` (`torch._scaled_mm`) outperforms both `bf16` and `int8` (`33,917 tok/s`).
- **`H2` (Controlled `FineWeb-Edu` Ablation of `Solution A + B` vs. `Solution A` Alone)**:
  - At matched `900`-step (`7.37M`-token, `B = 16, T = 512`) budget on genuine `FineWeb-Edu`, `w26_15gb_dyadic_convex_traceless_b16_900steps` (`Solution A + B`: `use_convex_traceless_born=True` + `use_dyadic_hierarchical_born=True`) achieves lower Validation Cross-Entropy than `Wave 25` (`w25_8gb_convex_traceless_born_900steps`: `5.4465` Val CE) and `Wave 21` (`5.4734` Val CE), with `kraus_stiefel_err <= 0.05`.
- **`H3` (`24,576`-Token Batch & Scaled `d = 384, L = 8` Network Scaling on `FineWeb-Edu` Breaking `5.0162` Val CE)**:
  - Training at `B_tok = 24,576` (`B = 48, T = 512`) on genuine `FineWeb-Edu` under the `15.0 GB` VRAM cap:
    1. `w26_15gb_b48_d256_L6_dyadic_fp8_1200steps` (`10.64M` params, `1,200` steps = `29.49M` tokens, `fp8` readout) achieves `>= 36,000 tok/s` average training throughput (completing `29.49M` tokens in `< 14 minutes` vs. `19.19 min` in `Wave 25` and `40.30 min` in `Wave 21`).
    2. Scaling the network to `w26_15gb_b48_d384_L8_dyadic_fp8_1200steps` (`d = 384, L = 8, H = 6, mid_detail_dim = 96`, `18.49M` params) at `B_tok = 24,576` breaks the all-time `FineWeb-Edu` `29.49M`-token Validation Cross-Entropy record (`< 5.0162` nats, targeting `< 5.00` nats) while maintaining `peak_vram_gb <= 15.0` and `kraus_stiefel_err <= 0.05`.

---

## 3. Planned Architectural & Hardware Changes

1. [`src/qdlblocks/functional.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/functional.py):
   - Add `_L2NormalizeFn(torch.autograd.Function)` for `bfloat16`/`float16` tensors outside `torch.compile` so `l2_normalize` saves only `y` (in `x.dtype`) and `r_inv` (`(..., 1)`), eliminating two duplicate `(B, T, D)` `float32` saved buffers per normalization call.
2. [`src/qdlblocks/layers/wavelet_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/wavelet_cholesky.py):
   - Use `math.gcd(self.dim, min(int(max_block_size), self.dim))` in `BlockCholeskyLinear.__init__` so `embed_dim = 384` (`coarse_dim = 48, block_size = 16`) is supported cleanly.
   - Preserve `torch.bfloat16` under `torch.amp.autocast` in `BlockCholeskyLinear.forward` and `WaveletCholeskyMultiscaleBranch._work_dtype`, and checkpoint `_vcycle_impl` when `psi.numel() >= 4_194_304`.
3. [`src/qdlblocks/layers/overcomplete_two_beam.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_two_beam.py):
   - Skip second-beam synthesis when `return_second_beam=False`, checkpoint `_mz_phase_elementwise` when `z1_raw.numel() >= 4_194_304`, and keep `OvercompleteTwoBeamInterference.__init__` within the `<= 8` argument ceiling.
4. [`src/qdlblocks/layers/direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py):
   - Add `gemm_precision: str = "bf16"` supporting `"bf16"`, `"fp8"` (`torch._scaled_mm` with `torch.float8_e4m3fn`), and `"int8"` (`torch._int_mm`), use `chunk_sz = 2048` on GPUs with `>= 14 GiB` VRAM, and wrap the compiled chunk loop in `with torch.no_grad():`.
5. [`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py):
   - Add `use_dyadic_hierarchical_born`, `dyadic_coarse_alpha`, `dyadic_coarse_ce_weight`, and `readout_gemm_precision` to `Radical2Pow16Config`.
   - Implement coarse-root detector reinforcement in `_synthesize_raw_detectors` and `compute_dyadic_coarse_aux_loss` in `Tied2Pow16WaveletCholeskyPyramid`.
   - Enable `use_flash_born = fast_chol` across all block types and checkpoint `_wavelet_kronecker_pathway` in `FusedKroneckerWaveletUNetBlock` when `psi_a.numel() >= 4_194_304`.

---

## 4. Verification & Test Plan

1. **Network Sizer & Hardware Telemetry (`scripts/rtx5060ti_15gb_batch24576_network_sizer.py`)**:
   - Generates `.cache/rtx5060ti_15gb_batch24576_network_sizer.json` verifying `bfloat16` activations and benchmarking all 10 candidate configurations across `bf16`, `fp8`, and `int8` at `B_tok = 24,576`.
2. **End-to-End `Wave 26` Verification Runner (`examples/51_verify_plan041_15gb_batch24576_dyadic_scaling.py`)**:
   - Executes `H1`, `H2`, and `H3` and writes `.cache/hypothesis_experiments_041.json`.
3. **Unit & Hypothesis Test Suites**:
   - `tests/test_plan041_15gb_batch24576_dyadic_scaling.py` and `tests/test_hypothesis_041_15gb_batch24576_dyadic_scaling.py`.

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Gate)

1. **Existing Scripts Used**:
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py`
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/imagenet1k_quantum_block_ablation_profiler.py`
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/fineweb_convex_vs_concave_slope_tracker.py`
2. **`Build-Before-Launch` Scripts Built for Plan 041**:
   - [`scripts/rtx5060ti_15gb_batch24576_network_sizer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/rtx5060ti_15gb_batch24576_network_sizer.py): Automated `bfloat16` activation dtype verifier and `24,576`-token batch (`B = 48, T = 512`) throughput/VRAM profiler across `d in {256, 384, 512}` and `bf16` / `fp8` / `int8` readout GEMMs under `torch.cuda.set_per_process_memory_fraction(0.95, device=0)`.
3. **`Nice-to-Have` Candidate Scripts Logged in `scripts/TOOLING_BACKLOG.md`**:
   - `scripts/dyadic_coarse_to_fine_entropy_profiler.py`: Decomposes `FineWeb-Edu` token Cross-Entropy (`50,257` classes) into 8-bit coarse dyadic subtree entropy (`256` roots) + 8-bit conditional intra-subtree leaf entropy (`256` leaves/root).
   - `scripts/fp8_vs_bf16_born_logit_drift_auditor.py`: Measures cosine similarity, max logit error, and KL divergence between `fp8` (`torch._scaled_mm`) and `bf16` (`torch.mm`) Born-Boltzmann readouts across training checkpoints.
