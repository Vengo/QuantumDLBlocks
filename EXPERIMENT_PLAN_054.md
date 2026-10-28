# EXPERIMENT PLAN 054: Extending Triton Flash-Krylov to Arbitrary Multiples of 32 ($d_{\text{complex}} \in \{32, 64, 96, 128, 160, 192\}$), 5-Minute Wall-Clock Tournament, & 900M-Token Production Run (`Wave 39`)

**Date**: 2026-10-25
**Wave**: `Wave 39` (`W39`)
**Plan ID**: `EXPERIMENT_PLAN_054`
**Target Hardware**: Local Workstation `NVIDIA GeForce RTX 5060 Ti` (`16 GB` / `15,941 MiB` VRAM, Blackwell `sm_120`)

---

## 1. Motivation & Mathematical Opportunity

In `Wave 38` ([`EXPERIMENT_PLAN_053.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_053.md)), our custom Triton & Rank-1 Associative Krylov kernels ([`flash_complex_krylov_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_complex_krylov_triton.py)) cut `GeminiPackedComplexRoPEGPT` VRAM in half (`2,727.7 MiB` at $d_{\text{complex}} = 128$) and boosted training throughput by `2.52x` (`20,068 tok/s`).

However, two major questions and optimization opportunities emerge when optimizing for **fixed wall-clock training (a 5-minute sprint)** and scaling the winner to **900M tokens**:

1. **Extending Triton Norm-RoPE & Flash-Krylov Kernels to Smaller $d_{\text{complex}}$ Multiples of 32 ($d_{\text{complex}} \in \{32, 64, 96, 128, 160, 192\}$) and Flexible Register Widths $d_h \in \{16, 32\}$**:
   - In `Wave 38`, `_fused_complex_norm_rope_fwd_kernel` and `_fused_complex_norm_rope_bwd_kernel` were restricted to `d_reg == 32`. Consequently:
     * Register dimensions $d_h = 16$ (needed for multi-register $d_{\text{complex}} = 32 = 2 \times 16$ or $d_{\text{complex}} = 64 = 4 \times 16$ or $d_{\text{complex}} = 96 = 6 \times 16$) fell back to unfused PyTorch.
     * Even more importantly, all full-backbone residual normalizations `complex_l2_normalize(psi_re, psi_im, dim=-1)` and FFN hidden normalizations of width $d \in \{32, 64, 96, 128, 160, 192, 256, 384, 512\}$ (`2L + 3` calls per step) fell back to unfused PyTorch because `d != 32`.
   - By adding masked power-of-2 column tiling (`BLOCK_D = triton.next_power_of_2(d_dim)`, `mask_d = offs_d < d_dim`) to `_fused_complex_norm_rope_fwd_kernel` and `_fused_complex_norm_rope_bwd_kernel`, **every single complex normalization and rotary embedding** across both register widths $d_h \in \{8, 16, 24, 32, 64\}$ and full model widths $d_{\text{complex}} \in \{32, 64, 96, 128, 160, 192, 256, 384, 512, 1024\}$ executes in one fused `@triton.jit` SRAM pass.

2. **Scale-Invariant Krylov Normalization Identity (Eliminating the $O(T^3)$ `w_gemm @ g_sq` Gram Matrix Multiplication)**:
   - In `Wave 38`'s `factored_complex_krylov_step`, lines 475–480 computed the exact McWeeny trace $\operatorname{Tr}(\rho_t^2) = \sum_{s, s' \le t} w_{ts} w_{ts'} |\langle v_s \mid v_{s'} \rangle_{\mathbb{C}}|^2$ via three $(T \times T)$ matrix multiplications (`g_re`, `g_im`, and `wg = torch.matmul(w_gemm, g_sq)`), where `w_gemm @ g_sq` is a $(512 \times 512) \times (512 \times 512)$ GEMM costing $2 T^3 = 2.68 \times 10^8$ FLOPs per register (over `50%` of the entire attention layer's FLOPs!).
   - Yet when `normalize_krylov_ket=True` (the canonical setting in `GeminiPackedComplexRoPEGPT`), the output $|u_t^{\text{raw}}\rangle = \frac{|z_t\rangle}{\operatorname{Tr}(\rho_t^2) + \varepsilon}$ is immediately projected onto the unit complex sphere $\mathbb{C}\mathbb{S}^{d_h - 1}$:
     $$|u_t\rangle = \operatorname{normalize}_{\mathbb{C}}\!\left(\frac{\rho_t^2 |\bar{v}_t\rangle}{\operatorname{Tr}(\rho_t^2)}\right) \equiv \frac{\rho_t^2 |\bar{v}_t\rangle}{\|\rho_t^2 |\bar{v}_t\rangle\|_2}$$
     Because $\operatorname{Tr}(\rho_t^2) > 0$ is a positive scalar per $(b, h, t)$, dividing by $\operatorname{Tr}(\rho_t^2)$ prior to unit-sphere normalization cancels out identically in both the forward pass and the tangent-space backward pass ($\frac{\partial}{\partial c}\operatorname{normalize}(c z) \equiv 0$)!
   - By normalizing $|z_t\rangle = \rho_t^2 |\bar{v}_t\rangle$ directly via `fused_complex_norm_rope_triton` when `normalize_output=True` (and recording the $O(T d_h)$ Rayleigh purity $\|y_t^{(1)}\|_2^2 = \langle \bar{v}_t \mid \rho_t^2 \mid \bar{v}_t \rangle$ for telemetry), we eliminate **all three $(T \times T)$ Gram matrix multiplications** and combine the 4 real/imag Krylov value GEMMs into 2 concatenated Tensor-Core GEMMs (`y1_cat = wa1_re @ v_cat + wa1_im @ v_skew`).

3. **The 5-Minute Wall-Clock Pareto Tradeoff Across $d_{\text{complex}} \in \{32, 64, 96, 128, 160, 192\}$**:
   - In a fixed **5-minute (`300s`) wall-clock budget**, smaller $d_{\text{complex}}$ models process significantly more tokens (`tok/s` $\times 300\text{s}$) at the cost of fewer complex registers $H$, whereas wider models have higher per-token capacity but see fewer tokens in 5 minutes.
   - Testing $d_{\text{complex}} \in \{32, 64, 96, 128, 160, 192\}$ (all multiples of 32) head-to-head in a calibrated 5-minute tournament identifies the true wall-clock Pareto winner on the RTX 5060 Ti, which we then launch for a **`900M`-token (`0.9B` tokens) long-horizon run**.

---

## 2. Observable & Quantitatively Falsifiable Hypotheses

### Hypothesis `H1` (Arbitrary-Width Masked Triton Norm-RoPE & Scale-Invariant Krylov Fast-Path Parity)
- **Independent Variable**:
  1. Comparing `fused_complex_norm_rope_triton` against PyTorch reference across both power-of-2 and non-power-of-2 complex dimensions $D \in \{16, 24, 32, 64, 96, 128, 160, 192, 256\}$.
  2. Comparing the scale-invariant `flash_complex_krylov_attention(..., normalize_output=True)` fast-path (which skips the $O(T^3)$ `w_gemm @ g_sq` McWeeny trace division and normalizes $\rho_t^2 |\bar{v}_t\rangle$ directly onto $\mathbb{C}\mathbb{S}^{d_h - 1}$) against the explicit `Wave 37` Hermitian matrix reference (`normalize_output=True`) in forward and backward passes across register dimensions $d_h \in \{16, 32\}$ and register counts $H \in \{2, 3, 4, 5, 6\}$.
- **Dependent Observables**:
  - `max_nonpow2_norm_rope_err_fp32`: Maximum absolute forward error of `fused_complex_norm_rope_triton` across $D \in \{16, 24, 32, 64, 96, 128, 160, 192, 256\}$ in `float32`.
  - `max_fast_krylov_unit_err_fp32`: Maximum absolute difference between `flash_complex_krylov_attention(..., normalize_output=True)` and the explicit Hermitian reference followed by unit-sphere normalization.
  - `min_fast_krylov_grad_cos`: Minimum cosine similarity of $(\nabla Q, \nabla K, \nabla V)$ between the scale-invariant `normalize_output=True` fast-path and the explicit Hermitian reference.
- **Quantitative Pass Criteria**:
  - `max_nonpow2_norm_rope_err_fp32 < 1e-6`.
  - `max_fast_krylov_unit_err_fp32 < 1e-5`.
  - `min_fast_krylov_grad_cos > 0.999`.

### Hypothesis `H2` (Throughput & Memory Scaling Across Multiples of 32: $d_{\text{complex}} \in \{32, 64, 96, 128, 160, 192\}$)
- **Independent Variable**: Benchmarking `GeminiPackedComplexRoPEGPT` ($L = 6, B = 16, T = 512$) across the 6 multiples of 32 via [`scripts/complex_krylov_rtx5060ti_vram_sizer.py --mode small_d_sweep`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complex_krylov_rtx5060ti_vram_sizer.py):
  - `d32c_H2_dh16_L6`: $d_{\text{complex}} = 32, H = 2, d_h = 16$ ($d_{\text{real}} = 64$)
  - `d64c_H2_dh32_L6`: $d_{\text{complex}} = 64, H = 2, d_h = 32$ ($d_{\text{real}} = 128$)
  - `d96c_H3_dh32_L6`: $d_{\text{complex}} = 96, H = 3, d_h = 32$ ($d_{\text{real}} = 192$)
  - `d128c_H4_dh32_L6`: $d_{\text{complex}} = 128, H = 4, d_h = 32$ ($d_{\text{real}} = 256$)
  - `d160c_H5_dh32_L6`: $d_{\text{complex}} = 160, H = 5, d_h = 32$ ($d_{\text{real}} = 320$)
  - `d192c_H6_dh32_L6`: $d_{\text{complex}} = 192, H = 6, d_h = 32$ ($d_{\text{real}} = 384$)
- **Dependent Observables**:
  - `d128_tok_per_s`: Steady-state training throughput at $d_{\text{complex}} = 128, L = 6$.
  - `max_small_d_tok_per_s`: Maximum steady-state training throughput achieved among $d_{\text{complex}} \in \{32, 64, 96\}$.
  - `all_float32_leaks`: Total `float32` activation leaks across all 6 configurations.
- **Quantitative Pass Criteria**:
  - `all_float32_leaks == 0`.
  - `d128_tok_per_s >= 28000.0` (`>= 1.40x` faster than `Wave 38`'s `20,068 tok/s` and `>= 3.5x` faster than `Wave 37`'s `7,952 tok/s`).
  - `max_small_d_tok_per_s >= 38000.0`.

### Hypothesis `H3` (5-Minute Wall-Clock Tournament on `FineWeb-Edu` Across Multiples of 32)
- **Independent Variable**: Training the candidate `d_complex` multiples of 32 on `FineWeb-Edu` under an identical **wall-clock training budget per arm** (with total steps $S_i = \lfloor T_{\text{budget}} \cdot \text{tok\_per\_s}_i / 8192 \rfloor$ calibrated from warmup throughput so every arm completes its full Smith one-cycle cosine decay right at the end of the wall-clock budget).
- **Dependent Observables**:
  - `winner_run_key`, `winner_d_complex`, and `winner_val_ce`: The configuration achieving the lowest Validation Cross-Entropy at the end of its wall-clock run.
- **Quantitative Pass Criteria**:
  - `winner_val_ce < 6.0000` (improving over `Wave 38`'s 300-step `6.2452` / `6.2951` by `> 0.24` nats within a single 5-minute wall-clock run!).

### Hypothesis `H4` (Launching & Verifying the 900M-Token Production Run of the Tournament Winner)
- **Independent Variable**: Training the winning `d_complex` configuration from `H3` on the `FineWeb-Edu` token stream toward a **`900M`-token (`900,000,000` tokens) horizon** (`batch_size=16, seq_len=512`, `109,863` total steps, streaming live `[STEP]` and `[EVAL]` checkpoints via `StandardizedExperimentLogger` to `.cache/live_logs/`, `.cache/model_cards/`, `.cache/benchmarks.db`, and the WebUI).
- **Dependent Observables**:
  - Live milestone Validation Cross-Entropy at `10M`, `25M`, `50M`, `100M`, ..., `900M` tokens.
- **Quantitative Pass Criteria**:
  - Validated launch and live WebUI telemetry ingestion (`val_ce < 5.75` by `15M` tokens and continuous monotonic decrease along the `900M`-token trajectory).

---

## 3. Planned File Changes & Mathematical Specifications

### 3.1 [MODIFY] [`src/qdlblocks/kernels/flash_complex_krylov_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/flash_complex_krylov_triton.py)

1. **Top-of-File Constants Update (Rule 7)**:
   - Add `TRITON_COMPLEX_MIN_DIM: int = 8`
   - Add `TRITON_COMPLEX_MAX_DIM: int = 2048`
   - Update `DEFAULT_KRYLOV_HEAD_CHUNK: int = 8` (so all models with $H \le 8$, i.e. $d_{\text{complex}} \le 256$, execute in a single batched cuBLAS call with zero head-chunk loop).

2. **Masked Arbitrary-Width `_fused_complex_norm_rope_fwd_kernel` & `_fused_complex_norm_rope_bwd_kernel`**:
   - Add `d_dim: int` parameter to both kernels and set `BLOCK_D = triton.next_power_of_2(d_reg)`.
   - Compute `offs_d = tl.arange(0, BLOCK_D)`, `mask_d = offs_d < d_dim`, `base_offset = row_idx * d_dim + offs_d`, and `rope_offset = t_idx * d_dim + offs_d`.
   - Use `mask=mask_d, other=0.0` on all `tl.load` calls and `mask=mask_d` on all `tl.store` calls.
   - Update `fused_complex_norm_rope_triton` dispatch condition from `d_reg == TRITON_COMPLEX_REG_DIM` to `TRITON_COMPLEX_MIN_DIM <= d_reg <= TRITON_COMPLEX_MAX_DIM`.

3. **Scale-Invariant Fast-Path in `factored_complex_krylov_step` and `flash_complex_krylov_attention`**:
   - Update signatures (`<= 8` args):
     ```python
     def factored_complex_krylov_step(
         q_rot_re: torch.Tensor,
         q_rot_im: torch.Tensor,
         k_rot_re: torch.Tensor,
         k_rot_im: torch.Tensor,
         v_re: torch.Tensor,
         v_im: torch.Tensor,
         use_vkrylov: bool = True,
         normalize_output: bool = True,
     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
     ```
     and:
     ```python
     def flash_complex_krylov_attention(
         q_rot_re: torch.Tensor,
         q_rot_im: torch.Tensor,
         k_rot_re: torch.Tensor,
         k_rot_im: torch.Tensor,
         v_re: torch.Tensor,
         v_im: torch.Tensor,
         probe_mode: str = "v_krylov",
         normalize_output: bool = True,
     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
     ```
   - Inside `factored_complex_krylov_step`:
     * Compute `y1_cat = torch.matmul(wa1_re, v_cat) + torch.matmul(wa1_im, v_skew)` and `z_cat = torch.matmul(wa2_re, v_cat) + torch.matmul(wa2_im, v_skew)` (2 concatenated GEMMs per Krylov step instead of 4 separate real/imag GEMMs + `torch.cat`).
     * When `normalize_output=True`:
       - Compute detached Rayleigh-Krylov telemetry in $O(T d_h)$ time:
         `y1_norm_sq = y1_cat.float().square().sum(dim=-1).clamp_min(DEFAULT_COMPLEX_KERNEL_EPS)`
         `z_norm = torch.sqrt(z_cat.float().square().sum(dim=-1) + DEFAULT_COMPLEX_KERNEL_EPS)`
         `raw_probe_norm = (z_norm / y1_norm_sq).mean().detach()`
         `rayleigh_purity = y1_norm_sq.mean().detach()`
       - Normalize `z_re = z_cat[..., :dh]`, `z_im = z_cat[..., dh:]` directly via `fused_complex_norm_rope_triton(z_re, z_im, None, None)` and return `(u_re, u_im, rayleigh_purity, raw_probe_norm)` — completely bypassing the $O(T^3)$ `g_re`, `g_im`, and `wg = torch.matmul(w_gemm, g_sq)` GEMMs!
     * When `normalize_output=False`:
       - Compute exact `g_re`, `g_im`, `g_sq`, `wg`, `tr_sq` and return unnormalized `z / (tr_sq + eps)` so all existing unnormalized unit tests continue to pass with `< 1e-6` error.

### 3.2 [MODIFY] [`src/qdlblocks/layers/packed_complex_register_density_attn.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/packed_complex_register_density_attn.py)
1. **`complex_l2_normalize`**:
   - Dispatch to `fused_complex_norm_rope_triton(x_re, x_im, None, None, eps=eps)` whenever `x_re.is_cuda and dim in (-1, x_re.ndim - 1) and 8 <= x_re.size(-1) <= 2048 and not torch.compiler.is_compiling()`, accelerating all register, residual, and FFN normalizations across every multiple of 32.
2. **`PackedComplexRegisterDensityAttention.forward_complex`**:
   - When calling `flash_complex_krylov_attention`, pass `normalize_output=self.normalize_krylov_ket` and avoid redundant post-normalization when `self.normalize_krylov_ket` is `True`.

### 3.3 [MODIFY] [`src/qdlblocks/models/gemini_packed_complex_rope_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/gemini_packed_complex_rope_gpt.py) & [`src/qdlblocks/benchmarks/wave_ingestor.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/wave_ingestor.py)
1. **`GeminiPackedComplexRoPEConfig`**:
   - Support any `d_complex >= 16` divisible by `n_registers` where `d_register = d_complex // n_registers >= 8` (including `d_complex = 32` with `n_registers = 2, d_register = 16` or `n_registers = 1, d_register = 32`, and `d_complex in {64, 96, 128, 160, 192, 256}` with `d_register = 32`).
2. **`build_gemini_complex_optimizers`**:
   - Route 2D complex matrices with `min(param.shape) >= 16 and max(param.shape) <= MAX_COMPLEX_MUON_DIM` to `StiefelMuon` so even $d_{\text{complex}} = 32$ and $d_h = 16$ projections receive Riemannian Polar Express updates.
   - For $d_{\text{real}} < 256$ ($d_{\text{complex}} \in \{32, 64, 96\}$), allow gentle $\mu\text{P}$ inverse-width LR scaling `d_ratio = 256.0 / float(model.cfg.real_equiv_dim)` with `muon_scale = min(1.35, d_ratio ** 0.15) * depth_scale` and `embed_scale = min(1.50, d_ratio ** 0.35)`.
3. **Register `"W39"` in [`src/qdlblocks/benchmarks/wave_ingestor.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/wave_ingestor.py)**.

### 3.4 [MODIFY] [`scripts/complex_krylov_rtx5060ti_vram_sizer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complex_krylov_rtx5060ti_vram_sizer.py)
- Add `--mode {max_vram,small_d_sweep}` CLI flag (default `"max_vram"`, preserving backward compatibility).
- When `--mode small_d_sweep` is selected, benchmark the 6 multiples of 32 (`d_complex in {32, 64, 96, 128, 160, 192}` at $L = 6, B = 16, T = 512$), reporting `total_params_M`, `float32_activation_leak_count`, `peak_vram_mib`, `tok_per_s`, `est_5min_steps`, and `est_5min_tokens_M`, and saving the report to `--json-out` (default `.cache/complex_krylov_small_d_sweep.json`).

### 3.5 [NEW] [`tests/test_plan054_small_d_flash_krylov_triton.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan054_small_d_flash_krylov_triton.py)
- **`test_h1_nonpow2_norm_rope_and_scale_invariant_krylov_parity`**: Verifies `fused_complex_norm_rope_triton` across $D \in \{16, 24, 32, 64, 96, 128, 160, 192, 256\}$ (`< 1e-6` error) and `flash_complex_krylov_attention(..., normalize_output=True)` forward (`< 1e-5`) and backward gradient cosine similarity (`> 0.999`) across $d_h \in \{16, 32\}$ and $H \in \{2, 3, 4, 5, 6\}$.
- **`test_h2_small_d_multiples_of_32_forward_backward`**: Verifies end-to-end forward + backward + optimizer step + unitary retraction for `GeminiPackedComplexRoPEGPT` at $d_{\text{complex}} \in \{32, 64, 96, 128, 160, 192\}$ with zero `float32` activation leaks under CUDA `bfloat16` autocast.

### 3.6 [NEW] [`examples/64_verify_plan054_small_d_5min_tournament_and_900m_launch.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/64_verify_plan054_small_d_5min_tournament_and_900m_launch.py)
- Supports two clean modes via CLI flags:
  1. `--stage tournament` (`--wall-clock-sec 300`, `--candidates 32,64,96,128,160,192`):
     - Runs the `H1` parity audit and `H2` throughput audit.
     - Trains each selected multiple of 32 (`d_complex`) for the exact wall-clock budget (`--wall-clock-sec`, default `300.0` seconds = 5.0 minutes; or `--Quick-screen-sec` if specified, followed by full `300.0` second runs on the top contenders), calibrating each arm's Smith one-cycle cosine LR schedule to its measured `tok/s` so every arm finishes at `tau = 1.0` right at the wall-clock limit.
     - Selects the lowest-Val-CE winner, writes `.cache/hypothesis_experiments_054.json`, and syncs to the WebUI.
  2. `--stage run_900m` (`--d-complex <winner>`, `--n-registers <H>`, `--d-register <dh>`, `--num-layers 6`, `--target-tokens-m 900.0`):
     - Launches the winning configuration on the full `FineWeb` / `FineWeb-Edu` token stream for **`900M` tokens** (`109,863` steps at `8,192` tok/step), logging `[STEP]` every 100 steps and `[EVAL]` every 2,000 steps (`~16.4M` tokens) via `StandardizedExperimentLogger` to `.cache/live_logs/`, `.cache/model_cards/`, `.cache/benchmarks.db`, and live-reloading the WebUI.

---

## 4. Verification & Testing Plan

1. **Unit Tests (`Plan 052`, `Plan 053`, & `Plan 054`)**:
   ```bash
   PYTHONPATH=src /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/tests/test_plan052_packed_complex_register_density_attn.py && PYTHONPATH=src /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/tests/test_plan053_flash_complex_krylov_and_bf16_scaling.py && PYTHONPATH=src /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/tests/test_plan054_small_d_flash_krylov_triton.py
   ```
2. **AST Signature & SLOC Linter (`Rules 1–8`)**:
   ```bash
   PYTHONPATH=src /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/kernels/flash_complex_krylov_triton.py src/qdlblocks/layers/packed_complex_register_density_attn.py src/qdlblocks/models/gemini_packed_complex_rope_gpt.py scripts/complex_krylov_rtx5060ti_vram_sizer.py tests/test_plan054_small_d_flash_krylov_triton.py examples/64_verify_plan054_small_d_5min_tournament_and_900m_launch.py
   ```
3. **Small-`d_complex` Multiples-of-32 Throughput & 5-Min Budget Sizer (`H2`)**:
   ```bash
   PYTHONPATH=src /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/complex_krylov_rtx5060ti_vram_sizer.py --mode small_d_sweep --json-out .cache/complex_krylov_small_d_sweep.json
   ```
4. **5-Minute Wall-Clock Tournament & 900M-Token Production Launch (`H3` & `H4`)**:
   ```bash
   PYTHONPATH=src /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/examples/64_verify_plan054_small_d_5min_tournament_and_900m_launch.py --stage tournament --wall-clock-sec 300
   ```

---

## 5. Operational Tooling & Script Opportunities

1. **Existing Scripts Used**:
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py`
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload`
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once`
2. **`Build-Before-Launch` Scripts (Upgraded in This Plan)**:
   - [`scripts/complex_krylov_rtx5060ti_vram_sizer.py --mode small_d_sweep`](file:///home/festivity/agy/QuantumDLBlocks/scripts/complex_krylov_rtx5060ti_vram_sizer.py): Profiles all multiples of 32 ($d_{\text{complex}} \in \{32, 64, 96, 128, 160, 192\}$) on the RTX 5060 Ti, computing exact `tok/s`, `VRAM (MiB)`, `est_5min_steps`, and `est_5min_tokens_M` before launching the tournament.
3. **`Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md` & Surfaced to User)**:
   - `scripts/wall_clock_chinchilla_pareto_predictor.py`: Combines measured `tok/s(d_complex)` from `complex_krylov_rtx5060ti_vram_sizer.py` with empirical Chinchilla scaling exponents $(\alpha, \beta)$ to predict the optimal width $d_{\text{complex}}^*(T_{\text{wall}})$ for any target wall-clock budget $T_{\text{wall}} \in \{1\text{ min}, 5\text{ min}, 30\text{ min}, 6\text{ hr}\}$.

---

## 6. Constants Ledger & Empirical Results (`Wave 39`)

### 6.1 Top-of-File Named Constants Ledger (Rule 7)
| Constant Name | File | Value | Mathematical / Physical Rationale |
| :--- | :--- | :---: | :--- |
| `TRITON_COMPLEX_MIN_DIM` | `flash_complex_krylov_triton.py` | `8` | Minimum complex dimension dispatched to masked Triton norm + RoPE kernel. |
| `TRITON_COMPLEX_MAX_DIM` | `flash_complex_krylov_triton.py` | `2048` | Maximum complex dimension dispatched to masked Triton norm + RoPE kernel. |
| `DEFAULT_KRYLOV_HEAD_CHUNK` | `flash_complex_krylov_triton.py` | `8` | Register slice size so all models with $H \le 8$ ($d_{\text{complex}} \le 256$) execute in one batched cuBLAS call. |
| `MIN_COMPLEX_MUON_DIM` | `gemini_packed_complex_rope_gpt.py` | `16` | Minimum 2D matrix dimension routed to `StiefelMuon` so $d_{\text{complex}} = 32$ and $d_h = 16$ matrices are orthogonalized. |

### 6.2 Empirical Results Summary (`Wave 39`)

#### 1. `H1` Non-Power-of-2 Triton Norm+RoPE & Scale-Invariant Krylov Parity
- `max_nonpow2_norm_rope_err_fp32`: **`5.96e-08`** (`< 1e-6` threshold — **PASS**)
- `max_fast_krylov_unit_err_fp32`: **`1.34e-06`** (`< 1e-5` threshold — **PASS**)
- `min_fast_krylov_grad_cosine_fp32`: **`1.000000`** (`> 0.999` threshold — **PASS**)

#### 2. `H2` Multiples-of-32 `d_complex` Throughput & VRAM Profile (`B = 16, T = 512, L = 6`)
| Tier | `d_complex` | $H$ | $d_h$ | `d_real` | Params | F32 Leaks | Train VRAM | Throughput (`tok/s`) | 5-Min Steps | 5-Min Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `d32c_H2_dh16_L6_B16` | **32** | 2 | 16 | 64 | **3.35M** | `0` | **693.6 MiB** | **48,604 tok/s** | 1,779 | **14.58M** |
| `d64c_H2_dh32_L6_B16` | **64** | 2 | 32 | 128 | **6.94M** | `0` | **964.9 MiB** | **43,805 tok/s** | 1,604 | **13.14M** |
| `d96c_H3_dh32_L6_B16` | **96** | 3 | 32 | 192 | **10.79M** | `0` | **1,261.1 MiB** | **33,669 tok/s** | 1,232 | **10.10M** |
| `d128c_H4_dh32_L6_B16` | **128** | 4 | 32 | 256 | **14.88M** | `0` | **1,665.8 MiB** | **28,286 tok/s** | 1,035 | **8.49M** |
| `d160c_H5_dh32_L6_B16` | **160** | 5 | 32 | 320 | **19.22M** | `0` | **2,072.1 MiB** | **23,687 tok/s** | 867 | **7.11M** |
| `d192c_H6_dh32_L6_B16` | **192** | 6 | 32 | 384 | **23.82M** | `0` | **2,482.3 MiB** | **20,491 tok/s** | 750 | **6.15M** |

#### 3. `H3` 5-Minute (`300s`) Wall-Clock Tournament Leaderboard on `FineWeb-Edu`
| Run Key | `d_complex` ($H \times d_h$) | Params | 5-Min Steps (`Tokens`) | Speed (`tok/s`) | `t=75s` Val CE | `t=150s` Val CE | `t=225s` Val CE | **`t=300s` Final Val CE (`PPL`)** | Final Top-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w39_5min_tourney_d32c_H2_dh16_L6_300s` | `32` ($2 \times 16$) | 3.35M | `1,643` (`13.46M`) | `44,844` | `6.4843` | `6.3753` | `6.3158` | **`6.2643`** (`525.46`) | `15.28%` |
| `w39_5min_tourney_d64c_H2_dh32_L6_300s` | `64` ($2 \times 32$) | 6.94M | `1,481` (`12.13M`) | `40,427` | `6.3632` | `6.1743` | `6.0591` | **`5.9914`** (`399.96`) | `16.39%` |
| `w39_5min_tourney_d96c_H3_dh32_L6_300s` | `96` ($3 \times 32$) | 10.79M | `1,174` (`9.62M`) | `32,055` | **`6.3194`** | **`6.1287`** | `6.0000` | **`5.9219`** (`373.13`) | `17.00%` |
| **`w39_5min_tourney_d128c_H4_dh32_L6_300s`** | **`128` ($4 \times 32$)** | **14.88M** | **`988` (`8.09M`)** | **`26,951`** | `6.4007` | `6.1391` | **`5.9798`** | **`5.8945` (`363.03`) 🏆** | **`17.51%`** |

- **Key Wall-Clock Scaling Discovery**:
  - For short wall-clock budgets ($t \le 150\text{s} = 2.5\text{ min}$), **`d_complex = 96` ($H=3, d_h=32$, `10.79M` params, `32,055 tok/s`)** achieves the lowest validation loss (`6.3194` at `75s`, `6.1287` at `150s`), outperforming `d_complex = 128` because its `+19%` higher token throughput outweighs capacity in the early token regime.
  - Between $t = 150\text{s}$ and $t = 225\text{s}$ (`~6M` tokens), **`d_complex = 128` ($H=4, d_h=32$, `14.88M` params, `26,951 tok/s`)** crosses over and wins the 5-minute (`300s`) tournament at **`5.8945` Val CE (`363.03` PPL, `17.51%` Top-1)** — a **`-0.4006` nat improvement** over `Wave 38`'s 300-step `d_complex = 128` baseline (`6.2951`) and **`-0.3507` nats** better than `Wave 38`'s `104.49M`-param max-VRAM run (`6.2452`).
  - Even tiny **`d_complex = 64` (`6.94M` params, `40,427 tok/s`)** reaches **`5.9914` Val CE (`399.96` PPL, `16.39%` Top-1)** in 5 minutes (`-0.2538` nats below `Wave 38`), and ultra-compact **`d_complex = 32` (`3.35M` params, `44,844 tok/s`)** reaches **`6.2643` Val CE (`15.28%` Top-1)** using just `752 MiB` of training VRAM.

#### 4. `H4` Live 900M-Token Production Run (`w39_winner_d128c_H4_dh32_L6_900Mtok`)
- Launched in background (`task-1276`) for **`900.0M` tokens (`109,863` steps)** across 9 `FineWeb` shards (`100M` tokens/shard) with asynchronous background shard prefetching (`FineWebMultiShardStream`), streaming live telemetry to `.cache/live_logs/w39_winner_d128c_H4_dh32_L6_900Mtok.log` and the WebUI (`:8765`).

