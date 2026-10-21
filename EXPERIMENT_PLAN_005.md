# EXPERIMENT_PLAN_005: Analytic-Gradient L2-Tiled POVM Cross-Entropy for HBM Memory Reduction & Roofline Arithmetic Intensity

**Document ID**: `EXPERIMENT_PLAN_005` (Revision 2)  
**Status**: Revised for Blind Sub-Agent Review  
**Target Repository**: `/home/festivity/agy/QuantumDLBlocks`  

---

## 1. Executive Summary & Self-Contained Background

### 1.1 Systems Bottleneck in Large-Vocabulary Quantum POVM Readout
In `QuantumDLBlocks`, the 10-qubit `QuantumBornGPT` model ($d=1024, D_{\text{seed}}=128, V=50,304$) processes $N = B \times T$ tokens per micro-batch (e.g., $B=32, T=512 \implies N = 16,384$).
At the output layer, the normalized seed states $Z \in \mathbb{R}^{N \times D_{\text{seed}}}$ are measured against the normalized vocabulary POVM dictionary $M \in \mathbb{R}^{V \times D_{\text{seed}}}$ scaled by inverse Born temperature $\beta \in \mathbb{R}$:
$$L = \beta \, Z M^\top \in \mathbb{R}^{N \times V}, \qquad \mathcal{L}(Z, M, \beta; y) = \frac{1}{N} \sum_{i=1}^{N} \left(\ln \sum_{v=0}^{V-1} \exp(\beta \langle z_i \mid m_v \rangle) - \beta \langle z_i \mid m_{y_i} \rangle\right).$$

### 1.2 Hardware Roofline & Memory Diagnosis
When computing $\mathcal{L}$ via standard PyTorch autograd (`F.cross_entropy(F.linear(Z, M) * beta, y)`) or large-chunk checkpointing (`chunk_sz = 16384`):
1. **Massive Intermediate Logit Materialization in HBM**:
   - Materializing the `[16384, 50304]` logit tensor $L$ in FP32 (`3.30 GB`) or BF16 (`1.65 GB`) exceeds the `96 MiB` (`100,663,296` bytes) L2 cache of an NVIDIA RTX PRO 6000 Blackwell GPU by $> 16\times$, forcing every logit read/write out to off-chip HBM!
2. **Eliminating HBM Spill via Analytic-Gradient L2-Cache-Sized Token Tiling**:
   - The exact analytical gradients of $\mathcal{L}$ with respect to $Z \in \mathbb{R}^{N \times D_{\text{seed}}}$, $M \in \mathbb{R}^{V \times D_{\text{seed}}}$, and $\beta \in \mathbb{R}$ decompose additively over small token tiles $Z_b \in \mathbb{R}^{B_{\text{tile}} \times D_{\text{seed}}}$ (where $B_{\text{tile}} \in \{256, 512, 1024\}$ so the single in-place $[B_{\text{tile}}, V]$ tile buffer fits inside on-chip L2 cache!):
     For each token tile $b$ of size $B_{\text{tile}}$:
     $$\ell_b = \beta \, Z_b M^\top \in \mathbb{R}^{B_{\text{tile}} \times V}, \qquad P_b = \exp(\ell_b - l_b) \in \mathbb{R}^{B_{\text{tile}} \times V},$$
     $$G_b = \frac{1}{N} \left(P_b - \mathbf{E}_{y_b}\right) \in \mathbb{R}^{B_{\text{tile}} \times V} \quad (\text{computed in-place in the } [B_{\text{tile}}, V] \text{ tile buffer!}),$$
     $$\frac{\partial \mathcal{L}}{\partial Z_b} = \beta \, G_b M \in \mathbb{R}^{B_{\text{tile}} \times D_{\text{seed}}}, \qquad \frac{\partial \mathcal{L}}{\partial M} \mathrel{+}= \beta \, G_b^\top Z_b \in \mathbb{R}^{V \times D_{\text{seed}}}, \qquad \frac{\partial \mathcal{L}}{\partial \beta} \mathrel{+}= \frac{1}{\beta} \sum_{i, v} G_{b, i, v} \ell_{b, i, v}.$$
   - By implementing a custom `torch.autograd.Function` (`_FusedTiledPOVMCEFn`) and a custom `@jax.custom_vjp` (`fused_tiled_povm_cross_entropy_jax` using `jax.lax.scan` over tiles) that:
     1. Compute the forward loss in tiles of size $B_{\text{tile}}$ saving **only** the 1D per-token log-sum-exp vector $l_i = \ln \sum_v \exp(\ell_{i, v}) \in \mathbb{R}^N$ (`64 KiB` for $N=16384$ instead of `1.65 - 3.30 GB` of logits), and
     2. In the backward pass, reconstruct $P_b = \exp(\beta Z_b M^\top - l_b)$ tile-by-tile using the saved $l_b$ (avoiding a second `logsumexp` reduction pass!) and overwrite $P_b$ in-place with $G_b = \frac{1}{N}(P_b - \mathbf{E}_{y_b})$ before accumulating $\nabla_{Z_b}, \nabla_M, \nabla_\beta$:
        - The single in-place $[B_{\text{tile}}, V]$ tile working set is `tile_working_set_bytes = eff_tile * vocab_size * dtype_bytes`, which drops by a factor of $N / B_{\text{tile}}$ (e.g., **$8\times$ smaller** at $B_{\text{tile}} = 256$ vs $N = 2048$, and **$32\times$ smaller** at $B_{\text{tile}} = 512$ vs $N = 16384$).
        - Saved activation memory between forward and backward drops from $O(N \cdot V)$ to $O(N)$ ($l \in \mathbb{R}^N$).

---

## 2. Observable Hypotheses

### Hypothesis H1: Exact Analytical Gradient Equivalence & $O(N / B_{\text{tile}})$ Peak Memory Reduction
- **Mechanism**: Compare standard un-tiled dense autograd cross-entropy `F.cross_entropy(F.linear(Z.float(), M.float()) * beta.float(), y)` against `fused_tiled_povm_cross_entropy(Z, M, beta, y, tile_size=B_tile)` for $N = 2048, D_{\text{seed}} = 64, V = 8192$ and $B_{\text{tile}} = 256$.
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Bit-Accurate Forward Loss & Backward Gradients**:
     - Forward scalar loss matches dense FP32 reference within $|\mathcal{L}_{\text{tiled}} - \mathcal{L}_{\text{dense}}| \le 10^{-5}$.
     - For `float32` inputs, analytical gradients $(\nabla_Z \mathcal{L}, \nabla_M \mathcal{L}, \nabla_\beta \mathcal{L})$ match dense autograd reference within relative $L_2$ error $\le 10^{-4}$ and max absolute error `atol=1e-4` in both PyTorch and JAX.
     - For `bfloat16` inputs in PyTorch, gradients match dense `bfloat16`-cast reference within `atol=5e-3, rtol=1e-2`.
  2. **Transient Logit Memory Footprint Reduction**:
     - For $N = 2048, V = 8192, \text{dtype\_bytes} = 2$, the in-place tile working set `peak_logit_buffer_mb` at `tile_size = 256` (`256 * 8192 * 2 = 4.0 MiB`) is **$8\times$ smaller** than un-tiled `tile_size = 2048` (`2048 * 8192 * 2 = 32.0 MiB`), and the saved activation tensor `lse` is `8.0 KiB` vs `32.0 MiB` (`4,096x` smaller).

### Hypothesis H2: Roofline Arithmetic Intensity Scaling Across the L2 Cache Capacity Threshold
- **Mechanism**: Define the exact FLOPs and HBM Byte Traffic roofline model `estimate_povm_head_roofline(num_tokens, seed_dim, vocab_size, tile_size, *, dtype_bytes=2, l2_cache_bytes=96*1024*1024)`:
  - Single in-place tile working set: `tile_working_set_bytes = float(eff_tile * vocab_size * dtype_bytes)` where `eff_tile = min(num_tokens, tile_size)`.
  - For $N = 16384, D_{\text{seed}} = 128, V = 50304, \text{dtype\_bytes} = 2$ on a $96\text{ MiB}$ (`100,663,296` bytes) L2 cache (`Blackwell G4`), `tile_size = 512` has working set `512 * 50304 * 2 = 51,511,296` bytes (`49.125 MiB <= 96.0 MiB`), so `fits_in_l2 == True`! In contrast, un-tiled `tile_size = 16384` has working set `1,648,361,472` bytes (`1572.0 MiB > 96.0 MiB`), so `fits_in_l2 == False` and spills `[N, V]` logits/gradients to HBM.
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Roofline Arithmetic Intensity Increase**: For $N = 16384, D_{\text{seed}} = 128, V = 50304, \text{dtype\_bytes} = 2, \text{l2\_cache\_bytes} = 96 \times 1024 \times 1024$:
     $$\text{arithmetic\_intensity}(\text{tile\_size}=512) \ge 2.5 \times \text{arithmetic\_intensity}(\text{tile\_size}=16384).$$
  2. **Monotonic Peak Buffer Scaling**: For $N = 2048, D_{\text{seed}} = 64, V = 16384$:
     $$\text{peak\_logit\_buffer\_mb}(\text{tile\_size}=256) \le 0.125 \times \text{peak\_logit\_buffer\_mb}(\text{tile\_size}=2048).$$

---

## 3. Planned Code Changes (Disjoint New Modules & Files)

To allow conflict-free parallel implementation alongside other ongoing tasks, all changes are placed in **new, dedicated files**:

### 3.1 PyTorch Analytic-Gradient L2-Tiled POVM Cross-Entropy (`src/qdlblocks/kernels/fused_povm_ce.py`)

Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/kernels/fused_povm_ce.py` implementing:

1. **`estimate_povm_head_roofline(num_tokens: int, seed_dim: int, vocab_size: int, tile_size: int, *, dtype_bytes: int = 2, l2_cache_bytes: int = 96 * 1024 * 1024) -> dict[str, float | bool]`**:
   - **Validation**: Raises `ValueError` if `num_tokens < 1`, `seed_dim < 1`, `vocab_size < 1`, `tile_size < 1`, or `dtype_bytes < 1`.
   - **Exact Computation**:
     ```python
     eff_tile = min(num_tokens, tile_size)
     gemm_flops = float(8 * num_tokens * seed_dim * vocab_size)
     tile_working_set_bytes = float(eff_tile * vocab_size * dtype_bytes)
     fits_in_l2 = bool(tile_working_set_bytes <= float(l2_cache_bytes))
     base_hbm_bytes = float((4 * num_tokens * seed_dim + 4 * vocab_size * seed_dim) * dtype_bytes + num_tokens * 12)
     if fits_in_l2:
         total_hbm_bytes = base_hbm_bytes
     else:
         spill_hbm_bytes = float(6 * num_tokens * vocab_size * dtype_bytes)
         total_hbm_bytes = base_hbm_bytes + spill_hbm_bytes
     arithmetic_intensity = gemm_flops / max(1.0, total_hbm_bytes)
     peak_logit_buffer_mb = tile_working_set_bytes / (1024.0 * 1024.0)
     saved_activation_kb = float(num_tokens * 4) / 1024.0
     return {
         "gemm_flops": gemm_flops,
         "total_hbm_bytes": total_hbm_bytes,
         "arithmetic_intensity": round(arithmetic_intensity, 2),
         "fits_in_l2": fits_in_l2,
         "peak_logit_buffer_mb": round(peak_logit_buffer_mb, 3),
         "saved_activation_kb": round(saved_activation_kb, 3),
     }
     ```

2. **`class _FusedTiledPOVMCEFn(torch.autograd.Function)`** and **`fused_tiled_povm_cross_entropy(z: torch.Tensor, m_vocab: torch.Tensor, inv_temp: torch.Tensor, targets: torch.Tensor, *, tile_size: int = 1024) -> torch.Tensor`**:
   - **Inputs & Validation**:
     - `z`: 2D floating tensor of shape `(N, D)`
     - `m_vocab`: 2D floating tensor of shape `(V, D)`
     - `inv_temp`: scalar (`0`-dim or `1`-element) floating tensor $\beta > 0$
     - `targets`: 1D integer tensor of shape `(N,)` with values in `[0, V - 1]`
     - `tile_size`: integer $B_{\text{tile}} \ge 1$
     - Raises `ValueError` if `tile_size < 1`, `z.ndim != 2`, `m_vocab.ndim != 2`, `z.size(1) != m_vocab.size(1)`, `targets.ndim != 1`, or `targets.size(0) != z.size(0)`.
   - **Forward (`ctx, z, m_vocab, inv_temp, targets, tile_size`)**:
     - Both forward and backward accumulate the tile GEMMs in `torch.float32` (`F.linear(z_b.float(), m_vocab.float())`) so forward and backward are numerically consistent for both `float32` and `bfloat16` inputs:
       ```python
       N = z.size(0)
       lse = torch.empty(N, device=z.device, dtype=torch.float32)
       loss_sum = torch.zeros((), device=z.device, dtype=torch.float32)
       beta_f = float(inv_temp.detach().float().item())
       m_f = m_vocab.detach().float()
       for start in range(0, N, tile_size):
           end = min(start + tile_size, N)
           z_b = z[start:end].detach().float()
           y_b = targets[start:end]
           logits_b = F.linear(z_b, m_f) * beta_f
           lse_b = torch.logsumexp(logits_b, dim=-1)
           tgt_logits_b = logits_b.gather(1, y_b.unsqueeze(1)).squeeze(1)
           lse[start:end] = lse_b
           loss_sum = loss_sum + torch.sum(lse_b - tgt_logits_b)
       ctx.save_for_backward(z, m_vocab, inv_temp, targets, lse)
       ctx.tile_size = tile_size
       return loss_sum / float(N)
       ```
   - **Backward (`ctx, grad_output`)**:
     ```python
     z, m_vocab, inv_temp, targets, lse = ctx.saved_tensors
     N, D = z.shape
     tile_size = ctx.tile_size
     beta_f = float(inv_temp.detach().float().item())
     scale = float(grad_output.detach().float().item()) / float(N)
     m_f = m_vocab.detach().float()
     dz = torch.empty((N, D), device=z.device, dtype=torch.float32)
     dm = torch.zeros_like(m_f)
     dbeta = torch.zeros((), device=z.device, dtype=torch.float32)
     for start in range(0, N, tile_size):
         end = min(start + tile_size, N)
         z_b = z[start:end].detach().float()
         y_b = targets[start:end]
         lse_b = lse[start:end].unsqueeze(1)
         raw_dot = F.linear(z_b, m_f)
         probs_b = torch.exp(raw_dot * beta_f - lse_b)
         probs_b.scatter_add_(
             1,
             y_b.unsqueeze(1),
             torch.full((end - start, 1), -1.0, device=z.device, dtype=torch.float32),
         )
         dbeta = dbeta + torch.sum(probs_b * raw_dot) * scale
         probs_b.mul_(scale * beta_f)
         dz[start:end] = torch.matmul(probs_b, m_f)
         dm.addmm_(probs_b.transpose(0, 1), z_b)
     return dz.to(z.dtype), dm.to(m_vocab.dtype), dbeta.reshape_as(inv_temp).to(inv_temp.dtype), None, None
     ```

---

### 3.2 JAX Analytic-Gradient L2-Tiled POVM Cross-Entropy (`src/qdlblocks_jax/fused_povm_ce.py`)

Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/fused_povm_ce.py` implementing:
1. **`estimate_povm_head_roofline_jax(...)`**: Calls the exact same roofline formula as `estimate_povm_head_roofline`.
2. **Internal Positional `@jax.custom_vjp` Function & Public Keyword Wrapper**:
   - Define internal positional function `_fused_tiled_povm_ce_jax(z: jax.Array, m_vocab: jax.Array, inv_temp: jax.Array, targets: jax.Array, tile_size: int) -> jax.Array` decorated with `@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))`:
     - Uses `jax.lax.scan` over `n_tiles = N // tile_size` tiles of shape `(tile_size, D)` in `jnp.float32`, saving only `(z, m_vocab, inv_temp, targets, lse)` where `lse` has shape `(n_tiles, tile_size)`.
     - In `_bwd(tile_size, res, g_out)`, scans over tiles to reconstruct `probs_b = jnp.exp(raw_dot * inv_temp_f - lse_b[:, None])`, subtracts one-hot targets `jax.nn.one_hot(y_b, V, dtype=jnp.float32)`, and accumulates `dz`, `dm`, `dbeta`.
   - Define public wrapper `fused_tiled_povm_cross_entropy_jax(z: jax.Array, m_vocab: jax.Array, inv_temp: jax.Array, targets: jax.Array, *, tile_size: int = 1024) -> jax.Array`:
     - Validates `tile_size >= 1`, `z.ndim == 2`, `m_vocab.ndim == 2`, `z.shape[1] == m_vocab.shape[1]`, `targets.ndim == 1`, `targets.shape[0] == z.shape[0]`, and `z.shape[0] % tile_size == 0` (raising `ValueError` otherwise).
     - Calls `return _fused_tiled_povm_ce_jax(z, m_vocab, inv_temp, targets, int(tile_size))`.

---

### 3.3 Standalone Controlled Hypothesis Verification Script (`examples/15_hypothesis_fused_povm_roofline.py`)

Create `/home/festivity/agy/QuantumDLBlocks/examples/15_hypothesis_fused_povm_roofline.py` that runs in `< 30` seconds, prints a roofline and accuracy summary table, asserts all quantitative criteria of Hypotheses H1 and H2, and writes `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_005.json`:
1. **Experiment 1 (H1 — Exact Gradient Equivalence & Saved/Transient Memory Reduction)**:
   - Evaluates `fused_tiled_povm_cross_entropy` (`tile_size=256`) vs. dense autograd (`F.cross_entropy(F.linear(z.float(), m.float()) * beta.float(), y)`) for $N = 2048, D_{\text{seed}} = 64, V = 8192$ in both PyTorch and JAX.
   - Asserts:
     - Forward loss difference $\le 10^{-5}$.
     - Relative $L_2$ error and max absolute error of $\nabla_Z, \nabla_M, \nabla_\beta \le 10^{-4}$ in both PyTorch and JAX.
     - `peak_logit_buffer_mb(tile_size=256) == 4.0` vs `peak_logit_buffer_mb(tile_size=2048) == 32.0` ($8\times$ smaller), and `saved_activation_kb == 8.0`.
2. **Experiment 2 (H2 — Roofline Arithmetic Intensity & Tile-Size Sweep)**:
   - Evaluates `estimate_povm_head_roofline` across `tile_size` $\in \{256, 512, 1024, 2048, 4096, 16384\}$ for the FineWeb 100M configuration ($N = 16384, D_{\text{seed}} = 128, V = 50304$, Blackwell `96 MiB` L2 cache) and benchmarks live execution across tile sizes on $N = 2048, D_{\text{seed}} = 64, V = 16384$.
   - Asserts:
     - `arithmetic_intensity(tile_size=512) >= 2.5 * arithmetic_intensity(tile_size=16384)`.
     - `peak_logit_buffer_mb(tile_size=256) <= 0.125 * peak_logit_buffer_mb(tile_size=2048)`.
3. **Exact JSON Artifact Schema (`.cache/hypothesis_experiments_005.json`)**:
   - Keys: `{"h1_gradient_and_memory": {"loss_diff_torch": ..., "max_grad_err_torch": ..., "max_grad_err_jax": ..., "peak_buffer_mb_tiled": 4.0, "peak_buffer_mb_dense": 32.0, "saved_activation_kb": 8.0, "passed": True}, "h2_roofline_scaling": {"fineweb_sweep": {...}, "ai_tile512": ..., "ai_tile16384": ..., "ai_ratio": ..., "passed": True}}`.

---

## 4. Testing & Verification Plan

### 4.1 Unit Tests (`tests/test_hypothesis_005_fused_povm_ce.py`)
Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_005_fused_povm_ce.py` testing:
1. **`test_fused_tiled_povm_ce_exact_gradients_torch`**:
   - Verifies forward loss and backward gradients (`z.grad`, `m_vocab.grad`, `inv_temp.grad`) match dense `F.cross_entropy(F.linear(z.float(), m.float()) * beta.float(), y)` across multiple `tile_size` values (`64, 128, 200` on $N = 500$, including non-divisible $N \bmod \text{tile\_size} \ne 0$):
     - For `float32` inputs: `atol=1e-4, rtol=1e-4`.
     - For `bfloat16` inputs: `atol=5e-3, rtol=1e-2`.
2. **`test_fused_tiled_povm_ce_exact_gradients_jax`**:
   - Verifies `@jax.jit`-compiled `jax.value_and_grad(lambda z, m, b: fused_tiled_povm_cross_entropy_jax(z, m, b, y, tile_size=64), argnums=(0, 1, 2))` matches dense JAX cross-entropy within `atol=1e-4`.
3. **`test_estimate_povm_head_roofline_l2_transition`**:
   - Verifies `fits_in_l2 == True` at `tile_size = 512` (`49.125 MiB <= 96 MiB`) and `fits_in_l2 == False` at `tile_size = 16384` (`1572.0 MiB > 96 MiB`) for $N=16384, D_{\text{seed}}=128, V=50304$, with $\ge 2.5\times$ arithmetic intensity jump.
4. **`test_hypothesis_005_script_execution`**:
   - Executes `examples/15_hypothesis_fused_povm_roofline.py` and verifies that `.cache/hypothesis_experiments_005.json` is written with `"passed": True` for both H1 and H2.
