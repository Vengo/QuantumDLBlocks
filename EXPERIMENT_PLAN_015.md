# Experiment Plan 015: True Cross-Layer First-Layer Value Residual ($V^{(0)}$), Cubic ReLU² Norm Control for $T=512$ Speedrun Alignment, and Fused Triton Purified Born Attention

## 1. Executive Summary & Motivation

Live telemetry from our Wave-6 and Wave-7 (`899.48M`-token FineWeb-Edu) runs and local kernel profiling revealed three critical mathematical and systems findings under our fixed **`block_size = 512` (`T = 512`) validation benchmark invariant**:

1. **Early-Epoch Quantum Fix Validated (`-0.2016 nats` at `100M` Tokens), but Triton Kernel Bypassed by CLI Flag**:
   - In Wave-7, our four-part `QuantumBornGPT` Early-Epoch Fix (`--ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn`) reduced `50M`-token validation cross-entropy from `4.7826` to `4.6119` (`-0.1707 nats`) and `100M`-token validation cross-entropy from `4.6328` to `4.4312` (**`-0.2016 nats`**, `-18.8 PPL`) on `w7_900m_qborn_earlyfix_frame4d_synergy512`, and from `4.5459` to `4.4027` (`-0.1432 nats`) on `w7_900m_qborn_earlyfix_asym1024_L8`.
   - However, `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` passed `use_flash_born=(not purified_born_attn)` (`False`) when instantiating `QuantumBornGPT`, which bypassed our SRAM-tiled Triton kernel `flash_purified_born_attention` and fell back to the chunked Python autograd loop (`62k–79k tok/s`). Enabling `use_flash_born=True` increases `SubspaceProjectorBornAttention(use_purified_attn=True)` throughput by **$4.59\times$ (`602,220 tok/s` vs. `131,302 tok/s`)** and reduces attention peak VRAM by **$33\%$ (`1.31 GB` vs. `1.97 GB`)**.

2. **Colinearity of Same-Layer `v_proj(x_0)` vs. True Cross-Layer First-Layer Value Residual ($V^{(0)}$)**:
   - In both `SubspaceProjectorBornAttention` (`src/qdlblocks/layers/attention.py`) and `ClassicalModdedBlock` (`src/qdlblocks/models/classical_modded_gpt.py`), value residual mixing was previously computed by projecting the initial embedding $x_0$ (or $\psi_0$) through **Layer $l$'s own value matrix $W_V^{(l)}$**:
     $$V_{\text{same}}^{(l)} = (1 - \alpha_l) W_V^{(l)} x_{\text{in}}^{(l)} + \alpha_l W_V^{(l)} x_0 = W_V^{(l)} \big((1 - \alpha_l) x_{\text{in}}^{(l)} + \alpha_l x_0\big)$$
   - Because each block *already* blends $x_0$ into $x_{\text{in}}^{(l)}$ via the input highway gate right before attention, applying the same matrix $W_V^{(l)}$ to $x_0$ again is **100% linearly colinear** with the input highway gate (adding zero new subspace rank outside $\text{col}(W_V^{(l)})$) while executing $(L - 1)$ redundant `v_proj(x_0)` matrix multiplications per forward pass and $2(L - 1)$ per backward pass.
   - In true ResFormer / `modded-nanogpt` value residual mixing, **Layer 0** computes $V^{(0)} = W_V^{(0)} x_{\text{in}}^{(0)}$ once and passes $V^{(0)}$ directly to Layers $l = 1, \dots, L-1$:
     $$V_{\text{cross}}^{(l)} = (1 - \alpha_l) W_V^{(l)} x_{\text{in}}^{(l)} + \alpha_l V^{(0)} = (1 - \alpha_l) W_V^{(l)} x_{\text{in}}^{(l)} + \alpha_l W_V^{(0)} x_{\text{in}}^{(0)}$$
     Because $W_V^{(0)} \neq W_V^{(l)}$, $V_{\text{cross}}^{(l)}$ spans a genuine two-operator subspace that allows $W_V^{(0)}$ to specialize as a shared token-value dictionary while freeing $W_V^{(l)}$ ($l \ge 1$) to represent deep multi-hop context—and **eliminates $(L - 1)$ `v_proj` GEMMs per step**.

3. **Cubic Degree-3 `ReLU^2` Norm Explosion & $T = 512$ Train-Validation Context Alignment in `ClassicalModdedGPT`**:
   - In `ClassicalModdedBlock`, the attention sublayer (`SDPA(RMSNorm(Q), RMSNorm(K), V) @ W_O^T`) is homogeneous of **degree 2** in the 2D weight matrices ($\propto \|W_V\|_2 \|W_O\|_2 \sim \sigma^2$), whereas the `ReLU^2` MLP sublayer ($\text{ReLU}(\text{RMSNorm}(x) W_1^\top)^2 W_2^\top$) is homogeneous of **degree 3** in the 2D weight matrices ($\propto \|W_1\|_2^2 \|W_2\|_2 \sim \sigma^3$).
   - In Wave-6 and Wave-7, running `StiefelMuon` on `ClassicalModdedGPT` with `stiefel_penalty = 0.0` over $6{,}858\text{–}13{,}725$ steps caused 2D weight Frobenius norms to grow by $\sim 8.5\times$ (`IsoErr` reaching `70.26–78.75`). Because $\sigma^3 / \sigma^2 = \sigma \approx 8.5$, the `ReLU^2` MLP branch output grows nearly an order of magnitude faster than the attention branch output, drowning out multi-head attention and inflating gradient variance prior to cooldown.
   - Furthermore, because our nanoGPT speedrun validation benchmark is strictly fixed at **`block_size = 512` (`T_val = 512`)**, training at $T_{\text{train}} = 1024$ (`w7_900m_classical_speedrun_12L_t1024`) caused `Val CE (T=512)` to lag behind `Train CE (T=1024)` by `+0.022 nats` (`3.6842` vs. `3.6618` at `650M` tok) because 50% of training tokens (`positions 512..1023`) updated attention heads for context distances $> 512$ never evaluated during `T_val = 512` validation.
   - By aligning $T_{\text{train}} = 512$ with $T_{\text{val}} = 512$, bounding 2D weight singular values in `ClassicalModdedGPT` via continuous Riemannian Stiefel retraction (`stiefel_penalty = 0.005`), decaying WSD cooldown all the way to `min_lr_ratio = 0.0`, and matching larger global batch sizes (`grad_accum = 4..8`, `262,144–524,288` tok/step), we eliminate cubic `ReLU^2` norm blowup and focus 100% of training updates on the $T = 512$ validation distribution.

---

## 2. Observable Hypotheses & Quantitative Pass/Fail Criteria

### Hypothesis 1 (`H1` — True Cross-Layer First-Layer Value Residual $V^{(0)}$ vs. Colinear Same-Layer `v_proj(x_0)`):
- **Claim**: Replacing same-layer $V_{\text{same}}^{(l)} = (1 - \alpha_l) W_V^{(l)} x_{\text{in}}^{(l)} + \alpha_l W_V^{(l)} x_0$ with true cross-layer $V_{\text{cross}}^{(l)} = (1 - \alpha_l) W_V^{(l)} x_{\text{in}}^{(l)} + \alpha_l V^{(0)}$ (where $V^{(0)} = W_V^{(0)} x_{\text{in}}^{(0)}$ is computed once in Layer 0 and passed to Layers $1 \dots L-1$) introduces non-zero orthogonal subspace energy outside $\text{col}(W_V^{(l)} [x^{(l)}; x_0])$, eliminates $(L - 1)$ redundant `v_proj` calls per forward pass, and lowers validation cross-entropy.
- **Observable Metrics & Pass Criteria**:
  1. **Cross-Layer Orthogonal Subspace Energy**: For a 6-layer model ($d = 128, H = 4$) on random inputs $(B=4, T=64)$, let $r_{\text{orth}}^{(l)}$ be the relative Frobenius norm of the component of $V^{(l)}$ orthogonal to the span of $\{W_V^{(l)} x_{\text{in}}^{(l)}, W_V^{(l)} x_0\}$:
     $$r_{\text{orth}}^{(l)} = \frac{\|(I - P_{\text{same}}^{(l)}) V^{(l)}\|_F}{\|V^{(l)}\|_F + \epsilon}$$
     Under `true_layer0_v_residual=False` (same-layer `v_proj(x_0)`), $\max_{l \ge 1} r_{\text{orth}}^{(l)} < 10^{-5}$. Under `true_layer0_v_residual=True` (cross-layer $V^{(0)}$), $\text{mean}_{l \ge 1} r_{\text{orth}}^{(l)} \ge 0.15$.
  2. **Computational Call Count**: Across a 6-layer forward pass, `true_layer0_v_residual=True` invokes `v_proj` exactly $L = 6$ times instead of $2L = 12$ times (`v_proj_calls_saved == 6`).
  3. **Controlled Multi-Hop Training Improvement**: On a 120-step controlled synthetic language modeling benchmark ($V = 256, d = 128, L = 6, H = 4, T = 64$), enabling `true_layer0_v_residual=True` reduces final validation cross-entropy by **$\ge 0.06\text{ nats}$** compared to `true_layer0_v_residual=False` for both `QuantumBornGPT` and `ClassicalModdedGPT`.

### Hypothesis 2 (`H2` — Cubic Degree-3 `ReLU^2` MLP Norm Explosion vs. Bounded-Norm Muon & WSD Zero-Floor Cooldown):
- **Claim**: In `ClassicalModdedBlock`, unconstrained `StiefelMuon` (`stiefel_penalty = 0.0`) causes 2D weight singular values $\sigma(W)$ to inflate over training steps, which scales the degree-3 homogeneous `ReLU^2` MLP output ($\propto \sigma^3$) faster than the degree-2 attention output ($\propto \sigma^2$), skewing the MLP-to-Attention RMS ratio $\rho_{\text{mlp/attn}} = \text{RMS}(\text{mlp\_out}) / \text{RMS}(\text{attn\_out})$ and degrading generalization. Enabling continuous Riemannian retraction (`stiefel_penalty = 0.005`) with `true_layer0_v_residual=True` and WSD cooldown to `min_lr_ratio = 0.0` bounds `IsoErr` and improves validation cross-entropy at $T_{\text{val}} = 512$.
- **Observable Metrics & Pass Criteria**:
  1. **Homogeneity Scaling Verification**: Scaling all 2D block weights $(W_Q, W_K, W_V, W_O, W_1, W_2)$ in `ClassicalModdedBlock` by a scalar factor $c = 4.0$ scales $\|\text{attn\_out}\|_2$ by exactly $c^2 = 16.0\times$ (since $Q, K$ are RMSNormed and $V, W_O$ are linear) and scales $\|\text{mlp\_out}\|_2$ by exactly $c^3 = 64.0\times$ (since $\text{ReLU}(h W_1^\top)^2 W_2^\top$ is quadratic in $W_1$ and linear in $W_2$), causing $\rho_{\text{mlp/attn}}(c) / \rho_{\text{mlp/attn}}(1)$ to equal $c = 4.0$ within $< 10^{-4}$ relative error.
  2. **Training Norm Stability & Loss Reduction**: In a controlled 120-step training test ($V = 256, d = 128, L = 6, T = 64$), unconstrained `stiefel_penalty = 0.0` causes `IsoErr` to grow $> 3.0\times$ higher than `stiefel_penalty = 0.005`, while `stiefel_penalty = 0.005` + `true_layer0_v_residual = True` + `wsd_min_lr_ratio = 0.0` keeps `IsoErr <= 1.5` and achieves **$\ge 0.08\text{ nats}$** lower final validation cross-entropy.

### Hypothesis 3 (`H3` — Fused Triton `flash_purified_born_attention` Throughput & Equivalence):
- **Claim**: Routing `SubspaceProjectorBornAttention(use_flash_born=True, use_purified_attn=True)` through the SRAM-tiled Triton kernel `flash_purified_born_attention` matches the reference forward and backward pass of `purified_born_attention_weights` while delivering $\ge 3.0\times$ higher token throughput and $\ge 20\%$ lower peak VRAM than the `use_flash_born=False` chunked Python fallback.
- **Observable Metrics & Pass Criteria**:
  1. **Forward & Backward Numerical Parity**: On CUDA `bfloat16` inputs $(B=4, H=8, T=128, D_h=64)$, `max_abs_diff(out_triton, out_ref) < 5e-3` and relative gradient cosines for $(\nabla Q, \nabla K, \nabla V, \nabla \gamma, \nabla \tau)$ are all **$\ge 0.995$**.
  2. **Throughput & Memory Speedup**: At $(B=64, H=16, T=512, D_h=64)$, `use_flash_born=True` achieves **$\ge 3.0\times$** the throughput (`tok/s`) and **$\le 0.80\times$** the peak CUDA memory of `use_flash_born=False`.

---

## 3. Planned Code & Architectural Changes

### 3.1 Operational Scripting Classification (`Build-Before-Launch` vs. `Nice-to-Have`)
- **`Build-Before-Launch`**:
  - None required beyond existing `scripts/colab_fleet.py` and `scripts/monitor_colab_runs.py` (which already handle auto-provisioning, push, launch, and harvest).
- **`Nice-to-Have` (Logged to `scripts/BACKLOG.md`)**:
  - Add a `--compare-waves w6,w7,w8` CLI flag to `scripts/colab_fleet.py` to print the aligned 50M-token trajectory comparison table directly from `.cache/live_logs/`.

### 3.2 New Diagnostic & Layer Utilities: `src/qdlblocks/layers/value_residual_and_norm_control.py` (and JAX Mirror `src/qdlblocks_jax/value_residual_and_norm_control.py`)
1. `compute_cross_layer_v_orthogonality_metrics(v_mixed: torch.Tensor, x_in: torch.Tensor, x_0: torch.Tensor, w_v_layer: torch.Tensor, eps: float = 1e-8) -> dict[str, float]`:
   - Computes the projection of `v_mixed` (flattened to `(B*T, D)`) onto the subspace spanned by $U_1 = x_{\text{in}} W_V^{(l)\top}$ and $U_2 = x_0 W_V^{(l)\top}$, returning `orth_residual_ratio`, `parallel_energy_ratio`, and `total_v_rms`.
2. `compute_mlp_attn_homogeneity_metrics(block: nn.Module, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scale_factor: float = 4.0) -> dict[str, float]`:
   - Evaluates $\|\text{attn\_out}\|_2$ and $\|\text{mlp\_out}\|_2$ at weight scale $1.0$ and $c = \text{scale\_factor}$, returning `attn_scale_ratio` ($\approx c^2$), `mlp_scale_ratio` ($\approx c^3$), and `dominance_growth_ratio` ($\approx c$).
3. Mirror pure-JAX implementations `compute_cross_layer_v_orthogonality_metrics_jax` and `compute_mlp_attn_homogeneity_metrics_jax` in `src/qdlblocks_jax/value_residual_and_norm_control.py`.

### 3.3 True Cross-Layer $V^{(0)}$ in `SubspaceProjectorBornAttention`, `QuantumBornBlock`, `QuantumBornGPT`, `ClassicalModdedBlock`, and `ClassicalModdedGPT`
1. **`SubspaceProjectorBornAttention` (`src/qdlblocks/layers/attention.py`)**:
   - Add optional parameter `v_0: torch.Tensor | None = None` and `return_v: bool = False` to `forward(...)`:
     - Compute `V = self.v_proj(psi).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)`.
     - If `self.value_residual_mixing and self.v_mix_logit is not None`:
       - If `v_0 is not None` (true cross-layer $V^{(0)}$ from Layer 0): mix `V = (1.0 - alpha) * V + alpha * v_0.to(V.dtype)` directly without calling `self.v_proj` a second time!
       - Else if `psi_0 is not None` (legacy fallback when `v_0` is not provided): compute `V_0 = self.v_proj(l2_normalize(psi_0, dim=-1, eps=self.eps))...`.
     - If `return_v=True`, return `(out, V_raw)` where `V_raw` is the pre-mixed `self.v_proj(psi)` tensor `(B, H, T, D_h)` from Layer 0.
2. **`QuantumBornBlock` (`src/qdlblocks/layers/born_block.py`) & `QuantumBornGPT` (`src/qdlblocks/models/born_gpt.py`)**:
   - Add `true_layer0_v_residual: bool = False` to `QuantumBornGPT.__init__` and `v_0: torch.Tensor | None = None, return_v0: bool = False` to `QuantumBornBlock.forward`.
   - When `true_layer0_v_residual=True`, Layer 0 captures `v_0` (`V` from Layer 0's `v_proj`) and passes `v_0` to Layers $1 \dots L-1$ (both in `QuantumBornGPT.forward` and in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py:forward_backbone`).
3. **`ClassicalModdedBlock` & `ClassicalModdedGPT` (`src/qdlblocks/models/classical_modded_gpt.py`)**:
   - Add `true_layer0_v_residual: bool = False` to `ClassicalModdedGPT.__init__` and `v_0: torch.Tensor | None = None, return_v0: bool = False` to `ClassicalModdedBlock.forward`.
   - When `v_0 is not None` in `ClassicalModdedBlock.forward`, mix `v = (1.0 - alpha_v) * v + alpha_v * v_0.to(v.dtype)` directly without calling `self.v_proj` a second time; in Layer 0 (`v_0 is None`), capture `v_0 = v` before SDPA so Layer 0 does not redundantly mix `v` with itself and passes `v_0` to Layers $1 \dots L-1$.
4. **CLI Benchmark Integration (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)**:
   - Keep validation strictly fixed at `block_size=min(512, T_ep)` (`512`).
   - Ensure `use_flash_born=True` on line 980 so `--purified-born-attn` always uses the fused Triton kernel `flash_purified_born_attention`.
   - Add `--true-layer0-v-residual` and `--classical-stiefel-penalty` (float, default `0.0`, set to `0.005` in Wave-8 bounded-norm runs) and `--wsd-min-lr-ratio` (float, default `0.10`, set to `0.0` for full WSD cooldown).

---

## 4. Verification & Testing Plan

1. **Unit Tests (`tests/test_hypothesis_015_layer0_v_and_norm_control.py`)**:
   - `test_cross_layer_v_orthogonality_and_gemm_reduction`: Verifies `orth_residual_ratio < 1e-5` for same-layer `v_proj(x_0)` vs. `>= 0.15` for cross-layer $V^{(0)}$, and confirms `v_proj` call count drops from $2L$ to $L$ in both `QuantumBornGPT` and `ClassicalModdedGPT`.
   - `test_classical_relu2_cubic_vs_attn_quadratic_homogeneity`: Verifies exact $c^2$ attention scaling vs. $c^3$ `ReLU^2` MLP scaling under weight scaling $c = 4.0$.
   - `test_flash_purified_born_attention_parity_and_gradients`: Verifies CUDA `bfloat16` forward/backward parity between `flash_purified_born_attention` and `purified_born_attention_weights`.
   - `test_jax_torch_parity_015`: Verifies parity between PyTorch and JAX implementations in `value_residual_and_norm_control.py`.
2. **Controlled Hypothesis Script (`examples/25_hypothesis_layer0_v_and_cubic_norm_control.py`)**:
   - Runs controlled `H1`, `H2`, and `H3` experiments, asserts all quantitative pass criteria, and writes `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_015.json`.
