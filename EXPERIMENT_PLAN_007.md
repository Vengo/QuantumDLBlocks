# Experiment Plan 007: Breaking the Linear Seed Nullspace Bottleneck via Overcomplete Born Seed Readout Adapters (`OvercompleteSeedReadoutAdapter`)

## 1. Executive Summary & Context

In our Wave-1 FineWeb 100M benchmark suite (`100M` training tokens per epoch, `4,999,488`-token validation slice, `vocab_size = 50304`, `embed_dim = 1024`), every single architectural intervention that expanded capacity at the seed readout interface ($D_{\text{seed}}$) produced a major validation loss improvement:
- **`gpu_exp03` (`D_seed = 256` vs `D_seed = 128` Cosine, 3 Epochs / 300M tok)**: Validation cross-entropy improved by **`-0.1307` nats** (`4.2181` vs `4.3488` CE), beating even the 5-epoch (500M tok) `D_seed = 128` record (`4.2676` CE) in 3 epochs and raising the empirical seed state effective rank from $R_{\text{eff}}(\rho_{\text{seed}}) = 26.2$ to $41.5$.
- **`gpu_slowrun` (`D_seed = 256` vs `D_seed = 128` WSD + Strided MTP=3, 3 Epochs / 300M tok)**: Validation cross-entropy improved by **`-0.1264` nats** (`4.2516` vs `4.3780` CE).
- **`tpu_exp01` (`D_seed = 256` vs `D_seed = 128` on Cloud TPU v5e-1, 3 Epochs / 300M tok)**: Validation cross-entropy improved from `4.4849` to **`4.4592` CE** ($R_{\text{eff}}(\rho_{\text{seed}}) = 49.7$).
- **`gpu_exp06` (Plan 004 `MultiBeamKrausReadout` with $K = 4$ beams at `D_seed = 128`)**: Validation cross-entropy improved from `4.3488` to **`4.3270` CE** (`-0.0218` nats).

### Why Is the Standard Seed Collapse Layer Such a Severe Bottleneck?

In `QuantumBornGPT` (`tied_stiefel` and `symmetric_dual` modes), the $D$-dimensional contextual quantum state $\psi \in \mathbb{S}^{D-1}$ ($D = 1024$) is collapsed into the $D_{\text{seed}}$-dimensional seed space $\mathbb{S}^{D_{\text{seed}}-1}$ before computing vocabulary logits against $M_{\text{seed}} \in \mathbb{R}^{V \times D_{\text{seed}}}$:
$$\psi_{\text{final}} = \mathrm{Normalize}(R_{\text{rot}} \psi), \qquad \phi_{\text{seed}} = \mathrm{Normalize}(W_{\text{lift}}^\top \psi_{\text{final}}) \in \mathbb{S}^{D_{\text{seed}}-1}$$
Because scalar normalization commutes with linear matrix multiplication ($\mathrm{Normalize}(c \cdot v) = \mathrm{Normalize}(v)$ for any $c > 0$), the composition of $R_{\text{rot}} \in \mathbb{R}^{D \times D}$ and $W_{\text{lift}}^\top \in \mathbb{R}^{D_{\text{seed}} \times D}$ is **mathematically equivalent to a single linear matrix** $W_{\text{eff}} = W_{\text{lift}}^\top R_{\text{rot}} \in \mathbb{R}^{D_{\text{seed}} \times D}$:
$$\phi_{\text{seed}}(\psi) = \mathrm{Normalize}(W_{\text{eff}} \psi)$$
This purely linear collapse suffers from two fundamental mathematical limitations:
1. **$(D - D_{\text{seed}})$-Dimensional Linear Nullspace Blindness**: By the rank-nullity theorem, $\operatorname{rank}(W_{\text{eff}}) \le D_{\text{seed}}$, so $\dim(\ker(W_{\text{eff}})) \ge D - D_{\text{seed}}$. At $D = 1024, D_{\text{seed}} = 128$, at least **$896$ orthogonal dimensions ($87.5\%$ of the backbone Hilbert space $\mathcal{H}_D$)** lie in the nullspace $\ker(W_{\text{eff}})$ and are completely invisible to the seed readout!
2. **Purely Degree-1 Spherical Harmonic Expressivity**: The linear map $W_{\text{eff}} \psi$ only extracts degree-1 spherical harmonics ($\ell = 1$) of $\psi \in \mathbb{S}^{D-1}$. It cannot extract non-linear quadratic Born interference observables $|\langle u_m \mid \psi \rangle|^2$ ($\ell = 2$ spherical harmonics) from the contextual state $\psi$ prior to vocabulary readout.

This document specifies **Experiment Plan 007**, which tests whether augmenting the $D \to D_{\text{seed}}$ seed collapse layer with an **Overcomplete Born Seed Readout Adapter (`OvercompleteSeedReadoutAdapter`)** breaks the $(D - D_{\text{seed}})$-dimensional linear nullspace bottleneck, increases the effective quantum state rank $R_{\text{eff}}(\rho_{\text{seed}})$, and lowers validation cross-entropy at fixed $D_{\text{seed}}$.

---

## 2. Observable Hypotheses & Quantitative Predictions

### Hypothesis 1 (H1 — Breaking Linear Nullspace Blindness & Expanding Input-Sensitivity Rank via Overcomplete Born Seed Readout):
Let $\psi \in \mathbb{S}^{D-1}$ be a unit-norm contextual quantum state ($D = 128$) collapsed into a compact seed state $\phi_{\text{seed}} \in \mathbb{S}^{D_{\text{seed}}-1}$ ($D_{\text{seed}} = 32$).
- **Baseline Linear Seed Collapse (`LinearSeedCollapse`)**:
  $$\phi_{\text{lin}}(\psi) = \mathrm{Normalize}(W_{\text{lin}} \psi), \qquad W_{\text{lin}} \in \mathbb{R}^{D_{\text{seed}} \times D}$$
  Because $\operatorname{rank}(W_{\text{lin}}) \le D_{\text{seed}}$, the expected input-sensitivity (pullback metric) matrix over $N$ isotropic states $\{\psi_i\}_{i=1}^N \subset \mathbb{S}^{D-1}$,
  $$G(\phi) = \frac{1}{N} \sum_{i=1}^N J_\phi(\psi_i)^\top J_\phi(\psi_i) \in \mathbb{R}^{D \times D}, \qquad J_\phi(\psi) = \frac{\partial \phi(\psi)}{\partial \psi} \in \mathbb{R}^{D_{\text{seed}} \times D},$$
  has spectral effective rank $R_{\text{eff}}(G(\phi_{\text{lin}})) \le D_{\text{seed}} = 32$, and any vector $v_\perp \in \ker(W_{\text{lin}})$ yields zero directional sensitivity $\|J_{\phi_{\text{lin}}}(\psi) v_\perp\|_2 = 0$.
- **Overcomplete Born Seed Readout Adapter (`OvercompleteSeedReadoutAdapter`)**:
  Combines the linear Stiefel seed path $W_{\text{lin}} \psi \in \mathbb{R}^{D_{\text{seed}}}$ with an overcomplete tight-frame quadratic Born interference branch $U_{\text{frame}} \in \mathbb{R}^{M \times D}$ ($M = 2 D = 256$ unit-norm frame rows in $\mathbb{R}^D$, initialized via Welch-bound tight-frame optimization):
  $$q(\psi) = \frac{D}{\sqrt{D_{\text{seed}}}} \left((U_{\text{frame}} \psi)^{\odot 2} - \frac{1}{D} \mathbf{1}_M\right) \in \mathbb{R}^M$$
  $$\phi_{\text{over}}(\psi) = \mathrm{Normalize}\!\left(\phi_{\text{lin}}(\psi) + g_{\text{seed}} \odot (W_{\text{down}} q(\psi))\right) \in \mathbb{S}^{D_{\text{seed}}-1}$$
  where $\frac{D}{\sqrt{D_{\text{seed}}}}$ restores spherical coordinate variance parity between the $O(1/D)$ quadratic Born intensities $(U_{\text{frame}}\psi)^{\odot 2} - \frac{1}{D}\mathbf{1}_M$ and the unit-sphere state $\phi_{\text{lin}}(\psi) \in \mathbb{S}^{D_{\text{seed}}-1}$ (even when $W_{\text{down}} \in \mathbb{R}^{D_{\text{seed}} \times M}$ is retracted to unit singular values by `StiefelMuon`), and $g_{\text{seed}} \in \mathbb{R}^{D_{\text{seed}}}$ is a learnable channel gate (initialized to $g_0 = 0.5$).
- **Observable Prediction (H1)**:
  In a controlled experiment ($D = 128, D_{\text{seed}} = 32, M = 256, N = 512$ states on $\mathbb{S}^{D-1}$):
  1. The pullback input-sensitivity effective rank $R_{\text{eff}}(G) = \exp\!\left(-\sum_{j=1}^D \hat{\lambda}_j \ln \hat{\lambda}_j\right)$ (where $\hat{\lambda}_j = \lambda_j / \sum_k \lambda_k$ are the normalized eigenvalues of $G(\phi) \in \mathbb{R}^{D \times D}$) will satisfy:
     $$R_{\text{eff}}(G(\phi_{\text{lin}})) \le D_{\text{seed}} = 32.0, \qquad R_{\text{eff}}(G(\phi_{\text{over}})) \ge 2.2 \times R_{\text{eff}}(G(\phi_{\text{lin}})) \ge 70.0$$
  2. When fitting a synthetic multi-subspace target observable $y(\psi) \in \mathbb{S}^{D_{\text{seed}}-1}$ that depends on both $\operatorname{row}(W_{\text{lin}})$ and its orthogonal complement $\ker(W_{\text{lin}})$ over `250` steps of AdamW (`lr = 3e-3`), `OvercompleteSeedReadoutAdapter` will achieve at least **`3.0x` lower held-out test MSE** than `LinearSeedCollapse` (which is mathematically blind to $\ker(W_{\text{lin}})$ even when $W_{\text{lin}}$ is trained, because any single linear projection has rank $\le D_{\text{seed}}$).

### Hypothesis 2 (H2 — Higher Seed Quantum State Rank $R_{\text{eff}}(\rho_{\text{seed}})$ & Lower Validation CE in `QuantumBornGPT` at Fixed $D_{\text{seed}}$):
When `OvercompleteSeedReadoutAdapter` is integrated into `QuantumBornGPT` (`readout_mode = "tied_stiefel"`, enabled via `use_overcomplete_seed_adapter: bool = True`):
- **Observable Prediction (H2)**:
  In a controlled 3-arm language modeling benchmark (`vocab_size = 512`, `embed_dim = 64`, `num_heads = 4`, `num_layers = 2`, `block_size = 64`, `200` training steps at `batch_size = 16`):
  - **Arm A (`linear_seed_d16`)**: Standard `QuantumBornGPT` with `seed_dim = 16`, `use_overcomplete_seed_adapter = False`.
  - **Arm B (`overcomplete_seed_d16`)**: `QuantumBornGPT` with `seed_dim = 16`, `use_overcomplete_seed_adapter = True` ($M = 2 D = 128$).
  - **Arm C (`linear_seed_d32`)**: Expanded linear seed reference with `seed_dim = 32`, `use_overcomplete_seed_adapter = False`.
  We predict that at step 200:
  1. Arm B (`overcomplete_seed_d16`) will achieve a strictly lower validation cross-entropy than Arm A (`linear_seed_d16`) by at least **`0.10` nats** ($\text{Val CE}_B \le \text{Val CE}_A - 0.10$), recovering at least **$50\%$** of the validation CE gain of doubling `seed_dim` to $32$ (Arm C) without increasing the vocabulary table width $D_{\text{seed}}$!
  2. Arm B (`overcomplete_seed_d16`) will achieve a strictly higher seed state effective quantum rank $R_{\text{eff}}(\rho_{\text{seed}})$ than Arm A (`linear_seed_d16`):
     $$R_{\text{eff}}(\rho_{\text{seed}, B}) \ge 1.15 \times R_{\text{eff}}(\rho_{\text{seed}, A})$$

---

## 3. Planned Code & Architectural Changes

### 3.1 PyTorch Module (`src/qdlblocks/layers/seed_adapter.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/seed_adapter.py` (and export in `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/__init__.py`) containing:

1. `compute_input_sensitivity_effective_rank(module: nn.Module, psi: torch.Tensor, eps: float = 1e-12) -> dict[str, float]`:
   - Given a callable `module` mapping `(N, D) -> (N, D_seed)` and a batch of inputs `psi` of shape `(N, D)`:
   - Computes the exact Jacobian $J(\psi_i) \in \mathbb{R}^{D_{\text{seed}} \times D}$ per sample (using `torch.func.vmap(torch.func.jacrev(...))` or explicit autograd per output coordinate over the batch) to form the empirical pullback metric:
     $$G = \frac{1}{N} \sum_{i=1}^N J(\psi_i)^\top J(\psi_i) \in \mathbb{R}^{D \times D}$$
   - Computes the eigenvalues $\lambda_1 \ge \dots \ge \lambda_D \ge 0$ of $G$ via `torch.linalg.eigvalsh(G.to(torch.float64)).clamp(min=0.0)`.
   - Normalizes $\hat{\lambda}_j = \lambda_j / (\sum_k \lambda_k + \varepsilon)$ and returns:
     - `"pullback_eff_rank"`: $\exp\!\left(-\sum_j \hat{\lambda}_j \ln(\hat{\lambda}_j + \varepsilon)\right)$
     - `"pullback_participation_ratio"`: $1.0 / \sum_j \hat{\lambda}_j^2$
     - `"trace_G"`: $\sum_j \lambda_j$

2. `OvercompleteSeedReadoutAdapter(nn.Module)`:
   - Constructor signature:
     ```python
     class OvercompleteSeedReadoutAdapter(nn.Module):
         def __init__(
             self,
             embed_dim: int,
             seed_dim: int,
             frame_expansion: int = 2,
             init_gate: float = 0.5,
             tight_frame_init_steps: int = 80,
             eps: float = 1e-12,
         ) -> None:
     ```
   - Validates `embed_dim >= 1`, `seed_dim >= 1`, `frame_expansion >= 1`.
   - Sets `self.frame_dim = frame_expansion * embed_dim` ($M = \text{frame\_expansion} \times D$).
   - Parameters:
     - `self.U_frame = nn.Parameter(W_init)` of shape `(frame_dim, embed_dim)`, initialized via `optimize_overcomplete_tight_frame(frame_dim, embed_dim, steps=tight_frame_init_steps, lr=0.05)` from `..layers.overcomplete_frame` so that its rows start as unit vectors near a Welch-bounded tight frame.
     - `self.W_down = nn.Parameter(torch.empty(seed_dim, frame_dim))`, initialized via `nn.init.orthogonal_(self.W_down)` (which produces orthonormal rows since `seed_dim <= frame_dim`).
     - `self.seed_gate = nn.Parameter(torch.full((seed_dim,), float(init_gate), dtype=torch.float32))`.
   - Method `forward(self, psi_final: torch.Tensor, phi_linear: torch.Tensor) -> torch.Tensor`:
     - Inputs:
       - `psi_final`: Normalized contextual state on $\mathbb{S}^{D-1}$ of shape `(..., embed_dim)`.
       - `phi_linear`: Base linear seed state on $\mathbb{S}^{D_{\text{seed}}-1}$ of shape `(..., seed_dim)` (e.g., from `hilbert_lift.collapse_to_seed(psi_final)`).
     - Normalizes rows of `U_frame` to unit $\ell_2$ norm:
       `U_unit = l2_normalize(self.U_frame, dim=-1, eps=self.eps).to(psi_final.dtype)`
     - Computes overcomplete frame projections:
       `z = F.linear(psi_final, U_unit)` of shape `(..., frame_dim)`
     - Computes centered and spherical-variance-normalized quadratic Born intensities on $\mathbb{S}^{D-1}$:
       `scale = float(self.embed_dim) / math.sqrt(float(self.seed_dim))`
       `q = (z * z - (1.0 / float(self.embed_dim))) * scale` of shape `(..., frame_dim)`
     - Projects `q` into seed space and scales by `seed_gate`:
       `delta_seed = F.linear(q, self.W_down.to(psi_final.dtype)) * self.seed_gate.to(psi_final.dtype)`
     - Combines with `phi_linear` and normalizes onto $\mathbb{S}^{D_{\text{seed}}-1}$:
       `return l2_normalize(phi_linear + delta_seed, dim=-1, eps=self.eps)`
   - Method `frame_metrics(self) -> dict[str, float]`:
     - Delegates to `frame_coherence_metrics(self.U_frame, eps=self.eps)` and adds `"mean_abs_seed_gate": float(self.seed_gate.detach().abs().mean().item())`.

### 3.2 Integration into `QuantumBornGPT` (`src/qdlblocks/models/born_gpt.py`) and CLI (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)
1. In `QuantumBornGPT.__init__`:
   - Add keyword argument `use_overcomplete_seed_adapter: bool = False` (default `False` so existing behavior is 100% unchanged unless enabled).
   - Store `self.use_overcomplete_seed_adapter = bool(use_overcomplete_seed_adapter)`.
   - If `use_overcomplete_seed_adapter` is `True`, instantiate:
     ```python
     self.seed_adapter: OvercompleteSeedReadoutAdapter | None = OvercompleteSeedReadoutAdapter(
         embed_dim=embed_dim,
         seed_dim=seed_dim,
         frame_expansion=2,
         eps=eps,
     )
     ```
     Otherwise set `self.seed_adapter = None`.
2. In `QuantumBornGPT.forward` and `QuantumBornGPT.forward_mtp_logits`:
   - Whenever `phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)` is computed (in both `tied_stiefel` and `symmetric_dual` branches), if `self.seed_adapter is not None`, immediately apply:
     ```python
     phi_seed = self.seed_adapter(psi_final, phi_seed)
     ```
3. In `/home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py`:
   - Add CLI flag `--overcomplete-seed-adapter` (`action="store_true"`) and pass `use_overcomplete_seed_adapter=args.overcomplete_seed_adapter` to `QuantumBornGPT(...)` on the PyTorch path.
   - Ensure `seed_adapter.U_frame` and `seed_adapter.W_down` are routed to the Stiefel optimizer group (`stiefel_params`), while `seed_adapter.seed_gate` (1D parameter) is routed to `adam_params`.

### 3.3 Pure JAX Functional Parity (`src/qdlblocks_jax/seed_adapter.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/seed_adapter.py` (and export in `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/__init__.py`) containing:
- `overcomplete_seed_adapter_forward_jax(psi_final: jax.Array, phi_linear: jax.Array, U_frame: jax.Array, W_down: jax.Array, seed_gate: jax.Array, eps: float = 1e-12) -> jax.Array`:
  - Pure JAX implementation of the exact same forward equation:
    - Row-normalizes `U_frame` via `l2_normalize_jax(U_frame, axis=-1, eps=eps)`
    - Computes `z = jnp.matmul(psi_final, jnp.swapaxes(U_unit, -1, -2))`
    - Computes `scale = float(psi_final.shape[-1]) / math.sqrt(float(phi_linear.shape[-1]))`
    - Computes `q = (z * z - (1.0 / float(psi_final.shape[-1]))) * scale`
    - Computes `delta_seed = jnp.matmul(q, jnp.swapaxes(W_down, -1, -2)) * seed_gate`
    - Returns `l2_normalize_jax(phi_linear + delta_seed, axis=-1, eps=eps)`

### 3.4 Controlled Verification Script (`examples/17_hypothesis_overcomplete_seed_adapter.py`)
Create `/home/festivity/agy/QuantumDLBlocks/examples/17_hypothesis_overcomplete_seed_adapter.py` that:
1. **Runs Experiment 1 (H1 — Pullback Input-Sensitivity Rank & Nullspace Target Fitting)**:
   - Sets `torch.manual_seed(42)`, $D = 128$, $D_{\text{seed}} = 32$, $M = 256$, $N_{\text{train}} = 512$, $N_{\text{test}} = 256$.
   - Constructs a fixed linear seed projection $W_{\text{lin}} \in \mathbb{R}^{32 \times 128}$ (orthonormal rows) and evaluates `compute_input_sensitivity_effective_rank` on:
     - `LinearSeedCollapse`: $\phi_{\text{lin}}(\psi) = \mathrm{Normalize}(W_{\text{lin}} \psi)$
     - `OvercompleteSeedReadoutAdapter`: $\phi_{\text{over}}(\psi) = \text{adapter}(\psi, \phi_{\text{lin}}(\psi))$
   - Verifies $R_{\text{eff}}(G(\phi_{\text{lin}})) \le 32.0 + 10^{-4}$ and $R_{\text{eff}}(G(\phi_{\text{over}})) \ge 2.2 \times R_{\text{eff}}(G(\phi_{\text{lin}}))$.
   - Constructs a ground-truth multi-subspace target $y(\psi) = \mathrm{Normalize}\!\left(W_{\text{lin}} \psi + 0.8 \cdot P_{\text{null}} \left((V_{\text{null}} \psi)^{\odot 2} - \frac{1}{D}\mathbf{1}\right)\right) \in \mathbb{S}^{D_{\text{seed}}-1}$ where $V_{\text{null}} \in \mathbb{R}^{64 \times 128}$ lies in $\ker(W_{\text{lin}})$ ($V_{\text{null}} W_{\text{lin}}^\top = 0$).
   - Trains a trainable single-matrix linear collapse $\hat{\phi}_{\text{lin}}(\psi) = \mathrm{Normalize}(\hat{W} \psi)$ ($\hat{W} \in \mathbb{R}^{32 \times 128}$) vs a trainable `OvercompleteSeedReadoutAdapter` for `250` steps of AdamW (`lr = 3e-3`) minimizing $\text{MSE}(\hat{\phi}(\psi), y(\psi))$.
   - Verifies that `OvercompleteSeedReadoutAdapter` achieves at least **`3.0x` lower held-out test MSE** than the trainable linear collapse.
2. **Runs Experiment 2 (H2 — 3-Arm `QuantumBornGPT` Seed Capacity Benchmark)**:
   - Compares Arm A (`linear_seed_d16`), Arm B (`overcomplete_seed_d16`), and Arm C (`linear_seed_d32`) on a synthetic contextual multi-subspace sequence task (`vocab_size = 512`, `embed_dim = 64`, `num_heads = 4`, `num_layers = 2`, `block_size = 64`, `steps = 200`, `batch_size = 16`, `lr = 3e-3`, `readout_mode = "tied_stiefel"`).
   - Measures final `val_ce` and `seed_eff_rank` ($R_{\text{eff}}(\rho_{\text{seed}})$ on the pre-readout seed states $\phi_{\text{seed}}$).
   - Verifies $\text{Val CE}_B \le \text{Val CE}_A - 0.10$ and $R_{\text{eff}}(\rho_{\text{seed}, B}) \ge 1.15 \times R_{\text{eff}}(\rho_{\text{seed}, A})$.
3. Saves all quantitative metrics to `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_007.json`.

---

## 4. Verification & Unit Testing Plan (`tests/test_hypothesis_007_seed_adapter.py`)

Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_007_seed_adapter.py` with the following unit tests:
1. `test_overcomplete_seed_adapter_unit_norm_and_gradients()`:
   - Verifies that `OvercompleteSeedReadoutAdapter(embed_dim=64, seed_dim=16)` outputs strictly unit-norm seed states (`torch.allclose(out.norm(dim=-1), torch.ones_like(out[..., 0]), atol=1e-5)`) for input shape `(4, 16, 64)` and produces finite, non-zero gradients for `psi_final`, `phi_linear`, `U_frame`, `W_down`, and `seed_gate`.
2. `test_input_sensitivity_effective_rank_expansion()`:
   - Verifies that `compute_input_sensitivity_effective_rank` returns `pullback_eff_rank <= 16.01` for a linear projection $\mathbb{R}^{64} \to \mathbb{S}^{15}$ and `pullback_eff_rank >= 36.0` when wrapped with `OvercompleteSeedReadoutAdapter(embed_dim=64, seed_dim=16)`.
3. `test_quantum_born_gpt_with_overcomplete_seed_adapter()`:
   - Verifies that `QuantumBornGPT(vocab_size=128, embed_dim=32, seed_dim=8, num_heads=4, num_layers=2, readout_mode="tied_stiefel", use_overcomplete_seed_adapter=True, mtp_horizons=2)` executes forward, MTP loss, and `forward_mtp_logits` cleanly with finite gradients on `seed_adapter` parameters.
4. `test_pytorch_jax_seed_adapter_equivalence()`:
   - Verifies that `OvercompleteSeedReadoutAdapter` in PyTorch and `overcomplete_seed_adapter_forward_jax` in JAX produce identical outputs (`max_abs_diff < 1e-5`) given identical weights and inputs.
