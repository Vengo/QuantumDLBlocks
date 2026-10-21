# Experiment Plan 008: Decoupling Two-Beam Bilinear Interference from Seed-Space Capacity Scaling ($D_{\text{seed}} \in \{256, 384, 512, 1024\}$)

## 1. Executive Summary & Context

In our Wave-2 FineWeb 100M experiments, separating the results by **Backbone MLP (`d_ff`)** and **Seed Readout Head** revealed a critical interaction:
- Runs that preserved the **Two-Beam Bilinear MLP (`TwoBeamQuantumInterference`, `d_ff = 1536x2`)** achieved our top three 3-epoch validation cross-entropies:
  1. `w2_gpu_seed1024` (`D_seed = 1024`, `1536x2` Two-Beam, Linear Seed): **`4.1527` Val CE (`63.6` PPL, `30.85%` Top-1, `51.41%` Top-5)**
  2. `w2_gpu_seed384` (`D_seed = 384`, `1536x2` Two-Beam, Linear Seed): **`4.1725` Val CE (`64.9` PPL, `30.84%` Top-1, `51.15%` Top-5)**
  3. `w2_gpu_adapter256` (`D_seed = 256`, `1536x2` Two-Beam, Plan 007 `OvercompleteSeedReadoutAdapter`): **`4.1849` Val CE (`65.7` PPL, `30.87%` Top-1 — highest Top-1 in project history)**
- By contrast, runs where `--overcomplete-frame` replaced `TwoBeamQuantumInterference` (`1536x2`) with single-beam `OvercompleteFrameInterference` (`2048`) learned faster at Epoch 0.5 (`4.9099` vs `5.1701` CE), but finished `~0.02–0.025` nats higher at Epoch 3.0 (`w2_gpu_adapter384` at `4.1940` CE and `w2_gpu_seed512` at `4.2009` CE).

### Mathematical Root Cause: Degree-3 Cross-Beam Interference vs. Degree-2 Single-Beam Intensity
In `TwoBeamQuantumInterference`, two distinct projections $z_1 = W_{\text{beam1}} \psi \in \mathbb{R}^{M_2}$ and $z_2 = W_{\text{beam2}} \psi \in \mathbb{R}^{M_2}$ interact multiplicatively:
$$h_{\text{2beam}}(\psi) = z_1^{\odot 2} \odot \mathrm{SiLU}(z_2) \in \mathbb{R}^{M_2}$$
Because each coordinate $h_m(\psi) = \langle w_{1,m}, \psi \rangle^2 \, \mathrm{SiLU}(\langle w_{2,m}, \psi \rangle)$ is a **degree-3 cross-subspace observable** coupling two distinct Hilbert directions $w_{1,m} \ne w_{2,m}$, it spans mixed spherical harmonics ($\ell = 1, 2, 3$) and breaks antipodal sign symmetry ($h_{\text{2beam}}(-\psi) \ne \pm h_{\text{2beam}}(\psi)$).
In contrast, single-beam `OvercompleteFrameInterference` computes $h_{\text{1beam}}(\psi) = (W_{\text{up}}\psi)^{\odot 2} \odot \mathrm{sign}(W_{\text{up}}\psi)$, where each channel only depends on a **single** direction $w_m$, so it cannot compute cross-subspace products $\langle w_{1,m}, \psi \rangle^2 \langle w_{2,m}, \psi \rangle$ within a single layer.

This document specifies **Experiment Plan 008**, which:
1. Introduces quantitative diagnostics (`single_beam_signed_born_features`, `two_beam_bilinear_born_features`, `compute_interference_feature_rank`, `compute_cross_beam_coherence`, and `compute_cross_subspace_regression_mse`) in `src/qdlblocks/layers/seed_scaling_diagnostics.py` (and JAX parity in `src/qdlblocks_jax/seed_scaling_diagnostics.py`) to measure the degree-3 cross-subspace expressivity gap between Two-Beam Bilinear Interference (`1.5d x 2`) and Single-Beam Overcomplete Interference (`2.0d`).
2. Tests the clean factorial interaction of preserving `TwoBeamQuantumInterference` in the backbone while scaling `OvercompleteSeedReadoutAdapter` across $D_{\text{seed}}$.

---

## 2. Observable Hypotheses & Quantitative Predictions

### Hypothesis 1 (H1 — Degree-3 Cross-Subspace Feature Rank & Expressivity of Two-Beam vs. Single-Beam Interference):
Let $\psi \in \mathbb{S}^{d-1}$ ($d = 64$) be an isotropic batch of unit-norm quantum states ($N_{\text{train}} = 1024$ training states and $N_{\text{test}} = 512$ held-out test states, generated with `torch.manual_seed(42)` by normalizing i.i.d. $\mathcal{N}(0, 1)$ vectors).
We compare two **architecture-ratio-matched** interference maps mirroring the FineWeb 100M configurations (`d_ff = 2048` single-beam = $2.0 d$ vs `d_ff = 1536x2` two-beam = $1.5 d$ per beam):
- **Map A (`SingleBeamFrame2d`, $M_1 = 2.0 d = 128$)**:
  $$f_{\text{1beam}}(\psi) = (W_{\text{up}} \psi)^{\odot 2} \odot \mathrm{sign}(W_{\text{up}} \psi) \in \mathbb{R}^{128}$$
  where $W_{\text{up}} \in \mathbb{R}^{128 \times 64}$ is initialized via `optimize_overcomplete_tight_frame(128, 64, steps=150, lr=0.05, seed=42)`.
- **Map B (`TwoBeamBilinear15d`, $M_2 = 1.5 d = 96$ per beam)**:
  $$f_{\text{2beam}}(\psi) = (W_1 \psi)^{\odot 2} \odot \mathrm{SiLU}(W_2 \psi) \in \mathbb{R}^{96}$$
  where $W_1 \in \mathbb{R}^{96 \times 64}$ and $W_2 \in \mathbb{R}^{96 \times 64}$ are initialized via `optimize_overcomplete_tight_frame(96, 64, steps=150, lr=0.05, seed=101)` and `seed=202` respectively.
- **Synthetic Cross-Subspace Degree-3 Target $y(\psi) \in \mathbb{R}^{32}$**:
  Construct an orthogonal matrix $U_A \in \mathbb{R}^{64 \times 64}$ via QR decomposition of a seeded Gaussian matrix (`seed=777`), and define $U_B \in \mathbb{R}^{64 \times 64}$ by cyclically shifting the rows of $U_A$ by 1 (`U_B = torch.roll(U_A, shifts=1, dims=0)`), guaranteeing that $U_B$ is also an orthogonal basis and $\langle u_{A,i}, u_{B,i}\rangle = 0$ for every channel $i \in \{1, \dots, 64\}$. Let $P_{\text{proj}} \in \mathbb{R}^{32 \times 64}$ have orthonormal rows (`nn.init.orthogonal_`, `seed=888`), and define:
  $$y(\psi) = d^{3/2} \, P_{\text{proj}}\!\left((U_A \psi)^{\odot 2} \odot (U_B \psi)\right) \in \mathbb{R}^{32}$$
- **Observable Prediction (H1)**:
  1. **Closed-Form Ridge Regression (`compute_cross_subspace_regression_mse`, $\lambda = 10^{-4}$)**:
     Using centered features $\tilde{F} = F - \bar{F}$ and centered targets $\tilde{Y} = Y - \bar{Y}$ with closed-form ridge weights $W^* = (\tilde{F}_{\text{train}}^\top \tilde{F}_{\text{train}} + \lambda I_M)^{-1} \tilde{F}_{\text{train}}^\top \tilde{Y}_{\text{train}}$, `TwoBeamBilinear15d` ($M_2 = 96$) will achieve lower held-out test MSE than `SingleBeamFrame2d` ($M_1 = 128$) despite having fewer output channels ($96$ vs $128$), and when both maps are trained end-to-end for `250` steps of AdamW (`lr = 1e-2`, minimizing $\text{MSE}(\hat{y}(\psi), y(\psi))$ where $\hat{y}(\psi) = W_{\text{down}} f(\psi)$ with trainable $W_{\text{up}}, W_{\text{down}}$ for Map A and trainable $W_1, W_2, W_{\text{down}}$ for Map B, dynamically row-normalizing $W_{\text{up}}, W_1, W_2$ on each forward pass via `single_beam_signed_born_features` / `two_beam_bilinear_born_features` scaled by $d^{3/2}$), `TwoBeamBilinear15d` will achieve at least **`3.0x` lower held-out test MSE** than `SingleBeamFrame2d` ($\text{MSE}_{\text{2beam, e2e}} \le \text{MSE}_{\text{1beam, e2e}} / 3.0$).

### Hypothesis 2 (H2 — Monotonic Scaling of `TwoBeamQuantumInterference` + `OvercompleteSeedReadoutAdapter` Across $D_{\text{seed}}$):
In `QuantumBornGPT` (`vocab_size = 512`, `embed_dim = 64`, `num_heads = 4`, `num_layers = 2`, `hidden_dim = 96`, `block_size = 64`, `readout_mode = "tied_stiefel"`, `use_two_beam_mlp = True`, `200` steps of AdamW at `lr = 3e-3`, `batch_size = 16`):
- **Synthetic Contextual Multi-Subspace Dataset**:
  Generated deterministically (`torch.manual_seed(42)`) using a teacher `QuantumBornGPT` (`vocab_size = 512, embed_dim = 64, seed_dim = 64, num_heads = 4, num_layers = 2, use_two_beam_mlp = True, povm_temperature = 0.015`) that maps `256` random prompt prefixes into argmax/low-temperature token continuations (`200` training sequences and `56` validation sequences of length `65`).
- Compare 4 configurations at `seed_dim = 24`:
  1. **Arm 1 (`1beam_linear_d24`)**: Backbone blocks use `OvercompleteFrameInterference(embed_dim=64, frame_expansion=2.0)` + Linear Seed readout (`use_overcomplete_seed_adapter=False`, `seed_dim=24`).
  2. **Arm 2 (`1beam_adapter_d24`)**: Backbone blocks use `OvercompleteFrameInterference(embed_dim=64, frame_expansion=2.0)` + `OvercompleteSeedReadoutAdapter` (`use_overcomplete_seed_adapter=True`, `seed_dim=24`).
  3. **Arm 3 (`2beam_linear_d24`)**: Backbone blocks use `TwoBeamQuantumInterference(embed_dim=64, hidden_dim=96)` (`use_two_beam_mlp=True`) + Linear Seed readout (`use_overcomplete_seed_adapter=False`, `seed_dim=24`).
  4. **Arm 4 (`2beam_adapter_d24`)**: Backbone blocks use `TwoBeamQuantumInterference(embed_dim=64, hidden_dim=96)` (`use_two_beam_mlp=True`) + `OvercompleteSeedReadoutAdapter` (`use_overcomplete_seed_adapter=True`, `seed_dim=24`).
- **Observable Prediction (H2)**:
  1. Preserving the Two-Beam backbone with `OvercompleteSeedReadoutAdapter` (**Arm 4**) will achieve the **lowest validation cross-entropy of all 4 arms**, beating **Arm 2** (`1beam_adapter_d24`) by at least **`0.05` nats** ($\text{Val CE}_{\text{Arm 4}} \le \text{Val CE}_{\text{Arm 2}} - 0.05$) and beating **Arm 3** (`2beam_linear_d24`) by at least **`0.15` nats** ($\text{Val CE}_{\text{Arm 4}} \le \text{Val CE}_{\text{Arm 3}} - 0.15$).
  2. Across `seed_dim` $\in \{16, 24, 32\}$ with the Two-Beam backbone + `OvercompleteSeedReadoutAdapter`, validation cross-entropy will decrease strictly monotonically ($\text{Val CE}_{d=32} < \text{Val CE}_{d=24} < \text{Val CE}_{d=16}$) while empirical seed quantum rank $R_{\text{eff}}(\rho_{\text{seed}})$ (computed via `effective_quantum_rank(empirical_state_density(phi_seed))` on the pre-readout seed state $\phi_{\text{seed}} \in \mathbb{S}^{D_{\text{seed}}-1}$ over the validation set) increases strictly monotonically ($R_{\text{eff}, 32} > R_{\text{eff}, 24} > R_{\text{eff}, 16}$).

---

## 3. Planned Code & Architectural Changes

### 3.1 PyTorch Diagnostics Module (`src/qdlblocks/layers/seed_scaling_diagnostics.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/seed_scaling_diagnostics.py` containing:
1. `single_beam_signed_born_features(psi: torch.Tensor, W_up: torch.Tensor, eps: float = 1e-12) -> torch.Tensor`:
   - Accepts `psi` of shape `(..., D)` and `W_up` of shape `(M, D)`.
   - Row-normalizes `W_unit = l2_normalize(W_up, dim=-1, eps=eps).to(psi.dtype)`.
   - Computes `z = F.linear(psi, W_unit)` of shape `(..., M)` and returns `(z * z) * torch.sign(z)`.
2. `two_beam_bilinear_born_features(psi: torch.Tensor, W_beam1: torch.Tensor, W_beam2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor`:
   - Accepts `psi` of shape `(..., D)` and `W_beam1, W_beam2` of shape `(M, D)`.
   - Row-normalizes `W1_unit = l2_normalize(W_beam1, dim=-1, eps=eps).to(psi.dtype)` and `W2_unit = l2_normalize(W_beam2, dim=-1, eps=eps).to(psi.dtype)`.
   - Computes `z1 = F.linear(psi, W1_unit)` and `z2 = F.linear(psi, W2_unit)` of shape `(..., M)`.
   - Returns `(z1 * z1) * F.silu(z2)`.
3. `compute_interference_feature_rank(features: torch.Tensor, eps: float = 1e-12) -> dict[str, float]`:
   - Accepts `features` of arbitrary leading dimensions `(..., M)` and flattens to `flat = features.reshape(-1, features.shape[-1]).to(torch.float64)` of shape `(N, M)`.
   - If `N <= 1`, returns `{"feature_eff_rank": 1.0, "feature_participation_ratio": 1.0, "feature_trace": 0.0}`.
   - Centers across dim `0`: `f_c = flat - flat.mean(dim=0, keepdim=True)`.
   - Computes covariance `C = (f_c.T @ f_c) / float(N - 1)` of shape `(M, M)`.
   - Computes eigenvalues `eigvals = torch.linalg.eigvalsh(C).clamp(min=0.0)`.
   - Normalizes `p = eigvals / (eigvals.sum() + eps)`.
   - Returns:
     - `"feature_eff_rank"`: `float(torch.exp(-(p * torch.log(p + eps)).sum()).item())`
     - `"feature_participation_ratio"`: `float((1.0 / ((p * p).sum() + eps)).item())`
     - `"feature_trace"`: `float(eigvals.sum().item())`
4. `compute_cross_beam_coherence(W_beam1: torch.Tensor, W_beam2: torch.Tensor, eps: float = 1e-12) -> dict[str, float]`:
   - Row-normalizes `W1_u = l2_normalize(W_beam1.detach().float(), dim=-1, eps=eps)` and `W2_u = l2_normalize(W_beam2.detach().float(), dim=-1, eps=eps)` of shape `(M, D)`.
   - Computes row-wise inner products `row_cos = (W1_u * W2_u).sum(dim=-1)` of shape `(M,)`.
   - Returns:
     - `"rms_paired_beam_coherence"`: `float(torch.sqrt((row_cos * row_cos).mean()).item())`
     - `"max_paired_beam_coherence"`: `float(row_cos.abs().max().item())`
5. `compute_cross_subspace_regression_mse(features_train: torch.Tensor, targets_train: torch.Tensor, features_test: torch.Tensor, targets_test: torch.Tensor, ridge_lambda: float = 1e-4) -> dict[str, float]`:
   - Flattens `features_train` to `(N_train, M)`, `targets_train` to `(N_train, K)`, `features_test` to `(N_test, M)`, `targets_test` to `(N_test, K)` in `torch.float64`.
   - Computes training means `f_mean = features_train.mean(dim=0, keepdim=True)` and `y_mean = targets_train.mean(dim=0, keepdim=True)`.
   - Centers `F_tr = features_train - f_mean`, `Y_tr = targets_train - y_mean`, `F_te = features_test - f_mean`, `Y_te = targets_test - y_mean`.
   - Solves `W = torch.linalg.solve(F_tr.T @ F_tr + ridge_lambda * torch.eye(M, dtype=torch.float64, device=F_tr.device), F_tr.T @ Y_tr)`.
   - Returns `{"train_mse": float(((F_tr @ W - Y_tr) ** 2).mean().item()), "test_mse": float(((F_te @ W - Y_te) ** 2).mean().item())}`.

### 3.2 Pure JAX Parity (`src/qdlblocks_jax/seed_scaling_diagnostics.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/seed_scaling_diagnostics.py` containing:
- `single_beam_signed_born_features_jax(psi: jax.Array, W_up: jax.Array, eps: float = 1e-12) -> jax.Array`
- `two_beam_bilinear_born_features_jax(psi: jax.Array, W_beam1: jax.Array, W_beam2: jax.Array, eps: float = 1e-12) -> jax.Array`
- `compute_interference_feature_rank_jax(features: jax.Array, eps: float = 1e-12) -> dict[str, jax.Array]`
- `compute_cross_beam_coherence_jax(W_beam1: jax.Array, W_beam2: jax.Array, eps: float = 1e-12) -> dict[str, jax.Array]`

### 3.3 Controlled Verification Script (`examples/18_hypothesis_twobeam_seed_scaling.py`)
Create `/home/festivity/agy/QuantumDLBlocks/examples/18_hypothesis_twobeam_seed_scaling.py` to:
1. Run Experiment 1 (H1 — Cross-Subspace Degree-3 Target Fitting: Ridge Regression + End-to-End 250-Step Training) and verify that `TwoBeamBilinear15d` achieves lower ridge test MSE and $\ge 3.0\times$ lower end-to-end test MSE than `SingleBeamFrame2d`.
2. Run Experiment 2 (H2 — 4-Arm `1beam` vs `2beam` $\times$ `linear` vs `seed_adapter` comparison at `seed_dim = 24`, plus `seed_dim` $\in \{16, 24, 32\}$ monotonic scaling curve) and verify all quantitative predictions.
3. Write all results to `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_008.json`.

---

## 4. Verification & Unit Testing Plan (`tests/test_hypothesis_008_twobeam_seed_scaling.py`)

Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_008_twobeam_seed_scaling.py` with 4 unit tests:
1. `test_single_vs_twobeam_feature_shapes_and_gradients()`:
   - Verifies shapes `(B, T, M)` and finite, non-zero gradients for `single_beam_signed_born_features` and `two_beam_bilinear_born_features`.
2. `test_interference_feature_rank_and_cross_beam_coherence()`:
   - Verifies that `compute_interference_feature_rank` works on both 2D `(N, M)` and 3D `(B, T, M)` inputs (and handles `N=1` gracefully), and that orthogonalized paired beams `(W_beam1, W_beam2)` yield `rms_paired_beam_coherence < 1e-5`.
3. `test_twobeam_degree3_cross_subspace_advantage()`:
   - Verifies on the synthetic degree-3 cross-subspace target ($d = 32, M_1 = 64, M_2 = 48$) that end-to-end trained `two_beam_bilinear_born_features` (`150` steps AdamW, `lr=1e-2`) achieves $\ge 2.5\times$ lower held-out test MSE than `single_beam_signed_born_features`.
4. `test_pytorch_jax_diagnostics_equivalence()`:
   - Verifies that PyTorch and JAX implementations of `single_beam_signed_born_features`, `two_beam_bilinear_born_features`, `compute_interference_feature_rank`, and `compute_cross_beam_coherence` match within `max_abs_diff < 1e-5`.
