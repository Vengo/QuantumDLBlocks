# Experiment Plan 010: Phase-Sensitive Two-Beam Bilinear Seed Readout Adapter (`TwoBeamSeedReadoutAdapter`) — Breaking Antipodal Sign Degeneracy on the Seed Nullspace

## 1. Executive Summary & Context

In **Experiment Plan 007**, introducing `OvercompleteSeedReadoutAdapter` at the $D \to D_{\text{seed}}$ seed collapse interface improved 3-epoch FineWeb 100M validation cross-entropy at `D_seed = 256` from `4.2181` to **`4.1849` CE (`65.7` PPL)** and achieved **`30.87%` Top-1 accuracy** (our highest Top-1 accuracy across all 24 runs).

### Mathematical Limitation of Single-Beam Quadratic Seed Adapters: Antipodal Sign Blindness on $\ker(W_{\text{lin}})$
Recall the forward equation of `OvercompleteSeedReadoutAdapter` (Plan 007):
$$q(\psi) = \frac{D}{\sqrt{D_{\text{seed}}}}\left((U_{\text{frame}}\psi)^{\odot 2} - \frac{1}{D}\mathbf{1}_M\right), \qquad \phi_{\text{over}}(\psi) = \mathrm{Normalize}\!\left(\phi_{\text{lin}}(\psi) + g_{\text{seed}} \odot (W_{\text{down}} q(\psi))\right)$$
Consider any decomposition of the contextual state $\psi = \psi_\parallel + \psi_\perp \in \mathbb{S}^{D-1}$, where $\psi_\parallel \in \operatorname{row}(W_{\text{lin}})$ (the $D_{\text{seed}}$-dimensional linear seed subspace) and $\psi_\perp \in \ker(W_{\text{lin}})$ (the $(D - D_{\text{seed}})$-dimensional linear nullspace).
- If $\psi$ lies primarily in $\ker(W_{\text{lin}})$ (or when we flip the sign of $\psi$: $\psi \mapsto -\psi$), note that:
  1. For any pure nullspace state $\psi_\perp \in \ker(W_{\text{lin}})$, $\phi_{\text{lin}}(\psi_\perp) = 0$, and $q(-\psi_\perp) = (U_{\text{frame}}(-\psi_\perp))^{\odot 2} - \frac{1}{D}\mathbf{1}_M = q(\psi_\perp)$!
  2. Therefore, `OvercompleteSeedReadoutAdapter` is **strictly invariant under sign inversion on $\ker(W_{\text{lin}})$**: $\phi_{\text{over}}(-\psi_\perp) = +\phi_{\text{over}}(\psi_\perp)$! It can only extract **even-degree ($\ell = 2$) spherical harmonics** from the $(D - D_{\text{seed}})$-dimensional nullspace, leaving all **odd-degree ($\ell = 1, 3$) phase/sign information** in $\ker(W_{\text{lin}})$ unrecoverable!

### Solution: Phase-Sensitive Two-Beam Bilinear Seed Readout Adapter (`TwoBeamSeedReadoutAdapter`)
We introduce `TwoBeamSeedReadoutAdapter`, which equips the $D \to D_{\text{seed}}$ seed collapse interface with **two mutually orthogonal Welch-bounded overcomplete tight frames** $U_1, U_2 \in \mathbb{R}^{M \times D}$ ($M = 2D$, initialized via `optimize_orthogonal_pair_tight_frames` so $\langle \hat{u}_{1,m}, \hat{u}_{2,m}\rangle \approx 0$):
1. **Centered Degree-2 Born Intensity Branch ($\ell = 2$, even)**:
   $$q(\psi) = \frac{D}{\sqrt{D_{\text{seed}}}}\left((\hat{U}_1 \psi)^{\odot 2} - \frac{1}{D}\mathbf{1}_M\right) \in \mathbb{R}^M$$
2. **Phase-Sensitive Degree-3 Cross-Beam Interference Branch ($\ell = 1, 2, 3$, mixed even/odd)**:
   $$b(\psi) = \frac{D^{3/2}}{\sqrt{D_{\text{seed}}}} \left((\hat{U}_1 \psi)^{\odot 2} \odot \mathrm{SiLU}(\hat{U}_2 \psi)\right) \in \mathbb{R}^M$$
3. **Combined Spherical Seed Collapse**:
   $$\phi_{\text{2beam}}(\psi) = \mathrm{Normalize}\!\left(\phi_{\text{lin}}(\psi) + g_{\text{seed}} \odot W_{\text{down}}\big(q(\psi) + \eta_{\text{beam}} \, b(\psi)\big)\right) \in \mathbb{S}^{D_{\text{seed}}-1}$$
   where $\eta_{\text{beam}} \in \mathbb{R}$ is a learnable scalar mixing weight (initialized to `0.6`).

Because $(\hat{U}_1 \psi)^{\odot 2}$ is even in $\psi$ while $\mathrm{SiLU}(\hat{U}_2 \psi)$ is asymmetric/odd in $\psi$, $b(-\psi_\perp) \ne b(\psi_\perp)$ for any nullspace vector $\psi_\perp \in \ker(W_{\text{lin}})$, completely breaking antipodal sign degeneracy on $\ker(W_{\text{lin}})$!

---

## 2. Observable Hypotheses & Quantitative Predictions

### Hypothesis 1 (H1 — Breaking Nullspace Antipodal Sign Degeneracy & Fitting Odd/Degree-3 Nullspace Observables):
Let $D = 128$, $D_{\text{seed}} = 32$, and $M = 2D = 256$. Let $W_{\text{lin}} \in \mathbb{R}^{32 \times 128}$ have orthonormal rows, and let $\psi_\perp \in \ker(W_{\text{lin}}) \cap \mathbb{S}^{127}$ be unit-norm states in the $96$-dimensional nullspace of $W_{\text{lin}}$ ($W_{\text{lin}} \psi_\perp = 0$).
- Define the **Nullspace Antipodal Separation Metric**:
  $$\Delta_{\text{antipodal}}(\phi) = \mathbb{E}_{\psi_\perp \in \ker(W_{\text{lin}}) \cap \mathbb{S}^{D-1}}\!\left[\|\phi(\psi_\perp) - \phi(-\psi_\perp)\|_2\right]$$
- **Observable Prediction (H1)**:
  1. For `OvercompleteSeedReadoutAdapter` (Plan 007 single-beam quadratic adapter), $\Delta_{\text{antipodal}}(\phi_{\text{over}}) = 0.0$ (up to floating-point precision $\le 10^{-6}$) because $q(-\psi_\perp) = q(\psi_\perp)$.
  2. For `TwoBeamSeedReadoutAdapter` (Plan 010), $\Delta_{\text{antipodal}}(\phi_{\text{2beam}}) \ge 0.50$, proving that antipodal states $\pm \psi_\perp \in \ker(W_{\text{lin}})$ are mapped to distinct points on $\mathbb{S}^{D_{\text{seed}}-1}$!
  3. On a synthetic **phase-sensitive nullspace target** $y(\psi) = \mathrm{Normalize}\!\left(W_{\text{lin}}\psi + 0.9 \, P_{\text{null}}\big((V_1 \psi)^{\odot 2} \odot \mathrm{SiLU}(V_2 \psi)\big)\right) \in \mathbb{S}^{D_{\text{seed}}-1}$ (where $V_1, V_2 \in \mathbb{R}^{64 \times 128}$ lie in $\ker(W_{\text{lin}})$) trained for `250` steps of AdamW (`lr = 3e-3`), `TwoBeamSeedReadoutAdapter` will achieve at least **`3.0x` lower held-out test MSE** than `OvercompleteSeedReadoutAdapter` (Plan 007) and at least **`20.0x` lower held-out test MSE** than `LinearSeedCollapse`!

### Hypothesis 2 (H2 — Superior Validation CE & Top-1 Accuracy in `QuantumBornGPT` at Fixed $D_{\text{seed}}$):
In a controlled 3-arm `QuantumBornGPT` language modeling benchmark (`vocab_size = 512`, `embed_dim = 64`, `seed_dim = 16`, `num_heads = 4`, `num_layers = 2`, `block_size = 64`, `readout_mode = "tied_stiefel"`, `200` training steps at `batch_size = 16`):
- **Arm A (`linear_seed_d16`)**: Standard linear seed collapse (`use_overcomplete_seed_adapter = False`, `use_two_beam_seed_adapter = False`).
- **Arm B (`single_beam_seed_adapter_d16`)**: Plan 007 `OvercompleteSeedReadoutAdapter` (`use_overcomplete_seed_adapter = True`).
- **Arm C (`two_beam_seed_adapter_d16`)**: Plan 010 `TwoBeamSeedReadoutAdapter` (`use_two_beam_seed_adapter = True`).
- **Observable Prediction (H2)**:
  At step 200:
  1. **Arm C (`two_beam_seed_adapter_d16`)** will achieve strictly lower validation cross-entropy than **Arm B (`single_beam_seed_adapter_d16`)** by at least **`0.06` nats** ($\text{Val CE}_C \le \text{Val CE}_B - 0.06$) and lower than **Arm A (`linear_seed_d16`)** by at least **`0.40` nats** ($\text{Val CE}_C \le \text{Val CE}_A - 0.40$).
  2. **Arm C** will achieve the highest seed state effective quantum rank $R_{\text{eff}}(\rho_{\text{seed}})$ among all three arms ($R_{\text{eff}, C} \ge R_{\text{eff}, B} > R_{\text{eff}, A}$).

---

## 3. Planned Code & Architectural Changes

### 3.1 PyTorch Module (`src/qdlblocks/layers/two_beam_seed_adapter.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/two_beam_seed_adapter.py` (and export in `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/__init__.py`) containing:

1. `compute_nullspace_antipodal_separation(adapter_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], W_lin: torch.Tensor, num_samples: int = 256, seed: int = 42, eps: float = 1e-12) -> float`:
   - Given a linear projection matrix `W_lin` of shape `(D_seed, D)` (`D > D_seed`), computes an orthonormal basis $Q_\perp \in \mathbb{R}^{(D - D_{\text{seed}}) \times D}$ for $\ker(W_{\text{lin}})$ via SVD of `W_lin`.
   - Samples `num_samples` random unit vectors $\psi_\perp \in \ker(W_{\text{lin}}) \cap \mathbb{S}^{D-1}$ (`coeffs @ Q_perp`, normalized).
   - Evaluates `phi_pos = adapter_fn(psi_perp, torch.zeros(num_samples, D_seed, device=W_lin.device, dtype=W_lin.dtype))` and `phi_neg = adapter_fn(-psi_perp, torch.zeros(num_samples, D_seed, device=W_lin.device, dtype=W_lin.dtype))`.
   - Returns `float(torch.norm(phi_pos - phi_neg, dim=-1).mean().item())`.

2. `TwoBeamSeedReadoutAdapter(nn.Module)`:
   - Constructor signature:
     ```python
     class TwoBeamSeedReadoutAdapter(nn.Module):
         def __init__(
             self,
             embed_dim: int,
             seed_dim: int,
             frame_expansion: int = 2,
             init_gate: float = 0.5,
             init_beam_eta: float = 0.6,
             tight_frame_init_steps: int = 80,
             eps: float = 1e-12,
         ) -> None:
     ```
   - Validates `embed_dim >= 1`, `seed_dim >= 1`, `frame_expansion >= 1`.
   - Sets `self.frame_dim = frame_expansion * embed_dim` ($M = \text{frame\_expansion} \times D$).
   - Initializes `U1_init = optimize_overcomplete_tight_frame(self.frame_dim, embed_dim, steps=tight_frame_init_steps, lr=0.05)` and constructs `U2_init` orthogonalized row-wise against `U1_init` and refined via `tight_frame_init_steps // 2` steps of tight-frame + row-orthogonality optimization.
   - Parameters:
     - `self.U_beam1 = nn.Parameter(U1_init)` of shape `(frame_dim, embed_dim)`
     - `self.U_beam2 = nn.Parameter(U2_init)` of shape `(frame_dim, embed_dim)`
     - `self.W_down = nn.Parameter(torch.empty(seed_dim, frame_dim))`, initialized via `nn.init.orthogonal_(self.W_down)`
     - `self.seed_gate = nn.Parameter(torch.full((seed_dim,), float(init_gate), dtype=torch.float32))`
     - `self.beam_eta = nn.Parameter(torch.tensor(float(init_beam_eta), dtype=torch.float32))`
   - Method `forward(self, psi_final: torch.Tensor, phi_linear: torch.Tensor) -> torch.Tensor`:
     - Row-normalizes `U1 = l2_normalize(self.U_beam1, dim=-1, eps=self.eps).to(psi_final.dtype)` and `U2 = l2_normalize(self.U_beam2, dim=-1, eps=self.eps).to(psi_final.dtype)`.
     - Computes `z1 = F.linear(psi_final, U1)` and `z2 = F.linear(psi_final, U2)` of shape `(..., frame_dim)`.
     - Computes spherical-variance-normalized degree-2 intensity:
       `scale_q = float(self.embed_dim) / math.sqrt(float(self.seed_dim))`
       `q = (z1 * z1 - (1.0 / float(self.embed_dim))) * scale_q`
     - Computes spherical-variance-normalized degree-3 cross-beam interference:
       `scale_b = (float(self.embed_dim) ** 1.5) / math.sqrt(float(self.seed_dim))`
       `b = (z1 * z1) * F.silu(z2) * scale_b`
     - Combines `combined = q + self.beam_eta.to(psi_final.dtype) * b`.
     - Projects `delta_seed = F.linear(combined, self.W_down.to(psi_final.dtype)) * self.seed_gate.to(psi_final.dtype)`.
     - Returns `l2_normalize(phi_linear + delta_seed, dim=-1, eps=self.eps)`.

### 3.2 Integration into `QuantumBornGPT` (`src/qdlblocks/models/born_gpt.py`) and CLI (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)
1. In `QuantumBornGPT.__init__`:
   - Add keyword argument `use_two_beam_seed_adapter: bool = False`.
   - If `use_two_beam_seed_adapter` is `True`, instantiate `self.seed_adapter = TwoBeamSeedReadoutAdapter(embed_dim=embed_dim, seed_dim=seed_dim, frame_expansion=2, eps=eps)` (taking precedence if both flags are passed).
2. In `/home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py`:
   - Add CLI flag `--two-beam-seed-adapter` (`action="store_true"`) and pass `use_two_beam_seed_adapter=two_beam_seed_adapter` to `QuantumBornGPT(...)`.
   - Route `beam_eta` (0D/1D scalar parameter) to `adamw_params` alongside `seed_gate`, and `U_beam1, U_beam2, W_down` to `muon_params`.

### 3.3 Pure JAX Functional Parity (`src/qdlblocks_jax/two_beam_seed_adapter.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/two_beam_seed_adapter.py` (and export in `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/__init__.py`) containing:
- `two_beam_seed_adapter_forward_jax(psi_final: jax.Array, phi_linear: jax.Array, U_beam1: jax.Array, U_beam2: jax.Array, W_down: jax.Array, seed_gate: jax.Array, beam_eta: jax.Array, eps: float = 1e-12) -> jax.Array`

### 3.4 Controlled Verification Script (`examples/20_hypothesis_two_beam_seed_adapter.py`)
Create `/home/festivity/agy/QuantumDLBlocks/examples/20_hypothesis_two_beam_seed_adapter.py` to run H1 and H2 and write `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_010.json`.

---

## 4. Verification & Unit Testing Plan (`tests/test_hypothesis_010_two_beam_seed_adapter.py`)

Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_010_two_beam_seed_adapter.py` with 4 unit tests:
1. `test_two_beam_seed_adapter_unit_norm_and_gradients()`:
   - Verifies unit-norm outputs on $\mathbb{S}^{D_{\text{seed}}-1}$ and finite, non-zero gradients for `psi_final`, `phi_linear`, `U_beam1`, `U_beam2`, `W_down`, `seed_gate`, and `beam_eta`.
2. `test_nullspace_antipodal_symmetry_breaking()`:
   - Verifies that `compute_nullspace_antipodal_separation` is `< 1e-5` for `OvercompleteSeedReadoutAdapter` and `>= 0.50` for `TwoBeamSeedReadoutAdapter`.
3. `test_quantum_born_gpt_with_two_beam_seed_adapter()`:
   - Verifies `QuantumBornGPT(..., use_two_beam_seed_adapter=True, mtp_horizons=2)` forward and backward pass.
4. `test_pytorch_jax_two_beam_seed_adapter_equivalence()`:
   - Verifies PyTorch and JAX forward parity within `max_abs_diff < 1e-5`.
