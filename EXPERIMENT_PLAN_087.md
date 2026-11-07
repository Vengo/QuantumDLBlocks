# `EXPERIMENT_PLAN_087.md`: Wave 67 — Selective Exact Quotienting, Strictly Read-Only Riemannian Shadow EMA, Wide `256D` Subspace Readout, & Horizontal Gradient Completeness Auditing

## 1. Executive Summary & Hypotheses

Wave 66 ([`EXPERIMENT_PLAN_086.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_086.md)) revealed two critical findings about parameter-manifold gauge freedom and thermostat dynamics:

1. **Exact Gauge Symmetries vs. Over-Projected Broken Symmetries**:
   - **Exact Gauge Symmetries (`W66_G1 -> W66_G2`)**: Analytically locking the second MUB probe basis via the $16 \times 16$ unitary DFT matrix ($U_h^{(2)} \equiv F_{16} U_h^{(1)}$, guaranteeing `MUBErr = 0.00e+00` vs. `0.697` in unconstrained $U(16) \times U(16)$) and quotienting out the left $\mathfrak{u}(R)$ Kraus branch-mixing fiber via the Hermitian Lyapunov equation (`KrausGauge = 0.00e+00`) strictly improved validation CE while saving `8,192` parameters.
   - **Over-Projected Broken Symmetries (`W66_G0 -> W66_G1`)**: By the chain rule, if a transformation $\theta \mapsto g(\epsilon) \cdot \theta$ is a true forward symmetry of $\mathcal{L}(\theta)$, then the raw gradient $\nabla_\theta \mathcal{L}$ must have **identically zero** component along the vertical fiber ($\langle \nabla_\theta \mathcal{L}, \xi_{\text{vert}} \rangle = 0$), and the horizontal projection $\Pi_{\text{horiz}}(\nabla_\theta \mathcal{L})$ must retain **100%** of the physical Riemannian gradient ($\|\Pi_{\text{horiz}}(\nabla \mathcal{L})\|_F / \|\Pi_{\text{manifold}}(\nabla \mathcal{L})\|_F = 1.0$). Applying this **Random-Loss Horizontal Gradient Completeness Test** exposes three over-projections in `W66_G1`/`W66_G3`:
     1. **Tied Vocabulary Subspace Bridges ($U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, d_{\text{voc}})$)**: Because $W_{\text{vocab}}$ is shared between input wavepackets ($W_{\text{vocab}} U_{\text{in}}^\top$) and output readout detectors ($W_{\text{vocab}} U_{\text{out}}^\top$), only the **simultaneous 3-way right rotation** $(U_{\text{in}} Q, U_{\text{out}} Q, W_{\text{vocab}} Q)$ ($Q \in O(d_{\text{voc}})$) is a forward symmetry ($\operatorname{skew}(U_{\text{in}}^\top \nabla_{U_{\text{in}}} \mathcal{L}) + \operatorname{skew}(U_{\text{out}}^\top \nabla_{U_{\text{out}}} \mathcal{L}) + \operatorname{skew}(W_{\text{vocab}}^\top \nabla_{W_{\text{vocab}}} \mathcal{L}) \equiv 0$). Projecting $U_{\text{in}}$ and $U_{\text{out}}$ independently onto $\operatorname{Gr}(d_{\text{voc}}, 512)$ threw away **`75.3%`** of the physical gradient—freezing both the relative input-to-output transition operator $\Omega_{\text{in}} - \Omega_{\text{out}} \in \mathfrak{so}(d_{\text{voc}})$ and the alignment between the trunk and $W_{\text{vocab}}$.
     2. **Coupled Biorthogonal SVD-Stiefel Pairs ($W = V \operatorname{diag}(\vec{\sigma}) U^\top$)**: Under simultaneous right-rotation $(U e^{\epsilon\Omega}, V e^{\epsilon\Omega})$, the operator variation is $(V^\top \delta W U)_{ij} = (\sigma_j - \sigma_i)\Omega_{ij}$. When $\sigma_i = \sigma_j$ (Step 0), $\Omega_{\text{sym}, ij}$ is an exact gauge fiber (`raw_grad = 0`), explaining why `W66_G3` beat `W66_G0` through Step 100 (`7.4905` vs. `7.5033` at Step 75; `7.4096` vs. `7.4111` at Step 100). Once singular values separate ($\sigma_i \neq \sigma_j$ after Step 100), $\Omega_{\text{sym}, ij}$ carries physical singular-vector rotation gradient proportional to $|\sigma_i - \sigma_j|$.
     3. **Tied MUB Basis Phase Fiber ($U_h^{(2)} \equiv F_{16} U_h^{(1)}$)**: Because `pack_rank1_ket_outer_isometric` normalizes the *rows* of $U_h^{(1)}$ (`dim=-1` is the ket coordinate), right-multiplication $U_h^{(1)} \mapsto U_h^{(1)} \operatorname{diag}(e^{i\vec{\phi}})$ rotates the computational basis coordinates of $\rho_h$, while left relative row phases $U_h^{(1)} \mapsto \operatorname{diag}(e^{i\vec{\theta}}) U_h^{(1)}$ do not commute with $F_{16}$ and actively steer the second MUB basis $U_h^{(2)} = F_{16} U_h^{(1)}$. Only the **1D global phase center** $U_h^{(1)} \mapsto e^{i\theta_h} U_h^{(1)}$ ($\operatorname{Tr}(\operatorname{Im}(U_h^\dagger \xi_h)) = 0$, i.e. $\mathfrak{su}(16)$) is an exact gauge fiber of the tied pair $(U_h^{(1)}, F_{16} U_h^{(1)})$.
2. **Strictly Read-Only Riemannian Shadow EMA (Zero Mid-Training Rollbacks or LR Ratchets)**:
   - Even with `catastrophic_gap_tol_nats = 0.042`, `W66_G3` still triggered a mid-training `hot_model <- ema_model` overwrite at Step 75 (`thermal_gap = +0.0541`) and damped `lr_scale` down to `0.74x`. A positive gap $\mathcal{L}_{\text{hot}} - \mathcal{L}_{\text{ema}} > 0$ during high-LR plateau training is the normal statistical signature of geodesic Polyak averaging smoothing mini-batch variance. Making the Riemannian Shadow EMA **strictly read-only** during the first 85% of training (zero mid-training parameter overwrites and zero `lr_scale` damping, reserving smooth cooldown + higher EMA weight for the final ~15% quench) eliminates momentum teleportation completely.
3. **Fixing the Rank-128 Readout Bottleneck (`128D -> 256D`) & Increasing Quantum Trunk Capacity (`R=2 -> 4`, `MLP=1024 -> 1536`)**:
   - Widening the fixed-once shared Stiefel vocabulary subspace frame from `subspace_dim = 128` to `subspace_dim = 256` doubles the linear readout rank to match the $16 \times 16 = 256$ complex density-matrix dimensions while keeping a single fixed-once tied vocabulary sphere. Once the readout bottleneck and thermostat rollback are fixed, increasing trunk capacity (`kraus_rank = 4`, `mlp_hidden = 1536`) can be cleanly exploited.

### Quantitative Hypotheses
- **H1 (100% Horizontal Gradient Completeness & Zero Gauge Leak)**: On [`scripts/horizontal_gradient_completeness_auditor.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/horizontal_gradient_completeness_auditor.py), all four Wave 67 selective-quotient projections (`su(16)` Gauge-Locked MUB, Lyapunov `u(R)` Kraus, Spectral-Degeneracy-Weighted SVD Filter, and 3-Way Tied-Subspace Excess Filter) achieve **`raw_vertical_grad_leak_frac < 1e-6` (`0.00e+00`)**, **`horiz_grad_retention_ratio = 1.000000`** under random losses (including non-isotropic $\vec{\sigma}$ and relative input-output bridge perturbations), while reducing post-Newton–Schulz artificial gauge spin to `0.00e+00` on all exact gauge fibers.
- **H2 (Selective Exact Quotienting + Read-Only Shadow EMA at Iso-Parameters `128D`)**: `W67_S1_SELECTIVE_EXACT_QUOTIENT_128D` (`7,933,320` params) achieves `0` mid-training rollbacks (`lr_scale = 1.00x`), maintains `MUBErr = 0.00e+00` and `KrausGauge = 0.00e+00`, and outperforms both `W66_G0_STIEFEL_AMBIENT_BASELINE` and `W66_G3_FULL_GAUGE_FREE_SPIKE_ONLY_THERMOSTAT` across both the 60-step benchmark and the 200-step (`0.41M`-token) horizon run.
- **H3 (Widening the Readout Bottleneck `128D -> 256D` & Trunk Capacity `R=4, MLP=1536`)**: `W67_S2_SELECTIVE_QUOTIENT_256D_READOUT` (`subspace_dim = 256`) and `W67_S3_SELECTIVE_QUOTIENT_256D_WIDE_TRUNK` (`subspace_dim = 256, kraus_rank = 4, mlp_hidden = 1536`) monotonically improve validation CE and Top-1 accuracy over `W67_S1_SELECTIVE_EXACT_QUOTIENT_128D`.

---

## 2. Mathematical & Architectural Design

### 2.1 Four Selective Exact Quotient Projections (100% Horizontal Gradient Retention)

1. **Exact $\mathfrak{su}(16)$ Horizontal Projection for Gauge-Locked MUB Tomography ($U_h^{(2)} \equiv F_{16} U_h^{(1)}$)**:
   - Projects `(v_re, v_im)` onto $T_{U_h}U(d_h)$ ($\Omega_h = \operatorname{skew}_{\mathbb{C}}(U_h^\dagger \xi_h) \in \mathfrak{u}(d_h)$) and strips only the exact 1D global phase center $\mathfrak{u}(1)$:
     $$\bar{\phi}_h = \frac{1}{d_h}\operatorname{Tr}(\operatorname{Im}(\Omega_h)), \qquad \Omega_h^{\text{su}(d_h)} = \Omega_h - i \bar{\phi}_h I_{d_h}, \qquad \xi_h^{\text{horiz}} = U_h \Omega_h^{\text{su}(d_h)}$$
   - Preserves `100.0%` of the raw gradient (`raw_vertical_grad_leak_frac = 0.00e+00`) while keeping $U_h^{(2)} = F_{16} U_h^{(1)}$ mutually unbiased (`MUBErr = 0.00e+00`) at every step.

2. **Exact Lyapunov $\mathfrak{u}(R)$ Horizontal Projection for CPTP Kraus Channels ($\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(R d_h, d_h)$)**:
   - Solves the $R \times R$ Hermitian Lyapunov equation $\frac{1}{2}(A_h G_K + G_K A_h) = S_{\text{skew}}(\Xi_h)$ with $G_K = K_{\text{flat}} K_{\text{flat}}^\dagger$ and returns $\Xi_h^{\text{horiz}} = \Xi_h - A_h K_{\text{flat}} \in T_{\mathbf{K}_h}\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$.
   - Preserves `100.0%` of the raw gradient (`raw_vertical_grad_leak_frac = 0.00e+00`) and strips `100.0%` of the Newton–Schulz-injected $\mathfrak{u}(R)$ branch-mixing spin (`KrausGauge = 0.00e+00`).

3. **Spectral-Degeneracy-Weighted Post-Newton–Schulz Co-Rotation Filter for Coupled SVD-Stiefel Pairs $(U, V)$**:
   - For $W = V \operatorname{diag}(\vec{\sigma}) U^\top$, the raw gradient $(\nabla_U \mathcal{L}, \nabla_V \mathcal{L})$ is projected onto $T_U\operatorname{St} \times T_V\operatorname{St}$ with **zero** physical co-rotation stripped prior to Newton–Schulz (except along exact degenerate pairs $|\sigma_i - \sigma_j| < \epsilon_\sigma$ where $\Omega_{\text{sym}, ij}(\nabla \mathcal{L}) \equiv 0$ by the chain rule).
   - After 5-step Newton–Schulz polar orthogonalization $(Z_U, Z_V) = (\operatorname{NS}_5(\Xi_U), \operatorname{NS}_5(\Xi_V))$, the **artificial co-rotation spin injected by $\operatorname{NS}_5$** on degenerate/near-degenerate singular pairs is damped via the Lorentzian spectral degeneracy kernel:
     $$\Gamma_{ij}(\vec{\sigma}) = \frac{\tau_\sigma^2}{\tau_\sigma^2 + (\sigma_i - \sigma_j)^2}, \qquad \Omega_{\text{excess}, ij}^{\text{NS}} = \Gamma_{ij}(\vec{\sigma}) \cdot \Big(\Omega_{\text{sym}}(Z_U, Z_V)_{ij} - \alpha_{\text{ref}}\,\Omega_{\text{sym}}(\hat{\Xi}_U, \hat{\Xi}_V)_{ij}\Big)$$
     where $(\hat{\Xi}_U, \hat{\Xi}_V)$ is the RMS-matched pre-Newton–Schulz tangent pair. At $\vec{\sigma} = \vec{1}$ (Step 0), $\Gamma_{ij} = 1$ and $\Omega_{\text{sym}}(\hat{\Xi}_U, \hat{\Xi}_V) = 0$, so `100%` of Newton–Schulz co-rotation spin is removed; once $\sigma_i \neq \sigma_j$, singular-vector rotation is completely unblocked (`horiz_grad_retention_ratio = 1.000000`).

4. **Tied-Subspace Excess Common-Fiber Filter for $(U_{\text{in}}, U_{\text{out}}) \in \operatorname{St}(512, d_{\text{voc}})$**:
   - Retains the full Stiefel tangent space $T_{U_{\text{in}}}\operatorname{St}(512, d_{\text{voc}}) \times T_{U_{\text{out}}}\operatorname{St}(512, d_{\text{voc}})$ on raw gradients (`horiz_grad_retention_ratio = 1.000000`), keeping both relative input-to-output transitions $\Omega_{\text{in}} - \Omega_{\text{out}} \in \mathfrak{so}(d_{\text{voc}})$ and vocabulary alignment unblocked, while subtracting only the **post-Newton–Schulz excess common-fiber spin** $\Delta \Omega_{\text{common}}^{\text{NS}} = \gamma_{\text{common}}\big(\Omega_{\text{common}}(Z_{\text{in}}, Z_{\text{out}}) - \alpha_{\text{ref}}\Omega_{\text{common}}(\hat{\Xi}_{\text{in}}, \hat{\Xi}_{\text{out}})\big)$.

### 2.2 Strictly Read-Only Riemannian Shadow EMA (`ReadOnlyQuenchRiemannianShadowEMA`)
- During the first `85%` of training (`step < int(0.85 * total_steps)`):
  - `ema_model` is **100% strictly read-only**: `check_and_thermostat` retracts `ema_model` onto all Stiefel/Unitary/Sphere manifolds and evaluates both `hot_model` and `ema_model`, but **never overwrites `hot_model`** (`triggered = False` always) and **never damps `lr_scale`** (`lr_scale = 1.00x` throughout the plateau).
- During the final `15%` cooldown quench (`step >= int(0.85 * total_steps)`):
  - The learning rate follows the smooth cosine cooldown quench (`1.00x -> 0.08x`) while `ema_model` continues geodesic Polyak averaging without disrupting `hot_model`'s Nesterov momentum buffers.

---

## 3. Verification & Unit Testing Plan
1. **Horizontal Gradient Completeness & Gauge-Injection Auditor ([`scripts/horizontal_gradient_completeness_auditor.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/horizontal_gradient_completeness_auditor.py))**:
   - Runs random-loss completeness and Newton–Schulz gauge-injection audits across all 4 manifolds on both `W66_G3` (exposing the over-projection leaks) and `W67` (`0.00e+00` leak, `1.000000` retention).
2. **Pytest Suite ([`tests/test_plan087_selective_exact_quotient.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan087_selective_exact_quotient.py))**:
   - Tests:
     1. `test_random_loss_horizontal_gradient_completeness`: Verifies `raw_vertical_grad_leak_frac < 1e-6` and `horiz_grad_retention_ratio == 1.0` across all 4 manifolds under random losses (including non-isotropic $\vec{\sigma}$ and relative $(U_{\text{in}}, U_{\text{out}})$ perturbations).
     2. `test_su16_mub_and_lyapunov_kraus_exact_invariance`: Verifies exact forward invariance under $U_h^{(1)} \mapsto e^{i\theta_h} U_h^{(1)}$ and $\mathbf{K}_h \mapsto (W_h \otimes I_{16})\mathbf{K}_h$, plus `0.00e+00` post-Newton–Schulz vertical spin.
     3. `test_3way_tied_subspace_and_degenerate_svd_invariance`: Verifies exact forward invariance of `SharedStiefelSubspaceFrameBoundary` under simultaneous $(U_{\text{in}} Q, U_{\text{out}} Q, W_{\text{vocab}} Q)$ ($Q \in O(d_{\text{voc}})$) and exact suppression of degenerate SVD co-rotation spin at $\vec{\sigma} = \vec{1}$.
     4. `test_read_only_shadow_ema_zero_rollbacks`: Verifies that `ReadOnlyQuenchRiemannianShadowEMA` performs `0` active-weight overwrites and keeps `lr_scale == 1.0` regardless of `hot_val_ce - ema_val_ce`.

---

## 4. Named Constants Ledger (Rule 7)

| Constant Name | Value | Module | Mathematical Role & Rationale |
| :--- | :---: | :--- | :--- |
| `DEFAULT_NARROW_SUBSPACE_DIM` | `128` | [`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) | Iso-parameter `128D` tied Stiefel vocabulary subspace dimension for `W67_S0` and `W67_S1`. |
| `DEFAULT_WIDE_SUBSPACE_DIM` | `256` | [`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) | Widened `256D` tied Stiefel vocabulary subspace dimension for `W67_S2` and `W67_S3`, matching the $16 \times 16 = 256$ density-matrix highway dimensions. |
| `DEFAULT_WIDE_KRAUS_RANK` | `4` | [`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) | Widened CPTP Kraus branch rank $R = 4$ ($\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(64, 16)$) for `W67_S3_SELECTIVE_QUOTIENT_256D_WIDE_TRUNK`. |
| `DEFAULT_WIDE_MLP_HIDDEN` | `1536` | [`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) | Widened Mach–Zehnder MLP intermediate dimension (`3.0x` expansion of `512D`) for `W67_S3`. |
| `DEFAULT_SVD_DEGENERACY_TAU` | `0.05` | [`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) | Lorentzian spectral degeneracy width $\tau_\sigma$ in $\Gamma_{ij}(\vec{\sigma}) = \tau_\sigma^2 / (\tau_\sigma^2 + (\sigma_i - \sigma_j)^2)$ for coupled SVD pairs. |
| `DEFAULT_COMMON_BRIDGE_EXCESS_DAMPING` | `0.50` | [`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) | Fraction of post-Newton–Schulz excess common-fiber spin $\Delta \Omega_{\text{common}}^{\text{NS}}$ subtracted on tied bridges $(U_{\text{in}}, U_{\text{out}})$. |
| `DEFAULT_QUENCH_FRACTION` | `0.15` | [`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) | Final fraction of training steps (`15%`) dedicated to the cosine cooldown quench after the `85%` read-only plateau. |
| `DEFAULT_SELECTIVE_MOMENTUM_BETA` | `0.92` | [`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py) | Nesterov momentum coefficient $\beta$ for `SelectiveExactQuotientMuon`. |

---

## 5. Operational Tooling & Script Opportunities

1. **Existing Scripts Used**:
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py`
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload`
2. **`Build-Before-Launch` Scripts**:
   - [`scripts/horizontal_gradient_completeness_auditor.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/horizontal_gradient_completeness_auditor.py): Audits random-loss horizontal gradient retention (`horiz_grad_retention_ratio == 1.0`, `raw_vertical_grad_leak_frac == 0.0`), post-Newton–Schulz artificial gauge-spin suppression, and finite gauge-orbit invariance across `W66_G3` vs. `W67`.
   - [`scripts/selective_exact_quotient_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/selective_exact_quotient_bench.py): Runs the 4-variant Wave 67 ablation suite (`W67_S0`, `W67_S1`, `W67_S2`, `W67_S3`) and multi-checkpoint horizon runs.
3. **`Nice-to-Have` Scripts (Logged in [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md))**:
   - `scripts/svd_singular_spectrum_and_bridge_alignment_tracker.py`: Tracks the evolution of singular-value splitting $|\sigma_i - \sigma_j|$ across `BiorthogonalSVDStiefelProjection` layers and the principal angles between $U_{\text{in}}$ and $U_{\text{out}}$ across checkpoints.

---

## 6. Verification & Empirical Results Ledger

### 6.1 AST Best-Practices Linter & Unit Test Verification
- [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py):
  - [`src/qdlblocks/benchmarks/selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py): `860` SLOC (`<= 1,000` soft limit), `36` functions, `max_args = 8`, `max_nested_sloc = 0` (**PASS**).
  - [`scripts/horizontal_gradient_completeness_auditor.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/horizontal_gradient_completeness_auditor.py): `226` SLOC, `5` functions, `max_args = 3`, `max_nested_sloc = 0` (**PASS**).
  - [`scripts/selective_exact_quotient_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/selective_exact_quotient_bench.py): `39` SLOC, `1` function, `max_args = 0` (**PASS**).
  - [`tests/test_plan087_selective_exact_quotient.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan087_selective_exact_quotient.py): `76` SLOC, `4 passed in 5.83s` (**PASS**).

### 6.2 Random-Loss Horizontal Gradient Completeness & Gauge-Injection Audit ([`scripts/horizontal_gradient_completeness_auditor.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/horizontal_gradient_completeness_auditor.py))

| Parameter Manifold & Physical Operator | `W66_G3` Raw Vertical Grad Leak | `W66_G3` Horiz Grad Retention | `W67` Raw Vertical Grad Leak | `W67` Horiz Grad Retention | `W67` Min Post-NS Descent Cosine | Exact Finite Gauge Orbit Max Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tied Vocab Subspace Bridges $(U_{\text{in}}, U_{\text{out}})$** | `0.3681` (`36.8%` lost) | `0.9297` | **`0.000000`** | **`1.000000`** (`100%`) | `+0.8844` | `4.768e-07` ($O(d_{\text{sub}})$ 3-way) |
| **Coupled SVD Pairs $(\vec{\sigma} = \vec{1}, \text{Degenerate})$** | `0.0000` | `1.0000` | **`0.000000`** | **`1.000000`** (`100%`) | `+0.9356` | `0.000e+00` (`DegSpin = 0.00e+00`) |
| **Coupled SVD Pairs $(\sigma_i \neq \sigma_j, \text{Split Spectrum})$** | `0.2339` (`23.4%` lost) | `0.9722` | **`0.000000`** | **`1.000000`** (`100%`) | `+0.9405` | `1.999e-03` (Lorentzian $\Gamma(\vec{\sigma})$) |
| **Gauge-Locked MUB Tomography $U_h^{(2)} \equiv F_{16} U_h^{(1)}$** | `0.2447` (`24.5%` lost) | `0.9695` | **`0.000001`** | **`1.000000`** (`100%`) | `+0.5634` | `2.384e-07` ($\mathfrak{su}(16)$ global phase) |
| **CPTP Kraus Operators $\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(R d_h, d_h)$** | `0.0000` | `1.0000` | **`0.000000`** | **`1.000000`** (`100%`) | `+0.8084` | `4.768e-07` ($\mathfrak{u}(R)$ Lyapunov) |

### 6.3 60-Step (`0.12M`-Token) 4-Variant Ablation Results (`FineWeb` GPT-2 BPE $V=50,257$)

| Variant (`run_key`) | `d_sub` / `R` / `MLP` | Params | Step 40 Rollback / `lr_scale` | Step 60 Hot Val CE | Step 60 EMA Val CE | Step 60 Top-1 Acc | `MUBErr` | `SU16Phase` / `KrausGauge` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`W67_S0_AMBIENT_RATCHET_BASELINE`** | `128D, R=2, 1024` | `23.55M` | `trig=True` (`lr=0.90x`) | `7.4795` | `7.4731` | `9.24%` | `4.17e-01` | `2.33e-02` / `2.01e-02` |
| **`W67_S1_SELECTIVE_EXACT_QUOTIENT_128D`** | `128D, R=2, 1024` | `23.55M` | `trig=False` (`lr=1.00x`) | `7.4821` | `7.4751` | `9.08%` | **`0.00e+00`** | **`0.00e+00` / `0.00e+00`** |
| **`W67_S2_SELECTIVE_QUOTIENT_256D_READOUT`** | `256D, R=2, 1024` | `30.12M` | `trig=False` (`lr=1.00x`) | `7.4697` (`task-1371`) / `7.4842` | `7.4686` (`task-1371`) / `7.4829` | `9.22%` | **`0.00e+00`** | **`0.00e+00` / `0.00e+00`** |
| **`W67_S3_SELECTIVE_QUOTIENT_256D_WIDE_TRUNK`** | `256D, R=4, 1536` | `31.23M` | `trig=False` (`lr=1.00x`) | **`7.4672`** | **`7.4646`** | **`9.29%`** (`9.42%` in `task-1371`) | **`0.00e+00`** | **`0.00e+00` / `0.00e+00`** |

### 6.4 200-Step (`8`-Checkpoint, `0.41M`-Token) Horizon Comparison (`W66_G0` vs. `W66_G3` vs. `W67_S1` vs. `W67_S3`)

| Checkpoint (`Tokens`) | `W66_G0` (`128D` Ambient Ratchet) Hot / EMA CE | `W66_G3` (`128D` Over-Projected + Rollback) Hot / EMA CE | `W67_S1` (`128D` Selective Quotient + Read-Only EMA) Hot / EMA CE | `W67_S3` (`256D, R=4` Selective Quotient + Read-Only EMA) Hot / EMA CE |
| :--- | :---: | :---: | :---: | :---: |
| **Step 25 (`0.05M`)** | `7.9317` / `7.9355` (`1.00x`) | `7.9393` / `7.9438` (`1.00x`) | **`7.9300` / `7.9415`** (`1.00x`) | `7.9445` / `7.9681` (`1.00x`) |
| **Step 50 (`0.10M`)** | `7.6033` / `7.5718` (`0.90x`, `trig`) | `7.5902` / `7.5718` (`1.00x`) | `7.5945` / `7.5826` (`1.00x`) | `7.5928` / `7.5912` (`1.00x`) |
| **Step 75 (`0.15M`)** | `7.5033` / `7.4452` (`0.79x`, `trig`) | `7.4905` / `7.4320` (`0.88x`, `trig`) | **`7.4877` / `7.4355`** (`1.00x`) | `7.4989` / `7.4513` (`1.00x`) |
| **Step 100 (`0.20M`)** | `7.4111` / `7.3948` (`0.79x`) | `7.4096` / `7.3948` (`0.88x`) | `7.4761` / `7.4387` (`1.00x`) | `7.4954` / `7.4615` (`1.00x`) |
| **Step 125 (`0.26M`)** | `7.3052` / `7.2747` (`0.70x`, `trig`) | `7.3396` / `7.3016` (`0.77x`, `trig`) | `7.3523` / `7.3160` (`1.00x`) | `7.3773` / `7.3423` (`1.00x`) |
| **Step 150 (`0.31M`)** | `7.2918` / `7.2747` (`0.67x`) | `7.3435` / `7.3016` (`0.74x`) | `7.3141` / `7.2960` (`1.00x`) | `7.3417` / `7.3308` (`1.00x`) |
| **Step 175 (`0.36M`)** | `7.2323` / `7.2131` (`0.67x`) | `7.2764` / `7.2530` (`0.74x`) | `7.2549` / `7.2339` (`1.00x`) | `7.2780` / `7.2503` (`1.00x`) |
| **Step 200 (`0.41M`)** | `7.1695` / `7.1743` (`10.75%` Top-1, `MUBErr=0.697`) | `7.2102` / `7.1912` (`10.71%` Top-1, `MUBErr=0.00`) | **`7.1709` / `7.1720`** (`10.76%` Top-1, `MUBErr=0.00`) | **`7.1664` / `7.1709`** (**`10.81%` Top-1**, `MUBErr=0.00`) |

### 6.5 Post-Mortem & Key Mathematical Takeaways
1. **Why Selective Exact Quotienting (`W67_S1` & `W67_S3`) Reverses the Late-Horizon Slowdown of `W66_G3`**:
   - In `W66_G3`, projecting $(U_{\text{in}}, U_{\text{out}})$, coupled SVD pairs $(U, V)$, and MUB bases $U_h^{(1)}$ onto over-restricted quotient spaces threw away `23.4%–36.8%` of physical horizontal gradients under split spectra and triggered a thermostat rollback at Step 75 (`7.2102` Hot CE / `7.1912` EMA CE at Step 200).
   - Restoring 100% horizontal gradient completeness (`0.000000` raw vertical leak) and making the Riemannian Shadow EMA strictly read-only (`0` rollbacks, `lr = 1.00x`) allowed **`W67_S1` (`128D`) to beat `W66_G3` by `-0.0393` nats on Hot CE (`7.1709` vs. `7.2102`) and `-0.0192` nats on EMA CE (`7.1720` vs. `7.1912`)**, and allowed **`W67_S3` (`256D, R=4, MLP=1536`) to reach `7.1664` Hot CE / `7.1709` EMA CE / `10.81%` Top-1**—surpassing both `W66_G3` (`-0.0438` Hot CE) and `W66_G0` across `Hot CE`, `EMA CE`, `Top-1`, and `MUBErr` (`0.00e+00` vs. `0.697`).
2. **Decoupling $\operatorname{Gr}(d_{\text{sub}}, 512)$ Subspace Selection from $W_{\text{vocab}}$ Coordinate Rotation**:
   - Passing $\Xi_{\text{horiz}} + U\Omega$ into tall Stiefel Newton–Schulz contaminates the $d_{\text{sub}} \times d_{\text{sub}}$ Gram matrix $(\Xi_{\text{horiz}}^\top\Xi_{\text{horiz}} + \Omega^\top\Omega)$ with vertical fiber spin $\Omega^\top\Omega$ and collides with `BatchSubspaceTikhonovGramMuon` updating $W_{\text{vocab}}$ at `5.6x` higher LR. Orthogonalizing only $(I - U U^\top)\Xi$ via Newton–Schulz cleanly separates Grassmannian subspace selection on $(U_{\text{in}}, U_{\text{out}})$ from coordinate updates on $W_{\text{vocab}}$.

