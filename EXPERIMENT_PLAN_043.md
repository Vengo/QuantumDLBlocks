# Experiment Plan 043 (`Wave 28`): Flag-Free Parseval $\mu\text{P}$ $\sqrt{d}$ Readout, Cholesky Metric Absorption, and Smith (`1506.01186`) Cyclical LR vs. WSD

**Date**: October 24, 2026  
**Wave**: `Wave 28` (`EXPERIMENT_PLAN_043.md`)  
**Hardware Target**: Local `NVIDIA GeForce RTX 5060 Ti` (`15.0 GB` VRAM ceiling via `torch.cuda.set_per_process_memory_fraction(0.95, device=0)`, native `bfloat16` & `float8_e4m3fn` tensor-core GEMMs)  
**Engineering Standard**: Enforces **Rules 1–6** of [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md), specifically **Rule 6 (Clean, Flag-Free Per-Experiment Modules Over Boolean-Flag Accumulation)**: `Wave 28` is implemented in dedicated, straight-line, flag-free modules ([`src/qdlblocks/layers/parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py) and [`src/qdlblocks/models/parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py)), leaving [`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py) and [`direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py) completely untouched and frozen.

---

## 1. Executive Summary & First-Principles Mathematical Derivation

### 1.1 Eliminating `PurityGate`, `kboson_k`, and Manual Clamps via Analytical Parseval $\mu\text{P}$ Scaling
In `Waves 24–27`, replacing the legacy concave $k$-boson logarithm $k \ln(1 + (\beta_t / k)\widehat{O}_{t,v})$ with `Solution A + B` (**Convex-Peak Traceless + Dyadic Coarse-to-Fine Born**) resolved peak-derivative saturation and broke our all-time `FineWeb-Edu` validation Cross-Entropy records. However, the legacy model class still carried three vestigial artifacts from the concave $k$-boson era:

1. **Redundant Division & Multiplication by `kboson_k` (`k_copies`) and Exact Unification of the Diffuse Channel**:
   - In the legacy concave readout, $k \ln(1 + (\beta_t / k)\widehat{O}_{t,v})$ depended nonlinearly on both $k$ and $\tilde{r}_t = \beta_t / k$.
   - Once the concave logarithm was removed in `Waves 24–27`, dividing by $k$ (`r_tilde = beta_t / k`) and immediately multiplying by $k$ (`logits = k * (r_tilde * O_hat + 0.72 * lam_diff * f_diff)`) became an exact algebraic no-op on the sharp channel ($k \cdot (\beta_t / k) = \beta_t$), while $k \approx 34.0$ multiplied by the manual `0.72` damping factor (`34.0 * 0.72 = 24.48`) acted only as a hidden multiplier of magnitude $\approx \beta_0(256) = 23.68$ on $\lambda_{\text{diff}}$.
   - **Exact Flag-Free Forward Equations in `parseval_mup_chunked_ce.py`**:
     Given unit-normalized query states $\psi_t, \psi_{\text{diff}, t} \in \mathbb{S}^{d-1}$ and unit-normalized vocabulary detectors $m_{\text{sharp}, v}, m_{\text{diff}, v} \in \mathbb{S}^{d-1}$ ($v \in \{0, \dots, V_{\text{valid}}-1\}$):
     - **Centered Sharp Cosine & Traceless Observable**:
       $$c_{t,v} = \langle \psi_t \mid m_{\text{sharp}, v} \rangle, \qquad \hat{c}_{t,v} = c_{t,v} - \frac{1}{V_{\text{valid}}}\sum_{u=0}^{V_{\text{valid}}-1} c_{t,u}$$
       $$\widehat{O}_{t,v} = \hat{c}_{t,v} + \gamma \left(\max(\hat{c}_{t,v}, 0)^2 - \tau_{t,+}\right), \qquad \tau_{t,+} = \frac{1}{V_{\text{valid}}}\sum_{u=0}^{V_{\text{valid}}-1} \max(\hat{c}_{t,u}, 0)^2, \qquad \sum_{v=0}^{V_{\text{valid}}-1} \widehat{O}_{t,v} = 0$$
     - **Centered Diffuse Kraus Bath Observable**:
       $$u_{t,v} = \langle \psi_{\text{diff}, t} \mid m_{\text{diff}, v} \rangle, \qquad f_{\text{diff}, t, v} = u_{t,v} + \frac{1}{2}\max(u_{t,v}, 0)^2, \qquad \hat{f}_{\text{diff}, t, v} = f_{\text{diff}, t, v} - \frac{1}{V_{\text{valid}}}\sum_{u=0}^{V_{\text{valid}}-1} f_{\text{diff}, t, u}$$
     - **Coarse Dyadic Tree Cluster Observable ($G = 256$ root clusters $g(v) = \text{perm}[v] \gg 8 \in \{0, \dots, 255\}$)**:
       $$c^{\text{coarse}}_{t, g} = \langle \psi_{\text{coarse}, t} \mid w_{\text{coarse}, g} \rangle, \qquad \hat{c}^{\text{coarse}}_{t, g} = c^{\text{coarse}}_{t, g} - \frac{1}{G}\sum_{h=0}^{G-1} c^{\text{coarse}}_{t, h}, \qquad \widehat{O}^{\text{coarse}}_{t, g} = \hat{c}^{\text{coarse}}_{t, g} + \gamma \max(\hat{c}^{\text{coarse}}_{t, g}, 0)^2$$
     - **Unified Parseval $\mu\text{P}$ Logit Equation (Zero `kboson_k`, Zero Clamps)**:
       Both the sharp observable $\widehat{O}_{t,v}$ and the diffuse observable $\hat{f}_{\text{diff}, t, v}$ are scaled by the **exact same analytical Parseval scale** $\beta(d, \tau) = \beta_0(d) \cdot (0.95 + 0.08\tau)$ where $\beta_0(d) = c^*\sqrt{d}$ ($c^* = 1.48$) and $\tau = s / S_{\text{total}} \in [0, 1]$ is the training progress fraction:
       $$\ell_{t,v} = \beta(d, \tau) \left[\widehat{O}_{t,v} + \lambda_{\text{diff}}(\tau) \, \hat{f}_{\text{diff}, t, v} + \alpha_{\text{coarse}} \, \widehat{O}^{\text{coarse}}_{t, g(v)}\right] + b_v$$
       where $\lambda_{\text{diff}}(\tau) = \lambda_{\text{start}} + (\lambda_{\text{end}} - \lambda_{\text{start}})\tau$ (`0.20 -> 0.05`). Because $\beta_0(256) = 23.68 \approx 34.0 \times 0.72 = 24.48$, multiplying $\lambda_{\text{diff}}(\tau)$ by $\beta(d, \tau)$ preserves the exact physical strength of the diffuse Kraus channel (`~4.74` at $\tau=0 \to$ `~1.22` at $\tau=1$) while scaling both channels identically as $\propto \sqrt{d}$ across all widths!

2. **Why `PurityGate` (`beta_t`) Required Manual Width-Dependent Clamps (`26.8` vs. `44.0`)**:
   - `ContextualBlochPurityGate` is a per-token MLP predicting a dynamic inverse temperature $\beta_t \in (0, 64]$ for each token position.
   - Because inflating $\beta_t$ on easy, high-frequency tokens is the fastest way for early gradients to reduce training loss before representations have sharpened, `PurityGate` races upward early in training, causing validation over-sharpening unless constrained by manual width-specific clamps (`beta_t <= 26.8` at $d=256$ vs. `beta_t <= 44.0` at $d > 256$).

3. **The Unit-Sphere Dual of Attention's $1/\sqrt{d}$ Scaling (`Analytical Parseval Frame Scale` $\beta_0(d) = c^*\sqrt{d}$)**:
   - **Why Euclidean Attention Divides by $\sqrt{d}$**: When $q, k \in \mathbb{R}^d$ have $\mathcal{O}(1)$ per-coordinate variance, $\|q\|_2, \|k\|_2 \sim \sqrt{d}$, so $\operatorname{Var}(\langle q \mid k \rangle) \sim d$ and dot products must be **divided by $\sqrt{d}$** to keep background logits $\mathcal{O}(1)$.
   - **Why Unit-Sphere Quantum Readout Must Multiply by $\sqrt{d}$**: Because both the query wavefunction $\psi_t \in \mathbb{S}^{d-1}$ and vocabulary detectors $m_{\text{sharp}, v} \in \mathbb{S}^{d-1}$ are unit-normalized ($\|\psi_t\|_2 = \|m_{\text{sharp}, v}\|_2 = 1$), the Welch / Parseval frame identity forces off-diagonal cosines ($v \ne y$) to have standard deviation $\sigma_{\text{cos}}(d) = \sqrt{(V - d)/(d(V - 1))} \approx 1/\sqrt{d}$.
   - For the $V - 1$ background tokens ($v \ne y$), $|\hat{c}_{t,v}| \sim 1/\sqrt{d}$ is small, so the linear term dominates ($\widehat{O}_{t,v} = \hat{c}_{t,v} + \mathcal{O}(1/d)$) and multiplying by $\beta_0(d) = c^*\sqrt{d}$ keeps the background logit standard deviation **strictly $\mathcal{O}(1)$ and invariant across all widths $d$**:
     $$\operatorname{Std}_{v \ne y}\!\left(\beta_0(d) \, \widehat{O}_{t,v}\right) = c^* \sqrt{\frac{V - d}{V - 1}} + \mathcal{O}\!\left(\frac{1}{\sqrt{d}}\right) \approx c^* = 1.48$$
   - Conversely, for an aligned target token ($\hat{c}_{t,y} = \mathcal{O}(1) > 0$), $\widehat{O}_{t,y} = \hat{c}_{t,y} + \gamma \hat{c}_{t,y}^2 - \gamma \tau_{t,+} = \mathcal{O}(1)$, so the target peak logit $\beta_0(d)\,\widehat{O}_{t,y} = c^*\sqrt{d}\,\widehat{O}_{t,y} \propto \sqrt{d}$ grows with network capacity $\sqrt{d}$ to overcome the $\ln(V) = 10.82$ nat vocabulary entropy.

### 1.2 How the $\mu\text{P}$ Zero-Shot Transfer & Cholesky Absorption Results Refine `Wave 28`
Following Yang & Hu et al. ([arXiv:2203.03466](https://arxiv.org/abs/2203.03466), *"Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"*), our two diagnostic probes refine the `Wave 28` design in three concrete ways:

1. **Zero-Shot Transfer of $c^* = 1.48$ From Proxy Width $d_{\text{proxy}} = 128$ (`scripts/unit_sphere_mup_transfer_grid_search.py`)**:
   - Sweeping $c \in \{1.10, 1.25, 1.38, 1.48, 1.60, 1.75, 1.95\}$ across $d \in \{128, 256, 384, 512\}$ proved that $c^* = 1.48$ is the **exact minimizer at every width $d$**, achieving **`0.000000` nats transfer regret** (compared to `0.3806` nats regret when transferring a fixed temperature $\beta^* = 16.74$ under Standard Parametrization).
   - **Implication for `Wave 28`**: `c_star = 1.48` is frozen from the $d = 128$ proxy calibration (`beta_0(256) = 23.68`, `beta_0(384) = 29.00`, `beta_0(512) = 33.49`) with **zero hyperparameter tuning at $d \in \{256, 384, 512\}$** and **zero width-conditional clamps**.

2. **Exact Order of Operations for Cholesky Metric Temperature Absorption (`scripts/cholesky_metric_temperature_absorber.py`)**:
   - **Crucial Ordering Relative to $\mathbb{S}^{d-1}$ Normalization**: Any scalar factor applied *before* $\mathbb{S}^{d-1}$ L2-normalization ($z / \|z\|_2$) would cancel out. Therefore, Cholesky metric absorption operates **after** $\psi_t \in \mathbb{S}^{d-1}$ and $m_{\text{sharp}, v} \in \mathbb{S}^{d-1}$ are unit-normalized onto $\mathbb{S}^{d-1}$:
     1. First, $\psi_t \in \mathbb{S}^{d-1}$ and $m_{\text{sharp}, v} \in \mathbb{S}^{d-1}$ are unit-normalized (`F.normalize(..., dim=-1)`).
     2. Next, a post-normalization positive-definite Cholesky metric bridge $A_{\beta} = L_{\beta} L_{\beta}^\top$ initialized at $L_{\beta} = \beta_0(d)^{1/4} I_d$ (so $A_{\beta} = L_{\beta} L_{\beta}^\top = \beta_0(d)^{1/2} I_d$) is applied to both the normalized query state ($\tilde{\psi}_t = A_{\beta} \psi_t = \beta_0(d)^{1/2} \psi_t$) and the normalized detector ($\tilde{m}_v = A_{\beta} m_{\text{sharp}, v} = \beta_0(d)^{1/2} m_{\text{sharp}, v}$).
     3. Their Euclidean inner product then automatically equals the $\beta_0(d)$-scaled cosine:
        $$\tilde{c}_{t,v} = \langle \tilde{\psi}_t \mid \tilde{m}_v \rangle = \langle \beta_0(d)^{1/2} \psi_t \mid \beta_0(d)^{1/2} m_{\text{sharp}, v} \rangle = \beta_0(d) c_{t,v}$$
     4. Evaluating the rectified-quadratic boost with rescaled coefficient $\tilde{\gamma} = \gamma / \beta_0(d)$ yields the exact $\beta_0(d)$-scaled traceless observable with `beta = 1.0` in the loss kernel:
        $$\widehat{\tilde{c}}_{t,v} + \frac{\gamma}{\beta_0(d)}\!\left(\max(\widehat{\tilde{c}}_{t,v}, 0)^2 - \tilde{\tau}_{t,+}\right) = \beta_0(d)\hat{c}_{t,v} + \frac{\gamma}{\beta_0(d)}\!\left(\beta_0(d)^2 \max(\hat{c}_{t,v}, 0)^2 - \beta_0(d)^2 \tau_{t,+}\right) = \beta_0(d)\widehat{O}_{t,v}$$
        reproducing the explicit $\beta_0(d)$ logits to machine precision (`max_logit_abs_err = 5.72e-6`).

3. **Spectral $\mu\text{P}$ Width-Transfer of `StiefelMuon` Learning Rate**:
   - Instead of hand-picking `lr_scale = 0.94` or `1.04` for different widths, `ParsevalMuPDyadicGPT` uses the principled spectral $\mu\text{P}$ width scaling rule $\eta_{\text{muon}}(d) = \eta_{\text{muon}}(256) \cdot \sqrt{256 / d}$ (`0.0260` at $d=256$, `0.0212` at $d=384$, `0.0184` at $d=512$).

### 1.3 Comparing Modded-NanoGPT (`WSD` + `StiefelMuon`) Against Leslie N. Smith (`arXiv:1506.01186`) Cyclical Learning Rates (`CLR`)
In [`arXiv:1506.01186`](https://arxiv.org/abs/1506.01186) (*"Cyclical Learning Rates for Training Neural Networks"*, WACV 2017), Leslie N. Smith introduced two foundational methods that directly address two weaknesses we observed in our Modded-NanoGPT Warmup-Stable-Decay (`WSD`) schedule:

1. **Smith's LR Range Test (`1506.01186` Section 3.3) Across Coupled Optimizer Groups**:
   - Our architecture uses `StiefelMuon` (for orthogonal Stiefel bases and attention projections, base LR $\eta_{\text{muon}, 0} = 0.026$) and `AdamW` (for wavelet tables $\eta_{\text{wav}, 0} = 0.011$, Cholesky/readout parameters $\eta_{\text{head}, 0} = 0.008$, and 1D norms $\eta_{\text{body}, 0} = 0.003$).
   - To preserve the relative update velocities across parameter groups during Smith's LR Range Test and CLR schedules, all parameter groups share a single dimensionless multiplier $m(s)$ relative to their base reference ratio ($1.0 : 0.423 : 0.308 : 0.115$), parameterized by the primary `StiefelMuon` learning rate $\eta(s) = m(s) \, \eta_{\text{muon}, 0}$.
   - In the **LR Range Test**, $\eta(s)$ is swept exponentially from $\eta_{\min} = 10^{-4}$ to $\eta_{\max\_\text{sweep}} = 0.12$ over $S_{\text{range}} = 40$ steps:
     - $\eta_{\text{base}}$ is extracted automatically as the learning rate where the smoothed loss derivative $dL / d\ln\eta$ first drops below `25%` of its steepest negative slope.
     - $\eta_{\max}$ is extracted automatically as the learning rate where the smoothed loss reaches its minimum before turning upward/diverging.

2. **Exact Step-by-Step Schedule Equations Over a `900`-Step (`7.37M`-Token) Budget ($\tau = s / S_{\text{total}} \in [0, 1]$)**:
   - **Modded-NanoGPT `WSD` (`Warmup-Stable-Decay`)**:
     - Warmup ($s \in [0, 45]$, $\tau \in [0, 0.05]$): $\eta(s) = \eta_{\max} \cdot (s / 45)$.
     - Stable Plateau ($s \in [45, 720]$, $\tau \in [0.05, 0.80]$, `75%` of training): $\eta(s) = \eta_{\max}$.
     - Cosine/Linear Decay ($s \in [720, 900]$, $\tau \in [0.80, 1.00]$): decays from $\eta_{\max}$ to $\eta_{\min} = 0.10 \, \eta_{\max}$.
   - **Smith (`1506.01186`) `triangular2` (2 Full Cycles, Half-Cycle Stepsize $S_{\text{half}} = S_{\text{total}} / 4 = 225$ Steps)**:
     - Cycle index $c = \lfloor s / (2 S_{\text{half}}) \rfloor \in \{0, 1\}$, intra-cycle phase $x = |s / S_{\text{half}} - 2c - 1| \in [0, 1]$:
       $$\eta_{\text{tri2}}(s) = \eta_{\text{base}} + \frac{\eta_{\max} - \eta_{\text{base}}}{2^c} \cdot \max(0, 1 - x)$$
     - Notice that $\eta_{\text{tri2}}(s)$ hits its **cooled-down cycle valleys ($\eta = \eta_{\text{base}}$)** at $s = 0$ ($\tau = 0.00$), $s = 450$ ($\tau = 0.50$, midpoint!), and $s = 900$ ($\tau = 1.00$, final step!), and hits its cycle peaks at $s = 225$ ($\tau = 0.25$, full peak $\eta_{\max}$) and $s = 675$ ($\tau = 0.75$, damped peak $(\eta_{\text{base}} + \eta_{\max})/2$).
   - **Smith `1cycle` (Asymmetric Single Triangle + Tail Annealing)**:
     - Upward ramp ($s \in [0, 270]$, $\tau \in [0, 0.30]$): $\eta_{\text{base}} \to \eta_{\max}$.
     - Downward ramp ($s \in [270, 900]$, $\tau \in [0.30, 1.00]$): cosine/linear descent $\eta_{\max} \to \eta_{\min} = 0.08 \, \eta_{\max}$, so at $\tau = 0.33$ ($s = 300$) and $\tau = 0.50$ ($s = 450$) the model is already smoothly cooling down rather than stuck on `WSD`'s $\eta_{\max}$ plateau.

---

## 2. Falsifiable Hypotheses & Acceptance Gates

### Hypothesis `H1`: Analytical Parseval $\sqrt{d}$ Scale Invariance, $\mu\text{P}$ Zero-Shot Transfer, & Cholesky Absorption
- **Claim**:
  1. Across $d \in \{128, 256, 384, 512, 768, 1024\}$ on $V = 50,257$ unit-sphere detectors, $\beta_0(d) = c^*\sqrt{d}$ ($c^* = 1.48$) keeps background logit standard deviation invariant (`bg_logit_std_cv_parseval < 0.015` vs. `> 0.30` for fixed $\beta$), and unifies both beams with `< 1e-6` algebraic error (`scripts/parseval_sqrt_d_scale_invariance_probe.py`).
  2. Calibrating $c^*$ on $d_{\text{proxy}} = 128$ transfers zero-shot to $d \in \{256, 384, 512\}$ with `<= 0.01` nats transfer regret (`scripts/unit_sphere_mup_transfer_grid_search.py`), and post-normalization Cholesky metric absorption ($L_{\beta} = \beta_0(d)^{1/4} I_d \implies A_{\beta} = \beta_0(d)^{1/2} I_d$ with $\tilde{\gamma} = \gamma / \beta_0(d)$) reproduces explicit $\beta_0(d)$ logits to `< 1e-4` (`scripts/cholesky_metric_temperature_absorber.py`).
- **Falsifiable Gate**:
  - Verified in `.cache/parseval_sqrt_d_scale_invariance_probe.json`, `.cache/unit_sphere_mup_transfer_grid_search.json`, and `.cache/cholesky_metric_temperature_absorber.json`.

### Hypothesis `H2`: Clean, Flag-Free `ParsevalMuPDyadicGPT` (`7.37M` Tokens / `900` Steps, `B = 16`, $d = 256$) Beats/Matches Legacy `PurityGate + kboson_k + Clamps`
- **Claim**: Implemented in the standalone, flag-free module [`src/qdlblocks/models/parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py) and loss kernel [`src/qdlblocks/layers/parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py) (with **zero `PurityGate`**, **zero `kboson_raw_k`**, **zero manual clamps**, and **zero boolean mode flags**), `ParsevalMuPDyadicGPT` at $d = 256$ ($\beta_0 = 1.48\sqrt{256} = 23.68$) matches or beats both `Wave 26` (`5.4447` Val CE) and `Wave 25` (`5.4465` Val CE) on the identical `7.37M`-token (`900`-step, `B = 16, T = 512`) `FineWeb-Edu` stream.
- **Falsifiable Gate**:
  1. `w28_flagfree_parseval_mup_b16_d256_900steps` achieves `final_val_ce <= 5.4447` with zero `PurityGate`, zero `kboson_raw_k`, and zero `torch.clamp` calls in the model or loss module.

### Hypothesis `H3`: Zero-Shot $\mu\text{P}$ Multi-Width Scaling (`d = 256, 384, 512`) With Zero Manual Clamps
- **Claim**: Using the exact same $c^* = 1.48$ transferred zero-shot from $d = 128$ ($\beta_0(256) = 23.68$, $\beta_0(384) = 29.00$, $\beta_0(512) = 33.49$), proportional wavelet detail ranks `(d // 2, d // 4)`, and spectral $\mu\text{P}$ learning-rate scaling $\eta(d) = \eta(256)\sqrt{256/d}$, `ParsevalMuPDyadicGPT` achieves strict monotonic Cross-Entropy improvement across widths:
  $$\text{CE}(d=512) < \text{CE}(d=384) < \text{CE}(d=256)$$
  at matched `FineWeb-Edu` token horizons (`7.37M` tokens, `B = 16, T = 512`) with zero width-conditional `if embed_dim > 256` branches.
- **Falsifiable Gate**:
  1. `ce_d512 < ce_d384 < ce_d256 <= 5.4447` at `7.37M` tokens.

### Hypothesis `H4`: Leslie N. Smith (`arXiv:1506.01186`) LR Range Test + Cyclical Learning Rate (`triangular2` / `1cycle`) vs. Modded-NanoGPT `WSD`
- **Claim**:
  1. Running Smith's (`1506.01186`) **LR Range Test** (`scripts/smith_clr_range_test_vs_wsd_profiler.py`) on `ParsevalMuPDyadicGPT` over an exponential sweep $\eta \in [10^{-4}, 0.12]$ automatically identifies the steepest-descent base bound $\eta_{\text{base}} \in [0.004, 0.015]$ and stability ceiling $\eta_{\max} \in [0.020, 0.045]$ in `40` steps without manual grid search.
  2. Training `ParsevalMuPDyadicGPT` with Smith's (`1506.01186`) **`triangular2` / `1cycle` boundary schedule** bounded by $[\eta_{\text{base}}, \eta_{\max}]$ eliminates the mid-training `WSD` plateau (achieving strictly lower intermediate Val CE at the $\tau = 0.50$ midpoint checkpoint) and matches or beats `WSD` at the final token horizon.
- **Falsifiable Gate**:
  1. Verified in `.cache/smith_clr_range_test_vs_wsd_profiler.json` and on `FineWeb-Edu` (`w28_smith_clr_b16_d256_900steps` vs. `w28_flagfree_parseval_mup_b16_d256_900steps`).

---

## 3. Planned Code & Architectural Changes (Rule 6 Flag-Free Architecture)

1. **Dedicated Flag-Free Loss Kernel ([`src/qdlblocks/layers/parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py))**:
   - A clean, `< 240`-line module implementing `parseval_mup_dyadic_chunked_ce`:
     $$\ell_{t,v} = \beta(d, \tau) \left[\widehat{O}_{t,v} + \lambda_{\text{diff}}(\tau) \hat{f}_{\text{diff}, t, v} + \alpha_{\text{coarse}} \widehat{O}^{\text{coarse}}_{t, g(v)}\right] + b_v$$
     with chunked log-sum-exp backward fusion, native `FP8` (`float8_e4m3fn`) / `BF16` tensor-core GEMMs, **zero `k_copies` argument**, **zero `is_convex` flag**, and **zero `torch.clamp` calls**.
2. **Dedicated Flag-Free Wave 28 Model ([`src/qdlblocks/models/parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py))**:
   - A clean, `< 340`-line module defining `ParsevalMuPDyadicConfig` and `ParsevalMuPDyadicGPT`.
   - Imports shared primitives (`Tied2Pow16WaveletCholeskyPyramid`, `RadicalWaveletCholeskyBlock`, `WaveletCholeskyUNetBlock`, `StiefelMuon`, `build_dyadic_2pow16_spectral_tree`) while containing **zero `PurityGate`**, **zero `kboson_raw_k`**, **zero manual clamps**, and **zero `use_*` boolean flags**.
   - Supports post-normalization Cholesky metric absorption (`absorb_beta_into_readout_cholesky()`) and spectral $\mu\text{P}$ optimizer construction (`build_parseval_mup_optimizers()`).
3. **Smith (`1506.01186`) LR Range Test & Cyclical Schedule Profiler ([`scripts/smith_clr_range_test_vs_wsd_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/smith_clr_range_test_vs_wsd_profiler.py))**:
   - Implements Smith's (`arXiv:1506.01186`) single-run exponential/linear LR Range Test, `triangular`, `triangular2`, `exp_range`, and `1cycle` schedules, comparing them head-to-head against Modded-NanoGPT `WSD` on `ParsevalMuPDyadicGPT`.
4. **Wave 28 Verification Runner ([`examples/53_verify_plan043_parseval_sqrt_d_noclamp_readout.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/53_verify_plan043_parseval_sqrt_d_noclamp_readout.py))**:
   - Straight-line script that imports `ParsevalMuPDyadicGPT`, verifies `H1`–`H4`, and writes [`.cache/hypothesis_experiments_043.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_043.json).

---

## 4. Verification & Testing Plan

- **Unit & Rule-6 Flag-Free AST Tests ([`tests/test_plan043_parseval_sqrt_d_noclamp_readout.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan043_parseval_sqrt_d_noclamp_readout.py))**:
  - Verifies via Python `ast` inspection that `src/qdlblocks/models/parseval_mup_dyadic_gpt.py` and `src/qdlblocks/layers/parseval_mup_chunked_ce.py` contain **zero `PurityGate` references**, **zero `kboson_raw_k` attributes**, **zero `clamp` calls on `beta`**, and **zero `use_*` / `is_*` boolean config fields**, while passing `scripts/ast_signature_and_length_linter.py`.
  - Verifies exact $\beta_0(d) = c^*\sqrt{d}$ scaling across $d \in \{128, 256, 384, 512\}$, Cholesky metric absorption equivalence (`max_abs_err < 1e-4`), and Smith (`1506.01186`) LR Range Test + `triangular2` / `1cycle` schedule bounds.
- **Hypothesis Gate Tests ([`tests/test_hypothesis_043_parseval_sqrt_d_noclamp_readout.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_043_parseval_sqrt_d_noclamp_readout.py))**:
  - Asserts `overall_pass is True` and all numerical gates for `H1`, `H2`, `H3`, and `H4` in `.cache/hypothesis_experiments_043.json`.

---

## 5. Operational Tooling & Script Opportunities

### 5.1 Existing & Newly Promoted Scripts Used
- [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py): Enforces Rules 1–6 across all new `Wave 28` files.
- [`scripts/parseval_sqrt_d_scale_invariance_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_sqrt_d_scale_invariance_probe.py): Verifies $\beta_0(d) = c^*\sqrt{d}$ background noise invariance across $d \in \{128, \dots, 1024\}$.
- [`scripts/unit_sphere_mup_transfer_grid_search.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/unit_sphere_mup_transfer_grid_search.py): Verifies zero-shot transfer of $c^* = 1.48$ across $d \in \{128, 256, 384, 512\}$.
- [`scripts/cholesky_metric_temperature_absorber.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/cholesky_metric_temperature_absorber.py): Verifies exact absorption of $\beta_0(d)^{1/2}$ into `BlockCholeskyLinear`.

### 5.2 `Build-Before-Launch` Scripts
- [`scripts/smith_clr_range_test_vs_wsd_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/smith_clr_range_test_vs_wsd_profiler.py):
  ```bash
  CUDA_VISIBLE_DEVICES="" /home/festivity/agy/QuantumDLBlocks/.venv/bin/python scripts/smith_clr_range_test_vs_wsd_profiler.py --json-out .cache/smith_clr_range_test_vs_wsd_profiler.json
  ```

### 5.3 `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/stiefel_manifold_curvature_lr_bound_probe.py`: Relates Smith's (`1506.01186`) empirical $\eta_{\max}$ from the LR Range Test to the analytical Stiefel retraction stability threshold $\eta_{\text{crit}} = 2 / L_{\text{Riemannian}}$.
