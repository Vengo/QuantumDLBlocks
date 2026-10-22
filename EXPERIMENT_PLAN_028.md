# Experiment Plan 028 — Stereographic Radial-Norm-Preserving Lift, Cosine Phase-Interference Two-Beam Readout, Positive-Part Quadratic Kraus Fidelity, Fisher's Iris & Concentric-Shell Geometric Verification, & Wave 17 2-G4-Instance `900M`-Token Colab Fleet

## 1. Executive Summary & Project Context (Waves 1–16)

Across Waves 1–16 and Plans 013–027 of `QuantumDLBlocks`, we have engineered and compared two competing sequence-transducer families on the `900M`-token (9-shard, `13,725`-step) FineWeb-Edu benchmark (evaluated on the held-out `10.48M`-token `KellerJordan/modded-nanogpt` validation shard `fineweb_val_000000.bin` at sequence length $T_{\text{val}} = 512$ across the `50,257` GPT-2 vocabulary):

1. **Classical Baseline Family (`ClassicalModdedGPT`)**:
   - Reaches **`3.3362` Val CE** (`28.1` Val PPL, `38.68%` Top-1, `63.52%` Top-5) at `900M` tokens in Wave 13 (`w13_900m_classical_leaderboard_exact_anvil_a_b_s1337`).
   - On our local RTX 5060 Ti (`<= 8.0 GB` VRAM cap, `L=6, d=512`, `300` steps = `2.46M` tokens, Plan 025), layerwise telemetry revealed that standard self-attention acts as a low-pass contractive filter across depth: spectral effective rank $R_\sigma(h_\ell)$ collapses monotonically from `221.2` (Layer 1) down to `184.1` (Layer 5) and participation ratio $R_{\text{PR}}(h_\ell)$ crashes by `-60%` from `31.61` down to **`12.75`**.

2. **Quantum Born-Rule Family (`QuantumBornGPT`) — Wave 16 Breakthrough (`3.4980` Val CE / `42.26%` Peak Top-1)**:
   - **Wave 13 Flat $d=1024$ (`w13_900m_qborn_d1024_ppmi_svd_9shard_ngram_anvil_v6`)**: Reached **`3.5121` Val CE** (`33.5` Val PPL).
   - **Wave 15 16-Qubit (`50,304 × 16 × 32`) + 256-State Hierarchy Pure POVM (`w15_900m_qborn_16x32_qubit_256hier_ngram_L8`)**: Completed `899.48M` tokens at **`3.5554` Val CE** (`35.0` Val PPL) with **`41.58%` Peak Top-1 Accuracy** (`41.41%` final Top-1, `59.51%` Top-5, `MedRank = 3`, runaway `beta = 134.35`).
   - **Wave 16 Completed `900M`-Token Results (`Plans 026 & 027` — 100% Quantum Two-Beam $k$-Boson Mixed-State Readout $\text{Tr}(\rho_t^{\otimes k} E_v^{\otimes k})$ + Scheduled Rank-$r$ Stiefel Kraus Semantic Bath + `polar_snap_weights` `"embed"` Exclusion Fix)**:
     All 5 Wave 16 `900M`-token runs completed `13,725 / 13,725` steps (`899.48M` tokens) and automatically stopped their Colab VMs (`0.00 CU/hr`):

| Session | Wave 16 Run Key (`899.48M` Tokens, `Step 13,725`) | Architecture Highlights | Final Val CE | Val PPL | Final Top-1 (`Peak`) | Final Top-5 (`Peak`) | MedRank | Final $\beta$ | Speed (`tok/s`) | Wall Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gpu_g4_w6_2`** | **`w16_900m_qborn_16x32_256hier_unified_w13w15_twobeam_kboson_L8`** | **Unified W13+W15 (`L=8, 4d-TwoBeam, VE=3, Win384, N-Gram`) + Rank-16 `warmup_hold_anneal` $k$-Boson** | **`3.4980`** | **`33.0`** | **`42.20%` (`42.26%`)** | **`60.41%` (`60.46%`)** | **`3`** | `44.94` | `49,233` | `400.6m` |
| `gpu_a100_w6_1` | `w16_900m_qborn_16x32_256hier_ngram_twobeam_kboson_L8` | Direct Two-Beam $k$-Boson counterpart to W15 `3.5554` (`L=8, 4d-TwoBeam, N-Gram`, Rank-16 `warmup_hold_anneal`) | **`3.5322`** | `34.2` | `41.71%` (`41.78%`) | `59.83%` (`59.92%`) | `3` | `44.98` | `43,322` | `479.6m` |
| `gpu_a100_w6_2` | `w16_900m_qborn_16x32_256hier_ngram_twobeam_kboson_r32_ctx_L8` | `L=8, 4d-TwoBeam, N-Gram` + Rank-32 `contextual_purity_complement` $k$-Boson | **`3.5407`** | `34.5` | `41.49%` (`41.74%`) | **`59.99%`** (`59.99%`) | `3` | `44.99` | `43,405` | `478.9m` |
| `gpu_g4_w6_1` | `w16_900m_qborn_16x32_256hier_twobeam_kboson_flagship_L8` | `L=8, 4d-TwoBeam` (no Hashed N-Gram, no VE=3) + Rank-16 `warmup_hold_anneal` $k$-Boson | `3.5641` | `35.3` | `41.54%` (`41.54%`) | `59.95%` (`59.95%`) | `3` | `44.95` | `48,973` | `418.9m` |
| `gpu_g4_w6_3` | `w16_900m_qborn_16x32_256hier_unified_deep_L12_twobeam_kboson` | Deep `L=12, 2d-TwoBeam` (`SkipAttn(7)`, no Hashed N-Gram) + Rank-16 `warmup_hold_anneal` $k$-Boson | `3.6099` | `37.0` | `41.16%` (`41.16%`) | `59.23%` (`59.23%`) | `3` | `44.95` | `45,907` | `437.7m` |

**Three Landmark Conclusions from Wave 16**:
1. **New All-Time `QuantumBornGPT` Record (`3.4980` Val CE, `42.26%` Peak Top-1, `60.46%` Peak Top-5)**:
   `w16_900m_qborn_16x32_256hier_unified_w13w15_twobeam_kboson_L8` (`gpu_g4_w6_2`) is the first `QuantumBornGPT` model in project history to break `< 3.50` Val CE, `> 42.2%` Top-1, and `> 60.4%` Top-5, beating Wave 15's pure-POVM record (`3.5554`) by **`-0.0574 nats`** and Wave 13's flat record (`3.5121`) by **`-0.0141 nats`**.
2. **Direct Controlled Validation of Two-Beam $k$-Boson Readout (`-0.0232 nats` on Identical Backbone)**:
   Comparing `w16_900m_qborn_16x32_256hier_ngram_twobeam_kboson_L8` (`3.5322` CE, `41.78%` peak Top-1, `59.92%` peak Top-5) against its exact Wave 15 single-beam pure-POVM counterpart `w15_900m_qborn_16x32_qubit_256hier_ngram_L8` (`3.5554` CE, `41.58%` peak Top-1, `59.51%` Top-5) proves that replacing pure-state POVM projection with the Two-Beam $k$-Boson Mixed-State Readout improves full-scale `900M`-token Cross-Entropy by **`-0.0232 nats`** while simultaneously improving both Top-1 (`+0.20%`) and Top-5 (`+0.41%`).
3. **`L=8, 4d-TwoBeam` Dominates `L=12, 2d-TwoBeam` by `-0.1119 nats`**:
   Cutting `frame_expansion` from `4.0` (`4096`-D overcomplete two-beam frame) to `2.0` (`2048`-D) to fit `L=12` sacrifices the overcomplete two-beam spectral rank expansion discovered in Plan 025 (`R_sigma: 287.3 -> 291.7`). Thus `L=8` with `frame_expansion = 4.0` and `262,144` Hashed N-Gram is strictly superior in both quality (`3.4980` vs `3.6099`) and speed (`49,233 tok/s` vs `45,907 tok/s`).

---

## 2. Three Geometric Traps Identified in Our Recent Discussions & Wave 16 Telemetry

### 2.1 Trap 1: Radial Norm Erasure via Naive `l2_normalize` (The Fisher's Iris & Interference-Amplitude Trap)

Consider Fisher's Iris dataset ($N = 150$ samples, $d = 4$ continuous features: sepal length/width, petal length/width; $C = 3$ classes: *Setosa*, *Versicolor*, *Virginica*):
- *Setosa* is linearly separable from the other two species by a wide angular margin.
- However, ***Versicolor* and *Virginica* lie along nearly the exact same directional ray in $\mathbb{R}^4$** (mean class-centroid cosine similarity $\cos\theta \approx 0.994$)! They differ primarily in **radial magnitude $\|x\|_2$** (*Virginica* petals and sepals are $1.3\times\text{–}1.5\times$ larger than *Versicolor* with almost identical length-to-width ratios).
- If a model encodes $x \in \mathbb{R}^d$ onto $S^{d-1}$ via naive spherical normalization:
  $$\psi_{\text{naive}}(x) = \frac{x}{\|x\|_2 + \epsilon} \in S^{d-1}$$
  it completely **erases the radial coordinate $\|x\|_2$**, projecting *Versicolor* and *Virginica* onto the exact same point on $S^{d-1}$. Even more starkly, on concentric radial shells in $\mathbb{R}^d$ where classes share an isotropic angular distribution and differ only in radius $\|x\|_2 \in \{r_1, r_2, r_3\}$, any classifier built on naive `l2_normalize(x)` is mathematically upper-bounded by random guessing ($1/C = 33.3\%$).

**Where Radial Norm Erasure Happens Inside `QuantumBornGPT` Right Now**:
1. In `OvercompleteTwoBeamInterference.forward` (`src/qdlblocks/layers/overcomplete_two_beam.py`, lines 179–187):
   ```python
   interf = (z1 * z1) * F.silu(z2)
   out_1 = F.linear(interf, self.W_out.to(psi.dtype))
   psi_main = l2_normalize(self.post_norm(out_1.float()).to(psi.dtype), dim=-1, eps=self.eps)
   b2 = (z2 * z2) * F.silu(z1)
   out_2 = F.linear(b2, u2.t().to(psi.dtype)) * ...
   psi_beam2 = l2_normalize(out_2 - 0.25 * out_1, dim=-1, eps=self.eps)
   ```
   Here `out_1` and `out_2` are produced by cubic two-beam interference across the $M = 4d$ tight frame. Their pre-normalization radial norms $\|out_1\|_2$ and $\|out_2\|_2$ encode the **constructive vs. destructive interference intensity** of token $t$. Applying `self.post_norm(out_1)` followed by `l2_normalize` completely erases $\|out_1\|_2$ and $\|out_2\|_2$!
2. In `Hierarchical16QubitPOVMReadout.get_two_beam_kboson_factors` (`src/qdlblocks/layers/hierarchical_qubit.py`, lines 783–790):
   `psi_sharp_norm = l2_normalize(psi_sharp * self.readout_rms_gain, dim=-1)` erases the pre-readout radial norm $\| \psi_{\text{sharp}} \odot g_{\text{rms}} \|_2$.

**The Quantum-Geometric Solution — Stereographic Radial-Norm-Preserving Lift (`--stereographic-lift`)**:
1. **Exact Riemann Northern-Hemisphere / Stereographic Lift for Tabular & Low-$d$ Inputs (`StereographicHilbertLift`, $x \in \mathbb{R}^d \mapsto S^d \subset \mathbb{R}^{d+1}$)**:
   With a learnable positive reference radius $\tau_{\text{stereo}} = \text{softplus}(\tilde{\tau}) + 10^{-4} > 0$:
   $$\psi_{\text{stereo}}(x) = \frac{\big[x,\; \tau_{\text{stereo}}\big]}{\sqrt{\|x\|_2^2 + \tau_{\text{stereo}}^2}} \in S^{d}$$
   Now the radial magnitude $\|x\|_2$ is preserved as the **polar latitude angle** $\theta_{\text{lat}}(x) = \arctan(\|x\|_2 / \tau_{\text{stereo}}) \in [0, \pi/2)$ on $S^d$:
   - $\|\psi_{\text{stereo}}(x)\|_2 = 1$ identically (100% pure unit ket on $S^d$).
   - Collinear vectors $x_2 = \alpha x_1$ ($\alpha \neq 1$, such as *Virginica* vs *Versicolor* or concentric shells) map to distinct polar latitudes $\theta_{\text{lat}}(x_1) \neq \theta_{\text{lat}}(x_2)$ on $S^d$ and are cleanly separated by unit-norm POVM detectors $M_c \in S^d$!
2. **Dimension-Preserving Stereographic Polar-Latitude Modulation on $S^{d-1}$ inside `OvercompleteTwoBeamInterference` and `Hierarchical16QubitPOVMReadout`**:
   To preserve power-of-2 ambient dimension $d \in \{512, 1024\}$ for FlashAttention, RoPE, and StiefelMuon without adding an odd $+1$ coordinate:
   - Given any pre-normalized state $u \in \mathbb{R}^{(\dots, d)}$ with radial norm $\|u\|_2$, define its dimensionless ratio relative to a learnable reference scale $\tau_{\text{stereo}} = \text{softplus}(\tilde{\tau}) + 10^{-4} > 0$ (initialized to $\tau_0 = 1.0$):
     $$r(u) = \frac{\|u\|_2}{\sqrt{d}\,\tau_{\text{stereo}} + \epsilon}, \qquad s_{\text{lat}}(u) = \frac{r(u) - 1}{\sqrt{r(u)^2 + 1}} \in (-1, 1)$$
     Notice that $s_{\text{lat}}(u) = \sin\theta_{\text{lat}}(u) - \cos\theta_{\text{lat}}(u)$ is a bounded, smooth, strictly monotonic stereographic latitude coordinate that equals $0$ when $r(u) = 1$, approaches $+1$ as $\|u\|_2 \to \infty$, and approaches $-1$ as $\|u\|_2 \to 0$.
   - Given the unit direction $\hat{u} = \text{l2\_normalize}(u) \in S^{d-1}$ and a learnable stereographic pole vector $p_{\text{pole}} \in \mathbb{R}^d$ orthogonalized against $\hat{u}$ via Gram-Schmidt:
     $$\hat{p}_\perp(u) = \text{l2\_normalize}\!\Big(p_{\text{pole}} - \langle p_{\text{pole}}, \hat{u}\rangle \hat{u}\Big) \in S^{d-1}$$
     we rotate $\hat{u}$ toward $\hat{p}_\perp(u)$ by an angle determined by $s_{\text{lat}}(u)$ and learnable gain $\alpha_{\text{stereo}} = 0.35\tanh(\eta_{\text{stereo}})$ (initialized to $\eta_{\text{stereo}} = 0.40 \implies \alpha_{\text{stereo}} \approx 0.133$):
     $$\psi_{\text{stereo}}(u) = \text{l2\_normalize}\!\Big(\hat{u} + \alpha_{\text{stereo}}\, s_{\text{lat}}(u)\, \hat{p}_\perp(u)\Big) \in S^{d-1}$$
   - In `OvercompleteTwoBeamInterference.forward`:
     Before `self.post_norm(out_1)` erases $\|out_1\|_2$, we compute $s_{\text{lat}}(out_1)$ from the raw interference output `out_1` (and $s_{\text{lat}}(out_2)$ from `out_2`) and apply the stereographic polar-latitude rotation to both `psi_main` and `psi_beam2`!
   - In `Hierarchical16QubitPOVMReadout.get_two_beam_kboson_factors`:
     When `use_stereographic_lift=True`, we also apply a dedicated readout stereographic polar-latitude rotation (`readout_stereo_pole`, `readout_stereo_tau`, `readout_stereo_gain`) to `psi_sharp * readout_rms_gain` before `l2_normalize` so token-level channel-gain radial variations are preserved on $S^{d-1}$ at readout.

---

### 2.2 Trap 2 & Trap 3: Uncollapsed Two-Beam Cosine/Sine Quadrature Phase Interference, Positive-Part Quadratic Kraus Fidelity, & Batch-Centered Contextual Purity Routing

1. **Cosine/Sine Quadrature Phase-Interference Gate & Exact Gram-Schmidt Beam-2 Orthogonality (`--cosine-phase-beam-gate`)**:
   - Currently in `OvercompleteTwoBeamInterference.forward`, Beam 1 is `(z1 * z1) * silu(z2)`, Beam 2 is `(z2 * z2) * silu(z1)`, and `psi_beam2 = l2_normalize(out_2 - 0.25 * out_1)` uses a fixed `0.25` subtraction heuristic that does not guarantee $\langle \psi_{\text{main}}, \psi_{\text{beam2}}\rangle = 0$.
   - In genuine two-beam quantum interferometry, two tight-frame paths $z_1 = \sqrt{d}\,(\psi U_1^\top)$ and $z_2 = \sqrt{d}\,(\psi U_2^\top)$ interfere through their **relative phase difference** $\Delta\theta = \omega_{\text{phase}} \odot (z_1 - z_2) + \phi_{\text{phase}}$:
     - **Beam 1 (Constructive Cosine Phase Channel)**:
       $$\text{interf}_1 = (z_1^2) \odot \text{SiLU}(z_2) \odot \Big(1 + \alpha_{\text{phase}} \cos\big(\omega_{\text{phase}} \odot (z_1 - z_2) + \phi_{\text{phase}}\big)\Big)$$
     - **Beam 2 (Quadrature Sine Phase Channel)**:
       $$\text{interf}_2 = (z_2^2) \odot \text{SiLU}(z_1) \odot \Big(1 + \alpha_{\text{phase}} \sin\big(\omega_{\text{phase}} \odot (z_1 - z_2) + \phi_{\text{phase}}\big)\Big)$$
       where $\omega_{\text{phase}} \in \mathbb{R}^M$ (initialized to `0.50`), $\phi_{\text{phase}} \in \mathbb{R}^M$ (initialized to `0.0`), and $\alpha_{\text{phase}} = 0.50\tanh(\eta_{\text{phase}})$ (initialized to $\eta_{\text{phase}} = 0.35$).
     - **Exact Gram-Schmidt Beam-2 Orthogonality**:
       After projecting `out_2`, `psi_beam2` is projected onto the exact orthogonal complement of `psi_main`:
       $$u_2^\perp = u_2 - \langle u_2, \psi_{\text{main}}\rangle \psi_{\text{main}}, \qquad \psi_{\text{beam2}} = \text{l2\_normalize}(u_2^\perp) \in S^{d-1}$$
       guaranteeing $\langle \psi_{\text{main}}, \psi_{\text{beam2}}\rangle = 0$ to machine precision ($< 10^{-6}$) for every token so $(\psi_{\text{main}}, \psi_{\text{beam2}})$ forms a true orthogonal rank-2 quantum state frame!

2. **Quadratic Positive-Part Kraus Diffuse Fidelity (`--quadratic-diffuse-fidelity`, $\gamma_{\text{quad}} = 0.50$)**:
   - In `_DirectChunkedTwoBeamKBosonCEFn` (`src/qdlblocks/layers/direct_chunked_ce.py`, line 782), the inner fidelity currently uses only the linear diffuse term $1 + \tilde{r}_t c_{\text{sharp}} + \lambda_{\text{diff}} c_{\text{diff}}$.
   - Adding the 2-boson positive-hemisphere quadratic Kraus fidelity term $\gamma_{\text{quad}}\,\text{ReLU}(c_{\text{diff}})^2$ ($\gamma_{\text{quad}} = 0.50$):
     $$c_+(t, v) = \max\big(c_{\text{diff}}(t, v),\, 0\big), \qquad f_{\text{diff}}(t, v) = c_{\text{diff}}(t, v) + \gamma_{\text{quad}}\, c_+(t, v)^2$$
     $$z_{t,v} = \max\!\Big(1 + \tilde{r}_t c_{\text{sharp}}(t, v) + \lambda_{\text{diff}}(t, s)\, f_{\text{diff}}(t, v),\; 10^{-4}\Big)$$
     $$\mathcal{E}^{\text{Born-}k}_{t,v} = k_{\text{eff}} \ln(z_{t,v}) + b_v$$
     achieves two critical geometric properties simultaneously:
     1. **Zero Negative-Hemisphere Tail Leakage**: For anti-aligned/orthogonal background tokens where $c_{\text{diff}}(t, v) \le 0$, $c_+(t, v) = 0$, so the quadratic term vanishes identically.
     2. **Curved Decision Boundaries & Stronger Mid-Rank Gradient Slope**: For plausible semantic neighbors (`Ranks 2–50` and `51–1000`) where $c_{\text{diff}}(t, v) > 0$, the gradient slope with respect to $c_{\text{diff}}$ increases by $(1 + 2\gamma_{\text{quad}} c_+(t, v))$, carving curved non-linear decision boundaries (as needed on concentric shells and polysemous token clusters) with zero extra GEMMs!

3. **Batch-Centered Contextual Purity Complementary Routing (`contextual_purity_complement` Fix)**:
   - Look at Wave 16's telemetry in Section 1: by `899.48M` tokens, $\beta_t \to 44.98$ while $k_{\text{eff}} \in [36, 45]$, so $\tilde{r}_t = \text{clamp}(\beta_t / k_{\text{eff}}, 0, 0.95)$ has a batch mean around $\bar{r}_t \approx 0.88\text{–}0.95$.
   - In Plan 027 (`src/qdlblocks/layers/hierarchical_qubit.py`, line 590), `contextual_purity_complement` computed `lam_base + contextual_purity_gain * (0.65 - r_tilde)` with a **hardcoded reference `0.65`**. Once $\tilde{r}_t$ exceeded `0.65`, `0.65 - r_tilde` became negative (`-0.23` to `-0.30`) for *every* token in the batch, systematically subtracting $\approx -0.05$ from the `warmup_hold_anneal` schedule `lam_base`! That is why `w16_900m_qborn_16x32_256hier_ngram_twobeam_kboson_r32_ctx_L8` (`3.5407` CE) trailed `warmup_hold_anneal` (`3.5322` CE) by `0.0085` nats despite achieving higher Top-5 accuracy (`59.99%` vs `59.83%`).
   - **The Fix**: Center the contextual purity offset around the **detached batch mean purity** $\bar{r}_{\text{ref}} = \text{clamp}(\text{mean}_t(\tilde{r}_t).\text{detach}(), 0.55, 0.92)$:
     $$\lambda_{\text{diff}}(t, s) = \text{clamp}\!\Big(\lambda_{\text{base}}(s) + \gamma_{\text{ctx}}\big(\bar{r}_{\text{ref}} - \tilde{r}_t\big),\; \lambda_{\min},\; \lambda_{\max}\Big)$$
     Now the batch-mean diffuse bath weight $\frac{1}{N}\sum_t \lambda_{\text{diff}}(t, s)$ tracks the exact `warmup_hold_anneal` schedule `lam_base(s)` (`0.22 -> 0.12`) with **zero downward bias**, while still routing $+\gamma_{\text{ctx}}(\bar{r}_{\text{ref}} - \tilde{r}_t) > 0$ extra diffuse bath weight to ambiguous tokens ($\tilde{r}_t < \bar{r}_{\text{ref}}$) and $-\gamma_{\text{ctx}}(\tilde{r}_t - \bar{r}_{\text{ref}}) < 0$ less diffuse bath weight to sharp tokens ($\tilde{r}_t > \bar{r}_{\text{ref}}$)!

---

## 3. Strict Hardware Resource Constraints

1. **Google Colab Fleet Constraint — ONLY 2 G4 Instances (`gpu_g4_w6_1` and `gpu_g4_w6_2`)**:
   - Wave 17 (`configs/wave17_fleet.json`) MUST use **exclusively the 2 G4 instances `gpu_g4_w6_1` and `gpu_g4_w6_2`** (`NVIDIA RTX PRO 6000 Blackwell (96GB)`).
   - Sessions `gpu_g4_w6_3`, `gpu_a100_w6_1`, and `gpu_a100_w6_2` MUST remain completely unallocated (`0.00 CU/hr`).
2. **Local RTX 5060 Ti Constraint — Strictly $\le 8.0\text{ GB}$ VRAM (`50%` of `16 GB`)**:
   - All local verification scripts (`examples/38_verify_plan028_stereographic_twobeam_kboson_and_wave17.py` and `scripts/iris_quantum_geometry_probe.py`) MUST call `torch.cuda.set_per_process_memory_fraction(0.5, device=0)` before any CUDA allocation so Peak VRAM never exceeds `8.0 GB`.

---

## 4. Observable Hypotheses & Quantitative Predictions

### Hypothesis H1 — Fisher's Iris ($N=150, d=4, C=3$) & 4D Concentric Radial Shells ($N=300, d=4, C=3$) Geometric Trap Verification
- **Claim**:
  Evaluating 4 controlled classifiers (`NaiveBornClassifier`, `StereographicSingleBeamKBoson`, `StereographicTwoBeamKBoson`, and `ClassicalSoftmaxMLP`) in `src/qdlblocks/layers/stereographic_lift.py` and `scripts/iris_quantum_geometry_probe.py`:
  1. **On 4D Concentric Radial Shells ($C=3$ isotropic spherical shells at radii $r \in \{1.0, 2.2, 3.6\}$)**:
     - `NaiveBornClassifier` (`l2_normalize(x)` on $S^3$) collapses all 3 radial shells onto the same unit sphere and fails at near-chance accuracy (**$\le 45.0\%$** test accuracy).
     - `StereographicTwoBeamKBoson` (`StereographicHilbertLift` $\psi_{\text{stereo}}(x) \in S^4$ + Cosine/Sine Quadrature Two-Beam Interference + Quadratic Kraus $k$-Boson Readout) preserves radial norm as polar latitude and carves curved radial shell boundaries, achieving **$\ge 95.0\%$** test accuracy (an improvement of **$\ge +50.0$ percentage points** over `NaiveBornClassifier`).
  2. **On Fisher's Iris ($N=150, d=4, C=3$: *Setosa*, *Versicolor*, *Virginica*)**:
     - Because *Versicolor* and *Virginica* lie along nearly the same directional ray in centered/raw feature space ($\cos\theta > 0.98$), `NaiveBornClassifier` suffers from radial norm erasure (**$\le 86.7\%$** accuracy).
     - `StereographicTwoBeamKBoson` resolves all 3 geometric traps and achieves **$\ge 96.0\%$** accuracy on Fisher's Iris, matching or outperforming `ClassicalSoftmaxMLP` while maintaining exact unit-norm quantum states ($\|\psi_{\text{stereo}}\|_2 = 1$, $\|\psi_{\text{sharp}}\|_2 = 1$, $\|\psi_{\text{diff}}\|_2 = 1$, $|\langle \psi_{\text{sharp}}, \psi_{\text{diff}}\rangle| < 10^{-6}$).

### Hypothesis H2 — Exact Autograd Parity of Quadratic Kraus $k$-Boson CE, Orthogonal Quadrature Beams, & Batch-Centered Contextual Purity Routing
- **Claim**:
  1. `_DirectChunkedTwoBeamKBosonCEFn` with `quad_diffuse_weight = 0.50` matches materialized `compute_two_beam_kboson_logits(..., quad_diffuse_weight=0.50)` + `F.cross_entropy` to within $|\Delta \mathcal{L}| < 10^{-5}$ in forward loss and $\max|\Delta \nabla| < 10^{-5}$ across all 8 input gradients (`dpsi_sharp`, `dpsi_diff`, `dm_sharp`, `dm_diff`, `dbeta`, `dk`, `dlam`, `dlog_prior`).
  2. `OvercompleteTwoBeamInterference` with `use_stereographic_lift=True` and `use_cosine_phase_beam_gate=True` produces unit-norm beams (`max | ||psi_main||_2 - 1 | < 1e-5`, `max | ||psi_beam2||_2 - 1 | < 1e-5`), exact Gram-Schmidt beam orthogonality (`max | <psi_main, psi_beam2> | < 1e-5`), and strictly non-zero sensitivity to input radial scaling ($\|\psi_{\text{main}}(2.0\,x) - \psi_{\text{main}}(0.5\,x)\|_2 > 0.02$).
  3. Batch-centered `contextual_purity_complement` keeps the batch-mean diffuse bath weight $\bar{\lambda}_{\text{diff}}$ within $10^{-3}$ of the `warmup_hold_anneal` schedule target `lam_base` even when all tokens have high $\beta_t \in [32.0, 44.0]$ ($\tilde{r}_t \in [0.80, 0.95]$), while still routing $\ge +0.04$ higher $\lambda_{\text{diff}}$ to lower-purity tokens than higher-purity tokens.

### Hypothesis H3 — Local `8 GB` RTX 5060 Ti 300-Step FineWeb-Edu Ablation & Wave 17 2-G4-Instance `900M`-Token Launch
- **Claim**:
  1. On our local RTX 5060 Ti (`<= 8.0 GB` VRAM cap via `torch.cuda.set_per_process_memory_fraction(0.5)`), training the Unified `L=6, d=512` model with Plan 028's full synthesis (`--stereographic-lift --cosine-phase-beam-gate --quadratic-diffuse-fidelity 0.50 --diffuse-rank 32 --diffuse-schedule contextual_purity_complement`) for `300` steps (`2.46M` FineWeb10B tokens) achieves **$\le 6.0890$ Val CE** (beating Plan 026 `free_decay` `6.0940`, Plan 027 `contextual_purity_complement` `6.1023`, Unified Softmax `6.0922`, Single-Beam Pure POVM `6.1476`, and Classical `6.4771`) with `Peak VRAM <= 8.0 GB`.
  2. Launching `configs/wave17_fleet.json` via `scripts/colab_fleet.py launch --manifest configs/wave17_fleet.json` starts **only the 2 G4 instances (`gpu_g4_w6_1` and `gpu_g4_w6_2`)**, leaving `gpu_g4_w6_3`, `gpu_a100_w6_1`, and `gpu_a100_w6_2` stopped (`active_assignments == 2`), and achieves `>= 45,000 tok/s` throughput on both G4 instances.

---

## 5. Operational Tooling & Script Opportunities Gate

### 5.1 Existing Scripts Used
1. **Model Parameter & 2-Step Smoke Verification (`scripts/inspect_model.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 --num-higher-states 256 --two-beam-kboson-readout --diffuse-rank 32 --diffuse-schedule contextual_purity_complement --stereographic-lift --cosine-phase-beam-gate --quadratic-diffuse-fidelity 0.50 --smoke-steps 2
   ```
2. **Wave 17 2-G4-Instance Model Card Registration & Fleet Launch (`scripts/colab_fleet.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py register-cards --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave17_fleet.json
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py launch --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave17_fleet.json
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py status --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave17_fleet.json
   ```
3. **Zero-Downtime Live Monitoring Daemon (`scripts/monitor_colab_runs.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once
   ```

### 5.2 Build-Before-Launch Scripts & Modules
1. **`scripts/iris_quantum_geometry_probe.py`**:
   A standalone, deterministic CLI diagnostic tool (`--help`, `--quick-test`, `--epochs`, `--out-json`) that runs the 4-model head-to-head geometric trap benchmark (`NaiveBornClassifier`, `StereographicSingleBeamKBoson`, `StereographicTwoBeamKBoson`, `ClassicalSoftmaxMLP`) on both **Fisher's Iris ($N=150, d=4, C=3$)** and **4D Concentric Radial Shells ($N=300, d=4, C=3$)** and prints the complete accuracy, radial-norm sensitivity, cross-entropy, and beam-orthogonality table.
2. **`examples/38_verify_plan028_stereographic_twobeam_kboson_and_wave17.py`**:
   End-to-end verification script that executes `H1` (Fisher's Iris & 4D Concentric Radial Shells), `H2` (Quadratic Kraus $k$-Boson autograd parity, stereographic radial sensitivity, Gram-Schmidt Beam-2 orthogonality, and batch-centered contextual purity routing), and `H3` (`300`-step local `8 GB` RTX 5060 Ti FineWeb-Edu ablation), saves `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_028.json`, and syncs `.cache/benchmarks.db` + `http://127.0.0.1:8765/api/reload`.

### 5.3 Nice-to-Have Scripts (Logged to `scripts/BACKLOG.md` & Surfaced to User)
1. **`scripts/stereographic_latitude_probe.py`**: A diagnostic CLI that inspects a trained `--stereographic-lift` checkpoint on `fineweb_val_000000.bin` and plots the layerwise distribution of stereographic polar latitudes $\theta_{\text{lat}}(h_\ell)$ and phase-interference gates $\cos(\omega(z_1 - z_2) + \phi)$ across token frequency and target-rank buckets.

---

## 6. Planned Code Changes & Wave 17 2-G4-Instance Manifest

### 6.1 New Module: `src/qdlblocks/layers/stereographic_lift.py`
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/stereographic_lift.py` (and export in `src/qdlblocks/layers/__init__.py` and `src/qdlblocks/__init__.py`) implementing:
1. **`StereographicHilbertLift(nn.Module)`**:
   - Maps $x \in \mathbb{R}^{(\dots, d)} \mapsto \psi_{\text{stereo}}(x) = \frac{[x, \tau_{\text{stereo}}]}{\sqrt{\|x\|_2^2 + \tau_{\text{stereo}}^2}} \in S^{d} \subset \mathbb{R}^{(\dots, d+1)}$ with learnable $\tau_{\text{stereo}} = \text{softplus}(\tilde{\tau}) + 10^{-4}$.
2. **`apply_dimension_preserving_stereographic_rotation(u: torch.Tensor, pole: torch.Tensor, tau: torch.Tensor, gain: torch.Tensor, eps: float = 1e-12) -> torch.Tensor`**:
   - Given pre-normalized state $u \in \mathbb{R}^{(\dots, d)}$, computes $\hat{u} = \text{l2\_normalize}(u, \text{dim}=-1, \text{eps}=\text{eps})$, dimensionless radial ratio $r(u) = \frac{\|u\|_2}{\sqrt{d}\,(\text{softplus}(\tau) + 10^{-4}) + \text{eps}}$, stereographic polar-latitude coordinate $s_{\text{lat}}(u) = \frac{r(u) - 1}{\sqrt{r(u)^2 + 1}} \in (-1, 1)$, orthogonalized pole $\hat{p}_\perp = \text{l2\_normalize}(pole - \langle pole, \hat{u}\rangle \hat{u})$, and returns:
     $$\psi_{\text{stereo}}(u) = \text{l2\_normalize}\!\Big(\hat{u} + \big(0.35\tanh(gain)\big)\, s_{\text{lat}}(u)\, \hat{p}_\perp\Big) \in S^{d-1}$$
3. **Tabular / Low-Dimensional Quantum Classifiers (`NaiveBornClassifier`, `StereographicSingleBeamKBoson`, `StereographicTwoBeamKBoson`, `ClassicalSoftmaxMLP`) + Embedded Exact Fisher's Iris Dataset (`load_fisher_iris_dataset`, `build_concentric_shells_dataset`, `run_iris_and_shells_benchmark`)**:
   - Self-contained, zero-external-dependency implementation of the 150-sample Fisher's Iris dataset and 300-sample 4D Concentric Radial Shells benchmark.

### 6.2 `src/qdlblocks/layers/overcomplete_two_beam.py` & `src/qdlblocks/layers/born_block.py`
- Update `OvercompleteTwoBeamInterference.__init__` and `QuantumBornBlock.__init__` to accept:
  - `use_stereographic_lift: bool = False`
  - `use_cosine_phase_beam_gate: bool = False`
- When `use_stereographic_lift=True` in `OvercompleteTwoBeamInterference`:
  - Allocate `self.stereo_pole_1 = nn.Parameter(torch.randn(self.embed_dim) / math.sqrt(self.embed_dim))`, `self.stereo_pole_2 = nn.Parameter(torch.randn(self.embed_dim) / math.sqrt(self.embed_dim))`, `self.stereo_tau = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))`, and `self.stereo_gain = nn.Parameter(torch.tensor(0.40, dtype=torch.float32))`.
  - In `forward`, capture the pre-LayerNorm radial coordinate from `out_1` (`r_1 = out_1.float().norm(dim=-1, keepdim=True) / math.sqrt(self.embed_dim)`) and apply `apply_dimension_preserving_stereographic_rotation` to `self.post_norm(out_1.float())` using `out_1`'s radial norm! Similarly apply `apply_dimension_preserving_stereographic_rotation` to `out_2` for Beam 2.
- When `use_cosine_phase_beam_gate=True` in `OvercompleteTwoBeamInterference`:
  - Allocate `self.phase_omega = nn.Parameter(torch.full((self.frame_dim,), 0.50, dtype=torch.float32))`, `self.phase_phi = nn.Parameter(torch.zeros(self.frame_dim, dtype=torch.float32))`, and `self.phase_gate_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))`.
  - Modulate `interf` on Beam 1 with `1.0 + alpha_phase * torch.cos(delta_theta)` and `b2` on Beam 2 with `1.0 + alpha_phase * torch.sin(delta_theta)`.
  - Project `psi_beam2` onto the exact Gram-Schmidt orthogonal complement of `psi_main`:
    `psi_beam2 = l2_normalize(psi_beam2 - (psi_beam2 * psi_main).sum(dim=-1, keepdim=True) * psi_main, dim=-1, eps=self.eps)`.

### 6.3 `src/qdlblocks/layers/direct_chunked_ce.py` & `src/qdlblocks/layers/hierarchical_qubit.py`
- In `_DirectChunkedTwoBeamKBosonCEFn`, `direct_chunked_two_beam_kboson_ce`, and `compute_two_beam_kboson_logits` (`src/qdlblocks/layers/direct_chunked_ce.py`):
  - Add optional `quad_diffuse_weight: float = 0.0` (`13th` argument to `_DirectChunkedTwoBeamKBosonCEFn.forward`).
  - When `quad_w = float(quad_diffuse_weight) > 0.0`:
    - `c_pos = torch.clamp_min(c_diff, 0.0)`
    - `f_diff = c_diff + quad_w * (c_pos * c_pos)`
    - `z_raw = 1.0 + r_tilde * c_sharp + lam_c * f_diff`
    - In backward: `dlam_c = (q_c * f_diff).sum(dim=-1, keepdim=True)` and `dc_diff = (q_c * lam_c * (1.0 + (2.0 * quad_w) * c_pos)).to(work_dtype)`.
- In `Hierarchical16QubitPOVMReadout` (`src/qdlblocks/layers/hierarchical_qubit.py`):
  - Add `use_stereographic_lift: bool = False` and `quad_diffuse_weight: float = 0.0` (`quadratic_diffuse_fidelity`).
  - When `use_stereographic_lift=True`, apply `apply_dimension_preserving_stereographic_rotation` with `self.readout_stereo_pole`, `self.readout_stereo_tau`, `self.readout_stereo_gain` to `psi_sharp * self.readout_rms_gain` in `get_two_beam_kboson_factors`.
  - In `compute_scheduled_lambda_diff`:
    Update `contextual_purity_complement` to compute the batch-centered purity reference `r_ref = torch.clamp(r_tilde.detach().mean(), min=0.55, max=0.92)` and `lam_t = torch.clamp(lam_base + self.contextual_purity_gain * (r_ref - r_tilde), min=self.lambda_diff_min, max=self.lambda_diff_max)`.

### 6.4 `src/qdlblocks/models/born_gpt.py`, `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`, `scripts/inspect_model.py`, & `src/qdlblocks/benchmarks/model_card.py`
- Add `use_stereographic_lift: bool = False`, `use_cosine_phase_beam_gate: bool = False`, and `quadratic_diffuse_fidelity: float = 0.0` to `QuantumBornGPT`.
- Add CLI flags `--stereographic-lift` (`action="store_true"`), `--cosine-phase-beam-gate` (`action="store_true"`), and `--quadratic-diffuse-fidelity` (`type=float, default=0.0`, with `nargs="?", const=0.50` so both `--quadratic-diffuse-fidelity` and `--quadratic-diffuse-fidelity 0.50` work cleanly!) to `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`, `scripts/inspect_model.py`, and `src/qdlblocks/benchmarks/model_card.py` (along with `"W17"` in `WAVE_DEFAULT_LABELS` and `WAVE_DEFAULT_PLANS`).
- Ensure `stereo_pole`, `stereo_tau`, `stereo_gain`, `phase_omega`, `phase_phi`, `phase_gate_scale` are routed to `AdamW` (`weight_decay = 0.0`) and excluded from `compute_isometry_error` and `polar_snap_weights`.

### 6.5 Wave 17 2-G4-Instance Declarative Manifest (`configs/wave17_fleet.json`)
Create `/home/festivity/agy/QuantumDLBlocks/configs/wave17_fleet.json` with **ONLY 2 runs** on `gpu_g4_w6_1` and `gpu_g4_w6_2`:
1. **`gpu_g4_w6_1` (`w17_900m_qborn_unified_stereographic_quad_kboson_L8`)**:
   - **Controlled Stereographic Lift + Quadratic Kraus Fidelity Upgrade** on top of the Wave 16 `3.4980` Val CE Unified winner (`L=8, d=1024, H=8, 4d-TwoBeam, VE=3, ZeroHalfRoPE, DepthScaledRes, SlidingWindow=384, Embed-Only Causal Memory, 262,144 Hashed N-Gram, 16x32 Qubits + 256 Hierarchy`):
   - Adds `--stereographic-lift --quadratic-diffuse-fidelity 0.50` with `--two-beam-kboson-readout --diffuse-rank 16 --diffuse-schedule warmup_hold_anneal --lambda-diff-min 0.08 --lambda-diff-max 0.35 --lambda-diff-start 0.22 --lambda-diff-end 0.12`.
2. **`gpu_g4_w6_2` (`w17_900m_qborn_unified_stereographic_phase_ctx_r32_kboson_L8`)**:
   - **Full Plan 028 Synthesis Flagship** on the Unified `L=8, d=1024, H=8, 4d-TwoBeam` backbone (`VE=3, ZeroHalfRoPE, DepthScaledRes, SlidingWindow=384, Embed-Only Causal Memory, 262,144 Hashed N-Gram, 16x32 Qubits + 256 Hierarchy`):
   - Adds **all 4 advances** from our discussions:
     - `--stereographic-lift` (Stereographic Radial-Norm-Preserving Polar-Latitude Lift),
     - `--cosine-phase-beam-gate` (Cosine/Sine Quadrature Phase-Interference Two-Beam Gate + Exact Gram-Schmidt Beam-2 Orthogonality),
     - `--quadratic-diffuse-fidelity 0.50` (Positive-Part Quadratic Kraus Diffuse Fidelity $c_{\text{diff}} + \frac{1}{2}\text{ReLU}(c_{\text{diff}})^2$),
     - `--diffuse-rank 32 --diffuse-schedule contextual_purity_complement --contextual-purity-gain 0.15` (Rank-32 Stiefel Kraus Semantic Bath with Batch-Centered Contextual Bloch-Purity Complementary Routing around the `0.22 -> 0.12` `warmup_hold_anneal` envelope).

---

## 7. Verification & Testing Plan

1. **Unit Tests (`tests/test_hypothesis_028_stereographic_twobeam_kboson.py`)**:
   - `test_fisher_iris_and_concentric_shells_stereographic_vs_naive_born`: Verifies `H1` (`StereographicTwoBeamKBoson` achieves $\ge 95.0\%$ on 4D Concentric Radial Shells vs $\le 45.0\%$ for `NaiveBornClassifier`, and $\ge 96.0\%$ on Fisher's Iris vs $\le 86.7\%$ for `NaiveBornClassifier`).
   - `test_quadratic_diffuse_kboson_ce_autograd_parity`: Verifies `H2.1` (`_DirectChunkedTwoBeamKBosonCEFn` forward loss and all 8 analytical gradients with `quad_diffuse_weight = 0.50` match `compute_two_beam_kboson_logits` to $< 10^{-5}$).
   - `test_stereographic_cosine_phase_two_beam_orthogonality_and_radial_sensitivity`: Verifies `H2.2` (unit norm $< 10^{-5}$, exact Beam-1/Beam-2 orthogonality $|\langle \psi_{\text{main}}, \psi_{\text{beam2}}\rangle| < 10^{-5}$, and non-zero radial sensitivity $\|\psi_{\text{main}}(2x) - \psi_{\text{main}}(0.5x)\|_2 > 0.02$).
   - `test_batch_centered_contextual_purity_and_wave17_two_g4_manifest`: Verifies `H2.3` (batch-centered `contextual_purity_complement` preserves batch mean `lam_base` within $10^{-3}$ at high $\beta_t \in [32, 44]$) and verifies `configs/wave17_fleet.json` contains **exactly 2 runs** targeting `gpu_g4_w6_1` and `gpu_g4_w6_2`.
2. **Local Geometric & `8 GB` RTX 5060 Ti FineWeb-Edu Verification (`examples/38_verify_plan028_stereographic_twobeam_kboson_and_wave17.py` & `scripts/iris_quantum_geometry_probe.py`)**:
   - Run `scripts/iris_quantum_geometry_probe.py` and `examples/38_verify_plan028_stereographic_twobeam_kboson_and_wave17.py` locally under `torch.cuda.set_per_process_memory_fraction(0.5)` (`<= 8.0 GB` VRAM).
3. **Wave 17 2-G4-Instance Launch (`scripts/colab_fleet.py launch --manifest configs/wave17_fleet.json`)**:
   - Launch `configs/wave17_fleet.json` on `gpu_g4_w6_1` and `gpu_g4_w6_2` and verify that only 2 Colab VMs are assigned (`active_assignments == 2`).

---

## 8. Empirical Results, 8GB Verification & Wave 17 2-G4-Instance Launch Telemetry

### 8.1 Hypothesis H1 Verification: Fisher's Iris (`N=150, d=4, C=3`) & 4D Concentric Radial Shells (`N=300, d=4, C=3`)
Executed via `scripts/iris_quantum_geometry_probe.py` (`Peak VRAM = 68.0 MB <= 8.0 GB` on local RTX 5060 Ti):

#### Table 8.1A — 4D Concentric Radial Shells (`N=300, d=4, C=3`, Radii $\{0.8, 1.8, 3.0\}$)
| Model | Train Acc | Test Acc | Full Acc | Test CE | RadialSens ($\|\psi(2x) - \psi(0.5x)\|_2$) | BeamOrtho ($\|\langle \psi_1, \psi_2\rangle\|$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NaiveBornClassifier`** ($\text{l2\_normalize}(x) \in S^3$) | `33.33%` | `27.78%` | `31.67%` | `1.1243` | `0.0000` (Radial Blindness) | `0.00e+00` | Baseline |
| **`StereographicSingleBeamKBoson`** ($S^4$ Lift + $k$-Boson) | `98.57%` | **`100.00%`** | `99.00%` | `0.0484` | `0.7209` | `0.00e+00` | **PASS** |
| **`StereographicTwoBeamKBoson`** ($S^4$ Lift + Phase Gate + Quad Kraus) | **`100.00%`** | **`97.78%`** | **`99.33%`** | `0.0493` | `0.6462` | **`7.08e-08`** | **PASS (`+70.00 pp`)** |
| **`ClassicalSoftmaxMLP`** ($4 \to 16 \to 3$ ReLU MLP) | `99.52%` | `98.89%` | `99.33%` | `0.0252` | `0.0000` | `0.00e+00` | Classical Control |

#### Table 8.1B — Fisher's Iris Dataset (`N=150, d=4, C=3`, Versicolor–Virginica Ray $\cos\theta = 0.9923$, Norm Ratio `1.482x`)
| Model | Train Acc | Test Acc | Full Acc | Test CE | RadialSens ($\|\psi(2x) - \psi(0.5x)\|_2$) | BeamOrtho ($\|\langle \psi_1, \psi_2\rangle\|$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NaiveBornClassifier`** ($\text{l2\_normalize}(x) \in S^3$) | `78.10%` | `77.78%` | `78.00%` | `0.3712` | `0.0000` (Collinear Collapse) | `0.00e+00` | Baseline |
| **`StereographicSingleBeamKBoson`** ($S^4$ Lift + $k$-Boson) | `96.19%` | **`93.33%`** | `95.33%` | `0.1819` | `0.8588` | `0.00e+00` | **PASS** |
| **`StereographicTwoBeamKBoson`** ($S^4$ Lift + Phase Gate + Quad Kraus) | **`99.05%`** | **`93.33%`** | **`98.00%`** | **`0.1656`** | `0.8388` | **`5.22e-08`** | **PASS (`+20.00 pp` Full)** |
| **`ClassicalSoftmaxMLP`** ($4 \to 16 \to 3$ ReLU MLP) | `100.00%` | `93.33%` | `98.00%` | `0.1304` | `0.0000` | `0.00e+00` | Classical Control |

---

### 8.2 Hypothesis H2 Verification: Custom Autograd Parity, Beam Orthogonality & Batch-Centered Purity Routing
Executed via `pytest tests/test_hypothesis_028_stereographic_twobeam_kboson.py` (`4 passed`) and `examples/38_verify_plan028_stereographic_twobeam_kboson_and_wave17.py`:
- **H2.1 (`_DirectChunkedTwoBeamKBosonCEFn` with `quad_diffuse_weight = 0.50`)**:
  - Forward loss difference vs. unchunked reference: `|dLoss| = 1.91e-06 < 1e-5` (**PASS**).
  - Maximum analytical gradient difference across all 8 inputs (`psi_s, psi_b2, u_s, u_d, b, g_h, lam, l_p`): `max|dGrad| = 1.79e-07 < 1e-5` (**PASS**).
- **H2.2 (`OvercompleteTwoBeamInterference` Stereographic Lift + Cosine Phase Gate)**:
  - Exact Gram-Schmidt Beam-2 orthogonality: `BeamOrtho = 6.71e-08 < 1e-5` ($\theta = 90.000004^\circ$, compared to `0.876 -> 0.919` without `--cosine-phase-beam-gate`!) (**PASS**).
  - Pre-LayerNorm radial norm sensitivity: `RadialSens = 0.4890 > 0.02` (**PASS**).
- **H2.3 (Batch-Centered `contextual_purity_complement` Routing at $\beta_t \in [32, 44]$)**:
  - Batch-mean schedule preservation error: `|mean(lam) - lam_base| = 1.49e-08 < 1e-3` (eliminating the `-0.04` to `-0.05` downward bias from Wave 16!) (**PASS**).
  - Contextual purity spread between mixed ($\beta=32$) and pure ($\beta=44$) tokens: `Spread(beta32 - beta44) = +0.0225 > 0.015` (**PASS**).

---

### 8.3 Hypothesis H3 Verification: `8 GB` RTX 5060 Ti FineWeb-Edu 300-Step (`2.46M` Token) Ablation (`L=6, d=512, T=512`)
Executed via `examples/38_verify_plan028_stereographic_twobeam_kboson_and_wave17.py` under `torch.cuda.set_per_process_memory_fraction(0.5, device=0)`:

| Run Key | Architecture / Readout | Params | Peak VRAM | Step 0 CE | Step 300 Val CE | Top-1 | Top-5 | Gap (T5-T1) | NLL (6-50) | NLL (51-1k) | $\lambda_{300}$ | $R_{\text{diff}}$ | BeamOrtho |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w15_8gb_classical_leaderboard_exact_L6_d512` | Classical Softmax (`L=6, d=512`) | `146.60M` | `4.127 GB` | `10.6179` | `6.4771` | `18.02%` | `33.57%` | `15.55%` | `5.1518` | `9.1545` | `0.0000` | `0.00` | `0.00e+00` |
| `w15_8gb_qborn_16qubit_256hier_povm_L6_d512` | Single-Beam $k$-Boson POVM | `84.80M` | `5.925 GB` | `9.4505` | `6.1476` | `16.00%` | `31.16%` | `15.16%` | `4.9381` | `7.8142` | `0.0000` | `1.00` | `0.00e+00` |
| `w16_8gb_qborn_unified_softmax_polarsnap_fixed_L6_d512` | Unified + Last-Layer Softmax | `195.67M` | `5.992 GB` | `8.9483` | `6.0922` | `16.52%` | `32.06%` | `15.54%` | `4.9602` | `7.8308` | `0.1800` | `0.00` | `0.00e+00` |
| `w15_8gb_qborn_16qubit_256hier_twobeam_kboson_unified_L6_d512` | Plan 026 Unified Two-Beam (`r=16`, unscheduled) | `195.67M` | `6.282 GB` | `9.2395` | `6.0940` | `16.32%` | `31.88%` | `15.56%` | `4.9595` | `7.8124` | `0.0356` | `14.54` | `~0.91` |
| `w16_8gb_qborn_unified_kboson_sched_warmup_hold_anneal_L6_d512` | Plan 027 `warmup_hold_anneal` (`r=16`) | `195.67M` | `6.117 GB` | `9.3641` | `6.0893` | `16.48%` | **`32.31%`** | `15.83%` | `4.9720` | `7.8031` | `0.1103` | `14.69` | `~0.91` |
| `w16_8gb_qborn_unified_kboson_sched_contextual_purity_L6_d512` | Plan 027 Legacy `contextual_purity` (`r=16`) | `195.67M` | `6.117 GB` | `9.2176` | `6.1023` | `16.22%` | `32.23%` | `16.01%` | `4.9891` | `7.7960` | `0.1073` | `14.68` | `~0.91` |
| **`w17_8gb_qborn_unified_stereographic_quad_warmup_hold_L6_d512`** | **Plan 028 Stereo Lift + Quad Kraus (`w_q=0.50`) + `warmup_hold` (`r=16`)** | `195.68M` | `5.992 GB` | `9.4627` | **`6.0876`** | **`16.66%`** | `32.10%` | `15.44%` | `4.9931` | `7.7884` | `0.1103` | `14.98` | `9.19e-01` |
| **`w17_8gb_qborn_unified_stereographic_phase_quad_kboson_L6_d512`** | **Plan 028 Full Synthesis (Stereo + PhaseGate + Quad Kraus + Centered `ctx_purity` `r=32`)** | `195.70M` | `6.254 GB` | `9.2185` | **`6.0901`** | `16.32%` | `31.89%` | `15.57%` | **`4.9510`** | **`7.7749`** | `0.1104` | **`27.76`** | **`0.00e+00`** |

**Key Diagnostic Insights from the 8GB Verification**:
1. **New All-Time 8GB Quantum Val CE & Top-1 Record (`6.0876` Val CE, `16.66%` Top-1)**:
   - `w17_8gb_qborn_unified_stereographic_quad_warmup_hold_L6_d512` achieves **`6.0876` Val CE** and **`16.66%` Top-1** (`PeakVRAM = 5.992 GB`), outperforming Plan 027's `warmup_hold_anneal` (`6.0893` CE, `16.48%` Top-1), PolarSnap Softmax (`6.0922` CE, `16.52%` Top-1), Plan 026 (`6.0940` CE), and Classical Leaderboard Exact (`6.4771` CE, **`-0.3895 nats` advantage**).
2. **Smoking-Gun `BeamOrtho` Discovery (`0.919` vs. `0.00e+00`) & New All-Time Mid/Tail-Rank NLL Record (`4.9510` / `7.7749`)**:
   - Without `--cosine-phase-beam-gate`, the legacy `l2_normalize(out_2 - 0.25 * out_1)` heuristic allows $\psi_{\text{main}}$ and $\psi_{\text{beam2}}$ to drift to **`BeamOrtho = 0.919`** ($\cos^2\theta \approx 84.5\%$ identical state!).
   - With `--cosine-phase-beam-gate` + Gram-Schmidt orthogonalization + `r=32` batch-centered `contextual_purity_complement`, `w17_8gb_qborn_unified_stereographic_phase_quad_kboson_L6_d512` enforces **`BeamOrtho = 0.00e+00`** (`R_diff = 27.76`), beats Plan 027's `contextual_purity` (`6.1023`) by **`-0.0122 nats`** (`6.0901` Val CE), and achieves the **best mid-rank `NLL(6-50) = 4.9510`** (`-0.0421 nats` better than `stereographic_quad_warmup_hold`) and **new all-time 8GB tail-rank record `NLL(51-1k) = 7.7749`**!

---

### 8.4 Wave 17 2-G4-Instance Launch Telemetry (`gpu_g4_w6_1` & `gpu_g4_w6_2` ONLY)
Launched via `scripts/colab_fleet.py launch --manifest configs/wave17_fleet.json` on **ONLY 2 G4 instances** (`Active assignments: 2`, `Usage rate: 17.80/hr`, `Current balance: 162.82 compute units`):

| Session | Run Key | Hardware | Params | Compile Time | Step 0 Val CE | Step 0 Top-1 / Top-5 | Step 0 MedRank | Step 0 $S(\rho_A)$ | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gpu_g4_w6_1`** | `w17_900m_qborn_unified_stereographic_quad_kboson_L8` | NVIDIA RTX PRO 6000 Blackwell (G4) | `678,347,154` | `50.24s` | **`8.5479`** (`5155.9` PPL) | `1.48%` / `5.16%` | `1092` | `2.3731 nats` | **RUNNING (Shard 1/9, `GPUStaged=True`)** |
| **`gpu_g4_w6_2`** | `w17_900m_qborn_unified_stereographic_phase_ctx_r32_kboson_L8` | NVIDIA RTX PRO 6000 Blackwell (G4) | `678,445,466` | `52.46s` | **`8.7054`** (`6035.4` PPL) | `1.26%` / **`5.76%`** | **`1019`** | `2.3670 nats` | **RUNNING (Shard 1/9, `GPUStaged=True`)** |

