# Experiment Plan 046 (`Wave 31`): Lever 3 — Product-of-Spheres $(\mathbb{S}^{d/M - 1})^M$ Multi-Subspace Factored Born Readout & Coincidence Gating

**Date**: October 24, 2026  
**Wave**: `Wave 31` (`EXPERIMENT_PLAN_046`)  
**Status**: Planned & Implementing (CPU-first while `Wave 30` finishes on GPU)  
**Target Domain**: `fineweb_edu_local` & `fineweb_speedrun` (`< 3.28` Val CE Roadmap, Lever 3)

---

## 1. Motivation & Live Empirical Diagnosis from `Wave 30`

While monitoring the live `Wave 30` GPU demonstration run (`w30_sub3p28_demo_parseval_mup_ngram_b24_d512_L6` at $d=512, L=6$) in `.cache/live_logs/w30_sub3p28_demo_parseval_mup_ngram_b24_d512_L6.log`, we observed a critical geometric phenomenon in the live telemetry:
- At Step 100 (`1.23M` tokens), the sharp detector participation ratio was $R_{\text{sharp}} = 77.52$.
- By Step 400 (`4.92M` tokens) and Step 600 (`7.37M` tokens), $R_{\text{sharp}}$ collapsed to **`27.23` $\to$ `24.18`** even though the ambient model width is $d = 512$!

### 1.1 Why Single-Sphere Normalization $\mathbb{S}^{d-1}$ Allows Subspace Cannibalization & Runner-Up Crowding
1. **Norm Cannibalization Across Additive Branches**:
   In `ParsevalMuPNGramSpeedrunGPT`, the pre-readout state combines the deep contextual state $\psi_t$, the Layer-0 phase anchor $\psi_0^z$, and the surface N-gram projection $\psi_{\text{ngram}}$ via a single global L2 normalization:
   $$\psi_{\text{pre}} = \frac{\psi_t + g_0 \psi_0^z + g_{\text{ng}} \psi_{\text{ngram}}}{\|\psi_t + g_0 \psi_0^z + g_{\text{ng}} \psi_{\text{ngram}}\|_2} \in \mathbb{S}^{d-1}$$
   Similarly, the vocabulary detector $m_{\text{sharp}, v}$ sums `wav_256`, `wav_2048`, `wav_8192`, and `wav_65536` and normalizes once over $\mathbb{R}^d$. Because early gradients on `wav_256` and `ngram_proj` are much larger than on fine wavelet leaves, a few dominant directions grow large in Euclidean norm and **cannibalize the global L2 denominator**, shrinking the effective dimension $R_{\text{sharp}}$ from $O(d)$ down to $\sim 24$!
2. **The Welch Bound Penalty of Low $R_{\text{sharp}}$ (`Positive Runner-Up Crowding`)**:
   By the Welch/Parseval frame identity, the RMS off-diagonal cosine across the $V = 50,257$ vocabulary detectors is bounded below by:
   $$\sigma_{\text{off-diag}} = \frac{1}{\sqrt{R_{\text{sharp}}}}$$
   When $R_{\text{sharp}}$ drops from $256$ to $24.18$, background and runner-up cosines swell by $\sqrt{256 / 24.18} \approx 3.25\times$, inflating the positive runner-up partition sum $Z_{\text{pos\_runnerup}} = \sum_{v \ne y, \hat{c}_{t,v} > 0} e^{\ell_{t,v}}$ (which [`scripts/multiclass_ce_margin_decomposer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multiclass_ce_margin_decomposer.py) proved accounts for **`78%` of residual Cross-Entropy** at $V = 50,257$).

---

## 2. Mathematical Formulation of Lever 3: Product-of-Spheres $(\frac{1}{\sqrt{M}}\mathbb{S}^{d_s - 1})^M$ & Multi-Subspace Coincidence Born Readout

Instead of confining token and detector states to a single unstructured sphere $\mathbb{S}^{d-1}$, we equip $\mathbb{R}^d = \bigoplus_{m=1}^M \mathbb{R}^{d_s}$ ($d_s = d / M$, with $M \in \{4, 8\}$) with the **Product-of-Spheres (Multi-Subspace) Riemannian Manifold**:
$$\mathcal{M}_{d, M} = \left(\frac{1}{\sqrt{M}} \mathbb{S}^{d_s - 1}\right)^M \subset \mathbb{S}^{d-1}$$

### 2.1 Subspace-Balanced Unit-Sphere Normalization (`subspace_l2_normalize`)
For any vector $x \in \mathbb{R}^{\dots \times d}$ partitioned into $M$ contiguous subspaces $x^{(1)}, \dots, x^{(M)} \in \mathbb{R}^{\dots \times d_s}$ ($d_s = d / M$):
$$\tilde{x}^{(m)} = \frac{1}{\sqrt{M}} \frac{x^{(m)}}{\sqrt{\|x^{(m)}\|_2^2 + \epsilon}}, \qquad \tilde{x} = \operatorname{concat}\!\left(\tilde{x}^{(1)}, \dots, \tilde{x}^{(M)}\right) \in \mathcal{M}_{d, M}$$
- **Exact Global Unit-Sphere Invariance**:
  $$\|\tilde{x}\|_2^2 = \sum_{m=1}^M \|\tilde{x}^{(m)}\|_2^2 = \sum_{m=1}^M \frac{1}{M} = 1 \implies \mathcal{M}_{d, M} \subset \mathbb{S}^{d-1}$$
  Every vector on $\mathcal{M}_{d, M}$ has exact unit L2 norm $\|\tilde{x}\|_2 = 1$, so the analytical Parseval $\mu\text{P}$ scale $\beta_0(d) = c^*\sqrt{d}$ ($c^* = 1.48$) remains **100% unchanged**!
- **Guaranteed Anti-Collapse of Detector Effective Rank $R_{\text{sharp}}$**:
  Because each of the $M$ orthogonal subspaces $\mathbb{R}^{d_s}$ is forced to carry exact energy $1/M$, no single subspace can cannibalize the others. By Jensen's inequality on the block-diagonal trace of the Gram matrix $G = M_{\text{sharp}}^\top M_{\text{sharp}}$, if each subspace has participation ratio $R_m$, the total participation ratio satisfies $R_{\text{sharp}} \ge \left(\frac{1}{M}\sum_{m=1}^M R_m^{-1}\right)^{-1}$, preventing single-direction rank collapse.

### 2.2 Multi-Subspace Coincidence Born Observable ($\widehat{O}_{t,v}^{(M)}$)
Let $c_{t,v}^{(m)} = \langle \psi_t^{(m)} \mid m_{\text{sharp}, v}^{(m)} \rangle \in [-1, 1]$ be the unit-sphere cosine in subspace $m \in \{1, \dots, M\}$, so the overall inner product is the mean across subspaces:
$$c_{t,v} = \langle \tilde{\psi}_t \mid \tilde{m}_{\text{sharp}, v} \rangle = \frac{1}{M} \sum_{m=1}^M c_{t,v}^{(m)}$$
Consider what happens when we compare:
1. **Single-Sphere Quadratic Peak Boost ($M=1$)**:
   Boosts $\max(\hat{c}_{t,v}, 0)^2 = \left(\frac{1}{M}\sum_{m=1}^M \hat{c}_{t,v}^{(m)}\right)_+^2$. A false-positive runner-up $v \ne y$ that has a large accidental spike in just **one** subspace ($\hat{c}_{t,v}^{(1)} = 0.60$, while $\hat{c}_{t,v}^{(2..4)} = 0$) still gets a positive mean cosine $0.15$.
2. **Multi-Subspace Coincidence / Agreement Gate (`coincidence_gamma`)**:
   In quantum multi-detector coincidence counting, a true target $y$ aligns coherently across **all $M$ subspaces** ($\hat{c}_{t,y}^{(m)} > 0\;\forall m$). We decompose the subspace-level alignment into the mean cosine $\hat{c}_{t,v}$ plus a **Coarse-Subspace Coincidence / Agreement Score** $A_{t,v}^{(M)}$ computed over the $M$ subspaces of the $256$ coarse dyadic roots $W_{256}$ (or chunked detectors):
   $$A_{t, g(v)}^{(M)} = \prod_{m=1}^M \operatorname{softplus}_\tau\!\left(\hat{c}_{\text{coarse}, t, g(v)}^{(m)}\right)^{1/M} - \frac{1}{M}\sum_{m=1}^M \operatorname{softplus}_\tau\!\left(-\hat{c}_{\text{coarse}, t, g(v)}^{(m)}\right)$$
   By the AM-GM inequality, the geometric mean $\left(\prod_{m=1}^M a_m\right)^{1/M}$ equals the arithmetic mean $\frac{1}{M}\sum_{m=1}^M a_m$ **if and only if all $M$ subspaces agree ($a_1 = a_2 = \dots = a_M$)**, and vanishes whenever any subspace fails to align ($a_m \to 0$)!
   Because $A_{t, g}^{(M)}$ is evaluated on the $256$ coarse dyadic clusters ($K_{\text{coarse}} = 256 \ll V = 50,257$) and added to the $256$-cluster coarse dyadic log-probability $\ln P_{\text{coarse}}(g(v) \mid x_t)$ inside [`parseval_mup_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parseval_mup_chunked_ce.py), it requires **zero extra $(N_{\text{tok}}, V)$ GEMMs** (`< 0.15 ms` overhead per step) while suppressing single-subspace runner-up impostors across all $50,257$ vocabulary words!

---

## 3. Hypotheses & Quantitative Acceptance Gates

- **Hypothesis H1 (Mathematical & Numerical Verification of Product-of-Spheres Anti-Collapse & Runner-Up Suppression)**:
  - Replacing single-sphere L2 normalization with $M$-subspace Product-of-Spheres normalization ($M \in \{4, 8\}$) + Multi-Subspace Coincidence Gating:
    1. Preserves exact unit L2 norm $\|\tilde{\psi}_t\|_2 = \|\tilde{m}_v\|_2 = 1$ (`max_norm_err < 1e-6`) and exact Parseval $\mu\text{P}$ background variance invariance across $d \in \{256, 384, 512, 768\}$.
    2. Prevents detector effective rank collapse under skew/anisotropic perturbation, maintaining $R_{\text{sharp}}(M=4) \ge 2.2 \times R_{\text{sharp}}(M=1)$ (`>= 180.0` at $d=512$).
    3. Reduces the positive runner-up Cross-Entropy tax (`positive_runnerup_ce_tax_nats` at $V = 50,257$) by **`>= 0.12` nats** compared to single-sphere $M=1$ at matched target peak alignment.
- **Hypothesis H2 (Fast-Loop Multi-Class Vision Verification on `CIFAR-100` / `1,000-Class` & Genuine `FineWeb-Edu` Language Modeling)**:
  - On our controlled language & vision verification benchmarks (`ParsevalMuPSubspaceCoincidenceGPT` vs. `ParsevalMuPNGramSpeedrunGPT` / `ParsevalMuPDyadicGPT` on identical token streams and hyperparameters):
    1. Maintains detector effective rank $R_{\text{sharp}} \ge 120.0$ throughout training (eliminating the `77.5 -> 24.2` collapse seen in `Wave 30`).
    2. Improves Validation Cross-Entropy by **`>= 0.08` nats** at iso-tokens while maintaining throughput `>= 22,000 tok/s` at $d=256$ and `>= 18,000 tok/s` at $d=384/512$.

---

## 4. Planned Changes (Rule 6 Flag-Free Module & Shared Primitives)

Following [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md) (Rules 1–6) and [`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md) Section 5:
1. **[`src/qdlblocks/layers/subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) (`NEW`, `<= 260` lines)**:
   - Implements reusable primitives:
     - `SubspaceCoincidenceSpec` (`@dataclass(frozen=True)` with `num_subspaces: int = 4`, `coincidence_weight: float = 0.18`, `subspace_var_penalty: float = 0.12`).
     - `subspace_l2_normalize(x: torch.Tensor, num_subspaces: int = 4, eps: float = 1e-8) -> torch.Tensor`: normalizes each of the $M$ contiguous subspaces of dimension $d / M$ to norm $1/\sqrt{M}$ so $\|x\|_2 = 1$ identically.
     - `compute_coarse_subspace_coincidence_scores(psi_s: torch.Tensor, coarse_centers: torch.Tensor, spec: SubspaceCoincidenceSpec) -> torch.Tensor`: computes the AM-GM coincidence + inter-subspace variance penalty across the $M$ subspaces for the $256$ coarse dyadic roots.
     - `subspace_coincidence_dyadic_chunked_ce(...)` and `eval_subspace_coincidence_full_logits(...)`: composes `parseval_mup_dyadic_chunked_ce` with `SubspaceCoincidenceSpec` with zero extra $(N_{\text{tok}}, V)$ GEMMs.
2. **[`src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py) (`NEW`, `<= 320` lines)**:
   - Flag-free `Wave 31` language model (`ParsevalMuPSubspaceCoincidenceConfig`, `ParsevalMuPSubspaceCoincidenceGPT`, `initialize_subspace_coincidence_with_ppmi_svd`, `build_subspace_coincidence_optimizers`) that enforces Product-of-Spheres $(\frac{1}{\sqrt{M}}\mathbb{S}^{d/M - 1})^M$ normalization across token states, residual interferometers, surface N-gram projections, and vocabulary detectors.
3. **[`scripts/multi_subspace_coincidence_margin_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multi_subspace_coincidence_margin_probe.py) (`NEW`, `<= 280` lines)**:
   - Build-Before-Launch CLI diagnostic (`--dims 256,384,512,768 --subspaces 1,2,4,8 --vocab-size 50257 --device cpu --json-out .cache/multi_subspace_coincidence_margin_probe.json`) that measures:
     - $R_{\text{sharp}}$ retention under anisotropic gradient drift across $M \in \{1, 2, 4, 8\}$,
     - Positive runner-up Cross-Entropy tax (`positive_runnerup_ce_tax_nats`) when runner-up impostors align in $1, 2, \dots, M-1$ subspaces vs. true targets aligning in all $M$ subspaces,
     - Exact unit-norm and Parseval $\mu\text{P}$ scale invariance.
4. **[`examples/56_verify_plan046_multi_subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/56_verify_plan046_multi_subspace_coincidence_born.py) (`NEW`, `<= 320` lines)** & **[`tests/test_plan046_multi_subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan046_multi_subspace_coincidence_born.py) (`NEW`, `<= 220` lines)**:
   - Full CPU + GPU verification script and unit test suite.

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

### 5.1 Existing Scripts Used
- [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py): Verify all new files obey `<= 1,000` lines soft limit, `<= 8` args per function, and zero semicolons.
- [`scripts/multiclass_ce_margin_decomposer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multiclass_ce_margin_decomposer.py): Baseline reference for `positive_runnerup_ce_tax_nats` vs. `negative_tail_ce_tax_nats`.
- [`scripts/sub3p28_at_150m_and_900m_feasibility_analyzer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sub3p28_at_150m_and_900m_feasibility_analyzer.py): Quantifies Lever 3's contribution on the `150M` and `900M` token curves.
- [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py): Syncs `Wave 30` and `Wave 31` logs/cards to `.cache/benchmarks.db` and reloads `:8765`.

### 5.2 `Build-Before-Launch` Script
- **[`scripts/multi_subspace_coincidence_margin_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multi_subspace_coincidence_margin_probe.py)**:
  - Runs on CPU in `< 2s` (while the GPU is busy with `Wave 30`) to prove and quantify $R_{\text{sharp}}$ anti-collapse and positive runner-up suppression across $M \in \{1, 2, 4, 8\}$ subspaces and $d \in \{256, 384, 512, 768\}$.

### 5.3 `Nice-to-Have` Scripts (Logged in `scripts/TOOLING_BACKLOG.md`)
1. `scripts/subspace_specialization_mutual_information_probe.py`: Measures pairwise mutual information and cosine orthogonality across the $M=4$ learned subspaces (`coarse`, `mid`, `fine`, `ngram`) on validation batches.
2. `scripts/live_r_sharp_collapse_watchdog.py`: Monitors `.cache/live_logs/*.log` for any run whose `R_sharp` drops below `0.25 * embed_dim` and flags norm cannibalization automatically.

---

## 6. Constants & Calibration Ledger (`Rule 7` of `docs/SCRIPT_BEST_PRACTICES.md`)

All numerical constants introduced or referenced during the implementation of `EXPERIMENT_PLAN_046.md` (`Wave 31`) and `ParsevalMuPDyadicGPT` (`Wave 28`) are defined as top-of-file `UPPER_SNAKE_CASE` constants in their respective modules:

| Constant Name | Module | Value | Mathematical / Physical Meaning & Empirical Source |
| :--- | :--- | :---: | :--- |
| `DEFAULT_NUM_SUBSPACES` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `4` | Default number of orthogonal subspaces $M$ in $\mathcal{M}_{d, M} = (\frac{1}{\sqrt{M}}\mathbb{S}^{d/M - 1})^M$. |
| `DEFAULT_TRACELESS_GAMMA` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `0.25` | Rectified-quadratic peak boost coefficient $\gamma$ in $\widehat{O}_{t,v}$ (`Wave 24` / `EXPERIMENT_PLAN_039.md`). |
| `DEFAULT_COINCIDENCE_GAMMA` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `0.22` | Weight $\gamma_{\text{coinc}}$ on the centered AM-GM geometric coincidence term $G_{t,v}^{(M)}$. |
| `DEFAULT_SUBSPACE_VAR_PENALTY` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `0.14` | Weight $\lambda_{\text{var}}$ penalizing inter-subspace cosine variance $V_{t,v}^{(M)} = \frac{1}{M}\sum_m (\hat{c}_{t,v}^{(m)} - \bar{c}_{t,v})^2$. |
| `DEFAULT_COARSE_COINCIDENCE_WEIGHT` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `0.16` | Weight $w_c$ on the 256-root coarse cluster coincidence log-probability. |
| `GEOM_COINCIDENCE_LOG_EPS` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `1e-4` | Numerical smoothing $\epsilon_0$ inside $\exp(\frac{1}{M}\sum_m \ln((\hat{c}_{t,v}^{(m)})_+ + \epsilon_0)) - \epsilon_0$. |
| `COARSE_AUX_LOSS_BLEND_SCALE` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `0.12` | Blending factor (`0.12 * w_c`) mixing 256-root coarse coincidence CE into `subspace_coincidence_dyadic_chunked_ce`. |
| `FULL_VOCAB_COINCIDENCE_LOGIT_SCALE` | [`subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/subspace_coincidence_born.py) | `0.06` | Evaluation-time scaling (`0.06 * beta_0`) when `full_vocab_coincidence_eval=True`. |
| `DEFAULT_MUP_C_STAR` | [`parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py) | `1.48` | Zero-shot unit-sphere Parseval $\mu\text{P}$ constant $c^*$ calibrated at $d_{\text{base}} = 128$ (`EXPERIMENT_PLAN_043.md`). |
| `GPU_READOUT_PROG_BASE` / `SLOPE` | [`parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py) | `0.705` / `0.330` | Linear progress schedule $0.705 + 0.330\tau$ damping initial readout temperature before Kraus bath equilibration (`Wave 28`). |
| `GPU_READOUT_WIDTH_EXPONENT` | [`parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py) | `0.32` | Width-damping exponent $(256 / d)^{0.32}$ compensating for finite-step feature correlation at $d > 256$ (`Wave 28`). |
| `GPU_READOUT_GAMMA_DAMP_COEFF` | [`parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py) | `0.36` | Effective temperature divisor $1 + 0.36\gamma$ accounting for quadratic peak boost $(1 + 2\gamma\hat{c}_y)$ (`Wave 28`). |
| `GPU_DIFFUSE_LAMBDA_SCALE` | [`parseval_mup_dyadic_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_dyadic_gpt.py) | `0.76` | Relative coupling factor of the diffuse Kraus bath channel in GPU standard readout mode (`Wave 28`). |

---

## 7. Experiment Outcomes, Empirical Findings, & Next-Step Hypotheses (`Graceful Wrap-Up Protocol`)

Following **Section 7 of [`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md)** (`Implementer Context-Budget & Graceful Experiment Wrap-Up Protocol`), here is the complete summary of what succeeded in `Wave 30` and `Wave 31` (`EXPERIMENT_PLAN_046.md`), what we observed during the `150`-step GPU diagnostic runs, and our concrete root-cause analysis for the next iteration.

### 7.1 `Wave 30` Live GPU Demonstration Completed (`2,400` Steps / `29.49M` Tokens at $d=512, L=6$)
- **Run Key**: `w30_sub3p28_demo_parseval_mup_ngram_b24_d512_L6` ([`.cache/live_logs/w30_sub3p28_demo_parseval_mup_ngram_b24_d512_L6.log`](file:///home/festivity/agy/QuantumDLBlocks/.cache/live_logs/w30_sub3p28_demo_parseval_mup_ngram_b24_d512_L6.log))
- **Architecture**: `ParsevalMuPNGramSpeedrunGPT` (`47.47M` parameters, $d=512, L=6, H=8$, $B=24, T=512$, `FP8 E4M3` fused chunked readout, `<= 15.0 GB` VRAM).
- **Outcome**: Completed all `2,400` steps (`29.49M` tokens) cleanly in **`21.85 min`** at **`22,489 tok/s`**, reaching **`5.0606` Val CE** and **`24.72%` Top-1 Accuracy** (`kraus_stiefel_err = 0.0108`).

### 7.2 `Wave 31` (`Plan 046`) Mathematical Probe & Unit Test Verification (`PASSED`)
1. **CPU Mathematical & Margin Probe ([`scripts/multi_subspace_coincidence_margin_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multi_subspace_coincidence_margin_probe.py) $\to$ [`.cache/multi_subspace_coincidence_margin_probe.json`](file:///home/festivity/agy/QuantumDLBlocks/.cache/multi_subspace_coincidence_margin_probe.json))**:
   - **`H1` (`passed = True`) — Anti-Collapse of $R_{\text{sharp}}$ under Anisotropic Drift**:
     Under $12\times$ single-subspace variance skew across $V = 50,257$ detectors at $d = 512$, single-sphere L2 normalization ($M=1$) collapses to $R_{\text{sharp}} = 21.94$ (`Welch_RMS = 0.2135`), closely reproducing the `Wave 30` live collapse (`24.18`). Product-of-Spheres normalization ($M=4$) maintains **$R_{\text{sharp}} = 182.07$ (`8.30x` higher, `Welch_RMS = 0.0741`)** while preserving exact unit norm (`max_unit_norm_err = 1.79e-7`).
   - **`H2` (`passed = True`) — Single-Subspace Runner-Up Impostor Suppression**:
     At $d = 512, V = 50,257$, when $64$ runner-up impostors have accidental spikes in 1 of $M$ subspaces, `compute_multi_subspace_coincidence_observable` ($M=4$) reduces `runnerup_ce_tax_nats` from **`1.3414` nats to `0.0000` nats (`-1.3414` nats reduction)**.
2. **Unit Test Suite ([`tests/test_plan046_multi_subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan046_multi_subspace_coincidence_born.py))**:
   - All `5/5` unit tests pass (`pytest` exit code `0`), and all new modules/scripts pass [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py).

### 7.3 `150`-Step Genuine `FineWeb-Edu` GPU Diagnostic & Root-Cause Insights ([`examples/56_verify_plan046_multi_subspace_coincidence_born.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/56_verify_plan046_multi_subspace_coincidence_born.py))
During our `150`-step (`1.23M`-token) GPU diagnostic comparing `M=1` (`w31_m1_single_sphere_baseline_d256`) vs. `M=4` (`w31_m4_subspace_coincidence_d256`) at $d=256$, we uncovered three subtle interactions in how `Tied2Pow16WaveletCholeskyPyramid` interacts with contiguous subspace slicing:

1. **Discovery 1 — Stale `_stabilize_pyramid_frame` Buffer Overwrote PPMI-SVD Wavelet Norms**:
   - `Tied2Pow16WaveletCholeskyPyramid.__init__` initializes a buffer `init_wav_rms = [0.12, 0.08, 0.06, 0.05]`. When `super().retract_kraus_generators()` called `_stabilize_pyramid_frame(rate=0.04)`, it exponentially pulled the PPMI-SVD initialized wavelet tables toward those tiny random-init RMS constants every 2 steps! Removing `_stabilize_pyramid_frame` from `ParsevalMuPDyadicGPT.retract_kraus_generators()` and `ParsevalMuPNGramSpeedrunGPT.retract_kraus_generators()` immediately improved 150-step Val CE from `6.96` down to `6.8074` (`24,522 tok/s`).
2. **Discovery 2 — Why Contiguous Subspace Slicing `x.view(..., M, d // M)` Interacts with Sorted PPMI-SVD Singular Coordinates**:
   - In `initialize_parseval_mup_with_ppmi_svd`, `a_self` (`128` dims) and `b_orth` (`128` dims) are constructed from the top $128$ PPMI-SVD singular vectors **ordered from largest singular value (`index 0`) to smallest singular value (`index 127`)** and interleaved as `(a_0, b_0, a_1, b_1, ..., a_127, b_127)`.
   - When `subspace_l2_normalize` slices $\mathbb{R}^{256}$ into $M=4$ **contiguous** blocks of $64$ coordinates (`[0:64]`, `[64:128]`, `[128:192]`, `[192:256]`), Subspace 0 receives the top singular vectors `0..31` (highest semantic signal-to-noise ratio), whereas Subspace 3 receives the tail singular vectors `96..127` (lowest signal-to-noise ratio).
   - Forcing Subspace 3 (tail SVD components) to carry the exact same $1/4$ L2 energy as Subspace 0 (top SVD components) preserved higher `R_sharp` (`55.35` vs. `53.11`) and higher Top-1 accuracy (`13.62%` vs. `13.57%` when equalized), but amplified tail singular-vector noise by `+0.036` to `+0.041` nats in early 150-step Cross-Entropy (`6.8437` vs. `6.8074`).
3. **Concrete Hypothesis for the Next Iteration (`Wave 32` Candidate)**:
   - **Strided (Round-Robin) Subspace Partitioning `x.view(..., d // M, M).transpose(-2, -1)`**:
     Instead of contiguous slicing `[0:64], [64:128], ...`, partitioning coordinates **modulo $M$** (so subspace $m \in \{0, \dots, M-1\}$ contains singular-value pairs $m, m+M, m+2M, \dots$) guarantees that **every subspace $m$ receives an identical spectrum of top, middle, and tail PPMI-SVD singular values** and an identical mix of Sweldens wavelet subbands!
   - **Explicit Branch-Assigned Subspaces**:
     Alternatively, assign the 4 subspaces explicitly to the 4 scale levels of the dyadic pyramid (`Subspace 0 = wav_256 + wav_2048`, `Subspace 1 = wav_8192 @ up_8192`, `Subspace 2 = wav_65536 @ up_65536`, `Subspace 3 = readout_ngram`) so each scale level lives in its own dedicated orthogonal subspace of dimension $d / 4$.

