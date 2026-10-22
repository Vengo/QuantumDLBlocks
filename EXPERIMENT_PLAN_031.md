# Experiment Plan 031 — Solving `Ranks 6–1000` Readout Calibration & Diffuse Rank Collapse on the Fast `L=4, d=256` & `L=6, d=512` Local FineWeb-Edu Testbed (`<= 8.0 GB` VRAM): Bipolar Particle-Hole $k$-Boson Log-Fidelity, Beam-2 LayerNorm Whitening, Concept-Cluster Bath, & Full-Ambient Qubit Scaling

## 1. Executive Summary & Empirical Diagnosis of Wave 17 (`300M` Tok) and Plan 030 (`L=4, d=256`)

In **Experiment Plan 030** ([`EXPERIMENT_PLAN_030.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_030.md)), we evaluated our Plan 029 synthetic-walk winner (`FastQuantumBornAttentionSynthLM`) and its scaled `L=4, d=256, H=4` counterpart on genuine FineWeb-Edu tokens locally on the RTX 5060 Ti (`<= 8.0 GB` VRAM cap), while monitoring Wave 17 (`configs/wave17_fleet.json`) on two Google Colab G4 instances (`gpu_g4_w6_1` and `gpu_g4_w6_2`) out to `300M` tokens (`Step 4,575`).

Comparing the Wave 17 (`300M`-token) trajectory and the Plan 030 (`L=4, d=256`, `900`-step) bucketed calibration telemetry side-by-side exposed **four exact mathematical and optimization bottlenecks** that explain why Wave 17 tracked within `0.003–0.015 nats` of Wave 16 (`3.8565` vs `3.8536` Val CE at `300M` tokens, on pace for `~3.50` Val CE instead of `< 3.28` Val CE), and why we stopped `gpu_g4_w6_1` and `gpu_g4_w6_2` at `300M` tokens (`0.00 CU/hr`, preserving `116.83 CU`) to solve calibration locally first:

### 1.1 Bottleneck #1: The Asymmetric Logarithmic Barrier (`k * ln(1 + x)`) Acts as an Anti-Softcap on Negative Cosine Fluctuations ($x < 0$) and Freezes `ContextualBlochPurityGate` at `r_tilde = 0.95`
In `direct_chunked_two_beam_kboson_ce` (`src/qdlblocks/layers/direct_chunked_ce.py`, lines 786–795), the $k$-boson log-fidelity was parameterized as:
$$x_{t,v} = \tilde{r}_t c_{\text{sharp}}(t,v) + \lambda_{\text{diff}} f_{\text{diff}}(t,v), \qquad \tilde{r}_t = \operatorname{clamp}\!\left(\frac{\beta_t}{k_{\text{eff}}}, 0.0, 0.95\right), \qquad \mathcal{E}_{t,v} = k_{\text{eff}} \ln(1 + x_{t,v}) + b_v$$
Look at the Taylor expansion of $k_{\text{eff}} \ln(1 + x)$ on the positive ($x > 0$) vs. negative ($x < 0$) axes:
- **For positive overlap ($x > 0$, i.e. Top-1 and Top-5 hits)**:
  $$k_{\text{eff}} \ln(1 + x) = k_{\text{eff}}\left(x - \frac{1}{2}x^2 + \frac{1}{3}x^3 - \cdots\right)$$
  The quadratic term $-\frac{1}{2}x^2$ gently saturates large positive overlaps while giving orthogonal/near-zero synonyms ($x \in [0, 0.25]$) a steep slope $\frac{k_{\text{eff}}}{1 + x}$. This is why our Quantum Born models achieve extraordinary Top-1 accuracy (`42.26%` on Wave 16, `19.61%` at Step 900 on local `L=4, d=256`).
- **For negative overlap ($x < 0$, which occurs on roughly 50% of `Ranks 6–1000` tokens in high-dimensional Stiefel space)**:
  $$k_{\text{eff}} \ln(1 - |x|) = -k_{\text{eff}}\left(|x| + \frac{1}{2}|x|^2 + \frac{1}{3}|x|^3 + \cdots\right) \xrightarrow{|x| \to 1^-} -\infty$$
  Instead of saturating negative excursions like `ClassicalModdedGPT`'s odd-symmetric softcap $C \tanh(u / C) \in [-C, +C]$, $\ln(1 - |x|)$ **diverges toward $-\infty$**!
  - At $k_{\text{eff}} = 38.0$ and $x = +0.40$: $+38 \ln(1.40) = \mathbf{+12.79\text{ nats}}$.
  - At $k_{\text{eff}} = 38.0$ and $x = -0.40$: $+38 \ln(0.60) = \mathbf{-19.41\text{ nats}}$ (`6.62 nats` harsher than the positive side!).
  - At $k_{\text{eff}} = 38.0$ and $x = -0.60$: $+38 \ln(0.40) = \mathbf{-34.82\text{ nats}}$!
  Because `~40%` of ground-truth tokens in FineWeb-Edu lie outside Top-5 (`Ranks 6–1000`), whenever a plausible mid-rank ground-truth token $y_t$ has a random negative cosine fluctuation $x_{t, y_t} \in [-0.45, -0.15]$, the $-\frac{k}{2}|x|^2 - \frac{k}{3}|x|^3$ log-barrier penalty inflates `NLL(6–50)` and `NLL(51–1000)`.
- **Why `ContextualBlochPurityGate` Lost 100% of Its Gradient After `125M` Tokens on Colab**:
  Because $\ln(1 + x)$ has a singularity at $x = -1$, `direct_chunked_two_beam_kboson_ce` had to clamp $\tilde{r}_t = \operatorname{clamp}(\beta_t / k_{\text{eff}}, 0.0, 0.95)$ with `r_mask = (r_raw > 0.0) & (r_raw < 0.95)`. On Colab Wave 16 and Wave 17, $\beta_t$ climbed from `28.57` to **`44.07`** by `150M` tokens (`r_raw = 44.07 / 38.0 = 1.16 > 0.95`), permanently zeroing `r_mask = 0` and freezing `ContextualBlochPurityGate` for the remaining `750M` tokens!

### 1.2 Bottleneck #2: Diffuse Beam Common-Mode DC Collapse (`diffuse_eff_rank: 15.47 -> 5.80`) Due to Missing Beam-2 `LayerNorm`
In `Attempt 3` (`w17_8gb_fast_synth_scaled_L4_d256_900steps`, `EXPERIMENT_PLAN_030.md`), the Shannon effective rank of the sharp beam stayed high (`sharp_eff_rank = 190.09 / 256`), yet the effective rank of the rank-$16$ diffuse Kraus bath collapsed monotonically from **`15.47` at Step 0 $\to$ `10.90` at Step 100 $\to$ `7.15` at Step 200 $\to$ `5.80` at Step 400**!
Inspecting `OvercompleteTwoBeamInterference.forward` (`src/qdlblocks/layers/overcomplete_two_beam.py`, lines 223–260) reveals the exact cause:
- Beam 1 (`out_1 = F.linear(interf, self.W_out)`) passes through `self.post_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)`, which centers and whitens the embedding channels before stereographic rotation / L2 normalization.
- Beam 2 (`b2 = (z2 * z2) * F.silu(z1)`, `out_2 = F.linear(b2, u2.t())`, `raw_b2 = out_2 - 0.25 * out_1`) had **no `LayerNorm` centering**! Because $z_2^2 \ge 0$ is strictly non-negative, `b2` develops a large positive DC mean across tokens, pulling `raw_b2` for every token in the batch toward a rank-1 common-mode direction and collapsing `diffuse_eff_rank` from `15.47` down to `5.80`.

### 1.3 Bottleneck #3: Unused `vocab_higher_affinity` Semantic Cluster Structure in the Diffuse Kraus Bath
In `Hierarchical16QubitPOVMReadout`, every token $v$ already possesses a learned semantic cluster distribution $\text{aff}_v \in \mathbb{R}^{M_{\text{hier}}}$ (`vocab_higher_affinity: (V, M_hier)`) over the $M_{\text{hier}} = 128$ or $256$ higher-level quantum concept states. However, `m_diff_r` only projected `m_sharp_norm` via `kraus_u_diff` (`(d, r)`), ignoring direct concept-cluster overlap. Coupling normalized `vocab_higher_affinity` directly into `m_diff_r` via an orthonormal bridge `hier_diffuse_proj: (M_hier, r)` allows all tokens in the same semantic concept cluster (`Ranks 2–50`) to share coherent diffuse bath amplitude!

### 1.4 Bottleneck #4: Under-Scaled Learning Rate (`0.003` vs `0.015`) & Non-Zero Weight Decay on Sparse Qubit Lookup Tables, Plus Half-Zero-Padded `L=6, d=512` Qubits
1. In `examples/40_verify_plan030_fast_synth_winner_on_fineweb_8gb.py`, the sparse per-token lookup tables `unigram_qubits` (`(50304, 16, A)`), `higher_states`, and `value_embeds` were placed in `body_group` with `lr = 0.003` and `weight_decay = 0.01`. Because `unigram_qubits` is row-normalized per sub-register on $(S^{A-1})^{16}$ and only receives gradients for tokens present in the current batch, `weight_decay = 0.01` with a tiny `lr = 0.003` starves mid-rank and rare tokens (`Ranks 51–1000`) of fast embedding alignment. Routing `unigram_qubits`, `higher_states`, and `value_embeds` to a dedicated **`weight_decay = 0.0` group with `lr = 0.015`** (`5x` higher) accelerates token subspace alignment across the vocabulary.
2. In `L=6, d=512` (`Attempt 4`), we previously used `16 × 16 = 256` qubits (`use_ortho_decouple=True`), leaving `50%` of the `d=512` ambient space zero-padded at $\psi_0$. Now that Plan 030's Directed Predecessor Gram-Schmidt Self-Decoupled PPMI-SVD (`decouple_self_in_full_ambient=True`) eliminates self-echo in full-ambient space (`1.19e-7`), upgrading `L=6, d=512` to **full-ambient `16 × 32 = 512` qubits (`use_ortho_decouple=False`)** activates 100% of the `d=512` Hilbert space from Step 0!

---

## 2. Mathematical Formulation of the Plan 031 Solutions

### 2.1 Bipolar Particle-Hole Symmetric $k$-Boson Log-Fidelity (`bipolar_hole_weight` $\alpha_{\text{hole}} \in [0, 1]$)
In `direct_chunked_two_beam_kboson_ce` and `compute_two_beam_kboson_logits` (`src/qdlblocks/layers/direct_chunked_ce.py`), let:
$$x_{t,v} = \tilde{r}_t c_{\text{sharp}}(t,v) + \lambda_{\text{diff}}(t) f_{\text{diff}}(t,v)$$
When `bipolar_hole_weight` $\alpha_{\text{hole}} > 0$ (default `0.85` when `--bipolar-kboson` is enabled; `0.0` for legacy asymmetric $k$-boson):
1. **Unclamped Purity Ratio ($\tilde{r}_t \in [0, 1.35]$)**:
   $$\tilde{r}_t = \operatorname{clamp}\!\left(\frac{\beta_t}{k_{\text{eff}}}, \, 0.0, \, 1.35\right)$$
   Because the bipolar log-fidelity has no singularity at $x = -1$, $\beta_t$ can vary freely up to $1.35 k_{\text{eff}} \approx 51.3$ without hitting the `0.95` clamp ceiling!
2. **Particle-Hole Symmetric Log-Fidelity Energy**:
   Decompose $x_{t,v} = x^+_{t,v} - x^-_{t,v}$ into its excitatory particle overlap $x^+_{t,v} = \max(x_{t,v}, 0) \ge 0$ and its inhibitory hole (antipodal) overlap $x^-_{t,v} = \max(-x_{t,v}, 0) \ge 0$. The $k_{\text{eff}}$-boson particle-hole log-fidelity is:
   $$\mathcal{E}^{\text{bipolar-}k}_{t,v} = k_{\text{eff}} \Big( \ln(1 + x^+_{t,v}) - \alpha_{\text{hole}} \ln(1 + x^-_{t,v}) \Big) + b_v = \begin{cases} +k_{\text{eff}} \ln(1 + x_{t,v}) + b_v, & x_{t,v} \ge 0 \\[6pt] -\alpha_{\text{hole}} k_{\text{eff}} \ln(1 + |x_{t,v}|) + b_v, & x_{t,v} < 0 \end{cases}$$
3. **Exact Analytical Backward Pass**:
   Let $\ell_{t,v} = \ln(1 + x^+_{t,v}) - \alpha_{\text{hole}} \ln(1 + x^-_{t,v})$ and $g_{t,v} = \frac{\partial \mathcal{L}}{\partial \mathcal{E}_{t,v}} = \frac{1}{N_{\text{valid}}}(P_{t,v} - \mathbb{I}[v = y_t])$. Then:
   $$\frac{\partial \mathcal{E}_{t,v}}{\partial k_{\text{eff}}} = \ell_{t,v} - \frac{\partial \mathcal{E}_{t,v}}{\partial x_{t,v}} c_{\text{sharp}}(t,v) \frac{\beta_t}{k_{\text{eff}}^2} \mathbb{I}\!\left[0 < \frac{\beta_t}{k_{\text{eff}}} < 1.35\right]$$
   $$\frac{\partial \mathcal{E}_{t,v}}{\partial x_{t,v}} = \frac{k_{\text{eff}}}{1 + |x_{t,v}|} \cdot \begin{cases} 1, & x_{t,v} \ge 0 \\ \alpha_{\text{hole}}, & x_{t,v} < 0 \end{cases}$$
   Notice that $1 + |x_{t,v}| \ge 1$ everywhere on $\mathbb{R}$: the denominator is **bounded away from zero for all $x_{t,v} \in (-\infty, +\infty)$**, completely eliminating numerical blowups and negative log-barrier distortion!

### 2.2 Beam-2 LayerNorm & Diffuse Bath Whitening + Concept-Cluster Bridge
1. **Beam-2 LayerNorm (`OvercompleteTwoBeamInterference.forward`)**:
   In `src/qdlblocks/layers/overcomplete_two_beam.py`, before applying stereographic rotation or L2 normalization to `raw_b2 = out_2 - 0.25 * out_1`, we apply parameter-free zero-mean unit-variance channel normalization `raw_b2_normed = F.layer_norm(raw_b2.float(), (self.embed_dim,), eps=1e-6).to(psi.dtype)` whenever `self.use_stereographic_lift or self.use_cosine_phase_beam_gate` is enabled, removing the non-negative $z_2^2$ common-mode DC bias.
2. **Diffuse Subspace Mode Whitening & Concept-Cluster Bridge (`Hierarchical16QubitPOVMReadout`)**:
   In `src/qdlblocks/layers/hierarchical_qubit.py`:
   - Add `self.hier_diffuse_proj: nn.Parameter | None = nn.Parameter(torch.empty(self.num_higher_states, self.diffuse_rank, dtype=torch.float32))` (initialized with orthonormal columns) and `self.hier_diffuse_gain = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))` when `self.num_higher_states > 0`.
   - In `get_two_beam_kboson_factors`:
     - Couple the normalized concept affinity `aff_norm = l2_normalize(self.vocab_higher_affinity.to(m_sharp_norm.dtype), dim=-1, eps=self.eps)` into `m_diff_r`:
       $$m_{\text{diff}, r} = \left(\frac{d}{r}\right)^{1/4} \Big( M_{\text{sharp}} U_{\text{diff}} + 0.35 \tanh(g_{\text{hier,diff}}) \, (\text{aff}_{\text{norm}} P_{\text{hier}}) \Big)$$
     - When `self.use_bipolar_kboson` is True, apply zero-mean batch-token centering and RMS mode equalization to the raw projection `p_diff = psi_beam2_norm @ self.kraus_v_diff` across tokens so all $r$ diffuse Kraus modes maintain equal variance (`diffuse_eff_rank >= 12.0`).

---

## 3. Observable Hypotheses & Quantitative Predictions

### Hypothesis H1 — Bipolar Particle-Hole $k$-Boson Log-Fidelity (`bipolar_hole_weight=0.85`) Eliminates Negative-Overlap Log-Barrier Distortion and Preserves Exact Analytical Gradients (`< 1e-5`)
- **Claim**:
  1. For any positive overlap $x_{t,v} \ge 0$, `direct_chunked_two_beam_kboson_ce(..., bipolar_hole_weight=0.85)` produces the **exact same** log-fidelity $k \ln(1 + x_{t,v})$ as legacy `bipolar_hole_weight=0.0`.
  2. For negative overlaps $x_{t,v} \in [-0.80, -0.20]$ (where legacy $k \ln(1 + x)$ suffers a severe concave downward penalty up to `-61 nats` at $k=38, x=-0.80$), `bipolar_hole_weight=0.85` bounds the negative energy smoothly (`-19.0 nats` at $x=-0.80$, reducing negative-tail distortion by **`> 3.0x`**) and remains finite and smooth even for $x_{t,v} < -1.0$.
  3. Analytical backward gradients of `direct_chunked_two_beam_kboson_ce(..., bipolar_hole_weight=0.85)` with respect to all 8 inputs (`psi_sharp`, `psi_diff`, `m_sharp`, `m_diff`, `beta_t`, `k_copies`, `lambda_diff`, `log_prior`) match PyTorch autograd reference (`compute_two_beam_kboson_logits` + `F.cross_entropy` in `float64`) to `< 1e-5` max relative/absolute error.

### Hypothesis H2 — Beam-2 LayerNorm + Bipolar $k$-Boson + Concept-Cluster Bath + Fast Qubit LR Improves `L=4, d=256` on FineWeb-Edu at Both Step 300 (`2.46M` Tok) and Step 900 (`7.37M` Tok) While Keeping `diffuse_eff_rank >= 12.0`
- **Claim**:
  On genuine FineWeb-Edu (`B_micro = 8, grad_accum = 2, T = 512`, `8,192` tok/step, evaluated on held-out `val_slice.bin` at $T_{\text{val}} = 512$, `<= 8.0 GB` VRAM cap):
  1. **Diffuse Rank Preservation**: Beam-2 LayerNorm + diffuse mode whitening prevents the Plan 030 diffuse rank collapse (`5.80–6.77`), maintaining **`diffuse_eff_rank >= 12.0`** (out of `16.0`) at both Step 300 and Step 900.
  2. **Step-300 (`2.46M` Tokens) Gain on `L=4, d=256`**:
     - `Arm A` (`w18_8gb_L4_d256_bipolar_kboson_300steps`, controlled readout + Beam-2 whitening upgrade at identical LRs) beats Plan 030 `Attempt 3` at Step 300 (`6.1391` Val CE), and
     - `Arm B` (`w18_8gb_L4_d256_bipolar_kboson_fast_qubit_lr_900steps`) reaches **`<= 5.9500` Val CE at Step 300 (`2.46M` tokens)** — breaking the `6.00` Val CE barrier in just `300` steps (`~215s`) with a 4-layer `d=256` model!
  3. **Step-900 (`7.37M` Tokens) Gain on `L=4, d=256`**:
     - At Step 900 (`7.37M` tokens), `Arm B` improves over Plan 030 `Attempt 3` (`5.6234` Val CE, `19.61%` Top-1, `36.52%` Top-5, `NLL(51-1k) = 7.7094`) by **`>= 0.12 nats`**, reaching **`<= 5.5000` Val CE** with lower `NLL(6-50)` and `NLL(51-1000)` and higher Top-1 and Top-5 accuracy!

### Hypothesis H3 — Full-Ambient `16×32` Qubits (`use_ortho_decouple=False`) + Bipolar $k$-Boson on `L=6, d=512` Shatters the 300-Step Local Record (`6.0823 -> <= 5.9200` Val CE)
- **Claim**:
  Upgrading the `L=6, d=512, H=8` Unified `QuantumBornGPT` (`Arm C`, `w18_8gb_L6_d512_full_ambient_16x32_bipolar_kboson_300steps`) to **Full-Ambient `16×32` Qubits (`use_ortho_decouple=False, decouple_self_in_full_ambient=True`) + Bipolar Particle-Hole $k$-Boson (`bipolar_hole_weight=0.85`) + Beam-2 LayerNorm + Concept-Cluster Diffuse Bath + Fast Qubit Embedding LR** breaks the `6.00` Val CE barrier in `300` steps (`2.46M` tokens), achieving **`<= 5.9200` Val CE** (**`>= 0.16 nats` better** than Plan 030's `6.0823` record!) while staying strictly within `Peak VRAM <= 8.0 GB`.

---

## 4. Operational Tooling & Script Opportunities Gate (Section 5 Compliance)

### 4.1 Existing Scripts Used
1. **Model Inspection & Smoke Verification (`scripts/inspect_model.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py --num-layers 4 --embed-dim 256 --num-heads 4 --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 128 --two-beam-kboson-readout --bipolar-kboson --bipolar-hole-weight 0.85 --stereographic-lift --cosine-phase-beam-gate --quadratic-diffuse-fidelity 0.50 --smoke-steps 2
   ```
2. **Zero-Downtime Live Monitoring Daemon (`scripts/monitor_colab_runs.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once
   ```

### 4.2 Build-Before-Launch Scripts & Modules
- **`examples/41_verify_plan031_bipolar_kboson_and_L4_d256_calibration.py`**:
  Deterministic local RTX 5060 Ti (`<= 8.0 GB` VRAM cap via `torch.cuda.set_per_process_memory_fraction(0.5, device=0)`) verification and benchmark suite that executes `H1`, `H2`, and `H3`, reuses the completed Plan 030 baselines from `.cache/hypothesis_experiments_030.json`, trains `Arm A` (`L=4, d=256`, `300` steps), `Arm B` (`L=4, d=256`, `900` steps), and `Arm C` (`L=6, d=512` Full-Ambient `16×32` Bipolar $k$-Boson, `300` steps), logs all 5-bucket Target-Rank Cross-Entropy Decompositions, saves `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_031.json`, `.cache/live_logs/w18_8gb_*.log`, `.cache/model_cards/w18_8gb_*.json`, and syncs `.cache/benchmarks.db` + `POST http://127.0.0.1:8765/api/reload`.

### 4.3 Nice-to-Have Scripts (Logged to `scripts/BACKLOG.md` & Surfaced to User)
1. **`scripts/bipolar_vs_concave_kboson_curve_plotter.py`**: A standalone CLI diagnostic that plots the exact log-fidelity curve $\mathcal{E}(x)$, slope $\partial \mathcal{E}/\partial x$, and empirical per-bucket cosine similarity distribution $p(x_{t, y_t} \mid \text{Rank Bucket})$ comparing legacy concave $k \ln(1+x)$ vs. Bipolar Particle-Hole $k$-Boson across validation checkpoints.

---

## 5. Planned Code Changes

### 5.1 `src/qdlblocks/layers/direct_chunked_ce.py`
- Extend `_DirectChunkedTwoBeamKBosonCEFn.forward`, `direct_chunked_two_beam_kboson_ce`, and `compute_two_beam_kboson_logits` with `bipolar_hole_weight: float = 0.0`:
  - When `bipolar_hole_weight <= 0.0`, preserve 100% legacy behavior (`r_tilde = clamp(beta / k, 0, 0.95)`, `z = clamp(1 + r_tilde * c_sharp + lam * f_diff, min=1e-4)`, `logits = k * ln(z) + b_v`) so all Plans 026–030 unit tests pass untouched.
  - When `bipolar_hole_weight > 0.0` (e.g. `0.85`):
    - Set `r_max_cap = 1.35`, `r_raw = beta_c / k_val`, `r_mask = ((r_raw > 0.0) & (r_raw < r_max_cap)).to(acc_dtype)`, `r_tilde = torch.clamp(r_raw, min=0.0, max=r_max_cap)`.
    - Compute $x = \tilde{r}_t c_{\text{sharp}} + \lambda_{\text{diff}} f_{\text{diff}}$, $x_+ = \operatorname{clamp\_min}(x, 0.0)$, $x_- = \operatorname{clamp\_min}(-x, 0.0)$.
    - Compute $\ell = \ln(1.0 + x_+) - \alpha_{\text{hole}} \ln(1.0 + x_-)$ and $\text{logits} = k_{\text{val}} \ell + b_v$.
    - Compute exact analytical backward derivative:
      $$q_{t,v} = g_{t,v} \cdot \frac{k_{\text{val}}}{1.0 + |x_{t,v}|} \cdot \begin{cases} 1.0, & x_{t,v} \ge 0 \\ \alpha_{\text{hole}}, & x_{t,v} < 0 \end{cases}$$

### 5.2 `src/qdlblocks/layers/overcomplete_two_beam.py`
- In `OvercompleteTwoBeamInterference.forward`:
  - When `self.use_stereographic_lift or self.use_cosine_phase_beam_gate` is True, center and whiten `raw_b2 = out_2 - 0.25 * out_1` via `raw_b2_normed = F.layer_norm(raw_b2.float(), (self.embed_dim,), eps=1e-6).to(psi.dtype)` before applying `apply_dimension_preserving_stereographic_rotation` (preserving `radial_norm_override=raw_b2.float().norm(p=2, dim=-1, keepdim=True)` so the pre-normalization radial norm of `raw_b2` is still encoded into the polar latitude!).

### 5.3 `src/qdlblocks/layers/hierarchical_qubit.py`, `src/qdlblocks/models/born_gpt.py`, `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`, `scripts/inspect_model.py`, `src/qdlblocks/benchmarks/model_card.py`, `src/qdlblocks/benchmarks/db.py`
- In `Hierarchical16QubitPOVMReadout`:
  - Add `use_bipolar_kboson: bool = False`, `bipolar_hole_weight: float = 0.0`, `hier_diffuse_proj: nn.Parameter | None`, and `hier_diffuse_gain: nn.Parameter | None`.
  - When `self.use_bipolar_kboson` is True:
    - Couple `vocab_higher_affinity` into `m_diff_r` via `hier_diffuse_proj` (`(M_hier, r)`).
    - Apply batch-mean centering and per-mode RMS stabilization to `psi_diff_r` so all $r$ diffuse Kraus modes stay active (`diffuse_eff_rank >= 12.0`).
- Wire `use_bipolar_kboson: bool = False` and `bipolar_hole_weight: float = 0.0` (default `0.85` when `--bipolar-kboson` is passed) through `QuantumBornGPT`, `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` (`--bipolar-kboson`, `--bipolar-hole-weight`, `--directed-self-decoupled-ppmi`), `scripts/inspect_model.py`, `src/qdlblocks/benchmarks/model_card.py`, and `src/qdlblocks/benchmarks/db.py` (ingesting `hypothesis_experiments_031.json`).

### 5.4 Verification & Unit Tests
- **`tests/test_hypothesis_031_bipolar_kboson_and_calibration.py`**:
  1. `test_h1_bipolar_kboson_positive_equivalence_negative_saturation_and_autograd`: Verifies exact positive-branch equivalence ($x \ge 0$), $>3\times$ reduction in negative-overlap barrier penalty ($x \in [-0.8, -0.2]$), finite behavior for $x < -1.0$, and $< 10^{-5}$ analytical backward gradient parity against `float64` PyTorch autograd across all 8 inputs.
  2. `test_h2_beam2_layernorm_and_concept_cluster_bath_rank_preservation`: Verifies `OvercompleteTwoBeamInterference` Beam-2 LayerNorm and `Hierarchical16QubitPOVMReadout` concept-cluster diffuse bath preserve `diffuse_eff_rank >= 12.0` on non-negative DC-shifted activations.
  3. `test_plan031_verification_artifact_schema_if_present`: Verifies `.cache/hypothesis_experiments_031.json` confirms `H1=PASS`, `H2=PASS`, and `H3=PASS`.
- **`examples/41_verify_plan031_bipolar_kboson_and_L4_d256_calibration.py`**:
  Executes `H1`, `H2`, and `H3` on the local RTX 5060 Ti under `<= 8.0 GB` VRAM and updates Section 6 of `EXPERIMENT_PLAN_031.md`.

---

## 6. Empirical Results & Root-Cause Diagnosis (Falsification of `--bipolar-kboson` & Hard Diffuse Whitening)

Running `examples/41_verify_plan031_bipolar_kboson_and_L4_d256_calibration.py` on our local RTX 5060 Ti (`<= 8.0 GB` VRAM) produced a clean, decisive negative result (`H1=PASS` on analytical autograd, `H2=FAIL` on FineWeb-Edu validation cross-entropy, terminated early after Step 300 of `L=4, d=256` to conserve GPU time):

| Model / Run Key | Backbone (`L, d, H`) | Step 0 Val CE (`ECE`) | Step 300 Val CE (`2.46M` Tok) | Step 300 Top-1 / Top-5 | `Gap(T5-T1)` | `R_diff` (`diffuse_eff_rank`) | Delta vs. Plan 030 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plan 030 Baseline**: `w17_8gb_fast_synth_ppmi_ngram_L2_d256` | `L=2, d=256, H=4` | `12.9310` (`0.2841`) | **`6.1960`** | **`16.16%` / `31.32%`** | **`15.15%`** | `9.16` *(Natural Zipfian)* | Baseline |
| **Plan 031 Whitened Bipolar**: `w18_8gb_L2_d256_bipolar_kboson_whitened_300steps` | `L=2, d=256, H=4` | `13.6701` (`0.3180`) | `6.3250` | `15.62%` / `30.60%` | `14.98%` | `16.00` *(Forced Flat)* | **`+0.1290 nats` (Worse)** |
| **Plan 030 Baseline**: `w17_8gb_fast_synth_scaled_L4_d256_900steps` | `L=4, d=256, H=4` | `12.5196` (`0.2637`) | **`6.1391`** | **`16.76%` / `32.23%`** | **`15.47%`** | `6.01` *(Natural Zipfian)* | Baseline |
| **Plan 031 Calibrated Bipolar**: `w18_8gb_L4_d256_bipolar_kboson_calibrated_900steps` | `L=4, d=256, H=4` | `13.1323` (`0.2986`) | `6.4489` | `15.22%` / `29.31%` | `14.09%` | `16.00` *(Forced Flat)* | **`+0.3098 nats` (Worse)** |

### Why Plan 031 Regressed by `+0.13` to `+0.31 Nats`
1. **The Negative-Hemisphere Softmax Denominator Explosion (`--bipolar-kboson`)**:
   - In a `50,257`-token vocabulary, roughly `25,000–49,000` background tokens on any step have negative inner product ($x < 0$).
   - Under Plan 030's concave log-fidelity $k_{\text{eff}} \ln(1 + x)$, a negative background token at $x = -0.35$ (\(k_{\text{eff}} \approx 36\)) sits at $36 \ln(0.65) = \mathbf{-15.51\text{ nats}}$, suppressing the `49,000` wrong tokens in the Softmax partition function $Z = \sum_v e^{\mathcal{E}_v}$.
   - Replacing $\ln(1 - |x|)$ on $x < 0$ with the convex $-0.85 \ln(1 + |x|)$ lifted that same background token at $x = -0.35$ to $-0.85 \times 36 \ln(1.35) = \mathbf{-9.18\text{ nats}}$ — making each negative-hemisphere background token **$e^{6.33} \approx 560\times$ larger** in the Softmax denominator $Z$ and stealing massive probability mass from `Rank 1` and `Ranks 2–5`!
2. **The `R_diff = 16.00` Cholesky Whitening Trap**:
   - English syntax in the 16-dimensional PPMI-initialized Kraus subspace naturally follows a **Zipfian singular value spectrum (`R_diff ~ 6 to 11`)**, where modes 1–6 capture high-variance syntactic manifolds and modes 10–16 are low-variance residual directions.
   - Forcing all 16 modes to have identical variance (`R_diff = 16.00` at Steps `0, 100, 200, 300`) crushed the top-6 syntactic modes down while amplifying noisy modes 10–16 by `5x–10x`.
3. **Action Taken**:
   - Reverted Beam-2 LayerNorm in `OvercompleteTwoBeamInterference` and kept `use_bipolar_kboson=False` by default so the codebase uses the **exact Plan 030 mathematical kernel**, and promoted the clean isolate tests (`Full-Ambient 16×32 = 512 Qubits` on `L=6, d=512` and `Gentle 2x Qubit Embedding LR lr=0.006, wd=0.0`) to **`EXPERIMENT_PLAN_032.md`**.

