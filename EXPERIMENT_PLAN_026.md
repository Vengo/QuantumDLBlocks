# EXPERIMENT PLAN 026: 100% Quantum Two-Beam Mixed-State Density Readout ($\text{Tr}(\rho_t^{\otimes k} E_v^{\otimes k})$), Non-Isometric Kraus Semantic Bath ($K_{\text{diff}} = U_{\text{diff}}^\top$), & $k$-Copy Bloch Fidelity

**Status:** Drafted for Independent Blind Review  
**Date:** 2026-10-20  
**Hardware Target:** Local NVIDIA GeForce RTX 5060 Ti (`16,311 MiB` total VRAM, strictly capped to **$\le 8.0\text{ GB}$ (`50%` VRAM)** via `torch.cuda.set_per_process_memory_fraction(0.5)`)  
**Predecessor Plans:** `EXPERIMENT_PLAN_023.md` (`50,304 × 16 × 32` Unigram Qubit Table & `256`-State Hierarchy), `EXPERIMENT_PLAN_024.md` (Last-Layer Softmax Baseline), `EXPERIMENT_PLAN_025.md` (5-Bucket Target-Rank Cross-Entropy Decomposition & 8 GB Classical vs. Quantum Dynamics)

---

## 1. Standalone Context, Root-Cause Diagnosis, & Why Pure Quantum Beats Ad-Hoc Softmax

### 1.1 Project Overview & The Wave 15 Top-1 vs. Cross-Entropy Paradox
In **QuantumDLBlocks**, we are developing **`QuantumBornGPT`**—a quantum-inspired language model operating on the unit hypersphere $S^{d-1}$ with Stiefel-orthogonalized operators ($W^\top W = I_d$), a `50,304 × 16 × 32` multi-register unigram qubit amplitude table, and `256` higher-level quantum states interacting via a $16 \times 256 \times 256$ unitary operator—benchmarked against **`ClassicalModdedGPT`** (`KellerJordan/modded-nanogpt` speedrun architecture) on `900M` FineWeb-Edu tokens and evaluated on the `10,485,760`-token `fineweb_val_000000.bin` validation split.

In Wave 15 (`w15_900m_qborn_16x32_qubit_256hier_ngram_L8` on `gpu_a100_w6_1`), our pure quantum architecture achieved a remarkable milestone at `724.6M` tokens (`Step 11,056 / 13,725`):
- **Validation Top-1 Accuracy:** **`40.87%`** (surpassing the `~38.2%` Top-1 of a finished `3.28` CE classical `modded-nanogpt` speedrun and our own `3.3362` CE 16-layer classical record's `38.80%` Top-1 by **+2.07 percentage points**!)
- **Validation Cross-Entropy:** **`3.6340`** (`Top-5 = 58.72%`, `Top-5 - Top-1` gap = **`17.85%`** vs **`22.13%`** in `ClassicalModdedGPT`).

### 1.2 Three Mathematical Culprits Identified from Code & Telemetry Inspection
Why does `QuantumBornGPT` dominate `Top-1` token prediction by `+2% to +5%` across the entire training trajectory, yet suffer a `~0.25–0.30 nat` Cross-Entropy penalty on `Ranks 2–500`?

1. **Culprit 1 — Premature Collapse of `OvercompleteTwoBeamInterference` Before Readout**:
   - Inside each `QuantumBornBlock`, `OvercompleteTwoBeamInterference` (`src/qdlblocks/layers/early_epoch_quantum.py`) expands $|\psi\rangle \in S^{d-1}$ into two overcomplete beams $u_1, u_2 \in \mathbb{R}^{4d}$, applies phase interference $g = \cos(\omega(u_1 - u_2) + \phi)$, and **immediately collapses both beams back into a single pure state vector $|\psi_{\text{out}}\rangle \in S^{d-1}$ at the end of every block**:
     $$\psi_{\text{out}} = \text{L2Normalize}\Big(W_{\text{down},1}(u_1 \odot g) + W_{\text{down},2}(u_2 \odot (1 - g))\Big) \in S^{d-1}$$
   - Consequently, at the final readout layer, the model has only a **single rank-1 pure state $|\psi_L\rangle \in S^{d-1}$** living in a high-rank Stiefel-orthogonalized space ($R_{\text{eff}}(\rho_{\text{seed}}) \approx 26\text{–}32$, `R_sigma = 239.9–270.4`).
   - By contrast, `ClassicalModdedGPT` collapses $R_{\text{eff}}(\rho_{\text{seed}})$ down to `~5`, using ~5 broad collinear modes to broadcast probability mass across 30–100 syntactically plausible replacement tokens (`Ranks 2–50`). A single high-rank orthogonal ray $|\psi_L\rangle$ is a laser-sharp `Top-1` sniper (`40.87%` Top-1), but because plausible synonyms in a Stiefel/PPMI-SVD space are nearly orthogonal ($\langle M_{\text{synonym}} | \psi_L \rangle \approx 0.02\text{–}0.05$), a single pure state cannot simultaneously illuminate the cloud of valid replacements.

2. **Culprit 2 — `ContextualBlochPurityGate` Runaway Inverse Temperature ($\beta \in [71, 122]$)**:
   - In `ContextualBlochPurityGate` (`src/qdlblocks/layers/early_epoch_quantum.py` line 264), `max_beta` defaults to `150.0`, allowing $\beta$ to climb to `71.6–122.1` in Wave 15.
   - When $\beta = 100$, a Top-1 token with cosine similarity $0.22$ and a plausible Rank-15 synonym with cosine similarity $0.05$ are separated by $\Delta z = 100 \times (0.22 - 0.05) = 17.0\text{ nats}$ ($e^{-17} \approx 4 \times 10^{-8}$), severely penalizing Cross-Entropy whenever the ground-truth token lands in `Ranks 2–500`.

3. **Culprit 3 — Vanishing Synonym Gradients Under Quadratic Born $|\langle M_v | \psi \rangle|^2$ & The Low-Rank Subspace Normalization Trap**:
   - Under naive quadratic Born $P(v) \propto |\langle M_v | \psi \rangle|^2$, when a plausible synonym is orthogonal ($c_v = \langle M_v | \psi \rangle \approx 0$), the numerator derivative $2 c_v \to 0$ vanishes in the partition function, while the log-barrier derivative $-2/c_v$ explodes on rare misses.
   - Furthermore, if we project a second diffuse beam $|\psi_2\rangle$ into a low-rank subspace $\mathbb{R}^r$ ($r = 16$) and **mistakenly L2-normalize both $U_{\text{diff}}^\top M_v$ and $U_{\text{diff}}^\top \psi_2$ inside $\mathbb{R}^r$**, any two random unrelated tokens in $\mathbb{R}^{16}$ have expected squared overlap $1/r = 1/16 = 0.0625$, dumping $>99\%$ of probability mass into the 50,000 background tokens!

---

## 2. The 100% Quantum Solution: Two-Beam Mixed-State $k$-Boson Bloch Fidelity Readout ($\text{Tr}(\rho_t^{\otimes k} E_v^{\otimes k})$)

Instead of abandoning quantum geometry for a classical Softmax head, we solve all three culprits directly within quantum measurement theory:

### 2.1 Extracting Both Uncollapsed Final-Layer Quantum Beams ($|\psi_{t,1}\rangle, |\psi_{t,2}\rangle \in S^{d-1}$)
In the final transformer block ($l = L - 1$), `OvercompleteTwoBeamInterference` produces two complementary interferometric wavepackets before final collapse:
- **Beam 1 (`Sharp High-Rank State`, $|\psi_{t,1}\rangle \in S^{d-1}$, $R_{\text{eff}} \approx 30$)**:
  The primary constructive-interference state $|\psi_{t,1}\rangle = \psi_{t,L} \in S^{d-1}$ that routes exact `Top-1` predictions.
- **Beam 2 (`Complementary Interferometric State`, $|\psi_{t,2}\rangle \in S^{d-1}$)**:
  The complementary destructive-interference wavepacket:
  $$|\psi_{t,2}\rangle = \text{L2Normalize}\Big(\psi_{\text{attn}, L-1} + \gamma_{\text{mlp}, L-1} \, \text{L2Normalize}\big(W_{\text{down}, 2}(u_2 \odot (1 - g_{\text{interf}}))\big)\Big) \in S^{d-1}$$

### 2.2 Non-Isometric Kraus Semantic Bath Operator ($K_{\text{diff}} = U_{\text{diff}}^\top \in \mathbb{R}^{r \times d}$, $r = 16$)
To broadcast probability amplitude across clusters of syntactically/semantically interchangeable tokens (`Ranks 2–50`) **without falling into the $1/r$ low-rank normalization trap or causing a Step-0 variance shock**:
- We introduce a learnable **Kraus Measurement Bath** $U_{\text{diff}} \in \mathbb{R}^{d \times r}$ and dual projector $V_{\text{diff}} \in \mathbb{R}^{d \times r}$ ($r = 16$).
- **Top-$r$ POVM Spectral Initialization ($U_{\text{diff}} = V_{\text{diff}} = V_{\text{SVD}, :r}(M_{\text{sharp}})$)**:
  - Since the vocabulary POVM matrix $M_{\text{sharp}} \in \mathbb{R}^{V \times d}$ is already initialized on $S^{d-1}$ from the 9-shard PPMI-SVD collocation basis, we initialize $U_{\text{diff}} = V_{\text{diff}} \in \mathbb{R}^{d \times r}$ directly as the **top-$r$ right singular vectors of $M_{\text{sharp}}$** (`_, _, Vh = torch.linalg.svd(M_sharp.float(), full_matrices=False); U_diff = V_diff = Vh[:r].T`) with a gentle initial mixing weight $\lambda_{\text{diff}, 0} \approx 0.18$.
  - This guarantees that at Step 0, `TwoBeamMixedStateReadout` starts seamlessly aligned with our proven `6.1475` POVM baseline while $U_{\text{diff}}$ and $V_{\text{diff}}$ immediately span the top-$r$ principal semantic co-occurrence axes of the vocabulary!
- **Exact $(d / r)^{1/4}$ Per-Side ($\sqrt{d / r}$ Bilinear) Variance Matching Without Per-Token $\mathbb{R}^r$ L2-Normalization**:
  - For two random unit vectors $M_v, \psi_{t,2} \in S^{d-1}$ projected onto an $r$-dimensional orthonormal subspace $U_{\text{diff}}, V_{\text{diff}}$, each projected vector $u_v = U_{\text{diff}}^\top M_v$ and $v_t = V_{\text{diff}}^\top \psi_{t,2}$ has expected norm $\sqrt{r / d}$, and their unscaled inner product $\langle u_v \mid v_t \rangle$ has standard deviation $\frac{\sqrt{r}}{d}$.
  - Meanwhile, the sharp channel $c_{\text{sharp}}(t, v) = \langle M_v \mid \psi_{t,1} \rangle$ in $\mathbb{R}^d$ has standard deviation $\frac{1}{\sqrt{d}} = \frac{1}{\sqrt{512}} \approx 0.0442$ (whereas scaling by the full $\frac{d}{r} = 32.0$ would give standard deviation $\frac{1}{\sqrt{r}} = 0.25$, which is $5.65\times$ too wide at Step 0!).
  - Therefore, we scale each Kraus factor by **$(d / r)^{1/4}$** ($\tilde{M}_{\text{diff}} = (d / r)^{1/4}\, M_{\text{sharp}} U_{\text{diff}}$ and $\tilde{\psi}_{t,2} = (d / r)^{1/4}\, \psi_{t,2} V_{\text{diff}}$, so the bilinear inner product is scaled by $\sqrt{d / r} = \sqrt{32} \approx 5.657$) **without** per-token L2-normalization inside $\mathbb{R}^r$:
    $$c_{\text{sharp}}(t, v) = \langle M_v \mid \psi_{t,1} \rangle \in [-1, 1]$$
    $$c_{\text{diff}}(t, v) = \sqrt{\frac{d}{r}} \, \langle U_{\text{diff}}^\top M_v \mid V_{\text{diff}}^\top \psi_{t,2} \rangle = \left\langle \left(\frac{d}{r}\right)^{1/4} U_{\text{diff}}^\top M_v \;\middle|\; \left(\frac{d}{r}\right)^{1/4} V_{\text{diff}}^\top \psi_{t,2} \right\rangle$$
  - This matches the **exact $\frac{1}{\sqrt{d}} \approx 0.0442$ Step-0 variance** of $c_{\text{sharp}}$ for random vectors, allows aligned semantic clusters to grow to $\mathcal{O}(1)$ during training, and keeps non-isometric norm zeroes intact for out-of-cluster tokens (completely avoiding the $1/r = 0.0625$ background collision floor!).
- **Zero-Materialization Augmented GEMM Trick**:
  Define the augmented vocabulary effect matrix $\widehat{M} = \big[\sqrt{\alpha_{\text{sharp}}}\, M,\; \sqrt{\alpha_{\text{diff}}}\, (M U_{\text{diff}})\big] \in \mathbb{R}^{V \times (d + r)}$ and augmented token state $\widehat{\psi}_t = \big[\sqrt{\alpha_{\text{sharp}}}\, \psi_{t,1},\; \sqrt{\alpha_{\text{diff}}}\, (\psi_{t,2} V_{\text{diff}})\big] \in \mathbb{R}^{N \times (d + r)}$.
  Then a **single matrix multiplication** $\widehat{\psi}_t \widehat{M}^\top$ computes the exact two-beam mixed-state inner product:
  $$c_{\text{mix}}(t, v) = \alpha_{\text{sharp}} \langle M_v \mid \psi_{t,1} \rangle + \alpha_{\text{diff}} \langle U_{\text{diff}}^\top M_v \mid V_{\text{diff}}^\top \psi_{t,2} \rangle$$
  with only $r / d = 16 / 512 = 3.1\%$ extra FLOPs!

### 2.3 $k$-Boson / Multi-Copy Bloch-Fidelity Born Readout ($\text{Tr}(\rho_t^{\otimes k} E_v^{\otimes k})$)
On the Bloch sphere, a mixed density state $\rho_t = \lambda_{\text{sharp}} |\psi_{t,1}\rangle\langle\psi_{t,1}| + \lambda_{\text{diff}} |\psi_{t,2}^{\text{bath}}\rangle\langle\psi_{t,2}^{\text{bath}}|$ has single-copy Bloch measurement fidelity against effect $E_v$:
$$F_{t,v}(\rho_t) = \text{Tr}(\rho_t E_v) = \frac{1 + \text{clamp}(c_{\text{mix}}(t, v), -0.95, 0.999)}{2} \in [0.025, 1.0]$$
Plus an explicit low-rank quadratic Born density bath term $\lambda_{\text{quad}} \, \frac{\|(M_v U_{\text{diff}}) \odot (\psi_{t,2} V_{\text{diff}})\|_1 + |\langle U_{\text{diff}}^\top M_v | V_{\text{diff}}^\top \psi_{t,2} \rangle|^2}{2}$:
$$F_{t,v}^{\text{total}}(\rho_t) = \lambda_{\text{sharp}} \left(\frac{1 + c_{\text{sharp}}(t, v)}{2}\right) + \lambda_{\text{diff}} \left(\frac{1 + c_{\text{diff}}(t, v)}{2} + \frac{1}{2} |c_{\text{diff}}(t, v)|_+^2\right) + \epsilon_{\text{floor}}$$
When measuring **$k$ coherent copies** ($\rho_t^{\otimes k}$ projected onto $E_v^{\otimes k}$) weighted by the empirical prior density $w_v = e^{b_v}$ (`povm_log_prior`), the exact multi-copy Born probability is:
$$P_{\text{Born-}k}(v \mid \rho_t) = \frac{w_v \, \big(F_{t,v}^{\text{total}}(\rho_t)\big)^k}{\sum_{j=1}^V w_j \, \big(F_{t,j}^{\text{total}}(\rho_t)\big)^k}$$
In log-probability space, the $k$-copy Born energy is:
$$\mathcal{E}^{\text{Born-}k}_{t,v} = k \, \ln\!\big(F_{t,v}^{\text{total}}(\rho_t)\big) + b_v$$
where $k \in [28.0, 48.0]$ is a learnable coherent copy number (initialized at $k_0 = 34.0 \approx \beta_0$, with `ContextualBlochPurityGate` clamped to $k_{\max} = 45.0$ so $\Delta \mathcal{E}$ matches the $\sim 8.6\text{ nat}$ separation at $c=0$ while self-dampening via $k / (1 + c_v)$ on already-won Top-1 hits!).

**Why $\mathcal{E}^{\text{Born-}k}_{t,v} = k \ln(F_{t,v}) + b_v$ Solves the Orthogonal Synonym Problem**:
Taking the derivative with respect to the overlap $c_v$:
$$\frac{\partial \mathcal{E}^{\text{Born-}k}_{t,v}}{\partial c_v} = \frac{k}{2 F_{t,v}} = \frac{k}{1 + c_v}$$
- When a plausible replacement token in `Ranks 2–50` is currently orthogonal ($c_v \approx 0$), its attractive gradient is **$k \approx 34.0$**!
- When the `Top-1` token is already aligned ($c_v \approx 0.6$), its gradient tapers down to **$k / 1.6 = 0.625 k \approx 21.25$**!
- Thus, **orthogonal synonyms in `Ranks 2–50` automatically receive a $1.6\times$ to $2.0\times$ stronger attractive gradient than already-won `Top-1` tokens**, pulling synonym clusters together on $S^{d-1}$ without sacrificing `Top-1` accuracy!

---

## 3. Observable Hypotheses & Quantitative Predictions

### Hypothesis H1: Exact Forward & Backward Parity of Chunked $k$-Boson Two-Beam Mixed-State Born Cross-Entropy (`direct_chunked_two_beam_kboson_ce`)
- **Claim**: Our zero-materialization chunked autograd kernel `direct_chunked_two_beam_kboson_ce` (evaluating $\mathcal{E}^{\text{Born-}k}_{t,v} = k \ln(F_{t,v}^{\text{total}}(\rho_t)) + b_v$ across `(psi_1, psi_2, M_sharp, M_diff, k_copies, log_prior)` without materializing full `(B*T, V)` tensors) matches the materialized PyTorch `float32` reference to within `< 1e-5` on loss and `< 1e-5` on all parameter gradients (`dpsi_1`, `dpsi_2`, `dM_sharp`, `dM_diff`, `dk`, `db`), while confirming that the gradient magnitude $|\partial \mathcal{E}_{t,v} / \partial c_v|$ at $c_v = 0.0$ is strictly **$1.60\times \pm 0.01\times$ larger** than at $c_v = 0.60$.

### Hypothesis H2: 100% Quantum Two-Beam $k$-Boson Mixed-State Readout vs. Uncalibrated Single-Beam POVM & Ad-Hoc Softmax (RTX 5060 Ti `<= 8.0 GB` VRAM)
- **Claim**: Under identical 300-step FineWeb-Edu training (`L=6, d=512, H=8, V=50,304`, `2.46M` tokens, `torch.cuda.set_per_process_memory_fraction(0.5)`), the 100% Quantum Two-Beam $k$-Boson Mixed-State Readout (`w15_8gb_qborn_16qubit_256hier_twobeam_kboson_L6_d512`) will:
  1. Outperform both the single-beam POVM (`w15_8gb_qborn_16qubit_256hier_povm_L6_d512`), the ad-hoc `--last-layer-softmax` model (`w15_8gb_qborn_16qubit_256hier_softmax_L6_d512`), AND the leaderboard-exact Classical baseline (`w15_8gb_classical_leaderboard_exact_L6_d512`) on held-out `KellerJordan/modded-nanogpt` Validation Cross-Entropy!
  2. Specifically reduce the conditional NLL in the **`Ranks 6–50` and `Ranks 51–1000` buckets** compared to single-beam POVM while preserving or improving `Top-1` and `Top-5` accuracy.

### Hypothesis H3: Taming `ContextualBlochPurityGate` (`max_beta = 45.0`) + Kraus Diffuse Rank $r=16$ Prevents Rank `6–500` Saturation
- **Claim**: Clamping `ContextualBlochPurityGate` to `max_beta = 45.0` (vs `150.0`) and initializing the rank-$16$ Kraus semantic bath $U_{\text{diff}} = V_{\text{diff}} \in \mathbb{R}^{d \times 16}$ from the top-$16$ right singular vectors of $M_{\text{sharp}}$ with $(d/r)^{1/4}$ per-side scaling produces a diffuse channel with effective rank $R_{\text{eff}} \in [4.0, 14.0]$ (matching `ClassicalModdedGPT`'s low-rank broadcast mode!) while Beam 1 preserves $R_{\text{eff}} \ge 25.0$ for sharp `Top-1` routing, all within `<= 8.0 GB` Peak VRAM.

---

## 4. Planned Code Changes

### 4.1 `src/qdlblocks/layers/early_epoch_quantum.py` & `src/qdlblocks/layers/born_block.py`
1. In `OvercompleteTwoBeamInterference.forward(self, psi: torch.Tensor, return_second_beam: bool = False)`:
   - Compute `out_1 = self.down_1(b1)` and `out_2 = self.down_2(b2)`.
   - `psi_main = l2_normalize(out_1 + out_2, dim=-1, eps=self.eps)`.
   - When `return_second_beam=True`, also compute `psi_beam2 = l2_normalize(out_2 - 0.25 * out_1, dim=-1, eps=self.eps)` and store `self.last_beam2 = psi_beam2` (returning `(psi_main, psi_beam2)` when `return_second_beam=True`).
2. In `ContextualBlochPurityGate.__init__`, lower default `max_beta: float = 45.0` (when `use_two_beam_kboson_readout=True` or configurable via `max_beta`).

### 4.2 `src/qdlblocks/layers/direct_chunked_ce.py`
- Add `_DirectChunkedTwoBeamKBosonCEFn` and `direct_chunked_two_beam_kboson_ce`:
  - Inputs: `psi_sharp: (N, D)`, `psi_diff_r: (N, R)`, `m_sharp: (V, D)`, `m_diff_r: (V, R)`, `k_copies: torch.Tensor`, `flat_y: (N,)`, `log_prior: (V,) | None = None`, `lambda_diff: float = 0.18`, `eps_floor: float = 0.02`, `chunk_size: int = 16384`, `valid_vocab_size: int | None = None`.
  - Computes chunked exact $k$-boson Bloch fidelity energy:
    $$c_{\text{sharp}} = \psi_{\text{sharp}} M_{\text{sharp}}^\top, \qquad c_{\text{diff}} = \psi_{\text{diff}, r} M_{\text{diff}, r}^\top$$
    $$F_{t,v} = (1 - \lambda_{\text{diff}}) \frac{1 + c_{\text{sharp}}}{2} + \lambda_{\text{diff}} \left(\frac{1 + c_{\text{diff}}}{2} + \frac{1}{2} \text{ReLU}(c_{\text{diff}})^2\right) + \epsilon_{\text{floor}}$$
    $$\mathcal{E}_{t,v} = k_{\text{copies}} \ln(F_{t,v}) + b_v$$
    and exact analytical backward gradients $\partial \mathcal{L} / \partial c_{\text{sharp}},\, \partial \mathcal{L} / \partial c_{\text{diff}},\, \partial \mathcal{L} / \partial k_{\text{copies}},\, \partial \mathcal{L} / \partial b_v$ in `float32` / `bfloat16` without materializing `(N, V)` tensors!

### 4.3 `src/qdlblocks/layers/hierarchical_qubit.py` & `src/qdlblocks/models/born_gpt.py`
- In `Hierarchical16QubitPOVMReadout`:
  - Add Kraus semantic bath parameters `self.kraus_u_diff = nn.Parameter(torch.empty(self.embed_dim, self.diffuse_rank))` and `self.kraus_v_diff = nn.Parameter(torch.empty(self.embed_dim, self.diffuse_rank))` (`diffuse_rank: int = 16`), `self.kboson_copies = nn.Parameter(torch.tensor(34.0, dtype=torch.float32))`, and `self.diffuse_mix_logit = nn.Parameter(torch.tensor(-0.20, dtype=torch.float32))` ($\lambda_{\text{diff}} = 0.40 \sigma(\text{logit}) \approx 0.18$).
  - Add `get_two_beam_kboson_factors(self, embedder, psi_sharp, psi_beam2, base_povm_head)` returning `(psi_sharp_norm, psi_diff_r, m_sharp_norm, m_diff_r, k_copies, lambda_diff)` where `scale_side = (self.embed_dim / float(self.diffuse_rank)) ** 0.25`, `m_diff_r = scale_side * (m_sharp_norm @ kraus_u_diff)` and `psi_diff_r = scale_side * (psi_beam2_norm @ kraus_v_diff)` **without L2-normalizing inside $\mathbb{R}^r$**!
  - In `initialize_hierarchical_16qubit_with_ppmi_svd`, initialize `kraus_u_diff` and `kraus_v_diff` as `U_diff = V_diff = Vh[:r].T` from the top-$r$ right singular vectors of the initial coupled POVM matrix $M_{\text{sharp}}$ (`torch.linalg.svd(M_sharp.float(), full_matrices=False)`) so Step-0 diffuse overlaps start smoothly from the proven `6.1475` POVM baseline and immediately capture the top-$r$ principal semantic co-occurrence axes!
- In `QuantumBornGPT`, add `use_two_beam_kboson: bool = False` and `diffuse_bath_rank: int = 16`.

### 4.4 Verification Tests & 8 GB RTX 5060 Ti Experimental Suite
1. **`tests/test_hypothesis_026_two_beam_kboson_readout.py`**:
   - Unit Test 1: Exact forward & backward parity of `direct_chunked_two_beam_kboson_ce` vs materialized PyTorch autograd reference (`|dLoss| < 1e-5`, max grad diff `< 1e-5`), plus verification of the $1/(1+c_v)$ orthogonal synonym gradient amplification law.
   - Unit Test 2: Verifies that `get_two_beam_kboson_factors` does NOT L2-normalize inside $\mathbb{R}^r$ (random token overlap in diffuse channel is $< 0.01$, avoiding the $1/r = 0.0625$ trap) and that diffuse channel effective rank $R_{\text{eff}}(\psi_{\text{diff}, r}) \in [3.0, 16.0]$.
   - Unit Test 3: End-to-end forward/backward + causality verification (`0.0` prefix leakage) of `QuantumBornGPT(..., use_two_beam_kboson=True)`.
2. **`examples/36_verify_plan026_two_beam_kboson_experiments.py`**:
   - Runs under `torch.cuda.set_per_process_memory_fraction(0.5)` (`<= 8.0 GB` VRAM) on our local RTX 5060 Ti, comparing our new 100% Quantum Two-Beam $k$-Boson Mixed-State Readout (`w15_8gb_qborn_16qubit_256hier_twobeam_kboson_L6_d512`) directly against the Plan 025 baselines (`Classical`, `Single-Beam POVM`, `Ad-Hoc Softmax`, and `Unified W13+W15`) across all 5 Target-Rank Cross-Entropy buckets!

---

## 5. Operational Tooling & Script Opportunities (Mandatory Gate)

### 5.1 Must-Have (Updated in This Plan)
- Extend `scripts/calibration_rank_histogram_probe.py` with `--include-twobeam-kboson` so single-command CLI calibration probes automatically include the Two-Beam $k$-Boson Mixed-State Readout alongside Classical, Single-Beam POVM, and Softmax.

### 5.2 Nice-to-Have (Appended to `scripts/BACKLOG.md`)
- **`scripts/kboson_fidelity_landscape_visualizer.py`**: Plots the empirical cosine similarity histogram $p(c_{\text{sharp}})$ and $p(c_{\text{diff}})$ and gradient gain $k / (2 F_{t,v})$ across target rank buckets `1`, `2–5`, `6–50`, `51–1000`, and `>1000`.
