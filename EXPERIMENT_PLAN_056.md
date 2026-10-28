# EXPERIMENT PLAN 056 (`Wave 41`): Rank-$R$ Multi-Meaning $16 \times 16 \times 16 \times 16 = 65,536$ Kronecker Wavefunction GPT & 100M-Token Unigram-Prior Complex Born Readout

## 1. Context, Motivation, and Post-Mortem Insights from `Wave 40`

In `Wave 40` ([`EXPERIMENT_PLAN_055.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_055.md)), [`TiedHoloBornComplexRoPEGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/tied_holo_born_complex_gpt.py) resolved the `Wave 39` detector rank collapse (`R_det: 243 -> 27`) by tying the initial token wavefunctions into the final layer, moving the measurement operator to the sequence side (`6.13x` fewer operator GEMM rows), and replacing hard retractions with smooth differentiable Unit-Sphere, Complex-Unitary, and Parseval-Isotropy penalties (`R_emb = 250.35 / 256`, `embed_norm_median = 1.0026`, `complex_unitary_err = 0.0246`).

Building on `Wave 40` and our architectural design alignment, **`Wave 41` (`EXPERIMENT_PLAN_056.md`)** introduces a fundamental redesign of how tokens are mapped to wavefunctions, how unitary operations act on them inside the network, and how output wavefunctions convert to token probabilities:

1. **Vocabulary-Scale Hilbert Space via $16 \times 16 \times 16 \times 16 = 65,536$ Kronecker Factorization (`5` Hierarchical Levels: $1 \to 16 \to 256 \to 4,096 \to 65,536$)**:
   - Ideally, a vocabulary of $V = 50,257$ tokens lives in a Hilbert space of dimension at least $V$ so every token can occupy a nearly orthogonal state with negligible background cross-talk. However, dense wavefunctions in $\mathbb{C}^{65,536}$ would require $50,304 \times 65,536 = 3.30\text{B}$ complex parameters (`26.4 GB`) just for the embedding table and $65,536 \times 65,536 = 4.29\text{B}$ parameters per unitary matrix.
   - By factoring $\mathbb{C}^{65,536}$ as the 4-way Kronecker tensor product:
     $$\mathcal{H}_{\text{Kron}} = \mathbb{C}^{16} \otimes \mathbb{C}^{16} \otimes \mathbb{C}^{16} \otimes \mathbb{C}^{16} \cong \mathbb{C}^{65,536}$$
     each Kronecker branch requires only $4 \times 16 = 64$ complex numbers (`512x` parameter compression per branch) and local unitary transformations act via **batched $16 \times 16$ complex unitary matrices** $\hat{U}_{r}^{(m)} \in U(16)$ ($4 \times 16^2 = 1,024$ complex parameters per branch—`16x` smaller than a single dense $128 \times 128$ matrix!).
   - Crucially, for two independent random unit states in $\mathbb{C}^{16}$, $\mathbb{E}\bigl[|\langle \psi^{(m)} \mid \phi^{(m)} \rangle_{\mathbb{C}^{16}}|^2\bigr] = \frac{1}{16}$. Across the 4 Kronecker factors, the expected squared background overlap is:
     $$\mathbb{E}\!\left[\prod_{m=0}^{3} \bigl|\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\phi}_{t,r}^{(m)} \rangle_{\mathbb{C}^{16}}\bigr|^2\right] = \left(\frac{1}{16}\right)^4 = \frac{1}{65,536} \approx 1.5259 \times 10^{-5}$$
     whereas a flat unit vector in $\mathbb{C}^{128}$ has background noise $\frac{1}{128} = 7.8125 \times 10^{-3}$—giving the $16^4$ Kronecker tensor product a **`512x` improvement in single-branch signal-to-background noise ratio** using the exact same `128` complex numbers per token (at $R = 2$ branches)!

2. **Rank-$R$ Multi-Meaning Superposition per Word (`Polysemy Without Cross-Sense Dilution`)**:
   - Natural language words are polysemous (e.g., `"bank"`, `"run"`, `"set"`, `"charge"` carry multiple distinct meanings depending on context). A Rank-1 pure product state $\bigotimes_{m=0}^3 \lvert \hat{\psi}_v^{(m)} \rangle$ forces a word with two orthogonal meanings $\lvert \text{sense}_0 \rangle, \lvert \text{sense}_1 \rangle$ into an average state whose 4-factor Kronecker overlap drops to $(1/\sqrt{2})^8 = 1/16 = 0.0625$ on both meanings.
   - Representing each vocabulary token $v \in \{0, \dots, V-1\}$ as a **Rank-$R$ entangled superposition of $R$ meaning branches** ($R \in \{2, 4\}$):
     $$\lvert \Psi_v \rangle = \sum_{r=0}^{R-1} \alpha_{v,r} \, \lvert \hat{\psi}_{v,r}^{(0)} \rangle \otimes \lvert \hat{\psi}_{v,r}^{(1)} \rangle \otimes \lvert \hat{\psi}_{v,r}^{(2)} \rangle \otimes \lvert \hat{\psi}_{v,r}^{(3)} \rangle \in \mathbb{C}^{16 \times 16 \times 16 \times 16}, \qquad \sum_{r=0}^{R-1} \alpha_{v,r}^2 = 1$$
     acts as a **quantum OR-gate across the $R$ meanings of word $v$** and a **quantum AND-gate across the 4 Kronecker factors $m \in \{0, 1, 2, 3\}$**, penalized by an explicit cross-meaning branch orthogonality regularizer $\mathcal{L}_{\text{branch}} = \frac{1}{V \binom{R}{2}}\sum_{v} \sum_{r < r'} \prod_{m=0}^3 |\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\psi}_{v,r'}^{(m)} \rangle_{\mathbb{C}^{16}}|^2$ so the $R$ branches of each word specialize to distinct senses in $\mathbb{C}^{65,536}$.

3. **Full `100M`-Token Unigram-Prior Wavefunction Weighting & Three Targeted `Wave 40` Refinements**:
   - Every token wavefunction is weighted by its empirical unigram probability $P_{\text{uni}}(v)$ computed over the full **`100M`-token FineWeb slice** (`.cache/fineweb_100M/train_full_100M.bin` or `fineweb_train_000001.bin`, where `49,850` of `50,257` tokens are observed and the remaining `407` unobserved tokens receive a positive floor $\epsilon_{\text{floor}} = 1.0$ count).
   - Incorporates the three post-mortem improvements identified at the close of `Wave 40`:
     1. **Phase-Balanced Reference Wave ($e^{i\pi/4} = \frac{1+i}{\sqrt{2}}$) + Multi-Scale Kronecker Convex Peak Amplification**: Eliminates the $16\times$ real-vs-imaginary gradient asymmetry near the origin while combining a 1-factor linear gradient highway $(1 - \eta_{\text{kron}})\bar{z}_{t,v,r}$ with 4-factor Kronecker product amplification $\eta_{\text{kron}} \prod_{m=0}^3 z_{t,v,r}^{(m)}$ and 8th-degree coincidence energy $\gamma_{\text{coinc}} \prod_{m=0}^3 |z_{t,v,r}^{(m)}|^2$.
     2. **Relaxed Per-Factor Parseval Isotropy Weight ($\lambda_{\text{iso}} = 0.05$)**: Keeps each $\mathbb{C}^{16}$ factor space well-conditioned (`R_factor >= 20 / 32`) without over-constraining semantic clustering.
     3. **Single Batched Tensor-Core GEMM (`torch.bmm`) for All $4 R$ Complex Factor Overlaps**: Stacking `[phi_re, phi_im]` and `[-phi_im, phi_re]` into shape `(4*R, 2*N, 32)` computes both the real and imaginary parts of all $4 R$ 16-dimensional complex inner products in a single `torch.bmm` per vocabulary chunk.

---

## 2. Testable Scientific Hypotheses (`H1`–`H4`)

### Hypothesis `H1`: $16 \times 16 \times 16 \times 16 = 65,536$ Kronecker Signal-to-Noise Ratio (`512x` Background Suppression) & Multi-Meaning Polysemy Capacity (`Rank-R` vs. `Rank-1`)
- **Independent Variable**: Wavefunction factorization and meaning rank $R \in \{1, 2, 4\}$ at matched per-factor dimension $d_f = 16$ ($M = 4$ factors, $\mathcal{H}_{\text{Kron}} = (\mathbb{C}^{16})^{\otimes 4} \cong \mathbb{C}^{65,536}$):
  1. `flat_c128`: Single flat unit vector $\lvert e_v \rangle \in \mathbb{C}^{128}$ (`256` real params/token).
  2. `kronecker_16x4_rank1`: Rank-$1$ 4-factor Kronecker product $\bigotimes_{m=0}^3 \lvert \hat{\psi}_{v,0}^{(m)} \rangle \in (\mathbb{C}^{16})^{\otimes 4}$ (`128` real params/token).
  3. `kronecker_16x4_rank2`: Rank-$2$ multi-meaning 4-factor Kronecker superposition $\sum_{r=0}^1 \alpha_{v,r} \bigotimes_{m=0}^3 \lvert \hat{\psi}_{v,r}^{(m)} \rangle$ (`258` real params/token—matched to `flat_c128`).
  4. `kronecker_16x4_rank4`: Rank-$4$ multi-meaning 4-factor Kronecker superposition $\sum_{r=0}^3 \alpha_{v,r} \bigotimes_{m=0}^3 \lvert \hat{\psi}_{v,r}^{(m)} \rangle$ (`516` real params/token).
- **Dependent Observables**:
  - Empirical random-impostor squared Kronecker overlap $\mathbb{E}_{u \ne v}\bigl[|\mathcal{K}_{u,v,r}|^2\bigr] = \mathbb{E}_{u \ne v}\bigl[\prod_{m=0}^3 |\langle \hat{\psi}_{u,r}^{(m)} \mid \hat{\psi}_{v,r}^{(m)} \rangle_{\mathbb{C}^{16}}|^2\bigr]$ vs. flat $\mathbb{C}^{128}$ random overlap $\mathbb{E}_{u \ne v}\bigl[|\langle e_u \mid e_v \rangle_{\mathbb{C}^{128}}|^2\bigr]$,
  - Polysemous $K$-sense target-match Kronecker energy $\min_{s \in \{0, \dots, K-1\}} \mathcal{C}(v, \Phi_{\text{sense } s})$ when word $v$ must respond to $K = 2$ orthogonal context states $\lvert \Phi_{\text{sense } 0} \rangle, \lvert \Phi_{\text{sense } 1} \rangle \in (\mathbb{C}^{16})^{\otimes 4}$,
  - Exact probability normalization $\max_t \bigl|\sum_{v=0}^{V-1} P_{\text{KronBorn}}(v \mid t) - 1.0\bigr|$ and chunked vs. unchunked autograd parity (`max_abs_grad_err`).
- **Quantitative Pass/Fail Criteria**:
  1. The 4-factor Kronecker random background overlap $\mathbb{E}_{u \ne v}[|\mathcal{K}_{u,v,r}|^2]$ is within $[1.0 \times 10^{-5}, 2.5 \times 10^{-5}]$ (theoretical $16^{-4} = 1.5259 \times 10^{-5}$), achieving at least a **`400x` reduction in background noise** compared to flat $\mathbb{C}^{128}$ ($1/128 = 7.8125 \times 10^{-3}$).
  2. On a 2-sense polysemous target test ($K = 2$ orthogonal context senses per word), `kronecker_16x4_rank2` and `kronecker_16x4_rank4` achieve **`>= 6.0x` higher** 4-factor coincidence $\prod_{m=0}^3 |\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\phi}_{s,r}^{(m)} \rangle|^2$ on both senses than `kronecker_16x4_rank1` ($0.50 \text{ vs. } (0.50)^4 = 0.0625$).
  3. `KroneckerMultiMeaningBornChunkedNLLFn` matches the full unchunked `float32` reference within `< 1e-5` forward loss error and `< 1e-4` backward gradient error on all 4 inputs (`phi_factors`, `psi_factors`, `branch_beta`, `branch_alpha`).

### Hypothesis `H2`: 100M-Token Empirical Unigram-Prior Wavefunction Scaling vs. Uniform & 2M-Slice Priors
- **Independent Variable**: Unigram prior source for vocabulary wavefunction weighting $w_{\text{uni}}(v) = (P_{\text{uni}}(v) / \max_{v'} P_{\text{uni}}(v'))^{\tau_{\text{uni}}}$ across FineWeb:
  1. `uniform_prior`: $w_{\text{uni}}(v) = 1.0$ for all $v \in \{0, \dots, V-1\}$ ($\tau_{\text{uni}} = 0.0$).
  2. `slice_2m_prior`: $P_{\text{uni}}(v)$ estimated from the first `2M` tokens (`6,502` unseen tokens, `33,249` tokens with `< 10` counts).
  3. `slice_100m_prior`: $P_{\text{uni}}(v)$ estimated from the full `100M`-token stream (`.cache/fineweb_100M/train_full_100M.bin` or `fineweb_train_000001.bin`, `49,850` observed tokens, `407` unseen tokens floored at $\epsilon_{\text{floor}} = 1.0$ count, $\tau_{\text{uni}} = 0.55$).
- **Dependent Observables**:
  - KL divergence $D_{\text{KL}}(P_{\text{val}} \,\|\, P_{\text{uni}})$ and Step-0 zero-context Born NLL on the held-out FineWeb validation shard (`val.bin`),
  - Unseen-token coverage (`6,502` unseen in `2M` vs. `407` unseen in `100M`), and
  - Held-out validation NLL (`val_ce`) and `top1_acc` after `150` training steps (`d_f = 16, M = 4, R = 2, L = 4`).
- **Quantitative Pass/Fail Criteria**:
  1. `slice_100m_prior` reduces unseen vocabulary tokens by **`> 15x`** vs. `slice_2m_prior` (`407` vs. `6,502`) and achieves lower KL divergence $D_{\text{KL}}(P_{\text{val}} \,\|\, P_{\text{uni}})$ to the held-out validation distribution than both `uniform_prior` and `slice_2m_prior`.
  2. `slice_100m_prior` achieves lower Step-0 validation NLL (`< 8.20` nats vs. `10.82` nats for `uniform_prior`) and lower Step-150 validation NLL (`val_ce`) than `uniform_prior`.

### Hypothesis `H3`: Batched $16 \times 16$ Factor Unitary Mixing, Cross-Meaning Branch Orthogonality, & Relaxed Parseval Isotropy ($\lambda_{\text{iso}} = 0.05$)
- **Independent Variable**: Regularization and multi-meaning specialization over `150` training steps on FineWeb (`R = 2, M = 4, d_f = 16, L = 4`):
  1. `overconstrained_iso_no_branch_orth`: High Parseval isotropy weight ($\lambda_{\text{iso}} = 0.50$, as in `Wave 40`) and zero cross-meaning branch orthogonality penalty ($\lambda_{\text{branch}} = 0.0$).
  2. `relaxed_iso_with_branch_orth`: Relaxed per-factor Parseval isotropy ($\lambda_{\text{iso}} = 0.05$), per-factor 16D sphere penalty ($\lambda_{\text{sphere}} = 2.0$), batched $16 \times 16$ unitary penalty ($\lambda_{\text{unitary}} = 0.08$), and cross-meaning 4-factor Kronecker branch orthogonality penalty ($\lambda_{\text{branch}} = 0.20$).
- **Dependent Observables**:
  - Mean per-factor participation ratio $\bar{R}_{\text{factor}} = \frac{1}{4R}\sum_{r=0}^{R-1}\sum_{m=0}^3 R_{\text{eff}}(\tilde{\Psi}_{r}^{(m)}) \in [1, 32]$ and combined 128D complex embedding participation ratio $R_{\text{emb}} \in [1, 128 R]$,
  - Mean cross-branch 4-factor Kronecker overlap $\bar{O}_{\text{branch}} = \frac{1}{V \binom{R}{2}} \sum_{v=0}^{V-1} \sum_{r < r'} \prod_{m=0}^3 |\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\psi}_{v,r'}^{(m)} \rangle_{\mathbb{C}^{16}}|^2$,
  - Median factor norm $\|\psi_{v,r}^{(m)}\|_{\mathbb{C}^{16}}$, $16 \times 16$ unitary drift, and held-out `val_ce`.
- **Quantitative Pass/Fail Criteria**:
  1. `relaxed_iso_with_branch_orth` keeps mean per-factor rank $\bar{R}_{\text{factor}} \ge 20.0 / 32.0$ and global $R_{\text{emb}} \ge 140.0 / 256.0$ while keeping cross-branch 4-factor overlap $\bar{O}_{\text{branch}} < 1.0 \times 10^{-3}$ (confirming the $R$ branches of each word remain nearly orthogonal in $\mathbb{C}^{65,536}$).
  2. `relaxed_iso_with_branch_orth` achieves lower validation NLL (`val_ce`) than `overconstrained_iso_no_branch_orth` while maintaining median factor norm in $[0.95, 1.15]$ and unitary drift $< 0.25$.

### Hypothesis `H4`: 5-Minute (`300s`) Head-to-Head Verification of `KroneckerMultiMeaning16x4GPT` (`Wave 41` vs. `Wave 40`)
- **Independent Variable**: 5-minute (`300s`) live training run of [`KroneckerMultiMeaning16x4GPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py) (`R = 2` meanings, $M = 4$ factors of $d_f = 16 \implies \mathbb{C}^{16 \times 16 \times 16 \times 16} = \mathbb{C}^{65,536}$, $L = 6$ layers, `100M`-token unigram prior) on the RTX 5060 Ti (`B = 16, T = 512`) vs. `Wave 40` (`w40_tied_holo_born_d128c_H4_dh32_L6_300s`, which finished at `val_ce = 6.9492`, `val_top1 = 12.40%`, `val_top50 = 46.13%`).
- **Dependent Observables**:
  - `val_ce`, `val_ppl`, `top1_acc`, `top50_acc`, `sharp_eff_rank` ($R_{\text{emb}}$), `mean_factor_rank`, `cross_branch_overlap`, `embed_norm_median`, and `complex_unitary_err` streamed live via `StandardizedExperimentLogger` to `.cache/live_logs/`, `.cache/model_cards/`, and `.cache/benchmarks.db`.
- **Quantitative Pass/Fail Criteria**:
  1. `KroneckerMultiMeaning16x4GPT` achieves a lower 5-minute validation NLL than `Wave 40` (`val_ce < 6.9492`, target `<= 6.65`) and higher `top1_acc` (`> 12.40%`) while maintaining `sharp_eff_rank >= 140.0`, `embed_norm_median in [0.95, 1.15]`, and `0` NaN/Inf events.

---

## 3. Planned Changes (Rule 1–8 Compliant Architecture & Modules)

Following **Rule 6** of [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md), we leave [`GeminiBornComplexPackedWithRoPE.py`](file:///home/festivity/agy/QuantumDLBlocks/GeminiBornComplexPackedWithRoPE.py), [`src/qdlblocks/models/gemini_packed_complex_rope_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/gemini_packed_complex_rope_gpt.py), and [`src/qdlblocks/models/tied_holo_born_complex_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/tied_holo_born_complex_gpt.py) completely untouched and create a dedicated, flag-free module [`src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py).

### 3.1 New Flag-Free Module: [`src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py)

1. **Top-of-File Named Constants (Rule 7)**:
   ```python
   DEFAULT_NUM_FACTORS: int = 4
   DEFAULT_FACTOR_DIM: int = 16
   DEFAULT_NUM_MEANINGS: int = 2
   DEFAULT_KRON_BOSON_P: float = 11.0
   DEFAULT_KRON_HOLO_ALPHA: float = 0.85
   DEFAULT_KRON_BLEND: float = 0.55
   DEFAULT_KRON_COINC_GAMMA: float = 0.45
   DEFAULT_KRON_EPS: float = 1e-6
   DEFAULT_REF_PHASE_COS_SIN: float = 0.7071067811865476  # 1 / sqrt(2) for e^{i*pi/4}
   DEFAULT_UNIGRAM_TEMP: float = 0.55
   DEFAULT_UNIGRAM_FLOOR_COUNT: float = 1.0
   DEFAULT_BRANCH_EPS: float = 1e-4
   DEFAULT_SPHERE_PENALTY_WEIGHT: float = 2.0
   DEFAULT_UNITARY_PENALTY_WEIGHT: float = 0.08
   DEFAULT_ISO_PENALTY_WEIGHT: float = 0.05
   DEFAULT_BRANCH_ORTH_WEIGHT: float = 0.20
   DEFAULT_VOCAB_CHUNK_SIZE: int = 4096
   DEFAULT_MUON_LR: float = 0.015
   DEFAULT_EMBED_LR: float = 0.008
   DEFAULT_GATE_LR: float = 0.003
   ```

2. **Frozen Configuration Dataclass (`KroneckerMultiMeaningConfig`, `<= 8` args in all functions/methods per Rule 2)**:
   ```python
   @dataclass(frozen=True)
   class KroneckerMultiMeaningConfig:
       vocab_size: int = 50304
       valid_vocab_size: int = 50257
       num_meanings: int = DEFAULT_NUM_MEANINGS
       num_factors: int = DEFAULT_NUM_FACTORS
       factor_dim: int = DEFAULT_FACTOR_DIM
       num_layers: int = 6
       block_size: int = 512
       boson_p: float = DEFAULT_KRON_BOSON_P
       holo_alpha: float = DEFAULT_KRON_HOLO_ALPHA
       kron_blend: float = DEFAULT_KRON_BLEND
       coinc_gamma: float = DEFAULT_KRON_COINC_GAMMA
       unigram_temp: float = DEFAULT_UNIGRAM_TEMP
       sphere_penalty_weight: float = DEFAULT_SPHERE_PENALTY_WEIGHT
       unitary_penalty_weight: float = DEFAULT_UNITARY_PENALTY_WEIGHT
       iso_penalty_weight: float = DEFAULT_ISO_PENALTY_WEIGHT
       branch_orth_weight: float = DEFAULT_BRANCH_ORTH_WEIGHT
       checkpoint_blocks: bool = False

       @property
       def d_complex_total(self) -> int:
           return int(self.num_meanings * self.num_factors * self.factor_dim)

       @property
       def n_registers(self) -> int:
           return max(1, self.d_complex_total // 32)

       @property
       def d_register(self) -> int:
           return 32
   ```
   - At the default `num_meanings = 2, num_factors = 4, factor_dim = 16`, `d_complex_total = 2 * 4 * 16 = 128` (`n_registers = 4, d_register = 32`), and the tensor-product Hilbert space is $(\mathbb{C}^{16})^{\otimes 4} \cong \mathbb{C}^{65,536}$.

3. **Batched $16 \times 16$ Factor Unitary & Multi-Meaning Mixer (`Kronecker16x4UnitaryMeaningMixer`)**:
   - Operates directly on the flattened complex state `(psi_re, psi_im)` of shape `(B, T, R * M * d_f)` by reshaping internally to `(B, T, R, M, d_f)` where $M = 4, d_f = 16$:
     - **Batched Factor Unitaries (`factor_u_re`, `factor_u_im`)** of shape `(R * M, d_f, d_f)` (`(8, 16, 16)` at $R = 2$):
       Initialized as independent random Haar complex unitary matrices in $U(16)$. Reshapes input `(psi_re, psi_im)` to `(B, T, K, d_f)` ($K = R \cdot M$) and applies complex matrix multiplication:
       ```python
       u1_re = torch.einsum("btkd,kdo->btko", x_re, u_re) - torch.einsum("btkd,kdo->btko", x_im, u_im)
       u1_im = torch.einsum("btkd,kdo->btko", x_re, u_im) + torch.einsum("btkd,kdo->btko", x_im, u_re)
       ```
       then reshapes `(u1_re, u1_im)` to `(B, T, R, M, d_f)`.
     - **Cross-Factor $4 \times 4$ Complex Entangler (`factor_mix_re`, `factor_mix_im`)** of shape `(R, M, M)` (`(2, 4, 4)` at $R = 2$):
       Initialized quantitatively near the identity as `factor_mix_re = eye(M).unsqueeze(0).repeat(R, 1, 1) + 0.02 * randn(R, M, M)` and `factor_mix_im = 0.02 * randn(R, M, M)`. Applied second across the factor axis $M$:
       ```python
       u2_re = torch.einsum("btrmd,rmn->btrnd", u1_re, fm_re) - torch.einsum("btrmd,rmn->btrnd", u1_im, fm_im)
       u2_im = torch.einsum("btrmd,rmn->btrnd", u1_re, fm_im) + torch.einsum("btrmd,rmn->btrnd", u1_im, fm_re)
       ```
     - **Cross-Meaning $R \times R$ Complex Branch Interferometer (`meaning_mix_re`, `meaning_mix_im`)** of shape `(M, R, R)` (`(4, 2, 2)` at $R = 2$):
       Initialized quantitatively near the identity as `meaning_mix_re = eye(R).unsqueeze(0).repeat(M, 1, 1) + 0.02 * randn(M, R, R)` and `meaning_mix_im = 0.02 * randn(M, R, R)`. Applied third across the meaning axis $R$:
       ```python
       u3_re = torch.einsum("btrmd,mrs->btsmd", u2_re, mm_re) - torch.einsum("btrmd,mrs->btsmd", u2_im, mm_im)
       u3_im = torch.einsum("btrmd,mrs->btsmd", u2_re, mm_im) + torch.einsum("btrmd,mrs->btsmd", u2_im, mm_re)
       ```
     - **Per-Factor Normalization & Gated Residual Update**:
       Normalizes `(u3_re, u3_im)` per factor along `dim=-1` ($d_f = 16$) via `complex_l2_normalize(u3_re, u3_im, dim=-1)`, blends with the reshaped input `(x_re, x_im)` using `gate = torch.tanh(self.mixer_gate)` (`mixer_gate = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))`), normalizes each `(b, t, r, m)` factor ket along `dim=-1` ($d_f = 16$) and rescales by $1/\sqrt{R \cdot M}$ before flattening back to `(B, T, R * M * d_f)` so the full register vector also has unit norm $\|\psi_{b,t}\|_{\mathbb{C}^{64R}} = 1.0$.
   - **Exact Signatures (`<= 8` args)**:
     ```python
     class Kronecker16x4UnitaryMeaningMixer(nn.Module):
         def __init__(self, cfg: KroneckerMultiMeaningConfig) -> None: ...
         def forward(
             self,
             psi_re: torch.Tensor,
             psi_im: torch.Tensor,
         ) -> tuple[torch.Tensor, torch.Tensor]: ...
     ```

4. **Pure Polynomial Multi-Meaning $16^4$ Kronecker Born Readout (`eval_kronecker_multimeaning_born_probs` & `KroneckerMultiMeaningBornChunkedNLLFn`)**:
   - **Packed Tensor Conventions (`<= 8` args)**:
     - `phi_factors`: Real tensor of shape `(N, R, M, 2 * d_f)` (`(N, R, 4, 32)`) where each `(t, r, m)` slice is the unit-norm real+imag concatenation `[\phi_{t,r,re}^{(m)}, \phi_{t,r,im}^{(m)}]` with $\|\text{phi\_factors}[t, r, m]\|_2 = 1.0$.
     - `psi_factors`: Real tensor of shape `(V, R, M, 2 * d_f)` (`(V, R, 4, 32)`) where $V = \text{valid\_vocab\_size} = 50,257$ and each `(v, r, m)` slice is the unit-norm tied vocabulary factor ket `[\psi_{v,r,re}^{(m)}, \psi_{v,r,im}^{(m)}]` with $\|\text{psi\_factors}[v, r, m]\|_2 = 1.0$.
     - `branch_beta`: Real tensor of shape `(N, R)` with $\sum_{r=0}^{R-1} \beta_{t,r}^2 = 1.0$.
     - `branch_alpha`: Real tensor of shape `(V, R)` with $\sum_{r=0}^{R-1} \alpha_{v,r}^2 = 1.0$.
     - `prior_weights`: Real tensor of shape `(V,)` containing the 100M-token unigram weights $w_{\text{uni},v} = (P_{\text{uni}}(v) / \max_{v'} P_{\text{uni}}(v'))^{\tau_{\text{uni}}} \in (0, 1]$.
   - **Single-`torch.bmm` Complex Factor Overlap Computation**:
     - Let $K = R \cdot M$ (`8` at $R=2, M=4$) and $D_2 = 2 \cdot d_f = 32$.
     - Split `phi_factors` into `phi_re = phi_factors[..., :d_f]` and `phi_im = phi_factors[..., d_f:]`, and construct `phi_skew = torch.cat([-phi_im, phi_re], dim=-1)` of shape `(N, R, M, 32)`.
     - Permute and stack `[phi_factors, phi_skew]` into `U_batched` of shape `(K, 2 * N, 32)` so rows `0 .. N-1` hold `[\phi_{re}, \phi_{im}]` and rows `N .. 2N-1` hold `[-\phi_{im}, \phi_{re}]`.
     - For any vocabulary chunk $c = [v_0 : v_1]$ of size $V_c = v_1 - v_0$, reshape `psi_factors[v_0:v_1]` to `W_chunk` of shape `(K, 32, V_c)` (via `.reshape(V_c, K, 32).permute(1, 2, 0).contiguous()`).
     - Execute a **single batched matrix multiplication** `Z_cat = torch.bmm(U_batched, W_chunk).float()` of shape `(K, 2 * N, V_c)`:
       - `x = Z_cat[:, :N, :].reshape(R, M, N, V_c)` is the exact real part $x_{r,m,t,v} = \operatorname{Re}\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\phi}_{t,r}^{(m)} \rangle_{\mathbb{C}^{16}}$, and
       - `y = Z_cat[:, N:, :].reshape(R, M, N, V_c)` is the exact imaginary part $y_{r,m,t,v} = \operatorname{Im}\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\phi}_{t,r}^{(m)} \rangle_{\mathbb{C}^{16}} = -\phi_{\text{im}}^\top \psi_{\text{re}} + \phi_{\text{re}}^\top \psi_{\text{im}}$.
   - **2-Level Binary Complex Product Tree for $16 \times 16 \times 16 \times 16$ Kronecker Overlap**:
     - For each branch $r \in \{0, \dots, R-1\}$:
       1. Mean 1-factor overlap:
          $$\bar{x}_{r} = \frac{1}{4}(x_{r,0} + x_{r,1} + x_{r,2} + x_{r,3}), \qquad \bar{y}_{r} = \frac{1}{4}(y_{r,0} + y_{r,1} + y_{r,2} + y_{r,3})$$
       2. Pair `(0, 1)` complex product ($16 \times 16 = 256$-dim intermediate space):
          $$p_{01,\text{re}} = x_{r,0} x_{r,1} - y_{r,0} y_{r,1}, \qquad p_{01,\text{im}} = x_{r,0} y_{r,1} + y_{r,0} x_{r,1}$$
       3. Pair `(2, 3)` complex product ($16 \times 16 = 256$-dim intermediate space):
          $$p_{23,\text{re}} = x_{r,2} x_{r,3} - y_{r,2} y_{r,3}, \qquad p_{23,\text{im}} = x_{r,2} y_{r,3} + y_{r,2} x_{r,3}$$
       4. Full 4-factor Kronecker product ($256 \times 256 = 65,536$-dim space):
          $$\mathcal{K}_{r,\text{re}} = p_{01,\text{re}} p_{23,\text{re}} - p_{01,\text{im}} p_{23,\text{im}}, \qquad \mathcal{K}_{r,\text{im}} = p_{01,\text{re}} p_{23,\text{im}} + p_{01,\text{im}} p_{23,\text{re}}$$
       5. Multi-scale composite branch amplitude ($\eta = \text{kron\_blend}$):
          $$\mathcal{M}_{r,\text{re}} = (1 - \eta)\bar{x}_{r} + \eta \mathcal{K}_{r,\text{re}}, \qquad \mathcal{M}_{r,\text{im}} = (1 - \eta)\bar{y}_{r} + \eta \mathcal{K}_{r,\text{im}}$$
   - **Coherent Multi-Meaning Sum & Phase-Balanced Holographic Born Intensity**:
     - Let $w_{r,t,v} = \beta_{t,r} \alpha_{v,r}$ (`branch_beta[:, r:r+1] * branch_alpha[v_0:v_1, r].unsqueeze(0)` of shape `(N, V_c)`).
     - Coherent sum over meaning branches $r \in \{0, \dots, R-1\}$:
       $$A_{\text{re}} = \sum_{r=0}^{R-1} w_{r,t,v} \mathcal{M}_{r,\text{re}}, \qquad A_{\text{im}} = \sum_{r=0}^{R-1} w_{r,t,v} \mathcal{M}_{r,\text{im}}$$
     - Incoherent 4-factor coincidence energy over meaning branches:
       $$\mathcal{C}_{t,v} = \sum_{r=0}^{R-1} w_{r,t,v}^2 \bigl(\mathcal{K}_{r,\text{re}}^2 + \mathcal{K}_{r,\text{im}}^2\bigr)$$
     - Phase-balanced holographic + coincidence Born intensity ($s_0 = \frac{1}{\sqrt{2}} = \text{DEFAULT\_REF\_PHASE\_COS\_SIN}$, $\alpha = \text{holo\_alpha}$, $\gamma_c = \text{coinc\_gamma}$, $\varepsilon_0 = \text{DEFAULT\_KRON\_EPS}$):
       $$I_{t,v} = \bigl(s_0 + \alpha A_{\text{re}}\bigr)^2 + \bigl(s_0 + \alpha A_{\text{im}}\bigr)^2 + \gamma_c \, \mathcal{C}_{t,v} + \varepsilon_0$$
     - Pure polynomial $p$-boson Born weight (`torch.pow`, zero `exp` / `softmax` / `logsumexp`):
       $$\tilde{P}_{t,v} = w_{\text{uni},v} \cdot (I_{t,v})^p, \qquad Z_t = \sum_{v'=0}^{V-1} \tilde{P}_{t,v'}, \qquad P_{\text{KronBorn}}(v \mid t) = \frac{\tilde{P}_{t,v}}{Z_t}$$
     - Per-token NLL: $\ell_t = \ln(Z_t) - \ln(\tilde{P}_{t, y_t})$, weighted by $\tilde{w}_t = 1/N$ (or normalized `token_weights`).

   - **Exact Analytical Backward Equations of `KroneckerMultiMeaningBornChunkedNLLFn`**:
     - **Forward Signature (`<= 8` args)**:
       ```python
       @staticmethod
       def forward(
           ctx: Any,
           phi_factors: torch.Tensor,
           psi_factors: torch.Tensor,
           branch_beta: torch.Tensor,
           branch_alpha: torch.Tensor,
           targets: torch.Tensor,
           prior_weights: torch.Tensor,
           hyperparams: torch.Tensor,
           token_weights: torch.Tensor | None = None,
       ) -> torch.Tensor:
       ```
       where `hyperparams` is a 4-element `float32` tensor `[boson_p, holo_alpha, kron_blend, coinc_gamma]`.
     - **In `backward(ctx, grad_out)`**:
       For each chunk $c = [v_0 : v_1]$:
       1. Recompute $x_{r,m}, y_{r,m}, \bar{x}_r, \bar{y}_r, p_{01}, p_{23}, \mathcal{K}_r, \mathcal{M}_r, A_{\text{re}}, A_{\text{im}}, \mathcal{C}, I_{t,v}$, and $P_{t,v} = \tilde{P}_{t,v} / Z_t$.
       2. Upstream gradient with respect to $\ln \tilde{P}_{t,v}$ and $I_{t,v}$:
          $$\delta_{t,v} = (\text{grad\_out} \cdot \tilde{w}_t) \cdot \bigl(P_{t,v} - \mathbb{I}[v = y_t]\bigr), \qquad g_I = \delta_{t,v} \cdot \frac{p}{I_{t,v}}$$
       3. Gradients with respect to $A_{\text{re}}, A_{\text{im}}$, and $\mathcal{C}_{t,v}$:
          $$g_{A,\text{re}} = g_I \cdot 2 \alpha (s_0 + \alpha A_{\text{re}}), \qquad g_{A,\text{im}} = g_I \cdot 2 \alpha (s_0 + \alpha A_{\text{im}}), \qquad g_{\mathcal{C}} = g_I \cdot \gamma_c$$
       4. For each branch $r \in \{0, \dots, R-1\}$ (with $w_r = \beta_{t,r} \alpha_{v,r}$ and $|\mathcal{K}_r|^2 = \mathcal{K}_{r,\text{re}}^2 + \mathcal{K}_{r,\text{im}}^2$):
          - Gradient wrt branch coupling weight $w_r$:
            $$g_{w_r} = g_{A,\text{re}} \mathcal{M}_{r,\text{re}} + g_{A,\text{im}} \mathcal{M}_{r,\text{im}} + 2 g_{\mathcal{C}} w_r |\mathcal{K}_r|^2$$
            Accumulate into `d_branch_beta[:, r] += (g_{w_r} * alpha_chunk[:, r].unsqueeze(0)).sum(dim=1)` and `d_branch_alpha[v_0:v_1, r] = (g_{w_r} * branch_beta[:, r:r+1]).sum(dim=0)`.
          - Gradient wrt $\mathcal{M}_{r,\text{re}}, \mathcal{M}_{r,\text{im}}$:
            $$g_{\mathcal{M}_r,\text{re}} = w_r \, g_{A,\text{re}}, \qquad g_{\mathcal{M}_r,\text{im}} = w_r \, g_{A,\text{im}}$$
          - Gradient wrt 4-factor Kronecker product $\mathcal{K}_{r,\text{re}}, \mathcal{K}_{r,\text{im}}$:
            $$g_{\mathcal{K}_r,\text{re}} = \eta \, g_{\mathcal{M}_r,\text{re}} + 2 g_{\mathcal{C}} w_r^2 \mathcal{K}_{r,\text{re}}, \qquad g_{\mathcal{K}_r,\text{im}} = \eta \, g_{\mathcal{M}_r,\text{im}} + 2 g_{\mathcal{C}} w_r^2 \mathcal{K}_{r,\text{im}}$$
          - Exact 2-level binary complex product tree pullback to $(x_{r,m}, y_{r,m})$:
            $$g_{p_{01},\text{re}} = g_{\mathcal{K}_r,\text{re}} p_{23,\text{re}} + g_{\mathcal{K}_r,\text{im}} p_{23,\text{im}}, \qquad g_{p_{01},\text{im}} = g_{\mathcal{K}_r,\text{im}} p_{23,\text{re}} - g_{\mathcal{K}_r,\text{re}} p_{23,\text{im}}$$
            $$g_{p_{23},\text{re}} = g_{\mathcal{K}_r,\text{re}} p_{01,\text{re}} + g_{\mathcal{K}_r,\text{im}} p_{01,\text{im}}, \qquad g_{p_{23},\text{im}} = g_{\mathcal{K}_r,\text{im}} p_{01,\text{re}} - g_{\mathcal{K}_r,\text{re}} p_{01,\text{im}}$$
            and adding the linear 1-factor term $g_{\bar{x}_r} = \frac{1 - \eta}{4} g_{\mathcal{M}_r,\text{re}}, \; g_{\bar{y}_r} = \frac{1 - \eta}{4} g_{\mathcal{M}_r,\text{im}}$:
            $$g_{x_{r,0}} = g_{\bar{x}_r} + g_{p_{01},\text{re}} x_{r,1} + g_{p_{01},\text{im}} y_{r,1}, \qquad g_{y_{r,0}} = g_{\bar{y}_r} + g_{p_{01},\text{im}} x_{r,1} - g_{p_{01},\text{re}} y_{r,1}$$
            $$g_{x_{r,1}} = g_{\bar{x}_r} + g_{p_{01},\text{re}} x_{r,0} + g_{p_{01},\text{im}} y_{r,0}, \qquad g_{y_{r,1}} = g_{\bar{y}_r} + g_{p_{01},\text{im}} x_{r,0} - g_{p_{01},\text{re}} y_{r,0}$$
            $$g_{x_{r,2}} = g_{\bar{x}_r} + g_{p_{23},\text{re}} x_{r,3} + g_{p_{23},\text{im}} y_{r,3}, \qquad g_{y_{r,2}} = g_{\bar{y}_r} + g_{p_{23},\text{im}} x_{r,3} - g_{p_{23},\text{re}} y_{r,3}$$
            $$g_{x_{r,3}} = g_{\bar{x}_r} + g_{p_{23},\text{re}} x_{r,2} + g_{p_{23},\text{im}} y_{r,2}, \qquad g_{y_{r,3}} = g_{\bar{y}_r} + g_{p_{23},\text{im}} x_{r,2} - g_{p_{23},\text{re}} y_{r,2}$$
       5. Batched Tensor-Core GEMM pullback from $(g_x, g_y)$ to `d_phi_factors` and `d_psi_factors[v_0:v_1]`:
          - Stack `G_cat = torch.cat([g_x.reshape(K, N, V_c), g_y.reshape(K, N, V_c)], dim=1)` of shape `(K, 2 * N, V_c)` in `W_chunk.dtype`.
          - Compute `d_W_chunk = torch.bmm(U_batched.transpose(1, 2), G_cat)` of shape `(K, 32, V_c)` $\implies$ permute into `d_psi_factors[v_0:v_1]`!
          - Compute `d_U_chunk = torch.bmm(G_cat, W_chunk.transpose(1, 2)).float()` of shape `(K, 2 * N, 32)`:
            Since `U_batched[:, :N]` is `[\phi_{re}, \phi_{im}]` and `U_batched[:, N:]` is `[-\phi_{im}, \phi_{re}]`:
            $$\text{d\_phi\_re} \mathrel{+}= \text{d\_U\_chunk}[:, :N, :d_f] + \text{d\_U\_chunk}[:, N:, d_f:]$$
            $$\text{d\_phi\_im} \mathrel{+}= \text{d\_U\_chunk}[:, :N, d_f:] - \text{d\_U\_chunk}[:, N:, :d_f]$$
       6. Return `(d_phi_factors.to(phi_factors.dtype), d_psi_factors.to(psi_factors.dtype), d_branch_beta.to(branch_beta.dtype), d_branch_alpha.to(branch_alpha.dtype), None, None, None, None)`.

   - **Exact `eval_kronecker_multimeaning_born_probs` Function Signature (`<= 8` args)**:
     ```python
     def eval_kronecker_multimeaning_born_probs(
         phi_factors: torch.Tensor,
         psi_factors: torch.Tensor,
         branch_beta: torch.Tensor,
         branch_alpha: torch.Tensor,
         prior_weights: torch.Tensor,
         hyperparams: torch.Tensor,
     ) -> torch.Tensor:
     ```
     Evaluates the exact normalized probability distribution `probs` of shape `(*phi_factors.shape[:-3], V)` in `float32` using `torch.pow(I_tv, boson_p)` with zero `exp` or `softmax`.

5. **Complete `KroneckerMultiMeaning16x4GPT` Model Class Contract, Forward Dataflow, and Diagnostic Methods (`<= 8` args per method)**:
   - **Parameters & Buffers in `__init__(self, cfg: KroneckerMultiMeaningConfig)`**:
     - `self.embed_re = nn.Parameter(torch.empty(cfg.vocab_size, cfg.num_meanings, cfg.num_factors, cfg.factor_dim))` (`(50304, 2, 4, 16)`)
     - `self.embed_im = nn.Parameter(torch.empty(cfg.vocab_size, cfg.num_meanings, cfg.num_factors, cfg.factor_dim))` (`(50304, 2, 4, 16)`)
     - `self.branch_logits = nn.Parameter(torch.zeros(cfg.vocab_size, cfg.num_meanings))` (`(50304, 2)`)
     - `self.register_buffer("prior_weights", torch.ones(cfg.vocab_size, dtype=torch.float32))`
     - `self.register_buffer("unigram_probs", torch.full((cfg.vocab_size,), 1.0 / float(cfg.valid_vocab_size), dtype=torch.float32))`
     - `self.blocks = nn.ModuleList([GeminiComplexRegisterBlock(...) for _ in range(cfg.num_layers)])`
     - `self.kron_mixers = nn.ModuleList([Kronecker16x4UnitaryMeaningMixer(cfg) for _ in range(cfg.num_layers)])`
     - `self.out_factor_u_re = nn.Parameter(torch.empty(cfg.num_meanings * cfg.num_factors, cfg.factor_dim, cfg.factor_dim))` (`(8, 16, 16)` initialized as Haar complex unitaries)
     - `self.out_factor_u_im = nn.Parameter(torch.empty(cfg.num_meanings * cfg.num_factors, cfg.factor_dim, cfg.factor_dim))`
     - `self.context_branch_bias = nn.Parameter(torch.zeros(cfg.num_meanings, dtype=torch.float32))`
     - `self.psi0_readout_gate = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))`
   - **100% `exp`/`softmax`-Free Parameterization of `branch_alpha` and `branch_beta`**:
     - **Vocabulary Branch Amplitudes (`compute_branch_alpha(self) -> torch.Tensor` of shape `(V, R)`)**:
       Uses an algebraic rational positive map $f(u) = \frac{1}{2}\bigl(u + \sqrt{u^2 + 1}\bigr) + \varepsilon_{\text{branch}} > 0$ (which is strictly positive, smooth, and uses only `sqrt` with **zero `exp` or `softplus`**) followed by L2 unit-sphere normalization across `dim=-1`:
       $$a_{v,r}^+ = \frac{1}{2}\Bigl(\text{branch\_logits}[v, r] + \sqrt{\text{branch\_logits}[v, r]^2 + 1.0}\Bigr) + \varepsilon_{\text{branch}}, \qquad \alpha_{v,r} = \frac{a_{v,r}^+}{\sqrt{\sum_{r'=0}^{R-1} (a_{v,r'}^+)^2}}$$
       so $\sum_{r=0}^{R-1} \alpha_{v,r}^2 = 1.0$ (`branch_logits = 0` at initialization gives $\alpha_{v,r} = 1/\sqrt{R}$).
     - **Context Branch Amplitudes (`branch_beta` of shape `(N, R)`)**:
       Before normalizing the post-operator context factors `(raw_phi_re, raw_phi_im)` of shape `(N, R, M, d_f)` along `dim=-1`, computes each branch's pre-normalization RMS amplitude $e_{t,r} = \sqrt{\frac{1}{M}\sum_{m=0}^{M-1}\bigl(\|\text{raw\_phi\_re}[t,r,m]\|_2^2 + \|\text{raw\_phi\_im}[t,r,m]\|_2^2\bigr) + 10^{-8}}$, adds the algebraic positive bias $b_r^+ = \frac{1}{2}(\text{context\_branch\_bias}[r] + \sqrt{\text{context\_branch\_bias}[r]^2 + 1.0})$, and L2-normalizes across `dim=-1`:
       $$\tilde{\beta}_{t,r} = e_{t,r} \cdot b_r^+ + \varepsilon_{\text{branch}}, \qquad \beta_{t,r} = \frac{\tilde{\beta}_{t,r}}{\sqrt{\sum_{r'=0}^{R-1} \tilde{\beta}_{t,r'}^2}}, \qquad \sum_{r=0}^{R-1} \beta_{t,r}^2 = 1.0$$
   - **Exact `forward` Dataflow & Readout Equations (`<= 8` args)**:
     ```python
     def forward(
         self,
         idx: torch.Tensor,
         targets: torch.Tensor | None = None,
         return_logits: bool | None = None,
         token_weights: torch.Tensor | None = None,
     ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
     ```
     1. **Input Token Lookup & Packing (`(B, T, R, M, d_f) -> (B, T, 64 * R)`)**:
        - Looks up `raw_re = self.embed_re[idx]` and `raw_im = self.embed_im[idx]` of shape `(B, T, R, M, d_f)` (cast to `bfloat16` when CUDA autocast is enabled).
        - Normalizes each factor `(b, t, r, m)` to unit norm along `dim=-1` ($d_f = 16$) via `complex_l2_normalize(raw_re, raw_im, dim=-1)`.
        - Scales each branch $r$ by $\alpha_{\text{idx}, r} / \sqrt{M}$ (`alpha_tok.unsqueeze(-1).unsqueeze(-1) / math.sqrt(M)`) so that when reshaped to `(B, T, R * M * d_f)` (`d_complex_total = 64 * R`), the packed vector `(psi0_re, psi0_im)` has exact unit norm $\sum_{r=0}^{R-1} \sum_{m=0}^{M-1} (\alpha_{r}/\sqrt{M})^2 \|\hat{\psi}_{r}^{(m)}\|^2 = 1.0$.
        - Inside `GeminiComplexRegisterBlock`, `(B, T, 64 * R)` is viewed as `(B, T, n_registers, d_register)` (`n_registers = 2 * R, d_register = 32`); inside `Kronecker16x4UnitaryMeaningMixer`, `(B, T, 64 * R)` is viewed as `(B, T, R, M, d_f)` (`(B, T, R, 4, 16)`).
     2. **Layer Loop**:
        - For each layer $l \in \{0, \dots, L-1\}$, runs `psi_re, psi_im, v0_re, v0_im = self.blocks[l](psi_re, psi_im, v0_re, v0_im)` followed by `psi_re, psi_im = self.kron_mixers[l](psi_re, psi_im)`.
     3. **Sequence Readout State (`phi_factors` of shape `(N, R, M, 2 * d_f)`)**:
        - Applies the conjugate skip and self-echo removal with `(psi0_re, psi0_im)` gated by `g0 = torch.tanh(self.psi0_readout_gate)` (identical to `Wave 40`), then reshapes `(psi_re, psi_im)` to `(N, K, d_f)` where $N = B \cdot T$ and $K = R \cdot M$.
        - Applies the last-layer batched $16 \times 16$ unitary operator `(self.out_factor_u_re, self.out_factor_u_im)` of shape `(K, d_f, d_f)`:
          ```python
          raw_phi_re = torch.einsum("nkd,kdo->nko", q_re, u_out_re) - torch.einsum("nkd,kdo->nko", q_im, u_out_im)
          raw_phi_im = torch.einsum("nkd,kdo->nko", q_re, u_out_im) + torch.einsum("nkd,kdo->nko", q_im, u_out_re)
          ```
        - Reshapes `(raw_phi_re, raw_phi_im)` to `(N, R, M, d_f)`, computes `branch_beta` of shape `(N, R)` from the pre-normalization branch norms + `self.context_branch_bias`, normalizes each factor along `dim=-1` ($d_f = 16$) via `phi_re, phi_im = complex_l2_normalize(raw_phi_re, raw_phi_im, dim=-1)`, and concatenates `phi_factors = torch.cat([phi_re, phi_im], dim=-1)` of shape `(N, R, M, 2 * d_f)`.
     4. **Tied Vocabulary Readout State (`psi_factors` of shape `(V, R, M, 2 * d_f)`)**:
        - For the active vocabulary $V = \text{valid\_vocab\_size} = 50,257$, normalizes `e_re, e_im = complex_l2_normalize(self.embed_re[:V], self.embed_im[:V], dim=-1)` along `dim=-1` ($d_f = 16$) and concatenates `psi_factors = torch.cat([e_re, e_im], dim=-1)` of shape `(V, R, M, 2 * d_f)`.
     5. **`return_logits` Contract**:
        - When `want_logits` is `False` (`targets is not None and not return_logits`), calls `loss = KroneckerMultiMeaningBornChunkedNLLFn.apply(phi_factors, psi_factors, branch_beta, branch_alpha, valid_targets, self.prior_weights[:V], hyperparams, token_weights)` and returns `(loss if return_logits is None else None, loss)`.
        - When `want_logits` is `True` (`targets is None` or `return_logits=True`), calls `probs = eval_kronecker_multimeaning_born_probs(phi_factors.reshape(B, T, R, M, 2 * d_f), psi_factors, branch_beta.reshape(B, T, R), branch_alpha, self.prior_weights[:V], hyperparams)` of shape `(B, T, V)` (`V = valid_vocab_size = 50,257`), sets `log_probs = torch.log(probs.clamp_min(1e-12))` of shape `(B, T, V)`, and returns `(log_probs, loss)`.
   - **Initialization, 100M-Token Unigram Prior, and Diagnostic Methods (`<= 8` args each)**:
     - `initialize_unigram_prior_from_tokens(self, token_stream: np.ndarray | torch.Tensor, floor_count: float = DEFAULT_UNIGRAM_FLOOR_COUNT) -> dict[str, float]`
     - `load_100m_unigram_prior(self, data_dir: str | Path = ".cache/fineweb_100M", max_tokens: int = 100_000_000) -> dict[str, float]`
     - `initialize_isotropic_kronecker(self, seed: int = 1337) -> dict[str, float]`
     - `compute_sharp_eff_rank(self) -> float`: Flattens the normalized active vocabulary factors `psi_factors` `(V, R, M, 2 * d_f)` into `E_flat` of shape `(V, 2 * d_complex_total)` (`(50257, 256)` at $R = 2$), centers across `dim=0`, computes covariance $\Sigma = \frac{1}{V} E_{\text{cent}}^\top E_{\text{cent}}$, and returns participation ratio $\frac{(\operatorname{Tr}\Sigma)^2}{\|\Sigma\|_F^2} \in [1, 256]$.
     - `compute_mean_factor_rank(self) -> float`: For each of the $4 R$ factor tables $\tilde{\Psi}_r^{(m)} \in \mathbb{R}^{V \times 32}$, centers across `dim=0`, computes $\Sigma_{r,m} = \frac{1}{V} \tilde{\Psi}_{r,\text{cent}}^{(m)\top} \tilde{\Psi}_{r,\text{cent}}^{(m)} \in \mathbb{R}^{32 \times 32}$ and participation ratio $R_{r,m} = \frac{(\operatorname{Tr}\Sigma_{r,m})^2}{\|\Sigma_{r,m}\|_F^2} \in [1, 32]$, and returns the mean $\frac{1}{4R}\sum_{r,m} R_{r,m}$.
     - `compute_cross_branch_overlap(self) -> float`: Returns the mean 4-factor Kronecker overlap $\frac{1}{V \binom{R}{2}}\sum_{v=0}^{V-1}\sum_{0 \le r < r' < R} \prod_{m=0}^3 |\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\psi}_{v,r'}^{(m)} \rangle_{\mathbb{C}^{16}}|^2$ in `float` (`0.0` when $R = 1$).

6. **Soft Kronecker Manifold, Unitary, Per-Factor Parseval Isotropy, & Cross-Meaning Branch Orthogonality Penalties (`compute_kronecker_manifold_and_isotropy_penalties`)**:
   ```python
   def compute_kronecker_manifold_and_isotropy_penalties(
       model: KroneckerMultiMeaning16x4GPT,
   ) -> tuple[torch.Tensor, dict[str, float]]:
   ```
   - Computes:
     1. `sphere_pen`: Mean of $(\|\psi_{v,r,\text{re}}^{(m)}\|_2^2 + \|\psi_{v,r,\text{im}}^{(m)}\|_2^2 - 1.0)^2$ over active vocabulary $v < V$, branches $r < R$, and factors $m < 4$ (also logging `embed_norm_median` and `embed_norm_max` over the $V \cdot R \cdot 4$ factor kets).
     2. `unitary_raw_sum`: Un-attenuated sum of $\frac{1}{d_W}(\|W_{\text{re}} W_{\text{re}}^\top + W_{\text{im}} W_{\text{im}}^\top - I_{d_W}\|_F^2 + \|W_{\text{im}} W_{\text{re}}^\top - W_{\text{re}} W_{\text{im}}^\top\|_F^2)$ across:
        - all square `ComplexLinear` modules (`in_features == out_features`),
        - all $16 \times 16$ factor unitaries (`factor_u_re/im` of shape `(K, 16, 16)` and `out_factor_u_re/im` of shape `(K, 16, 16)`), and
        - the $4 \times 4$ `factor_mix_re/im` `(R, 4, 4)` and $R \times R$ `meaning_mix_re/im` `(4, R, R)` mixers inside each `Kronecker16x4UnitaryMeaningMixer` (while excluding rectangular FFN `ComplexLinear` matrices), logging `complex_unitary_err` as the mean of $\sqrt{\text{term}_W}$.
     3. `iso_pen`: Mean over all $4 R$ normalized factor tables $\tilde{\Psi}_{r}^{(m)} \in \mathbb{R}^{V \times 32}$ of $32 \|\frac{1}{V} \tilde{\Psi}_{r,\text{cent}}^{(m)\top} \tilde{\Psi}_{r,\text{cent}}^{(m)} - \frac{1}{32} I_{32}\|_F^2$.
     4. `branch_orth_pen`: For $R \ge 2$, mean over $v < V$ and pairs $0 \le r < r' < R$ of the 4-factor Kronecker branch overlap $\prod_{m=0}^3 |\langle \hat{\psi}_{v,r}^{(m)} \mid \hat{\psi}_{v,r'}^{(m)} \rangle_{\mathbb{C}^{16}}|^2$ (for $R = 1$, `branch_orth_pen = 0.0`).
   - Combines `total_penalty = cfg.sphere_penalty_weight * sphere_pen + cfg.unitary_penalty_weight * unitary_raw_sum + cfg.iso_penalty_weight * iso_pen + cfg.branch_orth_weight * branch_orth_pen`.
   - Returns `(total_penalty, {"sphere_pen": ..., "unitary_pen": ..., "iso_pen": ..., "branch_orth_pen": ..., "embed_norm_median": ..., "embed_norm_max": ..., "complex_unitary_err": ...})`.

7. **Optimizer Builder (`build_kronecker_multimeaning_optimizers`, `<= 8` args)**:
   ```python
   def build_kronecker_multimeaning_optimizers(
       model: KroneckerMultiMeaning16x4GPT,
       base_muon_lr: float = DEFAULT_MUON_LR,
       base_embed_lr: float = DEFAULT_EMBED_LR,
       base_gate_lr: float = DEFAULT_GATE_LR,
       weight_decay: float = 0.0,
   ) -> tuple[StiefelMuon, torch.optim.AdamW]:
   ```
   - Routes all parameters with `p.ndim in (2, 3)` and `min(p.shape[-2:]) >= 16` (excluding `embed_re, embed_im, branch_logits`)—specifically all 2D `ComplexLinear` weights and the 3D `(K, 16, 16)` batched factor unitaries (`factor_u_re, factor_u_im, out_factor_u_re, out_factor_u_im`)—to `StiefelMuon(..., stiefel_penalty=0.0, use_normuon=True, use_twin_rail=True)`.
   - Routes `[model.embed_re, model.embed_im]` to `AdamW` group 0 (`lr=base_embed_lr, weight_decay=0.0, betas=(0.9, 0.95)`).
   - Routes all remaining parameters—including `model.branch_logits`, `model.context_branch_bias`, `model.psi0_readout_gate`, the small 3D mixers `factor_mix_re/im` `(R, 4, 4)` and `meaning_mix_re/im` `(4, R, R)` (where `min(p.shape[-2:]) < 16`), `mixer_gate`, and block scalar/1D gates—to `AdamW` group 1 (`lr=base_gate_lr, weight_decay=weight_decay, betas=(0.9, 0.95)`).

### 3.2 Export in [`src/qdlblocks/models/__init__.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/__init__.py) and Document in [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md)
- Export `KroneckerMultiMeaningConfig`, `Kronecker16x4UnitaryMeaningMixer`, `KroneckerMultiMeaning16x4GPT`, `KroneckerMultiMeaningBornChunkedNLLFn`, `eval_kronecker_multimeaning_born_probs`, `compute_kronecker_manifold_and_isotropy_penalties`, and `build_kronecker_multimeaning_optimizers` in `src/qdlblocks/models/__init__.py`.
- Document the `Wave 41` building blocks and CLI profiler in [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md).

---

## 4. Testing & Verification Plan

### 4.1 Unit Tests: [`tests/test_plan056_kronecker_multimeaning_16x4.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan056_kronecker_multimeaning_16x4.py)
1. `test_h1_kronecker_16x4_snr_and_polysemy_capacity`: Verifies that random unit states in $(\mathbb{C}^{16})^{\otimes 4}$ have expected 4-factor squared overlap $\approx 1/65,536$ (`>= 400x` lower background noise than flat $\mathbb{C}^{128}$), and that Rank-$2$ / Rank-$4$ Kronecker states achieve `>= 6.0x` higher 4-factor coincidence on 2 orthogonal word senses than Rank-$1$.
2. `test_h1_kronecker_chunked_born_autograd_parity`: Verifies that `eval_kronecker_multimeaning_born_probs` sums to `1.0` (`max_err < 1e-5`) with zero `exp`/`softmax` calls, and that `KroneckerMultiMeaningBornChunkedNLLFn` (`vocab_chunk_size = 64` on a synthetic $V = 256$ vocabulary) matches the unchunked `float32` reference within `< 1e-5` forward loss error and `< 1e-4` backward gradient error on all 4 differentiable inputs (`phi_factors`, `psi_factors`, `branch_beta`, `branch_alpha`) including non-uniform `token_weights`.
3. `test_h2_unigram_100m_prior_loading_and_kl_advantage`: Verifies header-safe token counting and positive floor smoothing (`floor_count = 1.0`) in `initialize_unigram_prior_from_tokens` / `load_100m_unigram_prior`, and confirms lower KL divergence and Step-0 Born NLL on validation tokens than uniform prior.
4. `test_h3_kronecker_penalties_and_branch_orthogonality`: Verifies `compute_kronecker_manifold_and_isotropy_penalties` (`sphere_pen`, `unitary_pen`, `iso_pen`, `branch_orth_pen`), confirms rectangular FFN layers are excluded from `unitary_pen`, and verifies that gradient descent on `branch_orth_pen` drives cross-branch 4-factor overlap `< 1e-3`.
5. `test_h4_end_to_end_bf16_smoke_and_rank_retention`: Runs a 5-step `bfloat16` training loop of `KroneckerMultiMeaning16x4GPT` (`R = 2, M = 4, d_f = 16, L = 2`) verifying zero `float32` activation leaks inside blocks, decreasing loss, and factor rank retention.

### 4.2 End-to-End Verification Runner: [`examples/66_verify_plan056_kronecker_multimeaning_16x4.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/66_verify_plan056_kronecker_multimeaning_16x4.py)
- Executes all four hypothesis audits (`H1`–`H4`), including a 5-minute (`300s`) live training run of `KroneckerMultiMeaning16x4GPT` (`R = 2, M = 4, d_f = 16, L = 6`) on `FineWeb` with `StandardizedExperimentLogger`, writing `.cache/hypothesis_experiments_056.json` and syncing to the WebUI (`:8765`).

### 4.3 Regression & AST Linter Gate
- Run `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python` on `tests/test_plan054_*.py`, `tests/test_plan055_*.py`, and `tests/test_plan056_*.py`, and run `scripts/ast_signature_and_length_linter.py` on all created/modified files (`0` violations).

---

## 5. Operational Tooling & Script Opportunities

### 5.1 Existing Scripts Used
1. **AST Signature & SLOC Linter**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py scripts/kronecker_16x4_meaning_and_snr_profiler.py examples/66_verify_plan056_kronecker_multimeaning_16x4.py tests/test_plan056_kronecker_multimeaning_16x4.py
   ```
2. **WebUI ModelCard & SQLite Sync**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload
   ```
3. **Live Experiment Monitor Snapshot**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once
   ```

### 5.2 `Build-Before-Launch` Script (Implemented in This Plan)
- [`scripts/kronecker_16x4_meaning_and_snr_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/kronecker_16x4_meaning_and_snr_profiler.py):
  - Standalone CLI diagnostic tool that:
    1. Benchmarks empirical random background overlap $\mathbb{E}_{u \ne v}[|\mathcal{K}_{u,v,r}|^2]$ (`~1/65,536`) vs. flat $\mathbb{C}^{128}$ (`1/128`),
    2. Profiles $K$-sense polysemy target coincidence across meaning ranks $R \in \{1, 2, 4\}$,
    3. Audits the 100M-token FineWeb unigram distribution vs. 2M-slice and uniform priors, and
    4. Benchmarks throughput (`tok/s`) and peak VRAM (`MiB`) of the single-`torch.bmm` chunked Kronecker Born readout across $R \in \{1, 2, 4\}$, writing `.cache/kronecker_16x4_meaning_and_snr_profile.json`.
  - Invocation:
    ```bash
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/kronecker_16x4_meaning_and_snr_profiler.py --ranks 1,2,4 --json-out .cache/kronecker_16x4_meaning_and_snr_profile.json
    ```

### 5.3 `Nice-to-Have` Scripts (Logged in [`scripts/TOOLING_BACKLOG.md`](file:///home/festivity/agy/QuantumDLBlocks/scripts/TOOLING_BACKLOG.md) & Surfaced to User)
1. `scripts/polysemy_branch_sense_disambiguation_inspector.py`: Inspects a trained `KroneckerMultiMeaning16x4GPT` checkpoint on polysemous probe sentences (e.g., `"river bank"` vs. `"central bank"`, `"apple tree"` vs. `"apple macbook"`) and visualizes which meaning branch $r \in \{0, \dots, R-1\}$ activates ($\beta_{t,r}$) and which top-10 vocabulary tokens have the highest 4-factor Kronecker overlap with each branch $\lvert \hat{\psi}_{v,r}^{(0..3)} \rangle$.
2. `scripts/kronecker_factor_entanglement_and_mutual_info_probe.py`: Measures per-factor participation ratios $R_{\text{factor}}^{(m)}$ ($m \in \{0, 1, 2, 3\}$), cross-factor correlation matrices, and quantum Schmidt entanglement entropy across the $16 \times 16 \times 16 \times 16$ hierarchy during training.

---

## 6. Experiment Outcomes, Constants Ledger, & Post-Mortem Analysis (`Wave 41`)

### 6.1 Hypothesis Verification Summary (`.cache/hypothesis_experiments_056.json` & `.cache/kronecker_16x4_meaning_and_snr_profile.json`)
- **All 4 Hypotheses (`H1`–`H4`) PASSED (`all_passed = True`)**:
  - **`H1` (`PASSED`) — $16^4 = 65,536$ Kronecker Coincidence SNR, Multi-Meaning Polysemy Capacity, & Chunked Autograd Parity**:
    - Random background collision floor dropped from `0.007807` (flat $\mathbb{C}^{128}$, theoretical $1/128 = 0.0078125$) to **`1.540e-5`** (`(C^16)^{x4} ~= C^{65,536}`, theoretical $1/65,536 = 1.5259\times 10^{-5}$), achieving a **`506.83x`** (`509.98x` in profiler) single-branch SNR improvement (`>= 400x` target) at unit self-overlap `1.000000`.
    - Across the 5-level Kronecker hierarchy (`1 -> 16 -> 256 -> 4,096 -> 65,536`), empirical background squared overlap decayed geometrically: `level_1 (16): 0.06253` $\to$ `level_2 (256): 0.003912` $\to$ `level_3 (4096): 0.0002453` $\to$ `level_4 (65536): 1.533e-5`.
    - For 2-sense polysemous tokens, `kronecker_16x4_rank2` and `kronecker_16x4_rank4` achieved `0.5000` coincidence energy vs. `0.0680` (`0.0625` theoretical) for `kronecker_16x4_rank1` (`7.35x`–`8.00x >= 6.0x` gain; `16.0x` unweighted branch-matched gain).
    - `eval_kronecker_multimeaning_born_probs` and `KroneckerMultiMeaningBornChunkedNLLFn` executed with **zero `exp`, `softmax`, or `logsumexp` calls**, `max_prob_sum_err = 1.19e-7 < 1e-5`, forward loss parity `0.0 < 1e-5`, and backward gradient parity `< 1.12e-8 < 1e-4` across all 4 differentiable inputs (`phi_factors`, `psi_factors`, `branch_beta`, `branch_alpha`).
  - **`H2` (`PASSED`) — `100M`-Token FineWeb Unigram Prior vs. `2M`-Slice & Uniform Priors**:
    - Counting the full `100M`-token FineWeb Shard 1 (`train_full_100M.bin`) increased observed vocabulary coverage from `43,755` (`slice_2m`, `6,502` unseen tokens) to **`49,850` observed tokens (`407` unseen tokens floored at `1.0`)** — a **`15.98x`** reduction in unseen tokens (`> 15x` target).
    - Validation KL divergence $D_{\text{KL}}(P_{\text{val}} \,\|\, P_{\text{uni}})$ dropped from `3.1902` nats (`uniform_prior`) and `0.0997` nats (`slice_2m_prior`) to **`0.0678` nats** (`slice_100m_prior`).
    - Step-0 validation Born NLL dropped from `10.9691` nats (`uniform_prior`) to **`8.0125` nats** (`< 8.20` nats target), and after 150 steps `slice_100m_prior` achieved the lowest validation NLL (`7.4015` vs. `7.4162` for `slice_2m_prior` and `7.6606` for `uniform_prior`).
  - **`H3` (`PASSED`) — Relaxed Per-Factor + Cross-Factor Parseval Isotropy (`lambda_iso = 0.05`) & Cross-Meaning Branch Orthogonality (`lambda_branch = 0.20`)**:
    - `relaxed_iso_with_branch_orth` achieved lower validation NLL (`7.3946` vs. `7.4140` nats) and higher Top-1 accuracy (`9.31%` vs. `9.08%`) than `overconstrained_iso_no_branch_orth` while preserving per-factor participation ratio $\bar{R}_{\text{factor}} = 31.86 / 32.0$ (`>= 20.0`), global embedding participation ratio $R_{\text{emb}} = 246.63 / 256.0$ (`>= 140.0`), cross-branch overlap $\bar{O}_{\text{branch}} = 1.16 \times 10^{-5} < 1.0 \times 10^{-3}$, median factor norm `1.0003` (`in [0.95, 1.15]`), and unitary drift `0.0615 < 0.25`.
  - **`H4` (`PASSED`) — 5-Minute (`300s`) Live `RTX 5060 Ti` Training Verification (`w41_kronecker_16x4_R2_M4_df16_L6_300s`)**:
    - Completed `224` steps (`1.84M` tokens at **`6,116.7 tok/s`**, `14.98M` parameters, `B = 16, T = 512, L = 6`) in `300.0s` with `nan_inf_events == 0`, `float32_activation_leak_count == 0`, and `peak_vram_mib = 12,665.5 MiB < 14,500 MiB`.
    - Validation Born NLL decreased monotonically across every evaluation checkpoint:
      - `Step 0 (0.0s)`: `Val CE = 8.0267` | `Top-1 = 3.44%` | `Top-50 = 33.31%` | `R_factor = 31.98/32` | `R_emb = 254.71/256` | `O_branch = 1.53e-5`
      - `Step 56 (76.2s, 0.46M tok)`: `Val CE = 7.5689` | `Top-1 = 8.94%` | `R_factor = 31.90/32` | `R_emb = 249.36/256` | `O_branch = 1.42e-5`
      - `Step 112 (150.9s, 0.92M tok)`: `Val CE = 7.3654` | `Top-1 = 9.77%` | `R_factor = 31.93/32` | `R_emb = 252.17/256` | `O_branch = 1.30e-5`
      - `Step 168 (225.4s, 1.38M tok)`: `Val CE = 7.2589` | `Top-1 = 10.65%` | `R_factor = 31.96/32` | `R_emb = 253.43/256` | `O_branch = 1.27e-5`
      - `Step 224 (300.0s, 1.84M tok)`: **`Val CE = 7.2066` (`PPL = 1348.24`) | `Top-1 = 11.06%` | `Top-50 = 42.38%` | `R_factor = 31.97/32` | `R_emb = 253.94/256` | `O_branch = 1.28e-5` | `U_err = 0.0360`**
    - Compared against the `Wave 40` 300s baseline (`Val CE = 8.7652, Top-1 = 4.19%, Top-50 = 26.82%`), `Wave 41` improved 5-minute validation NLL by **`-1.5586` nats** (`8.7652 -> 7.2066`), increased Top-1 accuracy by **`2.64x`** (`4.19% -> 11.06%`), and increased Top-50 accuracy from `26.82%` to **`42.38%`** while maintaining `9.41x` higher effective rank than `Wave 39` (`253.94` vs. `27.0`).

### 6.2 Rule 7 Constants & Calibration Ledger (`src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py`)
| Constant Name | Value | Provenance & Calibration Rationale |
| :--- | :---: | :--- |
| `DEFAULT_NUM_FACTORS` | `4` | 4 Kronecker tensor-product factors $(\mathbb{C}^{16})^{\otimes 4} \cong \mathbb{C}^{65,536}$. |
| `DEFAULT_FACTOR_DIM` | `16` | Complex dimension per Kronecker factor ($16^4 = 65,536 > V = 50,257$). |
| `DEFAULT_NUM_MEANINGS` | `2` | Default number of entangled meaning branches $R$ per vocabulary token. |
| `DEFAULT_KRON_BOSON_P` | `11.0` | Polynomial Born readout power $p$ (`torch.pow(I_tv, boson_p)` with zero `exp`/`softmax`). |
| `DEFAULT_KRON_HOLO_ALPHA` | `0.85` | Holographic modulation scale $\alpha$ against reference wave $e^{i\pi/4}$. |
| `DEFAULT_KRON_BLEND` | `0.55` | Multi-scale blend $\eta$ between 1-factor linear overlap $(1-\eta)\bar{z}$ and 4-factor Kronecker product $\eta\mathcal{K}$. |
| `DEFAULT_KRON_COINC_GAMMA` | `0.45` | 8th-degree 4-factor coincidence energy gain $\gamma_c \prod_{m=0}^3 |z^{(m)}|^2$. |
| `DEFAULT_UNIGRAM_TEMP` | `0.85` | Calibrated from initial `0.55` to `0.85` during `H2` profiler audit so the unigram prior matches FineWeb's Zipf slope and achieves Step-0 validation Born NLL `8.0125 < 8.20` nats (`-2.96` nats better than uniform prior). |
| `DEFAULT_VOCAB_CHUNK_SIZE` | `8192` | Calibrated from `4096` to `8192` (`7` chunks over $V = 50,257$) paired with `@torch.compile` fused `_compiled_kronecker_chunk_fwd_step` / `_compiled_kronecker_chunk_bwd_step`, accelerating `R=2` readout throughput by `5.06x` (`1,775.9 -> 8,983.4 tok/s`). |
| `DEFAULT_ISO_PENALTY_WEIGHT` | `0.05` | Relaxed Parseval isotropy weight combining per-factor $32 \times 32$ isotropy and cross-factor $256 \times 256$ decorrelation. |
| `DEFAULT_BRANCH_ORTH_WEIGHT` | `0.20` | 4-factor Kronecker cross-meaning branch orthogonality penalty weight ($\bar{O}_{\text{branch}} \approx 1.28 \times 10^{-5}$). |

### 6.3 Key Mathematical & Engineering Insights for Next Waves
1. **Joint Per-Factor + Cross-Factor Parseval Decorrelation**:
   - Penalizing only the diagonal $32 \times 32$ per-factor covariance blocks keeps individual factor tables isotropic ($\bar{R}_{\text{factor}} = 29.6 / 32$) but allows the 1-factor linear gradient highway $(1 - \eta)\bar{z}$ to correlate the $M = 4$ factors with one another, reducing the global 256D participation ratio $R_{\text{emb}}$ toward $256 / 4 \approx 64–115$. Including the off-diagonal cross-factor blocks via $32\|\Sigma_{\text{global}} - \frac{1}{32}I_{256}\|_F^2$ (`0.05 ms` overhead per step) keeps the 4 Kronecker factors mutually independent ($R_{\text{emb}} = 253.94 / 256.0$) and improves both `val_ce` and `top1_acc`.
2. **`@torch.compile` Register Fusion of the 2-Level Binary Complex Product Tree**:
   - Passing the batched GEMM output `z_cat = torch.bmm(u_batched, w_chunk)` in `bfloat16` directly into `@torch.compile` fused forward/backward helpers (`_compiled_kronecker_chunk_fwd_step` and `_compiled_kronecker_chunk_bwd_step`) and replacing `scatter_add_` with pointwise broadcast comparison `is_target = (v_idx == targets_col).float()` eliminates ~35 intermediate `(R, N, V_c)` `float32` HBM allocations per chunk, boosting `R = 2` readout throughput by `5.06x` and cutting peak VRAM by `30%`.

