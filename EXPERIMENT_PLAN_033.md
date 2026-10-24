# Experiment Plan 033 — Repository Script Best Practices, Consistently $N$-Dimensional ($O(N)$) Quantum Architectures, Soft Power-Law Spectral Floor ($R_{\text{diff}} \ge 6.5$), Local `8 GB` 3,600-Step (`29.5M`-Token) Scaling Study, Backlog CLI Promotion, & 2-Session Colab G4 Wave 18 (`900M`-Token) Launch

## 1. Executive Summary & Architectural Audit

### 1.1 Why Giant Scripts (`3,851` Lines) & `92–101`-Argument Functions Create Agent Confusion & Silent Drift
An audit of the repository's largest files and function signatures revealed the exact structural issue identified in code review:
- **[`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py)** is **`3,851` lines**, and its main entry function [`run_cuda_benchmark`](file:///home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py#L752-L854) takes **`101` named arguments** (grown from `~86` arguments across recent waves), most of which are boolean/scalar feature flags (`overcomplete_frame`, `overcomplete_two_beam`, `sparse_64qubit`, `hierarchical_16qubit`, `last_layer_softmax`, `two_beam_kboson_readout`, `bipolar_kboson`, etc.) selecting branches inside a single `2,100`-line nested function.
- **[`QuantumBornGPT.__init__`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/born_gpt.py#L47-L141)** in [`src/qdlblocks/models/born_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/born_gpt.py) takes **`92–93` named arguments**.
- **Per-Plan Verification Scripts (`examples/35_*.py` through [`examples/42_verify_plan032_clean_full_ambient_and_gentle_qubit_lr.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/42_verify_plan032_clean_full_ambient_and_gentle_qubit_lr.py))** are **`1,300–1,657` lines** each because they copy-paste optimizer construction, log-prior initialization, validation evaluation, and SQLite/Model-Card registration.

**Concrete Bug Caught by Auditing `examples/42_*.py` vs. `examples/11_*.py`**:
Because local verification scripts (`examples/42_*.py`) and the remote Colab runner (`examples/11_*.py`) maintained separate optimizer/initialization code paths:
1. `examples/42_*.py` initialized `model.povm_log_prior` with **Bose-saturated empirical log-unigram frequencies** (`alpha = 0.62`, top-16 knee saturation) and routed **`value_embeds`, `unigram_qubits`, and `higher_states`** into the dedicated **Gentle `2x` Zero-Weight-Decay Qubit Embedding `AdamW` group (`lr = 0.006, weight_decay = 0.0`)**.
2. In `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` (lines 2506–2535), `value_embeds` was omitted from `sparse_embed_ids` (falling into `adamw_2d` with `weight_decay = 0.01`), `sparse_lr` was `0.010` (`2.5 * 0.004`) instead of `0.006` (`2.0 * 0.003`), and Bose-saturated `initialize_povm_log_prior` was not called!

In **Plan 033**, we codify enforceable **Repository & Script Best Practices** in [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md) and [`AGENTS.md`](file:///home/festivity/agy/QuantumDLBlocks/AGENTS.md), introduce typed `@dataclass` configuration objects and shared optimizer/initialization builders in [`src/qdlblocks/benchmarks/config_models.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py), align `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` with Plan 032's exact Gentle `2x` Zero-WD Qubit LR + Bose-saturated log-prior, and ensure [`examples/43_verify_plan033_scaling_and_soft_spectral_floor.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/43_verify_plan033_scaling_and_soft_spectral_floor.py) stays strictly **`< 400` lines** with **`<= 8` arguments per function**.

---

## 2. Consistently $N$-Dimensional ("Equi-Dimensional $O(N)$") Quantum Architectures

### 2.1 Where Current Models Leave the $N$-Dimensional Space
In standard `QuantumBornGPT`, a token state nominally lives on $S^{N-1} \subset \mathbb{R}^N$, but four internal operations repeatedly change the dimension of the state space:
1. **Seed Bottleneck ($d_{\text{seed}} < N$)**: Legacy `IsometricHilbertLift` embedded tokens in $\mathbb{R}^{d_{\text{seed}}}$ ($128$ or $256$) before lifting to $\mathbb{R}^N$.
2. **Parity-Split Subspaces ($N/2 \oplus N/2$)**: When `use_ortho_decouple=True` (`16×16 = 256` in $N = 512$), Step-0 input states live in the $N/2$-dimensional even coordinate subspace (`0::2`) while readout detectors live in the $N/2$-dimensional odd coordinate subspace (`1::2`).
3. **Overcomplete Frame Expansion ($N \to 2N\text{ or }4N \to N$)**: `OvercompleteTwoBeamInterference` (`frame_expansion = 2.0` or `4.0`) lifts $|\psi_t\rangle \in S^{N-1}$ into two rectangular frame vectors $z_1, z_2 \in \mathbb{R}^{kN}$ ($k \in \{2, 4\}$) and projects back to $\mathbb{R}^N$ at every layer, consuming `~65–75%` of backbone FLOPs and parameters.
4. **Low-Rank Kraus Readout Bottleneck ($N \to r = 16$)**: At readout, the second beam $|\psi_{\text{beam2}, t}\rangle \in S^{N-1}$ is projected into a low-rank subspace $\mathbb{R}^{16}$ via $V_{\text{diff}} \in \text{St}(N, 16)$.

### 2.2 Four Equi-Dimensional ($N$-D All the Way Through) Quantum Architecture Designs
To model language as **consistently an $N$-dimensional quantum state space from token input to POVM readout**, we formulate four progressive designs:

1. **Design A — Strict Square-$O(N)$ Equi-Dimensional Two-Beam Network (`frame_expansion = 1.0`, Full-Ambient $J_{\text{anti}}$ $Q \times A = N$, `seed_dim = N`)**:
   - **Input State**: Full-ambient product registers ($Q \times A = N$, e.g. $16 \times 16 = 256$ for $N=256$ or $16 \times 32 = 512$ for $N=512$) produce $|\psi_t^{(0)}\rangle \in S^{N-1}$ occupying all $N$ coordinates of $\mathbb{R}^N$, decoupled from output POVM registers via the Anti-Diagonal Spectral-Reversal Isometry $J_{\text{anti}} \in O(A)$.
   - **Attention Sublayer**: `SubspaceProjectorBornAttention` maps $S^{N-1} \to S^{N-1}$ via square orthogonal matrices $W_Q, W_K, W_V, W_O \in O(N)$ followed by Stereographic Radial-Phase Lift onto $S^{N-1}$.
   - **Equi-Dimensional Two-Beam Interference (`frame_expansion = 1.0`)**: Instead of lifting $\mathbb{R}^N \to \mathbb{R}^{4N}$, both interference beams evolve via **square $N \times N$ orthogonal operators** $U_{\ell, 1}, U_{\ell, 2}, U_{\ell, \text{out}} \in O(N)$:
     $$z_1 = \sqrt{N}\, U_{\ell, 1} |\psi_t\rangle \in \mathbb{R}^N, \qquad z_2 = \sqrt{N}\, U_{\ell, 2} |\psi_t\rangle \in \mathbb{R}^N$$
     modulated by an $N$-dimensional quadrature phase gate $\Delta\theta = \omega \odot (z_1 - z_2) + \phi \in \mathbb{R}^N$ and projected back by $U_{\ell, \text{out}} \in O(N)$ onto orthogonal unit kets $(|\psi_{\text{main}}\rangle, |\psi_{\text{beam2}}\rangle) \in S^{N-1} \times S^{N-1}$.
   - **Why It Matters**: Every intermediate representation in every layer has exact dimension $N$, and square $N \times N$ `PolarExpress` Stiefel retractions cut MLP parameter count and retraction FLOPs by **`2x–4x`**!

2. **Design B — Complex Unitary $U(N) \cong \text{Sp}(2N) \cap O(2N)$ Persistent Coherent State ($|\Psi_t\rangle = |\psi_{\text{re}, t}\rangle + i |\psi_{\text{im}, t}\rangle \in \mathbb{C}^N$)**:
   - Treats the two orthogonal real $N$-vectors $(|\psi_{t,1}\rangle, |\psi_{t,2}\rangle)$ as the real and imaginary parts of a single persistent $N$-dimensional complex ket $|\Psi_t\rangle \in \mathbb{C}^N$ ($\|\Psi_t\|_2 = 1$), initialized at Step 0 with $\text{Re}|\Psi_0(v)\rangle = u_{\text{in}}(v)$ and $\text{Im}|\Psi_0(v)\rangle = J_{\text{anti}}(u_{\text{in}}(v))$.

3. **Design C — Full-$N$ Spectral-Envelope Kraus Readout (Replacing $r = 16$ Truncation with $U_{\text{diff}}, V_{\text{diff}} \in O(N)$ and Zipfian Envelope $g \in \mathbb{R}_+^N$)**:
   - Keeps the diffuse semantic channel in $\mathbb{R}^N$ via a learnable non-negative spectral filter $g \in \mathbb{R}_+^N$ bridging $U_{\text{diff}}, V_{\text{diff}} \in O(N)$.

4. **Design D — Contextual Householder Reflection Chains ($H(v_k) = I_N - 2 \hat{v}_k \hat{v}_k^\top \in O(N)$)**:
   - Parameterizes each $N \times N$ unitary transition as a product of $K$ contextual Householder reflections on $\mathbb{R}^N$, guaranteeing exact isometry ($\text{IsoErr} = 0$ identically) in $O(K N)$ time and memory.

In **Plan 033**, we directly benchmark **Design A (`frame_expansion = 1.0` Square-$O(N)$ Equi-Dimensional `QuantumBornGPT`)** on genuine FineWeb-Edu (`L=4, d=256`, `900` steps) alongside our `frame_expansion = 2.0` models!

---

## 3. Mathematical Formulation of the Soft Power-Law Spectral Floor (`kraus_spectral_floor = 6.5`)

In Plan 031, hard Cholesky whitening (`R_diff = 16.00`) regressed validation cross-entropy by `+0.13` to `+0.31 nats` because it flattened the natural Zipfian singular-value spectrum of the 16-mode Kraus bath on every step, amplifying noisy tail modes `10–16` by `5x–10x`. Conversely, over `900–3,600` steps without any floor, $R_{\text{diff}}$ drifts downward (`14.38 -> 11.25 -> 8.06 -> 6.02`) as the top 1–2 dominant modes outgrow the tail.

In [`Hierarchical16QubitPOVMReadout.apply_soft_powerlaw_spectral_floor`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/hierarchical_qubit.py#L920-L965), we implement a **Soft Power-Law Spectral Floor** with threshold $r_{\text{floor}} = 6.5$:
1. Given the raw diffuse projection $P = \Psi_{\text{beam2}} V_{\text{diff}} \in \mathbb{R}^{N_{\text{tok}} \times r}$ ($r = 16$), we compute the $r \times r$ Gram eigendecomposition $P^\top P = V \operatorname{diag}(s_1^2, \dots, s_r^2) V^\top$ in `< 0.02 ms` under `torch.no_grad()`, where $s_1 \ge s_2 \ge \dots \ge s_r > 0$, and evaluate the Shannon effective rank:
   $$p_j = \frac{s_j}{\sum_{m=1}^r s_m}, \qquad R_{\text{cur}} = \exp\!\left(-\sum_{j=1}^r p_j \ln p_j\right)$$
2. **Exact Identity When Healthy ($R_{\text{cur}} \ge r_{\text{floor}}$)**:
   - Whenever $R_{\text{cur}} \ge 6.5$, `apply_soft_powerlaw_spectral_floor` returns $P$ **unchanged** (`max_diff == 0.0`), leaving the natural Zipfian spectrum 100% untouched.
3. **Gentle Zipfian Power-Law Floor When $R_{\text{cur}} < r_{\text{floor}}$**:
   - When $R_{\text{cur}} < 6.5$, we define a natural Zipfian power-law floor profile anchored to $s_1$:
     $$s_{\text{zipf}, j} = 0.48 \, s_1 \, j^{-0.62}, \qquad s_{\text{target}, j} = \max(s_j, s_{\text{zipf}, j}), \qquad j \in \{1, \dots, r\}$$
   - We blend $s$ toward $s_{\text{target}}$ with deficit weight $\eta = \text{clamp}\!\left(\frac{r_{\text{floor}} - R_{\text{cur}}}{r_{\text{floor}} - 3.5},\, 0.25,\, 1.0\right)$, rescale to preserve exact Frobenius energy ($\|s_{\text{new}}\|_2 = \|s\|_2$), clamp per-mode gain ratios $\rho_j = \text{clamp}(s_{\text{new}, j} / s_j,\, 0.5,\, 2.2)$, and apply the detached $r \times r$ symmetric operator $M_{\text{scale}} = V \operatorname{diag}(\rho) V^\top$:
     $$P_{\text{floored}} = P M_{\text{scale}}$$
   - This guarantees: (i) zero SVD backward singularities, (ii) exact Frobenius norm preservation (`norm_ratio = 1.0000`), (iii) strict preservation of the Zipfian mode ordering $s_{\text{new}, 1} > s_{\text{new}, 2} > \dots > s_{\text{new}, r}$, and (iv) $R_{\text{diff}} \ge 6.5$ across multi-thousand-step runs.

---

## 4. Observable Hypotheses & Quantitative Predictions

### Hypothesis H1 — Soft Power-Law Spectral Floor (`kraus_spectral_floor = 6.5`) Is Exact Identity When $R_{\text{diff}} \ge 6.5$ and Lifts Decayed Spectra ($R_{\text{diff}} < 6.5$) Above `6.5` While Preserving Frobenius Energy & Zipfian Ordering
- **Claim**:
  1. On any diffuse activation tensor with $R_{\text{cur}} \ge 6.5$, `apply_soft_powerlaw_spectral_floor(raw_p, 6.5)` is bit-for-bit identical (`max_abs_diff == 0.0`) to `raw_p`.
  2. On a decayed activation tensor with $R_{\text{cur}} \in [4.5, 6.2] < 6.5$, `apply_soft_powerlaw_spectral_floor(raw_p, 6.5)` lifts $R_{\text{diff}}$ to **`>= 6.5` (`<= 11.0`, never flat `16.0`)**, preserves Frobenius norm within `1e-4`, and keeps gradients finite.

### Hypothesis H2 — Full-Ambient $J_{\text{anti}}$ (`16×16 = 256`) & Equi-Dimensional Square-$O(N)$ (`frame_expansion = 1.0`) vs. Parity-Split (`16×8 = 128`) at `900` Steps (`7.37M` FineWeb-Edu Tokens, `<= 8.0 GB` VRAM)
- **Claim**:
  On `L=4, d=256, H=4` (`900` steps = `7.37M` tokens on genuine FineWeb-Edu, evaluated on held-out `val_slice.bin` at $T_{\text{val}} = 512$):
  1. Full-Ambient $J_{\text{anti}}$ (`16×16 = 256`, `5.6092` Val CE) outperforms matched Parity-Split (`16×8 = 128`, `use_ortho_decouple=True`) because doubling per-register unigram amplitude capacity from `A=8` (`128` total amplitudes) to `A=16` (`256` total amplitudes) with $J_{\text{anti}}$ self-decoupling preserves full $d=256$ lexical resolution.
  2. The **Consistently $N$-Dimensional Square-$O(N)$ model (`frame_expansion = 1.0`, $d = 256$ all the way through every attention and two-beam interference matrix)** achieves **`> 1.25x` higher training throughput (`tok/s`)** and lower parameter count than `frame_expansion = 2.0` while converging below `< 5.72` Val CE at Step 900.

### Hypothesis H3 — Local `<= 8.0 GB` Multi-Thousand-Step Scaling Study (`3,600` Steps = `29.5M` FineWeb-Edu Tokens) With Soft Power-Law Spectral Floor (`kraus_spectral_floor = 6.5`) Beats Matched `ClassicalModdedGPT` and Breaks `< 5.10` Val CE
- **Claim**:
  Scaling `Isolate 3` + Soft Power-Law Spectral Floor (`w18_8gb_L4_d256_janti_soft_floor_3600steps`, `L=4, d=256, H=4, 16×16` Full-Ambient $J_{\text{anti}}$ + Gentle `2x` Zero-WD Qubit LR `0.006` + `kraus_spectral_floor = 6.5`) to **`3,600` steps (`29.5M` tokens)** on the local RTX 5060 Ti (`<= 8.0 GB` VRAM cap):
  1. Maintains $R_{\text{diff}} \ge 6.5$ all the way to Step 3,600 (preventing late-stage Kraus rank collapse below `6.0`).
  2. Outperforms the matched `3,600`-step `ClassicalModdedGPT` (`w18_8gb_classical_leaderboard_exact_L4_d256_3600steps`) at every evaluation checkpoint (`Steps 300, 900, 1800, 2700, 3600`) and reaches a new all-time local `8 GB` FineWeb-Edu record of **`<= 5.1000` Val CE** (`>= 23.0%` Top-1 accuracy).

---

## 5. Operational Tooling & Script Opportunities Gate (Section 5 Compliance)

### 5.1 Existing Scripts Used
1. **Model Parameter & Smoke Verification (`scripts/inspect_model.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py --num-layers 8 --embed-dim 512 --num-heads 8 --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 --num-higher-states 256 --two-beam-kboson-readout --stereographic-lift --quadratic-diffuse-fidelity 0.50 --smoke-steps 2
   ```
2. **Colab 2-G4 Fleet Launch & Status (`scripts/colab_fleet.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py launch --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave18_fleet.json
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py status --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave18_fleet.json
   ```
3. **Zero-Downtime Live Monitoring Daemon (`scripts/monitor_colab_runs.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once
   ```

### 5.2 Build-Before-Launch & Promoted Backlog Scripts (Implemented in Plan 033)
1. **`scripts/full_ambient_step0_calibration_audit.py` (Promoted from `scripts/BACKLOG.md` Plan 032)**:
   - Standalone CLI utility (`< 300` lines, `<= 8` args per function) that audits Step-0 inner-product distributions (`c_sharp`, `c_diff`, `max_self_overlap`, `self_register_overlap_max`, `directed_bigram_overlap_ratio`, and `Step 0 ECE/CE`) across `(Q, A, use_ortho_decouple, decouple_self_in_full_ambient)` configurations in `< 5s`.
2. **`scripts/vram_envelope_autotuner.py` (Promoted from `scripts/BACKLOG.md` Plan 025)**:
   - Standalone CLI utility (`< 300` lines, `<= 8` args per function) that profiles `(micro_batch_size, grad_accum, chunk_tokens)` under `torch.cuda.set_per_process_memory_fraction` to find the highest-throughput zero-OOM configuration within `--max-vram-gb 8.0`.
3. **`docs/SCRIPT_BEST_PRACTICES.md` & `src/qdlblocks/benchmarks/config_models.py`**:
   - Codifies the repository's script length (`<= 400` lines target / `<= 600` ceiling) and function signature (`<= 8` arguments via `@dataclass` configs) rules, and extracts shared `QuantumExperimentSpec`, `initialize_povm_log_prior`, `build_quantum_born_optimizers`, and evaluation helpers so verification scripts (`examples/43_*.py`) stay compact and never diverge from `examples/11_*.py`.
4. **`configs/wave18_fleet.json`**:
   - Declarative 2-session Google Colab G4 manifest (`gpu_g4_w6_1` `16×16` Parity-Split vs. `gpu_g4_w6_2` `16×32` Full-Ambient $J_{\text{anti}}$ on `L=8, d=512` for `900M` FineWeb-Edu tokens with Gentle `2x` Zero-WD Qubit LR, Bose-saturated log-prior, and Soft Power-Law Spectral Floor `kraus_spectral_floor = 6.5`).

### 5.3 Nice-to-Have Scripts (Logged to `scripts/BACKLOG.md`)
1. **`scripts/ast_signature_and_length_linter.py`**: Fast AST pre-commit check that flags any new Python file exceeding `600` lines or any new function exceeding `8` arguments without a `@dataclass` config.
2. **`scripts/equidimensional_on_spectrum_probe.py`**: Diagnostic CLI that compares singular-value spectra and layerwise angular velocity between overcomplete (`frame_expansion = 2.0/4.0`) and Equi-Dimensional Square-$O(N)$ (`frame_expansion = 1.0`) checkpoints.
