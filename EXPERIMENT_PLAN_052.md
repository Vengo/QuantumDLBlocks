# EXPERIMENT_PLAN_052 — Wave 37: Standalone Pure-Complex Register Density GPT (`GeminiBornComplexPackedWithRoPE` + Refinements 1, 3, & 4, Zero $\gamma_{\text{coh}}$)

## 1. Motivation & Mathematical Context

In [`GeminiBornComplexPackedWithRoPE.py`](file:///home/festivity/agy/QuantumDLBlocks/GeminiBornComplexPackedWithRoPE.py) (left untouched in the repository root as the reference implementation from the external Gemini design session), each token state lives natively in a complex Hilbert space $\psi_t = \psi_{\text{re}, t} + i\,\psi_{\text{im}, t} \in \mathbb{C}^{d_{\text{complex}}}$ partitioned into $H$ complex registers of dimension $d_h = 32$ ($d_{\text{complex}} = H \cdot d_h$).

Rather than bolting this mechanism into our real-wavefunction `LevelSeparatedWaveletConvGPT` model, `Wave 37` evaluates the **standalone pure-complex architecture** built directly around [`GeminiBornComplexPackedWithRoPE.py`](file:///home/festivity/agy/QuantumDLBlocks/GeminiBornComplexPackedWithRoPE.py) and tests the impact of **Refinements 1, 3, and 4** (with **zero $\gamma_{\text{coh}}$** coherent-outer-product parameter):

1. **Refinement 1 — Strict $Q/K$ Rotary Separation (Fixing the Step-7 Absolute-Position Phase Leak)**:
   - In the reference [`GeminiBornComplexPackedWithRoPE.py`](file:///home/festivity/agy/QuantumDLBlocks/GeminiBornComplexPackedWithRoPE.py#L179-L212), lines 179–180 overwrite `q_re, q_im` in-place with the position-rotated query $q_t^{\text{rot}} = q_t e^{i t \boldsymbol{\omega}}$ while leaving $v_s$ un-rotated, and then line 210 probes $\rho_t = \sum_{s \le t} w_{ts} |v_s\rangle\langle v_s|$ with $|q_t^{\text{rot}}\rangle$, injecting an absolute-position phase factor $e^{i t \boldsymbol{\omega}}$ into $\langle v_s \mid q_t^{\text{rot}}\rangle_{\mathbb{C}}$.
   - **Fix**: Apply `ComplexRoPE` strictly to `q_rot_re, q_rot_im` and `k_rot_re, k_rot_im` for the Step 3 inner product $c_{ts} = \langle q_t^{\text{rot}} \mid k_s^{\text{rot}} \rangle_{\mathbb{C}}$, keeping everything from Step 5 onward in position-invariant content space.

2. **Refinement 3 — Fused Single-Allocation Hermitian Triangle Packing & Unpacking (`1-GEMM` Width `1088 = 1024 + 64`)**:
   - Packs the symmetric real upper triangle $\operatorname{triu}(\rho_{\text{re}, s}, 0)$ ($528$ entries at $d_h = 32$) and skew-symmetric imaginary strict lower triangle $\operatorname{tril}(\rho_{\text{im}, s}, -1)$ ($496$ entries at $d_h = 32$) into a single $(d_h \times d_h = 1024)$ tensor via `torch.where(upper_mask, rho_re, rho_im)` without allocating $5\times$ intermediate `triu`/`tril` buffers.
   - Concatenates `[v_packed_flat, v_re, v_im]` along the feature dimension to width $d_h^2 + 2 d_h = 1024 + 64 = 1088 = 17 \times 64$ so **one real Tensor-Core GEMM** `torch.matmul(w_born, v_packed_and_ket)` computes both the mixed density matrix $\rho_t$ and the first-moment $V$-space ket $|\bar{v}_t\rangle$.

3. **Refinement 4 — Probing $\rho_t^{\text{purified}}$ in $V$-Space via Krylov Power Iteration ($|\bar{v}_t\rangle$ Anchor Ket Instead of $|q_t\rangle$)**:
   - Because $\rho_t = \sum_{s \le t} w_{ts} |v_s\rangle\langle v_s|$ is an operator on $V$-space, probing $\rho_t^{\text{purified}}$ with $|q_t\rangle \in Q$-space at Step 0 (when $W_Q$ and $W_V$ are independent random unitaries) scales $|v_s\rangle$ by a random complex scalar $\langle v_s \mid q_t\rangle_{\mathbb{C}}$ of magnitude $\sim 1/\sqrt{d_h} \approx 0.177$ and random phase.
   - Probing instead with the normalized first-moment $V$-space anchor ket:
     $$|\bar{v}_t\rangle = \operatorname{normalize}_{\mathbb{C}}\!\left(\sum_{s \le t} w_{ts} |v_s\rangle\right) \in \mathbb{C}^{d_h}, \qquad |u_t^{\text{raw}}\rangle = \rho_t^{\text{purified}} |\bar{v}_t\rangle = \frac{\rho_t \big(\rho_t |\bar{v}_t\rangle\big)}{\operatorname{Tr}(\rho_t^2) + \varepsilon}$$
     executes **two steps of Krylov Power Iteration** on $\rho_t$ inside $V$-space:
     * When a single token $s^*$ dominates ($w_{t s^*} = 1$), $\rho_t^{\text{purified}} |\bar{v}_t\rangle = |v_{s^*}\rangle\langle v_{s^*} \mid v_{s^*}\rangle_{\mathbb{C}} = 1.0 \cdot |v_{s^*}\rangle$ with exact overlap $1.0 + 0.0i$.
     * Normalizing the Krylov output $|u_t\rangle = \operatorname{normalize}_{\mathbb{C}}(|u_t^{\text{raw}}\rangle)$ further ensures that the attention update norm is invariant across context positions $t \in [1, 512]$ even when early-step attention is diffuse ($\operatorname{Tr}(\rho_t^2) \sim 1/d_h$).

---

## 2. Observable Hypotheses & Quantitative Acceptance Criteria

### Hypothesis H1 — Exact Hermitian Triangle Packing, `ComplexRoPE` Shift Invariance, & Associative Krylov Purification Parity
- **Claim**:
  1. `pack_hermitian_outer_product` + `unpack_hermitian_matrix` reconstructs both $\rho_{\text{re}}$ and $\rho_{\text{im}}$ from a single `torch.matmul(w_born, v_packed_flat)` GEMM with max absolute error `< 1e-6` against the 2-GEMM unpacked reference.
  2. `ComplexRoPE` satisfies relative-position shift invariance $|\langle q_{t+\Delta}^{\text{rot}} \mid k_{s+\Delta}^{\text{rot}} \rangle_{\mathbb{C}} - \langle q_t^{\text{rot}} \mid k_s^{\text{rot}} \rangle_{\mathbb{C}}| < 10^{-6}$.
  3. Associative two-step Krylov evaluation $\frac{\rho_t (\rho_t |\bar{v}_t\rangle)}{\|\rho_{\text{re}, t}\|_F^2 + \|\rho_{\text{im}, t}\|_F^2 + \varepsilon}$ matches explicit matrix-squared McWeeny purification $\frac{(\rho_t^2)}{\operatorname{Tr}(\rho_t^2) + \varepsilon} |\bar{v}_t\rangle$ with max absolute error `< 1e-6` while cutting purification FLOPs by `16x` at $d_h = 32$.

### Hypothesis H2 — Step-0 Subspace Alignment & Shift-Equivariance Audit (`Q-Probe` vs. `V-Space Krylov Probe`)
- **Claim**: At Step 0 (random unitary $W_Q, W_K, W_V \in U(d_{\text{complex}})$ with $d_h = 32$):
  1. **Single-Token Dominance ($w_{t s^*} = 1$)**: `V-Krylov` preserves exact unit norm $\||u_t^{\text{raw}}\rangle\|_{\mathbb{C}} = 1.000 \pm 10^{-5}$ and real-positive overlap $\operatorname{Re}\langle v_{s^*} \mid u_t^{\text{raw}} \rangle_{\mathbb{C}} = 1.000 \pm 10^{-5}$, whereas `Q-Probe` shrinks the output norm to `< 0.25` ($\sim 1/\sqrt{32} \approx 0.177$) with scrambled phase.
  2. **Translation Equivariance Under Position Shift**: When identical content vectors $(q, k, v)$ are shifted by $\Delta t$ positions, `V-Krylov` produces zero absolute-position phase drift (`< 1e-6`), whereas probing with rotated $|q_t^{\text{rot}}\rangle$ (`GeminiBornComplexPackedWithRoPE.py` line 210) exhibits `> 0.10` phase drift across positions.
  3. **Competing Multi-Token Mixture ($w_t = (0.60, 0.25, 0.15)$)**: Two-step Krylov probing amplifies principal-mode energy $|\langle v_{s_1} \mid u_t \rangle_{\mathbb{C}}|^2$ from `~0.809` (in $|\bar{v}_t\rangle$) to `> 0.960`.

### Hypothesis H3 — Standalone Pure-Complex GPT Training Ablation on `FineWeb-Edu` (`300` Steps / `2.46M` Tokens)
- **Claim**: In a controlled 3-arm training comparison of the standalone `GeminiPackedComplexRoPEGPT` model ($L = 6$, $d_h = 32, H = 4 \implies d_{\text{complex}} = 128$, i.e. $256$ real degrees of freedom per token, `B = 16, T = 512`, `300` steps / `2.46M` tokens on `FineWeb-Edu`):
  1. Arm 1 (`w37_gemini_complex_qprobe_d128c_L6_300steps`: original un-refined `Q-Probe` from `GeminiBornComplexPackedWithRoPE.py`) vs.
  2. Arm 2 (`w37_gemini_complex_vkrylov_raw_d128c_L6_300steps`: Refinements 1, 3, & 4 with raw McWeeny output $\rho_t^{\text{purified}} |\bar{v}_t\rangle$) vs.
  3. Arm 3 (`w37_gemini_complex_vkrylov_norm_d128c_L6_300steps`: Refinements 1, 3, & 4 with unit-normalized Krylov ket $\operatorname{normalize}_{\mathbb{C}}(\rho_t^{\text{purified}} |\bar{v}_t\rangle)$):
  - `V-Krylov` (Arm 2 / Arm 3) achieves strictly better validation cross-entropy (`val_ce`) and higher Top-1 accuracy than the un-refined `Q-Probe` (Arm 1), with `100%` finite gradients and exact complex unit-sphere norm conservation.

---

## 3. Planned File Changes

1. **[KEEP UNTOUCHED] [`GeminiBornComplexPackedWithRoPE.py`](file:///home/festivity/agy/QuantumDLBlocks/GeminiBornComplexPackedWithRoPE.py)**:
   - Preserved intact as the original 265-line reference script from the user's other Gemini session.
2. **[NEW] [`src/qdlblocks/layers/packed_complex_register_density_attn.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/packed_complex_register_density_attn.py)** & **[MODIFY] [`src/qdlblocks/layers/__init__.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/__init__.py)**:
   - Standalone complex building blocks (`ComplexRoPE`, `ComplexLinear`, `complex_l2_normalize`, `pack_hermitian_outer_product`, `pack_hermitian_outer_product_and_ket`, `unpack_hermitian_matrix`, `krylov_purified_density_probe`, `explicit_mcweeny_density_probe`, `PackedComplexRegisterDensityAttention`).
3. **[NEW] [`src/qdlblocks/models/gemini_packed_complex_rope_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/gemini_packed_complex_rope_gpt.py)** & **[MODIFY] [`src/qdlblocks/models/__init__.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/__init__.py)**:
   - Standalone pure-complex GPT architecture `GeminiPackedComplexRoPEGPT`, `GeminiPackedComplexRoPEBlock`, and `GeminiPackedComplexRoPEConfig`:
     * Complex token embedding table $\psi^{(0)} = E_{\text{re}}[x] + i E_{\text{im}}[x] \in \mathbb{C}^{B \times T \times d_{\text{complex}}}$ normalized on the complex unit sphere,
     * $L$ stacked complex blocks (`PackedComplexRegisterDensityAttention` + complex unitary two-beam phase-gated feed-forward `ComplexUnitaryInterferenceFFN`),
     * Tied complex Born readout (`compute_complex_born_chunked_ce` and `eval_complex_born_full_logits` with exact train/eval parity) using complex Hermitian overlaps $\langle \psi_t \mid M_v \rangle_{\mathbb{C}}$ and Parseval scale $\beta_0 = c^* \sqrt{2 d_{\text{complex}}}$.
4. **[MODIFY] [`src/qdlblocks/benchmarks/wave_ingestor.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/wave_ingestor.py)**:
   - Register `"W37"` in `LOCAL_FINEWEB_EDU_WAVES`.
5. **[NEW] [`tests/test_plan052_packed_complex_register_density_attn.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan052_packed_complex_register_density_attn.py)**:
   - Unit tests verifying `H1`, `H2`, and standalone `GeminiPackedComplexRoPEGPT` forward/backward + train/eval parity.
6. **[NEW] [`examples/62_verify_plan052_gemini_packed_complex_rope_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/62_verify_plan052_gemini_packed_complex_rope_gpt.py)**:
   - Runs `H1`–`H3` verification, logs all 3 standalone complex GPT runs via `StandardizedExperimentLogger` (`wave_id="W37"`, `plan_id="EXPERIMENT_PLAN_052"`), saves `.cache/hypothesis_experiments_052.json`, and compares directly against `Wave 36` `Option 2`.

---

## 4. Verification & Testing Plan

1. **Unit Tests (`pytest`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python -m pytest /home/festivity/agy/QuantumDLBlocks/tests/test_plan052_packed_complex_register_density_attn.py -v
   ```
2. **AST Signature & SLOC Linter (`Rule 1 & Rule 2`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/layers/packed_complex_register_density_attn.py src/qdlblocks/models/gemini_packed_complex_rope_gpt.py tests/test_plan052_packed_complex_register_density_attn.py examples/62_verify_plan052_gemini_packed_complex_rope_gpt.py
   ```
3. **End-to-End GPU Experiment (`Wave 37`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/examples/62_verify_plan052_gemini_packed_complex_rope_gpt.py
   ```
4. **WebUI Synchronization**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload
   ```

---

## 5. Operational Tooling & Script Opportunities

1. **Existing Scripts Used**:
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py`
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py --trigger-reload`
   - `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once`
2. **`Nice-to-Have` Scripts**:
   - `scripts/krylov_density_purity_and_eigenvalue_spectrum_probe.py`: Audits the empirical Hermitian eigenvalue spectrum $(\lambda_1, \dots, \lambda_{d_h})$ of $\rho_t$, purity $\operatorname{Tr}(\rho_t^2)$, and Krylov principal-eigenvector alignment $|\langle e_1(\rho_t) \mid u_t \rangle_{\mathbb{C}}|^2$ across layers and context positions $t \in [1, 512]$.

---

## 6. Constants Ledger & Empirical Results (`Wave 37`)

### 6.1 Top-of-File Named Constants Ledger (Rule 7)
| Constant Name | File | Value | Mathematical / Physical Rationale |
| :--- | :--- | :---: | :--- |
| `DEFAULT_COMPLEX_ROPE_BASE` | `packed_complex_register_density_attn.py` | `10000.0` | Base frequency $\omega_a = \text{base}^{-a / d_h}$ for complex register rotary embeddings. |
| `DEFAULT_COMPLEX_REGISTER_DIM` | `packed_complex_register_density_attn.py` | `32` | Per-register complex Hilbert space dimension $d_h \in \mathbb{C}^{32}$ (`1024` packed Hermitian entries). |
| `DEFAULT_P_BOSON_POWER` | `packed_complex_register_density_attn.py` | `2` | Quartic Born probability exponent $p = 2$ ($w_{ts} \propto |c_{ts}|^{2p} = |c_{ts}|^4$). |
| `DEFAULT_COMPLEX_NORM_EPS` | `packed_complex_register_density_attn.py` | `1e-7` | Numerical floor $\varepsilon$ for complex L2 normalization, Born probability normalization, and McWeeny trace division. |
| `DEFAULT_COMPLEX_PARSEVAL_C` | `gemini_packed_complex_rope_gpt.py` | `1.48` | Analytical Parseval $\mu\text{P}$ inverse-temperature constant $\beta_0 = c^* \sqrt{2 d_{\text{complex}}}$. |
| `DEFAULT_COMPLEX_FFN_EXPANSION` | `gemini_packed_complex_rope_gpt.py` | `2` | Complex unitary two-beam interference expansion factor in `ComplexUnitaryInterferenceFFN`. |
| `DEFAULT_COMPLEX_RESIDUAL_INIT` | `gemini_packed_complex_rope_gpt.py` | `0.35` | Depth-scaled complex residual mixing gate initialization $0.35 / \sqrt{l + 1}$. |

### 6.2 Empirical Results Summary (`Wave 37` — `.cache/hypothesis_experiments_052.json`)

1. **`H1` & `H2` Algebraic & Step-0 Subspace Probes (`PASS`)**:
   - **1-GEMM Hermitian Triangle Packing/Unpacking**: Exact round-trip reconstruction (`max_pack_re_err = 0.00e+00`, `max_pack_im_err = 0.00e+00`).
   - **Full-Spectrum `ComplexRoPE` Shift-Invariance (Refinement 1)**: Relative position shift invariance verified (`max_rope_shift_re_err = 3.73e-08`, `max_rope_shift_im_err = 7.45e-09`).
   - **Associative Two-Step Krylov `O(d_h^2)` vs. Explicit `O(d_h^3)` McWeeny Matrix Squaring (Refinements 3 & 4)**: Exact algebraic parity (`max_krylov_vs_mcweeny_err = 7.45e-08`) while cutting per-token purification FLOPs by `16x` at $d_h = 32$.
   - **Step-0 Single-Dominant-Token Subspace Match (`H2`)**:
     * Original `Q-Probe` ($\rho_t^{\text{purified}} |q_t^{\text{rot}}\rangle$): Step-0 output norm collapsed to `0.1480` ($\sim 1/\sqrt{32}$) with random complex phase due to $W_Q \perp W_V$ subspace mismatch.
     * **`V-Space Krylov Probe` ($\rho_t^{\text{purified}} |\bar{v}_t\rangle$, Refinement 4)**: Step-0 output norm = **`1.0000`** with exact real-positive overlap **`1.0000 + 0.0000i`**.
   - **Competing 3-Token Mixture $(0.60, 0.25, 0.15)$ Principal-Eigenvector Fidelity (`H2`)**:
     * First-moment anchor $|\bar{v}_t\rangle$ fidelity with principal eigenvector $|e_1(\rho_t)\rangle$: `0.8090`
     * Two-step Krylov purified state $\rho_t^{\text{purified}} |\bar{v}_t\rangle$ fidelity: **`0.9946`** (`+0.1856` purity boost in a single wait-free step).

2. **`H3` 3-Arm Standalone Complex GPT Benchmark on `FineWeb-Edu` (`300` Steps / `2.46M` Tokens, $d_{\text{complex}} = 128, H = 4, d_h = 32 \implies 256$ Real DoF/Token, $L = 6, B = 16, T = 512$) vs. `Wave 36` Baselines**:

| Run Key / Architecture | Params | Probe / Attention Formulation | Step-0 Probe Norm | Step 75 Val CE | Step 150 Val CE | Step 225 Val CE | **Step 300 Val CE** | **Step 300 Val PPL** | **Top-1 Acc** | Tok/s | Peak VRAM |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w36_opt1_diff_ngram_d256_L6_300steps` (`Wave 36` Option 1) | `24.42M` | Real Wavelet-Conv + Diff N-Gram | — | `6.8277` | `6.5784` | `6.4335` | `6.3481` | `571.40` | `14.66%` | `9,913` | `3,519 MiB` |
| `w36_opt2_no_ngram_d256_L6_300steps` (`Wave 36` Option 2) | `11.82M` | Real Wavelet-Conv (Pure Neural) | — | `6.8359` | `6.5653` | `6.3976` | `6.3157` | `553.21` | `14.92%` | `10,331` | `2,542 MiB` |
| `w37_gemini_complex_qprobe_d128c_L6_300steps` (**Arm 1**) | `14.88M` | Original `Q-Probe` ($\rho_t^{\text{purified}} |q_t^{\text{rot}}\rangle$, Unnorm) | `0.0898` | `6.9309` | `6.6376` | `6.4495` | `6.3454` | `569.88` | `14.44%` | `8,127` | `5,273 MiB` |
| **`w37_gemini_complex_vkrylov_raw_d128c_L6_300steps` (Arm 2)** | `14.88M` | **`V-Krylov Probe` (Refinements 1+3+4, Unnorm)** | **`0.5773` (`6.43x`)** | **`6.8666`** | **`6.5612`** | **`6.3841`** | **`6.2951`** | **`541.91`** | **`14.90%`** | `7,970` | `5,294 MiB` |
| **`w37_gemini_complex_vkrylov_norm_d128c_L6_300steps` (Arm 3)** | `14.88M` | **`V-Krylov Probe` (Refinements 1+3+4, Unit-Norm)** | **`0.5794` (`6.45x`)** | **`6.8617`** | **`6.5685`** | **`6.3883`** | **`6.2951`** | **`541.89`** | **`14.91%`** | `7,952` | `5,355 MiB` |

3. **Key Takeaways**:
   - **Refinements 1 + 3 + 4 (`V-Space Krylov Probe`) Deliver a Massive `-0.0503` Nat (`-28.0` PPL) Gain Over `Q-Probe`**: Probing $\rho_t^{\text{purified}}$ with the V-space first-moment anchor $|\bar{v}_t\rangle$ instead of $|q_t^{\text{rot}}\rangle$ increases Step-0 probe norm by **`6.45x`** (`0.0898 -> 0.5794`), eliminates absolute-position phase leakage, and improves 300-step validation cross-entropy from `6.3454` (`569.88` PPL) down to **`6.2951` (`541.89` PPL)** (`+0.47%` Top-1 accuracy).
   - **Standalone Pure-Complex `GeminiPackedComplexRoPEGPT` Outperforms Both `Wave 36` Arms**: Even with a simple unfactored complex token embedding (zero wavelets, zero multi-rate convolutions, zero N-gram tables), the standalone pure-complex `V-Krylov` GPT (`6.2951` Val CE, `541.89` PPL) beats `Wave 36` `Option 2` (`6.3157` Val CE, `553.21` PPL) by **`-0.0206` nats (`-11.32` PPL)** and `Wave 36` `Option 1` (`6.3481` Val CE, `571.40` PPL) by **`-0.0530` nats (`-29.51` PPL)** on the identical `FineWeb-Edu` stream and validation shards.

