# Experiment Plan 024 — Calibrated Last-Layer Softmax Readout for 16-Qubit Hierarchical `QuantumBornGPT` (`--last-layer-softmax`) & Wave 15 Probability Calibration Hotfix

---

## 1. Standalone Project Context, Wave 15 Live Telemetry Diagnosis, & Motivation for Plan 024

### 1.1 What `QuantumDLBlocks` Is Building and Where the Leaderboard Stands
**QuantumDLBlocks** is a dual-backend (`PyTorch + Triton` on GPU and `JAX + Pallas` on TPU) deep-learning library benchmarking **Quantum-Geometric Language Models** (`QuantumBornGPT`) head-to-head against **`ClassicalModdedGPT`** (our single-GPU port of the `KellerJordan/modded-nanogpt` speedrun architecture) on the **FineWeb-Edu `900M`-token (`9 × 100M` shard) streaming benchmark** evaluated on the official `KellerJordan/modded-nanogpt` `10,485,760`-token (`10 × 2^{20}`) held-out validation split (`fineweb_val_000000.bin`) at fixed context length $T_{\text{val}} = 512$ over the exact `50,257` GPT-2 vocabulary (`0..50256`).

Across Waves 1–14:
- **`ClassicalModdedGPT` All-Time Record**: `w12_900m_classical_leaderboard_exact_3ve_skip7_b131k` achieved **`3.3095` Val CE** (`27.4` PPL, `39.21%` Top-1, `61.3%` Top-5, `132.4m`), and in Wave 13 `w13_900m_classical_anvil2_full_catABC_b131k` achieved **`3.3583` Val CE** (`28.7` PPL, `38.66%` Top-1, `60.73%` Top-5).
- **`QuantumBornGPT` Previous All-Time Record (Wave 13)**: `w13_900m_qborn_v6_9shard_ppmi_svd_deep_L12_H8_d128` achieved **`3.5121` Val CE** (`33.5` PPL, `37.35%` Top-1, `58.91%` Top-5, `192.6m`) after `900M` tokens, and the 8-layer `w13_900m_qborn_v6_9shard_ppmi_svd_ngram_anvil_L8` achieved **`3.6182` Val CE** (`37.3` PPL, `36.51%` Top-1, `57.99%` Top-5) after `900M` tokens.

### 1.2 The Wave 15 Empirical Anomaly: Extraordinary Top-1 Accuracy (`37.16%` at `200M–275M` Tokens) vs Compressed Top-5 (`54.02%`) and Elevated Cross-Entropy (`3.9896`–`4.0129`)
In Wave 15 (Plan 023), we introduced the **`50,304 × 16 × 32` Multi-Register Unigram Qubit Amplitude Table** (`25.76M` parameters) coupled with **`256` (or `512`) Higher-Level Quantum States** (`256 × 16 × 32`) interacting via a per-register orthogonal operator (`16 × 256 × 256`).

By `200M–275M` tokens (Shard 3 of 9), the live Wave 15 runs revealed a striking empirical divergence between **Top-1 ranking accuracy** and **Cross-Entropy / Top-5 probability calibration**:

| Model (`Wave` / `Tokens Trained`) | Readout Architecture | Val CE (Nats) | Val PPL | Top-1 Acc | Top-5 Acc | Gap ($\text{Top-5} - \text{Top-1}$) | Readout Scale ($\beta$ or $\sigma_{\text{logit}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ClassicalModdedGPT` (`W13`, `@ 200M` tok)** | Unconstrained `lm_head` + `15*tanh(z/15)` + Softmax | `3.6810` | `39.7` | `35.40%` | `57.65%` | **`+22.25%`** | $\sigma_z = 2.68$ |
| **`ClassicalModdedGPT` (`W13`, `@ 900M` tok)** | Unconstrained `lm_head` + `15*tanh(z/15)` + Softmax | **`3.3583`** | **`28.7`** | **`38.66%`** | **`60.73%`** | **`+22.07%`** | $\sigma_z = 2.69$ |
| **`QuantumBornGPT` (`W13` `L=12`, `@ 200M` tok)** | Unit-Sphere Cosine POVM (`beta * <phi, M>`) | `3.9985` | `54.5` | `32.45%` | `53.21%` | `+20.76%` | $\beta = 66.44$ |
| **`QuantumBornGPT` (`W13` `L=12`, `@ 900M` tok)** | Unit-Sphere Cosine POVM (`beta * <phi, M>`) | `3.5121` | `33.5` | `37.35%` | `58.91%` | `+21.56%` | $\beta = 125.18$ |
| **`w15_900m_qborn_16x32_qubit_256hier_ngram_L8` (`@ 200M` tok)** | Unit-Sphere Hierarchical POVM (`beta * <phi, M>`) | `4.0129` | `55.3` | **`37.16%`** | `54.02%` | **`+16.86%`** | $\beta = 71.56$ |
| **`w15_900m_qborn_16x32_qubit_256hier_flagship_L8` (`@ 275M` tok)** | Unit-Sphere Hierarchical POVM (`beta * <phi, M>`) | `3.9896` | `54.0` | **`37.16%`** | `54.24%` | **`+17.08%`** | $\beta = 86.78$ |

Look at what this table shows:
1. **Representational Breakthrough on Top-1 (`37.16%` by `200M–275M` tokens)**:
   - At `200M` tokens, Wave 15's `16×32` qubit + `256`-state hierarchy (`37.16%` Top-1) is **`+4.71` percentage points ahead** of Wave 13's `L=12` quantum record at the same token count (`32.45%` Top-1), has already nearly matched Wave 13's **final `900M`-token** Top-1 (`37.35%`), and is within **`1.5%`** of `ClassicalModdedGPT`'s `900M`-token Top-1 (`38.66%`)!
2. **Severe Probability Calibration Bottleneck on Ranks 2–50 (`Top-5 - Top-1 = 16.86%` vs `22.2%`, and `Val CE = 4.01` vs `~3.45`)**:
   - In a well-calibrated language model on FineWeb-Edu, a `37.16%` Top-1 accuracy corresponds to a `~59.5%–61.5%` Top-5 accuracy ($\text{Gap} \approx 22\text{–}24.5\%$) and a validation cross-entropy of **`~3.40–3.52` nats**.
   - Instead, Wave 15's `Top-5 - Top-1` gap is compressed to **`16.86%`**, and its Cross-Entropy sits at **`3.9896–4.0129` nats** (`+0.50` nats higher than its Top-1 accuracy implies!).

### 1.3 Root-Cause Mathematical Diagnosis of the Last-Layer Calibration Bottleneck
Inspecting [`Hierarchical16QubitPOVMReadout`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/hierarchical_qubit.py#L425-L548), [`_DirectChunkedPOVMCEFn`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py#L168-L353), and [`initialize_hierarchical_16qubit_with_ppmi_svd`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/hierarchical_qubit.py#L551-L873) reveals four compounding mechanisms in the current unit-sphere POVM readout that cause this exact Rank 2–50 calibration failure:

1. **Root Cause 1 — Forced Unit-Sphere Row Normalization (`||M_v||_2 = 1`) + Spherical Tangent Projection Erases Per-Token Radial Norms**:
   - In `_DirectChunkedPOVMCEFn`, every vocabulary detector $M_v \in \mathbb{R}^{1024}$ is forcibly projected to unit length $\|M_v\|_2 \equiv 1$, and its gradient is projected onto the tangent plane of $S^{1023}$ (`dm_seed = (dm_norm - m_norm * <dm_norm, m_norm>) * r_inv`).
   - Furthermore, in `Hierarchical16QubitPOVMReadout.get_normalized_odd_qubits()`, every single `(v, q)` sub-register (`50,304 × 16 = 804,864` vectors of length `32`) is *also* forced to have equal norm $1/\sqrt{16}$ via `l2_normalize(..., dim=-1)`.
   - **Consequence**: Vocabulary tokens cannot learn distinct radial norms $\|w_v\|_2$, and individual tokens cannot weight their 16 qubit registers differently. On $S^{1023}$, a unit state $\phi_t$ can align closely with its #1 top choice $v_1$ (yielding `37.16%` Top-1!), but cannot simultaneously achieve large cosine similarity with 5–20 non-collinear plausible continuations $\{v_2, \dots, v_{20}\}$ unless those continuations have independent radial norms $\|w_v\|_2$.
2. **Root Cause 2 — Unbounded Logits Multiplied by Runaway Inverse Temperature ($\beta = 71.6 \to 86.8 \to 150$) Without Logit Soft-Capping**:
   - Because unit-sphere inner products $\langle \phi_t, M_v \rangle$ have small variance ($\sim 1/\sqrt{1024} \approx 0.031$), `PurityTemperatureGate` drives the inverse temperature $\beta$ up to **`71.6–86.8` by `200M–275M` tokens** (and `125–152` by `900M` tokens) with **no `logit_softcap`**.
   - At $\beta = 86.8$, a modest cosine difference $\Delta\cos = 0.08$ between Rank 1 and Rank 3 translates into a **$6.94\text{-nat}$ logit cliff**, assigning $>90\%$ probability to Rank 1 and crushing Ranks 2–50 to $<10^{-3}$, which inflicts a massive $-\log p(y_t)$ cross-entropy penalty whenever the true token lands in Ranks 2–50!
   - By contrast, `ClassicalModdedGPT` applies **`logit_softcap` ($z = 15 \tanh(u / 15)$)** before `Softmax`. Because $\frac{\partial z}{\partial u} = 1 - \tanh^2(u / 15) \to 0$ as $|u| \to 15$, once the #1 candidate is confidently ahead, its gradient saturates and the optimizer automatically redirects gradient energy into lifting and calibrating **Ranks 2–50**!
3. **Root Cause 3 — Half-Zero `base_povm_head[:, 0::2]`, `0.50` Gradient Dilution, & `weight_decay = 0.01` on the Readout Head**:
   - In `initialize_hierarchical_16qubit_with_ppmi_svd` (`line 789`), `base_povm_head[:, 0::2]` (`512` even coordinates) was initialized to `0.0`, while `get_coupled_povm_matrix` (`line 529`) computed `odd_part = 0.50 * base_povm_head[:, 1::2] + 0.50 * odd_coupled` (cutting gradients to `povm_odd_qubits` and `vocab_higher_affinity` in half!) and only coupled the `256` higher-level states into the odd coordinates `1::2`.
   - Furthermore, `povm_head` was placed in the `adamw_2d` parameter group with `lr = 0.004, weight_decay = 0.01`, whereas `ClassicalModdedGPT` optimizes its last-layer readout head (`lm_head`) with **`lr = 0.008, weight_decay = 0.0`**.
4. **Root Cause 4 — Over-Damped Isotropic Diversity (`iso_mix = 0.08 * (1 - alpha) = 0.02`) in `initialize_hierarchical_16qubit_with_ppmi_svd` & Uninitialized `value_embeds`**:
   - Multiplying `(1 - alpha)` by `0.08` at `line 684` of `hierarchical_qubit.py` gave `unigram_qubits` and `povm_odd_qubits` only `2%` isotropic Gaussian diversity vs `75%` collocation-smoothed `c_reg`, artificially lowering initial rank and slowing Rank 2–5 differentiation.

---

## 2. Mathematical & Architectural Specification of the Calibrated Last-Layer Softmax Readout (`--last-layer-softmax`)

### 2.1 Design Principle: Pure Quantum Backbone + Calibrated Hierarchical Softmax Last Layer
We keep the entire quantum backbone (`Hierarchical16QubitEmbedder`, `L` layers of `QuantumBornBlock` with `SubspaceProjectorBornAttention`, `OvercompleteTwoBeamInterference` `4d-TwoBeam`, and per-block `256`/`512`-state quantum hierarchy bridges) **100% unchanged** on the unit hypersphere $S^{d-1}$.

At the **final layer readout** (when `--last-layer-softmax` / `use_last_layer_softmax=True` is enabled), we replace the unit-sphere cosine + runaway-$\beta$ POVM collapse with a **Calibrated Hierarchical Softmax Readout**:

1. **Last-Layer Feature Calibration (`RMSNorm` + Learnable Channel Gain $\gamma_{\text{read}} \in \mathbb{R}^d$)**:
   Given the final quantum state $\psi_L \in \mathbb{R}^{B \times T \times d}$ ($d = 1024$), we map $\psi_L$ to the RMS-normalized feature vector:
   $$x_{\text{read}} = \gamma_{\text{read}} \odot \text{RMSNorm}(\psi_L) = \gamma_{\text{read}} \odot \left(\sqrt{d} \cdot \frac{\psi_L}{\|\psi_L\|_2 + \epsilon}\right) \in \mathbb{R}^{B \times T \times d}$$
   where $\gamma_{\text{read}} \in \mathbb{R}^d$ is initialized to $\mathbf{1}_d$. Because $\|x_{\text{read}}\|_2 \approx \sqrt{d} = 32$, standard inner products with $\mathcal{O}(1/\sqrt{d})$ readout weights naturally lie in $\mathcal{O}(1\text{–}10)$ without requiring an unstable $\beta \in [60, 150]$ multiplier.

2. **Unconstrained-Norm Two-Level Hierarchical Readout Weight ($W_{\text{soft}} \in \mathbb{R}^{V \times d}$)**:
   Inside [`Hierarchical16QubitPOVMReadout`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/hierarchical_qubit.py), we add `get_calibrated_softmax_weight(embedder, base_povm_head) -> torch.Tensor` which constructs the **unconstrained** `(V, d)` (`50,304 × 1024`) readout matrix once per step:
   - **Unconstrained Base Readout Matrix $W_{\text{base}} \in \mathbb{R}^{V \times d}$ (`base_povm_head`)**:
     Every row $v \in \{0, \dots, V-1\}$ has unconstrained radial norm in $\mathbb{R}^d$ (no `l2_normalize`!). At Step 0, its odd coordinates `1::2` are initialized to $\frac{1}{\sqrt{d}} u_{\text{odd}}(v)$ and its even coordinates `0::2` to $0$ (guaranteeing exact Step-0 input-output orthogonality $\langle \psi_0(v), W_{\text{soft}}[v] \rangle = 0$), while during training both `0::2` and `1::2` update freely with **`lr = 0.008, weight_decay = 0.0`**.
   - **Unconstrained `50,304 × 16 × 32` Multi-Register Qubit Detector Table ($U_{\text{odd}} \in \mathbb{R}^{V \times 512}$)**:
     Instead of clipping every `(v, q)` sub-register to unit norm (`get_normalized_odd_qubits`), we use the **unconstrained** `self.povm_odd_qubits.reshape(V, 512)` scaled by $\frac{1}{\sqrt{d}}$ so each token $v$ can freely adjust the norm of each of its 16 qubit registers!
   - **Full-Width Coarse-to-Fine `256`/`512`-State Hierarchical Readout Coupling ($\Delta W_{\text{hier}} \in \mathbb{R}^{V \times d}$)**:
     Given the $M$ interacted higher-level quantum states $\tilde{\Phi}_{\text{flat}} \in \mathbb{R}^{M \times 512}$ (`embedder.get_interacted_higher_states().reshape(M, 512) * embedder.subregister_scale`) and a learnable even-parity projection $W_{\text{hier\_even}} \in \mathbb{R}^{512 \times 512}$ (or per-register $16 \times 32 \times 32$ block-diagonal unitary `self.hier_even_proj` initialized to `0.0` so Step-0 self-overlap remains `0.0`), the vocabulary-to-hierarchy affinity matrix $A_{\text{vocab}\leftarrow\text{high}} \in \mathbb{R}^{V \times M}$ (`self.vocab_higher_affinity`) couples the $M$ higher-level states into **both** odd and even readout coordinates:
     $$\Delta W_{\text{odd}} = A_{\text{vocab}\leftarrow\text{high}} \, \tilde{\Phi}_{\text{flat}} \in \mathbb{R}^{V \times 512}, \qquad \Delta W_{\text{even}} = A_{\text{vocab}\leftarrow\text{high}} \, \text{reshape}\!\left(\text{einsum}(\text{"mqa,qba->mqb"}, \tilde{\Phi}, W_{\text{hier\_even}}), [M, 512]\right) \in \mathbb{R}^{V \times 512}$$
   - **Per-Token Learnable Radial Scale $s_{\text{tok}} \in \mathbb{R}^{V \times 1}$ (`self.token_radial_scale`, initialized to $\mathbf{1}_V$)**:
     Allows fast per-token radial norm adaptation across the `50,304` vocabulary:
     $$W_{\text{soft}} = s_{\text{tok}} \odot \Big( W_{\text{base}} + g_{\text{qubit}} \, \text{Interleave}(0, U_{\text{odd}}) + g_{\text{hier}} \, \text{Interleave}(\Delta W_{\text{even}}, \Delta W_{\text{odd}}) \Big) \in \mathbb{R}^{V \times d}$$

3. **Smooth Logit Soft-Capping + Softmax Cross-Entropy (`direct_chunked_softcap_linear_ce` with `bias` Support)**:
   - We extend `_DirectChunkedSoftcapLinearCEFn` and `direct_chunked_softcap_linear_ce` in [`src/qdlblocks/layers/direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py) to support an optional per-token bias vector `bias: torch.Tensor | None = None` (`(V,)`, passed from `model.povm_log_prior`), computing exact analytical gradients `(dx, dw, dbias)` in a single forward chunk loop (`0` GEMMs in `.backward()`):
     $$u_{t, v} = \langle x_{\text{read}, t}, W_{\text{soft}}[v] \rangle + b_v$$
     $$z_{t, v} = S \cdot \tanh\!\left(\frac{u_{t, v}}{S}\right) \qquad (S = 15.0\text{ default, configurable via }\texttt{--logit-softcap})$$
     $$\mathcal{L}_{\text{CE}} = -\frac{1}{N_{\text{valid}}} \sum_{t} \log \frac{\exp(z_{t, y_t})}{\sum_{v=0}^{50,256} \exp(z_{t, v})}$$

4. **Fixing `initialize_hierarchical_16qubit_with_ppmi_svd` Diversity & `value_embeds`**:
   - In `initialize_hierarchical_16qubit_with_ppmi_svd` ([`src/qdlblocks/layers/hierarchical_qubit.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/hierarchical_qubit.py#L684-L686)), we restore full isotropic-spectral blending `(1.0 - alpha_f) * z_even + alpha_f * c_reg` (instead of `0.08 * (1.0 - alpha_f)` which starved the embedding table of rank diversity), initialize `model.value_embeds` when present (for `L=12`), and scale `base_povm_head` / `povm_odd_qubits` appropriately when `use_last_layer_softmax=True`.

---

## 3. Observable Hypotheses & Quantitative Pass/Fail Criteria

- **Hypothesis H1 (Analytical Gradient Exactness, Step-0 Self-Overlap Orthogonality, & Rank-2–50 Calibration Geometry of `CalibratedHierarchicalSoftmaxReadout`)**:
  - **Prediction**:
    1. Extending `direct_chunked_softcap_linear_ce(x_2d, w_lm, targets, softcap=15.0, bias=b_prior, valid_vocab_size=50257)` with `bias` matches reference `F.cross_entropy(15.0 * torch.tanh((F.linear(x, w) + b) / 15.0)[:, :50257].float(), y)` to `< 1e-5` in forward loss and `< 1e-4` maximum relative/cosine error on all three analytical gradients `(dx, dw, dbias)`.
    2. After `initialize_hierarchical_16qubit_with_ppmi_svd(model, ...)`, a `QuantumBornGPT` with `use_hierarchical_16qubit=True, use_last_layer_softmax=True, logit_softcap=15.0` maintains exact Step-0 input-output self-overlap orthogonality $\max_w |\langle \psi_0(w) \mid W_{\text{soft}}[w] \rangle| < 10^{-6}$, while its readout weight rows $W_{\text{soft}}[v]$ are unconstrained during training (non-constant row norms $\|W_{\text{soft}}[v]\|_2$ with coefficient of variation $\text{CV}(\|W_{\text{soft}}\|_2) > 0.01$ after 20 steps) and its logits are strictly bounded in $(-15.0, +15.0)$.

- **Hypothesis H2 (Controlled Local RTX 5060 Ti FineWeb-Edu Benchmark: Last-Layer Softmax vs Pure Unit-Sphere POVM on `val_slice.bin`)**:
  - **Setup**: On our local **NVIDIA GeForce RTX 5060 Ti (16GB)**, train matched `16×32`-qubit + `256`-state hierarchical `QuantumBornGPT` models on non-repeating FineWeb-Edu tokens from `.cache/fineweb_100M/train_full_100M.bin` and evaluate on the exact `KellerJordan/modded-nanogpt` validation shard `.cache/fineweb_100M/val_slice.bin` (`fineweb_val_000000.bin`, sequential from `pos=0`, `50,257` active GPT-2 vocab):
    1. `w15_local_16x32_256hier_povm_control`: `use_hierarchical_16qubit=True, use_last_layer_softmax=False` (current Wave 15 unit-sphere POVM readout),
    2. `w15_local_16x32_256hier_last_layer_softmax`: `use_hierarchical_16qubit=True, use_last_layer_softmax=True, logit_softcap=15.0` (Calibrated Last-Layer Softmax),
    3. `w15_local_16x32_256hier_ngram_last_layer_softmax`: `use_hierarchical_16qubit=True, use_ngram_hash=True, use_last_layer_softmax=True, logit_softcap=15.0`.
  - **Quantitative Pass Criteria**:
    - `w15_local_16x32_256hier_last_layer_softmax` achieves **strictly lower held-out validation Cross-Entropy** (by **$\ge 0.15\text{ nats}$**), **strictly higher Top-5 accuracy**, and a **wider, better-calibrated $\text{Top-5} - \text{Top-1}$ gap** than `w15_local_16x32_256hier_povm_control` at identical step budget and learning rates, while maintaining $\ge 98\%$ of the training throughput (`tok/s`).

- **Hypothesis H3 (Wave 15 Hotfix Manifest & Selective 4-Session Relaunch Preserving `gpu_a100_w6_1`)**:
  - **Prediction**:
    - Creating [`configs/wave15_hotfix_softmax_fleet.json`](file:///home/festivity/agy/QuantumDLBlocks/configs/wave15_hotfix_softmax_fleet.json) and updating [`configs/wave15_fleet.json`](file:///home/festivity/agy/QuantumDLBlocks/configs/wave15_fleet.json) preserves `w15_900m_qborn_16x32_qubit_256hier_ngram_L8` on **`gpu_a100_w6_1`** completely untouched, while configuring the 4 hotfixed `900M`-token runs with `--last-layer-softmax --logit-softcap 15.0` on `gpu_g4_w6_1`, `gpu_g4_w6_2`, `gpu_g4_w6_3`, and `gpu_a100_w6_2` (where `gpu_a100_w6_2` runs `w15_900m_qborn_16x32_qubit_256hier_ngram_softmax_L8` as a direct head-to-head A/B counterpart to `gpu_a100_w6_1`!).

---

## 4. Planned Code & Configuration Changes

### 4.1 [`src/qdlblocks/layers/direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py)
- Extend `_DirectChunkedSoftcapLinearCEFn` and `direct_chunked_softcap_linear_ce` with `bias: torch.Tensor | None = None`:
  - Inside the forward chunk loop, after `raw = F.linear(xc_work, w_work)`, if `bias is not None`, add `bias` before applying `softcap * tanh(u / softcap)`.
  - If `need_dbias` is True, accumulate `dbias_accum.add_(logits.sum(dim=0))` after `logits` holds `dL/du = (p - y_onehot) * (1 - tanh^2(u / softcap)) * row_scale`, and return `dbias` in `backward`.
- Extend `sampled_softcap_linear_ce` with optional `bias: torch.Tensor | None = None`.

### 4.2 [`src/qdlblocks/layers/hierarchical_qubit.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/hierarchical_qubit.py)
- In `Hierarchical16QubitPOVMReadout`:
  - Add parameters:
    - `self.readout_rms_gain = nn.Parameter(torch.ones(self.embed_dim, dtype=torch.float32))`
    - `self.token_radial_scale = nn.Parameter(torch.ones(self.vocab_size, 1, dtype=torch.float32))`
    - `self.qubit_readout_gain = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))`
    - `self.hier_even_proj: nn.Parameter | None = nn.Parameter(torch.zeros(self.num_registers, self.register_dim, self.register_dim, dtype=torch.float32))` (when `self.num_higher_states > 0`)
  - Add method `format_readout_features(self, psi: torch.Tensor) -> torch.Tensor`:
    - Computes `F.rms_norm(psi, (self.embed_dim,), eps=self.eps) * self.readout_rms_gain.to(psi.dtype)`.
  - Add method `get_calibrated_softmax_weight(self, embedder: Hierarchical16QubitEmbedder, base_povm_head: torch.Tensor | None = None) -> torch.Tensor`:
    - Constructs the unconstrained `(vocab_size, embed_dim)` weight matrix $W_{\text{soft}}$ combining `base_povm_head`, unconstrained `povm_odd_qubits.reshape(V, flat_qubit_dim)`, and full-width `vocab_higher_affinity @ phi_tilde` (across both odd `1::2` and `hier_even_proj`-mapped even `0::2` coordinates), scaled by `self.token_radial_scale` and `1.0 / math.sqrt(self.embed_dim)` so Step-0 pre-softcap logits have unit variance $\mathcal{O}(1)$ and exact zero self-overlap $\langle \psi_0(w), W_{\text{soft}}[w] \rangle = 0$.
- In `initialize_hierarchical_16qubit_with_ppmi_svd`:
  - Fix `iso_mix` from `0.08 * (1.0 - alpha_f)` to `1.0 - alpha_f` so `u_even` and `u_odd` preserve full rank diversity alongside PPMI-SVD collocation structure (`collocation_cosine_ratio >= 6.0x`).
  - Also initialize `model.value_embeds` (when `getattr(model, "value_embeds", None) is not None`) using orthogonal parity-mixed PPMI-SVD embeddings.
  - Reset `readout.token_radial_scale.fill_(1.0)`, `readout.readout_rms_gain.fill_(1.0)`, and `readout.hier_even_proj.zero_()` so Step-0 self-overlap is `< 1e-6`.

### 4.3 [`src/qdlblocks/models/born_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/born_gpt.py), [`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py), & [`src/qdlblocks/benchmarks/model_card.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/model_card.py)
- Add `use_last_layer_softmax: bool = False` and `logit_softcap: float = 15.0` to `QuantumBornGPT.__init__`, `forward`, and `compute_logits`.
- Add CLI flags `--last-layer-softmax` (`action="store_true"`) and `--logit-softcap` (`type=float, default=15.0`) to `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`, `scripts/inspect_model.py`, and `src/qdlblocks/benchmarks/model_card.py`:
  - In `forward_backbone` (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`), when `getattr(m, "use_last_layer_softmax", False)` is True:
    - `phi_seed = m.hierarchical_povm_readout.format_readout_features(psi)` (or `F.rms_norm(psi, (m.embed_dim,), eps=m.eps)` if non-hierarchical),
    - `M_seed = m.hierarchical_povm_readout.get_calibrated_softmax_weight(m.hierarchical_qubit_embed, m.povm_readout.povm_head).to(phi_seed.dtype)` (or unconstrained `m.povm_readout.povm_head * (1.0 / math.sqrt(m.embed_dim))`),
    - `inv_t = torch.tensor(float(m.logit_softcap), device=phi_seed.device, dtype=torch.float32)`.
  - In `_chunked_povm_ce` and `evaluate_validation`:
    - When `getattr(model, "use_last_layer_softmax", False)` is True, dispatch directly to `direct_chunked_softcap_linear_ce(phi_2d, M_seed, flat_y, softcap=float(model.logit_softcap), chunk_size=16384, valid_vocab_size=FINEWEB_GPT2_VOCAB_SIZE, bias=log_prior)` (and in `evaluate_validation` diagnostic top-1/top-5 calculation, compute `raw = F.linear(zc_diag, M_seed[:v_active]) + bias; logits_diag = softcap * torch.tanh(raw / softcap)` and report `mean_beta_eval = float(logits_diag.detach().float().std(dim=-1).mean().item())`)!
  - In optimizer parameter group splitting (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`), when `last_layer_softmax` is True:
    - Exclude `hier_even_proj`, `readout_rms_gain`, `token_radial_scale`, and `qubit_readout_gain` from `StiefelMuon` (routing them to `AdamW` with `weight_decay=0.0`), and route `povm_head` / `povm_odd_qubits` / `vocab_higher_affinity` with `weight_decay=0.0` and `lr = 0.008` (matching `ClassicalModdedGPT`'s `lm_head`!).

### 4.4 Wave 15 Hotfix Manifests ([`configs/wave15_hotfix_softmax_fleet.json`](file:///home/festivity/agy/QuantumDLBlocks/configs/wave15_hotfix_softmax_fleet.json) & [`configs/wave15_fleet.json`](file:///home/festivity/agy/QuantumDLBlocks/configs/wave15_fleet.json))
- Create `configs/wave15_hotfix_softmax_fleet.json` containing ONLY the **4 hotfixed sessions** (`gpu_g4_w6_1`, `gpu_g4_w6_2`, `gpu_g4_w6_3`, `gpu_a100_w6_2`), leaving **`gpu_a100_w6_1` (`w15_900m_qborn_16x32_qubit_256hier_ngram_L8`) completely untouched**:
  1. `gpu_g4_w6_1`: `w15_900m_qborn_16x32_qubit_256hier_flagship_softmax_L8` (`L=8, d=1024, 4d-TwoBeam, 16x32 Qubit + 256-State Hierarchy + Calibrated Last-Layer Softmax, softcap=15.0`)
  2. `gpu_g4_w6_2`: `w15_900m_qborn_16x32_qubit_256hier_deep_L12_softmax` (`L=12, d=1024, H=8, d_h=128, 3VE, Skip7, 16x32 Qubit + 256-State Hierarchy + Calibrated Last-Layer Softmax, softcap=15.0`)
  3. `gpu_g4_w6_3`: `w15_900m_qborn_16x32_qubit_512hier_softmax_L8` (`L=8, d=1024, 4d-TwoBeam, 16x32 Qubit + 512-State Hierarchy + Calibrated Last-Layer Softmax, softcap=15.0`)
  4. `gpu_a100_w6_2`: `w15_900m_qborn_16x32_qubit_256hier_ngram_softmax_L8` (`L=8, d=1024, 4d-TwoBeam, 16x32 Qubit + 256-State Hierarchy + 262k Hashed 2/3-Gram + Calibrated Last-Layer Softmax, softcap=15.0` — exact A/B match on identical A100-80GB hardware against the untouched `w15_900m_qborn_16x32_qubit_256hier_ngram_L8` on `gpu_a100_w6_1`!)
- Update `configs/wave15_fleet.json` so the active catalog tracks all 5 runs (`w15_900m_qborn_16x32_qubit_256hier_ngram_L8` on `gpu_a100_w6_1` plus the 4 hotfixed `_softmax_` runs), while marking the 4 stopped pre-hotfix runs obsolete with clear reasons.

---

## 5. Operational Tooling & Script Opportunities (Mandatory Workflow Gate)

### 5.1 Existing Scripts Used
1. **Model Parameter Breakdown & Synthetic Smoke Test (`scripts/inspect_model.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 --num-higher-states 256 --last-layer-softmax --logit-softcap 15.0 --smoke-steps 2
   ```
2. **Selective 4-Session Hotfix Launch (`scripts/colab_fleet.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py register-cards --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave15_fleet.json
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py launch --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave15_hotfix_softmax_fleet.json
   ```
3. **Zero-Downtime Monitoring (`scripts/monitor_colab_runs.py`)**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once
   ```

### 5.2 `Build-Before-Launch` Script Enhancements
- Update [`scripts/inspect_model.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py) to support `--last-layer-softmax` and `--logit-softcap`.
- Verify that `scripts/colab_fleet.py launch --manifest configs/wave15_hotfix_softmax_fleet.json` only touches the sessions listed in `wave15_hotfix_softmax_fleet.json` (`gpu_g4_w6_1..3` and `gpu_a100_w6_2`) and never touches `gpu_a100_w6_1`.

### 5.3 `Nice-to-Have` Scripts (Logged to `scripts/BACKLOG.md` & Surfaced to User)
1. **`scripts/calibration_rank_histogram_probe.py`**: A standalone diagnostic CLI that evaluates a checkpoint or live model on `fineweb_val_000000.bin` and outputs a full rank-bucketed calibration table (`Rank 1`, `Ranks 2–5`, `Ranks 6–20`, `Ranks 21–100`, `Ranks 101+`, Expected Calibration Error `ECE`, and mean $-\log p(y_t)$ contribution per rank bucket) to visualize how `--last-layer-softmax` shifts probability mass into Ranks 2–50.

---

## 6. Verification Tests & Local RTX 5060 Ti Experimental Suite

1. **Unit & Mathematical Verification Suite (`tests/test_hypothesis_024_calibrated_last_layer_softmax.py`)**:
   - `test_direct_chunked_softcap_linear_ce_with_bias_exact_gradients`: Verifies forward loss and analytical gradients `(dx, dw, dbias)` against PyTorch autograd reference across `valid_vocab_size=50257` and `ignore_index=-1`.
   - `test_calibrated_hierarchical_softmax_readout_shapes_and_unconstrained_norms`: Verifies `get_calibrated_softmax_weight` and `format_readout_features` shapes, Step-0 zero self-overlap (`< 1e-6`), bounded logits in `(-15, +15)`, and non-unit unconstrained row norms after gradient steps.
   - `test_quantum_born_gpt_last_layer_softmax_forward_backward`: Verifies end-to-end `QuantumBornGPT(..., use_hierarchical_16qubit=True, use_last_layer_softmax=True, logit_softcap=15.0)` forward/backward pass and gradient flow into `unigram_qubits`, `higher_states`, `higher_interaction`, `povm_odd_qubits`, `vocab_higher_affinity`, `hier_even_proj`, `token_radial_scale`, and `povm_log_prior`.
   - `test_wave15_hotfix_manifest_leaves_a100_1_untouched`: Verifies `configs/wave15_hotfix_softmax_fleet.json` and `configs/wave15_fleet.json` Model Cards and confirms `gpu_a100_w6_1` is excluded from the hotfix launch manifest.
2. **Local RTX 5060 Ti Empirical Verification Script (`examples/34_verify_plan024_last_layer_softmax_experiments.py`)**:
   - Runs `H1`, `H2` (controlled multi-hundred-step FineWeb-Edu GPU benchmark on our local RTX 5060 Ti evaluating on `val_slice.bin` / `fineweb_val_000000.bin`), and `H3`, saves `.cache/hypothesis_experiments_024.json`, and ingests the verification telemetry into `.cache/benchmarks.db`.
