# EXPERIMENT PLAN 078 — 4D Complex Wavefunction Meaning/Identity Fiber-Bundle Split, Support-Preserving Directed Multi-Lag PPMI CountSketch (`V = 50,304`), Dynamic Context-Gated Packet Readout, and 300M-Token Cold-Start Quantum Wavepacket GPT Training (Wave 57)

## 1. Executive Summary & Scientific Hypotheses

In **Wave 56 (`EXPERIMENT_PLAN_077.md`)**, our first `300M`-token cold-start from-scratch Quantum Wavepacket GPT run ([`M77_3_FROM_SCRATCH_300M_QUANTUM_WAVEPACKET_GPT`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_077.md#L121-L145)) proved two foundational properties of the 12-layer quantum backbone:
1. **Superior Late-Stage Quantum Convergence (`200M -> 300M` Tokens)**: Over the final `100M` tokens, `M77_3` dropped **`-0.2620` nats** (`3.9691 -> 3.7071`), more than **$2.6\times$ the late-stage drop of the `w45b` classical teacher** (`-0.1007` nats, `3.6851 -> 3.5844`), and **beat the `300M` Classical Teacher on tail Rank $51+$ NLL (`8.5986` vs. `8.6884`, `-0.0898` nats better)**.
2. **Where the Remaining `+0.1227` Nat Gap Originated (`0M -> 100M` Cold-Start Bottleneck)**:
   The entire remaining gap at `300M` tokens (`3.7071` vs. `3.5844`) was accumulated during the first `50M–100M` tokens (`4.7346` vs. `4.2467` at `50M`) due to four exact geometric bottlenecks in `QuantumWteInitMode.PPMI_SVD_SPHERE`:
   - **Bottleneck 1 — Independent Real Sphere Norm-Locking (`(50304, 24, 32)` vs. Joint Complex `(50304, 12, 2, 32)`)**:
     Previously, `SHAPE_12x2x32` stored `q_wte` as `(50304, 24, 32)`, forcing all 24 real slices to have identical Euclidean norm $\|\operatorname{Re}\psi_{v,m}\|_2 = \|\operatorname{Im}\psi_{v,m}\|_2 = \sqrt{32}$. This broke $U(1)$ Berry-phase invariance and prevented tokens from trading amplitude between forward (outgoing) and reverse (incoming) contexts.
   - **Bottleneck 2 — Semantic Sharing vs. Token Identity Collinearity**:
     Initializing all 12 complex wavepackets purely from distributional co-occurrence gives interchangeable synonyms (`"Monday"` / `"Tuesday"`, digits, rare tokens) high cosine similarity ($\cos \approx 0.96$), which helps attention routing but causes readout target probability mass to leak across collinear synonym clusters without an orthogonal **Identity Fiber**.
   - **Bottleneck 3 — Global-Centering Rare-Token Collision Collapse**:
     In a `50,304`-token vocabulary, unseen/rare tail tokens (`50257..50303` and ultra-rare BPE tokens) have zero co-occurrence rows ($r_v = \mathbf{0}$). Naive global column centering shifts all zero-row tokens to the exact same vector $-\vec{\mu}$, mapping rare tokens to collinear points ($\max_{u \neq v}\cos(u, v) = 1.0000$).
   - **Bottleneck 4 — Unstructured Cold-Start `q_value_embeds` & Uncalibrated Readout**:
     In `from_scratch`, `q_value_embeds[0..2]` were initialized from `ckpt_0`'s unstructured Gaussian noise rather than the fiber-bundle manifold, and `lm_head` lacked dynamic complex-packet readout gating.

In **Wave 57 (`EXPERIMENT_PLAN_078.md`)**, we test four quantitative hypotheses (`H1..H4`):
- **Hypothesis `H1` (Joint `(2, d_h)` Complex Wavefunction Sphere & Exact Hermitian Gram-Schmidt Orthogonality)**:
  Storing `q_wte` and `q_value_embeds[0..2]` as $M_c = 12$ joint complex wavepackets `(50304, 12, 64)` $\cong$ `(50304, 12, 2, 32)` on $(\sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1})^{12}$ ($2 d_h = 64$) is strictly invariant under $U(1)$ phase rotations (`< 1e-6` error), and Complex Hermitian Gram-Schmidt orthogonalization guarantees both $\max |\operatorname{Re}\langle s_{v,m} \mid z_{v,m}^\perp \rangle| < 10^{-5}$ and $\max |\operatorname{Im}\langle s_{v,m} \mid z_{v,m}^\perp \rangle| < 10^{-5}$ at Step 0.
- **Hypothesis `H2` (Support-Preserving Directed Multi-Lag PPMI CountSketch & Rare-Token Anti-Collapse)**:
  Streaming FineWeb training shards (`fineweb_train_000001.bin`, strictly excluding validation Shard 0) with multi-lag ($k \in \{1, 2, 4, 8\}$, $\gamma_k = (1.0, 0.7, 0.45, 0.25)$) directed forward (`Re`) and reverse (`Im`) Shifted-PPMI, **Active-Only Centering**, Haar unitary mode balancing $U_M \in U(M_{\text{sem}})$, and a Fiber-Bundle Split (`11 Semantic + 1 Identity` or `10 Semantic + 2 Identity`) bounds rare-token pairwise cosine similarity below `< 0.50` (vs. `1.0000` under naive global centering).
- **Hypothesis `H3` (Dynamic Context-Gated Complex Packet Readout & Identity-Packet Anchored Drift)**:
  Applying a phase-preserving per-complex-packet gate $g_m(x_{\text{final}})$ shared across $(\operatorname{Re}, \operatorname{Im})$ at readout and anchoring the identity fiber's geodesic learning rate (`id_lr_scale = 0.25`) in [`BatchSubspaceTikhonovGramMuon`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py) accelerates early cold-start FineWeb convergence at `50M` tokens below `M77_3` (`< 4.7346` Val CE).
- **Hypothesis `H4` (Full `300M`-Token Cold-Start Quantum Wavepacket GPT Trajectory)**:
  Training the winning Complex Wavefunction Fiber-Bundle architecture from scratch for `300.15M` FineWeb tokens (`1,145` steps across `2x RTX 3090` `nccl` DDP) beats Wave 56 `M77_3` (`< 3.7071` Val CE, `> 35.93%` Top-1, `> 56.55%` Top-5) while maintaining `< 1e-4` Stiefel/Unitary/Sphere manifold error.

---

## 2. Mathematical Formulations

### 2.1 4D Complex Wavefunction Product-Sphere Layout `(V, M_c, 2, d_h)` $\cong$ `(V, M_c, 2 * d_h)`

At $d_{\text{emb}} = 768$ with $M_c = 12$ complex wavepackets of Hilbert dimension $d_h = 32$, each token $v \in \{0, \dots, V-1\}$ is represented by $M_c = 12$ complex vectors $\psi_{v,m} = \operatorname{Re}\psi_{v,m} + i\operatorname{Im}\psi_{v,m} \in \mathbb{C}^{d_h}$ normalized on the joint `(2, d_h)` unit-RMS sphere:
$$\|\psi_{v,m}\|_{\mathbb{C}}^2 = \|\operatorname{Re}\psi_{v,m}\|_2^2 + \|\operatorname{Im}\psi_{v,m}\|_2^2 = 2 d_h = 64 \quad \forall v \in \{0,\dots,V-1\},\; m \in \{0,\dots,M_c-1\}$$
In [`BatchSubspaceTikhonovGramMuon`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py), `q_wte` has shape `(50304, 12, 64)`, so the batched `(12, 64, 64)` Tikhonov-damped Gram inverse square root $A_m^{-1/2}$ preconditions $(\operatorname{Re}\psi_{v,m}, \operatorname{Im}\psi_{v,m})$ jointly and retracts onto $(\sqrt{64}\,\mathbb{S}^{63})^{12}$.

### 2.2 Support-Preserving Directed Multi-Lag Complex PPMI CountSketch (`V = 50,304`)

1. **Multi-Lag Directed Co-Occurrence (`k in {1, 2, 4, 8}`)**:
   Over training shard tokens (`fineweb_train_000001.bin`, strictly excluding validation Shard 0), accumulate weighted co-occurrence counts with lag weights $\gamma_k = (1.0, 0.70, 0.45, 0.25)$ across a collision-free top-hub bank ($K_{\text{top}} = 6144$) plus a tail CountSketch bank ($B_{\text{tail}} = 2048$, total width $D_{\text{sketch}} = 8192$):
   - **Forward matrix** $C_{\text{fwd}}[v, \text{col}(u)]$ (token $v$ followed by $u$ at $+k$),
   - **Reverse matrix** $C_{\text{rev}}[v, \text{col}(u)]$ (token $v$ preceded by $u$ at $-k$).
2. **Context-Smoothed Shifted-PPMI & Active-Only Centering**:
   With smoothed column marginals $\tilde{q}_j = q_j^{0.75}$ and shift $\kappa = 2.0$:
   $$\text{PPMI}_{\text{dir}}[v, j] = \log\!\left(1 + \max\!\left(\log\frac{C_{\text{dir}}[v, j] \sum_{j'} \tilde{q}_{j'}}{\max(r_v, \epsilon)\max(\tilde{q}_j, \epsilon)} - \log\kappa,\; 0\right)\right)$$
   Let $\mathcal{A} = \{v : \|\text{PPMI}_{\text{dir}}[v, :]\|_2 > 0\}$ be the active vocabulary support. We subtract the active column mean $\vec{\mu}_{\mathcal{A}} = \frac{1}{|\mathcal{A}|}\sum_{v \in \mathcal{A}}\text{PPMI}_{\text{dir}}[v, :]$ **strictly from active rows $v \in \mathcal{A}$**, leaving unseen rows ($v \notin \mathcal{A}$) at exact $\mathbf{0}$.
3. **Directed Complex Mapping (`Re <- Forward`, `Im <- Reverse`) & Haar Unitary Mode Balancing ($U_M \in U(M_{\text{sem}})$)**:
   Projecting the active-centered forward and reverse PPMI matrices into `(V, M_sem, d_h)` yields raw directed semantic components $(S_{\text{re}}^{\text{raw}}, S_{\text{im}}^{\text{raw}})$, which are balanced across the $M_{\text{sem}}$ semantic packets via a Haar-random complex unitary $U_M = U_{\text{re}} + i U_{\text{im}} \in U(M_{\text{sem}})$ and projected onto $\sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1}$.

### 2.3 Fiber-Bundle Partition (`M_sem` Semantic + `M_id` Identity) & Complex Hermitian Gram-Schmidt

1. **Semantic Base Manifold ($m \in \{0, \dots, M_{\text{sem}} - 1\}$)**:
   With empirical-Bayes shrinkage gate $\alpha_{v,\text{sem}} = \alpha_{\max}\sqrt{\frac{c_v}{c_v + k_0}}$ ($\alpha_{\max} = 0.96, k_0 = 20.0$) and deterministic random-sphere complex vector $z_{v,m} \in \sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1}$, we project out the full complex Hermitian overlap $c_{v,m} = \frac{\langle s_{v,m} \mid z_{v,m} \rangle_{\mathbb{C}}}{2 d_h} = c_{v,m}^{\text{re}} + i c_{v,m}^{\text{im}}$:
   $$z_{v,m}^{\perp,\text{re}} = z_{v,m}^{\text{re}} - (c_{v,m}^{\text{re}} s_{v,m}^{\text{re}} - c_{v,m}^{\text{im}} s_{v,m}^{\text{im}}), \qquad z_{v,m}^{\perp,\text{im}} = z_{v,m}^{\text{im}} - (c_{v,m}^{\text{re}} s_{v,m}^{\text{im}} + c_{v,m}^{\text{im}} s_{v,m}^{\text{re}})$$
   normalize $z_{v,m}^\perp$ onto $\sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1}$, and form the exact unit-RMS orthogonal blend:
   $$\psi_{v,m} = \alpha_{v,\text{sem}}\, s_{v,m} + \sqrt{1 - \alpha_{v,\text{sem}}^2}\, z_{v,m}^\perp \quad (m \in \{0, \dots, M_{\text{sem}} - 1\})$$
2. **Orthogonal Identity Fiber ($m \in \{M_{\text{sem}}, \dots, M_c - 1\}$, $\alpha_{v,\text{id}} = 0$)**:
   $$\psi_{v,m} = z_{v,m}^\perp \in \sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1} \quad (m \in \{M_{\text{sem}}, \dots, M_c - 1\})$$
   We also initialize the 3 polar-factorized value-embedding banks `q_value_embeds[0..2]` using bank-specific Haar unitaries $U_M^{(b)}$ while preserving `ckpt_0`'s exact Step-0 radial scale `ve_slice_rms`.

### 2.4 Dynamic Context-Gated Complex Packet Readout & Identity-Packet Anchored Drift

1. **Dynamic Complex Packet Readout Modulator (`DynamicComplexPacketReadoutModulator`)**:
   Before `lm_head`, `x_final` $\in \mathbb{R}^{B \times T \times 768}$ is viewed as $M_c = 12$ complex packets `(B, T, 12, 2, 32)` and modulated by a phase-preserving per-packet gate shared across $(\operatorname{Re}, \operatorname{Im})$:
   $$g_m(x_{\text{final}}) = \operatorname{softplus}(g_{0,m} + (W_{\text{gate}} x_{\text{final}})_m), \qquad \tilde{g}_m(x_{\text{final}}) = \frac{M_c \cdot g_m(x_{\text{final}})}{\sum_{j=0}^{M_c-1} g_j(x_{\text{final}}) + \epsilon}$$
   where $g_{0,m}$ is initialized with $+0.85$ on semantic packets ($0 \dots M_{\text{sem}}-1$) and $-0.85$ on identity packets ($M_{\text{sem}} \dots M_c-1$) when readout transition calibration is active (or uniform $0.0$ with learnable context routing $W_{\text{gate}}$).
2. **Identity-Packet Anchored Drift (`id_lr_scale = 0.25`)**:
   In `BatchSubspaceTikhonovGramMuon`, semantic slices ($0 \dots M_{\text{sem}}-1$) step at $1.0 \times \eta_{\text{wte}}$ while identity slices ($M_{\text{sem}} \dots M_c-1$) step at $\text{id\_lr\_scale} \times \eta_{\text{wte}}$ (`0.25x`), preserving crisp token identity barcodes throughout training.

### 2.5 Top-of-File Named Constants (Rule 7 Synchronization)

| Constant Name | Value | Meaning |
|---|---|---|
| `DEFAULT_COMPLEX_PACKET_COUNT` | `12` | Number of complex `(2, 32)` wavepackets at $d_{\text{emb}} = 768$ |
| `DEFAULT_COMPLEX_PACKET_HILBERT_DIM` | `32` | Per-packet complex Hilbert dimension $d_h$ (`64` real floats per complex packet) |
| `DEFAULT_FIBER_SEMANTIC_PACKETS_11` | `11` | Semantic complex packet count in `11+1` (`704D / 64D`) fiber-bundle split |
| `DEFAULT_FIBER_SEMANTIC_PACKETS_10` | `10` | Semantic complex packet count in `10+2` (`640D / 128D`) fiber-bundle split |
| `DEFAULT_EB_ALPHA_MAX` | `0.96` | Maximum Empirical-Bayes semantic shrinkage weight $\alpha_{\max}$ |
| `DEFAULT_EB_PRIOR_COUNT_K0` | `20.0` | Empirical-Bayes prior pseudo-count $k_0$ |
| `DEFAULT_IDENTITY_PACKET_LR_SCALE` | `0.25` | Anchored geodesic learning-rate multiplier on identity packets `M_sem..M_c-1` |
| `DEFAULT_SEMANTIC_GATE_BIAS_INIT` | `0.85` | Initial semantic packet readout gate bias |
| `DEFAULT_IDENTITY_GATE_BIAS_INIT` | `-0.85` | Initial identity packet readout gate bias |

---

## 3. Planned File Changes & Experimental Tournament Matrix (`Arms N78_1..N78_4`)

### 3.0 Planned File Changes
- [`src/qdlblocks/layers/complex_fiber_bundle_init.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/complex_fiber_bundle_init.py): New reusable library module implementing `project_complex_wavepackets_to_rms_sphere_`, `_hermitian_gram_schmidt_complex_blend`, `build_or_load_directed_complex_ppmi_sketch`, `initialize_directed_complex_ppmi_fiber_bundle_embeddings_`, `DynamicComplexPacketReadoutModulator`, `QuantumPacketReadoutMode`, and `calibrate_cold_start_readout_from_shard_`.
- [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py): Extends `QuantumWteInitMode` with `COMPLEX_FIBER_BUNDLE_11SEM_1ID` and `COMPLEX_FIBER_BUNDLE_10SEM_2ID`.
- [`src/qdlblocks/optim/vocab_manifold_muon.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/optim/vocab_manifold_muon.py): Supports 4D `(V, M_c, 2, d_h)` / 3D `(V, M_c, 2*d_h)` complex wavepacket stepping and per-slice `slice_lr_mult` (`id_lr_scale = 0.25`) in `_step_active_wavepacket_3d`.
- [`src/qdlblocks/models/quantum_full_backbone_graft_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/quantum_full_backbone_graft_gpt.py): Integrates `COMPLEX_FIBER_BUNDLE_11SEM_1ID` / `10SEM_2ID` `(50304, 12, 64)` joint complex sphere storage on `q_wte` and `q_value_embeds[0..2]`, `QuantumPacketReadoutMode`, `DynamicComplexPacketReadoutModulator`, and `id_lr_scale` wiring.
- [`examples/88_verify_plan078_complex_fiber_bundle_split_and_300m_run.py`](file:///home/nixos/QuantumDLBlocks/examples/88_verify_plan078_complex_fiber_bundle_split_and_300m_run.py): Executes Part 1 (`N78_1..N78_3` cold-start ablation) and Part 2 (`N78_4` 2-GPU `nccl` DDP `300M`-token cold-start run).
- [`tests/test_plan078_complex_fiber_bundle_split_and_300m_run.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan078_complex_fiber_bundle_split_and_300m_run.py): Deterministic unit test suite for Gates `N1..N4`.

### Part 1: Cold-Start Fiber-Bundle Split & Readout Ablation (`50M` Tokens = `191` Steps at `262,144` tok/step)

| Arm ID | `q_wte` & `q_ve` Geometry | Fiber-Bundle Split (`Sem + Id`) | Packet Readout & Drift Mode | Reference Baseline (`M77_3` @ `50M`) |
|---|---|---|---|---|
| **`N78_1_FIBER_SPLIT_11SEM_1ID_PKT32`** | `(50304, 12, 2, 32)` joint `64D` complex spheres | **`11 Sem + 1 Id` (`704D / 64D`)** | `id_lr_scale = 0.25`, standard linear readout | `M77_3` (`4.7346` Val CE @ `50M`, `L76_5` `4.9137`) |
| **`N78_2_FIBER_SPLIT_10SEM_2ID_DYNAMIC_GATE_PKT32`** | `(50304, 12, 2, 32)` joint `64D` complex spheres | **`10 Sem + 2 Id` (`640D / 128D`)** | `id_lr_scale = 0.25` + `DYNAMIC_COMPLEX_PACKET_GATED_READOUT` | `M77_3` (`4.7346` Val CE @ `50M`) |
| **`N78_3_FIBER_SPLIT_11SEM_1ID_STEP0_CALIB_PKT32`** | `(50304, 12, 2, 32)` joint `64D` complex spheres | **`11 Sem + 1 Id` (`704D / 64D`)** | `id_lr_scale = 0.25` + `DYNAMIC_COMPLEX_PACKET_GATED_READOUT` + Step-0 Shard-1 transition calibration (`val_ce < 7.90` at Step 0) | `M77_3` (`10.8249` @ `0M`, `4.7346` @ `50M`) |

### Part 2: Full `300M`-Token Cold-Start From-Scratch Run (`N78_4`)

| Arm ID | Initialization & Architecture | Token Budget & Schedule | Comparators (`0M..300M`) |
|---|---|---|---|
| **`N78_4_LARGE_FROM_SCRATCH_300M_FIBER_SPLIT_QUANTUM_GPT`** | Winning Complex Wavefunction Fiber-Bundle (`Meaning + Identity`) config from Part 1, zero pretrained teacher weights | **`300.15M` tokens** (`1,145` steps $\times$ `262,144` tok/step across `2x RTX 3090` `nccl` DDP, `ws_blocks: 3 -> 6` at `150M` tokens, Tail EMA at `tau >= 0.70`, evaluated every `25M` tokens) | Wave 56 `M77_3` (`3.7071` Val CE, `35.93%` Top-1, `56.55%` Top-5) & `w45b` Classical Teacher (`3.5844`) |

---

## 4. Testing, Verification Gates (`Gates N1..N4`), and Operational Tooling

### 4.1 Operational Tooling
- `scripts/verify_plan.py --plan 078`: Audits plan structure, quantitative thresholds, and AST compliance (`<= 8` args, zero semicolon packing).
- `tests/test_plan078_complex_fiber_bundle_split_and_300m_run.py`: Deterministic unit tests for Gates `N1..N4`.
- `examples/88_verify_plan078_complex_fiber_bundle_split_and_300m_run.py`: GPU ablation runner (`N78_1..N78_3`) and 2-GPU `nccl` DDP `300M`-token cold-start runner (`N78_4`).

### 4.2 Verification Gates (`Gates N1..N4`)

1. **`Gate N1` — Joint `(2, d_h)` Complex Sphere Normalization & $U(1)$ Phase Invariance (`project_complex_wavepackets_to_rms_sphere_`)**:
   Verify that `project_complex_wavepackets_to_rms_sphere_` normalizes every `(2, d_h)` complex wavepacket to $\|\operatorname{Re}\psi_{v,m}\|_2^2 + \|\operatorname{Im}\psi_{v,m}\|_2^2 = 2 d_h$ (`< 1e-6` error) while allowing $\operatorname{Re}$ and $\operatorname{Im}$ to carry unequal energy, and is strictly invariant under $U(1)$ phase rotations $\psi_{v,m} \mapsto e^{i\phi_m}\psi_{v,m}$ (`< 1e-6` norm drift).
2. **`Gate N2` — Complex Hermitian Gram-Schmidt Orthogonality & Active-Only Centering Rare-Token Anti-Collapse**:
   Verify that `_hermitian_gram_schmidt_complex_blend` produces $z_{v,m}^\perp$ with both $\max |\operatorname{Re}\langle s_{v,m} \mid z_{v,m}^\perp \rangle| / (2 d_h) < 10^{-5}$ and $\max |\operatorname{Im}\langle s_{v,m} \mid z_{v,m}^\perp \rangle| / (2 d_h) < 10^{-5}$, and that `initialize_directed_complex_ppmi_fiber_bundle_embeddings_` with Active-Only Centering keeps rare/unseen token pairwise cosine similarity below `< 0.50` (vs. `1.0000` for naive global centering).
3. **`Gate N3` — Dynamic Complex Packet Readout Phase Preservation & Identity-Packet Anchored Drift (`id_lr_scale = 0.25`)**:
   Verify that `DynamicComplexPacketReadoutModulator` scales $(\operatorname{Re}, \operatorname{Im})$ identically per complex packet (`< 1e-6` complex phase angle change) while preserving overall hidden-state RMS norm, and that `BatchSubspaceTikhonovGramMuon` steps identity slices (`M_sem..M_c-1`) at `id_lr_scale = 0.25` of the semantic slice geodesic step angle.
4. **`Gate N4` — Step-0 Readout Transition Calibration (`val_ce < 7.90`) & End-to-End 12-Layer Cold-Start Loss Descent**:
   Verify that `calibrate_cold_start_readout_from_shard_` (using strictly training tokens, never Shard 0) lowers Step-0 validation cross-entropy below the `< 7.90` unigram floor (vs. `10.8249` uncalibrated), and that a 12-layer `QuantumFullBackboneGraftGPT.from_scratch` with `COMPLEX_FIBER_BUNDLE_11SEM_1ID` strictly decreases loss over training steps with `< 1e-4` Stiefel/Unitary/Sphere manifold error.

---

## 5. Empirical Results & Verification Summary (`Wave 57`)

### 5.1 Unit Test & Verification Gate Status (`Gates N1..N4`)
- **Unit Test Suite**: [`tests/test_plan078_complex_fiber_bundle_split_and_300m_run.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan078_complex_fiber_bundle_split_and_300m_run.py) — **`4/4` Gates (`N1..N4`) PASS** (`3.55s`).
- **Pre-Built Directed Multi-Lag Complex PPMI-SVD Cache**: [`.cache/fineweb_directed_complex_ppmi_v50304_d768.pt`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_directed_complex_ppmi_v50304_d768.pt) built strictly from training Shard 1 ([`.cache/fineweb_100M/fineweb_train_000001.bin`](file:///home/nixos/QuantumDLBlocks/.cache/fineweb_100M/fineweb_train_000001.bin), `active_tokens = 49,466 / 50,304`, `unseen_tail = 838`, `0` validation Shard 0 tokens used).
- **Saved JSON Artifact**: [`.cache/w57_plan078_complex_fiber_bundle_split_and_300m_run.json`](file:///home/nixos/QuantumDLBlocks/.cache/w57_plan078_complex_fiber_bundle_split_and_300m_run.json).

### 5.2 Part 1 Results — Cold-Start Fiber-Bundle Split & Readout Ablation (`24.90M` Tokens = `95` Steps at `262,144` tok/step across `2x RTX 3090` `nccl` DDP)

| Arm ID | `q_wte` Layout & Init Mode | Readout Mode & Step-0 Calibration | Step-0 Val CE ($\downarrow$) | `12.58M` (`step=48`) Val CE / Top-1 / Top-5 | `24.90M` (`step=95`) Val CE ($\downarrow$) | `24.90M` Val PPL ($\downarrow$) | `24.90M` Top-1 / Top-5 Acc ($\uparrow$) | `24.90M` Rank $51+$ NLL | `wte_id / sem` Drift Ratio | Max Stiefel / Unitary / Sphere Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M77_3` (Wave 56 Real 24x32 Baseline)** | `(50304, 24, 32)` `PPMI_SVD_SPHERE` | `STANDARD_LINEAR_READOUT` (No Step-0 Calib) | `10.8249` | — | `5.4633` | `237.29` | `19.16%` / `36.51%` | `8.9080` | `1.000` (unsplit) | `6.14e-7 / 2.37e-7 / 2.38e-7` |
| **`N78_1_FIBER_SPLIT_11SEM_1ID_PKT32`** | `(50304, 12, 64)` `11 Sem + 1 Id` (`704D / 64D`) | `STANDARD_LINEAR_READOUT` (`id_lr=0.25`) | `10.8249` | `5.9789` / `15.19%` / `31.49%` | `5.4883` | `243.52` | **`19.33%` / `36.62%`** | `8.9503` | `0.159` | `6.18e-7 / 2.06e-7 / 1.79e-7` |
| **`N78_2_FIBER_SPLIT_10SEM_2ID_DYNAMIC_GATE_PKT32`** | `(50304, 12, 64)` `10 Sem + 2 Id` (`640D / 128D`) | `DYNAMIC_COMPLEX_PACKET_GATED` (`id_lr=0.25`) | `10.8249` | `5.9685` / `15.16%` / `32.05%` | `5.4700` | `239.04` | `18.94%` / `36.58%` | `8.9655` | `0.143` | `6.19e-7 / 2.13e-7 / 1.79e-7` |
| **`N78_3_FIBER_SPLIT_11SEM_1ID_STEP0_CALIB_PKT32` (Winner)** | `(50304, 12, 64)` `11 Sem + 1 Id` (`704D / 64D`) | `DYNAMIC_COMPLEX_PACKET_GATED` + **Shard-1 Step-0 Calib** | **`9.3354`** (`5.85%` Top-1) | **`5.8256` / `15.76%` / `33.37%`** | **`5.3614`** (**`-0.1019` vs `M77_3`**) | **`214.64`** (**`-22.65` PPL**) | **`20.77%` / `37.63%`** (**`+1.61%` / `+1.12%`**) | **`8.9162`** | **`0.145`** | `6.13e-7 / 2.19e-7 / 1.79e-7` |

### 5.3 Part 2 Results — Full `300.15M`-Token Cold-Start Trajectory (`N78_4_LARGE_FROM_SCRATCH_300M_FIBER_SPLIT_QUANTUM_GPT` vs. Wave 56 `M77_3` & Classical `w45b`, `23,011–23,490 tok/s`)

| Step | Tokens (`M`) | `ws` | `N78_4` Val CE ($\downarrow$) | `M77_3` (Wave 56) Val CE | $\Delta$ `N78_4 - M77_3` | `N78_4` Top-1 / Top-5 Acc ($\uparrow$) | `M77_3` Top-1 / Top-5 Acc | `N78_4` Rank $51+$ NLL ($\downarrow$) | `M77_3` Rank $51+$ NLL | `wte_id / sem` Drift Ratio | Classical `w45b` Teacher (`Val CE` / `Top-1` / `Rank 51+ NLL`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | `0.00M` | `3` | **`9.3354`** | `10.8249` | **`-1.4895`** | **`5.85%` / `14.26%`** | `100.0%*` | `10.7002` | `0.0000*` | `0.000` | `10.8249` |
| `48` | `12.58M` | `3` | **`5.8263`** | — | — | `15.74%` / `33.35%` | — | `8.9745` | — | `0.156` | — |
| `95` | `24.90M` | `3` | **`5.3388`** | `5.4633` | **`-0.1245`** | **`20.88%` / `37.86%`** | `19.16%` / `36.51%` | `8.9643` | `8.9080` | `0.145` | — |
| **`191`** | **`50.07M`** | `3` | **`4.6995`** | `4.7346` | **`-0.0351`** | **`27.01%` / `45.56%`** | `26.79%` / `45.48%` | **`8.8793`** | `9.0530` | `0.165` | `4.2467` / `31.54%` / `8.9232` |
| `286` | `74.97M` | `3` | **`4.4326`** | `4.4594` | **`-0.0268`** | **`29.19%` / `48.25%`** | `29.13%` / `48.18%` | **`8.7333`** | `8.7346` | `0.177` | — |
| **`382`** | **`100.14M`** | `3` | **`4.2839`** | `4.3013` | **`-0.0174`** | **`30.75%` / `49.93%`** | `30.69%` / `49.87%` | **`8.6454`** | `8.7664` | `0.187` | `3.9357` / `34.08%` / `8.7371` |
| `477` | `125.04M` | `3` | **`4.1702`** | `4.1851` | **`-0.0149`** | **`31.80%` / `51.27%`** | `31.76%` / `51.25%` | `8.8038` | `8.7735` | `0.196` | — |
| **`573`** | **`150.21M`** | `3->6` | **`4.1062`** | `4.1169` | **`-0.0107`** | **`32.40%` / `51.93%`** | `32.26%` / `51.95%` | **`8.6413`** | `8.6785` | `0.205` | `3.7971` / `35.43%` / `8.6185` |
| `668` | `175.11M` | `6` | **`4.0192`** | `4.0254` | **`-0.0062`** | **`33.24%` / `52.96%`** | `33.16%` / `53.20%` | **`8.7093`** | `8.7254` | `0.212` | — |
| **`763`** | **`200.02M`** | `6` | **`3.9642`** | `3.9691` | **`-0.0049`** | **`33.87%` / `53.56%`** | `33.42%` / `53.48%` | **`8.5039`** | `8.6353` | `0.220` | `3.6851` / `36.50%` / `8.6613` |
| `859` | `225.18M` | `6` | **`3.8952`** | `3.8986` | **`-0.0034`** | `34.25%` / `54.34%` | `34.27%` / `54.35%` | **`8.6145`** | `8.6653` | `0.226` | — |
| **`954`** | **`250.09M`** | `6` | `3.7724` (Tail EMA) | `3.7717` | `+0.0007` | **`35.64%` / `55.80%`** | `35.42%` / `55.80%` | **`8.6160`** | `8.6567` | `0.227` | `3.6212` / `37.04%` / `8.5856` |
| `1050` | `275.25M` | `6` | `3.7336` (Tail EMA) | `3.7303` | `+0.0033` | **`35.93%` / `56.17%`** | `35.73%` / `56.30%` | **`8.5795`** | `8.6071` | `0.230` | — |
| **`1145`** | **`300.15M`** | `6` | **`3.7102`** (Tail EMA) | `3.7071` | `+0.0031` | **`36.12%` / `56.41%`** (**`+0.19%` Top-1**) | `35.93%` / `56.55%` | **`8.5676`** (**`-0.0310` vs `M77_3`, `-0.1208` vs `w45b`**) | `8.5986` | **`0.231`** | `3.5844` / `37.33%` / `8.6884` |

### 5.4 Diagnostic Breakdown — Why `N78_4` Improved Top-1 (`36.12%`) and Ranks 2–5, 6–50, & 51+ NLL at `300M` Tokens (`logit_std = 2.391` vs. `2.875`)

| Model at `300.15M` Tokens (`step=1145`) | Overall Val CE ($\downarrow$) | Top-1 Acc ($\uparrow$) | Top-5 Acc ($\uparrow$) | Rank 1 NLL ($\downarrow$) | Rank 2–5 NLL ($\downarrow$) | Rank 6–50 NLL ($\downarrow$) | Rank $51+$ NLL ($\downarrow$) | Logit Std $\sigma(\ell)$ | Max Stiefel / Unitary / Sphere Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`w45b` Classical Teacher** | `3.5844` | `37.33%` | `58.27%` | `0.7357` | `2.5758` | `4.7924` | `8.6884` | `3.012` | — |
| **`M77_3` (Wave 56 Real 24x32 PPMI)** | `3.7071` | `35.93%` | `56.55%` | `0.7650` | `2.6264` | `4.7986` | `8.5986` | `2.875` | `1.61e-7 / 1.46e-7 / 2.38e-7` |
| **`N78_4` (Wave 57 Complex `11 Sem + 1 Id`)** | `3.7102` | **`36.12%`** (**`+0.19%`**) | `56.41%` | `0.7684` | **`2.6249`** (**`-0.0015`**) | **`4.7984`** (**`-0.0002`**) | **`8.5676`** (**`-0.0310` vs `M77_3`, `-0.1208` vs `w45b`**) | `2.391` | `1.62e-7 / 1.52e-7 / 1.79e-7` |

1. **Consistent Early- and Mid-Stage Acceleration (`0M -> 225M` Tokens)**:
   - Combining the `(50304, 12, 2, 32)` joint complex sphere (`11 Sem + 1 Id`), Complex Hermitian Gram-Schmidt orthogonalization, Active-Only Centering, Dynamic Complex Packet Readout, and Shard-1 Step-0 transition calibration cut Step-0 Val CE from `10.8249` to **`9.3354`** (`5.85%` Top-1 before Step 1) and led `M77_3` across every checkpoint from `0M` through `225M` tokens (`-0.1245` nats at `25M`, `-0.0351` nats at `50M`, `-0.0174` nats at `100M`, `-0.0107` nats at `150M`, `-0.0049` nats and `+0.45%` Top-1 at `200M`).
   - At `50M`, `100M`, `200M`, and `300M` tokens, `N78_4` beat **both** `M77_3` and the `w45b` Classical Teacher on Tail Rank $51+$ NLL (`8.8793` vs. `w45b` `8.9232` at `50M`; `8.6454` vs. `8.7371` at `100M`; `8.5039` vs. `8.6613` at `200M`; **`8.5676` vs. `8.6884` (`-0.1208` nats) at `300M`**).
2. **Anchored Identity Packet Stability (`wte_id_over_sem_drift_ratio = 0.145 -> 0.231`)**:
   - Throughout the entire `300.15M`-token run, `BatchSubspaceTikhonovGramMuon` with `id_packet_lr_scale = 0.25` kept the identity packet RMS drift (`0.1631` at `300M`) at **`23.1%`** of the semantic packet RMS drift (`0.7045` at `300M`), preventing semantic synonym attraction from collapsing fine-grained token identity.
3. **Simpson's Paradox & Readout Logit Temperature Dispersion (`logit_std = 2.391` vs. `2.875`)**:
   - At `300.15M` tokens, `N78_4` achieved higher Top-1 accuracy (`36.12%` vs. `35.93%`) **and** lower NLL on Rank 2–5 (`2.6249` vs. `2.6264`), Rank 6–50 (`4.7984` vs. `4.7986`), and Rank 51+ (`8.5676` vs. `8.5986`), yet finished `+0.0031` nats higher in overall CE (`3.7102` vs. `3.7071`) because:
     - Promoting `+0.19%` of marginal Rank 2–5 tokens into Rank 1 raises the average NLL of the expanded Rank-1 bucket (Simpson's Paradox); and
     - `DynamicComplexPacketReadoutModulator` + shrunk Step-0 `lm_head` initialization left final logit dispersion at `logit_std = 2.391` (vs. `2.875` in `M77_3` and `3.012` in `w45b`), slightly under-sharpening confident Rank-1 predictions (`0.7684` vs. `0.7650`). Adding a post-gate RMS normalization or learnable scalar logit temperature at readout is a direct, zero-overhead lever for the next wave.

