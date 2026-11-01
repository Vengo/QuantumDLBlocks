# EXPERIMENT PLAN 068: Biorthogonal SVD-Stiefel Wavepacket Projections, Rank-Compressed Stiefel Manifolds ($r \in \{128, 256, 512, 768\}$), and Fubini-Study Self-State Orthogonalization for Sub-`300M` Classical Teacher Reproduction & Exceeding (`Wave 52D`)

**Date**: 2026-11-01  
**Author**: Antigravity (QuantumDLBlocks Wave 52D)  
**Predecessors**: [`EXPERIMENT_PLAN_065.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_065.md), [`EXPERIMENT_PLAN_066.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_066.md), [`EXPERIMENT_PLAN_067.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_067.md), [`docs/SESSION_WRAPUP_WAVE50_QUANTUM_WAVEPACKET_GPT.md`](file:///home/nixos/QuantumDLBlocks/docs/SESSION_WRAPUP_WAVE50_QUANTUM_WAVEPACKET_GPT.md)  
**Status**: Approved (`0 Blockers, 0 Majors, 0 Nits` — Independent Reviewer `035f390f-cf8b-4cd9-b7fb-acf6643e4129`)

---

## 1. Executive Summary & Root-Cause Diagnosis of the `Plan 067` Asymptotic Floor (`3.6220` vs. `3.5845`)

In **Plan 067 (`Wave 52C`)**, combining `(V, 2, 512)` wavefunctions, rank-3 causal ensemble density-matrix I/O (`CAUSAL_ENSEMBLE_DM_IO`), and persistent floored co-distillation (`PERSISTENT_COSINE_FEAT_AND_CE`) enabled `P2_SUB300M_RECOVERY_VS_300M_TEACHER` (`0.33M` Layer-0 parameters — `21.5x` smaller than classical Layer 0) to reach **`val_ce = 3.6220`, `top1_acc = 36.97%`, `top5_acc = 58.03%`** in `1,200` steps (`9.83M` tokens = `3.28%` of `300M` tokens), within `+0.0375` nats of the `300M`-token Classical Teacher (`3.5845`, `37.31%` Top-1), while `P1_SUB50M_RECOVERY_VS_50M_TEACHER` reached **`val_ce = 4.2678`** (`+0.0211` nats above the `50M`-token Classical Teacher `4.2467`).

Direct mathematical and GPU ablation analysis of the `300M` and `50M` Classical Teachers' Layer 0 (`classical.blocks[0]`) vs. [`ConfigurableWavepacketStatMechBlock`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py) identified **four exact structural bottlenecks** responsible for the remaining `+0.0375`-nat gap:

1. **Bottleneck 1 — Rank-1 Kronecker Submatrix Bottleneck in `apply_factored_wavepacket_stiefel`**:
   - In [`apply_factored_wavepacket_stiefel(x, w_pkt, m_mode)`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/wavepacket_statmech_block.py#L207-L221), the effective $D \times D$ linear operator mapping input coordinate $(m, d_1)$ to output coordinate $(n, d_2)$ is:
     $$W_{\text{fact}}[(n, d_2), (m, d_1)] = m_{\text{mode}}[d_2, m, n] \cdot w_{\text{pkt}}[m, d_1, d_2]$$
   - For every $(m, d_2)$ subblock of shape $M_{\text{pkt}} \times d_{\text{pkt}}$ (e.g., $48 \times 16$), $W_{\text{fact}}$ is a **rank-1 outer product** between $m_{\text{mode}}[d_2, m, :] \in \mathbb{R}^{M_{\text{pkt}}}$ and $w_{\text{pkt}}[m, :, d_2] \in \mathbb{R}^{d_{\text{pkt}}}$.
   - Exact SVD of the pretrained `300M` Teacher's Layer-0 projections (`wq, wk, wv, wout`) reshaped to $(m=48, d_2=16, n=48, d_1=16)$ shows that the rank-1 $(m, d_2)$ outer-product subspace captures only **`15.05%` (`wq`), `15.59%` (`wk`), `14.43%` (`wv`), and `14.42%` (`wout`)** of the Frobenius energy required to interface with the frozen classical trunk (`blocks[0].mlp` and `blocks[1..11]`).
2. **Bottleneck 2 — Isotropic Singular-Value Constraint ($\sigma_i \equiv 1$) vs. Anisotropic Operator Spectrum ($\sigma_i \in [0.01, 25.89]$)**:
   - Because `w_pkt` and `m_mode` are pure orthogonal matrices multiplied by a single scalar `obs_scale`, all $768$ singular values are forced equal ($\sigma_1 = \dots = \sigma_{768}$).
   - In the pretrained `300M` Teacher, the singular values of `wq, wk, wv, wout` span **`[0.01 .. 25.89]`** (`median = 3.58`, `p10 = 9.45`), and the output channel RMS of `d_attn_c = blk0.out_proj(attn_heads)` spans **`[15.75 .. 436.42]`** (`27.7x` channel anisotropy). Forcing isotropic channel RMS on the exact teacher's own `d_attn_c` degrades validation CE by **`+0.1653` nats** (`3.6827 -> 3.8480` on `batch 0`).
3. **Bottleneck 3 — Layer-0 Residual Bypass Scale Clamping (`block_lambdas` clamped to `<= 2.0` vs. `24.23` in `300M` Teacher)**:
   - In the `300M` Teacher, `blk0.block_lambdas = [12.6139, 11.6139]` (`sum = 24.23`), producing `rms(x_in_c) = 24.25`, `rms(d_attn_c) = 47.89`, and `rms(x_mid_c) = 55.46` before adding `d_mlp_c` (`rms = 103.54`).
   - In Plans 065–067, [`restrict_quantum_scalars_`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py#L926-L945) clamped `block_lambdas` to `[0.50, 1.50]` / `[-0.50, 0.50]` and `obs_scale` to `[0.02, 0.60]`, restricting `rms(x_mid_q) <= 2.0` (`27x` smaller than `55.46`) and starving the direct `x_mid` residual bypass around `blk0.mlp` (`+0.1525` nats penalty even for the exact teacher: `3.6827 -> 3.8352`).
4. **Bottleneck 4 — Missing Fubini-Study Projective Self-State Orthogonalizer $(I - \tanh(\alpha_h) |\hat{\psi}_v\rangle\langle\hat{\psi}_v|)$**:
   - After causal quantum outer-product density evolution, the output wavepacket $|\psi_{\text{out}}\rangle$ contains a trivial self-echo component parallel to the local value ket $|\hat{\psi}_v\rangle$. Removing this self-echo via the rank-1 Fubini-Study orthogonal complement projector $(I - \tanh(\alpha_h)|\hat{\psi}_v\rangle\langle\hat{\psi}_v|)$ improves Step-0 validation CE by **`-0.0283` nats** on the `50M` Teacher (`4.3641 -> 4.3358`) and **`-0.0141` nats** on the `300M` Teacher (`3.6878 -> 3.6737`).

---

## 2. Three New Architectural & Optimization Axes (`Axes K, L, M`)

### 2.1. Axis K — Biorthogonal SVD-Stiefel Operator Geometry & Rank Compression (`QuantumStiefelProjMode`)

To preserve **strict Stiefel manifold geometry** (`||U^T U - I_r||_F = 0`, `||V^T V - I_r||_F = 0` stepped by `ModularRiemannianStiefelKrausMuon`) and **genuine interior Quantum Wavepacket Outer-Product Density-Matrix Attention (`CAUSAL_ENSEMBLE_DM_IO` + McWeeny Purification $\rho_{\text{pure}} = 3\rho_0^2 - 2\rho_0^3$ + Unitary $U(d_h)$ Evolution)** while eliminating Both Bottleneck 1 (rank-1 Kronecker subblocks) and Bottleneck 2 (isotropic singular values), we introduce `QuantumStiefelProjMode`:

- **`KRONEKER_FACTORED_STIEFEL`** (`Plan 067` Baseline, `0.18M - 0.33M` L0 params):
  Uses `apply_factored_wavepacket_stiefel(x, w_pkt, m_mode)`.
- **`BIORTHOGONAL_SVD_STIEFEL_R128`** ($r = 128$, **`0.79M` L0 params — `9.0x` smaller than classical Layer 0 `7.08M`**)
- **`BIORTHOGONAL_SVD_STIEFEL_R256`** ($r = 256$, **`1.58M` L0 params — `4.5x` smaller than classical Layer 0 `7.08M`**)
- **`BIORTHOGONAL_SVD_STIEFEL_R512`** ($r = 512$, **`3.15M` L0 params — `2.25x` smaller than classical Layer 0 `7.08M`**)
- **`BIORTHOGONAL_SVD_STIEFEL_R768`** ($r = \min(D_{\text{emb}}, 768)$, **`3.67M - 4.72M` L0 params — `1.5x - 1.9x` smaller than classical Layer 0 `7.08M`**)

#### Exact Mathematical Specification of `BiorthogonalSVDStiefelProjection`:
For each of the four Layer-0 quantum wavepacket boundary operators $\alpha \in \{q, k, v, \text{obs}\}$ mapping $\mathbb{R}^{D_{\text{in}}} \to \mathbb{R}^{D_{\text{out}}}$ (where $(D_{\text{in}}, D_{\text{out}}) = (768, D_{\text{emb}})$ for $\alpha \in \{q, k, v\}$ and $(D_{\text{emb}}, 768)$ for $\alpha = \text{obs}$), let $r_{\text{eff}} = \min(r, D_{\text{in}}, D_{\text{out}})$:
$$W_\alpha(x) = \left((x V_\alpha) \odot \vec{\sigma}_\alpha\right) U_\alpha^\top$$
where:
1. $U_\alpha \in \mathrm{St}(D_{\text{out}}, r_{\text{eff}})$ (`shape (D_out, r_eff)`, $U_\alpha^\top U_\alpha = I_{r_{\text{eff}}}$) and $V_\alpha \in \mathrm{St}(D_{\text{in}}, r_{\text{eff}})$ (`shape (D_in, r_eff)`, $V_\alpha^\top V_\alpha = I_{r_{\text{eff}}}$) are **tall column-orthonormal Stiefel manifold parameters** stepped by `ModularRiemannianStiefelKrausMuon`.
2. $\vec{\sigma}_\alpha \in \mathbb{R}^{r_{\text{eff}}}$ (`shape (r_eff,)`) is the learnable **singular-value / Hamiltonian coupling spectrum** stepped by `q_scalar_opt` (`AdamW`, `weight_decay = 0.0`, clamped to $[0.0, 50.0]$).
3. **Exact Teacher-SVD Procrustes Initialization at Step 0**:
   - Let $W_{\alpha, t} \in \mathbb{R}^{768 \times 768}$ be the corresponding slice of `classical.blocks[0].qkv_proj.weight` ($\alpha \in \{q, k, v\}$) or `classical.blocks[0].out_proj.weight` ($\alpha = \text{obs}$).
   - For $D_{\text{emb}} = 768$, compute thin SVD $W_{\alpha, t} = \tilde{U} \operatorname{diag}(\tilde{\sigma}) \tilde{V}^\top$ in `float32` and set:
     $$U_\alpha \leftarrow \tilde{U}_{:, :r_{\text{eff}}}, \qquad \vec{\sigma}_\alpha \leftarrow \tilde{\sigma}_{:r_{\text{eff}}}, \qquad V_\alpha \leftarrow \tilde{V}_{:, :r_{\text{eff}}}$$
   - For $D_{\text{emb}} \ne 768$ (e.g., $D_{\text{emb}} = 512$ in `SHAPE_1x2x256` or $D_{\text{emb}} = 1024$ in `SHAPE_1x2x512` / `SHAPE_1x2x512_PKT32`), let $B \in \mathrm{St}(\max(D_{\text{emb}}, 768), \min(D_{\text{emb}}, 768))$ be the `w_in_bridge` Stiefel frame. Project $W_{\alpha, t}$ into the $D_{\text{emb}}$ wavepacket frame before taking the top-$r_{\text{eff}}$ SVD:
     - For $\alpha \in \{q, k, v\}$: $\widehat{W}_\alpha = B W_{\alpha, t} \in \mathbb{R}^{D_{\text{emb}} \times 768}$ (if $D_{\text{emb}} > 768$) or $B^\top W_{\alpha, t} \in \mathbb{R}^{D_{\text{emb}} \times 768}$ (if $D_{\text{emb}} < 768$).
     - For $\alpha = \text{obs}$: $\widehat{W}_{\text{obs}} = W_{\text{out}, t} B^\top \in \mathbb{R}^{768 \times D_{\text{emb}}}$ (if $D_{\text{emb}} > 768$) or $W_{\text{out}, t} B \in \mathbb{R}^{768 \times D_{\text{emb}}}$ (if $D_{\text{emb}} < 768$).
     - Compute thin SVD $\widehat{W}_\alpha = \tilde{U} \operatorname{diag}(\tilde{\sigma}) \tilde{V}^\top$ and assign the top $r_{\text{eff}}$ singular triplets $(U_\alpha, \vec{\sigma}_\alpha, V_\alpha)$.
4. **Interior Quantum Outer-Product Density-Matrix Evolution**:
   - Inside `ConfigurableWavepacketStatMechBlock`, $q_{\text{wp}}, k_{\text{wp}}, v_{\text{wp}} \in \mathbb{R}^{B \times T \times M_{\text{pkt}} \times d_{\text{pkt}}}$ still form the **rank-3 causal ensemble density matrices** $\rho_j = \sum_{\delta=0}^2 w_\delta |\psi_{j-\delta}\rangle\langle\psi_{j-\delta}| \in \mathbb{C}^{d_{\text{pkt}} \times d_{\text{pkt}}}$, aggregate them via causal attention, purify $\rho_0 \mapsto \rho_{\text{pure}}$ via McWeeny purification, evolve via complex head unitary $U_h \rho_{\text{pure}} U_h^\dagger$ (`initialized to I_{d_h}` when using SVD-Stiefel warm-start so Step-0 phase is unperturbed), and modulate $|\psi_v\rangle \mapsto (I + \frac{1}{2} U_h \rho_{\text{pure}} U_h^\dagger) U_h |\psi_v\rangle$.
   - Because `_run_attention_and_evolution` returns unit-RMS wavepackets $y_{\text{wp}}$ with gain $1.5$ before `F.rms_norm`, we restore the per-head input value RMS norm $\|\psi_{v, h}\|_{\text{RMS}} = \sqrt{\frac{1}{2 R d_{\text{pkt}}} \|\psi_{v, h}\|_2^2 + \epsilon}$ before applying $W_{\text{obs}}$:
     $$y_{\text{scaled}, h} = \gamma_{\text{obs}} \cdot \|\psi_{v, h}\|_{\text{RMS}} \cdot y_{\text{wp}, h}$$
     where $\gamma_{\text{obs}}$ (`obs_scale`) is initialized to $\frac{2}{3} \approx 0.6667$ (and clamped to $[0.02, 5.00]$).

---

### 2.2. Axis L — Fubini-Study Projective Self-State Orthogonalization & Teacher-Calibrated Residual Highway (`QuantumSelfOrthoMode`)

We introduce `QuantumSelfOrthoMode`:
- **`NONE`**: Standard quantum wavepacket readout without self-state projection.
- **`FUBINI_STUDY_SELF_ORTHO`**:
  1. **Rank-1 Fubini-Study Orthogonal Complement Projector**:
     For each quantum head $h \in \{0, \dots, H_q - 1\}$, let $|\hat{\psi}_{v, h}\rangle = \frac{\psi_{v, h}}{\|\psi_{v, h}\|_2 + \epsilon} \in \mathbb{R}^{2 R d_{\text{pkt}}}$ be the unit-normalized mixed value wavepacket ket (`v_mixed`). Before applying the observable readout $W_{\text{obs}}$, project out the trivial self-echo component parallel to $|\hat{\psi}_{v, h}\rangle$:
     $$|\psi_{\text{ortho}, h}\rangle = |y_{\text{scaled}, h}\rangle - \tanh(\alpha_h) \langle \hat{\psi}_{v, h} \mid y_{\text{scaled}, h} \rangle |\hat{\psi}_{v, h}\rangle$$
     where $\alpha_h \in \mathbb{R}^{H_q}$ (`q_xsa_alpha`, `6` or `8` scalar parameters) is initialized from `classical.blocks[0].xsa_alpha` when $H_q = 6$ (or interpolated to $H_q \in \{4, 8\}$).
  2. **Teacher-Calibrated Layer-0 Highway & Gated Couplings**:
     - `quantum_block0.block_lambdas` is initialized directly from `classical.blocks[0].block_lambdas` (`[12.6139, 11.6139]` in `300M`, `[2.4047, 1.4047]` in `50M`) and its clamp interval in `restrict_quantum_scalars_` is widened to `[-30.0, 30.0]`.
     - In `encode_quantum_layer0`, the residual highway entering Layer 0 is computed cleanly in $768$D without per-packet norm flattening:
       $$x_{0, 768} = \operatorname{RMSNorm}_{768}\!\left(\operatorname{Smear}_{\text{causal}}\!\left(\operatorname{Bridge}_{\text{in}}(q_{\text{wte}}[\text{idx}])\right)\right) \cdot s_{\text{in}}, \qquad x_{\text{in}, 768} = (\lambda_{0,0} + \lambda_{0,1}) x_{0, 768}$$
       and the pre-MLP residual update adds $d_{\text{attn}, 768} = W_{\text{obs}}(g_{\text{attn}}(h) \odot \psi_{\text{ortho}})$ directly onto $x_{\text{in}, 768}$:
       $$x_{\text{mid}, 768} = x_{\text{in}, 768} + d_{\text{attn}, 768}$$
     - When `ve_coupling_mode == FROZEN_TEACHER_VE0_COUPLED`, `ve_0` is modulated by `classical.blocks[0].ve_gate` and `sa_lambdas` before mapping into the quantum value wavepacket space.
     - Inside `ConfigurableWavepacketStatMechBlock`, when $H_q = 6$ and $2 R d_{\text{pkt}} = 128$, query/key wavepackets use the teacher's `NativeLeaderboardRotary` (`base = 1024.0`, half-truncated rotary + `yarn_q_scale` + partial key offset) so rotary frequencies match Layers `1..11` exactly; for general $(H_q, R, d_{\text{pkt}})$ shapes (`SHAPE_1x2x256` with $H_q=4$, `SHAPE_1x2x512` / `SHAPE_1x2x512_PKT32` with $H_q=8$), a dedicated `NativeLeaderboardRotary(head_dim=2*R*d_pkt, base=1024.0)` buffer with identical `yarn_q_scale` is used.

---

### 2.3. Axis M — Wavefunction Shape & Outer-Product Hilbert Granularity under Biorthogonal SVD-Stiefel

We test how the four primary wavefunction geometries behave once freed from the rank-1 Kronecker bottleneck:
1. **`SHAPE_24x2x16` (`(V, 24, 2, 16)`, `768`D, $d_{\text{pkt}} = 16$, $16 \times 16$ $\rho$)**: Iso-dimension multi-packet wavepackets.
2. **`SHAPE_12x2x32` (`(V, 12, 2, 32)`, `768`D, $d_{\text{pkt}} = 32$, $32 \times 32$ $\rho$)**: Larger $32 \times 32$ outer-product density operators at fixed `768`D.
3. **`SHAPE_1x2x512_PKT32` (`(V, 2, 512)`, `1024`D, $d_{\text{pkt}} = 32$, $32 \times 32$ $\rho$)**: Fewer, larger `512`-complex-dim token wavefunctions.
4. **`SHAPE_1x2x256` (`(V, 2, 256)`, `512`D, $d_{\text{pkt}} = 16$, $16 \times 16$ $\rho$)**: Compact `256`-complex-dim token wavefunctions (`33%` smaller embedding table `25.76M` vs. `38.63M` AND max SVD-Stiefel rank $r_{\text{eff}} \le 512$, `3.15M` L0 params — `2.25x` smaller than classical Layer 0).

---

## 3. Experimental Design & Controlled Tournament Matrix (`Wave 52D`)

All arms use the deterministic FineWeb-Edu streaming loader (`batch_size = 8`, `seq_len = 1024`, `8,192` tokens/step) and evaluate on the exact `65,536`-token FineWeb validation split (`fineweb_val_000000.bin`) every `100` steps.

### 3.1. Phase-1 Controlled Ablation Tournament (`500` Steps = `4.10M` Tokens = `1.37%` of `300M` vs. `300M` Classical Teacher `val_ce = 3.5845`)

| Arm ID | `wf_shape` (`D_emb`, $d_{\text{pkt}}$) | `stiefel_proj_mode` (Axis K) | `self_ortho_mode` (Axis L) | Active L0 Params (vs. Classical `7.08M`) | Primary Hypothesis Tested |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`D0_KRONEKER_P067_REF`** | `SHAPE_1x2x512_PKT32` (`1024`D, $d_{\text{pkt}}=32$) | `KRONEKER_FACTORED_STIEFEL` | `NONE` | `0.28M` (`24.8x` smaller) | Plan 067 champion baseline (`G4`: `3.6458` @ `4.10M` tok). |
| **`D1_SVD_R128_24X2X16`** | `SHAPE_24x2x16` (`768`D, $d_{\text{pkt}}=16$) | `BIORTHOGONAL_SVD_STIEFEL_R128` | `FUBINI_STUDY_SELF_ORTHO` | **`0.79M` (`9.0x` smaller)** | Does rank-128 SVD-Stiefel (`0.79M` L0 params) break the `3.6220` Kronecker floor? |
| **`D2_SVD_R256_24X2X16`** | `SHAPE_24x2x16` (`768`D, $d_{\text{pkt}}=16$) | `BIORTHOGONAL_SVD_STIEFEL_R256` | `FUBINI_STUDY_SELF_ORTHO` | **`1.58M` (`4.5x` smaller)** | Rank-256 SVD-Stiefel pareto efficiency (`4.5x` smaller L0). |
| **`D3_SVD_R512_24X2X16`** | `SHAPE_24x2x16` (`768`D, $d_{\text{pkt}}=16$) | `BIORTHOGONAL_SVD_STIEFEL_R512` | `FUBINI_STUDY_SELF_ORTHO` | **`3.15M` (`2.25x` smaller)** | Rank-512 SVD-Stiefel (`2.25x` smaller L0). |
| **`D4_SVD_R768_24X2X16`** | `SHAPE_24x2x16` (`768`D, $d_{\text{pkt}}=16$) | `BIORTHOGONAL_SVD_STIEFEL_R768` | `FUBINI_STUDY_SELF_ORTHO` | `4.72M` (`1.50x` smaller) | Full-rank $r=768$ SVD-Stiefel with $16 \times 16$ outer-product $\rho$. |
| **`D5_SVD_R768_12X2X32`** | `SHAPE_12x2x32` (`768`D, $d_{\text{pkt}}=32$) | `BIORTHOGONAL_SVD_STIEFEL_R768` | `FUBINI_STUDY_SELF_ORTHO` | `4.73M` (`1.50x` smaller) | Does $32 \times 32$ outer-product $\rho$ ($d_{\text{pkt}}=32$) beat $16 \times 16$ ($d_{\text{pkt}}=16$) under SVD-Stiefel? |
| **`D6_SVD_R512_1X2X256`** | `SHAPE_1x2x256` (`512`D, $d_{\text{pkt}}=16$) | `BIORTHOGONAL_SVD_STIEFEL_R512` | `FUBINI_STUDY_SELF_ORTHO` | **`3.15M` (`2.25x` smaller, `-33%` `wte`)** | Can compact `(V, 2, 256)` wavefunctions match the `300M` teacher with `33%` fewer embedding params? |
| **`D7_SVD_R768_1X2X512_PKT32`** | `SHAPE_1x2x512_PKT32` (`1024`D, $d_{\text{pkt}}=32$) | `BIORTHOGONAL_SVD_STIEFEL_R768` | `FUBINI_STUDY_SELF_ORTHO` | `5.52M` (`1.28x` smaller) | Large `(V, 2, 512)` wavefunctions + $32 \times 32$ outer-product $\rho$ + SVD-Stiefel. |

### 3.2. Phase-2 Teacher Reproduction & Exceeding Runs (`800` Steps = `6.55M` Tokens)

1. **`E1_EXCEED_50M_TEACHER` (`800` Steps = `6.55M` Tokens = `13.1%` of `50M` vs. `50M` Classical Teacher `4.2467`, `31.54%` Top-1)**:
   - Runs `D5_SVD_R768_12X2X32` (or top Phase-1 configuration) grafted onto [`ckpt_step00191_0050M_m0050M.pt`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/ckpt_step00191_0050M_m0050M.pt) (`val_ce = 4.2467`, `top1_acc = 31.54%`) to verify **strictly exceeding (`< 4.2467`)** the `50M`-token Classical Teacher in `< 6.55M` tokens.
2. **`E2_REPRODUCE_300M_TEACHER` (`800` Steps = `6.55M` Tokens = `2.18%` of `300M` vs. `300M` Classical Teacher `3.5845`, `37.31%` Top-1)**:
   - Runs the winning Phase-1 configuration grafted onto [`ckpt_step01145_0300M_m0300M.pt`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/ckpt_step01145_0300M_m0300M.pt) (`val_ce = 3.5845`, `top1_acc = 37.31%`) to test whether we can **reproduce (`<= 3.5845`) or exceed** the `300M`-token Classical Teacher in `6.55M` tokens (`45.8x` fewer tokens).
3. **`E3_COMPACT_R256_300M_RECOVERY` (`800` Steps = `6.55M` Tokens vs. `300M` Classical Teacher `3.5845`)**:
   - Runs `D2_SVD_R256_24X2X16` (`1.58M` L0 params — `4.5x` smaller than classical Layer 0) for `800` steps (`6.55M` tokens) to map the sub-`2M`-parameter recovery frontier.

---

## 4. Module Architecture & Compliance with `docs/SCRIPT_BEST_PRACTICES.md`

To keep every source file well below the `1,000` SLOC soft limit (`Rule 1` of `docs/SCRIPT_BEST_PRACTICES.md`):
1. **New Companion Module**: [`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py) (`~320` SLOC):
   - Defines `QuantumStiefelProjMode` (`KRONEKER_FACTORED_STIEFEL`, `BIORTHOGONAL_SVD_STIEFEL_R128`, `BIORTHOGONAL_SVD_STIEFEL_R256`, `BIORTHOGONAL_SVD_STIEFEL_R512`, `BIORTHOGONAL_SVD_STIEFEL_R768`) and `QuantumSelfOrthoMode` (`NONE`, `FUBINI_STUDY_SELF_ORTHO`).
   - Implements `BiorthogonalSVDStiefelProjection(nn.Module)` (`U in St(D_out, r)`, `sigma in R^r`, `V in St(D_in, r)`) with `initialize_from_teacher_matrix_` and `real_stiefel_parameters()` / `scalar_parameters()`.
   - Implements `apply_fubini_study_self_ortho(y_scaled, v_mixed, xsa_alpha)` and `SVDStiefelWavepacketStatMechBlock(nn.Module)` wrapping `ConfigurableWavepacketStatMechBlock._run_attention_and_evolution` with Biorthogonal SVD-Stiefel projections, `NativeLeaderboardRotary(base=1024.0)` alignment, and Fubini-Study projective self-state orthogonalization.
2. **Integration in [`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py)** (`<= 995` SLOC):
   - Adds `stiefel_proj_mode: QuantumStiefelProjMode = QuantumStiefelProjMode.KRONEKER_FACTORED_STIEFEL` and `self_ortho_mode: QuantumSelfOrthoMode = QuantumSelfOrthoMode.NONE` to `QuantumFrontEndGraftConfig` (defaulting to existing behavior so all Plans 065–067 tests remain 100% backward-compatible).
   - Routes `FrozenTrunkQuantumFrontEndGPT` to `SVDStiefelWavepacketStatMechBlock` when `stiefel_proj_mode != QuantumStiefelProjMode.KRONEKER_FACTORED_STIEFEL`.

---

## 5. Verification Gates & Falsifiable Hypotheses (`tests/test_plan068_svd_stiefel_and_fubini_study_graft.py`)

- **Gate `H1` (`test_h1_biorthogonal_svd_stiefel_orthogonality_and_rank_bounds`)**:
  Verifies that for all `QuantumStiefelProjMode` ranks $r \in \{128, 256, 512, 768\}$ and wavefunction shapes (`SHAPE_24x2x16`, `SHAPE_12x2x32`, `SHAPE_1x2x256`, `SHAPE_1x2x512_PKT32`), every $U_\alpha, V_\alpha$ satisfies $\|U_\alpha^\top U_\alpha - I_{r_{\text{eff}}}\|_F < 10^{-5}$ and $\|V_\alpha^\top V_\alpha - I_{r_{\text{eff}}}\|_F < 10^{-5}$ both at initialization and after 3 steps of `FrozenTrunkQuantumOptimizerBundle`.
- **Gate `H2` (`test_h2_fubini_study_projective_self_state_orthogonalizer`)**:
  Verifies that `apply_fubini_study_self_ortho(y_scaled, v_mixed, xsa_alpha)`:
  1. Reduces to identity when $\alpha_h = 0$,
  2. Produces an output strictly orthogonal to $\hat{\psi}_{v, h}$ ($\langle \hat{\psi}_{v, h} \mid \psi_{\text{ortho}, h} \rangle = 0$ to `< 1e-5`) in the limit $\tanh(\alpha_h) \to 1$, and
  3. Propagates non-zero finite gradients to `xsa_alpha`, `y_scaled`, and `v_mixed`.
- **Gate `H3` (`test_h3_step0_teacher_calibration_improvement`)**:
  Verifies on a synthetic teacher that `BIORTHOGONAL_SVD_STIEFEL_R768` + `FUBINI_STUDY_SELF_ORTHO` achieves a strictly lower (`> 5x` smaller) Step-0 (`0`-token) Layer-0 output discrepancy $\|x_{1, q} - x_{1, c}\|_F / \|x_{1, c}\|_F$ than `KRONEKER_FACTORED_STIEFEL`.
- **Gate `H4` (`test_h4_end_to_end_loss_descent_and_strict_trunk_freeze`)**:
  Verifies that 6 optimization steps strictly decrease total training loss while all frozen classical parameters have `grad is None` and zero parameter drift (`max_abs_diff == 0.0`) and active quantum density purity remains genuinely mixed ($\operatorname{Tr}(\rho^2) < 0.95$).
