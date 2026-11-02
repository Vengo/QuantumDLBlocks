# EXPERIMENT PLAN 073: Full-Backbone (`Embeddings through Layer 11`) Quantum Wavepacket Replacement — Progressive Depth (`L0..L2`, `L0..L5`, `L0..L11`), Full-Block Mach-Zehnder Interferometers, and Whole-Network SVD-Stiefel Compression (`Wave 53A`)

**Date:** 2026-11-01  
**Wave:** `Wave 53A` (`EXPERIMENT_PLAN_073.md`)  
**Status:** Proposed for Independent Implementation-Readiness Review  
**Predecessor Waves:** `Waves 52A–52H` (`Plans 065–072`, `75` completed FineWeb GPU runs; all-time record `H72_1@1600`: `val_ce = 3.5763` (`-0.0082` nats below the `300M` Classical Teacher `3.5845`), `top1 = 37.47%` (`+0.16%`), `top5 = 58.36%` (`+0.10%`) at exact `1:1` Layer-0 parameter parity `r = 384`, plus `H72_7@1600` two-layer `Emb + L0 + L1` graft reaching `val_ce = 3.5837`, `top1 = 37.41%`, `top5 = 58.32%`).

---

## 1. Executive Summary & Scientific Objective

In `Plans 065–072`, we proved that replacing the initial token embeddings (`wte`) and Layer 0 (`blocks[0].attn`)—or Layers `0..1` (`blocks[0..1].attn`)—of a frozen pretrained 12-layer classical Transformer (`CompiledLeaderboardClassicalGPT`, [`ckpt_step01145_0300M_m0300M.pt`](file:///home/nixos/QuantumDLBlocks/.cache/checkpoints/w45b_classical_teacher_ckpt50m_l12_d768_b262k/ckpt_step01145_0300M_m0300M.pt), `val_ce = 3.5845`, `top1_acc = 37.31%`, `top5_acc = 58.26%`) with our **Biorthogonal SVD-Stiefel Quantum Wavepacket Outer-Product Architecture** (`(V, 12, 2, 32)` + `WF_IN_WF_OUT` + `ATTENDED_RHO_ON_LOCAL_V` / `LIOUVILLE_COMMUTATOR_RHO_V` + `BERRY_PHASE_COMPLEX_QK` + `FUBINI_STUDY_SELF_ORTHO`) strictly exceeds the `300M`-token Classical Teacher in `4.10M–13.11M` tokens (`1.37%–4.37%` of `300M`), even at `1:1` parameter parity ($r=384$), `2x` compression ($r=192$), `3x` compression ($r=128$), and `6x` compression ($r=64$).

The user now asks:
> *"This looks great! Let's see if this architecture can replace more of the network. Let's try the experiment again, but replacing everything from embeddings through the last layer and see whether these quantum inspired architectures work for more of the network."*

### 1.1. The Three Core Questions of `EXPERIMENT_PLAN_073.md` (`Wave 53A`)

1. **Can we replace progressive slices (`L0..L2` = `25%`, `L0..L5` = `50%` encoder half) and ultimately ALL 12 layers (`Embeddings + Layers 0..11`, through the last layer) with Quantum Wavepacket Outer-Product Attention (`EMB_PLUS_ALL_12_ATTN`) and reproduce or exceed the `300M`-token Classical Teacher (`3.5845` CE, `37.31%` Top-1, `58.26%` Top-5) well within `300M` tokens?**
2. **Can we go even further (`EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`) and replace 100% of the network from embeddings through Layer 11—including `wte`, `value_embeds[0..2]`, ALL 12 layers' attention sublayers (`blocks[0..11].attn`), AND ALL 12 layers' MLP sublayers (`blocks[0..11].mlp`) using Biorthogonal SVD-Stiefel Two-Beam Mach-Zehnder Quantum Interferometers—so that zero classical layers remain anywhere in the 12-layer backbone (leaving only the final vocabulary projection `lm_head`)?**
3. **Does our Biorthogonal SVD-Stiefel parameter compression ($r=384$ exact `1:1` attention parity and `0.625x` Mach-Zehnder MLP compression; $r=192$ `2x` whole-network attention compression; $r=128$ `3x` whole-network attention compression) hold when applied simultaneously across ALL 12 layers from embeddings through Layer 11?**

---

## 2. Root-Cause Analysis of Multi-Layer Compounding & Exact Mathematical Solutions

When scaling quantum wavepacket outer-product replacement from 1–2 layers (`L0..L1`) to **all 12 layers (`L0..L11`)**, three multi-layer compounding mechanisms must be handled explicitly:

### 2.1. Root Cause 1: Short-Window Layer `use_partial_key_offset` Mismatch (`Layers 1, 2, 3, 5, 6, 8, 9, 10`)
- In [`CompiledLeaderboardBlock.__init__`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/compiled_leaderboard_classical_gpt.py#L453-L456):
  ```python
  self.is_long_window_layer = bool(self.layer_idx in cfg.long_window_layers or self.layer_idx == cfg.num_layers - 1)
  self.use_partial_key_offset = bool(cfg.use_partial_key_offset and self.is_long_window_layer)
  ```
  Because `cfg.long_window_layers = (0, 4, 11)`, only layers $\ell \in \{0, 4, 11\}$ shift the upper half of key channels by 1 token (`k[:, :-1, :, 64:]`). On the 8 interior short-window layers $\ell \in \{1, 2, 3, 5, 6, 8, 9, 10\}$, `use_partial_key_offset` is `False`.
- Previously, [`SVDStiefelWavepacketStatMechBlock.forward`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py#L524-L527) shifted keys unconditionally (`if seq_len > 1 and self.head_dim == 128`). Across 8 short-window layers, an unintended 1-token key shift per layer corrupts key-query phase alignment.
- **Exact Fix**: `SVDStiefelWavepacketStatMechBlock` stores `self.use_partial_key_offset: bool = bool(getattr(teacher_blk, "use_partial_key_offset", False))`, `self.sdpa_scale: float = float(getattr(teacher_blk, "attn_scale", 0.12))`, and `self.q_ve_gate: nn.Linear | None` (with `self.ve_part_ch = int(getattr(teacher_blk, "ve_part_ch", 6))` copied from `teacher_blk.ve_gate` when `teacher_blk.ve_gate is not None`, and included in `scalar_parameters()`), applying the half-key shift if and only if `self.use_partial_key_offset and seq_len > 1`.

### 2.2. Root Cause 2: Attention Temperature (`0.22` vs. `0.12`) & Entropy-Preserving Coherent Ket Branch (`Axis T`)
- In `CompiledLeaderboardBlock`, attention logits are scaled by `attn_scale = 0.12`. In `ConfigurableWavepacketStatMechBlock`, `DEFAULT_WAVEPACKET_SDPA_SCALE = 0.22` was used, and the post-SDPA ket $|\bar{v}_{t,h,r}\rangle = \sum_{s \le t} A_{t,s}^{(h)} |\hat{v}_{s,h,r}\rangle$ was unit-normalized (`inv_vn = rsqrt(v_norm_sq * v_rank)`) and RMSNormed (`F.rms_norm(y_wp, (d_h,))`) before multiplying by the local token's `v_head_rms`.
- While post-SDPA RMSNorm works well for 1–2 layers (`+0.058` nats at Step 0), across 11 stacked attention layers it erases two physical quantities:
  1. **Attention-entropy amplitude modulation**: When $A_{t,s}^{(h)}$ is diffuse across many context tokens $s \le t$, destructive interference among incoherent kets reduces $\|\sum_{s \le t} A_{t,s}^{(h)} v_{s,h}\|_2$; when $A_{t,s}^{(h)}$ is peaked on a coherent token, $\|\sum_{s \le t} A_{t,s}^{(h)} v_{s,h}\|_2$ remains large.
  2. **Inter-mode relative energy**: Forcing each mode $r \in \{1, \dots, R\}$ to equal norm $1/\sqrt{R}$ in the ket branch alters the relative singular-value weighting learned in $W_v$.
- **Exact Fix (`QuantumKetAmplitudeMode.ENTROPY_PRESERVING_COHERENT_KET`)**:
  In each quantum wavepacket attention block $\ell \in \{0, \dots, 11\}$:
  1. **Trace-1 Outer-Product Density Branch & Explicit Traceless Subtraction**: Each token $s$ forms its trace-1 Hermitian density matrix $\rho_{s,h}^{\text{tok}} = \sum_{r=1}^R |\hat{v}_{s,h,r}\rangle\langle\hat{v}_{s,h,r}| \in \mathbb{C}^{d_h \times d_h}$ from trace-normalized kets $|\hat{v}_{s,h,r}\rangle = \frac{|v_{\text{src}, s, h, r}\rangle}{\sqrt{R (\|v_{\text{src}, s, h, r}\|_2^2 + \epsilon)}}$ (where $v_{\text{src}} = h_{\text{pkt}}$ for `ATTENDED_RHO_ON_LOCAL_V` or $v_{\text{mixed}}$ otherwise). Causal SDPA with `scale = self.sdpa_scale` (`0.12`, calibrated from `teacher_blk.attn_scale`) aggregates $\rho_{t,h}^{(0)} = \sum_{s \le t} A_{t,s}^{(h)} \rho_{s,h}^{\text{tok}} \in \mathbb{C}^{d_h \times d_h}$, which is purified via McWeeny/Fermi-Dirac cooling $\rho_{t,h}^{(0)} \mapsto \rho_{\text{pure}, t,h}$ and evolved via $U_h \in U(d_h)$ (or CPTP Kraus) into `(evol_re, evol_im)` with $\operatorname{Tr}(\text{evol\_re}) = 1$. We then explicitly subtract the maximally mixed identity component to form the traceless Hermitian operator:
     $$\mathring{\rho}_{\text{evol}, R} = \text{evol\_re} - \frac{1}{d_h}I_{d_h}, \qquad \mathring{\rho}_{\text{evol}, I} = \text{evol\_im}$$
     so that the identity trace does not shift `obs_scale = 1.0` at Step 0.
  2. **Entropy-Preserving Coherent $V$-Ket Branch**: The $(k_\rho + 1)$-th SDPA value chunk carries the unnormalized $V$-space wavepacket $v_{\text{mixed}, s, h, r} \in \mathbb{C}^{d_h}$, so causal SDPA aggregates the true coherent superposition:
     $$|\bar{v}_{t, h, r}\rangle = \sum_{s \le t} A_{t, s}^{(h)} |v_{\text{mixed}, s, h, r}\rangle \in \mathbb{C}^{d_h}$$
  3. **Direct Outer-Product Operator Action on $V$-Space Wavefunctions**:
     Let $|\tilde{v}_{t,h,r}\rangle = |\bar{v}_{t,h,r}\rangle U_h^\dagger$ be the unitarily evolved attended wavefunction, and for `ATTENDED_RHO_ON_LOCAL_V`, let $|\tilde{v}_{\text{loc}, t, h, r}\rangle = \frac{\|\tilde{v}_{t,h}\|_F}{\|v_{\text{mixed}, t, h}\|_F + \epsilon} |v_{\text{mixed}, t, h, r}\rangle U_h^\dagger$ be the local $V$-wavefunction RMS-matched to the attended ket envelope $\|\tilde{v}_{t,h}\|_F$. Then the traceless evolved outer-product density operator $\mathring{\rho}_{\text{evol}, t, h} \in \mathbb{C}^{d_h \times d_h}$ acts directly on the $V$-space wavefunctions with a learnable per-layer coupling $\beta_{\rho}^{(\ell)} = \text{rho\_coupling\_scale}^{(\ell)} \in [0.0, 1.0]$:
     $$|\psi_{\text{out}, t, h, r}\rangle = \text{obs\_scale}^{(\ell)} \Big( |\tilde{v}_{t, h, r}\rangle + \beta_{\rho}^{(\ell)} |\tilde{v}_{\text{target}, t, h, r}\rangle \mathring{\rho}_{\text{evol}, t, h} \Big)$$
     Followed by Fubini-Study projective self-state orthogonalization (`apply_fubini_study_self_ortho`), `q_attn_gate`, and `obs_proj`!
  - Notice that when $\beta_{\rho}^{(\ell)}$ is initialized to $\beta_{\rho,\text{init}} = 0.25 / \sqrt{N_{\text{q\_layers}}}$ (e.g., $\beta_{\rho,\text{init}} = 0.08$ for 11–12 layers, clamped to `[0.01, 1.00]`) and $\text{obs\_scale}^{(\ell)} = 1.0$, at Step 0 every single layer $\ell \in \{0..11\}$ preserves both quantum outer-product gradients ($\nabla_{U_h}\mathcal{L} \neq 0$, $\nabla_{\beta_\rho}\mathcal{L} \neq 0$) AND high Step-0 cosine alignment across all 12 layers simultaneously!

### 2.3. Root Cause 3: Replacing All 12 MLPs (`blocks[0..11].mlp`) with Calibrated Biorthogonal SVD-Stiefel Two-Beam Mach-Zehnder Interferometers (`Axis T`)
When we replace **not only all 12 attention sublayers, but also all 12 MLP sublayers (`blocks[0..11].mlp`)** (`EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`), how do we formulate the feed-forward channel as a genuine Riemannian quantum interferometer without losing the `300M`-token teacher's knowledge?
- We introduce [`SVDStiefelMachZehnderInterferometer`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/quantum_full_backbone_graft_gpt.py):
  - **Excitation Operator on $\mathrm{St}(D_{\text{hidden}}, r_{\text{mlp}}) \times \mathbb{R}_+^{r_{\text{mlp}}} \times \mathrm{St}(D_{\text{model}}, r_{\text{mlp}})$**:
    `up_proj = BiorthogonalSVDStiefelProjection(d_model=768, d_out=3072, rank=r_mlp)` maps $h_{\text{mlp}}^{(\ell)} = \operatorname{RMSNorm}(x_{\text{mid}}^{(\ell)}) \in \mathbb{R}^{B \times T \times 768}$ to the $3072$-D wavepacket excitation state ($1536$ complex pairs):
    $$S^{(\ell)} = \text{up\_proj}(h_{\text{mlp}}^{(\ell)}) \in \mathbb{R}^{B \times T \times 3072}$$
  - **Coherent Beam-Splitter Rotation & Two-Beam Mach-Zehnder Interference**:
    Viewing $S^{(\ell)}$ as $1536$ complex 2D modes $(S_{\text{re}}, S_{\text{im}}) \in \mathbb{R}^{B \times T \times 1536 \times 2}$, a learnable per-layer coherent beam-splitter angle $\theta_{\text{mz}}^{(\ell)} \in \mathbb{R}$ (initialized to $\theta_{\text{mz},\text{init}} = 0.04$) forms the orthogonal quadrature partner beam:
    $$D_{\text{re}}^{(\ell)} = -\tanh(\theta_{\text{mz}}^{(\ell)})\,S_{\text{im}}^{(\ell)}, \qquad D_{\text{im}}^{(\ell)} = \tanh(\theta_{\text{mz}}^{(\ell)})\,S_{\text{re}}^{(\ell)}$$
    The two interferometer arms are $A^{(\ell)} = S^{(\ell)} + D^{(\ell)}$ and $B^{(\ell)} = S^{(\ell)} - D^{(\ell)}$. With learnable constructive/destructive balance $\gamma_{\text{mz}}^{(\ell)} \in \mathbb{R}$ (initialized to $\gamma_{\text{mz},\text{init}} = 0.25$), the Mach-Zehnder detector output is:
    $$Z_{\text{MZ}}^{(\ell)} = \frac{1}{2}\big(\operatorname{ReLU}(A^{(\ell)})^2 + \operatorname{ReLU}(B^{(\ell)})^2\big) + \frac{\tanh(\gamma_{\text{mz}}^{(\ell)})}{4}\big(\operatorname{ReLU}(A^{(\ell)})^2 - \operatorname{ReLU}(B^{(\ell)})^2\big)$$
    - Because both $\theta_{\text{mz},\text{init}} = 0.04 \neq 0$ and $\gamma_{\text{mz},\text{init}} = 0.25 \neq 0$, both $\nabla_{\theta_{\text{mz}}}\mathcal{L} \neq 0$ and $\nabla_{\gamma_{\text{mz}}}\mathcal{L} \neq 0$ are strictly non-zero at Step 0 while introducing $< 0.1\%$ Step-0 perturbation relative to $\operatorname{ReLU}(S^{(\ell)})^2$!
  - **Readout Operator on $\mathrm{St}(D_{\text{model}}, r_{\text{mlp}}) \times \mathbb{R}_+^{r_{\text{mlp}}} \times \mathrm{St}(D_{\text{hidden}}, r_{\text{mlp}})$ & Scalar Clamping Bounds**:
    `down_proj = BiorthogonalSVDStiefelProjection(d_in=3072, d_out=768, rank=r_mlp)` maps $Z_{\text{MZ}}^{(\ell)}$ back to $\mathbb{R}^{B \times T \times 768}$:
    $$x_{\text{out}}^{(\ell)} = x_{\text{mid}}^{(\ell)} + \text{mz\_scale}^{(\ell)} \cdot \text{down\_proj}(Z_{\text{MZ}}^{(\ell)})$$
    In `SVDStiefelMachZehnderInterferometer.restrict_scalars_()`, parameters are clamped to bounded physical intervals: `up_proj.clamp_sigma_()`, `down_proj.clamp_sigma_()`, `mz_scale.clamp_(0.05, 5.0)`, `mz_theta.clamp_(-1.5, 1.5)`, and `mz_gamma.clamp_(-2.0, 2.0)`.
  - **Riemannian Parameter Count of `SVDStiefelMachZehnderInterferometer`**:
    - Classical `fc1 + fc2`: $2 \times 3072 \times 768 = \mathbf{4,718,592}$ (`4.72M`) parameters per layer (`56.62M` across 12 layers).
    - Quantum `SVDStiefelMachZehnderInterferometer` at rank $r_{\text{mlp}}$: $2 \times (3072 + 768 + 1) \times r_{\text{mlp}} + 3 = 7682\,r_{\text{mlp}} + 3$:
      - At $r_{\text{mlp}} = 768$: `5.90M` params/layer (exact rank-768 SVD calibration of `fc1` and `fc2` on Stiefel manifolds).
      - At $r_{\text{mlp}} = 384$: **`2.95M` params/layer (`0.625x` = `37.5%` smaller than classical MLP!)** (`35.40M` vs. `56.62M` across 12 layers!).
      - At $r_{\text{mlp}} = 192$: **`1.47M` params/layer (`0.3125x` = `3.2x` smaller than classical MLP!)** (`17.70M` vs. `56.62M` across 12 layers!).

---

## 3. Architecture of `QuantumFullBackboneGraftGPT` (`src/qdlblocks/models/quantum_full_backbone_graft_gpt.py`)

We create a clean, standalone module [`src/qdlblocks/models/quantum_full_backbone_graft_gpt.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/models/quantum_full_backbone_graft_gpt.py) (`<= 650` SLOC, all functions `<= 8` arguments) implementing:

### 3.1. Strategy Enums (Rule 3 of `docs/SCRIPT_BEST_PRACTICES.md`)
1. **`QuantumBackboneDepthScope(str, Enum)`**:
   - `EMB_PLUS_L0_TO_L2_ATTN = "emb_plus_l0_to_l2_attn"` (`wte` + Layers `0, 1, 2` quantum attention = `3` layers, `25%` of backbone)
   - `EMB_PLUS_L0_TO_L5_ATTN = "emb_plus_l0_to_l5_attn"` (`wte` + Layers `0..5` quantum attention = `6` layers, `50%` encoder half)
   - `EMB_PLUS_ALL_12_ATTN = "emb_plus_all_12_attn"` (`wte` + Layers `0..11` quantum attention = **all `12` layers, `100%` of attention through the last layer**)
   - `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ = "emb_plus_all_12_full_blocks_mz"` (`q_wte` + `q_value_embeds[0..2]` + Layers `0..11` quantum attention + Layers `0..11` `SVDStiefelMachZehnderInterferometer` + quantum `smear_gate` + quantum `skip_weights` + quantum `layer7_lambdas` = **100% of the network from embeddings through Layer 11!**)
2. **`QuantumKetAmplitudeMode(str, Enum)`**:
   - `POST_RMSNORM_KET = "post_rmsnorm_ket"`
   - `ENTROPY_PRESERVING_COHERENT_KET = "entropy_preserving_coherent_ket"`
3. **`QuantumMLPChannelMode(str, Enum)`**:
   - `FROZEN_CLASSICAL_MLP = "frozen_classical_mlp"`
   - `SVD_STIEFEL_MACH_ZEHNDER = "svd_stiefel_mach_zehnder"`

### 3.2. Complete Layer-by-Layer Forward Contract (`l = 0 .. 11`)
In `QuantumFullBackboneGraftGPT`:
1. **Token Wavepacket Sphere Embeddings (`q_wte`), Exact Polar-Factorized Quantum Value Sphere Embeddings (`q_value_embeds`), & Standalone Backbone Ownership**:
   - `q_wte` $\in \mathbb{R}^{V \times M_{\text{slices}} \times d_{\text{sphere}}}$ on $(\sqrt{d_{\text{sphere}}}\,\mathbb{S}^{d_{\text{sphere}}-1})^{M_{\text{slices}}}$, initialized via `initialize_teacher_procrustes_sphere_embeddings_`.
   - When `depth_scope == EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`:
     - The 3 token value embedding tables `value_embeds[0..2]` are replaced by 3 quantum wavepacket sphere tables `q_value_embeds[0..2]` $\in \mathbb{R}^{V \times M_{\text{slices}} \times d_{\text{sphere}}}$ on the unit-RMS sphere $(\sqrt{d_{\text{sphere}}}\,\mathbb{S}^{d_{\text{sphere}}-1})^{M_{\text{slices}}}$ (stepped via `BatchSubspaceTikhonovGramMuon` alongside `q_wte`), paired with a detached per-slice radial profile buffer `ve_slice_rms[b]` $\in \mathbb{R}^{V \times M_{\text{slices}} \times 1}$ (the exact pre-normalization slice RMS of `teacher.value_embeds[b].weight` in wavepacket coordinates) and a learnable per-bank scalar `ve_bank_scales` $\in \mathbb{R}^3$ (initialized to `1.0`, clamped to `[0.2, 5.0]`). Looking up token `idx` reconstructs `ve_b(idx) = ve_bank_scales[b] * (q_value_embeds[b][idx] * ve_slice_rms[b][idx])`, keeping `q_value_embeds[b]` strictly on the unit-RMS sphere (`q_wte_sphere_err < 1e-5`) while reproducing `teacher.value_embeds[b](idx)` to machine precision at Step 0!
     - `QuantumFullBackboneGraftGPT` also owns `self.q_smear_gate`, `self.q_smear_lambda`, `self.q_skip_weights`, `self.q_rope`, and Layer 7's `self.q_layer7_lambdas` (`nn.Parameter` initialized from `teacher.blocks[7].block_lambdas`, clamped to `[-30.0, 30.0]`) so `forward_features(idx)` executes all 12 layers (`0..11`) without ever accessing `self.classical.blocks`.
2. **Active Quantum Layers Set $\mathcal{L}_{\text{attn}}$ & $\mathcal{L}_{\text{mlp}}$**:
   - For `EMB_PLUS_L0_TO_L2_ATTN`: $\mathcal{L}_{\text{attn}} = \{0, 1, 2\}$, $\mathcal{L}_{\text{mlp}} = \emptyset$.
   - For `EMB_PLUS_L0_TO_L5_ATTN`: $\mathcal{L}_{\text{attn}} = \{0, 1, 2, 3, 4, 5\}$, $\mathcal{L}_{\text{mlp}} = \emptyset$.
   - For `EMB_PLUS_ALL_12_ATTN`: $\mathcal{L}_{\text{attn}} = \{\ell \in \{0..11\} : \neg\text{skip\_attn}(\ell)\} = \{0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11\}$, $\mathcal{L}_{\text{mlp}} = \emptyset$.
   - For `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`: $\mathcal{L}_{\text{attn}} = \{0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11\}$ and $\mathcal{L}_{\text{mlp}} = \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11\}$ (**all 12 blocks completely replaced**; note that at Layer 7 where `skip_attention=True`, `q_block_lambdas[7]` + `quantum_mlps[7]` execute the full Layer-7 block with zero classical parameters!).
3. **U-Net Encoder-Decoder Skip Connections & Value Residual Highway (`v_0`)**:
   - Preserves the exact 12-layer U-Net topology: encoder layers $\ell \in \{0..5\}$ push their post-MLP state $x_{\text{out}}^{(\ell)}$ onto `encoder_states`, and decoder layers $\ell \in \{6..11\}$ add $\sigma(\text{skip\_weights}[\ell - 6]) \cdot \text{encoder\_states.pop()}$ before layer $\ell$.
   - Layer 0 outputs `v_0` (projected from `v_wp` via `w_in_bridge`), which is passed to interior layers when `interior_value_mode == "v0_residual"`.
4. **Multi-Layer Trajectory Co-Distillation Warmup (`compute_multilayer_alignment_loss`)**:
   - During `0 <= training_step < interface_warmup_steps`:
     A single `with torch.no_grad():` call to `classical.extract_layer_trajectories(idx)` returns `x_0_c`, `layer_outputs[0..11]`, and `final_norm_x`.
     The alignment loss averages scale-invariant cosine MSE over all replaced layers $\ell \in \mathcal{L}_{\text{replaced}} = \mathcal{L}_{\text{attn}} \cup \mathcal{L}_{\text{mlp}}$ plus the initial embedding `x_0` and final normed feature state `x_final`:
     $$\mathcal{L}_{\text{align}}(t) = \lambda(t) \left[ 0.5\,\mathcal{D}_{\cos}(x_0^q, x_0^c) + \frac{1}{|\mathcal{L}_{\text{replaced}}|} \sum_{\ell \in \mathcal{L}_{\text{replaced}}} \mathcal{D}_{\cos}(x_{\text{out}}^{(\ell, q)}, x_{\text{out}}^{(\ell, c)}) + 0.5\,\mathcal{D}_{\cos}(x_{\text{final}}^q, x_{\text{final}}^c) \right]$$
     where $\mathcal{D}_{\cos}(a, b) = \frac{1}{2}\operatorname{MSE}(\operatorname{RMSNorm}(a), \operatorname{RMSNorm}(b))$ and $\lambda(t) = \frac{w_{\text{align}}}{2}\big(1 + \cos(\pi t / T_{\text{align}})\big)$ decays smoothly to **`0.0` at `t = interface_warmup_steps`**, after which the model trains under **100% pure Cross-Entropy (`ce_loss`)**.

---

## 4. Eight-Arm Experimental Matrix (`I73_1 .. I73_8`)

All 8 arms use the winning `(V, 12, 2, 32)` wavepacket factorization ($H=6, R=2, d_{\text{pkt}}=32$, $32 \times 32$ complex outer products per head), `WF_IN_WF_OUT`, `BERRY_PHASE_COMPLEX_QK`, `FUBINI_STUDY_SELF_ORTHO`, `TEACHER_PROCRUSTES_SPHERE`, `FROZEN_TEACHER_VE0_COUPLED`, `ENTROPY_PRESERVING_COHERENT_KET`, and `interface_warmup_steps = 250` on the `300M`-token Classical Teacher (`val_ce = 3.5845`, `top1 = 37.31%`, `top5 = 58.26%`), trained for **`800` steps (`6.55M` FineWeb tokens = `2.18%` of `300M`, evaluating every `200` steps)**:

| Arm ID | Depth Scope (`Replaced Layers`) | Attention SVD-Stiefel Rank (`Total Q-Attn Params` vs. `25.95M` Classical) | MLP Mode (`Total Q-MLP Params` vs. `56.62M` Classical) | Interior Coupling | Scientific Question Tested |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`I73_1_DEPTH3_L0_TO_L2_ATTN_R384`** | `EMB_PLUS_L0_TO_L2_ATTN` (`wte + L0..L2`, `25%` depth) | **`R384` (`7.12M` vs `7.08M` = `1.00x` Parity)** | `FROZEN_CLASSICAL_MLP` | `ATTENDED_RHO_ON_LOCAL_V` | Does replacing `wte` + the first 3 layers (`L0..L2`) at `1:1` parity beat the `300M` Teacher (`3.5845`)? |
| **`I73_2_DEPTH6_ENCODER_L0_TO_L5_ATTN_R384`** | `EMB_PLUS_L0_TO_L5_ATTN` (`wte + L0..L5`, `50%` encoder half) | **`R384` (`14.24M` vs `14.16M` = `1.00x` Parity)** | `FROZEN_CLASSICAL_MLP` | `ATTENDED_RHO_ON_LOCAL_V` | Does replacing `wte` + the **entire 6-layer U-Net encoder half (`L0..L5`)** at `1:1` parity beat `3.5845`? |
| **`I73_3_ALL12_ATTN_R768_FULL_RANK`** | **`EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11`)** | `R768` (`52.08M` across 11 attn layers) | `FROZEN_CLASSICAL_MLP` | `ATTENDED_RHO_ON_LOCAL_V` | Can replacing `wte` + **all attention layers from Layer 0 through Layer 11** ($r=768$) beat `3.5845`? |
| **`I73_4_ALL12_ATTN_R384_1TO1_PARITY`** | **`EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11`)** | **`R384` (`26.11M` vs `25.95M` = `1.00x` Parity!)** | `FROZEN_CLASSICAL_MLP` | `ATTENDED_RHO_ON_LOCAL_V` | Can replacing `wte` + **all 12 layers' attention at exact `1:1` parameter parity** beat `3.5845`? |
| **`I73_5_ALL12_ATTN_R192_2X_COMPRESSED`** | **`EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11`)** | **`R192` (`13.12M` vs `25.95M` = `0.50x` `2x` Smaller!)** | `FROZEN_CLASSICAL_MLP` | `ATTENDED_RHO_ON_LOCAL_V` | Can we **compress every attention layer in the 12-layer network by `2x`** (`13.12M` vs `25.95M`) and still beat `3.5845`? |
| **`I73_6_ALL12_ATTN_R128_3X_COMPRESSED`** | **`EMB_PLUS_ALL_12_ATTN` (`wte + All 12 Layers L0..L11`)** | **`R128` (`8.80M` vs `25.95M` = `0.34x` `3x` Smaller!)** | `FROZEN_CLASSICAL_MLP` | `ATTENDED_RHO_ON_LOCAL_V` | Can we **compress every attention layer in the 12-layer network by `3x`** (`8.80M` vs `25.95M`) and match/beat `3.5845`? |
| **`I73_7_ALL12_FULL_BLOCKS_MZ_R768`** | **`EMB_PLUS_ALL_12_FULL_BLOCKS_MZ` (`100%` Backbone `wte + ve + L0..L11 Attn & MLP`)** | `R768` (`52.08M` Q-Attn) | **`SVD_STIEFEL_MACH_ZEHNDER` (`R768`, `70.80M` Q-MLP)** | `ATTENDED_RHO_ON_LOCAL_V` | **100% Full-Backbone Quantum Replacement**: when **every embedding, attention layer, and MLP from `wte` through Layer 11** is quantum, does it beat `3.5845`? |
| **`I73_8_ALL12_FULL_BLOCKS_MZ_R384_COMPRESSED`** | **`EMB_PLUS_ALL_12_FULL_BLOCKS_MZ` (`100%` Backbone `wte + ve + L0..L11 Attn & MLP`)** | **`R384` (`26.11M` = `1.00x` Q-Attn)** | **`SVD_STIEFEL_MACH_ZEHNDER` (`R384`, `35.40M` = `0.625x` Q-MLP!)** | `ATTENDED_RHO_ON_LOCAL_V` | **Compressed 100% Full-Backbone Quantum Replacement**: `61.51M` total quantum trunk params vs. `82.58M` classical (`25.5%` smaller trunk across all 12 layers)! |

### 4.1. Dual-GPU Execution Plan
- **GPU 0 (`cuda:0`)**: Runs `I73_1_DEPTH3_L0_TO_L2_ATTN_R384`, `I73_2_DEPTH6_ENCODER_L0_TO_L5_ATTN_R384`, `I73_3_ALL12_ATTN_R768_FULL_RANK`, `I73_4_ALL12_ATTN_R384_1TO1_PARITY` -> saves to `.cache/w53a_plan073_part1_cuda0.json`.
- **GPU 1 (`cuda:1`)**: Runs `I73_5_ALL12_ATTN_R192_2X_COMPRESSED`, `I73_6_ALL12_ATTN_R128_3X_COMPRESSED`, `I73_7_ALL12_FULL_BLOCKS_MZ_R768`, `I73_8_ALL12_FULL_BLOCKS_MZ_R384_COMPRESSED` -> saves to `.cache/w53a_plan073_part2_cuda1.json`.

---

## 5. Verification Gates & Best-Practices Compliance (`tests/test_plan073_full_backbone_quantum_replacement.py`)

Before launching the GPU runs, `tests/test_plan073_full_backbone_quantum_replacement.py` must pass 4 deterministic CPU verification gates:
- **Gate H1 (Multi-Layer & Full-Block Parameter Isolation and Zero Classical Trunk Leakage)**:
  - Verifies that in `EMB_PLUS_ALL_12_ATTN`, all 11 active attention layers (`0..6, 8..11`) are replaced by `SVDStiefelWavepacketStatMechBlock`, `use_partial_key_offset` is `True` only on long-window layers `(0, 4, 11)` and `False` on short-window layers, and zero parameters in `model.classical` have `requires_grad=True`.
  - Verifies that in `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`, all 12 layers (`0..11`) have active `SVDStiefelMachZehnderInterferometer` modules and `q_value_embeds` on the unit-RMS sphere, and during `forward_features(idx)` (inference/validation) `model.classical.blocks` are never invoked.
- **Gate H2 (Mach-Zehnder Interferometer Calibration & Non-Zero Quadrature Gradients)**:
  - Verifies that `SVDStiefelMachZehnderInterferometer` calibrated from `teacher_blk` matches the teacher MLP output at Step 0 with cosine similarity `> 0.999` at full rank, and that a single backward pass produces strictly non-zero gradients on `up_proj.u_stiefel`, `up_proj.v_stiefel`, `down_proj.u_stiefel`, `down_proj.v_stiefel`, `mz_theta`, and `mz_gamma`.
- **Gate H3 (End-to-End 12-Layer Forward, Backward, Riemannian Manifold Step & Telemetry)**:
  - Runs 2 steps of `QuantumFullBackboneGraftGPT` on a synthetic CPU teacher for both `EMB_PLUS_ALL_12_ATTN` (`r=192`) and `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ` (`r=384`), asserting finite loss, strictly non-zero gradients across all active quantum layers `0..11`, Stiefel orthogonality error `< 1e-4`, and `q_wte` / `q_value_embeds` sphere RMS error `< 1e-4` after `opts.step()`.
- **Gate H4 (`docs/SCRIPT_BEST_PRACTICES.md` AST Audit)**:
  - Verifies `<= 1,000` SLOC (excluding comments/docstrings), `<= 8` arguments per function (excluding `self`/`cls`), strategy enums, top-of-file constants, and zero semicolon statement packing across `src/qdlblocks/models/quantum_full_backbone_graft_gpt.py`, `src/qdlblocks/layers/quantum_svd_stiefel_graft.py`, `tests/test_plan073_full_backbone_quantum_replacement.py`, and `examples/83_verify_plan073_full_backbone_quantum_replacement.py`.
