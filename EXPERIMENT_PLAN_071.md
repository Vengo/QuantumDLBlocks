# EXPERIMENT PLAN 071 — Wave 52G: $(V, 12, 2, 32)$ Synergistic Interior Physics (`WF_IN_WF_OUT`, `CUBIC_FERMI_DIRAC`, `CPTP_KRAUS_RANK_K`), U(1) Berry-Phase Complex Hermitian Query-Key Coupling, and `12.29M`-Token Extended Recovery

**Date**: 2026-11-01
**Author**: Antigravity (Autonomous Overnight Session — Wave 52G)
**Predecessor**: [`EXPERIMENT_PLAN_070.md`](file:///home/nixos/QuantumDLBlocks/EXPERIMENT_PLAN_070.md) (Wave 52F — Sub-Parity & 1:1 Parameter-Matched SVD-Stiefel Ranks, Liouville Commutator & Multi-Layer Grafting)
**Status**: Approved — Reviewed by Independent Context-Free Reviewer (`egm-implementation-readiness`, `0 Blockers, 0 Majors, 0 Nits`)

---

## 1. Executive Summary & Motivation

Across **Waves 52A–52E (`Plans 065–069`, `46` completed GPU runs)**, we established four foundational results:
1. **Wavefunction Granularity (`Plan 068` & `Plan 069`)**: Larger per-packet complex Hilbert dimension $d_{\text{pkt}} = 32$ (`(V, 12, 2, 32)`, `D6`: `3.5823` CE at `500` steps, `E3`: **`3.5811` CE** at `800` steps) strictly outperforms `(V, 24, 2, 16)` (`3.5839`), `(V, 2, 512)` (`3.5872`), `(V, 48, 2, 8)` (`3.5881`), `(V, 256, 2, 2)` (`3.5947`), and `(V, 2, 256)` (`3.5960`).
2. **State I/O Contract (`Plan 069`)**: Once boundary projections are calibrated via Biorthogonal SVD-Stiefel, the user's **Pure Complex Wavefunction I/O (`WF_IN_WF_OUT`, `W5`: `3.5839` CE)** and **Hybrid Born Observable I/O (`WF_IN_DENSITY_OBS_OUT`, `W6`: `3.5843` CE, `37.36%` Top-1)** slightly outperform `CAUSAL_ENSEMBLE_DM_IO` (`W1`: `3.5845` CE) on `(V, 24, 2, 16)`.
3. **Purification & Channel Physics (`Plan 069`)**: **Cubic Fermi-Dirac purification ($3\rho^2 - 2\rho^3$, `W2`: `3.5840` CE)** and **Rank-4 CPTP Kraus channels (`W4`: `3.5843` CE, `37.37%` Top-1)** outperform baseline Quadratic McWeeny + Unitary $U(16)$ (`W1`: `3.5845` CE, `37.34%` Top-1) on `(V, 24, 2, 16)`.

While **Wave 52F (`EXPERIMENT_PLAN_070.md`)** is currently running on `cuda:0` and `cuda:1` to evaluate sub-parity/1:1 parameter ranks ($r \in \{64, 128, 192, 256, 384\}$), Liouville commutator coupling, and multi-layer (`Emb + L0 + L1`) replacement, **Wave 52G (`EXPERIMENT_PLAN_071.md`)** investigates three synergistic questions:

1. **Axis Q — Synergistic Interior Physics (`WF_IN_WF_OUT`, `CUBIC_FERMI_DIRAC`, `CPTP_KRAUS_RANK_K`, `WF_IN_DENSITY_OBS_OUT`) on the Winning $(V, 12, 2, 32)$ ($d_{\text{pkt}}=32$) Geometry**:
   - In `Plan 069`, `W1..W6` were evaluated on $(V, 24, 2, 16)$ ($d_{\text{pkt}}=16$). We have not yet tested `WF_IN_WF_OUT`, `CUBIC_FERMI_DIRAC`, `CPTP_KRAUS_RANK_K`, or their combination on the winning **$32 \times 32$ per-packet Hilbert space `(V, 12, 2, 32)`**! Because a $32 \times 32$ density matrix has twice as many eigenvalues ($32$ vs. $16$), Cubic Fermi-Dirac purification ($3\rho^2 - 2\rho^3$) and Rank-4 CPTP Kraus evolution have an even stronger spectral effect on `(V, 12, 2, 32)`.
2. **Axis R — U(1) Berry-Phase Complex Hermitian Query-Key Attention Scoring (`QuantumQKPhaseMode`)**:
   - Standard real dot-product SDPA between $Q_{t, h} = (Q_{\text{re}}, Q_{\text{im}})$ and $K_{s, h} = (K_{\text{re}}, K_{\text{im}})$ computes only $\operatorname{Re}\langle Q_{t, h} | K_{s, h} \rangle = Q_{\text{re}}\cdot K_{\text{re}} + Q_{\text{im}}\cdot K_{\text{im}}$, discarding the imaginary part $\operatorname{Im}\langle Q_{t, h} | K_{s, h} \rangle = Q_{\text{re}}\cdot K_{\text{im}} - Q_{\text{im}}\cdot K_{\text{re}}$!
   - By applying a learnable per-head, per-mode U(1) Berry phase rotation $e^{i \Phi_{h, r, d}}$ (`berry_qk_phase` $\in \mathbb{R}^{H \times R \times d_h}$, initialized to $0$ so Step-0 teacher calibration is exact) to the complex query ket $(Q_{\text{re}}, Q_{\text{im}})$ prior to SDPA:
     $$Q'_{\text{re}} = \cos(\Phi) Q_{\text{re}} - \sin(\Phi) Q_{\text{im}}, \qquad Q'_{\text{im}} = \sin(\Phi) Q_{\text{re}} + \cos(\Phi) Q_{\text{im}}$$
     the fused FlashAttention/SDPA dot product $Q'_{t, h} \cdot K_{s, h}$ **exactly computes the phase-shifted Hermitian inner product**:
     $$Q'_{t, h} \cdot K_{s, h} = \sum_{r, d} \left[\cos(\Phi_{h,r,d})\operatorname{Re}\langle Q_{t,h,r,d}|K_{s,h,r,d}\rangle + \sin(\Phi_{h,r,d})\operatorname{Im}\langle Q_{t,h,r,d}|K_{s,h,r,d}\rangle\right] = \operatorname{Re}\langle e^{-i\Phi_h} Q_{t,h} | K_{s,h} \rangle$$
     capturing both real and imaginary quantum interference in attention weights with **zero custom kernel overhead**!
3. **Axis S — Extended Sub-`300M` Horizon (`1,200` Steps = `9.83M` Tokens = `3.28%` of `300M`) at Both Full Rank ($r=768$) and Exact 1:1 Parameter Parity ($r=384$, `2.36M` L0 Params)**:
   - Pushing the unified $(V, 12, 2, 32)$ champion configuration to `1,200` steps (`9.83M` tokens) on the `300M` Classical Teacher (`3.5845` CE) at both $r=768$ and exact 1:1 parameter parity ($r=384$, `2.36M` L0 params = $1.00\times$ Classical Layer-0 Attention).

---

## 2. Mathematical & Algorithmic Specifications

### 2.1 Axis R: U(1) Berry-Phase Complex Hermitian Query-Key Coupling (`QuantumQKPhaseMode`)

We define a new strategy enum `QuantumQKPhaseMode(str, Enum)` in [`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py) (re-exported in `quantum_frontend_graft.py`):
- `QuantumQKPhaseMode.STANDARD_REAL_DOT = "standard_real_dot"` (default baseline: $\Phi = 0$)
- `QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK = "berry_phase_complex_qk"`

When `qk_phase_mode == QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK`:
1. `SVDStiefelWavepacketStatMechBlock.__init__` registers `self.berry_qk_phase = nn.Parameter(torch.zeros(self.num_heads, self.v_rank, self.d_h, dtype=torch.float32), requires_grad=True)` (when `STANDARD_REAL_DOT`, `requires_grad=False`).
2. `self.berry_qk_phase` is included in `SVDStiefelWavepacketStatMechBlock.scalar_parameters()` when `requires_grad` is `True`, and clamped to $[-\pi, \pi]$ (`berry_qk_phase.clamp_(-3.1416, 3.1416)`) in `restrict_scalars_()`.
3. In `SVDStiefelWavepacketStatMechBlock.forward`, after rotary embedding `q_rot` `(B, T, H, 2 * R * d_h)` is computed, if `self.qk_phase_mode == QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK`, helper `apply_berry_qk_phase(q_rot, self.berry_qk_phase)` reshapes `q_rot` to `(B, T, H, R, 2, d_h)`:
   ```python
   def apply_berry_qk_phase(q_rot: torch.Tensor, berry_phase_hrd: torch.Tensor) -> torch.Tensor:
       """Rotate complex query kets `(B, T, H, 2*R*d_h)` by learnable U(1) Berry phase `berry_phase_hrd` `(H, R, d_h)`."""
       bsz, seq_len, n_h, head_dim = q_rot.shape
       v_rank, d_h = berry_phase_hrd.size(1), berry_phase_hrd.size(2)
       qc = q_rot.view(bsz, seq_len, n_h, v_rank, 2, d_h)
       qr, qi = qc[..., 0, :], qc[..., 1, :]
       phi = berry_phase_hrd.to(q_rot.dtype).view(1, 1, n_h, v_rank, d_h)
       c_phi, s_phi = torch.cos(phi), torch.sin(phi)
       qr_out = c_phi * qr - s_phi * qi
       qi_out = s_phi * qr + c_phi * qi
       return torch.stack([qr_out, qi_out], dim=-2).reshape(bsz, seq_len, n_h, head_dim)
   ```
   Notice that at initialization ($\Phi = 0$), $\cos(\Phi) = 1, \sin(\Phi) = 0$, so `q_rot` is **bit-identical** to `STANDARD_REAL_DOT` at Step 0, preserving 100% of the Procrustes/SVD teacher calibration!

---

## 3. Experimental Design (`8` Arms Across Phase 1 & Phase 2)

### Phase 1: 6-Arm $(V, 12, 2, 32)$ Synergistic Interior Physics & Berry-Phase QK Tournament (`500` Steps = `4.10M` Tokens on `300M` Teacher `3.5845`)

All Phase-1 arms use `wf_shape = SHAPE_12x2x32` ($d_{\text{pkt}}=32$), `wte_init_mode = TEACHER_PROCRUSTES_SPHERE`, `self_ortho_mode = FUBINI_STUDY_SELF_ORTHO`, `recovery_schedule_mode = PERSISTENT_COSINE_FEAT_AND_CE` (`warmup_steps = 300`), and train for `500` steps (`4.10M` tokens) on the `300M` Classical Teacher (`3.5845` CE, `37.31%` Top-1, `58.26%` Top-5):

| Arm ID | SVD-Stiefel Rank | State I/O Mode | Purification Mode | Channel Mode | QK Phase Mode | Scientific Question |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`G71_1_PKT32_PURE_WF_IO`** | `R768` (`4.72M`) | **`WF_IN_WF_OUT`** | `QUADRATIC_MCWEENY` | `UNITARY_U16` | `STANDARD_REAL_DOT` | Does pure wavefunction I/O (`W5` winner) improve `(V, 12, 2, 32)` beyond `D6` (`3.5823`)? |
| **`G71_2_PKT32_CUBIC_FD`** | `R768` (`4.72M`) | `CAUSAL_ENSEMBLE_DM_IO` | **`CUBIC_FERMI_DIRAC`** | `UNITARY_U16` | `STANDARD_REAL_DOT` | Does Cubic Fermi-Dirac ($3\rho^2 - 2\rho^3$, `W2` winner) improve $32 \times 32$ density purification? |
| **`G71_3_PKT32_KRAUS_RANK4`** | `R768` (`4.73M`) | `CAUSAL_ENSEMBLE_DM_IO` | `QUADRATIC_MCWEENY` | **`CPTP_KRAUS_RANK_K`** | `STANDARD_REAL_DOT` | Does Rank-4 CPTP Kraus evolution (`W4` Top-1 winner) improve $32 \times 32$ wavepacket transport? |
| **`G71_4_PKT32_BERRY_PHASE_QK`** | `R768` (`4.72M`) | `CAUSAL_ENSEMBLE_DM_IO` | `QUADRATIC_MCWEENY` | `UNITARY_U16` | **`BERRY_PHASE_COMPLEX_QK`** | Does U(1) Berry-phase Hermitian QK inner product improve causal outer-product attention? |
| **`G71_5_PKT32_SYNERGY_FULL`** | `R768` (`4.72M`) | **`WF_IN_WF_OUT`** | **`CUBIC_FERMI_DIRAC`** | `UNITARY_U16` | **`BERRY_PHASE_COMPLEX_QK`** | Full synergy of `WF_IN_WF_OUT` + `CUBIC_FERMI_DIRAC` + `BERRY_PHASE_COMPLEX_QK` on `(V, 12, 2, 32)`. |
| **`G71_6_PKT32_SYNERGY_R384_PARITY`** | **`R384` (`2.36M`, 1:1 Parity)** | **`WF_IN_WF_OUT`** | **`CUBIC_FERMI_DIRAC`** | `UNITARY_U16` | **`BERRY_PHASE_COMPLEX_QK`** | Full synergy at **exact 1:1 Layer-0 parameter parity** (`2.36M` params) with Classical Attention! |

### Phase 2: Extended `1,200`-Step (`9.83M`-Token = `3.28%` of `300M`) Champion Runs
- **`G71_7_CHAMPION_R768_1200STEPS`**: Runs the winning `(V, 12, 2, 32)` configuration at $r=768$ for `1,200` steps (`9.83M` tokens) on the `300M` Classical Teacher (`3.5845` CE) to establish our best overall validation cross-entropy and Top-1/Top-5 accuracy.
- **`G71_8_CHAMPION_R384_PARITY_1200STEPS`**: Runs the winning `(V, 12, 2, 32)` configuration at **exact 1:1 parameter parity ($r=384$, `2.36M` L0 params)** for `1,200` steps (`9.83M` tokens) on the `300M` Classical Teacher (`3.5845` CE).

---

## 4. Files to Create / Modify & Verification Gates (`H1..H4`)

1. **[`src/qdlblocks/layers/quantum_svd_stiefel_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_svd_stiefel_graft.py)** (`< 650` SLOC):
   - Add `QuantumQKPhaseMode` enum (`STANDARD_REAL_DOT`, `BERRY_PHASE_COMPLEX_QK`) and `apply_berry_qk_phase(q_rot, berry_phase_hrd)`.
   - Add `qk_phase_mode: QuantumQKPhaseMode = QuantumQKPhaseMode.STANDARD_REAL_DOT` to `SVDStiefelWavepacketStatMechBlock.__init__`, register `self.berry_qk_phase`, return it in `scalar_parameters()` when `requires_grad` is `True`, and clamp it to `[-3.1416, 3.1416]` in `restrict_scalars_()`.
2. **[`src/qdlblocks/layers/quantum_frontend_graft.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_frontend_graft.py)** (`<= 995` SLOC):
   - Re-export `QuantumQKPhaseMode` and `apply_berry_qk_phase`, add `qk_phase_mode: QuantumQKPhaseMode = QuantumQKPhaseMode.STANDARD_REAL_DOT` to `QuantumFrontEndGraftConfig`, and pass `self.graft_cfg.qk_phase_mode` into `SVDStiefelWavepacketStatMechBlock`.
3. **[`tests/test_plan071_pkt32_synergy_and_berry_phase_qk.py`](file:///home/nixos/QuantumDLBlocks/tests/test_plan071_pkt32_synergy_and_berry_phase_qk.py)** (`< 250` SLOC):
   - **Gate H1**: Verifies `apply_berry_qk_phase` norm preservation ($\|Q'\|_2 = \|Q\|_2$), exact identity at $\Phi = 0$, and exact analytical Hermitian inner product $Q'_t \cdot K_s = \operatorname{Re}(e^{-i\Phi}\langle Q_t | K_s \rangle)$ for non-zero $\Phi$.
   - **Gate H2**: Verifies finite gradients to `berry_qk_phase` and clamping in `restrict_quantum_scalars_()` under `BERRY_PHASE_COMPLEX_QK`.
   - **Gate H3**: Verifies all 6 Phase-1 arm configs (`G71_1..G71_6`) instantiate and run forward+backward+step cleanly on `(V, 12, 2, 32)`.
   - **Gate H4**: AST best-practices audit (`<= 1,000` SLOC, `<= 8` args per function) across all modified and new files.
4. **[`examples/81_verify_plan071_pkt32_synergy_and_berry_phase_qk.py`](file:///home/nixos/QuantumDLBlocks/examples/81_verify_plan071_pkt32_synergy_and_berry_phase_qk.py)** (`< 200` SLOC):
   - CLI benchmark runner executing `G71_1..G71_8` on `cuda:0` and `cuda:1`.
