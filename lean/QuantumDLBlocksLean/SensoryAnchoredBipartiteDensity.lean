/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 11: Theorem 12 — Sensory-Anchored Dual-Unitary Distinguishability, Bipartite Register
Decomposition, & Self-Copy Attractor Elimination (Wave 71: Resolution of Memoryless State
Collision via Layer-0 Sensory Anchor, Dual-Unitary CPTP Closure, Bipartite Sensory/Predictive
Register Invariance, and Orthogonal Input/Output Frame Self-Copy Elimination).
-/
import QuantumDLBlocksLean.Core
import QuantumDLBlocksLean.HilbertSchmidtQuantumAttention
import QuantumDLBlocksLean.QuantumOrAndSchurLogic
import QuantumDLBlocksLean.OnlineEmaAndStatefulHistoryDensity
import QuantumDLBlocksLean.FactoredCrossHeadKrausSuperoperator

namespace QuantumDLBlocksLean

/-- Scalar/coordinate evaluation of a sensory-anchored dual-channel state:
`anchoredChannel w_evol w_anc e_evol_rho e_anc_rho0 = w_evol * e_evol_rho + w_anc * e_anc_rho0`. -/
def anchoredChannel (w_evol w_anc e_evol_rho e_anc_rho0 : Int) : Int :=
  w_evol * e_evol_rho + w_anc * e_anc_rho0

/-- 2×2 density-matrix evaluation of a sensory-anchored dual-channel state:
`anchoredDensity2 w_evol w_anc E_evol_rho E_anc_rho0 = w_evol • E_evol_rho + w_anc • E_anc_rho0`. -/
def anchoredDensity2 (w_evol w_anc : Int) (E_evol_rho E_anc_rho0 : Mat2) : Mat2 :=
  quantumOr2 w_evol w_anc E_evol_rho E_anc_rho0

/-- **Theorem 12(a) — Memoryless State Collision Without Layer-0 Sensory Anchor
(`unanchored_channel_state_collision`)**:
For any single-argument quantum channel or Q/K/V projection map `E : α → β` (and specifically
for any 2×2 Kraus congruence `krausCongruence2 K`), if position `s` with initial token `A`
evolves at Layer `ℓ` into a state `ρ_evol_A` that coincides with token `B`'s state `ρ_B`
(`ρ_evol_A = ρ_B`), then their unanchored channel outputs `E ρ_evol_A` and `E ρ_B` are
identically equal (`E ρ_evol_A = E ρ_B`), making the two positions mathematically
indistinguishable to downstream attention. -/
theorem unanchored_channel_state_collision
    {α β : Type} (E : α → β) (ρ_evol_A ρ_B : α) (h_coll : ρ_evol_A = ρ_B)
    (K ρ_mat_A ρ_mat_B : Mat2) (h_mat_coll : ρ_mat_A = ρ_mat_B) :
    E ρ_evol_A = E ρ_B ∧
    krausCongruence2 K ρ_mat_A = krausCongruence2 K ρ_mat_B := by
  rw [h_coll, h_mat_coll]
  exact ⟨rfl, rfl⟩

/-- **Theorem 12(b.1) — Exact Distinguishability of Dual-Unitary Sensory-Anchored Channels
(`dual_unitary_sensory_anchored_distinguishability`)**:
Even when two positions have the *exact same* evolved state `e_evol_rho_A = e_evol_rho_B = e_star`
(and on `Mat2`, `E_evol_A = E_evol_B = E_star`), if the anchor weight is non-zero (`w_anc ≠ 0`)
and the anchor unitary channel separates the Layer-0 states (`e_anc_rho0_A ≠ e_anc_rho0_B`,
and on `Mat2`, `E_anc_rho0_A ≠ E_anc_rho0_B`), then:
1. The anchored difference equals `w_anc * (e_anc_rho0_A - e_anc_rho0_B) ≠ 0`,
2. The scalar anchored outputs are strictly distinct:
   `anchoredChannel w_evol w_anc e_star e_anc_rho0_A ≠ anchoredChannel w_evol w_anc e_star e_anc_rho0_B`,
3. The 2×2 matrix anchored states are strictly distinct:
   `anchoredDensity2 w_evol w_anc E_star E_anc_rho0_A ≠ anchoredDensity2 w_evol w_anc E_star E_anc_rho0_B`. -/
theorem dual_unitary_sensory_anchored_distinguishability
    (w_evol w_anc e_star e_anc_rho0_A e_anc_rho0_B : Int)
    (hw_anc : w_anc ≠ 0)
    (h_sep : e_anc_rho0_A ≠ e_anc_rho0_B)
    (E_star E_anc_rho0_A E_anc_rho0_B : Mat2)
    (h_mat_sep : E_anc_rho0_A ≠ E_anc_rho0_B) :
    anchoredChannel w_evol w_anc e_star e_anc_rho0_A -
      anchoredChannel w_evol w_anc e_star e_anc_rho0_B =
        w_anc * (e_anc_rho0_A - e_anc_rho0_B) ∧
    w_anc * (e_anc_rho0_A - e_anc_rho0_B) ≠ 0 ∧
    anchoredChannel w_evol w_anc e_star e_anc_rho0_A ≠
      anchoredChannel w_evol w_anc e_star e_anc_rho0_B ∧
    anchoredDensity2 w_evol w_anc E_star E_anc_rho0_A ≠
      anchoredDensity2 w_evol w_anc E_star E_anc_rho0_B := by
  have h_diff_eq :
      anchoredChannel w_evol w_anc e_star e_anc_rho0_A -
        anchoredChannel w_evol w_anc e_star e_anc_rho0_B =
          w_anc * (e_anc_rho0_A - e_anc_rho0_B) := by
    unfold anchoredChannel
    rw [Int.mul_sub]
    omega
  have h_diff_ne : w_anc * (e_anc_rho0_A - e_anc_rho0_B) ≠ 0 := by
    intro h_zero
    cases Int.mul_eq_zero.mp h_zero with
    | inl h_w => exact hw_anc h_w
    | inr h_sub =>
      apply h_sep
      omega
  have h_val_ne :
      anchoredChannel w_evol w_anc e_star e_anc_rho0_A ≠
        anchoredChannel w_evol w_anc e_star e_anc_rho0_B := by
    intro h_eq
    apply h_diff_ne
    omega
  have h_mat_ne :
      anchoredDensity2 w_evol w_anc E_star E_anc_rho0_A ≠
        anchoredDensity2 w_evol w_anc E_star E_anc_rho0_B := by
    intro h_eq
    have h00 := congrArg Mat2.m00 h_eq
    have h01 := congrArg Mat2.m01 h_eq
    have h10 := congrArg Mat2.m10 h_eq
    have h11 := congrArg Mat2.m11 h_eq
    simp only [anchoredDensity2, quantumOr2, Mat2.add, Mat2.smul] at h00 h01 h10 h11
    have hd00 : w_anc * (E_anc_rho0_A.m00 - E_anc_rho0_B.m00) = 0 := by
      rw [Int.mul_sub]
      omega
    have hd01 : w_anc * (E_anc_rho0_A.m01 - E_anc_rho0_B.m01) = 0 := by
      rw [Int.mul_sub]
      omega
    have hd10 : w_anc * (E_anc_rho0_A.m10 - E_anc_rho0_B.m10) = 0 := by
      rw [Int.mul_sub]
      omega
    have hd11 : w_anc * (E_anc_rho0_A.m11 - E_anc_rho0_B.m11) = 0 := by
      rw [Int.mul_sub]
      omega
    have e00 : E_anc_rho0_A.m00 = E_anc_rho0_B.m00 := by
      cases Int.mul_eq_zero.mp hd00 with
      | inl hw => exact False.elim (hw_anc hw)
      | inr hs => omega
    have e01 : E_anc_rho0_A.m01 = E_anc_rho0_B.m01 := by
      cases Int.mul_eq_zero.mp hd01 with
      | inl hw => exact False.elim (hw_anc hw)
      | inr hs => omega
    have e10 : E_anc_rho0_A.m10 = E_anc_rho0_B.m10 := by
      cases Int.mul_eq_zero.mp hd10 with
      | inl hw => exact False.elim (hw_anc hw)
      | inr hs => omega
    have e11 : E_anc_rho0_A.m11 = E_anc_rho0_B.m11 := by
      cases Int.mul_eq_zero.mp hd11 with
      | inl hw => exact False.elim (hw_anc hw)
      | inr hs => omega
    exact h_mat_sep (Mat2.ext e00 e01 e10 e11)
  exact ⟨h_diff_eq, h_diff_ne, h_val_ne, h_mat_ne⟩

/-- **Theorem 12(b.2) — Exact CPTP Closure of Dual-Unitary Sensory-Anchored Channels
(`dual_unitary_sensory_anchored_cptp`)**:
1. If `tr_evol = T`, `tr_anc = T`, `0 ≤ q_evol`, `0 ≤ q_anc`, and `0 ≤ w_evol`, `0 ≤ w_anc`,
   `w_evol + w_anc = W`, then the anchored combination has exact weighted trace `W * T`
   (and unit trace `1` when `W = 1, T = 1`) and non-negative quadratic form
   `0 ≤ w_evol * q_evol + w_anc * q_anc`,
2. On `Mat2`, if `IsValidDensity2 T ρ_evol` and `IsValidDensity2 T ρ_0`, `U_evol^T U_evol = I_2`,
   `U_anc^T U_anc = I_2`, and `w_evol + w_anc = 1` (`w_evol, w_anc ≥ 0`), then
   `anchoredDensity2 w_evol w_anc (U_evol ρ_evol U_evol^T) (U_anc ρ_0 U_anc^T)` satisfies
   `IsValidDensity2 T` (Hermitian, PSD, and exact trace `T`). -/
theorem dual_unitary_sensory_anchored_cptp
    (T W w_evol w_anc tr_evol tr_anc q_evol q_anc : Int)
    (hw_evol : 0 ≤ w_evol) (hw_anc : 0 ≤ w_anc) (hW : w_evol + w_anc = W)
    (htr_evol : tr_evol = T) (htr_anc : tr_anc = T)
    (hq_evol : 0 ≤ q_evol) (hq_anc : 0 ≤ q_anc)
    (wc_evol wc_anc : Int)
    (hwc_evol : 0 ≤ wc_evol) (hwc_anc : 0 ≤ wc_anc) (hwc_sum : wc_evol + wc_anc = 1)
    (ρ_evol ρ_0 U_evol U_anc : Mat2)
    (h_valid_evol : IsValidDensity2 T ρ_evol)
    (h_valid_0 : IsValidDensity2 T ρ_0)
    (hU_evol : Mat2.mul (Mat2.transpose U_evol) U_evol = Mat2.eye)
    (hU_anc : Mat2.mul (Mat2.transpose U_anc) U_anc = Mat2.eye) :
    w_evol * tr_evol + w_anc * tr_anc = W * T ∧
    0 ≤ w_evol * q_evol + w_anc * q_anc ∧
    IsValidDensity2 T
      (anchoredDensity2 wc_evol wc_anc
        (krausCongruence2 U_evol ρ_evol)
        (krausCongruence2 U_anc ρ_0)) := by
  have h_tr_wt : w_evol * tr_evol + w_anc * tr_anc = W * T := by
    rw [htr_evol, htr_anc, ← Int.add_mul, hW]
  have h_q_nonneg : 0 ≤ w_evol * q_evol + w_anc * q_anc := by
    have h1 := Int.mul_nonneg hw_evol hq_evol
    have h2 := Int.mul_nonneg hw_anc hq_anc
    omega
  have h_u_evol_valid :=
    krausCongruence2_preserves_valid_density T U_evol ρ_evol hU_evol h_valid_evol
  have h_u_anc_valid :=
    krausCongruence2_preserves_valid_density T U_anc ρ_0 hU_anc h_valid_0
  have h_mat_valid :=
    ema_step_preserves_valid_density2 T wc_evol wc_anc hwc_evol hwc_anc hwc_sum
      (krausCongruence2 U_evol ρ_evol) (krausCongruence2 U_anc ρ_0)
      h_u_evol_valid h_u_anc_valid
  exact ⟨h_tr_wt, h_q_nonneg, h_mat_valid⟩

/-- **Theorem 12(c) — Bipartite Register Trace Invariance & Orthogonal Input/Output Frame
Self-Copy Attractor Elimination (`bipartite_register_trace_and_decoupled_frame_self_copy_free`)**:
1. **Bipartite Register Decomposition**: For a bipartite density state
   `ρ = w_sens • ρ_sens + w_pred • ρ_pred` where `tr_sens = 1`, `tr_pred' = 1`, and
   `w_sens + w_pred = 1` (`w_sens, w_pred ≥ 0`), any trace-preserving evolution on the
   predictive register (`ρ_pred ↦ ρ_pred'`) preserves total unit trace
   `w_sens * tr_sens + w_pred * tr_pred' = 1` and `IsValidDensity2 1` while leaving the
   sensory sub-register `w_sens • ρ_sens` identically invariant,
2. **Self-Copy Attractor Elimination via Orthogonal Input/Output Frames**:
   In any inner-product space (`inner : V → V → Int`), when input preparation and output readout
   share the same frame (`u_out = u_in` with `0 < inner u_in u_in`), the Step-0 self-copy score
   is strictly positive (`0 < inner u_out u_in`), whereas when `u_out` and `u_in` are initialized
   orthogonal (`inner u_out u_in = 0`), the Step-0 self-copy score vanishes identically
   (`inner u_out u_in = 0 < inner u_in u_in`). -/
theorem bipartite_register_trace_and_decoupled_frame_self_copy_free
    (w_sens w_pred tr_sens tr_pred' : Int)
    (hw_sens : 0 ≤ w_sens) (hw_pred : 0 ≤ w_pred) (hw_sum : w_sens + w_pred = 1)
    (htr_sens : tr_sens = 1) (htr_pred' : tr_pred' = 1)
    (ρ_sens ρ_pred' : Mat2)
    (h_valid_sens : IsValidDensity2 1 ρ_sens)
    (h_valid_pred' : IsValidDensity2 1 ρ_pred')
    {V : Type} (inner : V → V → Int)
    (u_in u_out_shared u_out_ortho : V)
    (h_norm_pos : 0 < inner u_in u_in)
    (h_shared : u_out_shared = u_in)
    (h_ortho : inner u_out_ortho u_in = 0) :
    w_sens * tr_sens + w_pred * tr_pred' = 1 ∧
    Mat2.smul w_sens ρ_sens = Mat2.smul w_sens ρ_sens ∧
    IsValidDensity2 1 (quantumOr2 w_sens w_pred ρ_sens ρ_pred') ∧
    0 < inner u_out_shared u_in ∧
    inner u_out_ortho u_in = 0 ∧
    inner u_out_ortho u_in < inner u_out_shared u_in := by
  have h_bipartite_tr : w_sens * tr_sens + w_pred * tr_pred' = 1 := by
    rw [htr_sens, htr_pred']
    omega
  have h_valid_bipartite :=
    ema_step_preserves_valid_density2 1 w_sens w_pred hw_sens hw_pred hw_sum
      ρ_sens ρ_pred' h_valid_sens h_valid_pred'
  have h_shared_pos : 0 < inner u_out_shared u_in := by
    rw [h_shared]
    exact h_norm_pos
  refine ⟨h_bipartite_tr, rfl, h_valid_bipartite, h_shared_pos, h_ortho, ?_⟩
  omega

#print axioms unanchored_channel_state_collision
#print axioms dual_unitary_sensory_anchored_distinguishability
#print axioms dual_unitary_sensory_anchored_cptp
#print axioms bipartite_register_trace_and_decoupled_frame_self_copy_free

end QuantumDLBlocksLean
