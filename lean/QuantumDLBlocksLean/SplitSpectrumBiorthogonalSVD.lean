/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 2: Theorem 2 — Split-Spectrum Biorthogonal SVD Normal-Space Decomposition
(Wave 67/68: `project_coupled_svd_raw_exact_degenerate`, `filter_coupled_svd_post_ns_corotation`,
and `BiorthogonalSVDStiefelProjection`).
-/
import QuantumDLBlocksLean.Core

namespace QuantumDLBlocksLean

/-- The `(i, j)` coordinate entry of `U^T dM V` for `M(U, σ, V) = U diag(σ) V^T`
under vertical skew-symmetric perturbations `(δU, δV) = (U Ω_U, V Ω_V)`:
`(U^T dM V)_{ij} = (Ω_U diag(σ) - diag(σ) Ω_V)_{ij} = σ_j (Ω_U)_{ij} - σ_i (Ω_V)_{ij}`. -/
def svdCoordDiff (σ_i σ_j ωU_ij ωV_ij : Int) : Int :=
  σ_j * ωU_ij - σ_i * ωV_ij

/-- Under skew-symmetry `(Ω_U)_{ji} = -(Ω_U)_{ij}` and `(Ω_V)_{ji} = -(Ω_V)_{ij}`,
the transposed `(j, i)` coordinate entry of `U^T dM V` is `-(σ_i (Ω_U)_{ij} - σ_j (Ω_V)_{ij})`. -/
theorem svdCoordDiff_skew_transpose (σ_i σ_j ωU_ij ωV_ij : Int) :
    svdCoordDiff σ_j σ_i (-ωU_ij) (-ωV_ij) = -(σ_i * ωU_ij - σ_j * ωV_ij) := by
  unfold svdCoordDiff
  rw [Int.mul_neg, Int.mul_neg]
  omega

/-- Algebraic determinant elimination identity isolating `ωU_ij` and `ωV_ij` from the
coupled `(i, j)` and `(j, i)` SVD differential equations. -/
theorem svd_split_spectrum_determinant_elimination
    (σ_i σ_j ωU_ij ωV_ij : Int) :
    (σ_j * σ_j - σ_i * σ_i) * ωU_ij =
      σ_j * svdCoordDiff σ_i σ_j ωU_ij ωV_ij +
      σ_i * svdCoordDiff σ_j σ_i (-ωU_ij) (-ωV_ij) ∧
    (σ_j * σ_j - σ_i * σ_i) * ωV_ij =
      σ_i * svdCoordDiff σ_i σ_j ωU_ij ωV_ij +
      σ_j * svdCoordDiff σ_j σ_i (-ωU_ij) (-ωV_ij) := by
  unfold svdCoordDiff
  constructor
  · simp only [Int.sub_mul, Int.mul_sub, Int.mul_neg, ← Int.mul_assoc, Int.mul_comm σ_i σ_j]
    omega
  · simp only [Int.sub_mul, Int.mul_sub, Int.mul_neg, ← Int.mul_assoc, Int.mul_comm σ_i σ_j]
    omega

/-- **Theorem 2(a) — Degenerate Singular Values Synchronous Spin
(`theorem2a_degenerate_spectrum_synchronous_spin`)**:
When `σ_i = σ_j = σ ≠ 0`, the vertical kernel condition
`svdCoordDiff σ σ ωU_ij ωV_ij = 0 ∧ svdCoordDiff σ σ (-ωU_ij) (-ωV_ij) = 0`
is equivalent to exact synchronous spin `ωU_ij = ωV_ij`. -/
theorem theorem2a_degenerate_spectrum_synchronous_spin
    (σ ωU_ij ωV_ij : Int) (hσ : σ ≠ 0) :
    (svdCoordDiff σ σ ωU_ij ωV_ij = 0 ∧ svdCoordDiff σ σ (-ωU_ij) (-ωV_ij) = 0) ↔
      ωU_ij = ωV_ij := by
  unfold svdCoordDiff
  constructor
  · intro ⟨h1, _⟩
    have h_diff : σ * (ωU_ij - ωV_ij) = 0 := by
      rw [Int.mul_sub]
      omega
    cases Int.mul_eq_zero.mp h_diff with
    | inl h_zero => exact False.elim (hσ h_zero)
    | inr h_eq => omega
  · intro h_eq
    subst h_eq
    omega

/-- **Theorem 2(b) — Split-Spectrum Trivial Vertical Kernel
(`theorem2b_split_spectrum_zero_vertical_spin`)**:
When `σ_i^2 ≠ σ_j^2`, the coupled skew-symmetric vertical kernel system
`svdCoordDiff σ_i σ_j ωU_ij ωV_ij = 0 ∧ svdCoordDiff σ_j σ_i (-ωU_ij) (-ωV_ij) = 0`
forces `ωU_ij = 0 ∧ ωV_ij = 0`. -/
theorem theorem2b_split_spectrum_zero_vertical_spin
    (σ_i σ_j ωU_ij ωV_ij : Int) (h_split : σ_j * σ_j ≠ σ_i * σ_i) :
    (svdCoordDiff σ_i σ_j ωU_ij ωV_ij = 0 ∧ svdCoordDiff σ_j σ_i (-ωU_ij) (-ωV_ij) = 0) ↔
      (ωU_ij = 0 ∧ ωV_ij = 0) := by
  have h_det_ne : σ_j * σ_j - σ_i * σ_i ≠ 0 := by omega
  have ⟨h_elim_u, h_elim_v⟩ := svd_split_spectrum_determinant_elimination σ_i σ_j ωU_ij ωV_ij
  constructor
  · intro ⟨h1, h2⟩
    rw [h1, h2] at h_elim_u h_elim_v
    simp only [Int.mul_zero, Int.add_zero] at h_elim_u h_elim_v
    have hu : ωU_ij = 0 := by
      cases Int.mul_eq_zero.mp h_elim_u with
      | inl h_abs => exact False.elim (h_det_ne h_abs)
      | inr h_z => exact h_z
    have hv : ωV_ij = 0 := by
      cases Int.mul_eq_zero.mp h_elim_v with
      | inl h_abs => exact False.elim (h_det_ne h_abs)
      | inr h_z => exact h_z
    exact ⟨hu, hv⟩
  · intro ⟨hu, hv⟩
    subst hu
    subst hv
    unfold svdCoordDiff
    simp

/-- Helper: strictly positive distinct integers have distinct squares (`σ_i ≠ σ_j → σ_j^2 ≠ σ_i^2`). -/
theorem sq_ne_sq_of_pos_ne (σ_i σ_j : Int) (hi : 0 < σ_i) (hj : 0 < σ_j) (hne : σ_i ≠ σ_j) :
    σ_j * σ_j ≠ σ_i * σ_i := by
  intro h_eq
  have h_fact : (σ_j - σ_i) * (σ_j + σ_i) = σ_j * σ_j - σ_i * σ_i := by
    rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_comm σ_i σ_j]
    omega
  have h_prod_zero : (σ_j - σ_i) * (σ_j + σ_i) = 0 := by omega
  cases Int.mul_eq_zero.mp h_prod_zero with
  | inl h_sub => omega
  | inr h_sum => omega

/-- **Corollary 2(c) — Non-Zero Vertical Spin on Positive Split Spectrum Alters Operator
(`theorem2c_positive_split_spectrum_physical_spin`)**:
When `0 < σ_i`, `0 < σ_j`, and `σ_i ≠ σ_j`, any non-zero vertical spin `(ωU_ij ≠ 0 ∨ ωV_ij ≠ 0)`
produces a strictly non-zero physical operator perturbation, proving why
`project_coupled_svd_raw_exact_degenerate` must mask by `|σ_i - σ_j| = 0`. -/
theorem theorem2c_positive_split_spectrum_physical_spin
    (σ_i σ_j ωU_ij ωV_ij : Int) (hi : 0 < σ_i) (hj : 0 < σ_j) (hne : σ_i ≠ σ_j)
    (h_spin : ωU_ij ≠ 0 ∨ ωV_ij ≠ 0) :
    svdCoordDiff σ_i σ_j ωU_ij ωV_ij ≠ 0 ∨ svdCoordDiff σ_j σ_i (-ωU_ij) (-ωV_ij) ≠ 0 := by
  have h_split : σ_j * σ_j ≠ σ_i * σ_i := sq_ne_sq_of_pos_ne σ_i σ_j hi hj hne
  have h_iff := (theorem2b_split_spectrum_zero_vertical_spin σ_i σ_j ωU_ij ωV_ij h_split).mp
  by_cases h1 : svdCoordDiff σ_i σ_j ωU_ij ωV_ij = 0
  · by_cases h2 : svdCoordDiff σ_j σ_i (-ωU_ij) (-ωV_ij) = 0
    · have ⟨hu, hv⟩ := h_iff ⟨h1, h2⟩
      cases h_spin with
      | inl h_nu => exact False.elim (h_nu hu)
      | inr h_nv => exact False.elim (h_nv hv)
    · exact Or.inr h2
  · exact Or.inl h1

#print axioms theorem2a_degenerate_spectrum_synchronous_spin
#print axioms theorem2b_split_spectrum_zero_vertical_spin
#print axioms theorem2c_positive_split_spectrum_physical_spin

end QuantumDLBlocksLean
