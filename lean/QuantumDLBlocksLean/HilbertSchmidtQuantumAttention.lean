/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 6: Theorem 6 — Hilbert–Schmidt Born Similarity Positivity, Isometric Outer-Product Flattening,
& Exact Right `U(R)`-Gauge Invariance (Wave 69 `EXPERIMENT_PLAN_089.md`: Pure Density-Matrix Attention).
-/
import QuantumDLBlocksLean.Core

namespace QuantumDLBlocksLean

/-- Matrix multiplication on `Mat2` is associative: `(A * B) * C = A * (B * C)`. -/
theorem mat2_mul_assoc (A B C : Mat2) :
    Mat2.mul (Mat2.mul A B) C = Mat2.mul A (Mat2.mul B C) := by
  cases A
  cases B
  cases C
  ext <;>
    simp [Mat2.mul, Int.add_mul, Int.mul_add, ← Int.mul_assoc] <;>
    omega

/-- Transpose of a matrix product reverses order: `(A * B)^T = B^T * A^T`. -/
theorem mat2_transpose_mul (A B : Mat2) :
    Mat2.transpose (Mat2.mul A B) = Mat2.mul (Mat2.transpose B) (Mat2.transpose A) := by
  cases A
  cases B
  ext <;> simp [Mat2.transpose, Mat2.mul] <;> ac_nf

/-- Multiplication by the 2×2 identity matrix on the left: `I * A = A`. -/
theorem mat2_eye_mul (A : Mat2) : Mat2.mul Mat2.eye A = A := by
  cases A
  ext <;> simp [Mat2.mul, Mat2.eye]

/-- Outer-product density matrix `ρ(F) = F F^T` from a Cholesky/Purification factor `F`. -/
def outerDensity2 (F : Mat2) : Mat2 :=
  Mat2.mul F (Mat2.transpose F)

/-- Outer-product density matrix `ρ(F) = F F^T` is always symmetric/Hermitian. -/
theorem outerDensity2_transpose (F : Mat2) :
    Mat2.transpose (outerDensity2 F) = outerDensity2 F := by
  unfold outerDensity2
  rw [mat2_transpose_mul, Mat2.transpose_transpose]

/-- Every integer square is non-negative: `0 ≤ x * x`. -/
theorem int_sq_nonneg (x : Int) : 0 ≤ x * x := by
  by_cases h : 0 ≤ x
  · exact Int.mul_nonneg h h
  · have hneg : 0 ≤ -x := by omega
    have hprod : 0 ≤ (-x) * (-x) := Int.mul_nonneg hneg hneg
    simp only [Int.neg_mul, Int.mul_neg, Int.neg_neg] at hprod
    exact hprod

/-- Frobenius squared norm of any 2×2 matrix is non-negative: `0 ≤ ‖A‖_F^2`. -/
theorem mat2_frobNormSq_nonneg (A : Mat2) : 0 ≤ Mat2.frobNormSq A := by
  unfold Mat2.frobNormSq Mat2.frobInner
  have h00 := int_sq_nonneg A.m00
  have h01 := int_sq_nonneg A.m01
  have h10 := int_sq_nonneg A.m10
  have h11 := int_sq_nonneg A.m11
  omega

/-- Algebraic cyclic trace / Frobenius identity:
`Tr(ρ_Q ρ_K) = ⟨ρ_Q, ρ_K⟩_F = ‖Q^T K‖_F^2` for `ρ_Q = Q Q^T` and `ρ_K = K K^T`. -/
theorem hilbert_schmidt_trace_eq_gram_frobNormSq (Q K : Mat2) :
    Mat2.tr (Mat2.mul (outerDensity2 Q) (outerDensity2 K)) =
      Mat2.frobNormSq (Mat2.mul (Mat2.transpose Q) K) ∧
    Mat2.frobInner (outerDensity2 Q) (outerDensity2 K) =
      Mat2.frobNormSq (Mat2.mul (Mat2.transpose Q) K) := by
  cases Q with
  | mk q00 q01 q10 q11 =>
    cases K with
    | mk k00 k01 k10 k11 =>
      unfold outerDensity2 Mat2.tr Mat2.frobNormSq Mat2.frobInner Mat2.mul Mat2.transpose
      constructor <;>
        simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc] <;>
        ac_nf

/-- **Theorem 6(a) — Strict Non-Negativity of Hilbert–Schmidt Born Similarity & Exact Vacuum-Floor
Normalization (`theorem6a_hilbert_schmidt_born_nonneg_and_vacuum_norm`)**:
1. For any Cholesky/Purification factors `Q, K`, the Hilbert–Schmidt Born overlap satisfies
   `Tr(ρ_Q ρ_K) = ‖Q^T K‖_F^2 ≥ 0` without any `exp`/`softmax`,
2. For any sequence of non-negative Hilbert–Schmidt similarities `S : Fin n → Int` (`∀ j, 0 ≤ S j`)
   and positive vacuum rest-mass floor `ε > 0`, the denominator `D = (∑_j S j) + ε` is strictly
   positive (`0 < D`) and the sum of numerators plus `ε` equals `D` identically. -/
theorem theorem6a_hilbert_schmidt_born_nonneg_and_vacuum_norm
    (Q K : Mat2) (n : Nat) (S : Fin n → Int) (hS : ∀ j : Fin n, 0 ≤ S j)
    (ε : Int) (hε : 0 < ε) :
    Mat2.tr (Mat2.mul (outerDensity2 Q) (outerDensity2 K)) =
      Mat2.frobNormSq (Mat2.mul (Mat2.transpose Q) K) ∧
    0 ≤ Mat2.tr (Mat2.mul (outerDensity2 Q) (outerDensity2 K)) ∧
    0 ≤ finSum n S ∧
    0 < finSum n S + ε ∧
    (finSum n S) + ε = (finSum n S + ε) := by
  have ⟨h_tr_eq, _⟩ := hilbert_schmidt_trace_eq_gram_frobNormSq Q K
  have h_frob_nonneg := mat2_frobNormSq_nonneg (Mat2.mul (Mat2.transpose Q) K)
  have h_sum_nonneg := finSum_nonneg n S hS
  refine ⟨h_tr_eq, ?_, h_sum_nonneg, ?_, rfl⟩
  · rw [h_tr_eq]
    exact h_frob_nonneg
  · omega

/-- **Theorem 6(b) — Isometric Outer-Product Flattening Identity
(`theorem6b_isometric_outer_product_flattening`)**:
For `ρ_Q = Q Q^T` and `ρ_K = K K^T`:
1. Both `ρ_Q` and `ρ_K` are symmetric (`ρ_Q.m10 = ρ_Q.m01`, `ρ_K.m10 = ρ_K.m01`),
2. The packed symmetric inner product `(ρ_Q)_{00}(ρ_K)_{00} + (ρ_Q)_{11}(ρ_K)_{11} + 2(ρ_Q)_{01}(ρ_K)_{01}`
   equals the full Frobenius inner product `⟨ρ_Q, ρ_K⟩_F`, the matrix trace `Tr(ρ_Q ρ_K)`,
   and the factor Gram squared Frobenius norm `‖Q^T K‖_F^2`. -/
theorem theorem6b_isometric_outer_product_flattening (Q K : Mat2) :
    let ρQ := outerDensity2 Q
    let ρK := outerDensity2 K
    ρQ.m00 * ρK.m00 + ρQ.m11 * ρK.m11 + 2 * (ρQ.m01 * ρK.m01) =
      Mat2.frobInner ρQ ρK ∧
    Mat2.frobInner ρQ ρK = Mat2.tr (Mat2.mul ρQ ρK) ∧
    Mat2.tr (Mat2.mul ρQ ρK) = Mat2.frobNormSq (Mat2.mul (Mat2.transpose Q) K) := by
  dsimp only
  have ⟨h_tr, h_frob⟩ := hilbert_schmidt_trace_eq_gram_frobNormSq Q K
  have h_sym_Q : (outerDensity2 Q).m10 = (outerDensity2 Q).m01 :=
    congrArg Mat2.m01 (outerDensity2_transpose Q)
  have h_sym_K : (outerDensity2 K).m10 = (outerDensity2 K).m01 :=
    congrArg Mat2.m01 (outerDensity2_transpose K)
  refine ⟨?_, ?_, h_tr⟩
  · unfold Mat2.frobInner
    rw [h_sym_Q, h_sym_K]
    omega
  · rw [h_frob, h_tr]

/-- **Theorem 6(c) — Exact Right `U(R)`-Gauge Invariance of Outer-Product Density Matrices &
Hilbert–Schmidt Attention (`theorem6c_right_unitary_gauge_invariance`)**:
If `U_Q U_Q^T = I` and `U_K U_K^T = I`, then right-multiplying `Q ↦ Q U_Q` and `K ↦ K U_K`
leaves `ρ_Q = Q Q^T`, `ρ_K = K K^T`, `Tr(ρ_Q ρ_K)`, and `‖Q^T K‖_F^2` identically unchanged. -/
theorem theorem6c_right_unitary_gauge_invariance
    (Q K U_Q U_K : Mat2)
    (hUQ : Mat2.mul U_Q (Mat2.transpose U_Q) = Mat2.eye)
    (hUK : Mat2.mul U_K (Mat2.transpose U_K) = Mat2.eye) :
    outerDensity2 (Mat2.mul Q U_Q) = outerDensity2 Q ∧
    outerDensity2 (Mat2.mul K U_K) = outerDensity2 K ∧
    Mat2.tr (Mat2.mul (outerDensity2 (Mat2.mul Q U_Q)) (outerDensity2 (Mat2.mul K U_K))) =
      Mat2.tr (Mat2.mul (outerDensity2 Q) (outerDensity2 K)) ∧
    Mat2.frobNormSq (Mat2.mul (Mat2.transpose (Mat2.mul Q U_Q)) (Mat2.mul K U_K)) =
      Mat2.frobNormSq (Mat2.mul (Mat2.transpose Q) K) := by
  have h_rhoQ : outerDensity2 (Mat2.mul Q U_Q) = outerDensity2 Q := by
    unfold outerDensity2
    rw [mat2_transpose_mul, mat2_mul_assoc, ← mat2_mul_assoc U_Q (Mat2.transpose U_Q) (Mat2.transpose Q),
        hUQ, mat2_eye_mul]
  have h_rhoK : outerDensity2 (Mat2.mul K U_K) = outerDensity2 K := by
    unfold outerDensity2
    rw [mat2_transpose_mul, mat2_mul_assoc, ← mat2_mul_assoc U_K (Mat2.transpose U_K) (Mat2.transpose K),
        hUK, mat2_eye_mul]
  have h_tr_inv :
      Mat2.tr (Mat2.mul (outerDensity2 (Mat2.mul Q U_Q)) (outerDensity2 (Mat2.mul K U_K))) =
        Mat2.tr (Mat2.mul (outerDensity2 Q) (outerDensity2 K)) := by
    rw [h_rhoQ, h_rhoK]
  have ⟨h_tr_gauged, _⟩ := hilbert_schmidt_trace_eq_gram_frobNormSq (Mat2.mul Q U_Q) (Mat2.mul K U_K)
  have ⟨h_tr_orig, _⟩ := hilbert_schmidt_trace_eq_gram_frobNormSq Q K
  refine ⟨h_rhoQ, h_rhoK, h_tr_inv, ?_⟩
  rw [← h_tr_gauged, h_tr_inv, h_tr_orig]

#print axioms theorem6a_hilbert_schmidt_born_nonneg_and_vacuum_norm
#print axioms theorem6b_isometric_outer_product_flattening
#print axioms theorem6c_right_unitary_gauge_invariance

end QuantumDLBlocksLean
