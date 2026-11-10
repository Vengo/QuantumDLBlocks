/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 7: Theorem 7 — Quantum "OR + AND" Density-Matrix Closure via Convex Superposition
and the Schur Product Theorem (Wave 69 `EXPERIMENT_PLAN_089.md` & `born_schur_density_lab.py`).
-/
import QuantumDLBlocksLean.Core
import QuantumDLBlocksLean.HilbertSchmidtQuantumAttention

namespace QuantumDLBlocksLean

/-- Quadratic form `⟨ψ | M | ψ⟩ = ψ_0 (M_{00} ψ_0 + M_{01} ψ_1) + ψ_1 (M_{10} ψ_0 + M_{11} ψ_1)`. -/
def quadForm2 (M : Mat2) (ψ0 ψ1 : Int) : Int :=
  ψ0 * (M.m00 * ψ0 + M.m01 * ψ1) + ψ1 * (M.m10 * ψ0 + M.m11 * ψ1)

/-- Quantum "OR" (weighted convex superposition) `w_A • ρ_A + w_B • ρ_B`. -/
def quantumOr2 (wA wB : Int) (ρA ρB : Mat2) : Mat2 :=
  Mat2.add (Mat2.smul wA ρA) (Mat2.smul wB ρB)

/-- Quantum "AND" (elementwise Schur / Hadamard coincidence product) `(ρ_A ∘ ρ_B)_{ij} = (ρ_A)_{ij} (ρ_B)_{ij}`. -/
def schurMul2 (A B : Mat2) : Mat2 :=
  ⟨A.m00 * B.m00, A.m01 * B.m01, A.m10 * B.m10, A.m11 * B.m11⟩

/-- Every factored outer-product density matrix `ρ = L L^T` has a sum-of-squares quadratic form:
`⟨ψ | L L^T | ψ⟩ = (L_{00} ψ_0 + L_{10} ψ_1)^2 + (L_{01} ψ_0 + L_{11} ψ_1)^2 ≥ 0`. -/
theorem outerDensity2_quadForm_eq_sum_sq (L : Mat2) (ψ0 ψ1 : Int) :
    quadForm2 (outerDensity2 L) ψ0 ψ1 =
      (L.m00 * ψ0 + L.m10 * ψ1) * (L.m00 * ψ0 + L.m10 * ψ1) +
      (L.m01 * ψ0 + L.m11 * ψ1) * (L.m01 * ψ0 + L.m11 * ψ1) ∧
    0 ≤ quadForm2 (outerDensity2 L) ψ0 ψ1 := by
  have h_eq :
      quadForm2 (outerDensity2 L) ψ0 ψ1 =
        (L.m00 * ψ0 + L.m10 * ψ1) * (L.m00 * ψ0 + L.m10 * ψ1) +
        (L.m01 * ψ0 + L.m11 * ψ1) * (L.m01 * ψ0 + L.m11 * ψ1) := by
    cases L
    unfold quadForm2 outerDensity2 Mat2.mul Mat2.transpose
    simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc]
    ac_nf
  refine ⟨h_eq, ?_⟩
  rw [h_eq]
  have h1 := int_sq_nonneg (L.m00 * ψ0 + L.m10 * ψ1)
  have h2 := int_sq_nonneg (L.m01 * ψ0 + L.m11 * ψ1)
  omega

/-- Every factored outer-product density matrix `ρ = L L^T` has non-negative diagonal entries and trace. -/
theorem outerDensity2_diag_and_tr_nonneg (L : Mat2) :
    0 ≤ (outerDensity2 L).m00 ∧
    0 ≤ (outerDensity2 L).m11 ∧
    0 ≤ Mat2.tr (outerDensity2 L) := by
  simp only [outerDensity2, Mat2.tr, Mat2.mul, Mat2.transpose]
  have h00 := int_sq_nonneg L.m00
  have h01 := int_sq_nonneg L.m01
  have h10 := int_sq_nonneg L.m10
  have h11 := int_sq_nonneg L.m11
  refine ⟨by omega, by omega, by omega⟩

/-- **Theorem 7(a) — Quantum "OR" Additive / Convex Superposition Closure
(`theorem7a_quantum_or_convex_superposition_closure`)**:
For any factored PSD Hermitian density matrices `ρ_A = L_A L_A^T` and `ρ_B = L_B L_B^T`
and non-negative weights `w_A ≥ 0, w_B ≥ 0`, `ρ_OR = w_A ρ_A + w_B ρ_B` is:
1. Symmetric/Hermitian: `ρ_OR^T = ρ_OR`,
2. Positive semidefinite: `∀ ψ_0 ψ_1, ⟨ψ | ρ_OR | ψ⟩ = w_A ⟨ψ | ρ_A | ψ⟩ + w_B ⟨ψ | ρ_B | ψ⟩ ≥ 0`,
3. Trace-linear and non-negative: `Tr(ρ_OR) = w_A Tr(ρ_A) + w_B Tr(ρ_B) ≥ 0`. -/
theorem theorem7a_quantum_or_convex_superposition_closure
    (LA LB : Mat2) (wA wB : Int) (hwA : 0 ≤ wA) (hwB : 0 ≤ wB) :
    let ρA := outerDensity2 LA
    let ρB := outerDensity2 LB
    let ρOR := quantumOr2 wA wB ρA ρB
    Mat2.transpose ρOR = ρOR ∧
    (∀ ψ0 ψ1 : Int,
      quadForm2 ρOR ψ0 ψ1 = wA * quadForm2 ρA ψ0 ψ1 + wB * quadForm2 ρB ψ0 ψ1 ∧
      0 ≤ quadForm2 ρOR ψ0 ψ1) ∧
    Mat2.tr ρOR = wA * Mat2.tr ρA + wB * Mat2.tr ρB ∧
    0 ≤ Mat2.tr ρOR := by
  dsimp only
  have h_symA := outerDensity2_transpose LA
  have h_symB := outerDensity2_transpose LB
  have ⟨_, _, h_trA_nonneg⟩ := outerDensity2_diag_and_tr_nonneg LA
  have ⟨_, _, h_trB_nonneg⟩ := outerDensity2_diag_and_tr_nonneg LB
  refine ⟨?_, ?_, ?_, ?_⟩
  · have hA10 : (outerDensity2 LA).m10 = (outerDensity2 LA).m01 := congrArg Mat2.m01 h_symA
    have hB10 : (outerDensity2 LB).m10 = (outerDensity2 LB).m01 := congrArg Mat2.m01 h_symB
    unfold quantumOr2 Mat2.transpose Mat2.add Mat2.smul
    ext <;> simp [hA10, hB10]
  · intro ψ0 ψ1
    have h_q_lin :
        quadForm2 (quantumOr2 wA wB (outerDensity2 LA) (outerDensity2 LB)) ψ0 ψ1 =
          wA * quadForm2 (outerDensity2 LA) ψ0 ψ1 + wB * quadForm2 (outerDensity2 LB) ψ0 ψ1 := by
      unfold quantumOr2 quadForm2 Mat2.add Mat2.smul
      simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc, Int.mul_comm ψ0 wA, Int.mul_comm ψ0 wB,
        Int.mul_comm ψ1 wA, Int.mul_comm ψ1 wB]
      ac_nf
    have ⟨_, h_qA⟩ := outerDensity2_quadForm_eq_sum_sq LA ψ0 ψ1
    have ⟨_, h_qB⟩ := outerDensity2_quadForm_eq_sum_sq LB ψ0 ψ1
    have h_wAqA := Int.mul_nonneg hwA h_qA
    have h_wBqB := Int.mul_nonneg hwB h_qB
    refine ⟨h_q_lin, ?_⟩
    omega
  · simp only [quantumOr2, Mat2.tr, Mat2.add, Mat2.smul, Int.mul_add]
    omega
  · have h_tr_lin :
        Mat2.tr (quantumOr2 wA wB (outerDensity2 LA) (outerDensity2 LB)) =
          wA * Mat2.tr (outerDensity2 LA) + wB * Mat2.tr (outerDensity2 LB) := by
      simp only [quantumOr2, Mat2.tr, Mat2.add, Mat2.smul, Int.mul_add]
      omega
    have h_wAtrA := Int.mul_nonneg hwA h_trA_nonneg
    have h_wBtrB := Int.mul_nonneg hwB h_trB_nonneg
    omega

/-- General Rank-`R_A × R_B` Khatri–Rao (row-wise Kronecker) entry identity for the Schur product:
`((∑_r L_{A,ir} L_{A,jr}) * (∑_s L_{B,is} L_{B,js})) = ∑_r ∑_s (L_{A,ir} L_{B,is}) * (L_{A,jr} L_{B,js})`. -/
theorem schur_khatri_rao_entry_factorization
    (RA RB : Nat)
    (LA_i LA_j : Fin RA → Int)
    (LB_i LB_j : Fin RB → Int) :
    (finSum RA (fun r => LA_i r * LA_j r)) * (finSum RB (fun s => LB_i s * LB_j s)) =
      finSum RA (fun r => finSum RB (fun s => (LA_i r * LB_i s) * (LA_j r * LB_j s))) := by
  let S_B := finSum RB (fun s => LB_i s * LB_j s)
  rw [Int.mul_comm (finSum RA _) S_B, ← finSum_const_mul RA S_B]
  apply finSum_congr
  intro r
  rw [Int.mul_comm S_B (LA_i r * LA_j r), ← finSum_const_mul RB (LA_i r * LA_j r)]
  apply finSum_congr
  intro s
  simp only [← Int.mul_assoc]
  ac_nf

/-- **Theorem 7(b) — Quantum "AND" Schur / Hadamard Product Theorem
(`theorem7b_quantum_and_schur_product_theorem`)**:
For any factored PSD Hermitian density matrices `ρ_A = L_A L_A^T` and `ρ_B = L_B L_B^T`,
their elementwise Schur coincidence product `ρ_AND = ρ_A ∘ ρ_B` satisfies:
1. **Khatri–Rao Sum-of-Squares Factorization**: For every `ψ = (ψ_0, ψ_1)`,
   `⟨ψ | (ρ_A ∘ ρ_B) | ψ⟩ = ∑_{r,s ∈ {0,1}} ((L_A)_{0,r}(L_B)_{0,s} ψ_0 + (L_A)_{1,r}(L_B)_{1,s} ψ_1)^2`,
2. **Hermiticity**: `(ρ_A ∘ ρ_B)^T = ρ_A ∘ ρ_B`,
3. **Strict Positive Semidefiniteness**: `∀ ψ_0 ψ_1, 0 ≤ ⟨ψ | (ρ_A ∘ ρ_B) | ψ⟩`,
4. **Non-Negative Coincidence Trace**: `Tr(ρ_A ∘ ρ_B) = (ρ_A)_{00}(ρ_B)_{00} + (ρ_A)_{11}(ρ_B)_{11} ≥ 0`. -/
theorem theorem7b_quantum_and_schur_product_theorem (LA LB : Mat2) :
    let ρA := outerDensity2 LA
    let ρB := outerDensity2 LB
    let ρAND := schurMul2 ρA ρB
    (∀ ψ0 ψ1 : Int,
      quadForm2 ρAND ψ0 ψ1 =
        (LA.m00 * LB.m00 * ψ0 + LA.m10 * LB.m10 * ψ1) * (LA.m00 * LB.m00 * ψ0 + LA.m10 * LB.m10 * ψ1) +
        (LA.m00 * LB.m01 * ψ0 + LA.m10 * LB.m11 * ψ1) * (LA.m00 * LB.m01 * ψ0 + LA.m10 * LB.m11 * ψ1) +
        (LA.m01 * LB.m00 * ψ0 + LA.m11 * LB.m10 * ψ1) * (LA.m01 * LB.m00 * ψ0 + LA.m11 * LB.m10 * ψ1) +
        (LA.m01 * LB.m01 * ψ0 + LA.m11 * LB.m11 * ψ1) * (LA.m01 * LB.m01 * ψ0 + LA.m11 * LB.m11 * ψ1)) ∧
    Mat2.transpose ρAND = ρAND ∧
    (∀ ψ0 ψ1 : Int, 0 ≤ quadForm2 ρAND ψ0 ψ1) ∧
    Mat2.tr ρAND = ρA.m00 * ρB.m00 + ρA.m11 * ρB.m11 ∧
    0 ≤ Mat2.tr ρAND := by
  dsimp only
  have h_kr :
      ∀ ψ0 ψ1 : Int,
        quadForm2 (schurMul2 (outerDensity2 LA) (outerDensity2 LB)) ψ0 ψ1 =
          (LA.m00 * LB.m00 * ψ0 + LA.m10 * LB.m10 * ψ1) * (LA.m00 * LB.m00 * ψ0 + LA.m10 * LB.m10 * ψ1) +
          (LA.m00 * LB.m01 * ψ0 + LA.m10 * LB.m11 * ψ1) * (LA.m00 * LB.m01 * ψ0 + LA.m10 * LB.m11 * ψ1) +
          (LA.m01 * LB.m00 * ψ0 + LA.m11 * LB.m10 * ψ1) * (LA.m01 * LB.m00 * ψ0 + LA.m11 * LB.m10 * ψ1) +
          (LA.m01 * LB.m01 * ψ0 + LA.m11 * LB.m11 * ψ1) * (LA.m01 * LB.m01 * ψ0 + LA.m11 * LB.m11 * ψ1) := by
    intro ψ0 ψ1
    cases LA
    cases LB
    unfold quadForm2 schurMul2 outerDensity2 Mat2.mul Mat2.transpose
    simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc]
    ac_nf
  have h_symA : (outerDensity2 LA).m10 = (outerDensity2 LA).m01 :=
    congrArg Mat2.m01 (outerDensity2_transpose LA)
  have h_symB : (outerDensity2 LB).m10 = (outerDensity2 LB).m01 :=
    congrArg Mat2.m01 (outerDensity2_transpose LB)
  have ⟨hA00, hA11, _⟩ := outerDensity2_diag_and_tr_nonneg LA
  have ⟨hB00, hB11, _⟩ := outerDensity2_diag_and_tr_nonneg LB
  refine ⟨h_kr, ?_, ?_, rfl, ?_⟩
  · unfold schurMul2 Mat2.transpose
    ext <;> simp [h_symA, h_symB]
  · intro ψ0 ψ1
    rw [h_kr ψ0 ψ1]
    have s00 := int_sq_nonneg (LA.m00 * LB.m00 * ψ0 + LA.m10 * LB.m10 * ψ1)
    have s01 := int_sq_nonneg (LA.m00 * LB.m01 * ψ0 + LA.m10 * LB.m11 * ψ1)
    have s10 := int_sq_nonneg (LA.m01 * LB.m00 * ψ0 + LA.m11 * LB.m10 * ψ1)
    have s11 := int_sq_nonneg (LA.m01 * LB.m01 * ψ0 + LA.m11 * LB.m11 * ψ1)
    omega
  · simp only [schurMul2, Mat2.tr]
    have p00 := Int.mul_nonneg hA00 hB00
    have p11 := Int.mul_nonneg hA11 hB11
    omega

#print axioms theorem7a_quantum_or_convex_superposition_closure
#print axioms schur_khatri_rao_entry_factorization
#print axioms theorem7b_quantum_and_schur_product_theorem

end QuantumDLBlocksLean
