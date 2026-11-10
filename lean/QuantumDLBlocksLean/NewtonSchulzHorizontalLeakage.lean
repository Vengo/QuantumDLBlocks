/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 4: Theorem 4 — Non-Invariance of Horizontal Subspaces Under Odd-Degree Newton–Schulz Polynomials
(Waves 66–68: `filter_coupled_svd_post_ns_corotation`, `project_kraus_branch_gauge_horizontal`,
and `TangentStiefelMuon` post-Newton–Schulz horizontal re-projection).
-/
import QuantumDLBlocksLean.Core
import QuantumDLBlocksLean.QuotientHorizontalCompleteness

namespace QuantumDLBlocksLean

/-- Gram matrix `G = Ξ^T Ξ` of a 2×2 tangent matrix `Ξ`. -/
def gramMat2 (Ξ : Mat2) : Mat2 :=
  Mat2.mul (Mat2.transpose Ξ) Ξ

/-- Odd-degree (cubic/quintic) Newton–Schulz polynomial map
`Φ_{a,b,c}(Ξ) = a Ξ + b Ξ G + c Ξ G^2` where `G = Ξ^T Ξ`. -/
def newtonSchulzPoly (a b c : Int) (Ξ : Mat2) : Mat2 :=
  let G := gramMat2 Ξ
  let G2 := Mat2.mul G G
  Mat2.add
    (Mat2.add (Mat2.smul a Ξ) (Mat2.smul b (Mat2.mul Ξ G)))
    (Mat2.smul c (Mat2.mul Ξ G2))

/-- Vertical gauge skew-moment `Π_V(Ξ) = Ξ K^T - K Ξ^T = 2 * skew(Ξ K^T)` relative to `K`. -/
def verticalMoment (K Ξ : Mat2) : Mat2 :=
  Mat2.skew2 (Mat2.mul Ξ (Mat2.transpose K))

/-- Explicit 2×2 base operator `K = diag(1, 2)` representing non-isotropic branch/singular weights. -/
def witnessK : Mat2 :=
  ⟨1, 0, 0, 2⟩

/-- Explicit 2×2 horizontal tangent matrix `Ξ = [[1, 1], [2, 1]]` satisfying `Ξ K^T = [[1, 2], [2, 2]] = K Ξ^T`. -/
def witnessXi : Mat2 :=
  ⟨1, 1, 2, 1⟩

/-- Exact formula for the vertical gauge moment of `Φ_{a,b,c}(witnessXi)`:
`verticalMoment witnessK (newtonSchulzPoly a b c witnessXi) = [[0, -3b - 21c], [3b + 21c, 0]]`. -/
theorem witness_newtonSchulz_verticalMoment_formula (a b c : Int) :
    verticalMoment witnessK (newtonSchulzPoly a b c witnessXi) =
      ⟨0, -3 * b - 21 * c, 3 * b + 21 * c, 0⟩ := by
  ext <;>
    simp [verticalMoment, newtonSchulzPoly, gramMat2, witnessK, witnessXi,
      Mat2.skew2, Mat2.mul, Mat2.add, Mat2.sub, Mat2.smul, Mat2.transpose] <;>
    omega

/-- **Theorem 4(a) — Concrete Witness of Horizontal Subspace Non-Invariance Under Odd-Degree
Newton–Schulz Polynomials (`theorem4a_newton_schulz_horizontal_leakage`)**:
For the explicit 2×2 witness `(witnessK, witnessXi)`:
1. `witnessXi` is strictly horizontal: `verticalMoment witnessK witnessXi = 0`,
2. Its Gram matrix `G = witnessXi^T witnessXi` does not commute with `witnessK^T witnessK`,
3. The cubic term `witnessXi * G` leaks into the vertical subspace (`verticalMoment ≠ 0`),
4. Both standard cubic `(3, -1, 0)` and quintic `(15, -10, 3)` Newton–Schulz polynomials—and
   every polynomial with `b + 7c ≠ 0`—produce a strictly non-zero vertical gauge leak
   `verticalMoment witnessK (newtonSchulzPoly a b c witnessXi) ≠ 0`. -/
theorem theorem4a_newton_schulz_horizontal_leakage :
    verticalMoment witnessK witnessXi = Mat2.zero ∧
    Mat2.commutator (gramMat2 witnessXi) (Mat2.mul (Mat2.transpose witnessK) witnessK) ≠ Mat2.zero ∧
    verticalMoment witnessK (Mat2.mul witnessXi (gramMat2 witnessXi)) ≠ Mat2.zero ∧
    verticalMoment witnessK (newtonSchulzPoly 3 (-1) 0 witnessXi) ≠ Mat2.zero ∧
    verticalMoment witnessK (newtonSchulzPoly 15 (-10) 3 witnessXi) ≠ Mat2.zero ∧
    (∀ a b c : Int, b + 7 * c ≠ 0 →
      verticalMoment witnessK (newtonSchulzPoly a b c witnessXi) ≠ Mat2.zero) := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, ?_⟩
  intro a b c h_nondeg h_zero
  rw [witness_newtonSchulz_verticalMoment_formula a b c] at h_zero
  have h10 : 3 * b + 21 * c = 0 := congrArg Mat2.m10 h_zero
  apply h_nondeg
  omega

/-- **Theorem 4(b) — Post-Newton–Schulz Horizontal Re-Projection Restores Exact Horizontality
(`theorem4b_post_ns_horizontal_projection_restores_horizontality`)**:
For any orthogonal tangent projection `P : OrthogonalTangentProjection T` and any non-linear
retraction/polar map `Φ : T → T`, applying `P.projH` after `Φ(Ξ)` guarantees that the
vertical component vanishes identically: `P.projV (P.projH (Φ Ξ)) = P.zeroT`, and in the
Euclidean Stiefel `K = I_2` tangent split (`projH(X) = sym2(X)`, `projV(X) = skew2(X)`),
`Mat2.skew2 (Mat2.sym2 (newtonSchulzPoly a b c Ξ)) = Mat2.zero` for all `Ξ : Mat2`. -/
theorem theorem4b_post_ns_horizontal_projection_restores_horizontality
    {T : Type} (P : OrthogonalTangentProjection T) (Φ : T → T) (Ξ : T)
    (a b c : Int) (M : Mat2) :
    P.projV (P.projH (Φ Ξ)) = P.zeroT ∧
    Mat2.skew2 (Mat2.sym2 (newtonSchulzPoly a b c M)) = Mat2.zero := by
  constructor
  · have h_ortho := P.inner_projH_projV_eq_zero (Φ Ξ) (P.projH (Φ Ξ))
    have h_symm := P.projV_symm (P.projH (Φ Ξ)) (P.projV (P.projH (Φ Ξ)))
    rw [P.projV_idem (P.projH (Φ Ξ))] at h_symm
    have h_inner_zero : P.inner (P.projV (P.projH (Φ Ξ))) (P.projV (P.projH (Φ Ξ))) = 0 := by
      rw [h_symm]
      exact h_ortho
    exact (P.inner_eq_zero_iff (P.projV (P.projH (Φ Ξ)))).mp h_inner_zero
  · exact Mat2.skew2_sym2_eq_zero (newtonSchulzPoly a b c M)

#print axioms theorem4a_newton_schulz_horizontal_leakage
#print axioms theorem4b_post_ns_horizontal_projection_restores_horizontality

end QuantumDLBlocksLean
