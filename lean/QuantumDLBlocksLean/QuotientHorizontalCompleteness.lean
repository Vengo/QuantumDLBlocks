/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 1: Theorem 1 — Completeness of the Quotient Horizontal Projection & False-Symmetry Detection
(Waves 66–68: `project_su_dh_mub_horizontal`, `project_kraus_branch_gauge_horizontal`,
and the Wave 66 `W66_G3` false-symmetry diagnostic).
-/
import QuantumDLBlocksLean.Core

namespace QuantumDLBlocksLean

/-- An inner-product tangent space `T` equipped with an orthogonal projection `projV`
onto a candidate vertical gauge subspace `V ⊆ T` and horizontal complement
`projH v = subT v (projV v)`. -/
structure OrthogonalTangentProjection (T : Type) where
  zeroT : T
  addT : T → T → T
  subT : T → T → T
  inner : T → T → Int
  projV : T → T
  projV_zero : projV zeroT = zeroT
  projV_idem : ∀ v : T, projV (projV v) = projV v
  projV_symm : ∀ u v : T, inner (projV u) v = inner u (projV v)
  subT_zero : ∀ v : T, subT v zeroT = v
  subT_self : ∀ v : T, subT v v = zeroT
  add_sub_cancel : ∀ u v : T, addT (subT u v) v = u
  inner_zero_left : ∀ v : T, inner zeroT v = 0
  inner_sub_left : ∀ u v w : T, inner (subT u v) w = inner u w - inner v w
  inner_sub_right : ∀ u v w : T, inner u (subT v w) = inner u v - inner u w
  inner_nonneg : ∀ v : T, 0 ≤ inner v v
  inner_eq_zero_iff : ∀ v : T, inner v v = 0 ↔ v = zeroT

namespace OrthogonalTangentProjection

variable {T : Type} (P : OrthogonalTangentProjection T)

/-- Horizontal projection `projH v = v - projV v`. -/
def projH (v : T) : T :=
  P.subT v (P.projV v)

/-- Squared norm `‖v‖^2 = ⟨v, v⟩`. -/
def normSq (v : T) : Int :=
  P.inner v v

/-- Orthogonal decomposition `projH v + projV v = v`. -/
theorem projH_add_projV (v : T) : P.addT (P.projH v) (P.projV v) = v :=
  P.add_sub_cancel v (P.projV v)

/-- Horizontal vectors have zero vertical component: `⟨projH u, projV v⟩ = 0`. -/
theorem inner_projH_projV_eq_zero (u v : T) :
    P.inner (P.projH u) (P.projV v) = 0 := by
  unfold projH
  rw [P.inner_sub_left]
  have h1 : P.inner (P.projV u) (P.projV v) = P.inner u (P.projV (P.projV v)) :=
    P.projV_symm u (P.projV v)
  rw [P.projV_idem v] at h1
  omega

/-- Pythagorean identity: `‖v‖^2 = ‖projH v‖^2 + ‖projV v‖^2`. -/
theorem pythagoras_normSq (v : T) :
    P.normSq v = P.normSq (P.projH v) + P.normSq (P.projV v) := by
  unfold normSq projH
  rw [P.inner_sub_left, P.inner_sub_right, P.inner_sub_right]
  have hsym1 : P.inner v (P.projV v) = P.inner (P.projV v) (P.projV v) := by
    have h := P.projV_symm v (P.projV v)
    rw [P.projV_idem v] at h
    exact h.symm
  have hsym2 : P.inner (P.projV v) v = P.inner (P.projV v) (P.projV v) := by
    rw [P.projV_symm v v, hsym1]
  omega

end OrthogonalTangentProjection

/-- Forward differential `df : T → W` into an output inner-product space `W`. -/
structure ForwardDifferentialMap (T W : Type) (P : OrthogonalTangentProjection T) where
  zeroW : W
  subW : W → W → W
  innerW : W → W → Int
  df : T → W
  df_zero : df P.zeroT = zeroW
  df_sub : ∀ u v : T, df (P.subT u v) = subW (df u) (df v)
  subW_zero : ∀ w : W, subW w zeroW = w
  innerW_zero_right : ∀ w : W, innerW w zeroW = 0
  innerW_pos_of_ne_zero : ∀ w : W, w ≠ zeroW → 0 < innerW w w

/-- **Theorem 1(a) — Chain-Rule Horizontal Completeness (`theorem1a_chain_rule_horizontal_completeness`)**:
If `f` is gauge-invariant along `V` (`∀ v, df (projV v) = 0`), then:
1. `df (projH v) = df v` for all `v : T` (horizontal tangent preserves 100% of forward differential),
2. For any loss gradient `g : T` representing a pullback `∀ u, ⟨g, u⟩ = L_pullback (df u)` with `L_pullback 0 = 0`,
   we have `projV g = 0`, `projH g = g`, and `‖projH g‖^2 = ‖g‖^2` (100% horizontal gradient retention). -/
theorem theorem1a_chain_rule_horizontal_completeness
    {T W : Type} (P : OrthogonalTangentProjection T) (F : ForwardDifferentialMap T W P)
    (h_gauge_inv : ∀ v : T, F.df (P.projV v) = F.zeroW)
    (lossPullback : W → Int) (h_loss_zero : lossPullback F.zeroW = 0)
    (g : T) (h_grad : ∀ u : T, P.inner g u = lossPullback (F.df u)) :
    (∀ v : T, F.df (P.projH v) = F.df v) ∧
    P.projV g = P.zeroT ∧
    P.projH g = g ∧
    P.normSq (P.projH g) = P.normSq g := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro v
    unfold OrthogonalTangentProjection.projH
    rw [F.df_sub, h_gauge_inv v, F.subW_zero]
  · have h_inner_vg : P.inner g (P.projV g) = 0 := by
      rw [h_grad (P.projV g), h_gauge_inv g, h_loss_zero]
    have h_norm_vg : P.inner (P.projV g) (P.projV g) = 0 := by
      rw [P.projV_symm g (P.projV g), P.projV_idem g, h_inner_vg]
    exact (P.inner_eq_zero_iff (P.projV g)).mp h_norm_vg
  · have h_inner_vg : P.inner g (P.projV g) = 0 := by
      rw [h_grad (P.projV g), h_gauge_inv g, h_loss_zero]
    have h_norm_vg : P.inner (P.projV g) (P.projV g) = 0 := by
      rw [P.projV_symm g (P.projV g), P.projV_idem g, h_inner_vg]
    have h_vg_zero : P.projV g = P.zeroT :=
      (P.inner_eq_zero_iff (P.projV g)).mp h_norm_vg
    unfold OrthogonalTangentProjection.projH
    rw [h_vg_zero, P.subT_zero]
  · have h_inner_vg : P.inner g (P.projV g) = 0 := by
      rw [h_grad (P.projV g), h_gauge_inv g, h_loss_zero]
    have h_norm_vg : P.inner (P.projV g) (P.projV g) = 0 := by
      rw [P.projV_symm g (P.projV g), P.projV_idem g, h_inner_vg]
    have h_vg_zero : P.projV g = P.zeroT :=
      (P.inner_eq_zero_iff (P.projV g)).mp h_norm_vg
    have h_projH_eq : P.projH g = g := by
      unfold OrthogonalTangentProjection.projH
      rw [h_vg_zero, P.subT_zero]
    rw [h_projH_eq]

/-- **Theorem 1(b) — Converse / Wave 66 `W66_G3` False-Symmetry Theorem
(`theorem1b_false_symmetry_vertical_leak_and_norm_loss`)**:
If a proposed projection `projV'` is NOT a true gauge symmetry—i.e., there exists `v₀ : T`
such that `df (projV' v₀) ≠ 0`—then for the linear loss functional
`L(w) = ⟨df(projV' v₀), w⟩_W`, its pullback gradient `g` (`∀ u, ⟨g, u⟩_T = L(df u)`)
has strictly non-zero vertical leak (`projV' g ≠ 0`, `‖projV' g‖^2 > 0`) and strictly
loses norm under `projH'` (`‖projH' g‖^2 < ‖g‖^2`). -/
theorem theorem1b_false_symmetry_vertical_leak_and_norm_loss
    {T W : Type} (P : OrthogonalTangentProjection T) (F : ForwardDifferentialMap T W P)
    (v₀ : T) (h_false_sym : F.df (P.projV v₀) ≠ F.zeroW)
    (g : T)
    (h_grad : ∀ u : T, P.inner g u = F.innerW (F.df (P.projV v₀)) (F.df u)) :
    P.projV g ≠ P.zeroT ∧
    0 < P.normSq (P.projV g) ∧
    P.normSq (P.projH g) < P.normSq g := by
  have h_pos_w : 0 < F.innerW (F.df (P.projV v₀)) (F.df (P.projV v₀)) :=
    F.innerW_pos_of_ne_zero (F.df (P.projV v₀)) h_false_sym
  have h_pair : P.inner (P.projV g) v₀ = F.innerW (F.df (P.projV v₀)) (F.df (P.projV v₀)) := by
    rw [P.projV_symm g v₀, h_grad (P.projV v₀)]
  have h_vg_ne_zero : P.projV g ≠ P.zeroT := by
    intro h_zero
    rw [h_zero, P.inner_zero_left v₀] at h_pair
    omega
  have h_vg_norm_nonneg : 0 ≤ P.normSq (P.projV g) := P.inner_nonneg (P.projV g)
  have h_vg_norm_ne_zero : P.normSq (P.projV g) ≠ 0 := by
    intro h_norm_zero
    exact h_vg_ne_zero ((P.inner_eq_zero_iff (P.projV g)).mp h_norm_zero)
  have h_vg_norm_pos : 0 < P.normSq (P.projV g) := by omega
  have h_pyth : P.normSq g = P.normSq (P.projH g) + P.normSq (P.projV g) :=
    P.pythagoras_normSq g
  refine ⟨h_vg_ne_zero, h_vg_norm_pos, ?_⟩
  omega

#print axioms theorem1a_chain_rule_horizontal_completeness
#print axioms theorem1b_false_symmetry_vertical_leak_and_norm_loss

end QuantumDLBlocksLean
