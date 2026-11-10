/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 3: Theorem 3 — CPTP Kraus `u(R)` Gauge Invariance & Sylvester Horizontal Condition
(Waves 66–68: `project_kraus_branch_gauge_horizontal` and `compute_kraus_branch_gauge_fraction`).
-/
import QuantumDLBlocksLean.Core

namespace QuantumDLBlocksLean

/-- Kronecker delta `δ_{a, b}` on `Fin R`. -/
def kroneckerDelta {R : Nat} (a b : Fin R) : Int :=
  if a = b then 1 else 0

/-- Finite sum of identically zero terms is zero. -/
theorem finSum_zero (n : Nat) : finSum n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ k ih => simp [finSum, ih]

/-- Subtraction distributes over `finSum`: `∑_i (f(i) - g(i)) = (∑_i f(i)) - (∑_i g(i))`. -/
theorem finSum_sub (n : Nat) (f g : Fin n → Int) :
    finSum n (fun i => f i - g i) = finSum n f - finSum n g := by
  induction n with
  | zero =>
    simp [finSum]
  | succ k ih =>
    simp only [finSum]
    rw [ih]
    omega

/-- Summing against Kronecker delta collapses the sum: `∑_{b < R} δ_{a, b} f(b) = f(a)`. -/
theorem finSum_kroneckerDelta_mul :
    ∀ (R : Nat) (a : Fin R) (f : Fin R → Int),
      finSum R (fun b => kroneckerDelta a b * f b) = f a
  | 0, a, _ => Fin.elim0 a
  | R + 1, a, f => by
    simp only [finSum]
    by_cases h_last : a = ⟨R, Nat.lt_succ_self R⟩
    · subst h_last
      have h_zero_init :
          finSum R (fun b : Fin R => kroneckerDelta ⟨R, Nat.lt_succ_self R⟩ b.castSucc * f b.castSucc) = 0 := by
        have h_congr :
            (fun b : Fin R => kroneckerDelta ⟨R, Nat.lt_succ_self R⟩ b.castSucc * f b.castSucc) =
              (fun _ : Fin R => 0) := by
          funext b
          unfold kroneckerDelta
          have h_ne : (⟨R, Nat.lt_succ_self R⟩ : Fin (R + 1)) ≠ b.castSucc := by
            intro h_eq
            have h_val : R = b.val := congrArg Fin.val h_eq
            omega
          simp [h_ne]
        rw [h_congr]
        exact finSum_zero R
      rw [h_zero_init]
      unfold kroneckerDelta
      simp
    · have h_lt : a.val < R := by
        have h_ne_val : a.val ≠ R := by
          intro h_eq
          apply h_last
          ext
          exact h_eq
        omega
      let a_init : Fin R := ⟨a.val, h_lt⟩
      have h_a_eq : a = a_init.castSucc := by
        ext
        rfl
      have h_ih := finSum_kroneckerDelta_mul R a_init (fun b => f b.castSucc)
      have h_init_eq :
          finSum R (fun i : Fin R => kroneckerDelta a i.castSucc * f i.castSucc) =
            finSum R (fun b : Fin R => kroneckerDelta a_init b * f b.castSucc) := by
        apply finSum_congr
        intro b
        unfold kroneckerDelta
        rw [h_a_eq]
        by_cases hb : a_init = b
        · subst hb
          simp
        · have hb_cast : a_init.castSucc ≠ b.castSucc := by
            intro hc
            apply hb
            ext
            exact congrArg (@Fin.val (R + 1)) hc
          simp [hb, hb_cast]
      rw [h_init_eq, h_ih]
      unfold kroneckerDelta
      simp [h_last, ← h_a_eq]

/-- **Theorem 3(a) — Unitary Gauge Invariance of CPTP Kraus Channel & Trace-Preservation Condition
(`theorem3a_kraus_unitary_channel_and_tp_invariance`)**:
For any branch count `R`, if the unitary column-orthogonality relation
`∑_{r < R} W_conj(r, a) * W(r, b) = δ_{a, b}` holds, then for both:
1. the CPTP channel cross-terms `B_channel(a, b) = ⟨u, K_a ρ K_b^† v⟩`, and
2. the Stiefel trace-preservation cross-terms `B_tp(a, b) = ⟨u, K_a^† K_b v⟩`,
the mixed double sum `∑_a ∑_b (∑_r W_conj(r, a) W(r, b)) B(a, b)` equals the unmixed
diagonal sum `∑_a B(a, a)`. -/
theorem theorem3a_kraus_unitary_channel_and_tp_invariance
    (R : Nat)
    (W W_conj : Fin R → Fin R → Int)
    (h_unitary : ∀ a b : Fin R, finSum R (fun r => W_conj r a * W r b) = kroneckerDelta a b)
    (B_channel B_tp : Fin R → Fin R → Int) :
    finSum R (fun a => finSum R (fun b => finSum R (fun r => W_conj r a * W r b) * B_channel a b)) =
      finSum R (fun a => B_channel a a) ∧
    finSum R (fun a => finSum R (fun b => finSum R (fun r => W_conj r a * W r b) * B_tp a b)) =
      finSum R (fun a => B_tp a a) := by
  constructor
  · apply finSum_congr
    intro a
    have h_inner :
        (fun b : Fin R => finSum R (fun r => W_conj r a * W r b) * B_channel a b) =
          (fun b : Fin R => kroneckerDelta a b * B_channel a b) := by
      funext b
      rw [h_unitary a b]
    rw [h_inner]
    exact finSum_kroneckerDelta_mul R a (fun b => B_channel a b)
  · apply finSum_congr
    intro a
    have h_inner :
        (fun b : Fin R => finSum R (fun r => W_conj r a * W r b) * B_tp a b) =
          (fun b : Fin R => kroneckerDelta a b * B_tp a b) := by
      funext b
      rw [h_unitary a b]
    rw [h_inner]
    exact finSum_kroneckerDelta_mul R a (fun b => B_tp a b)

/-- Explicit 2-branch (`R = 2`) unfactored bilinear expansion showing that transforming
`K'_r = ∑_s W_{rs} K_s` directly preserves the sum of branch quadratic forms. -/
theorem theorem3a_two_branch_direct_expansion
    (w00 w01 w10 w11 : Int)
    (h00 : w00 * w00 + w10 * w10 = 1)
    (h11 : w01 * w01 + w11 * w11 = 1)
    (h01 : w00 * w01 + w10 * w11 = 0)
    (B00 B01 B10 B11 : Int) :
    (w00 * w00 * B00 + w00 * w01 * B01 + w01 * w00 * B10 + w01 * w01 * B11) +
    (w10 * w10 * B00 + w10 * w11 * B01 + w11 * w10 * B10 + w11 * w11 * B11) =
      B00 + B11 := by
  have h10 : w01 * w00 + w11 * w10 = 0 := by
    rw [Int.mul_comm w01 w00, Int.mul_comm w11 w10]
    exact h01
  have h_collect :
      (w00 * w00 * B00 + w00 * w01 * B01 + w01 * w00 * B10 + w01 * w01 * B11) +
      (w10 * w10 * B00 + w10 * w11 * B01 + w11 * w10 * B10 + w11 * w11 * B11) =
        (w00 * w00 + w10 * w10) * B00 +
        (w00 * w01 + w10 * w11) * B01 +
        (w01 * w00 + w11 * w10) * B10 +
        (w01 * w01 + w11 * w11) * B11 := by
    rw [Int.add_mul, Int.add_mul, Int.add_mul, Int.add_mul]
    omega
  rw [h_collect, h00, h01, h10, h11]
  omega

/-- First-order infinitesimal `u(R)` gauge invariance: for skew-Hermitian `A`
(`A_conj s r = -A r s`), the symmetrized first-order channel perturbation vanishes termwise. -/
theorem kraus_infinitesimal_skew_hermitian_cancel
    {R : Nat} (A A_conj : Fin R → Fin R → Int)
    (h_skew : ∀ r s : Fin R, A_conj s r = -A r s) (r s : Fin R) (B_sr : Int) :
    A r s * B_sr + A_conj s r * B_sr = 0 := by
  rw [h_skew r s, Int.neg_mul]
  omega

/-- Horizontal skew-moment entry `S_horiz(r, t) = ⟨Ξ_horiz,r, K_t⟩ - ⟨K_r, Ξ_horiz,t⟩`
for `Ξ_horiz,r = Ξ_r - ∑_s Ω_{rs} K_s` when `Ω` is skew-Hermitian (`Ω_conj t s = -Ω s t`)
and `M_{rs} = ⟨K_r, K_s⟩`:
`S_horiz(r, t) = (⟨Ξ_r, K_t⟩ - ⟨K_r, Ξ_t⟩) - ∑_s (Ω_{rs} M_{st} + M_{rs} Ω_{st})`. -/
def krausHorizSkewMoment
    (R : Nat)
    (inner_Xi_K inner_K_Xi M Ω : Fin R → Fin R → Int)
    (r t : Fin R) : Int :=
  (inner_Xi_K r t - inner_K_Xi r t) -
    finSum R (fun s => Ω r s * M s t + M r s * Ω s t)

/-- **Theorem 3(b) — Sylvester / Lyapunov Horizontal Orthogonality Condition
(`theorem3b_kraus_sylvester_horizontal_iff`)**:
Under skew-Hermitian `Ω ∈ u(R)` (`Ω_conj t s = -Ω s t`), the horizontal tangent
`Ξ_horiz,r = Ξ_r - ∑_s Ω_{rs} K_s` has zero vertical `u(R)` gauge moment
(`(⟨Ξ_r, K_t⟩ - ∑_s Ω_{rs} M_{st}) - (⟨K_r, Ξ_t⟩ - ∑_s M_{rs} Ω^*_{ts}) = 0` for all `r, t`)
if and only if `Ω` satisfies the Sylvester/Lyapunov equation
`∑_{s < R} (Ω_{rs} M_{st} + M_{rs} Ω_{st}) = ⟨Ξ_r, K_t⟩ - ⟨K_r, Ξ_t⟩`. -/
theorem theorem3b_kraus_sylvester_horizontal_iff
    (R : Nat)
    (inner_Xi_K inner_K_Xi M Ω Ω_conj : Fin R → Fin R → Int)
    (h_skew_Ω : ∀ s t : Fin R, Ω_conj t s = -Ω s t) :
    (∀ r t : Fin R,
      (inner_Xi_K r t - finSum R (fun s => Ω r s * M s t)) -
      (inner_K_Xi r t - finSum R (fun s => M r s * Ω_conj t s)) = 0) ↔
    (∀ r t : Fin R,
      finSum R (fun s => Ω r s * M s t + M r s * Ω s t) =
        inner_Xi_K r t - inner_K_Xi r t) := by
  have h_sum_rewrite :
      ∀ r t : Fin R,
        finSum R (fun s => Ω r s * M s t) - finSum R (fun s => M r s * Ω_conj t s) =
          finSum R (fun s => Ω r s * M s t + M r s * Ω s t) := by
    intro r t
    rw [← finSum_sub]
    apply finSum_congr
    intro s
    rw [h_skew_Ω s t, Int.mul_neg]
    omega
  constructor
  · intro h_ortho r t
    have h_rt := h_ortho r t
    have h_rw := h_sum_rewrite r t
    omega
  · intro h_sylv r t
    have h_rt := h_sylv r t
    have h_rw := h_sum_rewrite r t
    omega

#print axioms theorem3a_kraus_unitary_channel_and_tp_invariance
#print axioms theorem3a_two_branch_direct_expansion
#print axioms theorem3b_kraus_sylvester_horizontal_iff

end QuantumDLBlocksLean
