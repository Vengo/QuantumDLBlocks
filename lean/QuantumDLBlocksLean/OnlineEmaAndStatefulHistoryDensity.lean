/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 8: Theorem 8 — Online EMA N-Gram Density Matrices & Stateful Multi-Window Density
History Invariance (Wave 69 `EXPERIMENT_PLAN_089.md`: Pure Density-Matrix Architecture).
-/
import QuantumDLBlocksLean.Core
import QuantumDLBlocksLean.HilbertSchmidtQuantumAttention
import QuantumDLBlocksLean.QuantumOrAndSchurLogic

namespace QuantumDLBlocksLean

/-- Predicate characterizing a valid trace-`S` positive semidefinite Hermitian 2×2 density matrix:
1. Symmetric/Hermitian: `ρ^T = ρ`,
2. Positive semidefinite: `∀ ψ_0 ψ_1, 0 ≤ ⟨ψ | ρ | ψ⟩`,
3. Exact trace: `Tr(ρ) = S`. -/
def IsValidDensity2 (S : Int) (ρ : Mat2) : Prop :=
  Mat2.transpose ρ = ρ ∧
  (∀ ψ0 ψ1 : Int, 0 ≤ quadForm2 ρ ψ0 ψ1) ∧
  Mat2.tr ρ = S

/-- Single-step convex/EMA combination `(1 - η) ρ_ema + η ρ_obs` (with `w_decay + w_obs = 1`,
`w_decay ≥ 0`, `w_obs ≥ 0`) preserves Hermiticity, positive semidefiniteness, and exact trace `S`. -/
theorem ema_step_preserves_valid_density2
    (S w_decay w_obs : Int)
    (h_decay : 0 ≤ w_decay) (h_obs : 0 ≤ w_obs) (h_sum : w_decay + w_obs = 1)
    (ρ_ema ρ_obs : Mat2)
    (h_valid_ema : IsValidDensity2 S ρ_ema)
    (h_valid_obs : IsValidDensity2 S ρ_obs) :
    IsValidDensity2 S (quantumOr2 w_decay w_obs ρ_ema ρ_obs) := by
  rcases h_valid_ema with ⟨h_sym_ema, h_psd_ema, h_tr_ema⟩
  rcases h_valid_obs with ⟨h_sym_obs, h_psd_obs, h_tr_obs⟩
  refine ⟨?_, ?_, ?_⟩
  · have h_e10 : ρ_ema.m10 = ρ_ema.m01 := congrArg Mat2.m01 h_sym_ema
    have h_o10 : ρ_obs.m10 = ρ_obs.m01 := congrArg Mat2.m01 h_sym_obs
    unfold quantumOr2 Mat2.transpose Mat2.add Mat2.smul
    ext <;> simp [h_e10, h_o10]
  · intro ψ0 ψ1
    have h_q_lin :
        quadForm2 (quantumOr2 w_decay w_obs ρ_ema ρ_obs) ψ0 ψ1 =
          w_decay * quadForm2 ρ_ema ψ0 ψ1 + w_obs * quadForm2 ρ_obs ψ0 ψ1 := by
      unfold quantumOr2 quadForm2 Mat2.add Mat2.smul
      simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc,
        Int.mul_comm ψ0 w_decay, Int.mul_comm ψ0 w_obs,
        Int.mul_comm ψ1 w_decay, Int.mul_comm ψ1 w_obs]
      ac_nf
    rw [h_q_lin]
    have h1 := Int.mul_nonneg h_decay (h_psd_ema ψ0 ψ1)
    have h2 := Int.mul_nonneg h_obs (h_psd_obs ψ0 ψ1)
    omega
  · have h_tr_lin :
        Mat2.tr (quantumOr2 w_decay w_obs ρ_ema ρ_obs) =
          w_decay * Mat2.tr ρ_ema + w_obs * Mat2.tr ρ_obs := by
      simp only [quantumOr2, Mat2.tr, Mat2.add, Mat2.smul, Int.mul_add]
      omega
    rw [h_tr_lin, h_tr_ema, h_tr_obs, ← Int.add_mul, h_sum, Int.one_mul]

/-- Recursive causal EMA density stream over `T` steps:
`ρ_{t+1} = w_decay • ρ_t + w_obs • ρ_obs(t)`. -/
def runOnlineEmaStream (w_decay w_obs : Int) (ρ_init : Mat2) (obs : Nat → Mat2) : Nat → Mat2
  | 0 => ρ_init
  | t + 1 => quantumOr2 w_decay w_obs (runOnlineEmaStream w_decay w_obs ρ_init obs t) (obs t)

/-- **Theorem 8(a) — Online Causal EMA N-Gram Density Update Invariance by Induction Over `T`
(`theorem8a_online_ema_stream_induction`)**:
If the initial N-gram bucket state `ρ_init` and every observed rank-1/outer-product token state
`obs t` satisfy `IsValidDensity2 S`, and the EMA weights satisfy `0 ≤ w_decay`, `0 ≤ w_obs`,
`w_decay + w_obs = 1`, then for every stream horizon `T : Nat`, the online EMA state
`runOnlineEmaStream w_decay w_obs ρ_init obs T` is Hermitian, positive semidefinite, and has
exact trace `S`. -/
theorem theorem8a_online_ema_stream_induction
    (S w_decay w_obs : Int)
    (h_decay : 0 ≤ w_decay) (h_obs : 0 ≤ w_obs) (h_sum : w_decay + w_obs = 1)
    (ρ_init : Mat2) (obs : Nat → Mat2)
    (h_init : IsValidDensity2 S ρ_init)
    (h_obs_valid : ∀ t : Nat, IsValidDensity2 S (obs t))
    (T : Nat) :
    IsValidDensity2 S (runOnlineEmaStream w_decay w_obs ρ_init obs T) := by
  induction T with
  | zero =>
    simpa [runOnlineEmaStream] using h_init
  | succ t ih =>
    simp only [runOnlineEmaStream]
    exact ema_step_preserves_valid_density2 S w_decay w_obs h_decay h_obs h_sum
      (runOnlineEmaStream w_decay w_obs ρ_init obs t) (obs t) ih (h_obs_valid t)

/-- **Theorem 8(b) — Multi-Window Stateful Summary Density Carry-Over Invariance
(`theorem8b_multi_window_stateful_summary_invariance`)**:
1. Combining a local window density matrix `ρ_local` (`IsValidDensity2 S ρ_local`) and a
   carried-over summary density matrix `ρ_summary` (`IsValidDensity2 S ρ_summary`) with
   convex weights `α_local + α_summary = 1` (`α_local, α_summary ≥ 0`) preserves `IsValidDensity2 S`,
2. More generally, across any `W`-window bank with non-negative weights `α : Fin W → Int`
   (`∀ w, 0 ≤ α w`, `∑_{w < W} α w = 1`), window quadratic forms `Q_win : Fin W → Int`
   (`∀ w, 0 ≤ Q_win w`), and window traces `Tr_win : Fin W → Int` (`∀ w, Tr_win w = S`),
   the multi-window convex combination has non-negative quadratic form `0 ≤ ∑_w α_w Q_win(w)`
   and exact trace `∑_w α_w Tr_win(w) = S`. -/
theorem theorem8b_multi_window_stateful_summary_invariance
    (S α_local α_summary : Int)
    (h_loc : 0 ≤ α_local) (h_sum_w : 0 ≤ α_summary) (h_norm2 : α_local + α_summary = 1)
    (ρ_local ρ_summary : Mat2)
    (h_valid_loc : IsValidDensity2 S ρ_local)
    (h_valid_sum : IsValidDensity2 S ρ_summary)
    (W : Nat) (α : Fin W → Int) (hα_nonneg : ∀ w : Fin W, 0 ≤ α w) (hα_sum : finSum W α = 1)
    (Q_win Tr_win : Fin W → Int)
    (hQ_nonneg : ∀ w : Fin W, 0 ≤ Q_win w)
    (hTr_eq : ∀ w : Fin W, Tr_win w = S) :
    IsValidDensity2 S (quantumOr2 α_local α_summary ρ_local ρ_summary) ∧
    0 ≤ finSum W (fun w => α w * Q_win w) ∧
    finSum W (fun w => α w * Tr_win w) = S := by
  have h_two_win :=
    ema_step_preserves_valid_density2 S α_local α_summary h_loc h_sum_w h_norm2
      ρ_local ρ_summary h_valid_loc h_valid_sum
  have h_multi_psd : 0 ≤ finSum W (fun w => α w * Q_win w) := by
    apply finSum_nonneg
    intro w
    exact Int.mul_nonneg (hα_nonneg w) (hQ_nonneg w)
  have h_multi_tr : finSum W (fun w => α w * Tr_win w) = S := by
    have h_congr :
        (fun w : Fin W => α w * Tr_win w) = (fun w : Fin W => S * α w) := by
      funext w
      rw [hTr_eq w, Int.mul_comm]
    rw [h_congr, finSum_const_mul W S α, hα_sum, Int.mul_one]
  exact ⟨h_two_win, h_multi_psd, h_multi_tr⟩

#print axioms theorem8a_online_ema_stream_induction
#print axioms theorem8b_multi_window_stateful_summary_invariance

end QuantumDLBlocksLean
