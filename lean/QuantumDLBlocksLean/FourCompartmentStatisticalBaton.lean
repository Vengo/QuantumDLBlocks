/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 13: Theorem 14 — 4-Compartment Quantum Statistical Mechanics Baton Architecture:
Born-Gated Quantum Statistical Backoff, Entrywise Schur Subspace Intersection vs. Additive
Mixing (`CodingHairpin`), Deterministic Walsh–Hadamard Token-ID Zero-Shot Reversal & Online
Thermal Bath Trace Invariance, and the $O(T d^2)$ Distributive Law Shortcut for 1–2 Layers
(Wave 73 `EXPERIMENT_PLAN_093.md` & `four_compartment_baton_lm_lab.py`).
-/
import QuantumDLBlocksLean.Core
import QuantumDLBlocksLean.HilbertSchmidtQuantumAttention
import QuantumDLBlocksLean.QuantumOrAndSchurLogic

namespace QuantumDLBlocksLean

/-- 4-state diagonal density matrix `(p_G, p_C, p_A, p_U)` over `{G, C, A, U}` representing
nucleotide / token subspace populations in `CodingHairpinHybridNet`. -/
structure Diag4 where
  pG : Int
  pC : Int
  pA : Int
  pU : Int
  deriving Repr, DecidableEq

namespace Diag4

/-- Trace `Tr(ρ) = p_G + p_C + p_A + p_U`. -/
def tr (ρ : Diag4) : Int :=
  ρ.pG + ρ.pC + ρ.pA + ρ.pU

/-- Unnormalized purity `Tr(ρ^2) = p_G^2 + p_C^2 + p_A^2 + p_U^2`. -/
def puritySq (ρ : Diag4) : Int :=
  ρ.pG * ρ.pG + ρ.pC * ρ.pC + ρ.pA * ρ.pA + ρ.pU * ρ.pU

/-- Weighted additive superposition `w_1 ρ_1 + w_2 ρ_2`. -/
def addWeighted (w1 w2 : Int) (ρ1 ρ2 : Diag4) : Diag4 :=
  ⟨w1 * ρ1.pG + w2 * ρ2.pG,
   w1 * ρ1.pC + w2 * ρ2.pC,
   w1 * ρ1.pA + w2 * ρ2.pA,
   w1 * ρ1.pU + w2 * ρ2.pU⟩

/-- Entrywise Schur / Hadamard intersection `ρ_1 ⊙ ρ_2`. -/
def schur (ρ1 ρ2 : Diag4) : Diag4 :=
  ⟨ρ1.pG * ρ2.pG,
   ρ1.pC * ρ2.pC,
   ρ1.pA * ρ2.pA,
   ρ1.pU * ρ2.pU⟩

/-- Non-negativity predicate `ρ ≥ 0` entrywise. -/
def NonNeg (ρ : Diag4) : Prop :=
  0 ≤ ρ.pG ∧ 0 ≤ ρ.pC ∧ 0 ≤ ρ.pA ∧ 0 ≤ ρ.pU

/-- Local `K`-gram MPS syntax density state supporting `{G, C}` (scaled by 2: `(1, 1, 0, 0)`). -/
def rhoMpsGC : Diag4 := ⟨1, 1, 0, 0⟩

/-- Long-range exterior Born-attention density state supporting `{G, A}` (scaled by 2: `(1, 0, 1, 0)`). -/
def rhoSeqGA : Diag4 := ⟨1, 0, 1, 0⟩

/-- 4-Compartment Quantum Statistical Backoff assembly
`w_id ρ_id + w_deep (ρ_seq ⊙ ρ_ngram) + w_ngram (ρ_ngram ⊙ ρ_uni) + (w_uni + ε_vac) ρ_uni`. -/
def fourCompartmentBackoff
    (wId wDeep wNgram wUni epsVac : Int)
    (ρId ρSeq ρNgram ρUni : Diag4) : Diag4 :=
  let ρDeep := schur ρSeq ρNgram
  let ρNgUni := schur ρNgram ρUni
  let wVac := wUni + epsVac
  ⟨wId * ρId.pG + wDeep * ρDeep.pG + wNgram * ρNgUni.pG + wVac * ρUni.pG,
   wId * ρId.pC + wDeep * ρDeep.pC + wNgram * ρNgUni.pC + wVac * ρUni.pC,
   wId * ρId.pA + wDeep * ρDeep.pA + wNgram * ρNgUni.pA + wVac * ρUni.pA,
   wId * ρId.pU + wDeep * ρDeep.pU + wNgram * ρNgUni.pU + wVac * ρUni.pU⟩

end Diag4

/-- **Theorem 14(a) — 4-Compartment Quantum Statistical Backoff & Schur Subspace Intersection
(`theorem14a_four_compartment_backoff_and_schur_subspace_intersection`)**:
1. **Conjunctive `CodingHairpin` Separation (`Schur` vs. `Additive`)**:
   When local `K`-gram MPS syntax supports `{G, C}` (`rhoMpsGC = (1, 1, 0, 0)`) and non-local
   exterior Born-attention supports `{G, A}` (`rhoSeqGA = (1, 0, 1, 0)`):
   - Additive mixing `ρ_add = rhoSeqGA + rhoMpsGC = (2, 1, 1, 0)` has total trace `4`, leaks
     `p_C + p_A = 2` (`50%` of total probability mass) onto half-satisfied forbidden states,
     and fails rank-1 purity (`Tr(ρ_add^2) = 6 < 16 = Tr(ρ_add)^2`), whereas
   - Entrywise Schur intersection `ρ_schur = rhoSeqGA ⊙ rhoMpsGC = (1, 0, 0, 0)` isolates the
     unique conjunctive state `G` with **zero leakage** (`p_C + p_A + p_U = 0`) and achieves
     exact Rank-1 purity (`Tr(ρ_schur^2) = Tr(ρ_schur)^2 = 1`).
2. **Strict Positive Trace Rescue Under Cold / Unseen Higher-Order Overlaps**:
   For any non-negative compartment states `ρ_id, ρ_seq, ρ_ngram, ρ_uni` and non-negative Born
   gates `w_id, w_deep, w_ngram, w_uni ≥ 0` with vacuum floor `eps_vac > 0` and positive thermal
   bath trace `Tr(ρ_uni) > 0`, the assembled 4-compartment state `ρ_out` is entrywise
   non-negative and has strictly positive trace `Tr(ρ_out) ≥ eps_vac * Tr(ρ_uni) > 0` even when
   all higher-order Schur overlaps (`ρ_seq ⊙ ρ_ngram = 0` and `ρ_ngram ⊙ ρ_uni = 0`) vanish. -/
theorem theorem14a_four_compartment_backoff_and_schur_subspace_intersection
    (wId wDeep wNgram wUni epsVac : Int)
    (ρId ρSeq ρNgram ρUni : Diag4)
    (hwId : 0 ≤ wId) (hwDeep : 0 ≤ wDeep) (hwNgram : 0 ≤ wNgram) (hwUni : 0 ≤ wUni)
    (heps : 0 < epsVac)
    (hId : Diag4.NonNeg ρId) (hSeq : Diag4.NonNeg ρSeq)
    (hNgram : Diag4.NonNeg ρNgram) (hUni : Diag4.NonNeg ρUni)
    (hTrUni : 0 < Diag4.tr ρUni) :
    let ρAdd := Diag4.addWeighted 1 1 Diag4.rhoSeqGA Diag4.rhoMpsGC
    let ρSchur := Diag4.schur Diag4.rhoSeqGA Diag4.rhoMpsGC
    let ρOut := Diag4.fourCompartmentBackoff wId wDeep wNgram wUni epsVac ρId ρSeq ρNgram ρUni
    (Diag4.tr ρAdd = 4 ∧
     ρAdd.pC + ρAdd.pA = 2 ∧
     2 * (ρAdd.pC + ρAdd.pA) = Diag4.tr ρAdd ∧
     Diag4.puritySq ρAdd = 6 ∧
     ρSchur = ⟨1, 0, 0, 0⟩ ∧
     ρSchur.pC + ρSchur.pA + ρSchur.pU = 0 ∧
     Diag4.puritySq ρSchur = Diag4.tr ρSchur * Diag4.tr ρSchur) ∧
    Diag4.NonNeg ρOut ∧
    epsVac * Diag4.tr ρUni ≤ Diag4.tr ρOut ∧
    0 < Diag4.tr ρOut := by
  dsimp only
  rcases hId with ⟨hIdG, hIdC, hIdA, hIdU⟩
  rcases hSeq with ⟨hSeqG, hSeqC, hSeqA, hSeqU⟩
  rcases hNgram with ⟨hNgG, hNgC, hNgA, hNgU⟩
  rcases hUni with ⟨hUniG, hUniC, hUniA, hUniU⟩
  have hVacNonneg : 0 ≤ wUni + epsVac := by omega
  have hDeepG : 0 ≤ wDeep * (ρSeq.pG * ρNgram.pG) :=
    Int.mul_nonneg hwDeep (Int.mul_nonneg hSeqG hNgG)
  have hDeepC : 0 ≤ wDeep * (ρSeq.pC * ρNgram.pC) :=
    Int.mul_nonneg hwDeep (Int.mul_nonneg hSeqC hNgC)
  have hDeepA : 0 ≤ wDeep * (ρSeq.pA * ρNgram.pA) :=
    Int.mul_nonneg hwDeep (Int.mul_nonneg hSeqA hNgA)
  have hDeepU : 0 ≤ wDeep * (ρSeq.pU * ρNgram.pU) :=
    Int.mul_nonneg hwDeep (Int.mul_nonneg hSeqU hNgU)
  have hNgUniG : 0 ≤ wNgram * (ρNgram.pG * ρUni.pG) :=
    Int.mul_nonneg hwNgram (Int.mul_nonneg hNgG hUniG)
  have hNgUniC : 0 ≤ wNgram * (ρNgram.pC * ρUni.pC) :=
    Int.mul_nonneg hwNgram (Int.mul_nonneg hNgC hUniC)
  have hNgUniA : 0 ≤ wNgram * (ρNgram.pA * ρUni.pA) :=
    Int.mul_nonneg hwNgram (Int.mul_nonneg hNgA hUniA)
  have hNgUniU : 0 ≤ wNgram * (ρNgram.pU * ρUni.pU) :=
    Int.mul_nonneg hwNgram (Int.mul_nonneg hNgU hUniU)
  have hIdTermG : 0 ≤ wId * ρId.pG := Int.mul_nonneg hwId hIdG
  have hIdTermC : 0 ≤ wId * ρId.pC := Int.mul_nonneg hwId hIdC
  have hIdTermA : 0 ≤ wId * ρId.pA := Int.mul_nonneg hwId hIdA
  have hIdTermU : 0 ≤ wId * ρId.pU := Int.mul_nonneg hwId hIdU
  have hVacG : 0 ≤ (wUni + epsVac) * ρUni.pG := Int.mul_nonneg hVacNonneg hUniG
  have hVacC : 0 ≤ (wUni + epsVac) * ρUni.pC := Int.mul_nonneg hVacNonneg hUniC
  have hVacA : 0 ≤ (wUni + epsVac) * ρUni.pA := Int.mul_nonneg hVacNonneg hUniA
  have hVacU : 0 ≤ (wUni + epsVac) * ρUni.pU := Int.mul_nonneg hVacNonneg hUniU
  have hUniTermTr : 0 ≤ wUni * (ρUni.pG + ρUni.pC + ρUni.pA + ρUni.pU) :=
    Int.mul_nonneg hwUni (by unfold Diag4.tr at hTrUni; omega)
  have hEpsTrPos : 0 < epsVac * (ρUni.pG + ρUni.pC + ρUni.pA + ρUni.pU) :=
    Int.mul_pos heps hTrUni
  have hVacDecomp :
      (wUni + epsVac) * ρUni.pG + (wUni + epsVac) * ρUni.pC +
      (wUni + epsVac) * ρUni.pA + (wUni + epsVac) * ρUni.pU =
        wUni * (ρUni.pG + ρUni.pC + ρUni.pA + ρUni.pU) +
        epsVac * (ρUni.pG + ρUni.pC + ρUni.pA + ρUni.pU) := by
    simp only [Int.add_mul, Int.mul_add]
    omega
  refine ⟨by decide, ?_, ?_, ?_⟩
  · unfold Diag4.NonNeg Diag4.fourCompartmentBackoff Diag4.schur
    dsimp only
    refine ⟨by omega, by omega, by omega, by omega⟩
  · unfold Diag4.tr Diag4.fourCompartmentBackoff Diag4.schur
    dsimp only
    omega
  · unfold Diag4.tr Diag4.fourCompartmentBackoff Diag4.schur
    dsimp only
    omega

/-- 4D Walsh–Hadamard codeword `|b(a)⟩ ∈ {±1}^4` for `a ∈ Fin 4`. -/
def walshCode4 (a : Fin 4) (i : Fin 4) : Int :=
  match a.val, i.val with
  | 0, _ => 1
  | 1, 0 => 1
  | 1, 1 => -1
  | 1, 2 => 1
  | 1, _ => -1
  | 2, 0 => 1
  | 2, 1 => 1
  | 2, 2 => -1
  | 2, _ => -1
  | _, 0 => 1
  | _, 1 => -1
  | _, 2 => -1
  | _, _ => 1

/-- Inner product `⟨b(a) | b(c)⟩` of two 4D Walsh–Hadamard Token-ID codewords. -/
def walshInner4 (a c : Fin 4) : Int :=
  finSum 4 (fun i => walshCode4 a i * walshCode4 c i)

/-- Born fidelity `|⟨b(a) | b(c)⟩|^2` for Token-ID routing under `U_V = I`. -/
def walshBornFidelity4 (a c : Fin 4) : Int :=
  let s := walshInner4 a c
  s * s

/-- Online-ingested unigram bath convex step `ρ_uni^{(t)} = (S - η) ρ_uni^{(t-1)} + η ρ_batch`
with integer scale `S` (representing `(1 - η/S) ρ_prev + (η/S) ρ_batch`). -/
def onlineBathStep (scaleMinusEta eta : Int) (ρPrev ρBatch : Diag4) : Diag4 :=
  Diag4.addWeighted scaleMinusEta eta ρPrev ρBatch

/-- **Theorem 14(b) — Deterministic Walsh–Hadamard Token-ID Zero-Shot Reversal & Online Bath Invariance
(`theorem14b_walsh_hadamard_token_id_zero_shot_reversal_and_online_bath`)**:
1. **Zero-Shot Unseen Token-ID Orthonormality**:
   For every token ID `a, c ∈ Fin 4`, the parameter-free Walsh–Hadamard codewords satisfy exact
   self-fidelity `walshBornFidelity4 a a = 16` (`d^2`) and zero cross-ID leakage
   `a ≠ c → walshBornFidelity4 a c = 0` under identity value routing `U_V = I`, without requiring
   prior gradient updates on token `a`.
2. **Online-Ingested Thermal Bath Trace & Positivity Preservation**:
   If `ρ_prev` and `ρ_batch` have equal unit-scaled trace `Tr(ρ_prev) = Tr(ρ_batch) = T_0` and
   are entrywise non-negative, then for any online learning rate split `scaleMinusEta + eta = S`
   (`scaleMinusEta ≥ 0, eta ≥ 0`), the online-ingested bath `onlineBathStep scaleMinusEta eta ρPrev ρBatch`
   remains entrywise non-negative and satisfies exact trace invariance `Tr(ρ_next) = S * T_0`. -/
theorem theorem14b_walsh_hadamard_token_id_zero_shot_reversal_and_online_bath
    (scaleMinusEta eta T0 : Int)
    (ρPrev ρBatch : Diag4)
    (h1 : 0 ≤ scaleMinusEta) (h2 : 0 ≤ eta)
    (hPrev : Diag4.NonNeg ρPrev) (hBatch : Diag4.NonNeg ρBatch)
    (hTrPrev : Diag4.tr ρPrev = T0) (hTrBatch : Diag4.tr ρBatch = T0) :
    (∀ a : Fin 4, walshBornFidelity4 a a = 16) ∧
    (∀ a c : Fin 4, a ≠ c → walshBornFidelity4 a c = 0) ∧
    Diag4.NonNeg (onlineBathStep scaleMinusEta eta ρPrev ρBatch) ∧
    Diag4.tr (onlineBathStep scaleMinusEta eta ρPrev ρBatch) = (scaleMinusEta + eta) * T0 := by
  rcases hPrev with ⟨hpG, hpC, hpA, hpU⟩
  rcases hBatch with ⟨hbG, hbC, hbA, hbU⟩
  have hNextG : 0 ≤ scaleMinusEta * ρPrev.pG + eta * ρBatch.pG :=
    Int.add_nonneg (Int.mul_nonneg h1 hpG) (Int.mul_nonneg h2 hbG)
  have hNextC : 0 ≤ scaleMinusEta * ρPrev.pC + eta * ρBatch.pC :=
    Int.add_nonneg (Int.mul_nonneg h1 hpC) (Int.mul_nonneg h2 hbC)
  have hNextA : 0 ≤ scaleMinusEta * ρPrev.pA + eta * ρBatch.pA :=
    Int.add_nonneg (Int.mul_nonneg h1 hpA) (Int.mul_nonneg h2 hbA)
  have hNextU : 0 ≤ scaleMinusEta * ρPrev.pU + eta * ρBatch.pU :=
    Int.add_nonneg (Int.mul_nonneg h1 hpU) (Int.mul_nonneg h2 hbU)
  refine ⟨by decide, by decide, ⟨hNextG, hNextC, hNextA, hNextU⟩, ?_⟩
  have hTrCalc :
      Diag4.tr (onlineBathStep scaleMinusEta eta ρPrev ρBatch) =
        scaleMinusEta * Diag4.tr ρPrev + eta * Diag4.tr ρBatch := by
    unfold onlineBathStep Diag4.addWeighted Diag4.tr
    simp only [Int.mul_add]
    omega
  rw [hTrCalc, hTrPrev, hTrBatch, Int.add_mul]

/-- **Theorem 14(c) — Distributive Law Shortcut (`O(T d^2)`) & 2-Layer 4-Way Conjunction
(`theorem14c_distributive_factored_schur_and_two_layer_four_way_conjunction`)**:
1. **`O(T d^2)` Distributive Factored Schur Identity**:
   For any pair of linear attention-weighted density aggregates `ρ̄_A = w_{A1} ρ_{A1} + w_{A2} ρ_{A2}`
   and `ρ̄_B = w_{B1} ρ_{B1} + w_{B2} ρ_{B2}`, computing the entrywise Schur product *after*
   linear aggregation equals the sum over all `2 × 2 = 4` pairwise Schur intersections:
   `ρ̄_A ⊙ ρ̄_B = ∑_{i,j ∈ {1,2}} (w_{Ai} w_{Bj}) (ρ_{Ai} ⊙ ρ_{Bj})`.
2. **2-Layer 4-Way Conjunction Expansion**:
   When Layer 1 forms two 2-way Schur states `M_1 = ρ_1 ⊙ ρ_2` and `M_2 = ρ_3 ⊙ ρ_4`,
   a second layer's Schur product `M_1 ⊙ M_2` equals the associative 4-way entrywise conjunction
   `ρ_1 ⊙ ρ_2 ⊙ ρ_3 ⊙ ρ_4`. -/
theorem theorem14c_distributive_factored_schur_and_two_layer_four_way_conjunction
    (wA1 wA2 wB1 wB2 : Int)
    (ρA1 ρA2 ρB1 ρB2 : Mat2) :
    let ρBarA := quantumOr2 wA1 wA2 ρA1 ρA2
    let ρBarB := quantumOr2 wB1 wB2 ρB1 ρB2
    schurMul2 ρBarA ρBarB =
      Mat2.add
        (quantumOr2 (wA1 * wB1) (wA1 * wB2) (schurMul2 ρA1 ρB1) (schurMul2 ρA1 ρB2))
        (quantumOr2 (wA2 * wB1) (wA2 * wB2) (schurMul2 ρA2 ρB1) (schurMul2 ρA2 ρB2)) ∧
    schurMul2 (schurMul2 ρA1 ρA2) (schurMul2 ρB1 ρB2) =
      schurMul2 (schurMul2 (schurMul2 ρA1 ρA2) ρB1) ρB2 := by
  dsimp only
  refine ⟨?_, ?_⟩
  · cases ρA1
    cases ρA2
    cases ρB1
    cases ρB2
    unfold schurMul2 quantumOr2 Mat2.add Mat2.smul
    ext <;> simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc] <;> ac_nf
  · cases ρA1
    cases ρA2
    cases ρB1
    cases ρB2
    unfold schurMul2
    ext <;> simp only [← Int.mul_assoc]

#print axioms theorem14a_four_compartment_backoff_and_schur_subspace_intersection
#print axioms theorem14b_walsh_hadamard_token_id_zero_shot_reversal_and_online_bath
#print axioms theorem14c_distributive_factored_schur_and_two_layer_four_way_conjunction

end QuantumDLBlocksLean
