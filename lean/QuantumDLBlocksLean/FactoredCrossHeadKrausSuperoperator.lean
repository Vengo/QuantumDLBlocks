/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Module 9: Theorems 9 & 10 — Exact Algebraic Equivalence & Representational Power of the
Factored Cross-Head Kraus Superoperator (Wave 69/70: O(H M) Factored Congruence Equivalence,
Attention-Cholesky Commutation, CPTP Closure, and Strict Expressivity Separation Over
Scalar Orthostochastic Mixing).
-/
import QuantumDLBlocksLean.Core
import QuantumDLBlocksLean.KrausUnitaryGaugeSylvester
import QuantumDLBlocksLean.HilbertSchmidtQuantumAttention
import QuantumDLBlocksLean.QuantumOrAndSchurLogic
import QuantumDLBlocksLean.OnlineEmaAndStatefulHistoryDensity

namespace QuantumDLBlocksLean

/-- Right scalar multiplication distributes over `finSum`: `∑_i (f(i) * c) = (∑_i f(i)) * c`. -/
theorem finSum_mul_const (n : Nat) (f : Fin n → Int) (c : Int) :
    finSum n (fun i => f i * c) = finSum n f * c := by
  induction n with
  | zero =>
    simp [finSum]
  | succ k ih =>
    simp only [finSum]
    rw [ih, Int.add_mul]

/-- Matrix-valued finite sum `∑_{i < n} F(i)` on `Mat2`, defined entrywise via `finSum`. -/
def mat2FinSum (n : Nat) (F : Fin n → Mat2) : Mat2 :=
  ⟨finSum n (fun i => (F i).m00),
   finSum n (fun i => (F i).m01),
   finSum n (fun i => (F i).m10),
   finSum n (fun i => (F i).m11)⟩

/-- Pointwise equality of matrix families implies equal `mat2FinSum`. -/
theorem mat2FinSum_congr {n : Nat} {F G : Fin n → Mat2} (h : ∀ i : Fin n, F i = G i) :
    mat2FinSum n F = mat2FinSum n G := by
  unfold mat2FinSum
  ext <;> apply finSum_congr <;> intro i <;> rw [h i]

/-- Left multiplication distributes over `mat2FinSum`: `A * (∑_i F(i)) = ∑_i (A * F(i))`. -/
theorem mat2_mul_mat2FinSum (n : Nat) (A : Mat2) (F : Fin n → Mat2) :
    Mat2.mul A (mat2FinSum n F) = mat2FinSum n (fun i => Mat2.mul A (F i)) := by
  ext <;>
    simp only [Mat2.mul, mat2FinSum, ← finSum_const_mul, ← finSum_add]

/-- Right multiplication distributes over `mat2FinSum`: `(∑_i F(i)) * B = ∑_i (F(i) * B)`. -/
theorem mat2FinSum_mul_mat2 (n : Nat) (F : Fin n → Mat2) (B : Mat2) :
    Mat2.mul (mat2FinSum n F) B = mat2FinSum n (fun i => Mat2.mul (F i) B) := by
  ext <;>
    simp only [Mat2.mul, mat2FinSum, ← finSum_mul_const, ← finSum_add]

/-- Fubini / commutation of finite sums over `Fin M × Fin H`:
`∑_{m < M} ∑_{h' < H} f(m, h') = ∑_{h' < H} ∑_{m < M} f(m, h')`. -/
theorem finSum_comm (M H : Nat) (f : Fin M → Fin H → Int) :
    finSum M (fun m => finSum H (fun h' => f m h')) =
      finSum H (fun h' => finSum M (fun m => f m h')) := by
  induction M with
  | zero =>
    simp only [finSum]
    symm
    exact finSum_zero H
  | succ k ih =>
    simp only [finSum]
    rw [ih (fun m h' => f m.castSucc h'), ← finSum_add]

/-- Commutation of double matrix sums over `Fin M × Fin H`. -/
theorem mat2FinSum_comm (M H : Nat) (F : Fin M → Fin H → Mat2) :
    mat2FinSum M (fun m => mat2FinSum H (fun h' => F m h')) =
      mat2FinSum H (fun h' => mat2FinSum M (fun m => F m h')) := by
  unfold mat2FinSum
  ext <;> apply finSum_comm

/-- Single-branch Kraus congruence `K ρ K^T`. -/
def krausCongruence2 (K ρ : Mat2) : Mat2 :=
  Mat2.mul (Mat2.mul K ρ) (Mat2.transpose K)

/-- Kraus congruence distributes over `mat2FinSum`:
`K (∑_i F(i)) K^T = ∑_i (K F(i) K^T)`. -/
theorem krausCongruence2_mat2FinSum (n : Nat) (K : Mat2) (F : Fin n → Mat2) :
    krausCongruence2 K (mat2FinSum n F) =
      mat2FinSum n (fun i => krausCongruence2 K (F i)) := by
  unfold krausCongruence2
  rw [mat2_mul_mat2FinSum, mat2FinSum_mul_mat2]

/-- Scalar multiplication commutes with Kraus congruence:
`K (w • ρ) K^T = w • (K ρ K^T)`. -/
theorem krausCongruence2_smul (K : Mat2) (w : Int) (ρ : Mat2) :
    krausCongruence2 K (Mat2.smul w ρ) = Mat2.smul w (krausCongruence2 K ρ) := by
  cases K
  cases ρ
  unfold krausCongruence2 Mat2.mul Mat2.transpose Mat2.smul
  ext <;> simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc] <;> ac_nf

/-- Composition of Kraus congruences equals congruence by the composite operator `K_out * K_in`:
`K_out (K_in ρ K_in^T) K_out^T = (K_out K_in) ρ (K_out K_in)^T`. -/
theorem krausCongruence2_comp (K_out K_in ρ : Mat2) :
    krausCongruence2 K_out (krausCongruence2 K_in ρ) =
      krausCongruence2 (Mat2.mul K_out K_in) ρ := by
  unfold krausCongruence2
  rw [mat2_transpose_mul, mat2_mul_assoc K_out K_in ρ,
      ← mat2_mul_assoc (Mat2.mul K_out (Mat2.mul K_in ρ)) (Mat2.transpose K_in) (Mat2.transpose K_out),
      ← mat2_mul_assoc K_out (Mat2.mul K_in ρ) (Mat2.transpose K_in)]

/-- **Theorem 9(a) — Exact Algebraic Equivalence of Factored `O(H M)` vs. Naive `O(H^2 M)`
Cross-Head Kraus Superoperators (`factored_kraus_associative_parenthesization_eq`)**:
For any number of heads `H`, branch rank `M`, input density matrices `ρ : Fin H → Mat2`,
input operators `K_in : Fin H → Fin M → Mat2`, output operators `K_out : Fin H → Fin M → Mat2`,
and routing weights `w : Fin H → Fin H → Fin M → Int`, the 3-stage factored parenthesization:
`∑_{m < M} K_out(h, m) (∑_{h' < H} w(h, h', m) • (K_in(h', m) ρ(h') K_in(h', m)^T)) K_out(h, m)^T`
is identically equal to the naive `O(H^2 M)` loop with composite operators
`K_{h, h', m} = K_out(h, m) K_in(h', m)`:
`∑_{h' < H} ∑_{m < M} w(h, h', m) • ((K_out(h, m) K_in(h', m)) ρ(h') (K_out(h, m) K_in(h', m))^T)`. -/
theorem factored_kraus_associative_parenthesization_eq
    (H M : Nat)
    (ρ : Fin H → Mat2)
    (K_in K_out : Fin H → Fin M → Mat2)
    (w : Fin H → Fin H → Fin M → Int)
    (h : Fin H) :
    mat2FinSum M (fun m =>
      krausCongruence2 (K_out h m)
        (mat2FinSum H (fun h' =>
          Mat2.smul (w h h' m) (krausCongruence2 (K_in h' m) (ρ h'))))) =
    mat2FinSum H (fun h' =>
      mat2FinSum M (fun m =>
        Mat2.smul (w h h' m)
          (krausCongruence2 (Mat2.mul (K_out h m) (K_in h' m)) (ρ h')))) := by
  have h_step1 :
      mat2FinSum M (fun m =>
        krausCongruence2 (K_out h m)
          (mat2FinSum H (fun h' =>
            Mat2.smul (w h h' m) (krausCongruence2 (K_in h' m) (ρ h'))))) =
      mat2FinSum M (fun m =>
        mat2FinSum H (fun h' =>
          Mat2.smul (w h h' m)
            (krausCongruence2 (Mat2.mul (K_out h m) (K_in h' m)) (ρ h')))) := by
    apply mat2FinSum_congr
    intro m
    rw [krausCongruence2_mat2FinSum]
    apply mat2FinSum_congr
    intro h'
    rw [krausCongruence2_smul, krausCongruence2_comp]
  rw [h_step1]
  exact mat2FinSum_comm M H (fun m h' =>
    Mat2.smul (w h h' m) (krausCongruence2 (Mat2.mul (K_out h m) (K_in h' m)) (ρ h')))

/-- Packed symmetric 2×2 representation storing `(m00, m11, m01)`. -/
structure PackedSym2 where
  p00 : Int
  p11 : Int
  p01 : Int
  deriving Repr, DecidableEq

/-- Pack a symmetric 2×2 matrix into `PackedSym2`. -/
def packSym2 (M : Mat2) : PackedSym2 :=
  ⟨M.m00, M.m11, M.m01⟩

/-- Unpack `PackedSym2` back to a symmetric 2×2 matrix `Mat2`. -/
def unpackSym2 (P : PackedSym2) : Mat2 :=
  ⟨P.p00, P.p01, P.p01, P.p11⟩

/-- Weighted finite sum in `PackedSym2`. -/
def packedSym2WeightedSum (H : Nat) (w : Fin H → Int) (P : Fin H → PackedSym2) : PackedSym2 :=
  ⟨finSum H (fun h' => w h' * (P h').p00),
   finSum H (fun h' => w h' * (P h').p11),
   finSum H (fun h' => w h' * (P h').p01)⟩

/-- **Theorem 9(b) — Kraus Commutation with Causal Attention Sum & Isometric Packed Routing Linearity
(`kraus_commutes_with_causal_attention_sum` & `isometric_packed_routing_linearity`)**:
1. Applying `K_in (·) K_in^T` after the causal attention sum `∑_{s < S} A_s • (V_s V_s^T)`
   is identically equal to multiplying the Value Cholesky factors `K_in V_s` before the attention sum:
   `K_in (∑_{s < S} A_s • (V_s V_s^T)) K_in^T = ∑_{s < S} A_s • ((K_in V_s)(K_in V_s)^T)`,
2. For any family of symmetric matrices `σ : Fin H → Mat2` (`∀ h', (σ h')^T = σ h'`),
   linear routing in the packed space unpacks to the exact matrix linear combination
   `∑_{h' < H} w_{h'} • σ_{h'}`. -/
theorem kraus_commutes_with_causal_attention_sum
    (S : Nat) (K_in : Mat2) (A : Fin S → Int) (V : Fin S → Mat2) :
    krausCongruence2 K_in (mat2FinSum S (fun s => Mat2.smul (A s) (outerDensity2 (V s)))) =
      mat2FinSum S (fun s => Mat2.smul (A s) (outerDensity2 (Mat2.mul K_in (V s)))) := by
  rw [krausCongruence2_mat2FinSum]
  apply mat2FinSum_congr
  intro s
  rw [krausCongruence2_smul]
  have h_outer : krausCongruence2 K_in (outerDensity2 (V s)) = outerDensity2 (Mat2.mul K_in (V s)) := by
    unfold krausCongruence2 outerDensity2
    rw [mat2_transpose_mul,
        ← mat2_mul_assoc K_in (V s) (Mat2.transpose (V s)),
        mat2_mul_assoc (Mat2.mul K_in (V s)) (Mat2.transpose (V s)) (Mat2.transpose K_in)]
  rw [h_outer]

theorem isometric_packed_routing_linearity
    (H : Nat) (w : Fin H → Int) (σ : Fin H → Mat2)
    (h_sym : ∀ h' : Fin H, Mat2.transpose (σ h') = σ h') :
    unpackSym2 (packedSym2WeightedSum H w (fun h' => packSym2 (σ h'))) =
      mat2FinSum H (fun h' => Mat2.smul (w h') (σ h')) := by
  simp only [unpackSym2, packedSym2WeightedSum, packSym2, mat2FinSum, Mat2.smul]
  ext
  · rfl
  · rfl
  · apply finSum_congr
    intro h'
    dsimp only
    have h10 : (σ h').m01 = (σ h').m10 := congrArg Mat2.m10 (h_sym h')
    rw [h10]
  · rfl

/-- Congruence `K ρ K^T` preserves symmetry and positive semidefiniteness for any `K : Mat2`,
and preserves trace whenever `K^T K = I_2`. -/
theorem krausCongruence2_preserves_valid_density
    (S_tr : Int) (K ρ : Mat2)
    (h_iso : Mat2.mul (Mat2.transpose K) K = Mat2.eye)
    (h_valid : IsValidDensity2 S_tr ρ) :
    IsValidDensity2 S_tr (krausCongruence2 K ρ) := by
  rcases h_valid with ⟨h_sym, h_psd, h_tr⟩
  refine ⟨?_, ?_, ?_⟩
  · unfold krausCongruence2
    rw [mat2_transpose_mul, mat2_transpose_mul, Mat2.transpose_transpose, h_sym, mat2_mul_assoc]
  · intro ψ0 ψ1
    let φ0 := K.m00 * ψ0 + K.m10 * ψ1
    let φ1 := K.m01 * ψ0 + K.m11 * ψ1
    have h_quad_pullback :
        quadForm2 (krausCongruence2 K ρ) ψ0 ψ1 = quadForm2 ρ φ0 φ1 := by
      cases K
      cases ρ
      unfold quadForm2 krausCongruence2 Mat2.mul Mat2.transpose φ0 φ1
      simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc]
      ac_nf
    rw [h_quad_pullback]
    exact h_psd φ0 φ1
  · have h_cyclic :
        Mat2.tr (krausCongruence2 K ρ) =
          Mat2.tr (Mat2.mul (Mat2.mul (Mat2.transpose K) K) ρ) := by
      cases K
      cases ρ
      unfold krausCongruence2 Mat2.tr Mat2.mul Mat2.transpose
      simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc]
      ac_nf
    rw [h_cyclic, h_iso, mat2_eye_mul, h_tr]

/-- Trace of `mat2FinSum` equals the `finSum` of traces. -/
theorem tr_mat2FinSum (n : Nat) (F : Fin n → Mat2) :
    Mat2.tr (mat2FinSum n F) = finSum n (fun i => Mat2.tr (F i)) := by
  unfold Mat2.tr mat2FinSum
  rw [← finSum_add]

/-- Quadratic form of `mat2FinSum` equals the `finSum` of quadratic forms. -/
theorem quadForm2_mat2FinSum (n : Nat) (F : Fin n → Mat2) (ψ0 ψ1 : Int) :
    quadForm2 (mat2FinSum n F) ψ0 ψ1 = finSum n (fun i => quadForm2 (F i) ψ0 ψ1) := by
  simp only [quadForm2, mat2FinSum, ← finSum_mul_const, ← finSum_const_mul, ← finSum_add]

/-- **Theorem 10(a) — Exact CPTP Closure of Factored Cross-Head Kraus Superoperator
(`factored_cross_head_kraus_cptp_closure`)**:
When all input heads `ρ h'` satisfy `IsValidDensity2 S_tr (ρ h')`, the branch operators
`K_in h' m` and `K_out h m` are isometric (`K^T K = I_2`), and the non-negative routing
weights `w h h' m ≥ 0` satisfy `∑_{m < M} ∑_{h' < H} w(h, h', m) = 1`, the factored
superoperator output `E_factored(ρ)_h` satisfies `IsValidDensity2 S_tr` (Hermitian,
positive semidefinite, and exact trace `S_tr`, including `S_tr = 1`). -/
theorem factored_cross_head_kraus_cptp_closure
    (H M : Nat) (S_tr : Int)
    (ρ : Fin H → Mat2) (h_valid_ρ : ∀ h' : Fin H, IsValidDensity2 S_tr (ρ h'))
    (K_in K_out : Fin H → Fin M → Mat2)
    (h_iso_in : ∀ (h' : Fin H) (m : Fin M), Mat2.mul (Mat2.transpose (K_in h' m)) (K_in h' m) = Mat2.eye)
    (h_iso_out : ∀ (h : Fin H) (m : Fin M), Mat2.mul (Mat2.transpose (K_out h m)) (K_out h m) = Mat2.eye)
    (w : Fin H → Fin H → Fin M → Int)
    (hw_nonneg : ∀ (h h' : Fin H) (m : Fin M), 0 ≤ w h h' m)
    (h : Fin H)
    (hw_sum : finSum M (fun m => finSum H (fun h' => w h h' m)) = 1) :
    let out_h :=
      mat2FinSum M (fun m =>
        krausCongruence2 (K_out h m)
          (mat2FinSum H (fun h' =>
            Mat2.smul (w h h' m) (krausCongruence2 (K_in h' m) (ρ h')))))
    IsValidDensity2 S_tr out_h := by
  intro out_h
  have h_eq_naive :
      out_h =
        mat2FinSum M (fun m =>
          mat2FinSum H (fun h' =>
            Mat2.smul (w h h' m)
              (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h'))))) := by
    apply mat2FinSum_congr
    intro m
    rw [krausCongruence2_mat2FinSum]
    apply mat2FinSum_congr
    intro h'
    exact krausCongruence2_smul (K_out h m) (w h h' m) (krausCongruence2 (K_in h' m) (ρ h'))
  have h_branch_valid :
      ∀ (m : Fin M) (h' : Fin H),
        IsValidDensity2 S_tr (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h'))) := by
    intro m h'
    apply krausCongruence2_preserves_valid_density S_tr (K_out h m) _ (h_iso_out h m)
    exact krausCongruence2_preserves_valid_density S_tr (K_in h' m) (ρ h') (h_iso_in h' m) (h_valid_ρ h')
  rw [h_eq_naive]
  refine ⟨?_, ?_, ?_⟩
  · unfold mat2FinSum Mat2.transpose
    ext
    · rfl
    · apply finSum_congr
      intro m
      apply finSum_congr
      intro h'
      have h_sym_mh := (h_branch_valid m h').1
      have h10 : (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h'))).m10 =
                 (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h'))).m01 :=
        congrArg Mat2.m01 h_sym_mh
      simp [Mat2.smul, h10]
    · apply finSum_congr
      intro m
      apply finSum_congr
      intro h'
      have h_sym_mh := (h_branch_valid m h').1
      have h01 : (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h'))).m01 =
                 (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h'))).m10 :=
        congrArg Mat2.m10 h_sym_mh
      simp [Mat2.smul, h01]
    · rfl
  · intro ψ0 ψ1
    rw [quadForm2_mat2FinSum]
    apply finSum_nonneg
    intro m
    rw [quadForm2_mat2FinSum]
    apply finSum_nonneg
    intro h'
    have h_q_smul :
        quadForm2 (Mat2.smul (w h h' m) (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h')))) ψ0 ψ1 =
          w h h' m * quadForm2 (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h'))) ψ0 ψ1 := by
      unfold quadForm2 Mat2.smul
      simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc,
        Int.mul_comm ψ0 (w h h' m), Int.mul_comm ψ1 (w h h' m)]
    rw [h_q_smul]
    exact Int.mul_nonneg (hw_nonneg h h' m) ((h_branch_valid m h').2.1 ψ0 ψ1)
  · rw [tr_mat2FinSum]
    have h_inner_tr :
        (fun m : Fin M => Mat2.tr (mat2FinSum H (fun h' =>
          Mat2.smul (w h h' m) (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h')))))) =
        (fun m : Fin M => S_tr * finSum H (fun h' => w h h' m)) := by
      funext m
      rw [tr_mat2FinSum, ← finSum_const_mul]
      apply finSum_congr
      intro h'
      have h_tr_mh := (h_branch_valid m h').2.2
      have h_tr_smul :
          Mat2.tr (Mat2.smul (w h h' m) (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h')))) =
            w h h' m * Mat2.tr (krausCongruence2 (K_out h m) (krausCongruence2 (K_in h' m) (ρ h'))) := by
        simp only [Mat2.tr, Mat2.smul, Int.mul_add]
      rw [h_tr_smul, h_tr_mh, Int.mul_comm]
    rw [h_inner_tr, finSum_const_mul, hw_sum, Int.mul_one]

/-- 2×2 matrix determinant `det(A) = A_{00} A_{11} - A_{01} A_{10}` (positive iff rank = 2 for PSD `Mat2`). -/
def mat2Det (A : Mat2) : Int :=
  A.m00 * A.m11 - A.m01 * A.m10

/-- Rank-1 input state `|0⟩⟨0| = [[1, 0], [0, 0]]` supported strictly on the first basis vector. -/
def witnessRho0 : Mat2 :=
  ⟨1, 0, 0, 0⟩

/-- Orthogonal basis rotation operator `K_rot = [[0, -1], [1, 0]]` mapping `|0⟩ ↦ |1⟩`. -/
def witnessKRot : Mat2 :=
  ⟨0, -1, 1, 0⟩

/-- **Theorem 10(b) — Strict Expressivity Separation & Rank Amplification Over Scalar
Orthostochastic Mixing (`representational_power_strict_separation_and_rank_amplification`)**:
1. **Kernel Trap of Scalar Orthostochastic Mixing**: For ANY number of heads `H`, any scalar
   mixing weights `B : Fin H → Int`, and any vector `v = (v_0, v_1)` in the common kernel of
   all input heads (`∀ h', ⟨v | ρ_{h'} | v⟩ = 0`), the scalar-mixed state
   `ρ_scalar = ∑_{h' < H} B_{h'} • ρ_{h'}` remains trapped in the kernel:
   `⟨v | ρ_scalar | v⟩ = 0`,
2. **Concrete Separation & `1 → 2` Rank Amplification Witness**: On the rank-1 input
   `witnessRho0 = |0⟩⟨0|` (`det(witnessRho0) = 0`, `⟨e_1 | witnessRho0 | e_1⟩ = 0` for `e_1 = (0, 1)`),
   every scalar mix has `⟨e_1 | ρ_scalar | e_1⟩ = 0`, whereas the 2-branch factored Kraus
   superoperator with orthogonal operators `(I_2, witnessKRot)` produces a state `ρ_kraus` with:
   - `⟨e_1 | ρ_kraus | e_1⟩ = 1 > 0` (populates the orthogonal basis state `|1⟩`), and
   - `det(ρ_kraus) = 1 > 0` (strictly amplifies matrix rank from rank `1` to full rank `2`). -/
theorem representational_power_strict_separation_and_rank_amplification
    (H : Nat) (B : Fin H → Int) (ρ : Fin H → Mat2) (v0 v1 : Int)
    (h_kernel : ∀ h' : Fin H, quadForm2 (ρ h') v0 v1 = 0) :
    quadForm2 (mat2FinSum H (fun h' => Mat2.smul (B h') (ρ h'))) v0 v1 = 0 ∧
    quadForm2 witnessRho0 0 1 = 0 ∧
    mat2Det witnessRho0 = 0 ∧
    Mat2.mul (Mat2.transpose witnessKRot) witnessKRot = Mat2.eye ∧
    quadForm2
      (Mat2.add (krausCongruence2 Mat2.eye witnessRho0)
                (krausCongruence2 witnessKRot witnessRho0)) 0 1 = 1 ∧
    mat2Det
      (Mat2.add (krausCongruence2 Mat2.eye witnessRho0)
                (krausCongruence2 witnessKRot witnessRho0)) = 1 := by
  refine ⟨?_, by decide, by decide, by decide, by decide, by decide⟩
  rw [quadForm2_mat2FinSum]
  have h_zero_terms :
      (fun i : Fin H => quadForm2 (Mat2.smul (B i) (ρ i)) v0 v1) = (fun _ : Fin H => 0) := by
    funext h'
    have h_q_smul :
        quadForm2 (Mat2.smul (B h') (ρ h')) v0 v1 = B h' * quadForm2 (ρ h') v0 v1 := by
      unfold quadForm2 Mat2.smul
      simp only [Int.add_mul, Int.mul_add, ← Int.mul_assoc,
        Int.mul_comm v0 (B h'), Int.mul_comm v1 (B h')]
    rw [h_q_smul, h_kernel h', Int.mul_zero]
  rw [h_zero_terms]
  exact finSum_zero H

#print axioms factored_kraus_associative_parenthesization_eq
#print axioms kraus_commutes_with_causal_attention_sum
#print axioms isometric_packed_routing_linearity
#print axioms factored_cross_head_kraus_cptp_closure
#print axioms representational_power_strict_separation_and_rank_amplification

end QuantumDLBlocksLean
