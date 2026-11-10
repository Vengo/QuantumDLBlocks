/-
Copyright (c) 2026 QuantumDLBlocks Authors. All rights reserved.
Foundational dependency-free algebraic, matrix, complex, and finite-sum primitives
for machine-checked Riemannian Stiefel/Grassmann/Kraus/Lüders theorems (Waves 66–68).
-/

namespace QuantumDLBlocksLean

/-- Exact 2×2 matrix over `Int` for constructive block/tangent/Newton–Schulz calculations. -/
structure Mat2 where
  m00 : Int
  m01 : Int
  m10 : Int
  m11 : Int
  deriving Repr, DecidableEq

namespace Mat2

/-- Zero 2×2 matrix. -/
def zero : Mat2 := ⟨0, 0, 0, 0⟩

/-- Identity 2×2 matrix. -/
def eye : Mat2 := ⟨1, 0, 0, 1⟩

/-- Diagonal 2×2 matrix `diag(d0, d1)`. -/
def diag (d0 d1 : Int) : Mat2 := ⟨d0, 0, 0, d1⟩

/-- Matrix transpose `A^T`. -/
def transpose (A : Mat2) : Mat2 :=
  ⟨A.m00, A.m10, A.m01, A.m11⟩

/-- Matrix addition `A + B`. -/
def add (A B : Mat2) : Mat2 :=
  ⟨A.m00 + B.m00, A.m01 + B.m01, A.m10 + B.m10, A.m11 + B.m11⟩

/-- Matrix subtraction `A - B`. -/
def sub (A B : Mat2) : Mat2 :=
  ⟨A.m00 - B.m00, A.m01 - B.m01, A.m10 - B.m10, A.m11 - B.m11⟩

/-- Matrix negation `-A`. -/
def neg (A : Mat2) : Mat2 :=
  ⟨-A.m00, -A.m01, -A.m10, -A.m11⟩

/-- Scalar multiplication `c • A`. -/
def smul (c : Int) (A : Mat2) : Mat2 :=
  ⟨c * A.m00, c * A.m01, c * A.m10, c * A.m11⟩

/-- Matrix multiplication `A * B`. -/
def mul (A B : Mat2) : Mat2 :=
  ⟨A.m00 * B.m00 + A.m01 * B.m10,
   A.m00 * B.m01 + A.m01 * B.m11,
   A.m10 * B.m00 + A.m11 * B.m10,
   A.m10 * B.m01 + A.m11 * B.m11⟩

/-- Matrix trace `Tr(A)`. -/
def tr (A : Mat2) : Int :=
  A.m00 + A.m11

/-- Frobenius inner product `⟨A, B⟩_F = Tr(A^T B)`. -/
def frobInner (A B : Mat2) : Int :=
  A.m00 * B.m00 + A.m01 * B.m01 + A.m10 * B.m10 + A.m11 * B.m11

/-- Frobenius squared norm `‖A‖_F^2 = ⟨A, A⟩_F`. -/
def frobNormSq (A : Mat2) : Int :=
  frobInner A A

/-- Twice the skew-symmetric part `2 * skew(A) = A - A^T`. -/
def skew2 (A : Mat2) : Mat2 :=
  sub A (transpose A)

/-- Twice the symmetric part `2 * sym(A) = A + A^T`. -/
def sym2 (A : Mat2) : Mat2 :=
  add A (transpose A)

/-- Matrix commutator `[A, B] = A B - B A`. -/
def commutator (A B : Mat2) : Mat2 :=
  sub (mul A B) (mul B A)

@[ext]
theorem ext {A B : Mat2}
    (h00 : A.m00 = B.m00) (h01 : A.m01 = B.m01)
    (h10 : A.m10 = B.m10) (h11 : A.m11 = B.m11) : A = B := by
  cases A
  cases B
  simp_all

theorem transpose_transpose (A : Mat2) : transpose (transpose A) = A := by
  cases A
  rfl

theorem skew2_transpose (A : Mat2) : transpose (skew2 A) = neg (skew2 A) := by
  cases A
  ext <;> simp [skew2, transpose, sub, neg] <;> omega

theorem sym2_transpose (A : Mat2) : transpose (sym2 A) = sym2 A := by
  cases A
  ext <;> simp [sym2, transpose, add] <;> omega

theorem skew2_sym2_eq_zero (A : Mat2) : skew2 (sym2 A) = zero := by
  cases A
  ext <;> simp [skew2, sym2, transpose, sub, add, zero] <;> omega

end Mat2

/-- Exact Gaussian integer / complex pair `re + i * im` for unitary and Kraus gauge proofs. -/
structure ComplexInt where
  re : Int
  im : Int
  deriving Repr, DecidableEq

namespace ComplexInt

def zero : ComplexInt := ⟨0, 0⟩
def one : ComplexInt := ⟨1, 0⟩

def conj (z : ComplexInt) : ComplexInt :=
  ⟨z.re, -z.im⟩

def add (z w : ComplexInt) : ComplexInt :=
  ⟨z.re + w.re, z.im + w.im⟩

def sub (z w : ComplexInt) : ComplexInt :=
  ⟨z.re - w.re, z.im - w.im⟩

def neg (z : ComplexInt) : ComplexInt :=
  ⟨-z.re, -z.im⟩

def mul (z w : ComplexInt) : ComplexInt :=
  ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩

def normSq (z : ComplexInt) : Int :=
  z.re * z.re + z.im * z.im

@[ext]
theorem ext {z w : ComplexInt} (hr : z.re = w.re) (hi : z.im = w.im) : z = w := by
  cases z
  cases w
  simp_all

theorem conj_conj (z : ComplexInt) : conj (conj z) = z := by
  cases z
  ext <;> simp [conj]

theorem conj_add (z w : ComplexInt) : conj (add z w) = add (conj z) (conj w) := by
  cases z
  cases w
  ext <;> simp [conj, add] <;> omega

theorem conj_sub (z w : ComplexInt) : conj (sub z w) = sub (conj z) (conj w) := by
  cases z
  cases w
  ext <;> simp [conj, sub] <;> omega

end ComplexInt

/-- Finite summation `∑_{i < n} f(i)` over `Fin n → Int`. -/
def finSum : (n : Nat) → (Fin n → Int) → Int
  | 0, _ => 0
  | n + 1, f => finSum n (fun i => f i.castSucc) + f ⟨n, Nat.lt_succ_self n⟩

/-- Scalar multiplication distributes over `finSum`: `∑_i (c * f(i)) = c * (∑_i f(i))`. -/
theorem finSum_const_mul (n : Nat) (c : Int) (f : Fin n → Int) :
    finSum n (fun i => c * f i) = c * finSum n f := by
  induction n with
  | zero =>
    simp [finSum]
  | succ k ih =>
    simp only [finSum]
    rw [ih, Int.mul_add]

/-- Pointwise equality implies equal finite sums. -/
theorem finSum_congr {n : Nat} {f g : Fin n → Int} (h : ∀ i : Fin n, f i = g i) :
    finSum n f = finSum n g := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [finSum]
    rw [ih (fun i => h i.castSucc), h ⟨k, Nat.lt_succ_self k⟩]

/-- Non-negative terms sum to a non-negative total. -/
theorem finSum_nonneg (n : Nat) (f : Fin n → Int) (h : ∀ i : Fin n, 0 ≤ f i) :
    0 ≤ finSum n f := by
  induction n with
  | zero =>
    simp [finSum]
  | succ k ih =>
    simp only [finSum]
    have h1 : 0 ≤ finSum k (fun i => f i.castSucc) := ih (fun i => f i.castSucc) (fun i => h i.castSucc)
    have h2 : 0 ≤ f ⟨k, Nat.lt_succ_self k⟩ := h ⟨k, Nat.lt_succ_self k⟩
    omega

/-- Additivity of finite summation: `∑_i (f(i) + g(i)) = (∑_i f(i)) + (∑_i g(i))`. -/
theorem finSum_add (n : Nat) (f g : Fin n → Int) :
    finSum n (fun i => f i + g i) = finSum n f + finSum n g := by
  induction n with
  | zero =>
    simp [finSum]
  | succ k ih =>
    simp only [finSum]
    rw [ih]
    omega

end QuantumDLBlocksLean
