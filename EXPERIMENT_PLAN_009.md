# Experiment Plan 009: Overcomplete Two-Beam Quantum Interference (`OvercompleteTwoBeamInterference`) — Welch-Bounded $2d$ Tight Frames in Both Interfering Beams

## 1. Executive Summary & Context

Our Wave-2 FineWeb 100M experiments uncovered a striking complementary tradeoff between two backbone MLP designs:
1. **`OvercompleteFrameInterference` (Plan 006 — Single-Beam $M = 2d = 2048$ Welch-Bounded Tight Frame)**:
   - Achieved **much faster early-epoch feature discovery** at Epoch 0.5 (`50M` tokens): `w2_gpu_frame256` reached **`4.9099` Val CE (`135.6` PPL)** vs `5.1701` Val CE (`175.9` PPL) for `hyp_gpu_exp03` (`-0.2602` nats ahead at 50M tokens!).
   - Why? Because packing $M = 2d$ unit vectors near the Welch bound ($\mu_{\text{rms}} = \sqrt{\frac{M-d}{d(M-1)}}$) gives isotropic $2\times$ overcomplete coverage of $\mathbb{S}^{d-1}$ from step 0.
   - However, because `OvercompleteFrameInterference` used only a **single beam** $(W_{\text{up}}\psi)^{\odot 2} \odot \mathrm{sign}(W_{\text{up}}\psi)$, it lacked cross-beam multiplicative interactions and finished `~0.02` nats behind the Two-Beam MLP at Epoch 3.0 (`4.2344` vs `4.2181` CE).
2. **`TwoBeamQuantumInterference` (`d_ff = 1536x2` — Two-Beam Bilinear Interference)**:
   - Computes $(W_{\text{beam1}}\psi)^{\odot 2} \odot \mathrm{SiLU}(W_{\text{beam2}}\psi)$, which couples two independent projections per channel (degree-3 cross-subspace interference) and won at Epoch 3.0 (`4.1725` CE at `D_seed=384`, `4.1527` CE at `D_seed=1024`), but lacked Welch-bounded $2d$ tight-frame initialization, tight-frame Gram regularization, and paired-beam orthogonality regularization.

### The Synthesis: `OvercompleteTwoBeamInterference`
What if **both** interfering beams $W_{\text{beam1}}, W_{\text{beam2}} \in \mathbb{R}^{M \times d}$ ($M = 2d$) are constructed as **Welch-bounded Overcomplete Tight Frames** whose paired rows $(w_{1,m}, w_{2,m})$ are **mutually orthogonal** ($\langle \hat{w}_{1,m}, \hat{w}_{2,m}\rangle = 0$ for all $m \in \{1, \dots, M\}$)?
- When $\langle \hat{w}_{1,m}, \hat{w}_{2,m}\rangle = 0$, every channel $m \in \{1, \dots, 2d\}$ measures the coincidence between a **quantum intensity** $|\langle \hat{w}_{1,m}, \psi\rangle|^2$ in direction $\hat{w}_{1,m}$ and a **coherent gate amplitude** $\mathrm{SiLU}(\langle \hat{w}_{2,m}, \psi\rangle)$ in a strictly orthogonal direction $\hat{w}_{2,m} \perp \hat{w}_{1,m}$, while both $W_{\text{beam1}}$ and $W_{\text{beam2}}$ individually form Welch-bounded $2d$ tight frames over $\mathbb{R}^d$!

This document specifies **Experiment Plan 009**, which implements and verifies `OvercompleteTwoBeamInterference`.

---

## 2. Observable Hypotheses & Quantitative Predictions

### Hypothesis 1 (H1 — Simultaneous Welch-Bound Tightness & Zero Paired-Beam Coherence via Orthogonal-Pair Tight-Frame Optimization):
Let $d = 64$ and $M = 2d = 128$. Consider constructing a pair of overcomplete frames $(W_1, W_2) \in \mathbb{R}^{M \times d} \times \mathbb{R}^{M \times d}$ with unit-norm rows ($\|\hat{w}_{1,m}\|_2 = \|\hat{w}_{2,m}\|_2 = 1$) via `optimize_orthogonal_pair_tight_frames(M, d, steps=200, lr=0.08, seed=42)` minimizing the unified objective:
$$\mathcal{L}_{\text{2beam}}(W_1, W_2) = \frac{1}{2}\big(\mathcal{L}_{\text{frame}}(W_1) + \mathcal{L}_{\text{frame}}(W_2)\big) + \frac{1}{M} \sum_{m=1}^M \langle \hat{w}_{1,m}, \hat{w}_{2,m} \rangle^2$$
where $\mathcal{L}_{\text{frame}}(W) = \frac{\|\hat{W}^\top \hat{W} - \frac{M}{d} I_d\|_F^2}{(M/d)^2 d}$ is the scale-normalized tight-frame Gram penalty from Plan 006.
- **Observable Prediction (H1)**:
  1. At `steps = 200` (and even at `steps = 150`), both $W_1$ and $W_2$ will converge to Welch-bounded tight frames with `beam1_tight_frame_err <= 0.03`, `beam2_tight_frame_err <= 0.03`, and `beam1_rms_coherence <= 1.02 * welch_bound`, `beam2_rms_coherence <= 1.02 * welch_bound` ($\mu_{\text{Welch}} = \sqrt{\frac{64}{64 \times 127}} \approx 0.08874$), while maintaining combined `naive_iso_err >= 0.95` relative to $I_d$.
  2. Simultaneously, the RMS paired-beam coherence $\mu_{\text{pair}}(W_1, W_2) = \sqrt{\frac{1}{M}\sum_{m=1}^M \langle \hat{w}_{1,m}, \hat{w}_{2,m}\rangle^2}$ (`paired_beam_rms_coherence`) will drop from $\approx 1/\sqrt{d} = 0.125$ (random independent frames) to **`<= 0.005`** (a **`25x` reduction** in redundant self-overlap between paired beams).
  3. **Multi-Subspace Degree-3 Observable Regression Task**:
     Generate $N_{\text{train}} = 1024$ and $N_{\text{test}} = 512$ isotropic unit vectors $\psi \in \mathbb{S}^{63}$ (`torch.manual_seed(42)`). Construct a ground-truth degree-3 target observable $y(\psi) = \frac{1}{\sqrt{3}} \sum_{k=1}^3 \big((Q_{1,k} \psi \sqrt{d})^{\odot 2} \odot \mathrm{SiLU}(Q_{2,k} \psi \sqrt{d})\big) \in \mathbb{R}^{64}$, where $(Q_{1,k}, Q_{2,k}) \in \mathbb{R}^{64 \times 64} \times \mathbb{R}^{64 \times 64}$ are 3 independent orthogonal-basis pairs with $Q_{2,k} = \mathrm{roll}(Q_{1,k}, 1, \text{dims}=0)$.
     For any feature map $h(\psi) \in \mathbb{R}^M$, define the **feature covariance effective rank** as $R_{\text{eff}}(C_h) = \exp\!\left(-\sum_{i=1}^M p_i \ln(p_i + \varepsilon)\right)$ where $\lambda_i \ge 0$ are the eigenvalues of the centered empirical covariance $C_h = \frac{1}{N_{\text{train}}-1} \tilde{H}^\top \tilde{H} \in \mathbb{R}^{M \times M}$ and $p_i = \lambda_i / \sum_j \lambda_j$, and fit a closed-form ridge readout $W^* = (\tilde{H}_{\text{train}}^\top \tilde{H}_{\text{train}} + \lambda I_M)^{-1} \tilde{H}_{\text{train}}^\top \tilde{Y}_{\text{train}}$ with $\lambda = 10^{-3}$:
     - **Baseline 1 (`ortho_two_beam_1d`, $M = d = 64$)**: $h_{\text{1d}}(\psi) = (\hat{W}_{1,\text{ortho}} \psi \sqrt{d})^{\odot 2} \odot \mathrm{SiLU}(\hat{W}_{2,\text{ortho}} \psi \sqrt{d}) \in \mathbb{R}^{64}$ where $\hat{W}_{1,\text{ortho}}, \hat{W}_{2,\text{ortho}} \in \mathbb{R}^{64 \times 64}$ are orthonormal matrices with $\hat{W}_{2,\text{ortho}} = \mathrm{roll}(\hat{W}_{1,\text{ortho}}, 1, \text{dims}=0)$.
     - **Baseline 2 (`single_beam_overcomplete_2d`, $M = 2d = 128$)**: $h_{\text{single}}(\psi) = (\hat{W}_{\text{up}} \psi \sqrt{d})^{\odot 2} \odot \mathrm{sign}(\hat{W}_{\text{up}} \psi) \in \mathbb{R}^{128}$ where $\hat{W}_{\text{up}} \in \mathbb{R}^{128 \times 64}$ is a Welch-bounded tight frame.
     - **Proposed (`overcomplete_two_beam_2d`, $M = 2d = 128$)**: $h_{\text{2beam}}(\psi) = (\hat{W}_1 \psi \sqrt{d})^{\odot 2} \odot \mathrm{SiLU}(\hat{W}_2 \psi \sqrt{d}) \in \mathbb{R}^{128}$ where $(\hat{W}_1, \hat{W}_2)$ are optimized via `optimize_orthogonal_pair_tight_frames(128, 64, steps=200)`.
     `overcomplete_two_beam_2d` will achieve at least **`1.35x` higher feature covariance effective rank** ($R_{\text{eff}}(C_{h_{\text{2beam}}}) \ge 1.35 \times R_{\text{eff}}(C_{h_{\text{1d}}})$) and **`2.0x` lower held-out test MSE** than `ortho_two_beam_1d`, and at least **`2.0x` lower held-out test MSE** than `single_beam_overcomplete_2d`.

### Hypothesis 2 (H2 — Faster Early Convergence + Superior Final Validation CE in `QuantumBornGPT`):
In a controlled 3-arm `QuantumBornGPT` language modeling benchmark (`vocab_size = 512`, `embed_dim = 64`, `seed_dim = 32`, `num_heads = 4`, `num_layers = 2`, `block_size = 64`, `readout_mode = "tied_stiefel"`, `200` training steps at `batch_size = 16`, trained with AdamW at `lr = 3e-3` on `200` training sequences and `56` held-out validation sequences of length `65` generated deterministically with `torch.manual_seed(42)` from a frozen teacher `QuantumBornGPT(vocab_size=512, embed_dim=64, seed_dim=64, num_heads=4, num_layers=2, use_two_beam_mlp=True, povm_temperature=0.015)`):
- **Arm A (`single_beam_overcomplete_2d`)**: Backbone blocks `blk.mlp` replaced with `OvercompleteFrameInterference(embed_dim=64, frame_expansion=2.0)` + auxiliary mean `frame_penalty()` across layers with weight $\lambda_{\text{frame}} = 0.02$.
- **Arm B (`standard_two_beam_15d`)**: Backbone blocks use standard `TwoBeamQuantumInterference(embed_dim=64, hidden_dim=96)` (`1.5d`, no frame penalty).
- **Arm C (`overcomplete_two_beam_2d`)**: Backbone blocks `blk.mlp` replaced with `OvercompleteTwoBeamInterference(embed_dim=64, frame_expansion=2.0)` + auxiliary mean `frame_penalty()` across layers with weight $\lambda_{\text{2beam}} = 0.02$.
- **Observable Prediction (H2)**:
  At step 200, **Arm C (`overcomplete_two_beam_2d`)** will achieve strictly lower validation cross-entropy than both **Arm A** (by at least **`0.08` nats**, $\text{Val CE}_C \le \text{Val CE}_A - 0.08$) and **Arm B** (by at least **`0.08` nats**, $\text{Val CE}_C \le \text{Val CE}_B - 0.08$), while maintaining layer-mean `paired_beam_rms_coherence <= 0.08` and layer-mean `naive_iso_err >= 0.50`.

---

## 3. Planned Code & Architectural Changes

### 3.1 PyTorch Module (`src/qdlblocks/layers/overcomplete_two_beam.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_two_beam.py` containing:

1. `two_beam_frame_penalty(W_beam1: torch.Tensor, W_beam2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor`:
   - Given `W_beam1, W_beam2` of shape `(M, d)`:
   - Computes `p1 = tight_frame_penalty(W_beam1, eps=eps)` and `p2 = tight_frame_penalty(W_beam2, eps=eps)` (from `..layers.overcomplete_frame`).
   - Row-normalizes `U1 = l2_normalize(W_beam1, dim=-1, eps=eps)` and `U2 = l2_normalize(W_beam2, dim=-1, eps=eps)`.
   - Computes paired row-wise inner-product penalty `p_pair = ((U1 * U2).sum(dim=-1) ** 2).mean()`.
   - Returns `0.5 * (p1 + p2) + p_pair`.

2. `optimize_orthogonal_pair_tight_frames(num_vectors: int, dim: int, steps: int = 200, lr: float = 0.08, seed: int = 42, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]`:
   - Uses `torch.enable_grad()` and a local `torch.Generator(device=device).manual_seed(seed)`.
   - Initializes `W1` and `W2` of shape `(num_vectors, dim)` with row-normalized Gaussian vectors, explicitly projecting `W2` rows orthogonal to `W1` rows at step 0 (`W2 = l2_normalize(W2 - (W2 * W1).sum(dim=-1, keepdim=True) * W1, dim=-1)`).
   - If `num_vectors <= dim`, orthonormalizes `W1` via QR decomposition and sets `W2 = torch.roll(W1, shifts=1, dims=0)` (when `num_vectors >= 2`) so both are exact orthonormal frames with zero paired coherence.
   - If `num_vectors > dim` and `steps > 0`, runs `steps` iterations of Adam (`lr=lr`) minimizing `two_beam_frame_penalty(W1, W2)`.
   - Returns `( l2_normalize(W1.detach(), dim=-1), l2_normalize(W2.detach(), dim=-1) )`.

3. `OvercompleteTwoBeamInterference(nn.Module)`:
   - Constructor signature:
     ```python
     class OvercompleteTwoBeamInterference(nn.Module):
         def __init__(
             self,
             embed_dim: int,
             frame_expansion: float = 2.0,
             target_iso_scale: float = 1.25,
             init_steps: int = 200,
             seed: int = 42,
             eps: float = 1e-12,
             device: torch.device | str | None = None,
         ) -> None:
     ```
   - Sets `self.frame_dim = int(round(embed_dim * frame_expansion))`.
   - Initializes `(W1_init, W2_init) = optimize_orthogonal_pair_tight_frames(self.frame_dim, embed_dim, steps=init_steps, seed=seed, device=device or "cpu")`.
   - Parameters:
     - `self.W_beam1 = nn.Parameter(W1_init * target_iso_scale)` of shape `(frame_dim, embed_dim)`
     - `self.W_beam2 = nn.Parameter(W2_init * target_iso_scale)` of shape `(frame_dim, embed_dim)`
     - `self.W_out = nn.Parameter(W1_init.t().contiguous() * (target_iso_scale * math.sqrt(embed_dim / self.frame_dim)))` of shape `(embed_dim, frame_dim)`
     - `self.post_norm = nn.LayerNorm(embed_dim, eps=1e-6)`
   - Method `forward(self, psi: torch.Tensor) -> torch.Tensor`:
     - Accepts `psi` of shape `(..., embed_dim)` (both 2D `(B, d)` and 3D `(B, T, d)`).
     - Row-normalizes `U1 = l2_normalize(self.W_beam1, dim=-1, eps=self.eps).to(psi.dtype)` and `U2 = l2_normalize(self.W_beam2, dim=-1, eps=self.eps).to(psi.dtype)`.
     - Computes `z1 = F.linear(psi, U1) * math.sqrt(float(self.embed_dim))` and `z2 = F.linear(psi, U2) * math.sqrt(float(self.embed_dim))`.
     - Computes degree-3 cross-beam interference:
       `interf = (z1 * z1) * F.silu(z2)`
     - Projects back to `embed_dim` via `out = F.linear(interf, self.W_out.to(psi.dtype))` and returns `l2_normalize(self.post_norm(out.float()).to(psi.dtype), dim=-1, eps=self.eps)`.
   - Method `frame_penalty(self) -> torch.Tensor`:
     - Returns `two_beam_frame_penalty(self.W_beam1, self.W_beam2, eps=self.eps)`.
   - Method `coherence_metrics(self) -> dict[str, float]`:
     - Computes `m1 = frame_coherence_metrics(self.W_beam1, eps=self.eps)` and `m2 = frame_coherence_metrics(self.W_beam2, eps=self.eps)` (from `..layers.overcomplete_frame`, where `welch_bound = sqrt((M - d) / (d * (M - 1)))` if `M > d` and `M > 1` else `0.0`, `tight_frame_err = ||U^T U - (M/d) I_d||_F / ((M/d) sqrt(d))`, `naive_iso_err = ||W^T W - I_d||_F / sqrt(d)`, and `rms_coherence = sqrt(sum_{i != j} (U U^T)_{ij}^2 / (M (M - 1)))` if `M > 1` else `0.0`).
     - Computes paired row cosine `U1 = l2_normalize(self.W_beam1.detach().float(), dim=-1, eps=self.eps)`, `U2 = l2_normalize(self.W_beam2.detach().float(), dim=-1, eps=self.eps)`, `paired_rms = float(torch.sqrt(((U1 * U2).sum(dim=-1) ** 2).mean()).item())`.
     - Returns:
       - `"beam1_tight_frame_err"`: `m1["tight_frame_err"]`
       - `"beam2_tight_frame_err"`: `m2["tight_frame_err"]`
       - `"beam1_rms_coherence"`: `m1["rms_coherence"]`
       - `"beam2_rms_coherence"`: `m2["rms_coherence"]`
       - `"welch_bound"`: `m1["welch_bound"]`
       - `"paired_beam_rms_coherence"`: `paired_rms`
       - `"naive_iso_err"`: `0.5 * (m1["naive_iso_err"] + m2["naive_iso_err"])`

### 3.2 Pure JAX Parity (`src/qdlblocks_jax/overcomplete_two_beam.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/overcomplete_two_beam.py` containing:
- `two_beam_frame_penalty_jax(W_beam1: jax.Array, W_beam2: jax.Array, eps: float = 1e-12) -> jax.Array`:
  - Exact JAX equivalent: `0.5 * (tight_frame_penalty_jax(W_beam1, eps=eps) + tight_frame_penalty_jax(W_beam2, eps=eps)) + jnp.mean(jnp.sum(U1 * U2, axis=-1) ** 2)`.
- `overcomplete_two_beam_metrics_jax(W_beam1: jax.Array, W_beam2: jax.Array, eps: float = 1e-12) -> dict[str, jax.Array]`:
  - Returns the exact same 7 keys (`"beam1_tight_frame_err"`, `"beam2_tight_frame_err"`, `"beam1_rms_coherence"`, `"beam2_rms_coherence"`, `"welch_bound"`, `"paired_beam_rms_coherence"`, `"naive_iso_err"`) using `frame_coherence_metrics_jax` on `W_beam1` and `W_beam2`.

### 3.3 CLI Integration (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`) & Verification Script (`examples/19_hypothesis_overcomplete_two_beam.py`)
1. In `/home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py`:
   - Add `--overcomplete-two-beam` CLI flag (`action="store_true"`). When enabled on the GPU backend, replace `blk.mlp` in each block with `OvercompleteTwoBeamInterference(embed_dim=1024, frame_expansion=2.0, seed=42 + 10 * l_idx, device=device).to(device)`. Because `compute_step_loss` already checks `hasattr(blk.mlp, "frame_penalty")` and adds `0.02 * mean(frame_penalty)`, `OvercompleteTwoBeamInterference` automatically receives the `0.02` auxiliary frame penalty! (On the TPU backend, `--overcomplete-two-beam` is a no-op / ignored just like `--overcomplete-frame`).
2. Create `/home/festivity/agy/QuantumDLBlocks/examples/19_hypothesis_overcomplete_two_beam.py` to run H1 and H2 and save `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_009.json` with top-level keys `"experiment_1_h1"` and `"experiment_2_h2"`.

---

## 4. Verification & Unit Testing Plan (`tests/test_hypothesis_009_overcomplete_two_beam.py`)

Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_009_overcomplete_two_beam.py` with 4 unit tests:
1. `test_orthogonal_pair_tight_frame_optimization()`:
   - Verifies that `optimize_orthogonal_pair_tight_frames(128, 64, steps=150)` achieves `beam1_tight_frame_err <= 0.03`, `beam2_tight_frame_err <= 0.03`, and `paired_beam_rms_coherence <= 0.005`.
2. `test_overcomplete_two_beam_forward_unit_norm_and_gradients()`:
   - Verifies that `OvercompleteTwoBeamInterference(embed_dim=32, frame_expansion=2.0)` outputs unit-norm vectors on $\mathbb{S}^{31}$ for both 2D `(4, 32)` and 3D `(2, 8, 32)` inputs and produces finite, non-zero gradients for `W_beam1`, `W_beam2`, and `W_out`.
3. `test_overcomplete_two_beam_breaks_antipodal_symmetry()`:
   - Verifies that `mlp(-psi)` is NOT equal to `mlp(psi)` or `-mlp(psi)` (`torch.norm(mlp(psi) - mlp(-psi)) > 0.1` and `torch.norm(mlp(psi) + mlp(-psi)) > 0.1`), confirming mixed even/odd spherical harmonic expressivity.
4. `test_pytorch_jax_two_beam_penalty_equivalence()`:
   - Verifies that `two_beam_frame_penalty` in PyTorch and `two_beam_frame_penalty_jax` in JAX agree within `1e-5`, and that all 7 keys of `coherence_metrics()` and `overcomplete_two_beam_metrics_jax()` agree within `1e-5`.
