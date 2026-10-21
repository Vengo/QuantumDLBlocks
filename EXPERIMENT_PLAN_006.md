# Experiment Plan 006: Overcomplete Tight Frames, Welch-Bound Coherence, and Controlled Isometry Relaxation (`EXPERIMENT_PLAN_006.md`)

## 1. Context & Empirical Motivation

Across Experiment Plans `001`–`005`, a central assumption was that keeping the Stiefel manifold isometry error
$$\text{IsoErr}(W) = \frac{\|W^\top W - I_c\|_F}{\sqrt{c}} \quad (\text{or } \frac{\|W W^\top - I_r\|_F}{\sqrt{r}} \text{ when } r < c)$$
as close to zero as possible is always beneficial. However, our live 10-session Google Colab FineWeb 100M fleet just revealed a striking empirical counterexample in the controlled head-to-head comparison between `gpu_exp01` and `gpu_exp02` on identical NVIDIA RTX PRO 6000 Blackwell (`G4`) hardware (`3` epochs, `WSD`, `MTP=3`, `mtp_stride=4`, decoupled temperatures, readout stop-gradient):

| Live Colab Run | Optimizer & Isometry Regime | `IsoErr` @ 50M tok | $R_{\text{eff}}(\rho_{\text{seed}})$ @ 50M tok | Val CE @ 50M tok | `IsoErr` @ 100M tok | $R_{\text{eff}}(\rho_{\text{seed}})$ @ 100M tok | Val CE @ 100M tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gpu_exp01`** | `TangentStiefelMuon` (Strict Isometry) | **`0.0004`** | `4.9` | `5.3668` (`214.2` PPL) | **`0.0004`** | `7.8` | `4.9605` (`142.7` PPL) |
| **`gpu_exp02`** | `StiefelMuon` (Relaxed Isometry) | **`1.2799`** | **`8.8`** | **`5.0074`** (`149.5` PPL) | **`1.1519`** | **`14.7`** | **`4.7640`** (`117.2` PPL) |

Pinning `IsoErr` to `0.0004` (`gpu_exp01`) **degraded** validation cross-entropy by **`+0.3594` nats** at 50M tokens and **`+0.1965` nats** at 100M tokens relative to allowing `IsoErr` to float around `1.15–1.28` (`gpu_exp02`), while cutting the effective quantum state rank $R_{\text{eff}}(\rho_{\text{seed}})$ in half (`7.8` vs `14.7`)! Moreover, `gpu_exp02` continued improving cleanly through `250M` tokens (`Val CE = 4.4748`, `R_eff = 23.0`).

### Physical Explanation: Overcomplete Bases (Tight Frames) in High-Dimensional Hilbert Spaces
In quantum mechanics and harmonic analysis, physicists frequently compute with **overcomplete bases** (coherent states, Positive Operator-Valued Measures / POVMs, and **Tight Frames** $\{\phi_i\}_{i=1}^M \subset \mathbb{R}^d$ where $M > d$, e.g. $M = 2d$ vectors packed into a $d$-dimensional Hilbert space):
1. **Welch Bound & Almost-Orthogonal Packing in High Dimensions**:
   When packing $M = \rho d$ unit-norm vectors ($\|\phi_i\|_2 = 1$, redundancy ratio $\rho = M/d = 2$) into $\mathbb{R}^d$, they cannot be mutually orthogonal. By the **Welch Bound**, the minimum achievable root-mean-square off-diagonal inner product (coherence) is:
   $$\mu_{\text{Welch}}(M, d) = \sqrt{\frac{M - d}{d(M - 1)}} \approx \sqrt{\frac{\rho - 1}{\rho \, d}}$$
   For $\rho = 2$ ($M = 2d$ vectors) in $d = 1024$, $\mu_{\text{Welch}} \approx 0.0221$: you can pack **twice as many unit vectors as dimensions** with pairwise overlaps of only $\sim 2.2\%$!
2. **Why Overcomplete Tight Frames Have $\mathcal{O}(1)$ Naive `IsoErr` Without Breaking the Algorithm**:
   An optimal overcomplete basis $W \in \mathbb{R}^{M \times d}$ ($M = 2d$, rows $\phi_i^\top$) satisfies the **Parseval Tight-Frame Resolution of the Identity**:
   $$W^\top W = \frac{M}{d} I_d = \rho I_d \quad (\text{with } \rho = 2)$$
   If one measures the naive Stiefel error against $I_d$, $\|W^\top W - I_d\|_F / \sqrt{d} = |\rho - 1| = 1.0$. Thus an overcomplete tight frame *naturally* exhibits `IsoErr ~ 1.0` while remaining a mathematically exact, energy-preserving resolution of the identity up to scale $\rho$!
3. **Why Quadratic Born Interference Benefits from $M = 2d$ Overcomplete Bases**:
   Because Born-rule layers compute elementwise quadratic probabilities/amplitudes $p_i = |\langle \psi | \phi_i \rangle|^2$, projecting $|\psi\rangle \in \mathbb{R}^d$ onto an orthogonal basis ($M = d$, `IsoErr = 0`) only probes $d$ commuting projectors. Projecting onto an **overcomplete $M = 2d$ frame** with small controlled off-diagonal coherence probes $2d$ non-commuting rank-1 POVM effects in parallel, nearly doubling the rank of the quadratic feature map!

---

## 2. Observable Hypotheses & Quantitative Predictions

### Hypothesis 1 (H1 — $M = 2d$ Overcomplete Tight-Frame Packing & Welch-Bound Coherence)
- **Claim**: Packing $M = 2d$ unit-norm vectors into $\mathbb{R}^d$ ($W \in \mathbb{R}^{2d \times d}$ with unit-norm rows) by minimizing the **Benedetto–Fickus Tight-Frame Potential**
  $$\mathcal{L}_{\text{frame}}(W) = \frac{1}{d} \left\| W^\top W - \frac{M}{d} I_d \right\|_F^2$$
  converges to an overcomplete tight frame whose:
  1. Naive Stiefel isometry error relative to $I_d$, $\text{IsoErr}(W) = \|W^\top W - I_d\|_F / \sqrt{d}$, is high ($\approx 1.0$, specifically $\in [0.85, 1.15]$ for $M = 2d$),
  2. **Tight-Frame Resolution Error** $\text{FrameErr}(W) = \|W^\top W - (M/d) I_d\|_F / \sqrt{d}$ is small ($< 0.05$), and
  3. **RMS Off-Diagonal Coherence** $\mu_{\text{rms}}(W) = \sqrt{\frac{1}{M(M-1)} \sum_{i \neq j} \langle \phi_i, \phi_j \rangle^2}$ is within $1.10\times$ of the theoretical Welch bound $\mu_{\text{Welch}}(2d, d) = \sqrt{\frac{2d - d}{d(2d - 1)}} = \frac{1}{\sqrt{2d - 1}}$.
- **Falsifiable Prediction**:
  Define the **effective feature covariance rank** $R_\sigma(\tilde{F})$ of the centered quadratic Born feature matrix $\tilde{F} = F - \frac{1}{N}\mathbf{1}\mathbf{1}^\top F \in \mathbb{R}^{N \times M}$ (where $F_{n,:} = (W \psi_n)^{\odot 2}$) via the Shannon entropy of its normalized singular values $\sigma_1 \ge \dots \ge \sigma_{\min(N,M)} > 0$:
  $$R_\sigma(\tilde{F}) = \exp\!\left(-\sum_i p_i \log p_i\right), \quad p_i = \frac{\sigma_i(\tilde{F})}{\sum_j \sigma_j(\tilde{F})}$$
  When used as the interference feature map $f_W(\psi) = (W \psi)^{\odot 2}$ on $d = 64$ ($M = 128 = 2d$) to fit a rank-32 quadratic quantum observable $y(\psi) = \psi^\top O \psi$ via ridge regression ($\alpha = 10^{-4}$, trained on $N_{\text{train}} = 1024$ unit states and evaluated on a held-out test set of $N_{\text{test}} = 256$ unit states), the $M = 2d$ overcomplete tight frame (`IsoErr ~ 1.0`, `FrameErr < 0.05`) achieves a **$\ge 1.6\times$ higher effective feature covariance rank $R_\sigma(\tilde{F})$** and a **$\ge 2.0\times$ lower held-out test MSE** than an exact $M = d$ orthogonal Stiefel basis (`IsoErr < 1e-4`).

### Hypothesis 2 (H2 — Controlled Isometry Error & Overcomplete $2d$ Interference in `QuantumBornGPT`)
- **Claim**: In `QuantumBornGPT` language modeling, strict Stiefel orthogonality (`IsoErr -> 0`) overly restricts singular-value gain and interference basis cardinality. Replacing the strictly orthogonal constraint with an **Overcomplete $2d$ Frame Interference Layer** ($M = 2d$ frame vectors in $\mathbb{R}^d$) coupled with **Controlled Tight-Frame Regularization** (allowing controlled `IsoErr` in $[0.4, 2.5]$ relative to $I_d$ while bounding `tight_frame_err`) outperforms both:
  - **Arm A (Strict Orthogonal Stiefel, $M = d$, `IsoErr < 0.01`)**, and
  - **Arm B (Strict Orthogonal Stiefel, $M = 2d$, `TangentStiefelMuon` forcing `IsoErr < 0.01`)**.
- **Falsifiable Prediction**:
  In a controlled 200-step synthetic Zipfian language modeling benchmark ($d = 64$, $D_{\text{seed}} = 32$, $H = 4$, $L = 2$, $T = 64$, $V = 256$, fixed seed `42`):
  - **Arm C (Overcomplete $2d$ Frame + Controlled Isometry Relaxation)** achieves a final validation cross-entropy **`val_ce` at least `0.08` nats lower** than Arm A (strict orthogonal $M = d$) and **at least `0.05` nats lower** than Arm B (strict Stiefel $M = 2d$), while achieving a strictly higher final seed effective quantum rank $R_{\text{eff}}(\rho_{\text{seed}})$.

---

## 3. Planned Architectural & Code Changes

### 3.1 Overcomplete Tight-Frame Metrics & Layer in PyTorch (`src/qdlblocks/layers/overcomplete_frame.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/overcomplete_frame.py` (and export all 5 symbols below in `src/qdlblocks/__init__.py`) providing:

1. `welch_bound_coherence(num_vectors: int, dim: int) -> float`:
   - If `dim < 1`: raise `ValueError(f"dim must be >= 1, got {dim}")`.
   - If `num_vectors <= dim`: return `0.0`.
   - If `num_vectors > dim`: return `math.sqrt((num_vectors - dim) / float(dim * (num_vectors - 1)))`.

2. `tight_frame_penalty(w: torch.Tensor, target_iso_scale: float = 1.25) -> torch.Tensor`:
   - Accepts a 2D tensor `w`. If `w.ndim != 2`, raise `ValueError`.
   - Orient `wf = w.float()` so that rows are the $M$ frame vectors and columns are the ambient dimension $d$: if `wf.shape[0] < wf.shape[1]`, set `wf = wf.transpose(0, 1)`. Let $M, d = \text{wf.shape}$.
   - Target Gram diagonal scale: $\gamma = \text{target\_iso\_scale}^2 \cdot (M / float(d))$.
   - Compute Gram matrix $G_{\text{dim}} = \text{wf}^\top \text{wf} \in \mathbb{R}^{d \times d}$ and return the differentiable scalar tensor:
     $$\mathcal{R}_{\text{frame}}(w) = \frac{1}{d} \left\| G_{\text{dim}} - \gamma I_d \right\|_F^2$$

3. `frame_coherence_metrics(w: torch.Tensor, eps: float = 1e-6) -> dict[str, float]`:
   - Accepts 2D tensor `w`. Orient `wf = w.detach().float()` so $M \ge d$ (if `wf.shape[0] < wf.shape[1]`, transpose `wf = wf.transpose(0, 1)`). Let $M, d = \text{wf.shape}$.
   - Normalize rows to unit L2 norm: $\hat{W} = \text{l2\_normalize}(\text{wf}, \text{dim}=-1, \text{eps}=eps) \in \mathbb{R}^{M \times d}$.
   - Redundancy ratio: `redundancy_ratio = float(M) / float(d)`.
   - Naive Stiefel Isometry Error relative to $I_d$ on the unnormalized (or scaled) matrix `wf`:
     $$\text{iso\_err\_vs\_identity} = \frac{\| \text{wf}^\top \text{wf} - I_d \|_F}{\sqrt{d}}$$
   - Tight-Frame Resolution Error on the unit-row matrix $\hat{W}$:
     $$\text{tight\_frame\_err} = \frac{\| \hat{W}^\top \hat{W} - (M / float(d)) I_d \|_F}{\sqrt{d}}$$
   - If $M \le 1$: `rms_coherence = 0.0`, `max_coherence = 0.0`.
   - If $M > 1$: let $G = \hat{W} \hat{W}^\top \in \mathbb{R}^{M \times M}$ and $G_{\text{off}} = G - \text{diag}(\text{diag}(G))$. Then:
     $$\text{rms\_coherence} = \sqrt{\frac{\sum_{i \neq j} G_{\text{off}, ij}^2}{M(M - 1)}}, \quad \text{max\_coherence} = \max_{i \neq j} |G_{\text{off}, ij}|$$
   - Return `dict[str, float]` with exact keys: `"redundancy_ratio"`, `"iso_err_vs_identity"`, `"tight_frame_err"`, `"rms_coherence"`, `"max_coherence"`, `"welch_bound"` (where `"welch_bound"` is `welch_bound_coherence(M, d)`).

4. `optimize_overcomplete_tight_frame(num_vectors: int, dim: int, steps: int = 250, lr: float = 0.1, seed: int = 42, device: torch.device | str = "cpu") -> torch.Tensor`:
   - If `dim < 1` or `num_vectors < dim`: raise `ValueError(f"Require num_vectors >= dim >= 1, got num_vectors={num_vectors}, dim={dim}")`.
   - Must work safely inside or outside `torch.no_grad()` by using the closed-form analytical gradient (no autograd needed):
     - If `num_vectors == dim`: generate random Gaussian $Z \in \mathbb{R}^{d \times d}$ from `torch.Generator(device=device).manual_seed(seed)`, compute QR decomposition $Q, R = \text{torch.linalg.qr}(Z)$, sign-stabilize $Q$, and return $Q$.
     - If `num_vectors > dim`: initialize $W \in \mathbb{R}^{M \times d}$ from `torch.Generator(device=device).manual_seed(seed)` with `torch.randn((num_vectors, dim), generator=gen, device=device, dtype=torch.float32)` and normalize rows $W \leftarrow \text{l2\_normalize}(W, \text{dim}=-1)$.
     - Let $\rho = M / float(d)$ and $I_d = \text{torch.eye}(d, \text{device}=device, \text{dtype}=\text{torch.float32})$.
     - For `steps` iterations:
       $$G_{\text{err}} = W^\top W - \rho I_d, \quad \nabla_W = \frac{4}{d} W G_{\text{err}}, \quad W \leftarrow \text{l2\_normalize}(W - \text{lr} \cdot \nabla_W, \text{dim}=-1)$$
     - Return $W \in \mathbb{R}^{M \times d}$ (unit-norm rows).

5. `OvercompleteFrameInterference(nn.Module)`:
   - Constructor signature:
     ```python
     class OvercompleteFrameInterference(nn.Module):
         def __init__(
             self,
             embed_dim: int,
             frame_expansion: float = 2.0,
             target_iso_scale: float = 1.25,
             seed: int = 42,
             eps: float = 1e-6,
         ) -> None:
     ```
   - Sets `self.embed_dim = int(embed_dim)`, `self.frame_expansion = float(frame_expansion)`, `self.target_iso_scale = float(target_iso_scale)`, `self.eps = float(eps)`.
   - Sets `self.num_vectors = max(self.embed_dim, int(round(self.embed_dim * self.frame_expansion)))` ($M$).
   - Initializes two distinct unit-row tight frames using `seed` and `seed + 1`:
     - `wa_unit = optimize_overcomplete_tight_frame(self.num_vectors, self.embed_dim, steps=80, seed=seed)`
     - `wb_unit = optimize_overcomplete_tight_frame(self.num_vectors, self.embed_dim, steps=80, seed=seed + 1)`
   - Registers parameters with names matching `TwoBeamQuantumInterference` (`w_a`, `w_b`, `w_down`, shape `[M, d]` for `w_a`/`w_b` and `[d, M]` for `w_down`):
     - `self.w_a = nn.Parameter((wa_unit * self.target_iso_scale).clone())`
     - `self.w_b = nn.Parameter((wb_unit * self.target_iso_scale).clone())`
     - `self.w_down = nn.Parameter((wa_unit.transpose(0, 1) * math.sqrt(self.embed_dim / float(self.num_vectors))).contiguous().clone())`
   - `forward(self, x: torch.Tensor) -> torch.Tensor`:
     - Computes $u_a = F.\text{linear}(x, \text{self.w\_a})$ and $u_b = F.\text{linear}(x, \text{self.w\_b})$ in $\mathbb{R}^{\dots \times M}$.
     - Computes interference activation:
       $$h = F.\text{silu}(u_a) \cdot u_b + 0.25 \cdot (u_a^2 - u_b^2)$$
     - Projects back to $\mathbb{R}^d$: $y = F.\text{linear}(h, \text{self.w\_down})$ and returns `l2_normalize(x + y, dim=-1, eps=self.eps)`.
   - `frame_penalty(self) -> torch.Tensor`:
     - Returns `0.5 * (tight_frame_penalty(self.w_a, self.target_iso_scale) + tight_frame_penalty(self.w_b, self.target_iso_scale))`.

---

### 3.2 JAX / XLA Counterpart (`src/qdlblocks_jax/overcomplete_frame.py`)
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/overcomplete_frame.py` (and export all 3 functions in `src/qdlblocks_jax/__init__.py`) providing:
1. `welch_bound_coherence_jax(num_vectors: int, dim: int) -> float`:
   - Exact same formula and validation (`dim < 1` raises `ValueError`, `num_vectors <= dim` returns `0.0`).
2. `tight_frame_penalty_jax(w: jax.Array, target_iso_scale: float = 1.25) -> jax.Array`:
   - Accepts 2D `jax.Array` `w`. If `w.ndim != 2`, raises `ValueError`.
   - Orients `wf = w.astype(jnp.float32)` so $M \ge d$ (`if wf.shape[0] < wf.shape[1]: wf = wf.T`). Let $M, d = \text{wf.shape}$.
   - Sets $\gamma = (\text{target\_iso\_scale}^2) \cdot (M / float(d))$ and returns $\frac{1}{d} \sum_{i,j} ((\text{wf}^\top \text{wf}) - \gamma I_d)_{ij}^2$.
3. `frame_coherence_metrics_jax(w: jax.Array, eps: float = 1e-6) -> dict[str, float]`:
   - Computes and returns the exact same 6-key dictionary (`"redundancy_ratio"`, `"iso_err_vs_identity"`, `"tight_frame_err"`, `"rms_coherence"`, `"max_coherence"`, `"welch_bound"`) as `frame_coherence_metrics` in PyTorch, handling $M \le 1$ by returning `0.0` for `"rms_coherence"` and `"max_coherence"`.

---

### 3.3 Controlled Verification Script (`examples/16_hypothesis_overcomplete_frames.py`)
Create `/home/festivity/agy/QuantumDLBlocks/examples/16_hypothesis_overcomplete_frames.py` that executes two deterministic experiments:

1. **Experiment 1 (H1 — $M = 2d$ Tight-Frame Packing vs $M = d$ Orthogonal Basis)**:
   - Sets $d = 64$. Constructs:
     - $W_{\text{ortho}} = \text{optimize\_overcomplete\_tight\_frame}(64, 64, \text{steps}=1, \text{seed}=42) \in \mathbb{R}^{64 \times 64}$, and
     - $W_{\text{frame}} = \text{optimize\_overcomplete\_tight\_frame}(128, 64, \text{steps}=250, \text{lr}=0.1, \text{seed}=42) \in \mathbb{R}^{128 \times 64}$.
   - Evaluates `frame_coherence_metrics` on both and asserts for $W_{\text{frame}}$:
     - `0.85 <= metrics_frame["iso_err_vs_identity"] <= 1.15`
     - `metrics_frame["tight_frame_err"] < 0.05`
     - `metrics_frame["rms_coherence"] <= 1.10 * metrics_frame["welch_bound"]`
   - Generates $N_{\text{train}} = 1024$ and $N_{\text{test}} = 256$ random unit vectors $\Psi_{\text{train}} \in \mathbb{R}^{1024 \times 64}$, $\Psi_{\text{test}} \in \mathbb{R}^{256 \times 64}$ (seeded with `seed=100`).
   - Constructs a rank-32 target symmetric observable $O = \sum_{r=1}^{32} c_r u_r u_r^\top \in \mathbb{R}^{64 \times 64}$ where $\{u_r\}_{r=1}^{32} \subset \mathbb{R}^{64}$ are random unit vectors (drawn independently, NOT aligned with $W_{\text{ortho}}$) and $c_r = (-1)^r / \sqrt{32}$, with targets $y(\psi) = \psi^\top O \psi$.
   - For each basis $W \in \{W_{\text{ortho}}, W_{\text{frame}}\}$:
     - Computes quadratic Born features $F_{\text{train}} = (\Psi_{\text{train}} W^\top)^{\odot 2}$ and $F_{\text{test}} = (\Psi_{\text{test}} W^\top)^{\odot 2}$.
     - Computes effective feature covariance rank $R_\sigma(\tilde{F}_{\text{train}})$ on column-centered $\tilde{F}_{\text{train}} = F_{\text{train}} - F_{\text{train}}.\text{mean}(\text{dim}=0, \text{keepdim}=True)$ via Shannon entropy of its singular values `torch.linalg.svdvals(F_centered)`.
     - Fits linear ridge regression weights $\hat{\theta} = (\tilde{F}_{\text{train}}^\top \tilde{F}_{\text{train}} + 10^{-4} I_M)^{-1} \tilde{F}_{\text{train}}^\top (y_{\text{train}} - \bar{y}_{\text{train}})$ and evaluates held-out `test_mse` on $\Psi_{\text{test}}$.
   - Asserts `rank_frame >= 1.6 * rank_ortho` and `mse_frame <= 0.5 * mse_ortho` ($\ge 2.0\times$ lower MSE).

2. **Experiment 2 (H2 — Controlled Isometry Error Sweet-Spot in `QuantumBornGPT`)**:
   - Generates deterministic synthetic Zipfian language modeling sequences (`vocab_size=256`, `seq_len=64`, `num_train=512`, `num_val=128`, `seed=42`).
   - Trains three `QuantumBornGPT` arms (`vocab_size=256`, `embed_dim=64`, `seed_dim=32`, `num_heads=4`, `num_layers=2`, `block_size=64`, `use_two_beam_mlp=True`, `use_flash_born=False`) for `200` steps (`batch_size=32`):
     - **Arm A (`strict_ortho_1d`)**: `hidden_dim=64` ($M = d$). Uses `TangentStiefelMuon` (`lr=0.03`, `momentum=0.95`, `ns_steps=5`, `retract_every=1`) on 2D block weights + `AdamW` (`lr=4e-3`) on embeddings/scalars.
     - **Arm B (`strict_ortho_2d`)**: `hidden_dim=128` ($M = 2d$). Uses `TangentStiefelMuon` (`lr=0.03`, `momentum=0.95`, `ns_steps=5`, `retract_every=1`) + `AdamW` (`lr=4e-3`).
     - **Arm C (`overcomplete_relaxed_2d`)**: Starts from the same model seed (`42`), then replaces `block.mlp` on each block $l \in \{0, 1\}$ with `OvercompleteFrameInterference(embed_dim=64, frame_expansion=2.0, target_iso_scale=1.25, seed=42 + 10 * l)`. Trains with `StiefelMuon` (`lr=0.03`, `stiefel_penalty=0.0`, `retract_every=1000`) on 2D weights + `AdamW` (`lr=4e-3`) on embeddings/scalars, optimizing `loss = ce_loss + 0.01 * sum(b.mlp.frame_penalty() for b in model.blocks)`.
   - At step 200, evaluates validation cross-entropy `val_ce`, seed effective quantum rank $R_{\text{eff}}(\rho_{\text{seed}})$ (`effective_quantum_rank` on `model.hilbert_lift.collapse_to_seed` states), and average MLP `IsoErr` relative to $I_d$ across `(b.mlp.w_a, b.mlp.w_b)` via `frame_coherence_metrics(w)["iso_err_vs_identity"]`.
   - Asserts:
     - `val_ce_C <= val_ce_A - 0.08` and `val_ce_C <= val_ce_B - 0.05`
     - `r_eff_C > r_eff_A` and `r_eff_C > r_eff_B`
     - `iso_err_A < 0.01`, `iso_err_B < 0.01`, and `0.4 <= iso_err_C <= 2.5`
   - Saves JSON summary to `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_006.json`.

---

## 4. Verification & Testing Plan

### 4.1 Unit Tests (`tests/test_hypothesis_006_overcomplete_frame.py`)
Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_006_overcomplete_frame.py` testing:
1. `test_welch_bound_and_tight_frame_optimization`: Verifies `welch_bound_coherence` edge cases ($M \le d$ returns `0.0`), `no_grad()` safety of `optimize_overcomplete_tight_frame(64, 32, steps=200)`, `tight_frame_err < 0.05`, `0.85 <= iso_err_vs_identity <= 1.15`, and `rms_coherence <= 1.05 * welch_bound_coherence(64, 32)`.
2. `test_overcomplete_frame_interference_forward_and_penalty`: Verifies that `w_a` and `w_b` are distinct at initialization, output shapes match input shapes, output vectors have unit L2 norm (`||y||_2 = 1`), gradients flow cleanly to `w_a`, `w_b`, `w_down`, and `frame_penalty()` is near zero (`< 0.05`) at initialization.
3. `test_jax_pytorch_frame_metrics_parity`: Verifies that `frame_coherence_metrics` vs `frame_coherence_metrics_jax` and `tight_frame_penalty` vs `tight_frame_penalty_jax` agree within `1e-5` (including $M \le 1$ edge case).
4. `test_overcomplete_quadratic_born_capacity`: Verifies that an $M = 2d$ tight frame achieves $\ge 1.6\times$ higher quadratic Born feature covariance rank $R_\sigma(\tilde{F})$ than an $M = d$ orthogonal basis.

### 4.2 Execution Commands
```bash
/home/festivity/agy/QuantumDLBlocks/.venv/bin/pytest /home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_006_overcomplete_frame.py -v
/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/examples/16_hypothesis_overcomplete_frames.py
```
