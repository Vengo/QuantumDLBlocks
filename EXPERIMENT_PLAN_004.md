# EXPERIMENT_PLAN_004: Multi-Beam Kraus Subspace POVM Readout to Overcome the Rank-$D_{\text{seed}}$ Vocabulary Bottleneck

**Document ID**: `EXPERIMENT_PLAN_004` (Revision 2)  
**Status**: Revised for Blind Sub-Agent Review  
**Target Repository**: `/home/festivity/agy/QuantumDLBlocks`  

---

## 1. Executive Summary & Self-Contained Background

### 1.1 Architectural Context & The Rank-$D_{\text{seed}}$ Readout Bottleneck
In `QuantumDLBlocks`, the 10-qubit `QuantumBornGPT` language model ($d = 1024 = 2^{10}$, vocabulary size $V = 50,304$, seed dimension $D_{\text{seed}} = 128$) uses a factorized isometric embedding (`IsometricHilbertLift`) to keep vocabulary parameter counts compact (`50,304 x 128 = 6.44M` parameters instead of `50,304 x 1024 = 51.51M`).

In the current `"tied_stiefel"` readout mode:
1. The $L$-layer quantum backbone produces a contextual state vector $|\psi_L\rangle \in \mathcal{S}^{d-1} \subset \mathbb{R}^{d}$ ($d = 1024$) for each token position $(b, t)$.
2. Before measuring against the $V = 50,304$ vocabulary states, the model collapses $|\psi_L\rangle \in \mathbb{R}^{d}$ down to a **single** $D_{\text{seed}}$-dimensional vector $|\phi_{\text{seed}}\rangle \in \mathcal{S}^{D_{\text{seed}}-1} \subset \mathbb{R}^{D_{\text{seed}}}$ ($D_{\text{seed}} = 128$) via $W_{\text{lift}}^\top \in \mathbb{R}^{d \times D_{\text{seed}}}$, and computes logits as a single rank-1 inner product:
   $$\ell_{n, v} = \beta \langle \phi_{\text{seed}, n} \mid M_{\text{seed}, v} \rangle, \qquad \Phi_{\text{seed}} \in \mathbb{R}^{N \times D_{\text{seed}}}, \quad M_{\text{seed}} \in \mathbb{R}^{V \times D_{\text{seed}}}.$$

### 1.2 Mathematical Diagnosis of the Expressivity Bottleneck
Consider the matrix of log-probability logits across $N$ contexts and $V$ vocabulary tokens:
$$L = \beta \, \Phi_{\text{seed}} M_{\text{seed}}^\top \in \mathbb{R}^{N \times V}.$$
1. **Strict Algebraic Rank Ceiling ($\text{rank}(L) \le D_{\text{seed}} \ll d$)**:
   - Because $\Phi_{\text{seed}} \in \mathbb{R}^{N \times D_{\text{seed}}}$ and $M_{\text{seed}} \in \mathbb{R}^{V \times D_{\text{seed}}}$, the logit matrix $L$ has **algebraic rank at most $D_{\text{seed}} = 128$**, discarding $d - D_{\text{seed}} = 896$ orthogonal dimensions of the 10-qubit ($d=1024$) Hilbert state $|\psi_L\rangle$ right at the output layer!
   - This is a primary reason why the 10-qubit model ($d=1024, D_{\text{seed}}=128$) reached `4.2676` Val CE on the 5M FineWeb validation set vs. the `3.183` Slowrun record: `87.5%` ($896 / 1024$) of the backbone Hilbert space is in the nullspace of the single-beam projection $W_{\text{lift}}^\top$!
2. **Unlocking Rank-$K D_{\text{seed}}$ Readout via Multi-Beam Kraus Subspace Projection**:
   - Suppose $K \cdot D_{\text{seed}} \le d$ for an integer number of quantum readout beams $K \ge 1$ (e.g., for $d = 64, D_{\text{seed}} = 16$, $K = 4$ orthogonal subspaces $\mathbb{R}^{64} = \bigoplus_{k=1}^{4} \mathbb{R}^{16}$; for $d=1024, D_{\text{seed}}=128$, up to $K=8$ beams).
   - An orthogonal readout operator $R_{\text{out}} \in \text{St}(K \cdot D_{\text{seed}}, d)$ projects $|\psi_L\rangle \in \mathbb{R}^d$ into $K$ normalized sub-beams $|\phi^{(k)}\rangle \in \mathcal{S}^{D_{\text{seed}}-1}$ ($k \in \{0, \dots, K-1\}$).
   - Beam $0$ measures against the base normalized dictionary $M^{(0)} = \text{L2Norm}(W_{\text{wte}}) \in \mathbb{R}^{V \times D_{\text{seed}}}$, while each auxiliary beam $k \in \{1, \dots, K-1\}$ measures against a beam-specific Kraus-perturbed dictionary $M^{(k)} = \text{L2Norm}(M^{(0)} + A_k B_k^\top)$ where $A_k \in \mathbb{R}^{V \times r_b}$ (initialized with scale `adapter_init_scale / sqrt(r_b)`) and $B_k \in \mathbb{R}^{D_{\text{seed}} \times r_b}$ (initialized with scale `1.0 / sqrt(D_seed)`):
     $$L = \beta \sum_{k=0}^{K-1} w_k \, \Phi^{(k)} \left(M^{(k)}\right)^\top \in \mathbb{R}^{N \times V}.$$
   - Because $L$ is a sum of $K$ independent beam projections spanning $K$ orthogonal subspaces of $|\psi_L\rangle$, its maximum algebraic rank increases from $D_{\text{seed}}$ to $\min(N, V, D_{\text{seed}} + (K-1) r_b) \le K \cdot D_{\text{seed}} \le d$!

---

## 2. Observable Hypotheses

### Hypothesis H1: Algebraic & Effective Rank Expansion of Multi-Beam Kraus Readout ($K$ Beams vs. 1 Beam)
- **Mechanism**:
  - Define the **Logit Matrix Effective Singular Rank** $R_{\sigma}(L)$ of a row-centered logit matrix $\tilde{L} = L - \frac{1}{N}\mathbf{1}\mathbf{1}^\top L \in \mathbb{R}^{N \times V}$ with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_m \ge 0$, normalized energy distribution $p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2 + \epsilon}$, and Shannon spectral entropy $H_\sigma(L) = -\sum_i p_i \ln p_i$:
    $$R_{\sigma}(L) = \exp\!\left(H_\sigma(L)\right) \in [1, \min(N, V)].$$
  - For a single-beam seed readout ($K = 1$) with seed dimension $D_{\text{seed}}$, $\text{rank}(L) \le D_{\text{seed}}$, so $R_{\sigma}(L) \le D_{\text{seed}}$ strictly for all inputs $\Psi \in \mathbb{R}^{N \times d}$.
  - For `MultiBeamKrausReadout` with $K = 4$ beams, $D_{\text{seed}} = 16$, $d = 64$, $r_b = 16$, and `adapter_init_scale = 1.0`, each row of $\Delta M_k = A_k B_k^\top$ has expected L2 norm $\alpha = 1.0$, yielding 16 common-mode squared singular values ($\propto 2.5$) and 48 differential-mode squared singular values ($\propto 0.5$), expanding the expected effective singular rank to $R_{\sigma}(L_{K=4}) \approx 46.8 \ge 32.0$.
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Strict Rank Ceiling Breaking**: For $d = 64, D_{\text{seed}} = 16, N = 128, V = 128, r_b = 16, \text{adapter\_init\_scale} = 1.0$ on full-rank isotropic states $\Psi \in \mathbb{R}^{128 \times 64}$:
     - Single-beam ($K=1$) satisfies $R_{\sigma}(L_{K=1}) \le 16.0 + 10^{-4}$.
     - Multi-beam ($K=4$) satisfies $R_{\sigma}(L_{K=4}) \ge 32.0$ and $R_{\sigma}(L_{K=4}) > 2.0 \times R_{\sigma}(L_{K=1})$.
     - Strict monotonic scaling across beam count: $R_{\sigma}(L_{K=4}) > R_{\sigma}(L_{K=2}) > R_{\sigma}(L_{K=1})$.

### Hypothesis H2: Neural Learning Capacity on High-Rank Contextual Hilbert-State Readout Tasks
- **Mechanism**:
  - Consider a contextual sequence prediction task where each context $(b, t)$ produces a normalized $d=64$-dimensional quantum state $|\psi_{b,t}\rangle \in \mathcal{S}^{63}$ and target token $y_{b,t} \in \{0, \dots, V-1\}$ ($V = 256$) generated by a ground-truth rank-$48$ measurement operator $W_{\text{true}} \in \mathbb{R}^{V \times d}$ spanning 3 of the 4 orthogonal $16$-dimensional subspaces of $\mathbb{R}^{64}$.
  - Train two readout models starting from the same base dictionary $W_{\text{wte}} \in \mathbb{R}^{256 \times 16}$ for $S = 120$ steps:
    - **Arm A (`Single-Beam Tied-Stiefel`, $K=1$)**: `MultiBeamKrausReadout(vocab_size=256, embed_dim=64, seed_dim=16, num_beams=1)` (rank ceiling $16 < 48$).
    - **Arm B (`Multi-Beam Kraus Readout`, $K=4$)**: `MultiBeamKrausReadout(vocab_size=256, embed_dim=64, seed_dim=16, num_beams=4, beam_adapter_rank=12, adapter_init_scale=1.0)` (rank ceiling $16 + 3 \times 12 = 52 \ge 48$).
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Higher Learned Logit Effective Rank**: Post-training validation logit matrix effective rank satisfies:
     $$R_{\sigma}(\text{Arm B}) \ge 1.35 \times R_{\sigma}(\text{Arm A}).$$
  2. **Lower Validation Cross-Entropy (`val_ce`)**: After $S = 120$ training steps:
     $$\text{val\_ce}(\text{Arm B}) \le \text{val\_ce}(\text{Arm A}) - 0.15.$$

---

## 3. Planned Code Changes (Disjoint New Modules & Files)

To allow conflict-free parallel implementation alongside other ongoing tasks, all changes are placed in **new, dedicated files**:

### 3.1 PyTorch Multi-Beam Kraus Readout (`src/qdlblocks/layers/kraus_readout.py`)

Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/kraus_readout.py`:

1. **Exact `l2_normalize` Definition**:
   - Import `l2_normalize` directly from `..functional` (`from ..functional import l2_normalize`), which computes `x / torch.clamp(torch.linalg.norm(x, dim=dim, keepdim=True), min=eps)`.

2. **`compute_logit_matrix_effective_rank(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor`**:
   - **Input**: Tensor `logits` of shape `(..., V)` with `logits.ndim >= 2`. Let $V = \text{logits.shape}[-1]$ and $N = \text{logits.numel}() // V$. Raises `ValueError` if `logits.ndim < 2` or $N < 2$ or $V < 2$.
   - **Computation**:
     ```python
     x = logits.reshape(N, V).float()
     x_c = x - x.mean(dim=0, keepdim=True)
     s = torch.linalg.svdvals(x_c)
     s_sq = s ** 2
     p = s_sq / (torch.sum(s_sq) + eps)
     ent = -torch.sum(p * torch.log(torch.clamp(p, min=eps)))
     return torch.exp(ent)
     ```

3. **`class MultiBeamKrausReadout(nn.Module)`**:
   - **Constructor Signature**:
     ```python
     def __init__(
         self,
         vocab_size: int,
         embed_dim: int = 1024,
         seed_dim: int = 128,
         num_beams: int = 4,
         beam_adapter_rank: int = 16,
         adapter_init_scale: float = 1.0,
         povm_temperature: float = 0.035,
         learnable_temperature: bool = True,
         eps: float = 1e-12,
     ) -> None:
     ```
   - **Validation**:
     - Raises `ValueError` if `vocab_size < 2`, `num_beams < 1`, `seed_dim < 1`, `embed_dim < seed_dim`, `num_beams * seed_dim > embed_dim`, `beam_adapter_rank < 1`, or `adapter_init_scale <= 0.0`.
   - **Parameters**:
     - `self.vocab_size = vocab_size`, `self.embed_dim = embed_dim`, `self.seed_dim = seed_dim`, `self.num_beams = num_beams`, `self.beam_adapter_rank = beam_adapter_rank`, `self.adapter_init_scale = adapter_init_scale`, `self.eps = eps`.
     - `self.readout_rot = nn.Linear(embed_dim, num_beams * seed_dim, bias=False)` initialized via `nn.init.orthogonal_(self.readout_rot.weight)` (weight shape `(num_beams * seed_dim, embed_dim)`).
     - If `num_beams > 1`:
       - `self.beam_vocab_a` of shape `(num_beams - 1, vocab_size, beam_adapter_rank)` initialized with mutually orthogonal columns across `(num_beams - 1) * beam_adapter_rank` (when `vocab_size >= (num_beams - 1) * beam_adapter_rank`, else per-beam orthogonal) scaled by `2.0 * adapter_init_scale * math.sqrt(float(vocab_size) / float(beam_adapter_rank))`
       - `self.beam_seed_b` of shape `(num_beams - 1, seed_dim, beam_adapter_rank)` initialized per beam via `nn.init.orthogonal_(..., gain=1.0)`
       - `self.beam_weights = nn.Parameter(torch.ones(num_beams, dtype=torch.float32) / math.sqrt(float(num_beams)))`
     - Else (`num_beams == 1`):
       - `self.register_parameter("beam_vocab_a", None)`
       - `self.register_parameter("beam_seed_b", None)`
       - `self.register_parameter("beam_weights", None)`
     - `init_beta = 1.0 / max(povm_temperature, 1e-4)`
     - If `learnable_temperature`: `self.inv_temp = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))`
     - Else: `self.register_buffer("inv_temp", torch.tensor(float(init_beta), dtype=torch.float32))`
   - **Forward Computation (`forward(self, psi: torch.Tensor, base_wte: torch.Tensor) -> torch.Tensor`)**:
     ```python
     beams = self.readout_rot(psi).reshape(*psi.shape[:-1], self.num_beams, self.seed_dim)
     beams = l2_normalize(beams, dim=-1, eps=self.eps)
     m_0 = l2_normalize(base_wte, dim=-1, eps=self.eps).to(psi.dtype)
     beta = torch.clamp(self.inv_temp, min=1.0, max=60.0).to(psi.dtype)
     if self.num_beams == 1:
         return F.linear(beams.squeeze(-2), m_0) * beta
     assert self.beam_vocab_a is not None and self.beam_seed_b is not None and self.beam_weights is not None
     w = self.beam_weights.to(psi.dtype)
     w = w / torch.clamp(torch.linalg.norm(w), min=self.eps)
     logits = w[0] * F.linear(beams[..., 0, :], m_0)
     for k in range(1, self.num_beams):
         a_k = self.beam_vocab_a[k - 1].to(psi.dtype)
         b_k = self.beam_seed_b[k - 1].to(psi.dtype)
         delta_m = torch.matmul(a_k, b_k.transpose(-2, -1))
         m_k = l2_normalize(m_0 + delta_m, dim=-1, eps=self.eps)
         logits = logits + w[k] * F.linear(beams[..., k, :], m_k)
     return logits * beta
     ```

---

### 3.2 JAX Multi-Beam Kraus Readout (`src/qdlblocks_jax/kraus_readout.py`)

Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/kraus_readout.py` importing `l2_normalize` from `.functional` (`x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), eps)`):

1. **`compute_logit_matrix_effective_rank_jax(logits: jax.Array, eps: float = 1e-8) -> jax.Array`**:
   - Validates `logits.ndim >= 2`, $N = \text{logits.size} // V \ge 2$, $V \ge 2$.
   - Computes `x = logits.reshape(N, V).astype(jnp.float32)`, `x_c = x - jnp.mean(x, axis=0, keepdims=True)`, `s = jnp.linalg.svd(x_c, compute_uv=False)`, `s_sq = s ** 2`, `p = s_sq / (jnp.sum(s_sq) + eps)`, `ent = -jnp.sum(p * jnp.log(jnp.maximum(p, eps)))`, and returns `jnp.exp(ent)`.

2. **`init_multibeam_kraus_params(key: jax.Array, vocab_size: int, embed_dim: int, seed_dim: int, num_beams: int = 4, beam_adapter_rank: int = 16, adapter_init_scale: float = 1.0, povm_temperature: float = 0.035) -> dict[str, jax.Array]`**:
   - Validates `vocab_size >= 2`, `num_beams >= 1`, `seed_dim >= 1`, `embed_dim >= seed_dim`, `num_beams * seed_dim <= embed_dim`, `beam_adapter_rank >= 1`, `adapter_init_scale > 0.0`.
   - Splits `key` into `k_rot, k_a, k_b = jax.random.split(key, 3)`.
   - Initializes orthogonal `readout_rot` of shape `(num_beams * seed_dim, embed_dim)` (matching PyTorch `nn.Linear.weight`) via QR decomposition on a `(embed_dim, num_beams * seed_dim)` normal matrix transposed to `(num_beams * seed_dim, embed_dim)`.
   - Returns dictionary with exact keys:
     - `"readout_rot"`: shape `(num_beams * seed_dim, embed_dim)`
     - `"inv_temp"`: scalar `jnp.array(1.0 / max(povm_temperature, 1e-4), dtype=jnp.float32)`
     - When `num_beams > 1`:
       - `"beam_vocab_a"`: shape `(num_beams - 1, vocab_size, beam_adapter_rank)` initialized via orthogonal QR columns scaled by `2.0 * adapter_init_scale * math.sqrt(float(vocab_size) / float(beam_adapter_rank))`
       - `"beam_seed_b"`: shape `(num_beams - 1, seed_dim, beam_adapter_rank)` initialized per beam via orthogonal QR columns with `gain=1.0`
       - `"beam_weights"`: shape `(num_beams,)` initialized to `jnp.ones((num_beams,), dtype=jnp.float32) / math.sqrt(float(num_beams))`
     - When `num_beams == 1`: `"beam_vocab_a"`, `"beam_seed_b"`, `"beam_weights"` are omitted (or not accessed).

3. **`multibeam_kraus_forward_jax(params: dict[str, jax.Array], psi: jax.Array, base_wte: jax.Array, eps: float = 1e-12) -> jax.Array`**:
   - Let `rot = params["readout_rot"]` of shape `(K * D_seed, d)` and `D_seed = base_wte.shape[-1]`, `K = rot.shape[0] // D_seed`.
   - `beams = jnp.matmul(psi, rot.T).reshape(*psi.shape[:-1], K, D_seed)`
   - `beams = l2_normalize(beams, axis=-1, eps=eps)`
   - `m_0 = l2_normalize(base_wte, axis=-1, eps=eps).astype(psi.dtype)`
   - `beta = jnp.clip(params["inv_temp"], 1.0, 60.0).astype(psi.dtype)`
   - If `K == 1`: `return jnp.matmul(jnp.squeeze(beams, axis=-2), m_0.T) * beta`
   - If `K > 1`:
     - `w = params["beam_weights"].astype(psi.dtype)`
     - `w = w / jnp.maximum(jnp.linalg.norm(w), eps)`
     - `logits = w[0] * jnp.matmul(beams[..., 0, :], m_0.T)`
     - Loop `k` in `range(1, K)`:
       - `a_k = params["beam_vocab_a"][k - 1].astype(psi.dtype)`
       - `b_k = params["beam_seed_b"][k - 1].astype(psi.dtype)`
       - `m_k = l2_normalize(m_0 + jnp.matmul(a_k, jnp.swapaxes(b_k, -2, -1)), axis=-1, eps=eps)`
       - `logits = logits + w[k] * jnp.matmul(beams[..., k, :], m_k.T)`
     - `return logits * beta`

---

### 3.3 Standalone Controlled Hypothesis Verification Script (`examples/14_hypothesis_multibeam_kraus_readout.py`)

Create `/home/festivity/agy/QuantumDLBlocks/examples/14_hypothesis_multibeam_kraus_readout.py` that runs in `< 30` seconds, prints a summary table, asserts all quantitative predictions of Hypotheses H1 and H2, and writes `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_004.json`:

1. **Experiment 1 (H1 — Algebraic & Effective Singular Rank Ceiling Across $K \in \{1, 2, 4\}$)**:
   - With `torch.manual_seed(42)`, generate full-rank normalized states `psi = l2_normalize(torch.randn(128, 64), dim=-1)` and `base_wte = l2_normalize(torch.randn(128, 16), dim=-1)`.
   - Evaluate `MultiBeamKrausReadout(vocab_size=128, embed_dim=64, seed_dim=16, num_beams=K, beam_adapter_rank=16, adapter_init_scale=1.0)` for $K \in \{1, 2, 4\}$.
   - Asserts:
     - `R_sigma(K=1) <= 16.0 + 1e-4`
     - `R_sigma(K=4) >= 32.0` and `R_sigma(K=4) > 2.0 * R_sigma(K=1)`
     - `R_sigma(K=4) > R_sigma(K=2) > R_sigma(K=1)`

2. **Experiment 2 (H2 — High-Rank Hilbert State Sequence Readout Capacity)**:
   - **Deterministic Synthetic Rank-48 Readout Dataset (`torch.manual_seed(100)`)**:
     - Set $d = 64, D_{\text{seed}} = 16, V = 256, N_{\text{train}} = 512, N_{\text{val}} = 256$.
     - Construct a fixed shared base dictionary `base_wte = l2_normalize(torch.randn(V, D_seed), dim=-1)` and a ground-truth rank-48 target measurement matrix $W_{\text{true}} \in \mathbb{R}^{V \times d}$ formed by concatenating 3 blocks of size $(V, 16)$ across dims `0..47` (with block 0 partially aligned with `base_wte` and blocks 1–2 orthonormal-column blocks) and zeros in dims `48..63`, normalized per row.
     - Generate prototype-clustered normalized Hilbert state batches `psi_train = l2_normalize(W_true[idx_train] + 0.50 * l2_normalize(torch.randn(N_train, d), dim=-1), dim=-1)` and `psi_val = l2_normalize(W_true[idx_val] + 0.50 * l2_normalize(torch.randn(N_val, d), dim=-1), dim=-1)`, with deterministic target labels `y_train = torch.argmax(psi_train @ W_true.T, dim=-1)` and `y_val = torch.argmax(psi_val @ W_true.T, dim=-1)`.
   - **Training Setup ($S = 120$ steps, full-batch or mini-batch `batch_size = 64`, `torch.optim.AdamW(..., lr=0.02, weight_decay=1e-4)`)**:
     - **Arm A ($K=1$)**: `MultiBeamKrausReadout(vocab_size=256, embed_dim=64, seed_dim=16, num_beams=1)`
     - **Arm B ($K=4$)**: `MultiBeamKrausReadout(vocab_size=256, embed_dim=64, seed_dim=16, num_beams=4, beam_adapter_rank=12, adapter_init_scale=1.0)`
     - Both arms optimize their parameters against `F.cross_entropy(readout(psi_batch, base_wte), y_batch)` (with `base_wte` held fixed so the experiment purely isolates the rank-$16$ vs. rank-$52$ readout capacity of `MultiBeamKrausReadout`!).
   - **Assertions**:
     - `final_logit_eff_rank(Arm B) >= 1.35 * final_logit_eff_rank(Arm A)`
     - `final_val_ce(Arm B) <= final_val_ce(Arm A) - 0.15`
   - **JSON Output Schema (`.cache/hypothesis_experiments_004.json`)**:
     - Keys: `{"h1_rank_scaling": {"r_sigma_k1": ..., "r_sigma_k2": ..., "r_sigma_k4": ..., "passed": True}, "h2_learning_capacity": {"arm_a_k1": {"val_ce": ..., "logit_eff_rank": ...}, "arm_b_k4": {"val_ce": ..., "logit_eff_rank": ...}, "passed": True}}`.

---

## 4. Testing & Verification Plan

### 4.1 Unit Tests (`tests/test_hypothesis_004_kraus_readout.py`)
Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_004_kraus_readout.py` testing:
1. **`test_logit_matrix_effective_rank_exact_bounds_torch_and_jax`**:
   - Verifies for a rank-1 outer product matrix $L = u v^\top + c$: `compute_logit_matrix_effective_rank` returns `1.0` (`atol=1e-4`).
   - Verifies for a centered matrix with $k = 12$ equal nonzero singular values: `compute_logit_matrix_effective_rank` and `@jax.jit`-compiled `compute_logit_matrix_effective_rank_jax` both return `12.0` (`atol=1e-4`).
2. **`test_multibeam_kraus_readout_rank_expansion_and_torch_jax_parity`**:
   - Verifies constructor validation (`ValueError` when `num_beams * seed_dim > embed_dim`), output shapes `(B, T, V)` for $K=1$ and $K=4$, finite gradients on all parameters (`readout_rot.weight`, `beam_vocab_a`, `beam_seed_b`, `beam_weights`, `inv_temp`), and exact numerical parity (`atol=1e-5`) between PyTorch `MultiBeamKrausReadout` and JAX `multibeam_kraus_forward_jax` when given identical parameter tensors for both $K=1$ and $K=4$.
3. **`test_hypothesis_004_script_execution`**:
   - Executes `examples/14_hypothesis_multibeam_kraus_readout.py` and verifies that `.cache/hypothesis_experiments_004.json` is written with `"passed": True` for both H1 and H2.
