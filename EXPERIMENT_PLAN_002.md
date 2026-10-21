# EXPERIMENT_PLAN_002: Observable Hypotheses on Multi-Horizon POVM Temperature Dynamics, Hardware MFU Roofline Recovery, and Hilbert Seed Subspace Capacity

**Document ID**: `EXPERIMENT_PLAN_002` (Revision 2)  
**Status**: Revised for Sub-Agent Review  
**Target Repository**: `/home/festivity/agy/QuantumDLBlocks`  

---

## 1. Executive Summary & Self-Contained Background

### 1.1 Architectural Context (`QuantumBornGPT` & `HaxQuantumBornGPT`)
`QuantumDLBlocks` implements a quantum-geometric autoregressive Transformer language model in both PyTorch (`src/qdlblocks/`) and JAX/Equinox/Haliax (`src/qdlblocks_jax/`):
1. **Isometric Hilbert Lift (`IsometricHilbertLift`)**:
   - Token IDs $x_{b,t} \in \{0, \dots, V-1\}$ index a seed embedding table $W_{\text{wte}} \in \mathbb{R}^{V \times D_{\text{seed}}}$, normalized onto the unit hypersphere $|\psi_{\text{seed}}\rangle = \text{L2Norm}(W_{\text{wte}}[x_{b,t}]) \in \mathcal{S}^{D_{\text{seed}}-1}$.
   - A column-orthonormal Stiefel matrix $W_{\text{lift}} \in \text{St}(D_{\text{seed}}, d)$ ($W_{\text{lift}}^\top W_{\text{lift}} = I_{D_{\text{seed}}}$, where $d = 2^n \ge D_{\text{seed}}$, e.g., $d=1024$ for a 10-qubit backbone and $D_{\text{seed}}=128$) lifts the seed state into the $d$-dimensional Hilbert space:
     $$|\psi_0\rangle = \text{L2Norm}\!\left(|\psi_{\text{seed}}\rangle W_{\text{lift}}\right) \in \mathcal{S}^{d-1}.$$
2. **Riemannian Stiefel Backbone (`QuantumBornBlock` & `StiefelMuon`)**:
   - $L$ layers of causal Subspace Born Attention ($A_{ij} = \max(0, \gamma_h \langle q_i | k_j \rangle)^2$ or complex phase $\gamma_h^2 |\langle q_i | k_j \rangle|^2$) and Squared-ReLU two-beam MLPs evolve $|\psi_0\rangle \mapsto |\psi_L\rangle \in \mathcal{S}^{d-1}$.
   - All 2D projection matrices are optimized on the Stiefel manifold via Newton-Schulz-5 orthogonalized momentum (`StiefelMuon`).
3. **POVM Readout Modes (`readout_mode` in `QuantumBornGPT`)**:
   - `QuantumBornGPT` supports three readout modes:
     1. **`"tied_stiefel"`**: The final backbone state $|\psi_L\rangle \in \mathbb{R}^{B \times T \times d}$ is rotated by $R_{\text{out}} \in \mathbb{R}^{d \times d}$ (`self.readout_rot`) and collapsed back to the $D_{\text{seed}}$-dimensional seed subspace via $W_{\text{lift}}^\top$:
        $$|\phi_{\text{seed}}\rangle = \text{L2Norm}\!\left(\text{L2Norm}(|\psi_L\rangle R_{\text{out}}^\top) W_{\text{lift}}^\top\right) \in \mathbb{R}^{B \times T \times D_{\text{seed}}}.$$
        Vocabulary POVM measurement vectors are formed as $M_{\text{seed}} = \text{L2Norm}(W_{\text{wte}} + \Delta_{\text{POVM}}) \in \mathbb{R}^{V \times D_{\text{seed}}}$. Next-token ($h=1$) logits are scaled by a learnable inverse Born temperature $\beta_1 = \text{clamp}(\text{self.tied\_inv\_temp}, 1.0, 60.0) \in \mathbb{R}$ (initialized to $\beta_1^{(0)} = 1 / \tau_0 = 1 / 0.035 \approx 28.5714$):
        $$\ell^{(1)}_{b,t,v} = \beta_1 \langle \phi_{\text{seed}, b, t} \mid M_{\text{seed}, v} \rangle, \qquad \mathcal{L}_{1} = \text{CrossEntropy}\!\left(\ell^{(1)}, Y\right).$$
     2. **`"symmetric_dual"`**: Identical to `"tied_stiefel"` except $R_{\text{out}} = I_d$, $\Delta_{\text{POVM}} = 0$ ($M_{\text{seed}} = \text{L2Norm}(W_{\text{wte}})$), and $\beta_1 = \text{self.\_dual\_inv\_temp} = 1.0 / \tau_0$ is a fixed Python float.
     3. **`"asymmetric"`**: Uses `self.povm_readout: ProjectivePOVMHead` (`src/qdlblocks/layers/measurement.py`), which stores a full-rank parameter `self.povm_readout.povm_head` of shape `(V, d)` and exposes a property `self.povm_readout.inv_temp -> torch.Tensor | float` (clamped to `[10.0, 100.0]` when `learnable_temperature=True`, or fixed float `1.0 / povm_temperature` otherwise). Logits are computed directly in $\mathbb{R}^d$ via $\ell^{(1)} = \beta_1 \langle \text{L2Norm}(\psi_L) \mid \text{L2Norm}(\text{povm\_head}) \rangle$.
4. **Multi-Token Quantum POVM Prediction (`mtp_horizons = H`)**:
   - When `mtp_horizons` $H > 1$, the model learns $H-1$ orthogonal Stiefel rotation matrices $U_1, \dots, U_{H-1} \in \text{O}(D_{\text{mtp}})$ (where $D_{\text{mtp}} = d$ in `"asymmetric"` mode and $D_{\text{mtp}} = D_{\text{seed}}$ in `"tied_stiefel"` / `"symmetric_dual"` modes) to predict future tokens $x_{t+1+h_{\text{idx}}}$ for $h_{\text{idx}} \in \{1, \dots, H-1\}$ (corresponding to lookahead horizon $h = h_{\text{idx}} + 1 \in \{2, \dots, H\}$):
     $$|z^{(h_{\text{idx}})}_{b,t}\rangle = \text{L2Norm}\!\left(|\phi_{b, t}\rangle U_{h_{\text{idx}}}^\top\right), \quad t \in \{0, \dots, T - 1 - h_{\text{idx}}\}.$$
   - **Current Implementation Flaw**: In both `src/qdlblocks/models/born_gpt.py` (lines 280–304) and `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` (lines 293–306), all horizons $h \in \{1, \dots, H\}$ share the **exact same scalar inverse temperature $\beta = \text{inv\_t}$**, backpropagate auxiliary horizon gradients directly into $M_{\text{seed}}$ and $\beta$, and execute the dense `[B*(T - h_idx), V]` vocabulary matmul on **every single token position** ($s=1$).

---

### 1.2 Empirical Diagnosis of the Live FineWeb 100M `WSD + MTP=3` Plateau
On the genuine `100,000,000`-token FineWeb training shard and `4,999,488`-token validation shard (`V = 50,304` padded vocabulary, $d=1024, D_{\text{seed}}=128, L=6, H_{\text{heads}}=16$), live telemetry from our `NVIDIA RTX PRO 6000 Blackwell` (`gpu_slowrun`) and `Google Cloud TPU v5e-1` (`tpu_slowrun`) runs revealed why `GPU WSD + MTP=3` plateaued around `4.69 - 4.71` Val CE between `100M` and `400M` tokens (`Epochs 1.0 - 4.0`) before plunging to `4.4881` at `Epoch 4.5` (`450M` tokens):

1. **Anomaly A — Shared-Temperature Collapse (`beta` drops from `28.57` to `25.10`, `gamma_h` drops from `8.00` to `6.65`)**:
   - In the single-horizon (`mtp_horizons = 1`) 5-epoch Cosine run, the learnable inverse temperature $\beta$ rose monotonically from `28.57 -> 28.98 -> 29.39 -> ... -> 32.67` and attention scale $\gamma_h$ rose `8.00 -> 9.45`, reaching `4.2676` Val CE (`71.3` PPL).
   - In the multi-horizon (`mtp_horizons = 3`) run, sharing a single scalar $\beta$ across $t+1$ (lower conditional entropy) and $t+2, t+3$ (much higher conditional entropy because intermediate tokens are unobserved) caused the high-uncertainty gradients $\frac{\partial \mathcal{L}_{h}}{\partial \beta}$ from $h=2, 3$ to pull the shared $\beta$ **downward** from `28.57 -> 26.54 -> 25.47 -> 25.10`. Simultaneously, un-detached auxiliary gradients pulled the shared seed state $|\phi_{\text{seed}}\rangle$ and dictionary $M_{\text{seed}}$ toward a blurry compromise across $(t+1, t+2, t+3)$, while un-decayed WSD `lr_mult = 1.0` kept Stiefel `IsoErr` pinned at `1.14 - 1.23` until the Step 6100 (`400M` tok) cooldown began.
2. **Anomaly B — Vocabulary-Head HBM Bandwidth Bottleneck (`22.4%` Throughput & MFU Drop)**:
   - With micro-batch $B=32, T=512$ ($N = 16,384$ tokens/step) and vocabulary $V = 50,304$, each horizon's vocabulary matmul + cross-entropy materializes a `[16384, 50304]` tensor (`1.65 GB` in BF16/FP32).
   - Computing all 3 horizons densely at every token position reduced Blackwell GPU speed from **`311,494 tok/s` (`28.6%` MFU)** to **`241,607 tok/s` (`22.1%` MFU)** (`-22.4%` throughput) and TPU v5e-1 speed from **`122,073 tok/s` (`28.6%` MFU)** to **`96,072 tok/s` (`22.5%` MFU)** (`-21.3%` throughput).

---

## 2. Observable Hypotheses

We formulate three falsifiable, quantitatively measurable hypotheses:

### Hypothesis H1: Decoupled Per-Horizon POVM Temperatures ($\boldsymbol{\beta} \in \mathbb{R}^H$) & Auxiliary Readout Stop-Gradient
- **Mechanism**:
  1. Replace the shared scalar inverse temperature on auxiliary horizons with a **horizon-specific learnable inverse temperature vector** $\boldsymbol{\beta}_{\text{aux}} = (\beta_2, \dots, \beta_H) \in \mathbb{R}^{H-1}$ alongside primary inverse temperature $\beta_1$, initialized with harmonic decay:
     $$\beta_h^{(0)} = \frac{1}{\tau_0 \sqrt{h}}, \qquad h \in \{2, \dots, H\},$$
     where $\tau_0 = \text{povm\_temperature}$ (default `0.035`, so $\beta_1^{(0)} \approx 28.5714$, $\beta_2^{(0)} \approx 20.2031$, $\beta_3^{(0)} \approx 16.4957$).
  2. When `mtp_detach_readout = True` (default `True`), auxiliary horizons ($h \ge 2$, i.e., $h_{\text{idx}} \ge 1$) evaluate logits against `M_seed.detach()` (PyTorch) / `jax.lax.stop_gradient(M_seed)` (JAX) and use their own scalar $\beta_{h_{\text{idx}}+1}$, so gradients from $h \ge 2$ update the backbone state, the Stiefel horizon rotation $U_{h_{\text{idx}}}$, and $\beta_{h_{\text{idx}}+1}$, **without** corrupting $\beta_1$ or the primary $t+1$ vocabulary POVM dictionary $M_{\text{seed}}$.
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Temperature Ordering & Trajectory**: After $S = 120$ training steps on autoregressive token sequences with `mtp_horizons = 3`, the learned temperatures in the decoupled model (`Arm B`) satisfy strict monotonic ordering $\beta_1(\text{Arm B}) > \beta_2(\text{Arm B}) > \beta_3(\text{Arm B})$, and the primary temperature satisfies $\beta_1(\text{Arm B}) > \beta_1(\text{Arm A}) + 1.0$, where `Arm A` is the shared-temperature, un-detached baseline (`mtp_decoupled_temp=False, mtp_detach_readout=False`).
  2. **Next-Token Validation Loss (`val_ce_t1`)**: At matched token budgets (`120` steps, `mtp_horizons = 3`), `Arm B` (`mtp_decoupled_temp=True, mtp_detach_readout=True`) achieves lower next-token ($h=1$) validation cross-entropy than `Arm A` by at least **`0.05` nats**:
     $$\text{val\_ce\_t1}(\text{Arm B}) \le \text{val\_ce\_t1}(\text{Arm A}) - 0.05.$$

### Hypothesis H2: Static-Shape Strided Subsampled Multi-Horizon POVM (`mtp_stride = s`) for Hardware Utilization Recovery
- **Mechanism**:
  - Primary horizon $h=1$ always computes the full dense cross-entropy over all $B \times T$ tokens.
  - For auxiliary horizons $h_{\text{idx}} \in \{1, \dots, H-1\}$ when `mtp_stride` $s > 1$ and $T_{\text{avail}} = T - h_{\text{idx}} > s$: before computing the Stiefel rotation and expensive vocabulary projection $z^{(h_{\text{idx}})} M_{\text{aux}}^\top \in \mathbb{R}^{N_{\text{aux}} \times V}$, we subsample along the sequence dimension with stride $s$ and **compile-time deterministic horizon phase offset** $o = (h_{\text{idx}} - 1) \bmod s$:
    $$\text{state slice}: \;\; [:, \; o : T - h_{\text{idx}} : s, \; :], \qquad \text{target slice}: \;\; [:, \; h_{\text{idx}} + o : T : s].$$
  - Because $o = (h_{\text{idx}} - 1) \bmod s$ depends only on static integers ($h_{\text{idx}}, s, T$), the resulting sequence length $T_{\text{sub}} = \lceil (T - h_{\text{idx}} - o) / s \rceil$ is **100% static across all steps**, requires zero mutable Python step counters, causes **zero `torch.compile` or `jax.jit` recompilations**, behaves identically in `train` and `eval` modes, staggers different horizons across complementary sequence offsets, and reduces auxiliary horizon FLOPs and HBM bytes accessed by a factor of $s$.
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Analytical FLOPs & HBM Traffic Reduction**: For $B=32, T=512, D_{\text{seed}}=128, V=50304, H=3$, setting `mtp_stride = 4` reduces auxiliary vocabulary-head FLOPs and HBM logit/gradient bytes by **$75\%$ ($4\times$)** compared to `mtp_stride = 1`.
  2. **Measured Throughput (`tok/s`) & MFU Recovery**: In the controlled large-vocabulary benchmark (`vocab_size=16384, embed_dim=128, seed_dim=64, batch_size=8, seq_len=128, mtp_horizons=3`), `mtp_stride = 4` achieves **`>= 15%` higher throughput (`tok_per_s`)** than `mtp_stride = 1`:
     $$\text{tok\_per\_s}(\text{stride}=4) \ge 1.15 \times \text{tok\_per\_s}(\text{stride}=1).$$

### Hypothesis H3: Hilbert Seed Subspace Bottleneck & Effective Quantum Rank $R_{\text{eff}}(\rho_{\text{seed}})$
- **Mechanism**:
  - In `tied_stiefel` readout, the $d$-dimensional quantum state $|\psi_L\rangle \in \mathcal{S}^{d-1}$ is projected onto the $D_{\text{seed}}$-dimensional subspace via $W_{\text{lift}}^\top \in \mathbb{R}^{d \times D_{\text{seed}}}$ to produce normalized seed state vectors $|\phi_{\text{seed}}^{(i)}\rangle \in \mathcal{S}^{D_{\text{seed}}-1}$ for $i \in \{1, \dots, N\}$.
  - Define the **Empirical State Ensemble Density Matrix** over a batch of $N$ normalized states $|\phi^{(i)}\rangle \in \mathbb{R}^{D_{\text{seed}}}$:
    $$\rho_{\text{seed}} = \frac{1}{N} \sum_{i=1}^{N} |\phi^{(i)}\rangle\langle\phi^{(i)}| \in \mathbb{R}^{D_{\text{seed}} \times D_{\text{seed}}}, \qquad \text{Tr}(\rho_{\text{seed}}) = 1, \quad \rho_{\text{seed}} \succeq 0.$$
  - Define two quantum-information capacity observables of $\rho_{\text{seed}}$ with eigenvalues $\{\lambda_j\}_{j=1}^{D_{\text{seed}}}$ ($\sum_j \lambda_j = 1$):
    1. **Effective Quantum Rank (Exponential von Neumann Entropy)**:
       $$R_{\text{eff}}(\rho_{\text{seed}}) = \exp\!\left(S(\rho_{\text{seed}})\right) = \exp\!\left(-\sum_{j=1}^{D_{\text{seed}}} \lambda_j \ln \lambda_j\right) \in [1, D_{\text{seed}}].$$
    2. **Purity Participation Ratio (Inverse Purity / Rényi-2 Rank)**:
       $$\text{PR}(\rho_{\text{seed}}) = \frac{1}{\text{Tr}(\rho_{\text{seed}}^2)} = \frac{1}{\sum_{j=1}^{D_{\text{seed}}} \lambda_j^2} \in [1, D_{\text{seed}}].$$
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Exact Mathematical Bounds**: For any batch of states, $1 \le \text{PR}(\rho_{\text{seed}}) \le R_{\text{eff}}(\rho_{\text{seed}}) \le \min(N, D_{\text{seed}})$. For an isotropic/maximally mixed ensemble ($\rho_{\text{seed}} = I_{D_{\text{seed}}} / D_{\text{seed}}$), $\text{PR}(\rho_{\text{seed}}) = R_{\text{eff}}(\rho_{\text{seed}}) = D_{\text{seed}}$; for a collapsed rank-1 ensemble ($|\phi^{(i)}\rangle = |\phi_0\rangle$), $\text{PR}(\rho_{\text{seed}}) = R_{\text{eff}}(\rho_{\text{seed}}) = 1.0$.
  2. **Capacity Scaling Law across $D_{\text{seed}} \in \{8, 16, 32, 64\}$ (with fixed $d = 128$)**: When training `QuantumBornGPT` on a 64-motif synthetic language task (`vocab_size=256`) with fixed backbone dimension $d=128$ across $D_{\text{seed}} \in \{8, 16, 32, 64\}$, post-training Effective Quantum Rank satisfies strict monotonic ordering:
     $$R_{\text{eff}}(64) > R_{\text{eff}}(32) > R_{\text{eff}}(16) > R_{\text{eff}}(8), \qquad \text{and} \qquad \text{val\_ce}(64) < \text{val\_ce}(8).$$

---

## 3. Planned Code Changes (Exact File Paths & Signatures)

### 3.1 New Quantum Capacity Observables in PyTorch (`src/qdlblocks/functional.py`) and JAX (`src/qdlblocks_jax/functional.py`)

Add three functions to **both** `src/qdlblocks/functional.py` (and export in `src/qdlblocks/__init__.py`) and `src/qdlblocks_jax/functional.py` (and export in `src/qdlblocks_jax/__init__.py`):

1. **`empirical_state_density(psi, eps: float = 1e-8)`**:
   - **Signature**:
     - PyTorch: `def empirical_state_density(psi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:`
     - JAX: `def empirical_state_density(psi: jax.Array, eps: float = 1e-8) -> jax.Array:`
   - **Input & Validation**:
     - Real or complex state tensor `psi` of shape `(..., D)`.
     - Raises `ValueError` if `psi.ndim < 1` or `psi.shape[-1] < 1` or `psi.numel() == 0` (PyTorch) / `psi.size == 0` (JAX).
   - **Computation**:
     - Let $D = \text{psi.shape}[-1]$ and $N = \text{psi.numel}() // D$ (or `psi.size // D` in JAX).
     - Reshape `psi` to 2D `(N, D)` and normalize each state vector to unit length along the last axis:
       - PyTorch: `psi_norm = l2_normalize(ensure_complex(psi.reshape(N, D)), dim=-1, eps=eps)`
       - JAX: `psi_norm = l2_normalize(ensure_complex(psi.reshape(N, D)), axis=-1, eps=eps)`
     - Compute empirical density matrix $\rho = \frac{1}{N} \psi_{\text{norm}}^\dagger \psi_{\text{norm}}$ of shape `(D, D)`:
       - PyTorch: `rho = torch.matmul(psi_norm.conj().transpose(-2, -1), psi_norm) / float(N)`
       - JAX: `rho = jnp.matmul(jnp.swapaxes(jnp.conj(psi_norm), -2, -1), psi_norm) / float(N)`
     - Symmetrize $\rho \leftarrow \frac{1}{2}(\rho + \rho^\dagger)$ and return `normalize_trace(rho, eps=eps)` of shape `(D, D)`.

2. **`effective_quantum_rank(psi_or_rho, eps: float = 1e-8, *, is_density: bool = False)`**:
   - **Signature**:
     - PyTorch: `def effective_quantum_rank(psi_or_rho: torch.Tensor, eps: float = 1e-8, *, is_density: bool = False) -> torch.Tensor:`
     - JAX: `def effective_quantum_rank(psi_or_rho: jax.Array, eps: float = 1e-8, *, is_density: bool = False) -> jax.Array:`
   - **Static `is_density` Dispatch (100% `jax.jit` and `torch.compile` safe, zero data-dependent branching)**:
     - When `is_density=False` (default): `psi_or_rho` is treated as a batch of state vectors of shape `(..., D)` (even when $N == D$, e.g., $\psi = I_{16}$). Compute $\rho = \text{empirical\_state\_density}(\text{psi\_or\_rho}, \text{eps}=\text{eps})$ of shape `(D, D)`.
     - When `is_density=True`: `psi_or_rho` must be a square density matrix of shape `(..., D, D)` (raise `ValueError` if `psi_or_rho.ndim < 2` or `psi_or_rho.shape[-2] != psi_or_rho.shape[-1]`). Convert via `rho = ensure_complex(psi_or_rho)`, symmetrize `rho = 0.5 * (rho + rho.conj().transpose(-2, -1))` (or `jnp.swapaxes`), and trace-normalize via `rho = normalize_trace(rho, eps=eps)`.
     - Compute entropy $S(\rho) = \text{von\_neumann\_entropy}(\rho, \text{eps}=\text{eps})$ and return $\exp(S(\rho))$ (`torch.exp` / `jnp.exp`).

3. **`purity_participation_ratio(psi_or_rho, eps: float = 1e-8, *, is_density: bool = False)`**:
   - **Signature**:
     - PyTorch: `def purity_participation_ratio(psi_or_rho: torch.Tensor, eps: float = 1e-8, *, is_density: bool = False) -> torch.Tensor:`
     - JAX: `def purity_participation_ratio(psi_or_rho: jax.Array, eps: float = 1e-8, *, is_density: bool = False) -> jax.Array:`
   - **Computation**:
     - Obtain Hermitian trace-normalized density matrix `rho` using the exact same `is_density: bool = False` rule as `effective_quantum_rank` (`empirical_state_density` when `is_density=False`; symmetrize + `normalize_trace` when `is_density=True`).
     - Compute purity $\gamma = \text{Re}(\text{Tr}(\rho^2)) = \sum_{i,j} |\rho_{ij}|^2$:
       - PyTorch: `purity = torch.sum(torch.abs(rho) ** 2, dim=(-2, -1))`
       - JAX: `purity = jnp.sum(jnp.abs(rho) ** 2, axis=(-2, -1))`
     - Return `1.0 / torch.clamp(purity, min=eps)` (PyTorch) or `1.0 / jnp.maximum(purity, eps)` (JAX).

---

### 3.2 Horizon-Specific POVM Temperatures & Strided MTP in PyTorch (`src/qdlblocks/models/born_gpt.py`)

Update `QuantumBornGPT` in `src/qdlblocks/models/born_gpt.py`:

1. **New Constructor Arguments in `QuantumBornGPT.__init__` (all with backward-compatible defaults)**:
   ```python
   mtp_horizons: int = 1,
   mtp_loss_weight: float = 0.25,
   mtp_stride: int = 1,
   mtp_detach_readout: bool = True,
   mtp_decoupled_temp: bool = True,
   ```
   - **Validation**:
     - If `mtp_horizons < 1`: raise `ValueError(f"mtp_horizons must be >= 1, got {mtp_horizons}")`.
     - If `mtp_stride < 1`: raise `ValueError(f"mtp_stride must be >= 1, got {mtp_stride}")`.
   - **Attributes & Parameter Registration**:
     - Store `self.mtp_stride = int(mtp_stride)`, `self.mtp_detach_readout = bool(mtp_detach_readout)`, `self.mtp_decoupled_temp = bool(mtp_decoupled_temp)`.
     - If `mtp_horizons > 1 and mtp_decoupled_temp`:
       ```python
       base_t = max(povm_temperature, 1e-4)
       init_betas = [1.0 / (base_t * math.sqrt(float(h))) for h in range(2, mtp_horizons + 1)]
       self.mtp_inv_temps: nn.Parameter | None = nn.Parameter(torch.tensor(init_betas, dtype=torch.float32))
       ```
     - Else (`mtp_horizons <= 1 or not mtp_decoupled_temp`):
       ```python
       self.register_parameter("mtp_inv_temps", None)
       ```
   - **Optimizer Routing (`split_muon_adamw_params`)**:
     - Because `self.mtp_inv_temps` has `ndim == 1`, `split_muon_adamw_params` automatically routes it to `AdamW` alongside `tied_inv_temp`.

2. **Exact Forward Pass Logic (`QuantumBornGPT.forward`) across All Readout Modes**:
   - **Primary Horizon ($h=1$, `t+1`)**:
     - In `"asymmetric"` mode (`self.povm_readout is not None`):
       - `logits = self.povm_readout(psi)`
       - `inv_t_primary = self.povm_readout.inv_temp` (either a scalar `torch.Tensor` or `float`)
     - In `"tied_stiefel"` mode:
       - `psi_final = l2_normalize(self.readout_rot(psi), dim=-1, eps=self.eps)`
       - `phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)`
       - `M_seed = l2_normalize(self.wte.weight + self.povm_head_seed_delta, dim=1, eps=self.eps).to(phi_seed.dtype)`
       - `inv_t_primary = torch.clamp(self.tied_inv_temp, min=1.0, max=60.0)`
       - `logits = F.linear(phi_seed, M_seed) * inv_t_primary.to(phi_seed.dtype)`
     - In `"symmetric_dual"` mode:
       - `psi_final = l2_normalize(psi, dim=-1, eps=self.eps)`
       - `phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)`
       - `M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)`
       - `inv_t_primary = self._dual_inv_temp`
       - `logits = F.linear(phi_seed, M_seed) * inv_t_primary`
   - **Auxiliary Horizons ($h_{\text{idx}} \in \{1, \dots, \text{mtp\_horizons}-1\}$) when `targets is not None`**:
     - Initialize `beta_1_val = float(inv_t_primary.detach().item()) if isinstance(inv_t_primary, torch.Tensor) else float(inv_t_primary)`.
     - Initialize `mtp_dict: dict[str, float] = {"ce_t+1": float(loss_t1.detach().item()), "beta_t+1": beta_1_val}`.
     - For `h_idx, u_rot in enumerate(self.mtp_rotations, start=1)`:
       - Let `T_avail = T - h_idx`. If `T_avail <= 0`: `break`.
       - **Static-Shape Strided Slicing**:
         - If `self.mtp_stride > 1 and T_avail > self.mtp_stride`:
           `offset = (h_idx - 1) % self.mtp_stride`
           `state_slice = slice(offset, T_avail, self.mtp_stride)`
           `target_slice = slice(h_idx + offset, T, self.mtp_stride)`
         - Else:
           `state_slice = slice(0, T_avail)`
           `target_slice = slice(h_idx, T)`
         - `tgt_h = targets[:, target_slice].reshape(-1)`
       - **Horizon Inverse Temperature $\beta_{h_{\text{idx}}+1}$**:
         - If `self.mtp_decoupled_temp and self.mtp_inv_temps is not None`:
           `beta_h = torch.clamp(self.mtp_inv_temps[h_idx - 1], min=1.0, max=60.0)`
         - Else:
           `beta_h = inv_t_primary if isinstance(inv_t_primary, torch.Tensor) else torch.tensor(float(inv_t_primary), device=idx.device, dtype=torch.float32)`
       - **Horizon Logits by Readout Mode**:
         - If `self.readout_mode == "asymmetric" and self.povm_readout is not None`:
           - `psi_h = l2_normalize(F.linear(psi[:, state_slice, :], u_rot.to(psi.dtype)), dim=-1, eps=self.eps)`
           - `M_full = l2_normalize(self.povm_readout.povm_head, dim=1, eps=self.eps).to(psi_h.dtype)`
           - `M_aux = M_full.detach() if self.mtp_detach_readout else M_full`
           - `logits_h = F.linear(psi_h, M_aux) * beta_h.to(psi_h.dtype)`
         - Else (`"tied_stiefel"` or `"symmetric_dual"`):
           - `assert phi_seed is not None and M_seed is not None`
           - `z_h = l2_normalize(F.linear(phi_seed[:, state_slice, :], u_rot.to(phi_seed.dtype)), dim=-1, eps=self.eps)`
           - `M_aux = M_seed.detach() if self.mtp_detach_readout else M_seed`
           - `logits_h = F.linear(z_h, M_aux) * beta_h.to(z_h.dtype)`
       - `loss_h = F.cross_entropy(logits_h.reshape(-1, self.vocab_size), tgt_h)`
       - `mtp_dict[f"ce_t+{h_idx + 1}"] = float(loss_h.detach().item())`
       - `mtp_dict[f"beta_t+{h_idx + 1}"] = float(beta_h.detach().item())`
       - `loss = loss + (self.mtp_loss_weight / float(h_idx)) * loss_h`
     - `self.last_mtp_losses = mtp_dict`

3. **Exact `QuantumBornGPT.forward_mtp_logits` Logic**:
   - Returns `[logits_t1, logits_t2, ..., logits_tH]` where each auxiliary horizon $h_{\text{idx}} \ge 1$ uses `beta_h = torch.clamp(self.mtp_inv_temps[h_idx - 1], min=1.0, max=60.0)` when `self.mtp_decoupled_temp and self.mtp_inv_temps is not None`, or `inv_t_primary` otherwise:
     - In `"asymmetric"` mode: `M_full = l2_normalize(self.povm_readout.povm_head, dim=1, eps=self.eps).to(psi.dtype)`, and for each `h_idx, u_rot`, `psi_h = l2_normalize(F.linear(psi, u_rot.to(psi.dtype)), dim=-1, eps=self.eps)`, `out_horizons.append(F.linear(psi_h, M_full) * beta_h.to(psi_h.dtype))`.
     - In `"tied_stiefel"` / `"symmetric_dual"` modes: for each `h_idx, u_rot`, `z_h = l2_normalize(F.linear(phi_seed, u_rot.to(phi_seed.dtype)), dim=-1, eps=self.eps)`, `out_horizons.append(F.linear(z_h, M_seed) * beta_h.to(z_h.dtype))`.

---

### 3.3 Feature Parity in Haliax/JAX (`src/qdlblocks_jax/haliax_modules.py` & `src/qdlblocks_jax/levanter_model.py`)

1. **`HaxQuantumBornGPT` in `src/qdlblocks_jax/haliax_modules.py`**:
   - Add static fields:
     - `mtp_stride: int = eqx.field(static=True)` (default `1`)
     - `mtp_detach_readout: bool = eqx.field(static=True)` (default `True`)
     - `mtp_decoupled_temp: bool = eqx.field(static=True)` (default `True`)
   - Add dynamic field:
     - `mtp_inv_temps: jax.Array` of shape `(max(0, mtp_horizons - 1),)` initialized to `jnp.array([1.0 / (max(povm_temperature, 1e-4) * math.sqrt(float(h))) for h in range(2, mtp_horizons + 1)], dtype=jnp.float32)`.
   - Validation in `__init__`: raise `ValueError` if `mtp_horizons < 1` or `mtp_stride < 1`.
   - In `HaxQuantumBornGPT.__call__`:
     - Extract normalized vocabulary detector array `m_norm_arr = hax_l2_normalize(self.povm_head.m_vocab, axis=self.Hilbert).rearrange((self.Vocab, self.Hilbert)).array` (shape `(V, d)`).
     - For each `h_idx, u_rot in enumerate(self.mtp_rotations, start=1)`:
       - Let `T_avail = self.Seq.size - h_idx`. If `T_avail <= 0`: `break`.
       - Compute `psi_rot = hax.dot(psi, u_rot, axis=self.Hilbert).rename({HilbertOut: self.Hilbert})` and `psi_h_arr = hax_l2_normalize(psi_rot, axis=self.Hilbert).rearrange(seq_axes + (self.Hilbert,)).array` (shape `(..., T, d)`).
       - If `self.mtp_stride > 1 and T_avail > self.mtp_stride`:
         `offset = (h_idx - 1) % self.mtp_stride`
         `psi_sub = psi_h_arr[..., offset : T_avail : self.mtp_stride, :]`
         `tgt_h = targets_ord[..., h_idx + offset : self.Seq.size : self.mtp_stride]`
       - Else:
         `psi_sub = psi_h_arr[..., :T_avail, :]`
         `tgt_h = targets_ord[..., h_idx:]`
       - `m_aux = jax.lax.stop_gradient(m_norm_arr) if self.mtp_detach_readout else m_norm_arr`
       - `beta_h = jnp.clip(self.mtp_inv_temps[h_idx - 1], 1.0, 60.0) if self.mtp_decoupled_temp else jnp.clip(self.povm_head.inv_temp, 1.0, 100.0)`
       - `logits_h = jnp.matmul(psi_sub, m_aux.T) * beta_h`
       - `log_p_h = jax.nn.log_softmax(logits_h, axis=-1)`
       - `nll_h = -jnp.take_along_axis(log_p_h, tgt_h[..., None], axis=-1).squeeze(-1)`
       - `loss = loss + (self.mtp_loss_weight / float(h_idx)) * jnp.mean(nll_h)`
   - In `HaxQuantumBornGPT.forward_mtp_logits`:
     - Uses `beta_h = jnp.clip(self.mtp_inv_temps[h_idx - 1], 1.0, 60.0) if self.mtp_decoupled_temp else jnp.clip(self.povm_head.inv_temp, 1.0, 100.0)` to scale `hax.dot(psi_h, m_norm, axis=self.Hilbert) * hax.NamedArray(beta_h, ())`.

2. **`HaxQuantumBornLMConfig` in `src/qdlblocks_jax/levanter_model.py`**:
   - Add fields `mtp_stride: int = 1`, `mtp_detach_readout: bool = True`, and `mtp_decoupled_temp: bool = True` to `HaxQuantumBornLMConfig` and forward them to `HaxQuantumBornGPT(...)` in `build()`.

3. **Optimizer Routing in `src/qdlblocks_jax/optim.py` (`build_quantum_optax_optimizer`)**:
   - Because `mtp_inv_temps` is a 1D `jax.Array` (`ndim == 1`, not a 2D `NamedArray` on the Stiefel manifold), `build_quantum_optax_optimizer` already routes it to the `AdamW` branch automatically.

---

### 3.4 Benchmark & Telemetry Updates in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`

Update `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` across CLI argument parsing, `run_cuda_benchmark`, and `run_tpu_benchmark`:

1. **New CLI Arguments in `parse_args()`**:
   - `--seed-dim` (`int`, default `128`): Seed Hilbert subspace dimension $D_{\text{seed}}$.
   - `--mtp-stride` (`int`, default `4`): Sequence subsampling stride $s \ge 1$ for auxiliary MTP horizons when `--mtp-horizons > 1`.
   - `--mtp-detach-readout` (` argparse.BooleanOptionalAction`, default `True`): Whether to detach `M_seed` on auxiliary MTP horizons.
   - `--mtp-decoupled-temp` (`argparse.BooleanOptionalAction`, default `True`): Whether to use horizon-specific inverse temperatures `mtp_inv_temps`.

2. **CUDA Path (`run_cuda_benchmark`) Updates**:
   - Pass `seed_dim=seed_dim`, `mtp_stride=mtp_stride`, `mtp_detach_readout=mtp_detach_readout`, `mtp_decoupled_temp=mtp_decoupled_temp` into `QuantumBornGPT(...)`.
   - In `compute_step_loss(m, idx, targets)`:
     - Primary horizon (`t+1`): unchanged (`loss_t1 = _chunked_povm_ce(phi_seed_3d.reshape(-1, m.seed_dim), M_seed, inv_t, targets.reshape(-1))`).
     - Auxiliary horizons (`h_idx` in `1 .. m.mtp_horizons - 1`):
       - Let `T_avail = T_cur - h_idx`.
       - Compute static strided slices `state_slice = slice((h_idx - 1) % m.mtp_stride, T_avail, m.mtp_stride)` and `target_slice = slice(h_idx + ((h_idx - 1) % m.mtp_stride), T_cur, m.mtp_stride)` when `m.mtp_stride > 1 and T_avail > m.mtp_stride`, else `slice(0, T_avail)` and `slice(h_idx, T_cur)`.
       - `z_h = l2_normalize(F.linear(phi_seed_3d[:, state_slice, :], u_rot.to(phi_seed_3d.dtype)), dim=-1, eps=m.eps).reshape(-1, m.seed_dim)`
       - `y_h = targets[:, target_slice].reshape(-1)`
       - `M_aux = M_seed.detach() if m.mtp_detach_readout else M_seed`
       - `beta_h = torch.clamp(m.mtp_inv_temps[h_idx - 1], min=1.0, max=60.0) if (m.mtp_decoupled_temp and m.mtp_inv_temps is not None) else inv_t`
       - `loss_h = _chunked_povm_ce(z_h, M_aux, beta_h, y_h)`
   - In `evaluate_validation`:
     - On `b_idx == 0`, `phi_seed` of shape `(eval_batch_size * block_size, model.seed_dim)` is already computed at line 365 (`phi_seed = phi_3d.reshape(-1, model.seed_dim)`).
     - Compute `seed_eff_rank = float(effective_quantum_rank(phi_seed[:512].float()).item())` and `seed_pr = float(purity_participation_ratio(phi_seed[:512].float()).item())` and include `"seed_eff_rank": round(seed_eff_rank, 2)` and `"seed_participation_ratio": round(seed_pr, 2)` in the returned dictionary.

3. **TPU Path (`run_tpu_benchmark`) Updates**:
   - Accept `seed_dim: int = 128`, `mtp_stride: int = 4`, `mtp_detach_readout: bool = True`, `mtp_decoupled_temp: bool = True`.
   - Set `D_seed = int(seed_dim)`.
   - When `mtp_horizons > 1`: initialize `params["mtp_inv_temps"] = jnp.array([1.0 / (0.035 * math.sqrt(float(h))) for h in range(2, mtp_horizons + 1)], dtype=jnp.float32)` (routed to `update_adam` since `is_muon_leaf("mtp_inv_temps", None)` is `False`).
   - In `loss_for_microbatch`:
     - For each `h_idx in range(1, mtp_horizons)`:
       - Let `T_avail = T - h_idx`.
       - Use static strided slice `offset = (h_idx - 1) % mtp_stride` (`phi_seed_3d[:, offset:T_avail:mtp_stride, :]` and `y_mb[:, h_idx + offset : T : mtp_stride]`) when `mtp_stride > 1 and T_avail > mtp_stride`.
       - Use `M_aux = jax.lax.stop_gradient(M_seed) if mtp_detach_readout else M_seed`.
       - Use `beta_h = jnp.clip(p["mtp_inv_temps"][h_idx - 1], 1.0, 60.0) if mtp_decoupled_temp else jnp.clip(inv_t, 1.0, 60.0)`.
       - Compute `loss_h = povm_ce_chunked(z_h, M_aux, beta_h, y_h)`.
   - In TPU `evaluate_validation(p)`:
     - On `b_idx == 0`, extract `phi_3d_0, _, _, layer_states_0 = encode_backbone(p, x_j, None, False)` and compute `phi_seed_0 = phi_3d_0.reshape(-1, D_seed)[:512].astype(jnp.float32)`.
     - Compute `seed_eff_rank = round(float(effective_quantum_rank(phi_seed_0)), 2)` and `seed_pr = round(float(purity_participation_ratio(phi_seed_0)), 2)` (imported from `qdlblocks_jax.functional`) and include `"seed_eff_rank"` and `"seed_participation_ratio"` in the returned dictionary.

---

### 3.5 Standalone Controlled Hypothesis Verification Script (`examples/12_hypothesis_capacity_and_mfu_experiments.py`)

Create a self-contained, reproducible experiment runner `examples/12_hypothesis_capacity_and_mfu_experiments.py` that runs on CPU or GPU in `< 45` seconds, prints a formatted summary table, asserts all quantitative hypothesis criteria, and writes `.cache/hypothesis_experiments_002.json`:

1. **Experiment 1 (Tests Hypothesis H1 — Shared vs. Decoupled Horizon Temperatures $\boldsymbol{\beta}$)**:
   - **Synthetic Multi-Horizon Markov Dataset**:
     - Vocabulary $V = 512$. Sequence transition rule: for each position $t$, token $x_{t+1} = (3 x_t + 7) \bmod V$ with probability $0.90$ (low conditional entropy at horizon $h=1$), while every 2nd step adds a random branch offset $\delta \in \{0, \dots, 31\}$ so $x_{t+2}$ and $x_{t+3}$ have higher conditional entropy than $x_{t+1}$.
   - **Model & Optimizer Configuration**:
     - `QuantumBornGPT(vocab_size=512, embed_dim=64, seed_dim=32, num_heads=4, num_layers=2, hidden_dim=128, block_size=64, readout_mode="tied_stiefel", learnable_temperature=True, mtp_horizons=3, mtp_loss_weight=0.5, mtp_stride=1)`
     - Trained for `120` steps (`batch_size=16, seq_len=64`) using `StiefelMuon` (`lr=0.025`) + `AdamW` (`lr=0.015`).
     - **Arm A (`Shared-Beta`)**: `mtp_decoupled_temp=False, mtp_detach_readout=False`.
     - **Arm B (`Decoupled-Beta + Detach`)**: `mtp_decoupled_temp=True, mtp_detach_readout=True`.
   - **Quantitative Assertions (matching Section 2 H1)**:
     - `beta_1(Arm B) > beta_2(Arm B) > beta_3(Arm B)`
     - `beta_1(Arm B) > beta_1(Arm A) + 1.0`
     - `val_ce_t1(Arm B) <= val_ce_t1(Arm A) - 0.05`

2. **Experiment 2 (Tests Hypothesis H2 — Strided Multi-Horizon POVM Throughput & MFU Scaling)**:
   - **Benchmark Configuration**:
     - `QuantumBornGPT(vocab_size=16384, embed_dim=128, seed_dim=64, num_heads=4, num_layers=2, hidden_dim=256, block_size=128, readout_mode="tied_stiefel", mtp_horizons=H, mtp_stride=s)` with `batch_size=8, seq_len=128` (`1,024` tokens/step).
     - Sweeps `(H=1, s=1)` (single-horizon baseline) and `(H=3, s)` for $s \in \{1, 2, 4, 8\}$ over `15` timed forward+backward steps (after `3` warmup steps).
   - **Quantitative Assertions (matching Section 2 H2)**:
     - `tok_per_s(H=3, s=4) >= 1.15 * tok_per_s(H=3, s=1)`
     - Analytical auxiliary vocabulary-head FLOPs for `(H=3, s=4)` are $\le 0.26 \times$ those of `(H=3, s=1)`.

3. **Experiment 3 (Tests Hypothesis H3 — Seed Hilbert Dimension $D_{\text{seed}} \in \{8, 16, 32, 64\}$ vs. Effective Quantum Rank $R_{\text{eff}}(\rho_{\text{seed}})$)**:
   - **Dataset & Model Configuration**:
     - 64-motif deterministic cycle language task over `vocab_size=256` (`batch_size=16, seq_len=64`).
     - Fixed backbone Hilbert dimension $d = 128$ (`7` qubits, `num_heads=4, num_layers=2, hidden_dim=256, readout_mode="tied_stiefel"`), sweeping `seed_dim` $D_{\text{seed}} \in \{8, 16, 32, 64\}$ for `100` steps each.
   - **Quantitative Assertions (matching Section 2 H3)**:
     - Strict monotonic ordering of post-training Effective Quantum Rank:
       `R_eff(64) > R_eff(32) > R_eff(16) > R_eff(8)`
     - Monotonic improvement of validation cross-entropy:
       `val_ce(64) < val_ce(8)`

---

## 4. Testing & Verification Plan

### 4.1 Unit Tests (`tests/test_hypothesis_002_features.py`)
Create `tests/test_hypothesis_002_features.py` covering both PyTorch (`qdlblocks`) and JAX/Haliax (`qdlblocks_jax`):
1. **`test_empirical_density_and_effective_rank_bounds_torch_and_jax`**:
   - **Rank-1 Pure State Batch**: When all $N=16$ rows of $\psi \in \mathbb{R}^{16 \times 16}$ equal the same unit vector $|\phi_0\rangle$, verifies `effective_quantum_rank(psi) == 1.0` and `purity_participation_ratio(psi) == 1.0` (`atol=1e-4`) in both PyTorch and JAX.
   - **Orthonormal Basis Batch ($\psi = I_{16}$, $N = D = 16$)**: With default `is_density=False`, verifies $\rho = \text{empirical\_state\_density}(I_{16}) = I_{16} / 16$, `effective_quantum_rank(I_16) == 16.0`, and `purity_participation_ratio(I_16) == 16.0` (`atol=1e-4`) in both PyTorch and JAX (including under `@jax.jit`).
   - **Explicit `is_density=True` Branch**: Verifies that passing an unnormalized positive-definite matrix $2.5 \cdot I_{16}$ with `is_density=True` trace-normalizes it and returns `effective_quantum_rank == 16.0` and `purity_participation_ratio == 16.0` (`atol=1e-4`).
   - **General Rényi-2 vs. von Neumann Inequality**: On random state batches of shape `(8, 32, 16)`, verifies $1.0 \le \text{PR}(\rho) \le R_{\text{eff}}(\rho) \le 16.0$.
2. **`test_decoupled_mtp_temperatures_and_detach_torch`**:
   - Tests `QuantumBornGPT` across all three readout modes (`"tied_stiefel"`, `"symmetric_dual"`, `"asymmetric"`) with `mtp_horizons=3, mtp_decoupled_temp=True, mtp_detach_readout=True` and with `mtp_decoupled_temp=False`.
   - Verifies `model.mtp_inv_temps` is `None` when `mtp_horizons=1` or `mtp_decoupled_temp=False`, and has shape `(2,)` with values $[1 / (\tau_0 \sqrt{2}), 1 / (\tau_0 \sqrt{3})]$ when `mtp_horizons=3, mtp_decoupled_temp=True`.
   - Verifies that with `mtp_detach_readout=True`, auxiliary horizon losses do not contribute gradients to `povm_head_seed_delta` or `tied_inv_temp`, while `mtp_inv_temps.grad` and `mtp_rotations[0].grad` are non-zero and finite.
   - Verifies `model.last_mtp_losses` contains `"ce_t+1"`, `"ce_t+2"`, `"ce_t+3"`, `"beta_t+1"`, `"beta_t+2"`, `"beta_t+3"`.
3. **`test_strided_mtp_shapes_and_jax_parity`**:
   - Tests `QuantumBornGPT` (PyTorch) and `HaxQuantumBornGPT` / `HaxQuantumBornLMConfig` (JAX/Haliax) with `mtp_horizons=3, mtp_stride=4, mtp_detach_readout=True, mtp_decoupled_temp=True`.
   - Verifies finite loss, finite gradients, static slice shapes across steps, and valid `forward_mtp_logits` output shapes.
4. **`test_full_pytest_suite_regression`**:
   - Run `/home/festivity/agy/QuantumDLBlocks/.venv/bin/pytest` across the entire `tests/` suite and verify 100% pass rate.
5. **`test_hypothesis_experiment_script_execution`**:
   - Run `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python examples/12_hypothesis_capacity_and_mfu_experiments.py` and verify all three empirical hypothesis experiments (`H1`, `H2`, `H3`) pass their quantitative assertions and write `.cache/hypothesis_experiments_002.json`.
