# Experiment Plan 013: Contextual Bloch-Purity Dynamic POVM Sharpness, Karpathy Overfitting Capacity, & Apples-to-Apples Classical Modded-NanoGPT Baseline

## 1. Context, Motivation & Problem Statement

Across Waves 1 through 5 on the 900M-token FineWeb speedrun benchmark (`vocab_size = 50,304`, `seq_len = 512`, `batch_size = 128`, `13,725` steps = `899.48M` tokens), `QuantumBornGPT` has progressed from an initial multi-epoch baseline of `4.3488` Val CE (`77.4` PPL) to:
- **Wave-4 Record (`107.6M` params)**: `w4_900m_full_synergy512_cosine` at **`3.9223` Val CE (`50.5` PPL, `32.98%` Top-1)**,
- **Wave-5 `#2` Record (`155.6M` params)**: `w5_900m_asym1024_deep9` (Full-Rank `50,304 × 1024` Asymmetric POVM + `L=9` Deep) at **`3.8825` Val CE (`48.5` PPL, `33.34%` Top-1, $R_{\text{eff}} = 72.62$)**, and
- **Wave-5 `#1` All-Time Record (`162.2M` params)**: `w5_900m_frame4d_synergy512` (`4d` `4096×2` Overcomplete Two-Beam Tight Frame + `D_seed=512` + `TwoBeamSeedReadoutAdapter`) at **`3.8765` Val CE (`48.3` PPL, `33.50%` Top-1, `212,020 tok/s`)**.

Two critical scientific questions now stand at the frontier of the project:

### Question 1: Final-Layer Sharpness & The Constant-Logit-Norm Theorem on $\mathbb{S}^{D-1}$
In `QuantumBornGPT` (`src/qdlblocks/models/born_gpt.py` and `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`), the final contextual state $|\phi_{b,t}\rangle \in \mathbb{S}^{D_{\text{out}}-1}$ (where $D_{\text{out}} = D_{\text{seed}}$ for tied/dual readouts or $D_{\text{out}} = d$ for asymmetric readout) and every vocabulary POVM measurement vector $|m_v\rangle \in \mathbb{S}^{D_{\text{out}}-1}$ ($v \in \{0, \dots, V-1\}$) are strictly L2-normalized to unit length ($\||\phi_{b,t}\rangle\|_2 = 1$, $\||m_v\rangle\|_2 = 1$). Logits are computed via a **single global scalar inverse temperature** $\beta = \text{inv\_temp} \in [1, 60]$:
$$\ell_{b,t,v} = \beta \langle \phi_{b,t} | m_v \rangle \in [-\beta, +\beta]$$

When the $V$ unit vocabulary vectors form an approximately isotropic tight frame on $\mathbb{S}^{D_{\text{out}}-1}$ ($M^\top M \approx \frac{V}{D_{\text{out}}} I_{D_{\text{out}}}$), the L2 norm of the logit vector $\ell_{b,t} = \beta M |\phi_{b,t}\rangle \in \mathbb{R}^V$ satisfies the **Constant-Logit-Norm Theorem**:
$$\|\ell_{b,t}\|_2 = \beta \sqrt{\langle \phi_{b,t} | M^\top M | \phi_{b,t} \rangle} \approx \beta \sqrt{\frac{V}{D_{\text{out}}}} = \text{constant across all token positions } (b, t)$$

This creates a fundamental **Heteroscedastic Sharpness Bottleneck**:
1. **In Pure Memorization (Karpathy Overfitting Regime, $N$ small records)**: Every target token $(b, t)$ is deterministic ($H^*(Y_{b,t} \mid X_{b,\le t}) = 0$). A single global $\beta$ can satisfy all tokens simultaneously once $D_{\text{out}}$ is large enough to avoid spherical crowding on $\mathbb{S}^{D_{\text{out}}-1}$. However, because $\beta$ is a single un-reparameterized scalar updated with AdamW (`lr = 0.004`) and clamped to `60.0`, output entropy $H(P_{b,t})$ remains $4\times\text{–}10\times$ higher than in an unconstrained Euclidean readout.
2. **In Real Language Modeling (Heteroscedastic Regime on 900M FineWeb Tokens)**: True conditional entropy $H^*(Y_{b,t} \mid X_{b,\le t})$ varies by over two orders of magnitude across token positions within the same sequence:
   - **High-certainty deterministic contexts** (e.g., subword continuations `" Einst"` $\to$ `"ein"`, `"San "` $\to$ `"Francisco"`) require high inverse temperature ($\beta_{b,t} \approx 75\text{–}120$) to concentrate probability mass on the unique continuation.
   - **Low-certainty branching contexts** (e.g., after a sentence period `". "`) require low inverse temperature ($\beta_{b,t} \approx 12\text{–}22$) to keep the distribution flat and avoid severe $-\log P(y_{b,t})$ penalties when plausible low-rank tokens occur.
   - With a single global scalar $\beta$, optimization is trapped in a dataset-wide compromise ($\beta \approx 38\text{–}43$ across all Wave-4 and Wave-5 runs), simultaneously **under-sharpening deterministic contexts** and **over-sharpening uncertain contexts**.

**Quantum-Physical Solution — Contextual Bloch-Ball Purity Readout (`ContextualBlochPurityGate`)**:
In quantum mechanics, a state prepared by an open or contextually uncertain channel resides inside the **Bloch ball** with purity radius $r_{b,t} \in [r_{\min}, r_{\max}]$ rather than on the pure surface $r \equiv 1$. Measuring a Bloch vector $\vec{r}_{b,t} = r_{b,t} |\phi_{b,t}\rangle$ against unit POVM effects $|m_v\rangle$ yields effective per-token inverse temperature:
$$u(\psi_{b,t}) = \text{SiLU}(\psi_{b,t} W_1^\top + b_1) W_2^\top + b_2, \qquad r(\psi_{b,t}) = \text{clamp}\!\left(\frac{\text{softplus}(u(\psi_{b,t}))}{\text{softplus}(b_{2,0})},\, \text{min}=r_{\min},\, \text{max}=r_{\max}\right)$$
$$\beta_{b,t} = \text{clamp}\!\left(\beta_0 \cdot r(\psi_{b,t}),\, \text{min}=1.0,\, \text{max}=\beta_{\max}\right)$$
where $\psi_{b,t} \in \mathbb{S}^{d-1}$ is the final unit-normalized Hilbert state before collapse, $W_1 \in \mathbb{R}^{H_g \times d}$ is orthogonal-initialized, $b_1 \in \mathbb{R}^{H_g}$ is zero-initialized, $W_2 \in \mathbb{R}^{1 \times H_g}$ is zero-initialized, and $b_2 \in \mathbb{R}^1$ is initialized to $b_{2,0} = 1.0$ so that at step 0, $u(\psi_{b,t}) \equiv b_{2,0}$ and $r(\psi_{b,t}) \equiv 1.0 \in [r_{\min}, r_{\max}]$ ($r_{\min} = 0.25, r_{\max} = 4.0, \beta_{\max} = 160.0$). Because `purity_gate` parameters are routed exclusively to `AdamW` (never `StiefelMuon`), the zero-initialized $W_2$ is preserved at step 0 and smoothly learns context-dependent sharpness modulation.

### Question 2: Apples-to-Apples Single-GPU Classical `Modded-NanoGPT` Baseline
To rigorously benchmark `QuantumBornGPT` against the strongest modern classical Transformer design (derived from `KellerJordan/modded-nanogpt` single-GPU architectural primitives, excluding `8×H100` distributed sharding and the 65B-parameter sharded n-gram lookup table), we need a first-class `ClassicalModdedGPT` model in `src/qdlblocks/models/classical_modded_gpt.py` and `--architecture {quantum_born,classical_modded}` in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` featuring:
1. Parameterless `RMSNorm` before attention and MLP,
2. Rotary Positional Embeddings (`UnitaryRoPE`) + QK-RMSNorm Causal Softmax Attention (`F.scaled_dot_product_attention`),
3. $\text{ReLU}^2$ MLP (`F.relu(x @ W1).square() @ W2`),
4. Learnable residual/input highway mixing and U-Net encoder-to-decoder skip connections matching `QuantumBornGPT`,
5. Untied Euclidean linear readout `lm_head` (`50,304 × d`) with logit softcapping (`softcap * torch.tanh(logits / softcap)` at `softcap = 30.0`),
6. Trained under the exact same `StiefelMuon` (`stiefel_penalty=0.0`, `retract_every=1000000`, i.e., pure Newton-Schulz Muon on 2D internal weights) + `AdamW` (on embeddings, `lm_head`, and scalar gates), same data stream (`FineWebMultiShardStream`), same batch size (`B=128, T=512`), and same evaluation cadence (`every 50M tokens`).

---

## 2. Concrete Hypotheses & Quantitative Pass/Fail Criteria

### Hypothesis 1 (H1 — Constant-Logit-Norm Bottleneck & Heteroscedastic Sharpness Recovery via `ContextualBlochPurityGate`)
- **Claim**:
  1. Under standard global-$\beta$ spherical readout ($\ell_{b,t} = \beta M |\phi_{b,t}\rangle$ with unit-norm $|\phi_{b,t}\rangle \in \mathbb{S}^{D_{\text{out}}-1}$ and unit-row $M \in \mathbb{R}^{V \times D_{\text{out}}}$), the coefficient of variation of the per-token logit L2 norm across token positions,
     $$\text{CV}(\|\ell\|_2) = \frac{\text{std}_{b,t}(\|\ell_{b,t}\|_2)}{\text{mean}_{b,t}(\|\ell_{b,t}\|_2)},$$
     is strictly bounded near zero ($\text{CV}(\|\ell\|_2) < 0.08$), proving that global-$\beta$ spherical readout cannot dynamically modulate logit energy across heteroscedastic contexts.
  2. On a controlled **Heteroscedastic Contextual Language Modeling Task** (`V = 512`, `d = 128`, `D_seed = 64`) where half of the contexts are **deterministic transitions** (true target entropy $H^* = 0.0$ nats, single ground-truth continuation with probability $1.0$) and half are **high-entropy branching transitions** (uniform over $K = 16$ valid continuations, true target entropy $H^* = \ln 16 \approx 2.7726$ nats), equipping `QuantumBornGPT` with `ContextualBlochPurityGate` (`use_dynamic_povm_beta=True`):
     - Achieves at least **`0.15` nats lower total validation Cross-Entropy** ($\text{CE}_{\text{dyn}} \le \text{CE}_{\text{static}} - 0.15$) than identical `QuantumBornGPT` with static global $\beta$,
     - Learns to separate effective inverse temperature by context type such that $\bar{\beta}_{\text{deterministic}} \ge 1.35 \cdot \bar{\beta}_{\text{branching}}$, and
     - Increases per-token logit norm modulation to $\text{CV}(\|\ell\|_2) \ge 0.15$.

### Hypothesis 2 (H2 — Karpathy Overfitting Capacity Scaling with $D_{\text{seed}}$ & Dynamic Purity Gate)
- **Claim**: On a controlled **Karpathy Sequence Memorization Benchmark** (`N = 32` sequence records of length `T = 64`, `2,048` total tokens over vocabulary `V = 1,024`, trained for `120` steps):
  1. Expanding seed readout capacity from `D_seed = 16` (`tied_stiefel`, static $\beta$) to `D_seed = 64` with `OvercompleteSeedReadoutAdapter` and `ContextualBlochPurityGate` reduces final memorization Training Cross-Entropy by at least **`50%`** ($\text{CE}_{D=64,\text{dyn}} \le 0.50 \cdot \text{CE}_{D=16,\text{static}}$) and achieves **`Top-1 Memorization Accuracy >= 98.0%`**.
  2. `ClassicalModdedGPT` with matched layer/head/width dimensions (`L = 2, d = 128, H = 4`) also achieves **`Top-1 Memorization Accuracy >= 98.0%`** and final memorization Training Cross-Entropy $\text{CE}_{\text{classical}} \le 0.10$, validating `ClassicalModdedGPT` as a high-capacity classical baseline.

---

## 3. Planned Code & Architectural Changes

### 3.1 New Module: `src/qdlblocks/layers/dynamic_povm_purity.py`
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/dynamic_povm_purity.py` implementing:
1. `compute_logit_norm_variation_metrics(logits: torch.Tensor, eps: float = 1e-8) -> dict[str, float]`:
   - Accepts `logits` of shape `(..., V)` (flattening leading batch/sequence dimensions to `(N_tok, V)` in `float32`).
   - Computes per-token L2 norms $n_i = \|\ell_i\|_2$, per-token softmax entropies $H_i = -\sum_v p_{i,v} \log(p_{i,v} + \epsilon)$, and per-token max probabilities $P_{\max,i} = \max_v p_{i,v}$.
   - Returns:
     - `"logit_norm_mean"`: `float(n.mean().item())`
     - `"logit_norm_std"`: `float(n.std(unbiased=False).item())`
     - `"logit_norm_cv"`: `float((n.std(unbiased=False) / (n.mean() + eps)).item())`
     - `"mean_entropy_nats"`: `float(H.mean().item())`
     - `"mean_max_prob"`: `float(P_max.mean().item())`
2. `class ContextualBlochPurityGate(nn.Module)`:
   - Constructor signature:
     ```python
     def __init__(
         self,
         embed_dim: int,
         hidden_dim: int = 64,
         r_min: float = 0.25,
         r_max: float = 4.0,
         init_bias: float = 1.0,
         max_beta: float = 160.0,
         eps: float = 1e-8,
     ) -> None:
     ```
   - Validates `embed_dim >= 1`, `hidden_dim >= 1`, `0.0 < r_min < r_max`, `max_beta >= 1.0`.
   - Defines a 2-layer bottleneck MLP directly on the unit-normalized Hilbert ket $\psi \in \mathbb{S}^{d-1}$ (no `LayerNorm` submodule is used, ensuring exact parity with JAX):
     - `self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=True)` initialized with `nn.init.orthogonal_(self.fc1.weight)` and `nn.init.zeros_(self.fc1.bias)`.
     - `self.fc2 = nn.Linear(hidden_dim, 1, bias=True)` initialized with `nn.init.zeros_(self.fc2.weight)` and `nn.init.constant_(self.fc2.bias, float(init_bias))`.
     - Stores `self.r_min = float(r_min)`, `self.r_max = float(r_max)`, `self.init_bias = float(init_bias)`, `self.max_beta = float(max_beta)`, `self.eps = float(eps)`, and precomputes `self._s0 = float(F.softplus(torch.tensor(float(init_bias), dtype=torch.float32)).item())`.
     - For any input `psi` of shape `(..., embed_dim)`:
       ```python
       psi_norm = l2_normalize(psi, dim=-1, eps=self.eps)
       h = F.silu(self.fc1(psi_norm.to(self.fc1.weight.dtype)))
       u = self.fc2(h).float()
       r = torch.clamp(F.softplus(u) / self._s0, min=self.r_min, max=self.r_max)
       ```
       At initialization (`fc2.weight == 0`, `fc2.bias == init_bias`), $u \equiv \text{init\_bias}$, so $r(\psi) \equiv 1.0$.
   - Method `forward(self, psi: torch.Tensor, base_inv_temp: torch.Tensor | float) -> torch.Tensor`:
     - Computes `r` of shape `(..., 1)` in `float32`, converts `base_inv_temp` to a `float32` tensor/scalar on `psi.device`, and returns:
       ```python
       beta_t = torch.clamp(base_t * r, min=1.0, max=self.max_beta)
       return beta_t
       ```
       of shape `(..., 1)` in `float32`.
   - Method `scale_logits(self, psi: torch.Tensor, cos_logits: torch.Tensor, base_inv_temp: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]`:
     - Computes `beta_t = self.forward(psi, base_inv_temp)` of shape `(..., 1)` and returns `(cos_logits * beta_t.to(cos_logits.dtype), beta_t)`.

### 3.2 New Module: `src/qdlblocks/models/classical_modded_gpt.py`
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/classical_modded_gpt.py` implementing:
1. `class ClassicalModdedBlock(nn.Module)`:
   - Constructor signature:
     ```python
     def __init__(
         self,
         embed_dim: int = 1024,
         num_heads: int = 16,
         hidden_dim: int = 2048,
         value_residual_mixing: bool = True,
         input_highway_mixing: bool = True,
         residual_scale: float = 1.0,
         eps: float = 1e-6,
     ) -> None:
     ```
   - Submodules & Parameters:
     - `self.embed_dim = int(embed_dim)`, `self.num_heads = int(num_heads)`, `self.head_dim = embed_dim // num_heads`, `self.hidden_dim = int(hidden_dim)`, `self.residual_scale = float(residual_scale)`, `self.eps = float(eps)`.
     - `self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)`
     - `self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)`
     - `self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)`
     - `self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)`
     - `self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=False)`
     - `self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=False)`
     - If `input_highway_mixing=True`: `self.highway_gate = nn.Parameter(torch.zeros(1, 1, embed_dim))` else `self.register_parameter("highway_gate", None)`.
     - If `value_residual_mixing=True`: `self.v_mix_gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1))` else `self.register_parameter("v_mix_gate", None)`.
   - Exact `forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, x_0: torch.Tensor | None = None) -> torch.Tensor`:
     1. **Input Highway Mixing**: If `self.highway_gate is not None and x_0 is not None`:
        `x = x + (0.15 * torch.tanh(self.highway_gate)).to(x.dtype) * x_0`
     2. **Pre-Attention RMSNorm & QKV**:
        `B, T, D = x.shape`
        `h = F.rms_norm(x, (D,), eps=self.eps)`
        `q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim)`
        `k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim)`
        `v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)`
     3. **RoPE + QK-RMSNorm**:
        Apply `apply_rotary_emb(q, cos, sin)` and `apply_rotary_emb(k, cos, sin)` (where `apply_rotary_emb` is imported from `qdlblocks.layers.rope`), transpose `q, k` to `(B, num_heads, T, head_dim)`, and apply `q = F.rms_norm(q, (self.head_dim,), eps=self.eps)` and `k = F.rms_norm(k, (self.head_dim,), eps=self.eps)`.
     4. **Value Residual Mixing**: If `self.v_mix_gate is not None and x_0 is not None`:
        Compute `v_0 = self.v_proj(F.rms_norm(x_0, (D,), eps=self.eps)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)` and mix:
        `alpha_v = (0.5 * torch.sigmoid(self.v_mix_gate)).to(v.dtype)`
        `v = (1.0 - alpha_v) * v + alpha_v * v_0`
     5. **Causal SDPA & Residual**:
        `attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)`
        `attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)`
        `x = x + self.residual_scale * self.out_proj(attn_out)`
     6. **Pre-MLP RMSNorm & $\text{ReLU}^2$ MLP**:
        `h_mlp = F.rms_norm(x, (D,), eps=self.eps)`
        `mlp_out = self.fc2(F.relu(self.fc1(h_mlp)).square())`
        `x = x + self.residual_scale * mlp_out`
        `return x`

2. `class ClassicalModdedGPT(nn.Module)`:
   - Constructor signature:
     ```python
     def __init__(
         self,
         vocab_size: int = 50304,
         embed_dim: int = 1024,
         num_heads: int = 16,
         num_layers: int = 6,
         hidden_dim: int = 2048,
         block_size: int = 1024,
         use_bigram_smear: bool = True,
         use_unet_interferometry: bool = True,
         value_residual_mixing: bool = True,
         input_highway_mixing: bool = True,
         depth_scaled_residual: bool = False,
         logit_softcap: float = 30.0,
         tie_embeddings: bool = False,
         eps: float = 1e-6,
     ) -> None:
     ```
   - Submodules & Parameters:
     - Computes `self.residual_scale = compute_depth_residual_scale(num_layers, base_layers=6, enabled=depth_scaled_residual)` (imported from `qdlblocks.layers.parameter_scaling`).
     - `self.wte = nn.Embedding(vocab_size, embed_dim)`
     - If `use_bigram_smear=True`: `self.smear_embed_gate = nn.Parameter(torch.zeros(1, 1, embed_dim))` else `self.register_parameter("smear_embed_gate", None)`.
     - `self.rope = UnitaryRoPE(head_dim=embed_dim // num_heads, max_seq_len=block_size)`
     - `self.blocks = nn.ModuleList([ClassicalModdedBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, value_residual_mixing=value_residual_mixing, input_highway_mixing=input_highway_mixing, residual_scale=self.residual_scale, eps=eps) for _ in range(num_layers)])`
     - If `use_unet_interferometry=True` and `num_layers >= 2`:
       `n_half = num_layers // 2`
       `self.unet_gates = nn.ParameterList([nn.Parameter(torch.eye(embed_dim)) for _ in range(n_half)])`
       `self.unet_scales = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim)) for _ in range(n_half)])`
       else `self.unet_gates = None`, `self.unet_scales = None`.
     - `self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)` (if `tie_embeddings=True`, `self.lm_head.weight = self.wte.weight`).
   - Exact `forward_features(self, idx: torch.Tensor, return_layer_states: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]`:
     - `B, T = idx.shape`
     - `x_0 = F.rms_norm(self.wte(idx), (self.embed_dim,), eps=self.eps)`
     - If `self.use_bigram_smear and T > 1 and self.smear_embed_gate is not None`:
       `x_prev = F.pad(x_0[:, :-1, :], (0, 0, 1, 0))`
       `eta = (0.06 * torch.tanh(self.smear_embed_gate)).to(x_0.dtype)`
       `x_0 = F.rms_norm(x_0 + eta * x_prev, (self.embed_dim,), eps=self.eps)`
     - `x = x_0`, `encoder_states = []`, `layer_states = []`, `n_half = self.num_layers // 2`, `cos, sin = self.rope.get_cos_sin(T)`
     - Loop `for layer_idx, block in enumerate(self.blocks)`:
       `x = block(x, cos=cos, sin=sin, x_0=x_0)`
       If `self.use_unet_interferometry and self.unet_gates is not None and self.unet_scales is not None`:
       - If `layer_idx < n_half`: `encoder_states.append(x)`
       - Else if `layer_idx >= self.num_layers - n_half`:
         `enc_idx = self.num_layers - 1 - layer_idx`
         `skip_x = encoder_states[enc_idx]`
         `u_scale = (0.35 * torch.tanh(self.unet_scales[enc_idx])).to(x.dtype)`
         `skip_rot = F.linear(skip_x, self.unet_gates[enc_idx].to(x.dtype))`
         `x = x + u_scale * F.rms_norm(skip_rot, (self.embed_dim,), eps=self.eps)`
       If `return_layer_states`: `layer_states.append(x)`
     - `x_final = F.rms_norm(x, (self.embed_dim,), eps=self.eps)`
     - `return x_final, layer_states`
   - Exact `compute_logits(self, x_final: torch.Tensor) -> torch.Tensor`:
     - `raw = F.linear(x_final, self.lm_head.weight.to(x_final.dtype))`
     - If `self.logit_softcap > 0.0`: `return self.logit_softcap * torch.tanh(raw / self.logit_softcap)`
     - Else: `return raw`
   - Exact `forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_layer_states: bool = False)`:
     - Calls `x_final, layer_states = self.forward_features(idx, return_layer_states=return_layer_states)` and `logits = self.compute_logits(x_final)`.
     - If `targets is not None`: `loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)` else `loss = None`.
     - If `return_layer_states`: returns `(logits, loss, layer_states)` else `(logits, loss)`.

### 3.3 Update `QuantumBornGPT` (`src/qdlblocks/models/born_gpt.py`)
1. Add `use_dynamic_povm_beta: bool = False` to `QuantumBornGPT.__init__`.
2. Store `self.use_dynamic_povm_beta = bool(use_dynamic_povm_beta)` and `self._last_mean_beta: float = float(1.0 / max(povm_temperature, 1e-4))`.
3. When `use_dynamic_povm_beta=True`, instantiate `self.purity_gate: ContextualBlochPurityGate | None = ContextualBlochPurityGate(embed_dim=embed_dim)` (else `None`).
4. In `QuantumBornGPT.forward`:
   - Across all 4 readout modes (`asymmetric`, `tied_stiefel`, `tied_vocab`, `symmetric_dual`), compute the unit-norm directional cosine similarity matrix `cos_sim` and scalar `inv_t_primary`:
     - For `asymmetric`: `phi_seed = l2_normalize(psi, dim=-1, eps=self.eps)`, `M_seed = l2_normalize(self.povm_readout.povm_head, dim=1, eps=self.eps).to(phi_seed.dtype)`, `inv_t_primary = torch.clamp(self.povm_readout.inv_temp, min=1.0, max=60.0)`, `cos_sim = F.linear(phi_seed, M_seed)`.
     - For `tied_stiefel` / `tied_vocab`: `psi_final = l2_normalize(self.readout_rot(psi), dim=-1, eps=self.eps)`, `phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)`, apply `self.seed_adapter(psi_final, phi_seed)` if `self.seed_adapter is not None` (which handles both 1-beam `OvercompleteSeedReadoutAdapter` and 2-beam `TwoBeamSeedReadoutAdapter`), form `M_seed`, `inv_t_primary = torch.clamp(self.tied_inv_temp, min=1.0, max=60.0)`, `cos_sim = F.linear(phi_seed, M_seed)`.
     - For `symmetric_dual`: `psi_final = l2_normalize(psi, dim=-1, eps=self.eps)`, `phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)`, apply `self.seed_adapter` if present, `M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)`, `inv_t_primary = float(self._dual_inv_temp)`, `cos_sim = F.linear(phi_seed, M_seed)`.
   - Then:
     - If `self.purity_gate is not None`:
       `logits, beta_t = self.purity_gate.scale_logits(psi, cos_sim, inv_t_primary)`
       `self._last_mean_beta = float(beta_t.detach().mean().item())`
     - Else:
       - In `asymmetric` mode (when `self.purity_gate is None`), preserve existing `logits = self.povm_readout(psi)` behavior and `self._last_mean_beta = float(inv_t_primary.detach().item())`.
       - In `tied_stiefel` / `tied_vocab` / `symmetric_dual` modes (when `self.purity_gate is None`), `logits = cos_sim * (inv_t_primary.to(phi_seed.dtype) if isinstance(inv_t_primary, torch.Tensor) else inv_t_primary)` and update `self._last_mean_beta`.

### 3.4 New JAX/Equinox/Haliax Module: `src/qdlblocks_jax/dynamic_povm_purity.py`
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/dynamic_povm_purity.py` implementing:
1. `compute_logit_norm_variation_metrics_jax(logits: jnp.ndarray, eps: float = 1e-8) -> dict[str, float]`:
   - Matches `compute_logit_norm_variation_metrics` in PyTorch (flattening to `(N_tok, V)`, computing L2 norms, softmax entropies, and max probabilities).
2. `contextual_bloch_purity_beta_jax(psi: jnp.ndarray, w1: jnp.ndarray, b1: jnp.ndarray, w2: jnp.ndarray, b2: jnp.ndarray, base_inv_temp: float = 28.57, r_min: float = 0.25, r_max: float = 4.0, init_bias: float = 1.0, max_beta: float = 160.0, eps: float = 1e-8) -> jnp.ndarray`:
   - Normalizes `psi_norm = psi / jnp.sqrt(jnp.sum(jnp.square(psi), axis=-1, keepdims=True) + eps)`.
   - Computes `h = jax.nn.silu(jnp.matmul(psi_norm, jnp.swapaxes(w1, -1, -2)) + b1)`.
   - Computes `u = jnp.matmul(h, jnp.swapaxes(w2, -1, -2)) + b2`.
   - Computes `s0 = jax.nn.softplus(jnp.asarray(init_bias, dtype=jnp.float32))`.
   - Computes `r = jnp.clip(jax.nn.softplus(u) / s0, r_min, r_max)`.
   - Returns `jnp.clip(base_inv_temp * r, 1.0, max_beta)` of shape `(..., 1)`.
Export both in `src/qdlblocks_jax/__init__.py` and export the PyTorch symbols (`ContextualBlochPurityGate`, `compute_logit_norm_variation_metrics`, `ClassicalModdedBlock`, `ClassicalModdedGPT`) in `src/qdlblocks/layers/__init__.py`, `src/qdlblocks/models/__init__.py`, and `src/qdlblocks/__init__.py`.

### 3.5 Update `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`
Add CLI arguments in `parse_args`:
- `--architecture` (`choices=["quantum_born", "classical_modded"]`, default `"quantum_born"`)
- `--dynamic-povm-beta` (`action="store_true"`, default `False`)
Wire both cleanly into `run_cuda_benchmark`, `run_tpu_benchmark`, and `main`:
1. **Optimizer Split (`adamw_name_patterns`)**:
   - Add `"purity_gate"`, `"lm_head"`, `"highway_gate"`, and `"v_mix_gate"` to `adamw_name_patterns` in `split_muon_adamw_params` so that `ContextualBlochPurityGate` parameters (`fc1`, `fc2`) and `ClassicalModdedGPT` output head / scalar gates are always optimized by `AdamW` (never `StiefelMuon`).
2. **`QuantumBornGPT` with `dynamic_povm_beta=True` on CUDA**:
   - Pass `use_dynamic_povm_beta=dynamic_povm_beta` to `QuantumBornGPT(...)`.
   - In `forward_backbone`, when `m.purity_gate is not None`, replace scalar `inv_t` with per-token `inv_t = m.purity_gate(psi, inv_t)` of shape `(B_cur, T_cur, 1)`.
   - In `_chunked_povm_ce`, when `inv_t.ndim >= 2`, flatten `inv_t_flat = inv_t.reshape(-1, 1)` and slice `inv_t_flat[i * c_sz : (i + 1) * c_sz]` into `compiled_chunk_ce` so each token $(b, t)$ scales its own row of logits `F.linear(zc, M_seed) * inv_t_c.to(zc.dtype)`.
   - In `evaluate_validation`, compute `logits = F.linear(phi_seed, M_seed) * inv_t.to(phi_seed.dtype)` (which broadcasts `(B, T, 1)` cleanly against `(B, T, V)`) and log `povm_beta = float(inv_t.mean().item())`.
3. **`ClassicalModdedGPT` (`architecture == "classical_modded"`) on CUDA**:
   - Instantiate `ClassicalModdedGPT(vocab_size=50304, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, hidden_dim=int(round(embed_dim * frame_expansion)) if (overcomplete_two_beam or overcomplete_frame) else hidden_dim, block_size=max_T, use_bigram_smear=True, use_unet_interferometry=True, value_residual_mixing=True, input_highway_mixing=True, depth_scaled_residual=depth_scaled_residual, logit_softcap=30.0)`.
   - Optimize 2D internal weights with Newton-Schulz Muon (`StiefelMuon(..., stiefel_penalty=0.0, retract_every=1_000_000)`) and embeddings/`lm_head`/gates with `AdamW`.
   - Log the exact same `[Epoch ...]` and `-> Step ...` format (`povm_beta` logs mean per-token logit standard deviation, `seed_eff_rank` logs effective rank of normalized `x_final` features, and `S(rho_A)` logs 10-qubit entanglement entropy of normalized `x_final[:, :1024]`).
4. **TPU Path (`run_tpu_benchmark`)**:
   - Accept `architecture: Literal["quantum_born", "classical_modded"] = "quantum_born"` and `dynamic_povm_beta: bool = False`. If `architecture == "classical_modded"`, raise `ValueError("ClassicalModdedGPT is only supported on the CUDA benchmark path")`.

### 3.6 Controlled Verification Script: `examples/23_hypothesis_dynamic_sharpness_and_classical_baseline.py`
Create `/home/festivity/agy/QuantumDLBlocks/examples/23_hypothesis_dynamic_sharpness_and_classical_baseline.py` that executes:
1. **Experiment H1 (Constant-Logit-Norm Theorem & Heteroscedastic Sharpness Recovery)**:
   - Measures $\text{CV}(\|\ell\|_2)$ on static-$\beta$ spherical readout vs. `ContextualBlochPurityGate` (`use_dynamic_povm_beta=True`).
   - Trains static-$\beta$ `QuantumBornGPT` vs. dynamic-$\beta$ `QuantumBornGPT` (`d=128, D_seed=64, L=2, H=4, V=512`) for `150` steps on the synthetic heteroscedastic language dataset (50% deterministic contexts with $H^* = 0$, 50% 16-way branching contexts with $H^* = \ln 16$).
   - Verifies $\text{CV}_{\text{static}}(\|\ell\|_2) < 0.08$, $\text{CV}_{\text{dyn}}(\|\ell\|_2) \ge 0.15$, $\bar{\beta}_{\text{deterministic}} \ge 1.35 \cdot \bar{\beta}_{\text{branching}}$, and $\text{CE}_{\text{dyn}} \le \text{CE}_{\text{static}} - 0.15$.
2. **Experiment H2 (Karpathy Overfitting Capacity Scaling & Classical Modded-NanoGPT Verification)**:
   - Runs the 32-record memorization benchmark (`N=32, T=64, V=1024`, `120` steps) comparing `QuantumBornGPT(D_seed=16, static beta)`, `QuantumBornGPT(D_seed=64 + SeedAdapter + DynamicBeta)`, and `ClassicalModdedGPT(L=2, d=128, H=4)`.
   - Verifies $\text{CE}_{D=64,\text{dyn}} \le 0.50 \cdot \text{CE}_{D=16,\text{static}}$, $\text{Top1}_{D=64,\text{dyn}} \ge 98.0\%$, $\text{Top1}_{\text{classical}} \ge 98.0\%$, and $\text{CE}_{\text{classical}} \le 0.10$.
3. Writes all metrics and pass/fail flags to `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_013.json`.

---

## 4. Testing & Verification Plan (`tests/test_hypothesis_013_dynamic_sharpness_and_classical.py`)

Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_013_dynamic_sharpness_and_classical.py` with 5 unit tests:
1. `test_constant_logit_norm_vs_dynamic_purity_gate`: Verifies that `ContextualBlochPurityGate` initializes to exact identity scaling ($r(\psi) \equiv 1.0$, $\beta_{b,t} \equiv \beta_0$ at step 0) and after gradient updates produces $\text{CV}(\|\ell\|_2) > 0.10$ while clamping $\beta_{b,t} \in [\beta_0 r_{\min}, \min(\beta_0 r_{\max}, \beta_{\max})]$.
2. `test_quantum_born_gpt_dynamic_povm_beta_forward_backward`: Verifies `QuantumBornGPT(..., use_dynamic_povm_beta=True)` across `readout_mode="tied_stiefel"` and `readout_mode="asymmetric"`, confirming finite loss and non-zero gradients on `purity_gate.fc1` and `purity_gate.fc2`.
3. `test_classical_modded_gpt_forward_backward_and_softcap`: Verifies `ClassicalModdedGPT` forward/backward shapes, logit softcapping (`|logits| <= 30.0`), U-Net skip connections, and value residual mixing.
4. `test_jax_dynamic_purity_parity`: Verifies numerical parity (`max_abs_diff < 1e-5`) between PyTorch `ContextualBlochPurityGate` / `compute_logit_norm_variation_metrics` and JAX `contextual_bloch_purity_beta_jax` / `compute_logit_norm_variation_metrics_jax`.
5. `test_heteroscedastic_sharpness_separation`: Runs a fast 60-step micro-benchmark confirming that `ContextualBlochPurityGate` assigns higher $\beta_{b,t}$ to deterministic contexts than to high-entropy branching contexts ($\bar{\beta}_{\text{det}} > \bar{\beta}_{\text{branch}}$) and achieves lower validation CE than static $\beta$.
