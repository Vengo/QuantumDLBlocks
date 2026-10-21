# Experiment Plan 014: Resolving the Early-Epoch Quantum Validation Lag & Closing the Single-GPU Classical Modded-NanoGPT Speedrun Gap (Wave-7 900M-Token Fleet)

## 1. Executive Summary & Empirical Motivation

Our completed **Wave-6 900M-Token FineWeb Speedrun Fleet (`Experiment Plan 013`)** established two landmark empirical results across `899.48M` unique tokens (`9` FineWeb shards, `13,725` steps at `65,536` tokens/step):

1. **Dynamic Bloch-Purity POVM Sharpness (`ContextualBlochPurityGate`, `--dynamic-povm-beta`) Set a New All-Time `QuantumBornGPT` Record (`3.8514` Val CE)**:
   - `w6_900m_asym1024_frame3d_dynbeta` (`179.9M` params, $L=8, d=1024$, `3d` Overcomplete Two-Beam Frame + Asymmetric 1024D POVM + `ContextualBlochPurityGate`) achieved **`3.8514` Val CE (`47.1` Val PPL, `33.57%` Top-1, `54.75%` Top-5, `MedRank = 3`, mean effective $\beta = 88.10$)** — improving by **`-0.0445` nats** over its exact Wave-5 static-$\beta$ counterpart (`w5_900m_asym1024_frame3d`, `3.8959` Val CE) and beating our previous all-time quantum record (`3.8765`).
   - Similarly, `w6_900m_frame4d_synergy512_dynbeta` achieved **`3.8857` Val CE (`48.7` PPL, `33.40%` Top-1, $\beta = 78.68$)**, improving by **`-0.0369` nats** over its Wave-5 static-$\beta$ counterpart (`3.9226`).
2. **Two Critical Open Questions Exposed by Wave-6**:
   - **Question A — Why Does `QuantumBornGPT` Lag `ClassicalModdedGPT` in the Early Epochs (`0–350M` Tokens)?**
     - In Wave-6, both `QuantumBornGPT` and `ClassicalModdedGPT` learn unigram/short-range statistics at the exact same rate up to Step 150 (`~10M` tokens: Train CE `6.9174` Quantum vs. `6.9373` Classical).
     - Between `10M` and `100M` tokens, however, `ClassicalModdedGPT` (`w6_900m_classical_modded_162m`) immediately forms contextual features — its 5-vs-5 bipartite entanglement entropy $S(\rho_A)$ drops from the random Page-curve ceiling `2.9539 nats` down to **`2.2957 nats`** by `50M` tokens (`Val CE = 4.4798`) and **`2.2636 nats`** by `100M` tokens (`Val CE = 4.2158`).
     - In contrast, `QuantumBornGPT` (`w6_900m_frame4d_synergy512_dynbeta`) remains frozen at the random Page-curve ceiling (`S(rho_A) = 2.9542 nats` at `50M` tokens, `2.9315 nats` at `100M` tokens) while its isometry error `IsoErr` spikes from `0.1044` to **`1.3475`**, holding `Val CE` at `4.7826` (`50M` tok) and `4.6328` (`100M` tok) — a **`+0.4170` nat** early-epoch lag! Only in the second half of training (`400M–900M` tokens), when cosine LR decay allows `StiefelMuon`'s Stiefel penalty to pull `IsoErr` back down (`1.35 -> 0.15`), does `QuantumBornGPT` rapidly close the gap.
   - **Question B — Why Did `ClassicalModdedGPT` Reach `3.5444` (`162M`) / `3.6003` (`124M`) Val CE Instead of Keller Jordan's `modded-nanogpt` `~3.28` Target on 900M Tokens?**
     - In Wave-6, to isolate geometry alone, we forced `ClassicalModdedGPT` into `QuantumBornGPT`'s exact 6-layer, $T=512$, `65,536`-token/step, non-Nesterov `StiefelMuon` (`aspect_scale=False`), single-AdamW-LR (`0.004`) harness.
     - Comparing our Wave-6 `ClassicalModdedGPT` configuration against `KellerJordan/modded-nanogpt` reveals five concrete configuration differences that account for the `3.5444 -> 3.28` gap:
       1. **Depth vs. Width ($L=6, d=1024$ vs. GPT-2's $L=12, d=768$)**: Using 6 layers instead of 12 halves sequential multi-hop composition depth.
       2. **Context Length ($T=512$ vs. $T=1024$)**: Training and evaluating at sequence length $T=512$ instead of $T=1024$ loses `~0.08–0.10` nats of long-range in-context compression on FineWeb.
       3. **Global Batch Size (`65,536` tok/step vs. `262,144–524,288` tok/step) & Weight Norm Blow-Up (`IsoErr = 56.65`)**: Running `13,725` steps at `65,536` tok/step (`lr = 0.025`) instead of `~1,716–3,431` gradient-accumulated steps (`262,144–524,288` tok/step) caused `ClassicalModdedGPT`'s weight Frobenius norms to blow up (`IsoErr` rising from `0.23 -> 56.65`) and injected $4\text{–}8\times$ higher per-step gradient variance.
       4. **Missing Nesterov Lookahead & Aspect-Ratio Scaling in `StiefelMuon`**: `modded-nanogpt`'s `Muon` applies **Nesterov momentum** (`g_nesterov = g + momentum * buf`) before `zeropower_via_newtonschulz5` and scales rectangular updates (`fc1` of shape `(4d, d)`) by $\sqrt{\max(1, d_{\text{out}} / d_{\text{in}})} = 2.0$ (`aspect_scale=True`).
       5. **Zero-Initialized Readout/Residual Projections & Decoupled Embedding Learning Rate**: Because `ClassicalModdedGPT` applies parameterless `F.rms_norm` directly to `wte(idx)`, `wte` requires a decoupled high AdamW learning rate (`lr_wte = 0.20` vs. `0.004` in Wave-6), `lm_head` (`lr_head = 0.008`) and residual output projections (`out_proj`, `fc2`) should be zero-initialized (`nn.init.zeros_`) so Step 0 starts at exact uniform entropy $\ln(50304) = 10.8258$ (rather than `11.3410`), and the LR schedule should use **Warmup-Stable-Decay (`wsd`)** linear cooldown.

---

## 2. Mathematical Diagnosis & Observable Hypotheses (`H1`, `H2`, `H3`)

### 2.1 Three Structural Causes of the Early-Epoch Quantum Lag (and Their Exact Fixes)

1. **Cause 1 — The Spherical Identity-Loop / Self-Repetition Trap (`SelfRep0 = 25.38%` at Step 0)**:
   - In `QuantumBornGPT` (`readout_mode in {"tied_stiefel", "tied_vocab"}`), `hilbert_lift` lifts the unit-norm token seed $\hat{e}_{x_t} \in \mathcal{S}^{D_{\text{seed}}-1}$ into $\mathcal{S}^{d-1}$ via the column-orthonormal Stiefel frame $W_{\text{lift}} \in \mathbb{R}^{d \times D_{\text{seed}}}$ ($W_{\text{lift}}^\top W_{\text{lift}} = I_{D_{\text{seed}}}$).
   - Because `unet_gates` and `readout_rot` are initialized to the identity matrix $I_d$ and `collapse_to_seed` projects by $W_{\text{lift}}^\top$, the identity residual highway satisfies:
     $$W_{\text{lift}}^\top \, I_d \, W_{\text{lift}} \, \hat{e}_{x_t} = \hat{e}_{x_t}$$
   - Since the output POVM detectors $M_{\text{seed}}$ are initialized to $\mathcal{N}(\text{wte})$, the residual stream at Step 0 maps every input token $x_t$ directly onto its own output detector $\hat{e}_{x_t}$ with cosine similarity $\approx +1.0$ (`logit` $\approx +28.57$), assigning **`25.38%` probability mass to predicting $x_{t+1} = x_t$**!
   - Even in `readout_mode="asymmetric"` (where `povm_head` is independently initialized), unit-norm POVM rows $\|m_v\|_2 = 1$ have zero radial freedom to represent the 5-order-of-magnitude Zipf unigram frequency distribution without rotating all 50,304 unit vectors into an anisotropic frequency cone during the first 1,000 steps.
   - **Fix 1 (`OrthoDecouple` + `POVMLogPrior`)**:
     - Initialize `readout_rot` and `unet_gates` as random orthogonal Stiefel matrices (`nn.init.orthogonal_`) so $\mathbb{E}[\langle W_{\text{lift}}^\top R_{\text{readout}} W_{\text{lift}} \hat{e}_{x_t}, \hat{e}_{x_t}\rangle] = 0$, reducing Step-0 self-repetition probability from `25.38%` to `< 0.1%`.
     - Add a learnable rank-1 POVM log-prior bias $b \in \mathbb{R}^V$ (representing general rank-1 POVM measure weights $E_v = e^{b_v} |m_v\rangle\langle m_v|$ in log-probability space: $\text{logits}_v = \beta(x)\langle \phi, m_v\rangle + b_v$) so the unigram Zipf distribution is absorbed in < 50 steps by $b_v$, leaving the spherical state $|\phi\rangle \in \mathcal{S}^{D-1}$ 100% dedicated to contextual mutual information.

2. **Cause 2 — Post-Sublayer Spherical Overwriting (`psi <- l2_normalize(psi + a)`) at Peak LR**:
   - In `QuantumBornBlock`, each sublayer updates the state via $\psi_{\text{next}} = \mathcal{N}(\psi + a)$. When `StiefelMuon` at peak LR drives sublayer weight norms up (`IsoErr` reaching `1.35` at `50M` tokens, so $\|a\|_2 \approx 2.5$), $\mathcal{N}(\psi + a)$ **overwrites $\psi$** and attenuates the backward residual highway Jacobian $\frac{\partial \psi_{\text{next}}}{\partial \psi}$ by $\frac{1}{\|\psi + a\|_2} \approx 0.40$ at every sublayer ($0.40^{12} \approx 1.7 \times 10^{-5}$ across 6 blocks!).
   - **Fix 2 (`PreNormalizedGatedSphericalResidual`)**:
     - Normalize each sublayer branch output onto the unit sphere $\hat{a} = \mathcal{N}(a)$ before combining with $\psi$, and weight it by a learnable per-sublayer tangent gate $\alpha_l \in \mathbb{R}$ initialized to $\alpha_0 = 0.25$:
       $$\psi_{\text{next}} = \mathcal{N}\!\left(\psi + \alpha_l \, \mathcal{N}(\text{sublayer}(\psi))\right)$$
     - Because $\|\mathcal{N}(\text{sublayer}(\psi))\|_2 = 1$ regardless of the sublayer's internal singular values, `StiefelMuon` updates can never overwrite the residual highway at peak LR, keeping the per-sublayer residual cosine similarity $\langle \psi_{\text{next}}, \psi\rangle = \frac{1}{\sqrt{1 + \alpha_0^2}} \approx 0.970$ at initialization and preserving a clean backward gradient highway across all layers.

3. **Cause 3 — Homogeneous Born Attention Scale Cancellation ($\gamma_h^2 / \gamma_h^2 = 1$) & Quadratic SNR Ceiling at $T \ge 512$**:
   - In `SubspaceProjectorBornAttention` and `flash_born_attention`, the unnormalized attention weight is $\text{prob}_{ij} = \text{ReLU}(\gamma_h \langle q_i, k_j\rangle)^2 = \gamma_h^2 \text{ReLU}(\langle q_i, k_j\rangle)^2$. Upon row-normalization by $\sum_{m \le i} \text{prob}_{im}$, **$\gamma_h^2$ cancels out identically** ($\frac{\gamma_h^2 p_{ij}}{\sum_m \gamma_h^2 p_{im}} = \frac{p_{ij}}{\sum_m p_{im}}$)!
   - Worse, because degree-2 Born weights lack exponential suppression of small positive random overlaps $\langle q_i, k_m\rangle \sim \mathcal{N}(0, 1/d_h)$, at context length $T = 512$ with head dimension $d_h = 64$, the $\sim 256$ positive random keys create a background floor $\sum_{m \neq j^*} \text{ReLU}(\langle q_i, k_m\rangle)^2 \approx \frac{T}{2 d_h} = 4.0$, capping even a perfectly aligned induction-head key ($\langle q_i, k_{j^*}\rangle = 1.0$) at $\frac{1}{1+4} = 20\%$ attention weight!
   - **Fix 3 (`CoherenceThresholdedBornAttention` with Non-Canceling $\gamma_h$ Sharpening)**:
     - Let $s_{ij} = \langle \hat{q}_i, \hat{k}_j\rangle \in [-1, 1]$ denote the unit-sphere query-key overlap. Introduce a learnable per-head coherence threshold $\tau_h = \tau_{\max} \cdot \text{sigmoid}(\tilde{\tau}_h) \in (0, \tau_{\max})$ with $\tau_{\max} = 0.35$, initialized via the exact inverse-sigmoid $\tilde{\tau}_h(0) = \ln\!\left(\frac{\tau_0}{\tau_{\max} - \tau_0}\right)$ where $\tau_0 = \frac{0.5}{\sqrt{d_h}} = 0.0625$ (for $d_h = 64$).
     - Define the canonical unnormalized purified Born weight across all PyTorch and JAX modules as:
       $$\text{prob}_{ij} = \left(\text{ReLU}\!\left(\gamma_h (s_{ij} - \tau_h)\right)\right)^4 + \epsilon_{\text{base}} \, \text{ReLU}(s_{ij})^2$$
       where $\gamma_h = \text{clamp}(\text{qk\_gain\_param}_h, 0.5, 20.0)$ (initialized to $\gamma_0 = \sqrt{d_h} = 8.0$) and $\epsilon_{\text{base}} = 0.05$.
     - Notice two critical properties of this canonical formula:
       1. Because $\gamma_h$ multiplies the quartic coherent branch $\text{ReLU}(\gamma_h (s_{ij} - \tau_h))^4 = \gamma_h^4 \text{ReLU}(s_{ij} - \tau_h)^4$ while the quadratic background floor $\epsilon_{\text{base}} \text{ReLU}(s_{ij})^2$ is unscaled by $\gamma_h$, **$\gamma_h$ does not cancel out** in $\frac{\text{prob}_{ij}}{\sum_{m \le i} \text{prob}_{im}}$ and receives a strong, non-zero gradient!
       2. Any coherent key with $s_{ij^*} = 0.85 > \tau_h = 0.0625$ is amplified by $(\gamma_h (0.85 - 0.0625))^4 = (8.0 \times 0.7875)^4 \approx 1575.3$, whereas the $\sim 256$ random keys have $s_{im} \sim \mathcal{N}(0, 1/64)$ mostly below or near $\tau_h = 0.0625$, lifting the target key's attention weight from `< 18%` to **`> 85%`**!

---

### 2.2 Concrete Hypotheses & Quantitative Pass/Fail Criteria

- **Hypothesis H1 (Eliminating Step-0 Self-Repetition & Accelerating Early-Epoch Quantum Convergence)**:
  - **Prediction**: On genuine FineWeb validation tokens (`fineweb_val_000000.bin`), equipping `QuantumBornGPT` with `OrthoDecouple + POVMLogPrior + PreNormalizedGatedSphericalResidual + CoherenceThresholdedBornAttention`:
    1. Reduces Step-0 self-repetition probability $P(x_{t+1} = x_t)$ from **`> 20.0%`** (`25.38%` in baseline `QuantumBornGPT`) to **`< 0.5%`** (matching `ClassicalModdedGPT`),
    2. Increases single-target attention concentration (the attention weight assigned to a planted induction-head key $\langle q_i, k_{j^*}\rangle = 0.85$ amidst $T = 512$ random unit keys in $\mathbb{R}^{64}$ at $\gamma_h = 8.0, \tau_h = 0.0625$) from **`< 18.0%`** under homogeneous degree-2 Born attention to **`>= 65.0%`** (`>= 3.5x` sharper attention SNR), and
    3. Achieves a validation Cross-Entropy reduction of **`>= 0.20 nats`** (`Val CE <= 6.20` vs. `>= 6.40` baseline) within a controlled 200-step early-epoch FineWeb training run (`B=16, T=256`).
- **Hypothesis H2 (Non-Zero Gradient Flow & Monotonic Sharpness of `CoherenceThresholdedBornAttention`)**:
  - **Prediction**: Whereas baseline `SubspaceProjectorBornAttention(use_flash_born=False, use_purified_attn=False)` has **identically zero gradient** with respect to `qk_gain_param` ($\|\nabla_{\gamma_h} \mathcal{L}\|_2 < 10^{-7}$) because $\gamma_h^2$ cancels in $\frac{\gamma_h^2 c_{ij}^2}{\sum_m \gamma_h^2 c_{im}^2}$, `SubspaceProjectorBornAttention(use_purified_attn=True)` exhibits strictly non-zero gradients $\|\nabla_{\gamma_h} \mathcal{L}\|_2 > 10^{-4}$ and $\|\nabla_{\tilde{\tau}_h} \mathcal{L}\|_2 > 10^{-4}$, and increasing $\gamma_h$ from `2.0` to `12.0` (with fixed $\tau_h = 0.0625, \epsilon_{\text{base}} = 0.05$) monotonically decreases the row-wise attention Shannon entropy $H(A_i) = -\sum_j A_{ij} \ln A_{ij}$ by **`>= 0.80 nats`**.
- **Hypothesis H3 (Closing the Classical Modded-NanoGPT Single-GPU Optimization Gap)**:
  - **Prediction**: Upgrading `ClassicalModdedGPT` and `StiefelMuon` with the genuine `modded-nanogpt` single-GPU primitives — (a) **Nesterov polar lookahead (`nesterov=True`) + aspect-ratio scaling (`aspect_scale=True`)** in `StiefelMuon`, (b) **zero-initialized `lm_head`, `out_proj`, and `fc2`** (`Step 0 Val CE = ln(50304) = 10.8258`), and (c) **decoupled embedding learning rate (`lr_wte = 0.20`, `lr_head = 0.008`, `lr_scalar = 0.02`)** — reduces validation Cross-Entropy by **`>= 0.12 nats`** over the Wave-6 `ClassicalModdedGPT` baseline at matched 200 steps on FineWeb tokens while keeping weight Frobenius drift (`IsoErr`) at least **`40%` lower**.

---

## 3. Planned Code & Architectural Changes

### 3.1 Operational Script Classification (`Build-Before-Launch` vs. `Nice-to-Have`)
Following our mandatory operational scripting gate in `AGENTS.md`:
- **Existing Scripts Reused**:
  - `scripts/colab_fleet.py`: Used for `status`, `push-code`, `launch`, and `harvest` of `configs/wave7_fleet.json`.
  - `scripts/inspect_model.py`: Used to verify parameter counts and optimizer group splits (`muon`, `adamw`).
  - `scripts/diagnose_early_epoch_gap.py`: Already built and used to isolate Causes 1–3.
- **`Build-Before-Launch` Upgrade (Implemented)**:
  - Updated `scripts/colab_fleet.py` (`cmd_harvest`) so if a remote Colab session has already disconnected after finishing its run, `cmd_harvest` automatically falls back to parsing the locally synced `.cache/live_logs/<key>.log` and `<key>_results.json` files without reporting a false `ERR(rc=1)`.
- **`Nice-to-Have` Deferred to `scripts/TOOLING_BACKLOG.md`**:
  - Automated multi-wave Pareto curve plotter (`Val CE` vs. `Wall-Clock Minutes` and `Tokens Seen`).

### 3.2 New PyTorch Module: `src/qdlblocks/layers/early_epoch_quantum.py`
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/early_epoch_quantum.py` providing:
1. `compute_self_repetition_metrics(logits: torch.Tensor, input_ids: torch.Tensor) -> dict[str, float]`:
   - Given `logits` of shape `(B, T, V)` and `input_ids` of shape `(B, T)`, computes:
     - `mean_self_repeat_prob`: mean probability $\frac{1}{BT}\sum_{b,t} P(y_{b,t} = x_{b,t})$ assigned to repeating the immediate input token $x_{b,t}$,
     - `top1_self_repeat_rate`: fraction of positions where $\arg\max_v \text{logits}_{b,t,v} == x_{b,t}$,
     - `mean_entropy_nats`: mean predictive Shannon entropy across the vocabulary.
2. `purified_born_attention_weights(q: torch.Tensor, k: torch.Tensor, gamma: torch.Tensor | float = 8.0, tau: torch.Tensor | float = 0.0625, eps_base: float = 0.05, causal: bool = True, eps: float = 1e-8) -> torch.Tensor`:
   - Given unit-normalized query and key tensors `q, k` of shape `(B, H, T, d_h)`:
     1. Computes raw cosine overlap $s = q k^\top \in [-1, 1]^{B \times H \times T \times T}$.
     2. Computes unnormalized purified weights using the single canonical formula:
        $$\text{prob}_{ij} = \left(\text{ReLU}\!\left(\gamma (s_{ij} - \tau)\right)\right)^4 + \epsilon_{\text{base}} \, \text{ReLU}(s_{ij})^2$$
     3. Applies lower-triangular causal masking when `causal=True` and row-normalizes:
        $$A_{ij} = \frac{\text{prob}_{ij}}{\sum_{m \le i} \text{prob}_{im} + \epsilon}$$
3. `compute_born_attention_snr_metrics(q: torch.Tensor, k: torch.Tensor, gamma_h: float = 8.0, tau_h: float = 0.0625, eps_base: float = 0.05) -> dict[str, float]`:
   - Given normalized query/key tensors `q, k` of shape `(B, H, T, d_h)` where key index `0` is a planted target key, compares standard degree-2 Born attention (`prob_std = ReLU(gamma_h * <q, k>)^2`) against `purified_born_attention_weights(q, k, gamma=gamma_h, tau=tau_h, eps_base=eps_base, causal=True)`, returning across query positions $i \ge T // 2$:
     - `std_target_weight`: mean attention weight on `j=0` under standard Born attention,
     - `purified_target_weight`: mean attention weight on `j=0` under purified Born attention,
     - `snr_gain_ratio`: `purified_target_weight / max(std_target_weight, 1e-8)`,
     - `std_entropy_nats` and `purified_entropy_nats`.
4. `apply_orthogonal_highway_decoupling(model: nn.Module, seed: int = 1337) -> int`:
   - Re-initializes `model.readout_rot.weight` and each matrix in `model.unet_gates` using deterministic orthogonal matrices (`nn.init.orthogonal_` with a seeded CPU generator) so $W_{\text{lift}}^\top R_{\text{readout}} W_{\text{lift}}$ has zero diagonal bias toward self-repetition. Returns the number of decoupled Stiefel matrices.

### 3.3 Upgrading `SubspaceProjectorBornAttention` (`src/qdlblocks/layers/attention.py`)
1. In `src/qdlblocks/layers/attention.py` (`SubspaceProjectorBornAttention.__init__`):
   - Add `use_purified_attn: bool = False`, `coherence_tau_init: float = 0.0625`, `coherence_tau_max: float = 0.35`, and `purified_eps_base: float = 0.05`.
   - When `use_purified_attn=True`:
     - Compute the inverse-sigmoid initialization scalar $\tilde{\tau}_0 = \ln\!\left(\frac{\tau_0}{\tau_{\max} - \tau_0}\right)$ where $\tau_0 = \text{clamp}(\text{coherence\_tau\_init}, 10^{-4}, \tau_{\max} - 10^{-4})$ and $\tau_{\max} = \text{coherence\_tau\_max} = 0.35$, and register `self.coherence_tau_raw = nn.Parameter(torch.full((1, self.num_heads, 1, 1), raw_init))` so $\tau_h(0) = \tau_{\max} \cdot \text{sigmoid}(\tilde{\tau}_h(0)) = \text{coherence\_tau\_init} = 0.0625$ exactly at Step 0.
   - When `use_purified_attn=False`, register `self.register_parameter("coherence_tau_raw", None)`.
2. In `SubspaceProjectorBornAttention.forward`:
   - When `self.use_purified_attn=True`:
     - Do **not** scale `Q` by `gamma` before calling `l2_normalize` (since `l2_normalize` is scale-invariant), and do **not** dispatch to the degree-2 `flash_born_attention` Triton kernel (which only computes degree-2 `ReLU(qk)^2` and only returns `dq, dk, dv`).
     - Instead, compute unit-normalized `Q = l2_normalize(Q, dim=-1, eps=self.eps)` and `K = l2_normalize(K, dim=-1, eps=self.eps)`, evaluate $\gamma_h = \text{clamp}(\text{self.qk\_gain\_param}, 0.5, 20.0)$ and $\tau_h = \text{self.coherence\_tau\_max} \cdot \text{sigmoid}(\text{self.coherence\_tau\_raw})$, and call `purified_born_attention_weights(Q, K, gamma=gamma_h, tau=tau_h, eps_base=self.purified_eps_base, causal=self.causal, eps=self.eps)` followed by `torch.matmul(born_weights, V)`.
     - Under `torch.compile`, TorchInductor fuses the pointwise `ReLU(gamma * (s - tau))^4 + eps_base * ReLU(s)^2` and reduction directly into a single compiled CUDA kernel while preserving exact autograd gradients `self.qk_gain_param.grad` and `self.coherence_tau_raw.grad` under both eager and `torch.compile` execution!

### 3.4 Upgrading `QuantumBornBlock` & `QuantumBornGPT` (`src/qdlblocks/layers/born_block.py` & `src/qdlblocks/models/born_gpt.py`)
1. In `QuantumBornBlock` (`src/qdlblocks/layers/born_block.py`):
   - Add `use_gated_spherical_residual: bool = False`, `residual_gate_init: float = 0.25`, and `use_purified_attn: bool = False`.
   - Pass `use_purified_attn=use_purified_attn` to `SubspaceProjectorBornAttention`.
   - When `use_gated_spherical_residual=True`, register learnable scalar parameters `self.attn_res_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))` and `self.mlp_res_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))`, and apply pre-normalized spherical residual updates in `forward`:
     $$\psi \leftarrow \mathcal{N}\!\left(\psi + \alpha_{\text{attn}} \, \mathcal{N}(a)\right), \qquad \psi \leftarrow \mathcal{N}\!\left(\psi + \alpha_{\text{mlp}} \, \mathcal{N}(h)\right)$$
2. In `QuantumBornGPT` (`src/qdlblocks/models/born_gpt.py`):
   - Add `use_ortho_decouple: bool = False`, `use_povm_log_prior: bool = False`, `use_gated_spherical_residual: bool = False`, and `use_purified_attn: bool = False`.
   - When `use_ortho_decouple=True`, call `apply_orthogonal_highway_decoupling(self)` at the end of `__init__` to initialize `self.readout_rot.weight` and `self.unet_gates` with orthogonal Stiefel matrices instead of `torch.eye`.
   - When `use_povm_log_prior=True`, register `self.povm_log_prior = nn.Parameter(torch.zeros(self.vocab_size))` and add `self.povm_log_prior.to(logits.dtype)` to the output logits in `QuantumBornGPT.forward` across all readout modes (`tied_stiefel`, `tied_vocab`, `symmetric_dual`, `asymmetric`).
   - **Why `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` also needs `povm_log_prior` wired in**: In `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`, to avoid materializing the full `(B*T, 50304)` logit tensor in GPU HBM at `B=128, T=512`, `run_cuda_benchmark` executes `forward_backbone(m, idx)` to obtain `(phi_seed, M_seed, inv_t)` and then computes cross-entropy in `16,384`-token chunks via `_chunked_povm_ce` / `_chunk_ce_fn` rather than calling `QuantumBornGPT.forward` directly. Therefore, we will also add `m.povm_log_prior` inside `_chunk_ce_fn` and `evaluate_validation` in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`.

### 3.5 Upgrading `StiefelMuon`, `ClassicalModdedGPT`, and Optimizer Group Routing (`src/qdlblocks/optim/muon.py`, `src/qdlblocks/models/classical_modded_gpt.py`, & `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)
1. In `StiefelMuon` (`src/qdlblocks/optim/muon.py`):
   - Add `nesterov: bool = True` (default `True`) to `StiefelMuon.__init__`.
   - In `StiefelMuon.step`, compute `g_in = g.add(buf, alpha=momentum) if nesterov else buf` and pass `g_in` into `zeropower_via_newtonschulz5(g_in, steps=ns_steps)`.
2. In `ClassicalModdedBlock` & `ClassicalModdedGPT` (`src/qdlblocks/models/classical_modded_gpt.py`):
   - Add `zero_init_residual_and_head: bool = False` to `ClassicalModdedGPT.__init__` and `zero_init_residual: bool = False` to `ClassicalModdedBlock.__init__`.
   - When `zero_init_residual_and_head=True`:
     - If `tie_embeddings=True` was also passed, force `self.tie_embeddings = False` so `wte.weight` retains its $\mathcal{N}(0, 1/\sqrt{d})$ initialization while `self.lm_head.weight` is an independent parameter initialized via `nn.init.zeros_(self.lm_head.weight)`.
     - Initialize `block.out_proj.weight` and `block.fc2.weight` via `nn.init.zeros_`.
     - This guarantees that `ClassicalModdedGPT(..., zero_init_residual_and_head=True)` starts Step 0 at exact uniform cross-entropy $\ln(V) = \ln(50304) \approx 10.825839$ and an exact identity residual highway.
3. **Explicit Optimizer Group Routing & Weight Decay Rules**:
   - In `split_muon_adamw_params` (`src/qdlblocks/optim/muon.py`), `scripts/inspect_model.py`, and `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`:
     - Only 2D weight matrices (`p.ndim == 2`) whose names do not match `adamw_name_patterns` are routed to `StiefelMuon`.
     - All four new quantum parameters have `p.ndim != 2` (`povm_log_prior` is 1D `(V,)`, `coherence_tau_raw` is 4D `(1, H, 1, 1)`, `attn_res_gate` and `mlp_res_gate` are 0D `()`), so they are automatically routed to `AdamW` without needing any changes to `adamw_name_patterns` (though we also include `"povm_log_prior"`, `"coherence_tau"`, `"res_gate"` in `adamw_name_patterns` for explicit clarity).
     - For `QuantumBornGPT`, all `AdamW` parameters use `base_adamw_lr = 0.004`, with `weight_decay = 0.0` on 0D/1D/4D bias/gate/prior/temperature parameters (`p.ndim != 2`) and `weight_decay = 0.01` on 2D embedding/POVM tables (`wte`, `povm_head`).
     - For `ClassicalModdedGPT` when `--zero-init-classical` is enabled:
       - `StiefelMuon` uses `lr = 0.025, momentum = 0.95, nesterov = True, aspect_scale = True, weight_decay = 0.01, decay_mode = "cautious", stiefel_penalty = 0.0`.
       - `AdamW` uses three explicit parameter groups (consistent across Section 1, Hypothesis H3, Section 3.5, and `examples/24`):
         1. `wte` group (`"wte" in name`): `lr = 0.20`, `weight_decay = 0.0` (since `F.rms_norm(wte(idx))` makes `wte` scale-invariant).
         2. `lm_head` group (`"lm_head" in name`): `lr = 0.008`, `weight_decay = 0.0`.
         3. Scalar/gate group (all remaining non-Muon parameters: `smear_embed_gate`, `highway_gate`, `v_mix_gate`, `unet_scales`): `lr = 0.02`, `weight_decay = 0.0`.

### 3.6 New JAX Counterpart: `src/qdlblocks_jax/early_epoch_quantum.py`
Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/early_epoch_quantum.py` implementing:
- `compute_self_repetition_metrics_jax(logits: jax.Array, input_ids: jax.Array) -> dict[str, jax.Array]`
- `purified_born_attention_weights_jax(q: jax.Array, k: jax.Array, gamma_h: float = 8.0, tau_h: float = 0.0625, eps_base: float = 0.05, causal: bool = True, eps: float = 1e-8) -> jax.Array` (using the exact same canonical formula $\text{ReLU}(\gamma_h (s_{ij} - \tau_h))^4 + \epsilon_{\text{base}} \text{ReLU}(s_{ij})^2$)
And export them in `src/qdlblocks_jax/__init__.py`.

---

## 4. Verification Plan (Unit Tests & Controlled Experiments)

### 4.1 Unit Tests (`tests/test_hypothesis_014_early_epoch_and_speedrun.py`)
Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_014_early_epoch_and_speedrun.py` with 5 unit tests:
1. `test_ortho_decouple_eliminates_step0_self_repetition`: Verifies on a `(B=4, T=64)` token batch (`vocab_size=512, embed_dim=128, seed_dim=64, num_heads=4, num_layers=4`) that baseline `QuantumBornGPT(use_ortho_decouple=False)` has Step-0 `mean_self_repeat_prob > 0.15`, whereas `QuantumBornGPT(use_ortho_decouple=True)` has `mean_self_repeat_prob < 0.01`.
2. `test_purified_born_attention_non_zero_gamma_and_tau_gradients`: Verifies that in standard `SubspaceProjectorBornAttention(embed_dim=128, num_heads=4, use_flash_born=False, use_purified_attn=False)`, `qk_gain_param.grad` is identically `0.0` (`< 1e-7`), whereas with `use_purified_attn=True`, both `qk_gain_param.grad` and `coherence_tau_raw.grad` are strictly non-zero (`> 1e-4`), and increasing `qk_gain_param` from `2.0` to `12.0` monotonically reduces row-wise attention Shannon entropy by `>= 0.80 nats`.
3. `test_purified_born_attention_snr_gain_at_t512`: Verifies `compute_born_attention_snr_metrics` on `(B=2, H=4, T=512, d_h=64)` unit vectors where key `j=0` has overlap `0.85` with queries `i >= 256` and keys `j >= 1` are random unit vectors in $\mathbb{R}^{64}$, confirming `std_target_weight < 0.18`, `purified_target_weight >= 0.65`, and `snr_gain_ratio >= 3.5`.
4. `test_gated_spherical_residual_and_povm_log_prior_forward_backward`: Verifies `QuantumBornGPT` with `use_ortho_decouple=True, use_povm_log_prior=True, use_gated_spherical_residual=True, use_purified_attn=True` across both `readout_mode="tied_stiefel"` and `readout_mode="asymmetric"`, checking that gradients populate on `povm_log_prior`, `coherence_tau_raw`, `attn_res_gate`, and `mlp_res_gate`.
5. `test_classical_zero_init_and_jax_parity`: Verifies that `ClassicalModdedGPT(vocab_size=512, embed_dim=128, num_heads=4, num_layers=4, zero_init_residual_and_head=True)` starts at exact uniform cross-entropy $\ln(512) \pm 10^{-5}$ at Step 0 even if `tie_embeddings=True` was passed, and verifies PyTorch-vs-JAX parity (`< 1e-5` max abs diff) for `purified_born_attention_weights_jax` and `compute_self_repetition_metrics_jax`.

### 4.2 Deterministic Specification of Controlled Experiment (`examples/24_hypothesis_early_epoch_and_classical_speedrun.py`)
Create `/home/festivity/agy/QuantumDLBlocks/examples/24_hypothesis_early_epoch_and_classical_speedrun.py` with fixed random seed `42` and write results to `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_014.json`:
- **Dataset**: Genuine FineWeb tokens loaded via `FineWebMultiShardStream` from `.cache/fineweb10B/fineweb_val_000000.bin` (first `1,000,000` tokens for training, next `131,072` tokens for validation; if offline, falls back to `generate_contextual_zipf_shard`).
- **Experiment H1 (Early-Epoch Quantum Convergence & Self-Repetition Elimination)**:
  - **Model Dimensions**: `QuantumBornGPT` with `vocab_size=50304, embed_dim=384, seed_dim=192, num_heads=6, num_layers=6, hidden_dim=768, block_size=256, readout_mode="tied_stiefel", use_flash_born=False`.
  - **Arms Compared**:
    - `baseline_quantum`: `use_ortho_decouple=False, use_povm_log_prior=False, use_gated_spherical_residual=False, use_purified_attn=False`.
    - `earlyfix_quantum`: `use_ortho_decouple=True, use_povm_log_prior=True, use_gated_spherical_residual=True, use_purified_attn=True, use_dynamic_povm_beta=True`.
  - **Training Protocol**: `200` steps, `batch_size=16, seq_len=256` (`4,096` tok/step = `819,200` tokens), `StiefelMuon(lr=0.025, momentum=0.95, nesterov=True, weight_decay=0.01, decay_mode="cautious", stiefel_penalty=0.005)` + `AdamW(lr=0.004)` with cosine LR decay and `20`-step warmup.
  - **Pass Criteria**:
    1. `earlyfix_quantum` Step-0 `mean_self_repeat_prob < 0.005` vs. `baseline_quantum > 0.20`,
    2. `compute_born_attention_snr_metrics` at `(B=2, H=4, T=512, d_h=64)` achieves `purified_target_weight >= 0.65` and `snr_gain_ratio >= 3.5` vs. `std_target_weight < 0.18`,
    3. `earlyfix_quantum` Step-200 `val_ce` improves over `baseline_quantum` Step-200 `val_ce` by **`>= 0.20 nats`**.
- **Experiment H2 (Gradient Flow & Monotonic Attention Entropy Reduction)**:
  - Evaluates `SubspaceProjectorBornAttention(embed_dim=256, num_heads=4, use_flash_born=False, use_purified_attn=True)` vs. `use_purified_attn=False` on `(B=4, T=128, D=256)`, verifying $\|\nabla_{\gamma_h} \mathcal{L}\|_2 > 10^{-4}$, $\|\nabla_{\tilde{\tau}_h} \mathcal{L}\|_2 > 10^{-4}$, and `entropy(gamma=2.0) - entropy(gamma=12.0) >= 0.80 nats`.
- **Experiment H3 (Classical Modded-NanoGPT Zero-Init + Decoupled `lr_wte` + Nesterov/Aspect-Scale Muon)**:
  - **Model Dimensions**: `ClassicalModdedGPT` with `vocab_size=50304, embed_dim=384, num_heads=6, num_layers=6, hidden_dim=1536, block_size=256`.
  - **Arms Compared**:
    - `wave6_classical_baseline`: `zero_init_residual_and_head=False`, `StiefelMuon(lr=0.025, nesterov=False, aspect_scale=False, weight_decay=0.01)` + single `AdamW(lr=0.004, weight_decay=0.01)`.
    - `speedrun_classical_upgraded`: `zero_init_residual_and_head=True`, `StiefelMuon(lr=0.025, nesterov=True, aspect_scale=True, weight_decay=0.01)` + 3-group `AdamW` (`lr_wte=0.20, lr_head=0.008, lr_scalar=0.02, weight_decay=0.0`).
  - **Training Protocol**: `200` steps, `batch_size=16, seq_len=256` on the exact same FineWeb token stream as H1.
  - **Pass Criteria**: `speedrun_classical_upgraded` achieves **`>= 0.12 nats`** lower Step-200 `val_ce` than `wave6_classical_baseline` and **`>= 40%`** lower Step-200 `iso_err`.

---

## 5. Wave-7 900M-Token FineWeb Speedrun Fleet (`configs/wave7_fleet.json`)

Once all unit tests and controlled experiments pass, we will launch the **Wave-7 900M-Token Fleet** across our 5 Colab GPUs via `scripts/colab_fleet.py`:

| Slot | Session | GPU | Run Key | Architecture & Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `gpu_g4_w6_1` | RTX PRO 6000 (96GB) | `w7_900m_qborn_earlyfix_frame4d_synergy512` | **`QuantumBornGPT` (`162M`, $L=6, d=1024$) + Full Early-Epoch Fix (`--ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn --dynamic-povm-beta`)** on `4d` Two-Beam + `Seed512` |
| **2** | `gpu_g4_w6_2` | RTX PRO 6000 (96GB) | `w7_900m_classical_speedrun_12L_t1024` | **`ClassicalModdedGPT` Speedrun Config ($L=12, d=768, H=12, M=3072, T=1024, \text{grad\_accum}=2, \text{WSD}$ + `--zero-init-classical`)** to test closing the `3.54 -> ~3.28` gap on a single GPU |
| **3** | `gpu_g4_w6_3` | RTX PRO 6000 (96GB) | `w7_900m_qborn_earlyfix_asym1024_L8` | **`QuantumBornGPT` (`180M`, $L=8, d=1024$, `asymmetric` + `3d` Frame + `dynbeta`) + Full Early-Epoch Fix (`--ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn`)** aiming to break our `3.8514` quantum record |
| **4** | `gpu_a100_w6_1` | A100-SXM4 (40GB) | `w7_900m_qborn_earlyfix_L12_d768` | **12-Layer Deep `QuantumBornGPT` ($L=12, d=768, H=12$, `3d` Frame + `asymmetric` + `dynbeta`) + Full Early-Epoch Fix** (apples-to-apples $L=12, d=768$ quantum counterpart to GPT-2 `124M`) |
| **5** | `gpu_a100_w6_2` | A100-SXM4 (40GB) | `w7_900m_classical_speedrun_6L_zeroinit` | **6-Layer `ClassicalModdedGPT` (`162M`, $L=6, d=1024, M=4096$) + `--zero-init-classical` + WSD schedule** (isolating the gain of zero-init + decoupled `wte` LR + Nesterov/aspect-scale `Muon` at $L=6, T=512$ vs. Wave-6's `3.5444`) |
