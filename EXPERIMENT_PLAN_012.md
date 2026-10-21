# EXPERIMENT PLAN 012: Multi-Axis Parameter Scaling from 100M to 210M+ Parameters — Circuit Depth $L$, Overcomplete $3d/4d$ Two-Beam Frame Width, Hilbert Space Width $d_{\text{model}}$, and Readout Reinvestment (`tied_vocab` & `asymmetric`)

**Status:** APPROVED (First-Pass Blind Review Approved by `194d5a7c-64ba-47df-ad03-06432bcd4e0d`)  
**Author:** Principal Research Scientist & Architect  
**Date:** 2026-10-17  

---

## 1. Executive Summary & Architectural Motivation

With Wave-4 now streaming 900M unique FineWeb tokens (`9 x 100M` shards in a single pass via `FineWebMultiShardStream`), the data-repetition memorization ceiling has been removed. This unlocks the next fundamental scaling question for the NanoGPT Speedrun: **where should we add parameters next to scale `QuantumBornGPT` from ~100M parameters to 124M–210M+ parameters for maximum validation cross-entropy reduction per parameter and per FLOP?**

### 1.1 Anatomy of Current Parameter Allocation vs. NanoGPT-124M

A direct parameter decomposition of our flagship Wave-3/Wave-4 models (`L=6, d=1024, H=16`) versus the standard NanoGPT-124M reference (`L=12, d=768, H=12, d_ff=3072, V=50304`) exposes a striking structural asymmetry:

| Model Configuration | Total Params $P_{\text{total}}$ | **Dense Backbone $P_{\text{backbone}}$** (Attn + MLP + U-Net) | Backbone Share (%) | Circuit Depth $L$ | Sparse Vocab / Lift ($wte + W_{\text{lift}}$) | Readout + Adapter ($M_{\text{out}} + \text{Adapter}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NanoGPT-124M Reference** (`L=12, d=768, d_ff=3072, tied wte`) | `123.59M` | **`84.93M`** (`28.31M` Attn + `56.62M` MLP) | **`68.7%`** | **`12`** | `38.63M` (tied) | `0.00M` (tied to `wte`) |
| **`w3_gpu_adapter384_2beam`** (`L=6, d=1024, D_seed=384, 1536x2`) | `99.58M` | **`56.62M`** (`25.17M` Attn + `28.31M` MLP + `3.15M` U-Net) | **`56.9%`** | **`6`** | `19.71M` | `23.25M` (`19.32M` $\Delta_{\text{povm}}$ + `3.93M` Rot/Ada) |
| **`w3_gpu_full_synergy512`** (`L=6, d=1024, D_seed=512, 2048x2`) | `135.72M` | **`66.06M`** (`25.17M` Attn + `37.75M` MLP + `3.15M` U-Net) | **`48.7%`** | **`6`** | `26.28M` | `32.05M` (`25.76M` $\Delta_{\text{povm}}$ + `6.29M` Rot/Ada) |

Two critical insights emerge from this table:
1. **Backbone Deficit (`56.6M` vs. `84.9M` and `L=6` vs. `L=12`)**: Our `L=6` models have **33% fewer non-embedding backbone parameters** (`56.62M` vs. `84.93M`) and **half the sequential compositional depth (`L=6` vs. `L=12`)** of NanoGPT-124M. Every backbone parameter receives a dense gradient across all $B \times T = 65,536$ token positions on every optimization step, whereas each row of the vocabulary tables ($wte, \Delta_{\text{povm}} \in \mathbb{R}^{V \times D_{\text{seed}}}$) is updated only when its specific token ID appears in the batch.
2. **Duplicate Untied Vocabulary Table in `tied_stiefel` (`povm_head_seed_delta`)**: In `readout_mode="tied_stiefel"`, the model allocates both `wte` ($V \times D_{\text{seed}}$) and an untied `povm_head_seed_delta` ($V \times D_{\text{seed}}$) initialized to zero, consuming $2 V D_{\text{seed}}$ parameters (`38.63M` at $D_{\text{seed}}=384$ and `51.51M` at $D_{\text{seed}}=512$). Now that `OvercompleteSeedReadoutAdapter` and `TwoBeamSeedReadoutAdapter` provide a high-rank nonlinear pullback from $\mathbb{S}^{d-1}$ to $\mathbb{S}^{D_{\text{seed}}-1}$ (plus orthogonal `readout_rot` $\in \mathbb{R}^{d \times d}$), tying the seed measurement frame directly to $M_{\text{seed}} = \mathcal{N}(W_{\text{wte}})$ (`readout_mode="tied_vocab"`) frees $19.32\text{M}$–$25.76\text{M}$ sparse vocabulary parameters that can be reinvested directly into additional dense backbone layers!

### 1.2 The Four High-Impact Parameter Scaling Axes for Wave-5

To systematically determine where added parameters yield the greatest validation cross-entropy reduction on the 900M-token FineWeb Speedrun, we formalize and test four orthogonal scaling axes spanning **124M to 210M+ parameters**:

1. **Axis 1 — Circuit Depth $L$ (`L=6` $\to$ `L=9`, `L=10`, `L=12` with Depth-Scaled Spherical Residuals $\alpha_{\text{res}}(L) = \sqrt{6/L}$)**:
   - Scaling `L=6 -> L=9` (`127.9M` total / `84.9M` backbone, matching NanoGPT-124M's exact `84.9M` backbone!) and `L=6 -> L=12` (`156.2M` total with `1536x2` or `177.2M` total with `2048x2`, `113.2M`–`132.1M` backbone, and `6` Mach-Zehnder U-Net skip gates).
   - To prevent deep hyperspherical state drift when stacking $2L$ sublayers on $\mathbb{S}^{d-1}$, we introduce **Depth-Scaled Spherical Residuals**:
     $$x_{\text{next}} = \mathcal{N}\!\left(x + \alpha_{\text{res}}(L) \, h(x)\right), \qquad \alpha_{\text{res}}(L) = \sqrt{\frac{6}{L}}$$
     Notice that at $L=6$, $\alpha_{\text{res}}(6) = 1.0$ (preserving 100% exact backward compatibility with Waves 1–4), while at $L=12$, $\alpha_{\text{res}}(12) = 1/\sqrt{2} \approx 0.7071$, keeping the cumulative angular variance across $2L$ spherical updates invariant to depth $L$.
2. **Axis 2 — Higher-Redundancy Overcomplete Two-Beam Frame MLP (`2d` [`2048x2`] $\to$ `3d` [`3072x2`] and `4d` [`4096x2`])**:
   - In `OvercompleteTwoBeamInterference`, the Hadamard interference vector $z = (F_1 x) \odot (F_2 x) \in \mathbb{R}^m$ samples rank-1 directions in $\text{Sym}^2(\mathbb{R}^d)$ ($\dim = d(d+1)/2 \approx 524,800$).
   - Scaling `frame_expansion` from $\rho = 2.0$ (`m=2048`, `6.29M`/layer) to **$\rho = 3.0$ (`m=3072`, `9.44M`/layer, matching NanoGPT's `d_ff=3072` $\to$ `137.3M` total at `L=6`)** and **$\rho = 4.0$ (`m=4096`, `12.58M`/layer $\to$ `156.2M` total at `L=6`)** doubles MLP capacity inside a single fused `[F_1; F_2]` GEMM at peak Tensor Core MFU.
   - We generalize the target isometry scale for frame redundancy $\rho \ge 1$ to:
     $$\sigma_{\text{target}}(\rho) = \sqrt{\frac{\rho + 3}{4}}$$
     which equals $\sqrt{1.25} \approx 1.1180$ at $\rho=2.0$ (matching `target_iso_scale=1.25` in squared Gram Frobenius convention or $\sqrt{1.25}$ in column/row scale), $\sqrt{1.50} \approx 1.2247$ at $\rho=3.0$, and $\sqrt{1.75} \approx 1.3229$ at $\rho=4.0$.
3. **Axis 3 — Hilbert Space Width $d_{\text{model}}$ (`d=1024, H=16` $\to$ `d=1280, H=20` and `d=1536, H=24`)**:
   - Expanding the ambient Hilbert sphere $\mathbb{S}^{d-1}$ from $d=1024$ (`16` heads $\times 64$) to $d=1280$ (`20` heads $\times 64$, `133.5M`–`151.2M` params at `L=6`) or $d=1536$ (`24` heads $\times 64$, `173.8M`–`198.9M` params at `L=6`) scales both attention subspace projector rank and MLP quadratic rank quadratically ($O(d^2)$) without adding sequential layer hops.
4. **Axis 4 — Readout Reinvestment (`tied_vocab` + `L=10` at 124M Params vs. Full-Rank `asymmetric` $d_{\text{model}}$ POVM Readout)**:
   - **`readout_mode="tied_vocab"`**: Keeps `readout_rot` ($\mathbb{R}^{d \times d}$), `seed_adapter`, and `tied_inv_temp`, while tying the seed POVM frame directly to $M_{\text{seed}} = \mathcal{N}(W_{\text{wte}})$ (no `povm_head_seed_delta`). At $D_{\text{seed}}=384$, this saves `19.32M` sparse vocab parameters, allowing an **`L=10` deep model (`118.0M` total with `1536x2` or `135.8M` total with `2048x2`)** to fit in the exact **~124M NanoGPT parameter budget**!
   - **`readout_mode="asymmetric"`**: Replaces the seed-projected readout with a full-rank $V \times d_{\text{model}}$ POVM measurement frame directly on $\mathbb{S}^{d-1}$, completely eliminating any readout subspace rank ceiling ($R_{\text{eff}}$ up to $d_{\text{model}} = 1024$).

---

## 2. Observable Hypotheses & Quantitative Pass/Fail Predictions

### Hypothesis H1 (Depth-Scaled Spherical Residuals $\alpha_{\text{res}}(L) = \sqrt{6/L}$ & Marginal Return of Backbone Depth vs. Sparse Vocab Tables)

1. **Mathematical Mechanism**:
   On $\mathbb{S}^{d-1}$, a sequence of $2L$ sublayers with unit residual scale ($\alpha_{\text{res}} = 1.0$) applies updates $x^{(k)} = \mathcal{N}(x^{(k-1)} + h^{(k)}(x^{(k-1)}))$. When each normalized sublayer output $h^{(k)}$ is approximately orthogonal to $x^{(k-1)}$ at initialization ($\|h^{(k)}\|_2 \approx 1$), the squared overlap with the initial lifted state $|\psi^{(0)}\rangle$ decays as $(1 + \alpha_{\text{res}}^2)^{-2L}$. Setting $\alpha_{\text{res}}(L) = \sqrt{6/L}$ stabilizes the expected total geodesic excursion $\sum_{l=1}^L \arccos|\langle \psi^{(l-1)} | \psi^{(l)} \rangle|$ across arbitrary depth $L \ge 6$.
2. **Quantitative Predictions (H1)**:
   - **H1a (Initial State Fidelity & Gradient Conditioning at $L=12$)**: In a randomly initialized 12-layer `QuantumBornGPT` (`L=12, d=128, D_seed=64`), enabling `depth_scaled_residual=True` ($\alpha_{\text{res}}(12) = \sqrt{6/12} \approx 0.7071$) increases initial end-to-end state fidelity $\mathbb{E}[|\langle \psi^{(0)} | \psi^{(L)} \rangle|^2]$ from $< 0.12$ (unscaled $\alpha_{\text{res}} = 1.0$) to **$\ge 0.18$**, keeps mean per-layer geodesic step size $\bar{\theta}_{\text{step}} \le 0.65\text{ rad}$, and reduces the across-layer gradient norm coefficient of variation ($\text{CV}(\|\nabla_{W^{(l)}} \mathcal{L}\|_F) = \text{std}/\text{mean}$) by **$\ge 15\%$**.
   - **H1b (Controlled Training & Marginal Return per Million Parameters)**: Over 250 steps on the contextual multi-shard benchmark (`examples/22_hypothesis_parameter_scaling_axes.py`), scaling circuit depth from $L=6$ to $L=12$ with `depth_scaled_residual=True` reduces final validation CE by **$\ge 0.08$ nats** over $L=6$ (and beats unscaled $L=12$ by $\ge 0.03$ nats), achieving a validation CE reduction per million added parameters ($\Delta \text{CE} / \Delta P_{\text{M}}$) that is **$\ge 2.0\times$ larger** than adding an untied vocabulary table of matching parameter size.

### Hypothesis H2 (Higher-Redundancy Overcomplete Two-Beam Frames $\rho \in \{2.0, 3.0, 4.0\}$ & Hilbert Space Width $d_{\text{model}}$ Scaling)

1. **Mathematical Mechanism**:
   In `OvercompleteTwoBeamInterference`, expanding the paired frames $F_1, F_2 \in \mathbb{R}^{m \times d}$ from $m = 2d$ ($\rho=2.0$) to $m = 3d$ ($\rho=3.0$) and $m = 4d$ ($\rho=4.0$) with target Gram scale $\tau(\rho) = 1.0 + 0.25(\rho - 1.0)$ (`1.25` at $\rho=2.0$, `1.50` at $\rho=3.0$, `1.75` at $\rho=4.0$) increases the number of distinct bilinear wave-interference channels $z_i = (F_1 x)_i (F_2 x)_i$ before projecting back to $\mathbb{S}^{d-1}$ via $W_{\text{out}} \in \mathbb{R}^{d \times m}$.
2. **Quantitative Predictions (H2)**:
   - **H2a (Monotonic Interference Feature Rank Expansion)**: Across a batch of $N = 1024$ isotropic states on $\mathbb{S}^{d-1}$ ($d=64$), the effective quantum rank $R_{\text{eff}}(Z_{\text{2beam}})$ of the post-Hadamard feature matrix $Z_{\text{2beam}} = (X F_1^\top) \odot (X F_2^\top) \in \mathbb{R}^{N \times m}$ increases strictly monotonically with frame redundancy $\rho$:
     $$R_{\text{eff}}(\rho=4.0) > R_{\text{eff}}(\rho=3.0) > R_{\text{eff}}(\rho=2.0)$$
     with **$R_{\text{eff}}(\rho=4.0) \ge 1.15 \times R_{\text{eff}}(\rho=2.0)$**.
   - **H2b (Controlled Validation CE Improvement for $\rho=3.0, 4.0$ and Wider $d_{\text{model}}$)**: In the controlled 250-step multi-shard benchmark (`examples/22_hypothesis_parameter_scaling_axes.py`), both (i) expanding `frame_expansion` from $\rho=2.0$ to $\rho=3.0$ / $\rho=4.0$ at fixed $(L=6, d=64)$ and (ii) expanding Hilbert width from $d=64$ to $d=96$ ($1.5\times$ width) at fixed $L=6$ improve final validation CE over the $(L=6, d=64, \rho=2.0)$ baseline by **$\ge 0.05$ nats**.

### Hypothesis H3 (Readout Parameter Reinvestment: `tied_vocab` + Deep Backbone vs. `tied_stiefel` and `asymmetric` Full-Rank Readout)

1. **Mathematical Mechanism**:
   In `readout_mode="tied_vocab"`, removing the duplicate $V \times D_{\text{seed}}$ `povm_head_seed_delta` matrix while retaining `readout_rot` ($\mathbb{R}^{d \times d}$), `seed_adapter`, and `tied_inv_temp` eliminates $V \times D_{\text{seed}}$ sparse parameters and allows reinvesting that exact parameter budget into $+4$ additional `QuantumBornBlock` layers ($L=6 \to L=10$). Meanwhile, `readout_mode="asymmetric"` measures $|\psi^{(L)}\rangle \in \mathbb{S}^{d-1}$ directly against a full-rank $V \times d$ POVM frame without collapsing through $\mathbb{R}^{D_{\text{seed}}}$.
2. **Quantitative Predictions (H3)**:
   - **H3a (Iso-Parameter `tied_vocab` Depth Reinvestment Wins)**: At matched total parameter count (within $\pm 5\%$), `QuantumBornGPT(num_layers=10, readout_mode="tied_vocab", depth_scaled_residual=True)` achieves a final validation CE at least **$0.06$ nats lower** than `QuantumBornGPT(num_layers=6, readout_mode="tied_stiefel")` in the controlled 250-step benchmark.
   - **H3b (Un-Bottlenecked Full-Rank `asymmetric` Readout Rank)**: `QuantumBornGPT(..., readout_mode="asymmetric")` produces pre-POVM output states on $\mathbb{S}^{d-1}$ whose effective quantum rank $R_{\text{eff}}(\Psi^{(L)})$ exceeds $D_{\text{seed}}$ ($R_{\text{eff}}(\Psi^{(L)}) > D_{\text{seed}}$ when $D_{\text{seed}} < d/2$) and reduces validation CE by **$\ge 0.05$ nats** over the `L=6` `tied_stiefel` baseline.

---

## 3. Planned Code & Architectural Changes

### 3.1 New Module: `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/parameter_scaling.py`

Create `src/qdlblocks/layers/parameter_scaling.py` (and export in `src/qdlblocks/layers/__init__.py` and `src/qdlblocks/__init__.py`) containing:

1. `default_overcomplete_target_iso_scale(frame_expansion: float) -> float`:
   - For `frame_expansion <= 1.0`, returns `1.0`.
   - For `frame_expansion > 1.0`, returns `float(1.0 + 0.25 * (frame_expansion - 1.0))` (so `frame_expansion=2.0 -> 1.25`, `3.0 -> 1.50`, `4.0 -> 1.75`).
2. `compute_depth_residual_scale(num_layers: int, base_layers: int = 6, enabled: bool = True) -> float`:
   - If `not enabled` or `num_layers <= base_layers`: returns `1.0`.
   - Otherwise: returns `float(math.sqrt(float(base_layers) / float(num_layers)))`.
3. `compute_parameter_scaling_breakdown(model: nn.Module) -> dict[str, int | float]`:
   - Inspects `model.named_parameters()` and returns exact counts and percentages:
     - `total_params`: int
     - `backbone_params`: int (all parameters in `blocks` and `unet_gates` / `unet_scales`)
     - `attention_params`: int (parameters in `.attn.`)
     - `mlp_params`: int (parameters in `.two_beam_mlp.`, `.moe.`, `.ff1.`, `.ff2.`)
     - `unet_params`: int (parameters in `unet_gates`, `unet_scales`)
     - `vocab_lift_params`: int (parameters in `hilbert_lift`)
     - `readout_params`: int (parameters in `povm_readout`, `readout_rot`, `povm_head_seed_delta`, `seed_adapter`, `tied_inv_temp`, `mtp_rotations`, `mtp_inv_temps`)
     - `backbone_fraction`: float (`backbone_params / max(1, total_params)`)
4. `compute_depth_fidelity_and_geodesic_profile(model: QuantumBornGPT, idx: torch.Tensor, targets: torch.Tensor | None = None) -> dict[str, float | list[float]]`:
   - Runs a forward pass with `return_layer_states=True` (and optional backward pass if `targets is not None`) to compute:
     - `end_to_end_fidelity`: $\mathbb{E}_{b,t}[|\langle \psi^{(0)}_{b,t} | \psi^{(L)}_{b,t} \rangle|^2]$
     - `layer_geodesic_steps_rad`: list of $L$ floats $\theta^{(l)} = \mathbb{E}_{b,t}[\arccos(\text{clamp}(|\langle \psi^{(l-1)}_{b,t} | \psi^{(l)}_{b,t} \rangle|, 0, 1 - 10^{-6}))]$
     - `cumulative_geodesic_length_rad`: float $\sum_{l=1}^L \theta^{(l)}$
     - `mean_geodesic_step_rad`: float $\frac{1}{L}\sum_{l=1}^L \theta^{(l)}$
     - `grad_norm_cv`: float coefficient of variation ($\text{std}/\text{mean}$) of per-layer parameter gradient Frobenius norms $\left\{\sqrt{\sum_{p \in \text{block}_l} \|\nabla_p \mathcal{L}\|_F^2}\right\}_{l=1}^L$ (or `0.0` if `targets is None`).

### 3.2 Updates to `QuantumBornBlock` (`src/qdlblocks/layers/born_block.py`) and `QuantumBornGPT` (`src/qdlblocks/models/born_gpt.py`)

1. **`QuantumBornBlock` (`src/qdlblocks/layers/born_block.py`)**:
   - Add `frame_expansion: float = 2.0` and `residual_scale: float = 1.0` to `__init__` (defaults preserve 100% backward compatibility).
   - When `self.use_overcomplete_two_beam` is `True`, pass `frame_expansion=frame_expansion` and `target_iso_scale=default_overcomplete_target_iso_scale(frame_expansion)` to `OvercompleteTwoBeamInterference`.
   - In `forward`, apply `self.residual_scale`:
     ```python
     if self.strict_spherical_residual:
         a = self.attn(x, cos=cos, sin=sin, psi_0=psi_0)
         x = self._spherical_norm(x + self.residual_scale * a)
         h = self._channel_forward(x)
         x = self._spherical_norm(x + self.residual_scale * h)
     else:
         x = x + self.residual_scale * self.attn(self._spherical_norm(x), cos=cos, sin=sin, psi_0=psi_0)
         h = self._channel_forward(self._spherical_norm(x))
         x = x + self.residual_scale * h
     ```
2. **`QuantumBornGPT` (`src/qdlblocks/models/born_gpt.py`)**:
   - Expand `ReadoutMode = Literal["asymmetric", "symmetric_dual", "tied_stiefel", "tied_vocab"]`.
   - Add `frame_expansion: float = 2.0` and `depth_scaled_residual: bool = False` to `QuantumBornGPT.__init__`.
   - Compute `res_scale = compute_depth_residual_scale(num_layers, base_layers=6, enabled=depth_scaled_residual)` and pass `frame_expansion=frame_expansion, residual_scale=res_scale` to each `QuantumBornBlock`.
   - For `readout_mode == "tied_vocab"`:
     - Allocate `self.readout_rot = nn.Linear(embed_dim, embed_dim, bias=False)` (`nn.init.eye_(self.readout_rot.weight)`) and `self.tied_inv_temp = nn.Parameter(torch.tensor(1.0 / max(povm_temperature, 1e-4)))`, while setting `self.register_parameter("povm_head_seed_delta", None)`.
     - In `forward`:
       ```python
       psi_final = l2_normalize(self.readout_rot(psi), dim=-1, eps=self.eps)
       phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)
       if self.seed_adapter is not None:
           phi_seed = self.seed_adapter(psi_final, phi_seed)
       if self.povm_head_seed_delta is not None:
           M_seed = l2_normalize(self.wte.weight + self.povm_head_seed_delta, dim=1, eps=self.eps).to(phi_seed.dtype)
       else:
           M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)
       inv_t_primary = torch.clamp(self.tied_inv_temp, min=1.0, max=60.0)
       logits = F.linear(phi_seed, M_seed) * inv_t_primary.to(phi_seed.dtype)
       ```

### 3.3 New JAX Module: `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/parameter_scaling.py`

Create `src/qdlblocks_jax/parameter_scaling.py` (and export in `src/qdlblocks_jax/__init__.py`) with:
- `default_overcomplete_target_iso_scale_jax(frame_expansion: float) -> float`
- `compute_depth_residual_scale_jax(num_layers: int, base_layers: int = 6, enabled: bool = True) -> float`
- `compute_depth_fidelity_and_geodesic_profile_jax(psi_0: jax.Array, layer_states: list[jax.Array], eps: float = 1e-6) -> dict[str, float | list[float]]`

### 3.4 Updates to `/home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py`

Add CLI flags to `parse_args`, `run_cuda_benchmark`, `run_tpu_benchmark`, and `main`:
- `--num-layers` (int, default `6`)
- `--embed-dim` (int, default `1024`)
- `--num-heads` (int, default `16`)
- `--hidden-dim` (int, default `1536`)
- `--frame-expansion` (float, default `2.0`)
- `--readout-mode` (str, default `"tied_stiefel"`, choices `["tied_stiefel", "tied_vocab", "symmetric_dual", "asymmetric"]`)
- `--depth-scaled-residual` (action `"store_true"`)

Update `forward_backbone`, `compute_step_loss`, and `evaluate_validation` in `run_cuda_benchmark` (and TPU config in `run_tpu_benchmark`):
- Support `readout_mode == "asymmetric"` (`phi_seed = psi`, `M_seed = l2_normalize(m.povm_readout.povm_head, dim=1)`, `inv_t = torch.clamp(m.povm_readout.inv_temp, min=1.0, max=60.0)`, `out_dim = m.embed_dim`), `readout_mode == "tied_vocab"` (`M_seed = l2_normalize(m.wte.weight, dim=1)`), and `readout_mode == "tied_stiefel"`.
- In `evaluate_validation`, when `model.embed_dim > 1024`, project/slice `sub_psi = l2_normalize(st.reshape(-1, model.embed_dim)[:128, :1024].float(), dim=-1)` before computing 10-qubit 5v5 `bipartite_entanglement_entropy`.
- Update the `Architecture:` header print line so `monitor_colab_runs.py` can parse `L={num_layers}`, `d={embed_dim}`, `d_ff={mlp_label}`, `Readout={readout_mode}`, and `Params: {n_params:,}`.

### 3.5 Controlled Hypothesis Script: `/home/festivity/agy/QuantumDLBlocks/examples/22_hypothesis_parameter_scaling_axes.py`

Create `examples/22_hypothesis_parameter_scaling_axes.py` which runs deterministic controlled experiments testing H1, H2, and H3 on the contextual multi-shard stream (`V=1024, d=64, D_seed=24, B=16, T=64, 250 steps across 5 non-repeating shards`) and writes `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_012.json`:
1. **Baseline (`L=6, d=64, D_seed=24, rho=2.0, tied_stiefel`)**.
2. **Sparse Vocab Expansion Control**: Adds an untied auxiliary vocabulary delta table to match the parameter increase of $L=12$ without adding backbone capacity.
3. **Axis 1 (`L=12, d=64, D_seed=24, rho=2.0, depth_scaled_residual=True` vs `depth_scaled_residual=False`)**: Verifies H1a (initial fidelity $\ge 0.18$, gradient norm CV reduction $\ge 15\%$) and H1b ($\Delta \text{CE} \le -0.08$ nats and $\ge 2.0\times$ higher $\Delta \text{CE}/\Delta P_{\text{M}}$ than sparse vocab expansion).
4. **Axis 2 (`L=6, d=64, rho=3.0` and `rho=4.0`)**: Verifies H2a ($R_{\text{eff}}(\rho=4.0) > R_{\text{eff}}(\rho=3.0) > R_{\text{eff}}(\rho=2.0)$ with $\ge 1.15\times$ ratio) and H2b ($\Delta \text{CE} \le -0.05$ nats).
5. **Axis 3 (`L=6, d=96, H=6, rho=2.0`)**: Verifies H2b width scaling ($\Delta \text{CE} \le -0.05$ nats).
6. **Axis 4 (`L=10, d=64, tied_vocab, depth_scaled_residual=True` and `L=6, d=64, asymmetric`)**: Verifies H3a (iso-parameter `tied_vocab` $L=10$ beats `tied_stiefel` $L=6$ by $\ge 0.06$ nats) and H3b (`asymmetric` readout $R_{\text{eff}}(\Psi^{(L)}) > D_{\text{seed}}$ and $\Delta \text{CE} \le -0.05$ nats).

---

## 4. Verification & Unit Testing Plan

Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_012_parameter_scaling.py` with 5 unit tests:
1. `test_parameter_scaling_breakdown_and_residual_scale()`: Verifies exact parameter accounting on `QuantumBornGPT` across `tied_stiefel`, `tied_vocab`, and `asymmetric` modes and verifies `compute_depth_residual_scale(6) == 1.0` and `compute_depth_residual_scale(12) == sqrt(0.5)`.
2. `test_depth_scaled_residual_preserves_fidelity_and_reduces_grad_cv()`: Verifies H1a on a 12-layer `QuantumBornGPT` (`depth_scaled_residual=True` achieves initial end-to-end fidelity $\ge 0.18$ and reduces layerwise gradient norm CV by $\ge 15\%$ compared to `depth_scaled_residual=False`).
3. `test_overcomplete_two_beam_3d_and_4d_rank_monotonicity()`: Verifies H2a (`OvercompleteTwoBeamInterference` with `frame_expansion` $\in \{2.0, 3.0, 4.0\}$ achieves $R_{\text{eff}}(4.0) > R_{\text{eff}}(3.0) > R_{\text{eff}}(2.0)$ and $R_{\text{eff}}(4.0) \ge 1.15 \times R_{\text{eff}}(2.0)$).
4. `test_tied_vocab_and_asymmetric_readout_forward_backward()`: Verifies `QuantumBornGPT` forward/backward passes, parameter savings of `tied_vocab` ($-\,V \times D_{\text{seed}}$ params vs `tied_stiefel`), and un-bottlenecked output state rank of `asymmetric` readout.
5. `test_jax_parameter_scaling_parity()`: Verifies numerical parity ($\le 10^{-5}$) between PyTorch `compute_depth_fidelity_and_geodesic_profile` and JAX `compute_depth_fidelity_and_geodesic_profile_jax`.
