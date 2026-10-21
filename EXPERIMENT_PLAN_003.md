# EXPERIMENT_PLAN_003: Tangent-Space Stiefel Muon Projection & Quadratic Retraction to Eliminate the WSD High-IsoErr Plateau

**Document ID**: `EXPERIMENT_PLAN_003` (Revision 2)  
**Status**: Revised for Blind Sub-Agent Review  
**Target Repository**: `/home/festivity/agy/QuantumDLBlocks`  

---

## 1. Executive Summary & Self-Contained Background

### 1.1 Architectural & Optimizer Context
In `QuantumDLBlocks`, 2D weight matrices $W \in \mathbb{R}^{r \times c}$ of the quantum-geometric Transformer (`QuantumBornGPT`) represent isometric quantum state preparations and unitary/orthogonal operators constrained to the compact **Stiefel manifold**:
$$\text{St}(r, c) = \begin{cases} \{W \in \mathbb{R}^{r \times c} : W^\top W = I_c\} & \text{if } r \ge c, \\ \{W \in \mathbb{R}^{r \times c} : W W^\top = I_r\} & \text{if } r < c. \end{cases}$$
Departure from the Stiefel manifold is measured by the normalized Frobenius **Isometry Error**:
$$\text{IsoErr}(W) = \begin{cases} \frac{\|W^\top W - I_c\|_F}{\sqrt{c}} & \text{if } r \ge c, \\ \frac{\|W W^\top - I_r\|_F}{\sqrt{r}} & \text{if } r < c. \end{cases}$$

In the current optimizer (`StiefelMuon` in `src/qdlblocks/optim/stiefel_muon.py`), each optimization step updates momentum $M_t = \mu M_{t-1} + G_t$, computes a 5-step Newton-Schulz orthogonal polar factor $U_t = \text{NS}_5(M_t) \in \mathbb{R}^{r \times c}$, and applies an **ambient Euclidean step** with periodic cubic retraction:
$$W_{t+1} = W_t - \eta_t U_t - \lambda_{\text{stiefel}} W_t (W_t^\top W_t - I_c).$$

### 1.2 Empirical Anomaly Observed in Live 500M-Token FineWeb Runs
During our 5-epoch (`500M` token) Warmup-Stable-Decay (`WSD`) run on `NVIDIA RTX PRO 6000 Blackwell` (`gpu_slowrun`), the learning rate multiplier was held at $\text{lr\_mult} = 1.0$ ($\eta_t = 0.025$) from `10M` to `400M` tokens (`Epochs 0.10` to `4.00`), followed by a cooldown from `400M` to `500M` tokens (`Epochs 4.00` to `5.00`):
- **Plateau Phase (`100M` to `400M` tokens, $\eta_t = 0.025$)**: Validation Cross-Entropy (`Val CE`) stalled completely between `4.7592` and `4.6911`, while `IsoErr` remained pinned at an off-manifold equilibrium of **`1.14 - 1.23`**.
- **Cooldown Phase (`400M` to `500M` tokens, $\eta_t \to 0.0025$)**: The instant $\eta_t$ decayed, `IsoErr` collapsed from **`1.1419` to `0.1771`** and `Val CE` dropped **`-0.3629` nats in a single epoch** (`4.6911 -> 4.3282`).

### 1.3 Mathematical Root Cause of the High-`IsoErr` Plateau
Why does stepping along $-\eta_t U_t$ push $W_t$ off the Stiefel manifold to $\text{IsoErr} \approx 1.20$ when $\eta_t = 0.025$?
- Even though $U_t = \text{NS}_5(M_t)$ is itself approximately semi-orthogonal ($U_t^\top U_t \approx I_c$ for $r \ge c$), **$U_t$ does NOT lie in the Riemannian tangent space $T_{W_t}\text{St}(r, c)$ at $W_t$**!
- The tangent space to $\text{St}(r, c)$ at $W \in \text{St}(r, c)$ is defined by the first-order constraint:
  $$T_W\text{St}(r, c) = \begin{cases} \{\Delta \in \mathbb{R}^{r \times c} : W^\top \Delta + \Delta^\top W = 0\} & \text{if } r \ge c, \\ \{\Delta \in \mathbb{R}^{r \times c} : W \Delta^\top + \Delta W^\top = 0\} & \text{if } r < c. \end{cases}$$
- For an arbitrary ambient direction $U \in \mathbb{R}^{r \times c}$, stepping $W_{t+1} = W_t - \eta U$ produces a **first-order $O(\eta)$ normal excursion** off the manifold whenever the symmetric alignment $\text{sym}(W_t^\top U) = \frac{1}{2}(W_t^\top U + U^\top W_t) \ne 0$:
  $$W_{t+1}^\top W_{t+1} - I_c = (W_t^\top W_t - I_c) - 2 \eta \, \text{sym}(W_t^\top U) + \eta^2 U^\top U.$$
- Conversely, if we apply the **Riemannian Stiefel Tangent Projection** $\Pi_{T_W}$:
  $$\Delta = \Pi_{T_W}(U) = \begin{cases} U - W \, \text{sym}(W^\top U) = U - \frac{1}{2} W (W^\top U + U^\top W) & \text{if } r \ge c, \\ U - \text{sym}(U W^\top) W = U - \frac{1}{2} (U W^\top + W U^\top) W & \text{if } r < c, \end{cases}$$
  both before and after $\text{NS}_5$ ($\Delta_{\text{tan}} = \Pi_{T_W}(\text{NS}_5(\Pi_{T_W}(G)))$, since right-multiplication of $\Pi_{T_W}(G)$ by the Newton-Schulz polynomial $P$ does not commute with $W^\top$), then for $W \in \text{St}(r, c)$, $W^\top \Delta_{\text{tan}} + \Delta_{\text{tan}}^\top W = 0$ identically! Thus, the first-order $O(\eta)$ off-manifold error **vanishes identically**, leaving only a second-order $O(\eta^2)$ curvature term $\eta^2 \Delta_{\text{tan}}^\top \Delta_{\text{tan}}$.
- Applying a single **First-Order Quadratic Stiefel Retraction** step after the tangent update:
  $$\mathcal{R}(W_{\text{tent}}) = \begin{cases} W_{\text{tent}} - \frac{1}{2} W_{\text{tent}} (W_{\text{tent}}^\top W_{\text{tent}} - I_c) & \text{if } r \ge c, \\ W_{\text{tent}} - \frac{1}{2} (W_{\text{tent}} W_{\text{tent}}^\top - I_r) W_{\text{tent}} & \text{if } r < c, \end{cases}$$
  cancels the $O(\eta^2)$ error as well, keeping $\text{IsoErr}(W_t) = O(\eta^4) \ll 0.05$ at every single step even at constant high learning rates ($\eta = 0.03 - 0.05$)!

---

## 2. Observable Hypotheses

### Hypothesis H1: First-Order Orthogonality of Riemannian Stiefel Tangent Projection
- **Mechanism**:
  - Let $W \in \text{St}(r, c)$ be an exact semi-orthogonal matrix ($W^\top W = I_c$ for $r \ge c$, or $W W^\top = I_r$ for $r < c$).
  - Construct a synthetic gradient $G = E / \|E\|_F + 0.5 W$ (where $E \sim \mathcal{N}(0, 1)^{r \times c}$ with fixed seed `42`), which contains both a tangential component and a normal component along $W$ (as arises when weight decay or unprojected momentum accumulates along $W$).
  - Compare the single-step ambient update (with aspect ratio scale $\alpha_{\text{aspect}} = \sqrt{\max(1, r/c)}$ and $\eta = 0.05$):
    $$W_{\text{raw}} = W - \eta \, \alpha_{\text{aspect}} \, \text{NS}_5(G)$$
    against the double-tangent-projected + quadratic-retracted update:
    $$W_{\text{tan}} = \mathcal{R}\!\left(W - \eta \, \alpha_{\text{aspect}} \, \Pi_{T_W}\!\left(\text{NS}_5(\Pi_{T_W}(G))\right)\right).$$
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Exact Skew-Symmetry in Tangent Space**: For $\Delta = \Pi_{T_W}(G)$:
     - For $r \ge c$ ($128 \times 64$): $\|W^\top \Delta + \Delta^\top W\|_F / \|G\|_F \le 10^{-5}$.
     - For $r < c$ ($64 \times 128$): $\|W \Delta^\top + \Delta W^\top\|_F / \|G\|_F \le 10^{-5}$.
  2. **Second-Order vs. First-Order Manifold Drift**: For both $(r, c) = (128, 64)$ and $(r, c) = (64, 128)$ at $\eta = 0.05$:
     - The ambient step suffers first-order normal drift: $\text{IsoErr}(W_{\text{raw}}) \ge 0.04$.
     - The tangent-projected + quadratic-retracted step satisfies:
       $$\text{IsoErr}(W_{\text{tan}}) \le 0.005, \qquad \text{and} \qquad \text{IsoErr}(W_{\text{tan}}) \le 0.10 \times \text{IsoErr}(W_{\text{raw}}).$$

### Hypothesis H2: Elimination of the High-`IsoErr` Plateau Under Constant High-LR WSD Training
- **Mechanism**: Compare standard `StiefelMuon` (`Arm A`, ambient update with `lr=0.03, weight_decay=0.01, decay_mode="cautious", stiefel_penalty=0.005, retract_every=4`) against `TangentStiefelMuon` (`Arm B`, tangent projection $\Pi_{T_W}$ before and after $\text{NS}_5$ + quadratic retraction $\mathcal{R}$ every step with `lr=0.03, retract_alpha=0.5, retract_every=1`) when training two identically initialized `QuantumBornGPT` models under a constant high learning rate ($\eta_{\text{muon}} = 0.03$, WSD stable plateau with zero LR decay) for $S = 100$ steps.
- **Observable Predictions & Quantitative Pass Criteria**:
  1. **Manifold Preservation on the WSD Stable Plateau**: After $S = 100$ steps at constant $\eta_{\text{muon}} = 0.03$, the mean model isometry error across all 2D Stiefel parameters satisfies:
     $$\text{IsoErr}(\text{Arm B}) \le 0.05, \qquad \text{and} \qquad \text{IsoErr}(\text{Arm B}) \le 0.15 \times \text{IsoErr}(\text{Arm A}).$$
  2. **Lower Validation Cross-Entropy Without Requiring Cooldown**: At the end of the $S = 100$ constant-LR plateau, `Arm B` (`TangentStiefelMuon`) achieves lower validation cross-entropy (`val_ce`) than `Arm A` (`StiefelMuon`) by at least **`0.05` nats**:
     $$\text{val\_ce}(\text{Arm B}) \le \text{val\_ce}(\text{Arm A}) - 0.05.$$

---

## 3. Planned Code Changes (Disjoint New Modules & Files)

To allow completely conflict-free parallel implementation alongside other ongoing tasks, all changes are encapsulated in **new, dedicated files**:

### 3.1 PyTorch Tangent-Space Stiefel Muon (`src/qdlblocks/optim/tangent_stiefel_muon.py`)

Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/optim/tangent_stiefel_muon.py` implementing:

1. **`project_stiefel_tangent(w: torch.Tensor, u: torch.Tensor) -> torch.Tensor`**:
   - **Input**: 2D tensors `w` and `u` of identical shape `(r, c)`. Raises `ValueError` if `w.ndim != 2` or `u.shape != w.shape`.
   - **Computation** (performed in `torch.float32`, then cast back to `u.dtype`):
     ```python
     wf = w.float()
     uf = u.float()
     r, c = wf.shape
     if r >= c:
         wt_u = torch.matmul(wf.transpose(-2, -1), uf)
         sym = 0.5 * (wt_u + wt_u.transpose(-2, -1))
         out = uf - torch.matmul(wf, sym)
     else:
         u_wt = torch.matmul(uf, wf.transpose(-2, -1))
         sym = 0.5 * (u_wt + u_wt.transpose(-2, -1))
         out = uf - torch.matmul(sym, wf)
     return out.to(u.dtype)
     ```

2. **`stiefel_quadratic_retract(w: torch.Tensor, alpha: float = 0.5) -> torch.Tensor`**:
   - **Input**: 2D tensor `w` of shape `(r, c)` and retraction coefficient `alpha: float = 0.5`. Raises `ValueError` if `w.ndim != 2` or `alpha < 0.0`.
   - **Computation** (in `torch.float32`, cast back to `w.dtype`):
     ```python
     wf = w.float()
     r, c = wf.shape
     if r >= c:
         gram = torch.matmul(wf.transpose(-2, -1), wf)
         eye = torch.eye(c, device=w.device, dtype=torch.float32)
         out = wf - alpha * torch.matmul(wf, gram - eye)
     else:
         gram = torch.matmul(wf, wf.transpose(-2, -1))
         eye = torch.eye(r, device=w.device, dtype=torch.float32)
         out = wf - alpha * torch.matmul(gram - eye, wf)
     return out.to(w.dtype)
     ```

3. **`class TangentStiefelMuon(torch.optim.Optimizer)`**:
   - **Constructor Signature**:
     ```python
     def __init__(
         self,
         params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
         lr: float = 0.02,
         momentum: float = 0.95,
         nesterov: bool = True,
         ns_steps: int = 5,
         retract_alpha: float = 0.5,
         retract_every: int = 1,
         tangent_project_before_ns: bool = True,
         tangent_project_after_ns: bool = True,
     ) -> None:
     ```
   - **Validation**:
     - Raises `ValueError` if `lr < 0.0`, `not (0.0 <= momentum < 1.0)`, `ns_steps < 1`, `retract_alpha < 0.0`, or `retract_every < 1`.
   - **Step Update (`@torch.no_grad() def step(self, closure=None)`)**:
     - For each parameter group `group` and each parameter `p` with `p.grad is not None`:
       - Let `g = p.grad`, `state = self.state[p]`.
       - If `"momentum_buffer"` not in `state`:
         `state["momentum_buffer"] = torch.zeros_like(p.data)`
         `state["step"] = 0`
       - `state["step"] += 1`
       - `buf = state["momentum_buffer"]`
       - `buf.mul_(group["momentum"]).add_(g)`
       - `g_dir = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf`
       - If `p.ndim == 2`:
         - If `group["tangent_project_before_ns"]`:
           `g_dir = project_stiefel_tangent(p.data, g_dir)`
         - `u = zeropower_via_newtonschulz5(g_dir, steps=group["ns_steps"])` (imported from `..functional`).
         - `aspect_scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))`
         - `u = u * aspect_scale`
         - If `group["tangent_project_after_ns"]`:
           `u = project_stiefel_tangent(p.data, u)`
         - `p.data.add_(u.to(p.dtype), alpha=-group["lr"])`
         - If `group["retract_alpha"] > 0.0 and (state["step"] % group["retract_every"] == 0)`:
           `p.data.copy_(stiefel_quadratic_retract(p.data, alpha=group["retract_alpha"]))`
       - Else (`p.ndim != 2`):
         - `p.data.add_(g_dir.to(p.dtype), alpha=-group["lr"])`

---

### 3.2 JAX Tangent-Space Stiefel Transformations (`src/qdlblocks_jax/tangent_stiefel.py`)

Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/tangent_stiefel.py` implementing pure-JAX `@jax.jit`-compatible equivalents with exact PyTorch parity:
1. **`project_stiefel_tangent_jax(w: jax.Array, u: jax.Array) -> jax.Array`**:
   - Exact JAX equivalent of `project_stiefel_tangent` for 2D `(r, c)` arrays (`r >= c` and `r < c`), computed in `jnp.float32` and cast back to `u.dtype`. Raises `ValueError` if `w.ndim != 2` or `u.shape != w.shape`.
2. **`stiefel_quadratic_retract_jax(w: jax.Array, alpha: float = 0.5) -> jax.Array`**:
   - Exact JAX equivalent of `stiefel_quadratic_retract` for 2D `(r, c)` arrays. Raises `ValueError` if `w.ndim != 2` or `alpha < 0.0`.
3. **`tangent_stiefel_muon_step_jax(w: jax.Array, g: jax.Array, m: jax.Array, *, lr: float | jax.Array = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, retract_alpha: float = 0.5, tangent_project_before_ns: bool = True, tangent_project_after_ns: bool = True) -> tuple[jax.Array, jax.Array]`**:
   - Updates `new_m = momentum * m + g`.
   - Computes `g_dir = (g + momentum * new_m) if nesterov else new_m`.
   - If `tangent_project_before_ns`: `g_dir = project_stiefel_tangent_jax(w, g_dir)`.
   - Runs `u = zeropower_via_newtonschulz5(g_dir, steps=ns_steps) * math.sqrt(max(1.0, w.shape[0] / w.shape[1]))` (from `.functional`).
   - If `tangent_project_after_ns`: `u = project_stiefel_tangent_jax(w, u)`.
   - Updates `w_tent = w - lr * u.astype(w.dtype)`.
   - If `retract_alpha > 0.0`: `w_next = stiefel_quadratic_retract_jax(w_tent, alpha=retract_alpha)` else `w_tent`.
   - Returns `(w_next, new_m)`.

---

### 3.3 Standalone Controlled Hypothesis Verification Script (`examples/13_hypothesis_tangent_stiefel_wsd.py`)

Create `/home/festivity/agy/QuantumDLBlocks/examples/13_hypothesis_tangent_stiefel_wsd.py` that runs on CPU or GPU in `< 30` seconds, prints a formatted comparison table, asserts all quantitative predictions of Hypotheses H1 and H2, and saves `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_003.json`:

1. **Experiment 1 (H1 — Tangent Space Orthogonality & Single-Step Drift)**:
   - For both $(r, c) = (128, 64)$ and $(r, c) = (64, 128)$ with `torch.manual_seed(42)`:
     - Construct $W \in \text{St}(r, c)$ via QR decomposition (`nn.init.orthogonal_`), and construct $G = E / \|E\|_F + 0.5 W$ where $E = \text{torch.randn}(r, c)$.
     - Compute $\Delta = \text{project\_stiefel\_tangent}(W, G)$ and assert skew-symmetry error $\le 10^{-5}$ in both PyTorch and JAX.
     - With $\eta = 0.05$ and $\alpha_{\text{aspect}} = \sqrt{\max(1.0, r/c)}$, compute $W_{\text{raw}} = W - \eta \alpha_{\text{aspect}} \text{NS}_5(G)$ and $W_{\text{tan}} = \mathcal{R}(W - \eta \alpha_{\text{aspect}} \Pi_{T_W}(\text{NS}_5(\Pi_{T_W}(G))))$.
     - Assert $\text{IsoErr}(W_{\text{raw}}) \ge 0.04$, $\text{IsoErr}(W_{\text{tan}}) \le 0.005$, and $\text{IsoErr}(W_{\text{tan}}) \le 0.10 \times \text{IsoErr}(W_{\text{raw}})$.

2. **Experiment 2 (H2 — Constant-LR WSD Plateau Training on `QuantumBornGPT`)**:
   - **Deterministic Synthetic Markov Dataset**:
     - With `np.random.default_rng(42)`, generate a transition table `next_tok = (np.arange(256) * 5 + 11) % 256` and a 2-gram context offset `ctx_offset = (np.arange(256) * 13) % 8`. Generate `train_data` of shape `(256, 65)` and `val_data` of shape `(64, 65)` where token $t+1$ is `(next_tok[x_t] + ctx_offset[x_{t-1}]) % 256` with probability `0.90` and uniform random in `[0, 255]` with probability `0.10`.
   - **Model & Parameter Routing**:
     - Instantiate two identically seeded (`torch.manual_seed(123)`) `QuantumBornGPT(vocab_size=256, embed_dim=64, seed_dim=32, num_heads=4, num_layers=2, hidden_dim=128, block_size=64, readout_mode="tied_stiefel", learnable_temperature=True)` models.
     - Split parameters using `split_muon_adamw_params(model, adamw_name_patterns=("wte", "povm_head", "embed", "unet_scales", "smear"))` (imported from `qdlblocks`).
     - Define model `IsoErr` as the arithmetic mean of $\text{IsoErr}(p)$ over all $p \in \text{muon\_params}$.
   - **Optimizer Arms (100 Steps at Constant WSD Plateau LR $\eta_{\text{muon}} = 0.03, \eta_{\text{adamw}} = 0.008$, `batch_size = 16`)**:
     - **Arm A (`StiefelMuon`)**: `StiefelMuon(muon_params, lr=0.03, momentum=0.95, weight_decay=0.01, decay_mode="cautious", stiefel_penalty=0.005, ns_steps=5, retract_every=4)` + `torch.optim.AdamW(adamw_params, lr=0.008, weight_decay=0.01)`.
     - **Arm B (`TangentStiefelMuon`)**: `TangentStiefelMuon(muon_params, lr=0.03, momentum=0.95, nesterov=True, ns_steps=5, retract_alpha=0.5, retract_every=1)` + `torch.optim.AdamW(adamw_params, lr=0.008, weight_decay=0.01)`.
   - **Assertions**:
     - `final_iso_err(Arm B) <= 0.05`
     - `final_iso_err(Arm B) <= 0.15 * final_iso_err(Arm A)`
     - `final_val_ce(Arm B) <= final_val_ce(Arm A) - 0.05`
   - **JSON Output Schema (`.cache/hypothesis_experiments_003.json`)**:
     - Keys: `{"h1_tangent_projection": {"tall_128x64": {...}, "wide_64x128": {...}, "passed": True}, "h2_wsd_plateau_training": {"arm_a_stiefel_muon": {...}, "arm_b_tangent_stiefel_muon": {...}, "passed": True}}`.

---

## 4. Testing & Verification Plan

### 4.1 Unit Tests (`tests/test_hypothesis_003_tangent_stiefel.py`)
Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_003_tangent_stiefel.py` testing:
1. **`test_project_stiefel_tangent_skew_symmetry_torch_and_jax`**:
   - Verifies $\|W^\top \Delta + \Delta^\top W\|_F / \|U\|_F < 10^{-5}$ for tall ($64 \times 32$), square ($32 \times 32$), and wide ($32 \times 64$, $\|W \Delta^\top + \Delta W^\top\|_F / \|U\|_F < 10^{-5}$) Stiefel matrices in both PyTorch and `@jax.jit`-compiled JAX.
2. **`test_stiefel_quadratic_retract_cubic_convergence_torch_and_jax`**:
   - Perturbs an orthonormal matrix $W_0$ by $\epsilon E$ ($\text{IsoErr} \approx 0.15$) and verifies that a single call to `stiefel_quadratic_retract` / `stiefel_quadratic_retract_jax` reduces `IsoErr` by $> 5\times$, and 3 iterations reduce `IsoErr` to $< 10^{-5}$.
3. **`test_tangent_stiefel_muon_optimizer_step_and_validation`**:
   - Verifies constructor input validation (`ValueError` on invalid `lr`, `momentum`, `ns_steps`, `retract_alpha`, `retract_every`), verifies both 2D and 1D parameter handling in `TangentStiefelMuon.step`, and verifies PyTorch-JAX single-step parity with `tangent_stiefel_muon_step_jax` within `atol=1e-5`.
4. **`test_hypothesis_003_script_execution`**:
   - Executes `examples/13_hypothesis_tangent_stiefel_wsd.py` and verifies that `.cache/hypothesis_experiments_003.json` is written with `"passed": True` for both H1 and H2.
