# Experiment Plan 019 — Exact `KellerJordan/modded-nanogpt` Leaderboard Parity (`<= 3.280` Target), Dashboard Stale-`RUNNING` Fix (`w7_900m_qborn_earlyfix_frame4d_synergy512`), & Wave-11 900M-Token Fleet

## 1. Executive Summary & Project Context (Self-Contained Overview for Independent Review)

### 1.1 What Is QuantumDLBlocks?
**QuantumDLBlocks** (`qdlblocks`) is a research and engineering project investigating whether **quantum-inspired neural architectures** built on **isometric/unitary state evolution** and **Born-rule / Positive Operator-Valued Measure (POVM) quadratic measurement readouts** can match or approach the sample efficiency and wall-clock throughput of state-of-the-art classical Transformer language models on the **FineWeb-Edu 10B** benchmark (tokenized with the GPT-2 vocabulary of $V = 50{,}257$ tokens, padded to $50{,}304$ for GPU tensor-core alignment).

Throughout the project, we compare two primary model families under identical hardware, dataset streams, and evaluation protocols:
1. **`QuantumBornGPT` (`quantum_born`)**:
   - Represents token hidden states as unit-norm wavefunctions $|\psi\rangle \in \mathbb{R}^d$ ($\|\psi\|_2 = 1$) evolving on the unit hypersphere $\mathcal{S}^{d-1}$.
   - Replaces standard softmax dot-product attention with **Purified Subspace-Projector Born Attention** (`flash_purified_born_attention`), a custom fused Triton kernel computing squared-amplitude Born probabilities $A_{ti} \propto (Q_t \cdot K_i)^2 + \tau (Q_t \cdot K_i)$ with causal, sliding-window, and document-boundary (`<|endoftext|>` token `50256`) masking.
   - Replaces the standard MLP with **Overcomplete Two-Beam Interference (`OvercompleteTwoBeamInterference`)**, projecting $|\psi\rangle$ through two orthogonal Stiefel-manifold frames $W_1, W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, interfering their non-linear activations, and projecting back via $W_{\text{out}}$ followed by spherical retraction $\psi \leftarrow \text{normalize}(\psi + \alpha \Delta\psi)$.
   - Computes next-token log-probabilities via an **Asymmetric Projective POVM Readout (`readout_mode="asymmetric"`)**, where each vocabulary token $v \in \{0, \dots, V-1\}$ has a unit-norm measurement effect vector $m_v \in \mathcal{S}^{d_{\text{seed}}-1}$ and logits are given by $z_v = \beta(x) \cdot (\langle m_v | \psi_{\text{seed}} \rangle^2 + \gamma_{\text{head}} \langle m_v | \psi_{\text{context}} \rangle) + b_{\text{prior}, v}$, where $\beta(x)$ is a context-dependent Bloch-purity inverse temperature (`ContextualBlochPurityGate`).
   - Optimizes 2D weight matrices on the Stiefel manifold $\{W : W^\top W = I\}$ using **`StiefelMuon`** (quintic Newton-Schulz polar decomposition of momentum gradients + continuous Riemannian Stiefel retraction penalty $\frac{1}{4}\|W^\top W - I\|_F^2$) and 1D/embedding parameters via **`AdamW`**.
2. **`ClassicalModdedGPT` (`classical_modded`)**:
   - Our classical Transformer baseline modeled after the **`KellerJordan/modded-nanogpt` NanoGPT Speedrun Leaderboard** (specifically including the **`PR#360`** milestone by `@YouJiacheng`, `@KoszarskyB`, `@Grad62304977`, and `@brendanh0gan`, which introduced a bank of 3 token value embeddings, scalar U-Net encoder-to-decoder skip connections, half-truncated RoPE, and interior attention-layer skipping), which reaches **$\le 3.280$ Validation Cross-Entropy (Val CE)** on FineWeb in $\sim 730\text{M}$–$900\text{M}$ training tokens.

**Core Telemetry & Schedule Definitions**:
- **Validation Cross-Entropy (`Val CE`) & Perplexity (`Val PPL`)**: Across all waves and both model families, validation loss (`Val CE`, in nats; Perplexity $\text{PPL} = e^{\text{Val CE}}$) is evaluated on the held-out FineWeb validation shard (`fineweb_val_000000.bin`) at a strictly fixed evaluation sequence length of **`block_size = 512` tokens (`T_val = 512`)**.
- **Isometry Error (`IsoErr`)**: Mean Stiefel manifold orthogonality deviation across all 2D backbone weight matrices $W \in \mathbb{R}^{m \times n}$, defined as $\text{IsoErr} = \frac{1}{N_{\text{mat}}} \sum_{W} \|W^\top W - I_n\|_F$ (when $m \ge n$) or $\|W W^\top - I_m\|_F$ (when $m < n$). For `QuantumBornGPT` with Stiefel retraction penalty, $\text{IsoErr} \approx 0.1$–$1.5$; for unconstrained `ClassicalModdedGPT`, $\text{IsoErr}$ measures how far the Euclidean weights have grown from unit spectral norm.
- **Warmup-Stable-Decay (`WSD`) Schedule (`wsd_lr_multiplier`)**: A 3-phase learning rate schedule that linearly warms up from $0$ to $1.0$ over `warmup_steps` (or starts directly at $1.0$ when `warmup_steps = 0`), holds constant at $1.0$ during the stable plateau, and linearly cools down over the final `wsd_decay_fraction` of training steps to `wsd_min_lr_ratio` (e.g., `WSD-0` denotes `wsd_min_lr_ratio = 0.0`).

---

### 1.2 Summary of Completed Waves (Waves 1–10) & Overnight Wave-10 Breakthroughs

| Wave / Run Key | Architecture Family | Token Budget | Key Architectural & Optimization Configuration | Best Val CE (`T_val=512`) | Val PPL | Throughput (`tok/s`) | Status / Outcome |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Baseline (`gpu_5ep`)** | `QuantumBornGPT` | `500M` (`5×100M`) | Early $L=6, d=768$ symmetric Born readout, cosine LR | `4.5446` | `94.1` | `83,421` | Historical baseline |
| **Wave 1 (`w1_g4_1`)** | `QuantumBornGPT` | `300M` (`3×100M`) | $L=6, d=1024$, asymmetric POVM readout, $d_{\text{seed}}=512$ | `4.0846` | `59.4` | `57,800` | Validated asymmetric POVM (`-0.460` nats) |
| **Wave 2B (`w2b_g4_3`)** | `QuantumBornGPT` | `900M` (`9×100M`) | $L=8, d=1024$, $2d$ Two-Beam MLP ($2048\times 2$), $T=512$ | `3.8295` | `46.0` | `84,565` | First sub-`3.85` quantum run |
| **Wave 4 (`w4_g4_3`)** | `QuantumBornGPT` | `900M` (`9×100M`) | $L=8, d=1024$, $4d$ Overcomplete Two-Beam ($4096\times 2$) | `3.8027` | `44.8` | `66,574` | Validated $4d$ frame expansion |
| **Wave 5 (`w5_g4_3`)** | `QuantumBornGPT` | `900M` (`9×100M`) | $L=8, d=1024$, Full-Width Asymmetric Seed ($d_{\text{seed}}=1024$) | `3.7764` | `43.7` | `83,039` | Eliminated $d_{\text{seed}}=512$ bottleneck |
| **Wave 6 (`w6_a100_1`)** | `ClassicalModdedGPT` | `900M` (`9×100M`) | Initial Classical Modded-NanoGPT ($L=12, d=768, H=12$, orthogonal init) | `3.6797` | `39.6` | `151,836` | First classical baseline (`IsoErr=159.3`) |
| **Wave 6 (`w6_g4_2`)** | `QuantumBornGPT` | `900M` (`9×100M`) | $L=8, d=1024$, Dynamic Bloch-Purity Gate $\beta(x)$ (`DynBeta`) | `3.7972` | `44.6` | `78,790` | Validated context-dependent $\beta(x)$ |
| **Wave 7 (`w7_g4_1` = `w7_900m_qborn_earlyfix_frame4d_synergy512`)** | `QuantumBornGPT` | `609.5M / 900M` (Preempted at Step `9300/13725`) | Unfused PyTorch `purified_born_attention_weights` + `OrthoDecouple` + `POVMLogPrior` + `GatedSphericalRes` + $4d$-Frame | `3.9365` (at `609.5M` tok) | `51.2` | `79,432` | Preempted when we built `2.79x` faster fused Triton `flash_purified_born_attention`; succeeded by `w9_g4_1` (`3.7495`) & `w10_g4_3` (`3.6926`) |
| **Wave 8 (`w8_a100_1`)** | `ClassicalModdedGPT` | `900M` (`9×100M`) | Added `--zero-init-classical`, `--true-layer0-v-residual`, `--wsd-min-lr-ratio 0.0` | `3.5882` | `36.2` | `160,653` | Improved classical by `-0.0915` nats |
| **Wave 9 (`w9_a100_1`)** | `ClassicalModdedGPT` | `900M` (`9×100M`) | Added PR#360 `VE=3`, scalar U-Net skip, `FusedQKV`, `SkipAttn(6)`, `ZeroHalfRoPE`, `Win=384`, `DocMask` (`3,429` steps, $B_{\text{tok}}=262\text{k}$) | **`3.4280`** | **`30.8`** | `136,754` | Current #1 Classical run (`-0.1602` nats vs Wave 8), but still `+0.1480` nats behind `3.280` |
| **Wave 9 (`w9_a100_2`)** | `ClassicalModdedGPT` | `900M` (`9×100M`) | Optimizer Ablation #2: Pure `AdamW` ($\text{lr}=0.003$) on identical PR#360 V2 model | `3.5025` | `33.2` | `141,210` | `StiefelMuon` beats `AdamW` by `-0.0745` nats |
| **Wave 9 (`w9_g4_1`)** | `QuantumBornGPT` | `900M` (`9×100M`) | $L=8, d=1024, d_{\text{seed}}=512$, $4d$-Frame + `VE=3` + Scalar U-Net + `FlashPurifiedBorn(Win=384, DocMask)` | `3.7495` | `42.5` | `102,418` | Direct fused-Triton successor to `w7_g4_1` |
| **Wave 9 (`w9_g4_2`)** | `QuantumBornGPT` | `900M` (`9×100M`) | $L=8, d=1024, d_{\text{seed}}=1024$ (`Asym-1024`), $3d$-Frame + `VE=3` + `FlashPurifiedBorn` | `3.7203` | `41.3` | `118,904` | Wave-9 Quantum Record |
| **Wave 10 (`w10_g4_1`)** | `ClassicalModdedGPT` | `900M` (`9×100M`) | $B_{\text{tok}}=131\text{k}$ (`6,858` steps), `muon_lr=0.04`, unmasked causal SDPA | `3.4498` | `31.5` | `152,072` | Weight norm exploded (`IsoErr=185.18`) due to unscaled momentum + small batch + no doc/window mask |
| **Wave 10 (`w10_g4_2`)** | `ClassicalModdedGPT` | `900M` (`9×100M`) | $B_{\text{tok}}=65\text{k}$ (`13,725` steps), `muon_lr=0.035`, unmasked causal SDPA | `3.5145` | `33.6` | `148,526` | Over-stepped (`13,725` steps) at small batch (`IsoErr=153.48`), confirming Muon requires large batch |
| **Wave 10 (`w10_a100_2`)** | `ClassicalModdedGPT` | `900M` (`9×100M`) | Optimizer Ablation #3: Mini-Batch `SGD+Nesterov` ($\text{lr}=0.05, m=0.95$) on PR#360 V2 model | `4.3528` | `77.7` | `133,472` | Completes 3-way 900M Optimizer Study (`StiefelMuon` `3.4280` vs `AdamW` `3.5025` vs `SGD` `4.3528`) |
| **Wave 10 (`w10_a100_1`)** | `QuantumBornGPT` | `900M` (`9×100M`) | **Deep $L=12$ QuantumBornGPT** ($d=1024, d_{\text{seed}}=1024$, $3d$-Frame, `SkipAttn(6)`, `VE=3`, `FlashPurifiedBorn`) | **`3.7077`** | **`40.8`** | `68,821` | **NEW #2 ALL-TIME QUANTUM RECORD** (validating $L=12$ deep quantum architecture!) |
| **Wave 10 (`w10_g4_3`)** | `QuantumBornGPT` | `900M` (`9×100M`) | **Unified $L=8, d=1024, d_{\text{seed}}=1024$ (`Asym-1024`) + $4d$-Frame ($4096\times 2$) + `VE=3` + `FlashPurifiedBorn`** | **`3.6926`** | **`40.1`** | **`103,936`** | **NEW #1 ALL-TIME QUANTUM RECORD (First Sub-`3.70` Quantum Born Language Model!)** |

---

## 2. Answering Question 1 & Fixing the Dashboard Stale-`RUNNING` Bug: What Happened to `w7_900m_qborn_earlyfix_frame4d_synergy512`?

### 2.1 Why `w7_900m_qborn_earlyfix_frame4d_synergy512` Stopped at `609.5M` Tokens (`Val CE = 3.9365`)
In Wave 7 (Plan 014), `w7_900m_qborn_earlyfix_frame4d_synergy512` was launched on session `gpu_g4_w6_1` as our first test combining the four Plan-014 early-epoch fixes (`--ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn --dynamic-povm-beta`) with a $4d$ Overcomplete Two-Beam frame ($4096\times 2$) at $T=512$.
1. **Why it was slow (`79,432 tok/s`) and got preempted at `609.5M` tokens**:
   - In Wave 7, `--purified-born-attn` used the pure-PyTorch eager/compiled fallback `purified_born_attention_weights()`, which materialized the full $(B, H, T, T)$ attention matrix in HBM and reduced throughput to `79.4k tok/s`.
   - While Wave 7 was running (`Step 9300 / 13725`, `609.50M` tokens, `Val CE = 3.9365`), we engineered the **fused Triton `flash_purified_born_attention` kernel** (`src/qdlblocks/kernels/flash_born.py`), which boosted purified Born attention throughput by **`2.79x`** (and added PR#360 `VE=3`, scalar U-Net skips, `FusedQKV`, `ZeroHalfRoPE`, and sliding-window document masking).
   - We therefore reassigned `gpu_g4_w6_1` to launch the upgraded Wave-8 / Wave-9 / Wave-10 fleets.
   - Its direct fused-Triton successors on the exact same $4d$-Frame + EarlyFix + Speedrun architecture **completed all `900M` tokens** and achieved our best quantum results ever:
     - **`w9_900m_qborn_speedrun_v2_frame4d_synergy512`**: Completed `900M` tokens (`Step 6858/6858`) at **`Val CE = 3.7495`** (`102,418 tok/s`),
     - **`w10_900m_qborn_speedrun_v3_asym1024_frame4d_L8`**: Added full-width `seed_dim = 1024` (`Asym-1024`) to `w9_900m_qborn_speedrun_v2_frame4d_synergy512` and completed `900M` tokens at **`Val CE = 3.6926`** (`103,936 tok/s`, **#1 All-Time Quantum Record**)!

### 2.2 Why the Dashboard Still Showed `w7_900m_qborn_earlyfix_frame4d_synergy512` as `RUNNING` (and the Fix)
Inspection of `scripts/monitor_colab_runs.py` (lines 1463–1492 and `parse_run_log` lines 421–433) reveals a subtle state-resolution bug:
- To prevent duplicate `colab exec` calls when multiple waves share the same Colab session name (`gpu_g4_w6_1`), `collect_snapshot()` only polls the *latest* active run on each session and reads older waves from their cached local log file `.cache/live_logs/<run_key>.log` (`lines 1471–1474`):
  ```python
  else:
      text = local_log.read_text(errors="replace") if local_log.exists() else ""
      parsed = parse_run_log(text, ps_alive=None, gpu_str="")
  ```
- Inside `parse_run_log(log_text, ps_alive=None)` (`lines 421–430`), when a log has `steps` (or `epochs`) and does **not** contain `"Completed 100M-Token"` (because it was preempted at `609.5M / 900M` tokens), `parse_run_log` checks `if ps_alive is False: status = "STOPPED" else: status = "RUNNING"`!
- Because `ps_alive=None` was passed for non-polled historical runs (and even when `local_log` had not been modified for $> 10$ hours!), `parse_run_log` defaulted `status` to `"RUNNING"`, causing 5 superseded incomplete Wave-7/Wave-8 runs (`w7_900m_qborn_earlyfix_frame4d_synergy512`, `w7_900m_qborn_earlyfix_synergy_t1024`, `w7_900m_qborn_purified_attn_ablation`, `w8_900m_qborn_layer0v_earlyfix_frame4d_s512`, `w8_900m_qborn_layer0v_earlyfix_t1024`) to display a stale green `RUNNING` badge forever!
- **Exact Fix in `scripts/monitor_colab_runs.py`**:
  1. When a run is **not** in `active_poll_keys` (`r["key"] not in active_poll_keys`), it is no longer the active process on its Colab session; furthermore, if `local_log.exists()` and `(time.time() - local_log.stat().st_mtime) > 300.0`, no process is writing to that log. In both cases, when the log does not contain `"Completed 100M-Token"`, pass `ps_alive=False` into `parse_run_log(...)` so `status` is accurately reported as `"STOPPED"` (`0` false `"RUNNING"` badges!).
  2. Add those 5 superseded incomplete Wave-7/Wave-8 runs (plus `w6_900m_classical_modded_gpt_baseline`, `w8_900m_classical_truev0_zeroinit_wsd0`, `w9_900m_classical_speedrun_v2_t512_b524k`, `w10_900m_classical_speedrun_v3_fastsdpa_b131k`, and `w10_900m_classical_speedrun_v3_b65k_13725steps`) to `DEFAULT_OBSOLETE_KEYS` with clear provenance reasons so the default active leaderboard stays clean while keeping them 1-click accessible via **`🗄️ Show Obsolete`**.

---

## 3. Forensic Analysis of Question 2: Why Did Our `ClassicalModdedGPT` Reach `3.4280` Instead of the Official `<= 3.280` Speedrun Record?

To determine with 100% mathematical certainty why our best `ClassicalModdedGPT` run (`w9_900m_classical_speedrun_v2_pr360`, `3.4280` Val CE at `900M` tokens) remained `+0.1480` nats behind the official `KellerJordan/modded-nanogpt` record (`3.2785` Val CE in `1,393` steps = `730.3M` tokens in `records/track_1_short/2025-01-16_Sub3Min/train_gpt.py`), we performed a line-by-line forensic diff between:
- **Official `KellerJordan/modded-nanogpt` (`2025-01-16_Sub3Min` `train_gpt.py`)**, which reaches `4.4043` at `65.5M` tok (`Step 125`), `3.7856` at `196.6M` tok (`Step 375`), `3.5286` at `393.2M` tok (`Step 750`), `3.3164` at `655.4M` tok (`Step 1250`), and **`3.2785` at `730.3M` tok (`Step 1393`)**, and
- **Our `ClassicalModdedGPT` (`src/qdlblocks/models/classical_modded_gpt.py`), `StiefelMuon` (`src/qdlblocks/optim/muon.py`), and `run_cuda_benchmark` (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)**.

We discovered **9 concrete mathematical and algorithmic discrepancies** in our implementation that compounded from Step 1:

### 3.1 Complete 9-Point Forensic Diff Table (`Official modded-nanogpt` vs. `Our ClassicalModdedGPT`)

| # | Subsystem | Official `modded-nanogpt` (`3.2785` Val CE in `730M` tok) | Our Waves 6–10 `ClassicalModdedGPT` (`3.4280` Val CE in `900M` tok) | Quantitative Impact & Failure Mechanism |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Muon Momentum Buffer Formula & Weight Decay** | **EMA (`lerp_`) + Zero Weight Decay**:<br>`buf.lerp_(g, 1 - momentum)`<br>`g = g.lerp_(buf, momentum)`<br>`weight_decay = 0.0` | **Unscaled Sum (`mul_ + add_`) + Cautious WD**:<br>`buf.mul_(m).add_(g)`<br>`g_in = g.add(buf, alpha=m)`<br>`weight_decay = 0.01` | When gradient norms vary across steps or get clipped, unscaled sum `buf.mul_(m).add_(g)` weights historical steps unevenly vs `lerp_`, and combining `lr=0.035–0.04` over `6,858–13,725` small-batch steps caused weight norms to explode (`IsoErr = 185.18` in `w10_g4_1`). |
| **2** | **Optimizer Learning Rates, Betas, Epsilon & Grad Clipping** | **No Grad Clipping**, `Muon(lr=0.05, mom=0.85->0.95)`, `Adam(betas=(0.8, 0.95), eps=1e-10, wd=0.0)` with:<br>• `embed + ve`: **`lr = 0.6`**<br>• `scalars`: **`lr = 0.04`**<br>• `lm_head`: **`lr = 0.008`** | Global `clip_grad_norm_(..., 1.0)` every step! `Muon(lr=0.025–0.04)`, `AdamW(betas=(0.9, 0.95), eps=1e-8)` with:<br>• `wte + ve`: **`lr = 0.20` (`3x` too low!)**<br>• `scalars`: **`lr = 0.02` (`2x` too low!)**<br>• `lm_head`: `lr = 0.008` | Global `clip_grad_norm_(1.0)` crushes `wte`, `ve`, and scalar gradients whenever backbone gradients exceed `1.0`, while `3x` lower `lr_embed` (`0.20` vs `0.60`) and `2x` lower `lr_muon` (`0.025` vs `0.05`) starve token and feature learning. |
| **3** | **LR Warmup & Cooldown Schedule (`get_lr`)** | **`0` Warmup Steps** (`lr_mult = 1.0` at Step 0!), constant `1.0` for first `60%` of steps, linear cooldown over `cooldown_frac = 0.40` to **`min_lr_ratio = 0.10`**. | **`150` Warmup Steps** (`11%` of run spent ramping from `0.0`!), `decay_fraction = 0.25` to `min_lr_ratio = 0.0`. | Because `out_proj`, `fc2`, and `lm_head` are initialized to **exact zeros**, the network has zero output gradients at Step 0 until symmetry is broken; wasting 150 steps at tiny warmup LRs delays feature emergence by `~80M` tokens. |
| **4** | **Attention Head Dimension (`num_heads=6, head_dim=128`) & Custom `attn_scale=0.12`** | **`num_heads = 6, head_dim = 128`** ($d = 768$) and custom softmax scale **`self.attn_scale = 0.12`** (which is $1.36\times$ sharper than $1/\sqrt{128} = 0.0884$). | **`num_heads = 12, head_dim = 64`** ($d = 768$) and default PyTorch SDPA scale $1/\sqrt{64} = 0.125$. | Halving `head_dim` from `128` to `64` (and thus active RoPE dimensions per head from `64` to `32`) halves per-head key/value subspace capacity. |
| **5** | **Half-Truncated RoPE Wavelength (`base = 1024`) & QK-Norm Order** | **RoPE `base = 1024.0`** with `linspace(0, 1, steps=head_dim//4)` + `zeros(head_dim//4)`, applied to half-chunks `x.chunk(2, dim=-1)` **AFTER** `q, k = norm(q), norm(k)`. | **RoPE `base = 10000.0`** ($10\times$ longer max wavelength!) with `arange(0, rot_d)/rot_d`, applied to even/odd pairs **BEFORE** `F.rms_norm(q)`. | With sliding windows of `128–1792` tokens, `base = 1024` concentrates all rotary frequencies in the local `< 1024`-token context window where attention actually operates. |
| **6** | **Token Value Embeddings (`ve`) Init, Un-Normed Mixing (`lambdas=[0.5, 0.5]`), & No `v_0` Conflict** | `ve` initialized with standard normal `N(0, 1)` in `bfloat16` (independent of `wte`). Mixed via unconstrained scalar pair `self.lambdas = [0.5, 0.5]`:<br>• Layers `0,1,2,9,10,11`: `v = lambdas[0]*v + lambdas[1]*ve`<br>• Layers `3,4,5,6,8`: `v = lambdas[0]*v`<br>**No `F.rms_norm(ve)` and no `v_0` mixing!** | `ve` initialized as a **clone of `wte.weight`**, normalized via `F.rms_norm(ve)` on every forward pass (erasing learned token magnitudes!), gated with `0.5*sigmoid(-1.1) = 0.125`, AND simultaneously mixed with Layer-0 `v_0` (`true_layer0_v_residual`) on the same layers! | Normalizing `ve` with `F.rms_norm` prevents `Adam(lr=0.6)` from scaling rare vs frequent token value magnitudes, while cloning `wte` and mixing `v_0` on top of `ve` creates redundant collinear value streams. |
| **7** | **Block `x0` Lambdas (`[1.0, 0.0]`), Pre-Block U-Net Skip (`skip_weights = ones(6)`), & Residual Scale (`1.0`)** | • Pre-block U-Net skip on decoder layers `6..11`: **`x = x + skip_weights[i] * skip.pop()` BEFORE `blocks[6+i]`** (`skip_weights = ones(6)`).<br>• Start of each block: **`x = lambdas[0]*x + lambdas[1]*x0`** (`lambdas = [1.0, 0.0]`).<br>• **`residual_scale = 1.0`** (`x = x + attn; x = x + mlp`). | • Post-block U-Net skip **AFTER `blocks[6+i]`** (so Layer 11's skip bypassed Layer 11's attention & MLP completely and went straight into `lm_head`!).<br>• Vector `0.15 * tanh(highway_gate)`.<br>• `--depth-scaled-residual` set **`residual_scale = 0.7071`**, shrinking all zero-init residual updates by `30%`. | Adding the U-Net skip *after* Layer 11 dumps raw Layer-0 encoder features directly into `x_final` without any decoder MLP processing, while `residual_scale = 0.7071` dampens backbone learning. |
| **8** | **Skipped Attention Layer (`layer_idx == 7`), `0.5 / sqrt(d)` Uniform Init, & Logit Softcap `15.0`** | • Skips attention on **`layer_idx == 7`** (2nd decoder block).<br>• Initializes `qkv_w` and `c_fc` with `uniform_(-bound, bound)` where `std = 0.5 / sqrt(d)` (`gain = 0.5`).<br>• Logit softcapping: `30 * sigmoid(logits / 7.5)` $\equiv$ **`15 * tanh(logits / 15) + 15` (effective cap `15.0`)**. | • Skipped attention on `layer_idx == 6` (1st decoder block right after bottleneck).<br>• Initialized `qkv_proj` and `fc1` with `orthogonal_(gain=1.0)` ($2\times$ larger initial spectral norm).<br>• Used `30 * tanh(logits / 30)` (**effective cap `30.0`**, $2\times$ looser). | Tighter logit softcapping (`15.0` vs `30.0`) prevents overconfident early token predictions and stabilizes large-LR (`lr=0.6`) embedding updates. |
| **9** | **Global Batch Size (`B_tok = 524,288` vs `262k/131k/65k`) & Progressive Long/Short Sliding-Window Schedule (`128 -> 1792`)** | **`B_tok = 524,288` tokens/step** (`1,393` steps = `730.3M` tok, or `1,716` steps = `900M` tok), sequential shard streaming, and progressive block-wise sliding-window schedule **`window_size = next_multiple_of_n(1728 * step / total_steps, n=128)`** with encoder `[long, short, short, short, long, short]` and decoder `[short, long, short, short, short, long]` (`short = long // 2`). | Ran at $B_{\text{tok}} = 65\text{k}$ (`13,725` steps) or $131\text{k}$ (`6,858` steps) or $262\text{k}$ (`3,429` steps) with fixed `window_size = 384` (or `0`) across all layers. | As noted in Muon's official docstring, Newton-Schulz normalizes every update to unit spectral norm regardless of gradient noise variance; large batch size ($B_{\text{tok}} = 262\text{k}$–$524\text{k}$) + progressive window curriculum (`128 -> 1792`) provides low-noise polar updates early and full long-range context late. |

---

## 4. Hypotheses & Quantitative Pass/Fail Predictions

### Hypothesis H1 — Exact Dashboard Process Liveness Resolution (`0` Stale `RUNNING` Badges)
- **Claim**: Updating `collect_snapshot()` in `scripts/monitor_colab_runs.py` so that any run not in `active_poll_keys` (or whose local log file has not been modified within `300` seconds while `ps_alive_for_run` is not `True`) passes `ps_alive=False` into `parse_run_log(...)` will transition all 5 superseded incomplete Wave-7/Wave-8 runs (including `w7_900m_qborn_earlyfix_frame4d_synergy512`) from `"RUNNING"` to `"STOPPED"`, and marking superseded Wave-6/7/8/9/10 exploratory runs as `obsolete: true` in `DEFAULT_OBSOLETE_KEYS` will leave `0` stale `"RUNNING"` runs in `runs_catalog`.
- **Quantitative Pass Criterion**:
  - `sum(1 for r in snap["runs_catalog"] if r["status"] == "RUNNING") == 0` when no remote Colab training process is active, and
  - `card["status"] == "STOPPED"` for `w7_900m_qborn_earlyfix_frame4d_synergy512`.

### Hypothesis H2 — Exact `modded-nanogpt` Architectural & Optimizer Parity (`leaderboard_exact=True`)
- **Claim**: Adding `leaderboard_exact: bool = False` to `ClassicalModdedBlock` and `ClassicalModdedGPT` (`src/qdlblocks/models/classical_modded_gpt.py`), `ema_momentum: bool = False` to `StiefelMuon` (`src/qdlblocks/optim/muon.py`), and `--leaderboard-exact-classical` + `--progressive-window` to `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` reproduces all 9 mathematical primitives of `KellerJordan/modded-nanogpt` (`2025-01-16_Sub3Min.txt`) and achieves strictly lower validation cross-entropy than the prior `ClassicalModdedGPT` configuration on a controlled 60-step FineWeb-shard convergence benchmark at matched token budget.
- **Quantitative Pass Criterion**:
  1. At initialization (`step = 0`), a `ClassicalModdedGPT(leaderboard_exact=True, vocab_size=50304, embed_dim=768, num_heads=6, num_layers=12, hidden_dim=3072)` outputs exact uniform zero logits (`0.0` everywhere via `15.0 * torch.tanh(raw / 15.0)`, which has identical softmax probabilities and identical cross-entropy gradients to `30.0 * torch.sigmoid(raw / 7.5)`) with initial loss $\ln(50304) = 10.82584 \pm 10^{-4}$ (matching `step:0/1393 val_loss:10.8258` in `2025-01-16_Sub3Min.txt`!).
  2. In a controlled 60-step training benchmark on real FineWeb tokens (`examples/29_verify_plan019_leaderboard_exact_classical.py`), the `leaderboard_exact=True` model + optimizer (`Muon(lr=0.05, ema_momentum=True, wd=0.0)` + `Adam(lr_wte=0.6, lr_scalar=0.04, lr_head=0.008, betas=(0.8, 0.95), eps=1e-10)`, `warmup_steps=0`, no grad clipping) improves final validation loss by **$\ge 0.15$ nats** over the Wave-9/10 `ClassicalModdedGPT` configuration at identical step count and token budget.

### Hypothesis H3 — Compiled Long/Short Progressive Window Schedule (`128 -> 1792`) with `T_val = 512` Invariant & `0` Dynamo Recompilations
- **Claim**: Implementing the `modded-nanogpt` per-layer long/short window schedule (`set_window_size(long_w)` updating a 0D GPU tensor buffer `_window_size_buf` and routing encoder `[long, short, short, short, long, short]` and reversed decoder `[short, long, short, short, short, long]` masks) inside `ClassicalModdedGPT` while keeping `evaluate_validation` strictly fixed at `block_size = 512` executes cleanly under `torch.compile` with **zero Dynamo recompilations across window size changes**, zero NaNs, and finite gradients across all parameter groups.
- **Quantitative Pass Criterion**:
  - `100%` of parameters in `ClassicalModdedGPT(leaderboard_exact=True)` receive finite, non-zero gradients by Step 2 (once Step 1 breaks the zero-init symmetry of `out_proj`, `fc2`, and `lm_head`), and `torch.compile` executes across multiple window size updates (`128 -> 256 -> 384 -> 512`) with `0` errors and `0` guard recompilations.

---

## 5. Detailed Technical Specification of Code Changes

### 5.1 Operational Classification (`Build-Before-Launch` vs. `Nice-to-Have`)
- **`Build-Before-Launch` (Implemented Now)**:
  1. Dashboard stale-`RUNNING` bug fix in `scripts/monitor_colab_runs.py` + `DEFAULT_OBSOLETE_KEYS` update.
  2. `ema_momentum: bool = False` (`buf.lerp_(g, 1 - m)` + `g.lerp(buf, m)`) in `StiefelMuon` (`src/qdlblocks/optim/muon.py`).
  3. `leaderboard_exact: bool = False` in `ClassicalModdedBlock` & `ClassicalModdedGPT` (`src/qdlblocks/models/classical_modded_gpt.py`) + 0D-tensor-backed per-layer long/short progressive sliding window support.
  4. `--leaderboard-exact-classical` and `--progressive-window` CLI flags in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` + Model Card parser support in `src/qdlblocks/benchmarks/model_card.py`.
  5. Wave-11 `900M`-token fleet manifest (`configs/wave11_fleet.json`).
- **`Nice-to-Have` (Logged to `scripts/BACKLOG.md`)**:
  - Multi-GPU DDP `NorMuon` + `PolarExpress` variance reduction (`apply_normuon_variance_reduction`) for multi-node H100 clusters.

### 5.2 Component 1: `StiefelMuon` EMA Momentum & Fused `qkv_proj` Slicing (`src/qdlblocks/optim/muon.py`)
1. Add `ema_momentum: bool = False` to `StiefelMuon.__init__(...)` and `defaults`.
2. In `StiefelMuon.step()`:
   ```python
   buf: torch.Tensor = state["momentum_buffer"]
   if group.get("ema_momentum", False):
       buf.lerp_(g, 1.0 - eff_momentum)
       g_in = g.lerp(buf, eff_momentum) if nesterov else buf
   else:
       buf.mul_(eff_momentum).add_(g)
       g_in = g.add(buf, alpha=eff_momentum) if nesterov else buf
   ```
   This preserves 100% backward compatibility with all existing tests when `ema_momentum=False` while giving exact line-by-line parity with `KellerJordan/modded-nanogpt` (`lines 209–210` of `2025-01-16_Sub3Min.txt`) when `ema_momentum=True`.
3. **Fused `qkv_proj` `(3*D, D)` Slicing into 3 Independent `(D, D)` Matrices (Already Present at Lines 183–197 of `src/qdlblocks/optim/muon.py`)**:
   - In `ClassicalModdedBlock`, `self.qkv_proj = nn.Linear(D, 3 * D, bias=False)` has a 2D weight tensor `p` of shape `(3 * D, D)`.
   - In `StiefelMuon.step()` (`lines 183–197`), whenever `p.size(0) == 3 * p.size(1)`, `StiefelMuon` views `p` and `g_in` as `p.view(3, D, D)` and `g_in.view(3, D, D)` so that the $W_Q, W_K, W_V$ slices are orthogonalized as **3 independent square `(D, D)` matrices** via `zeropower_via_newtonschulz5_batched` with `aspect_scale = sqrt(max(1, D / D)) = 1.0`. This is mathematically identical to `self.qkv_w = nn.Parameter(torch.empty(3, D, D))` in `2025-01-16_Sub3Min.txt` (`line 266`).

### 5.3 Component 2: `ClassicalModdedGPT` & `ClassicalModdedBlock` Exact Leaderboard Mode (`src/qdlblocks/models/classical_modded_gpt.py`)
1. **Half-Chunk `LeaderboardRotary` (`base = 1024.0`)**:
   - Add `LeaderboardRotary(nn.Module)` in `src/qdlblocks/models/classical_modded_gpt.py` matching lines 238–255 of `2025-01-16_Sub3Min.txt`:
     ```python
     class LeaderboardRotary(nn.Module):
         """Exact KellerJordan/modded-nanogpt half-truncated Rotary embedding (base=1024, chunk(2, dim=-1))."""

         def __init__(self, head_dim: int, max_seq_len: int = 65536) -> None:
             super().__init__()
             self.head_dim = int(head_dim)
             self.max_seq_len = int(max_seq_len)
             quarter = max(1, self.head_dim // 4)
             angular_freq = (1.0 / 1024.0) ** torch.linspace(0.0, 1.0, steps=quarter, dtype=torch.float32)
             angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(quarter)])
             t = torch.arange(self.max_seq_len, dtype=torch.float32)
             theta = torch.outer(t, angular_freq)
             self.register_buffer("cos_cached", theta.cos(), persistent=False)
             self.register_buffer("sin_cached", theta.sin(), persistent=False)

         def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
             return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
     ```
   - And helper `apply_leaderboard_rotary(x_bhtd: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor`:
     Given `x_bhtd` of shape `(B, H, T, D)` and `cos, sin` of shape `(T, D // 2)`:
     ```python
     c = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
     s = sin.unsqueeze(0).unsqueeze(0)
     x1, x2 = x_bhtd.float().chunk(2, dim=-1)
     y1 = x1 * c + x2 * s
     y2 = x1 * (-s) + x2 * c
     return torch.cat((y1, y2), dim=-1).type_as(x_bhtd)
     ```
2. **`ClassicalModdedBlock(..., leaderboard_exact: bool = False, attn_scale: float | None = None)`**:
   - When `leaderboard_exact=True`:
     - Weight initialization (`lines 230–233, 262–270, 293–295` of `2025-01-16_Sub3Min.txt`):
       - `std = 0.5 * (self.embed_dim ** -0.5)` and `bound = (3.0 ** 0.5) * std`.
       - `self.qkv_proj.weight` (or `q_proj, k_proj, v_proj`) and `self.fc1.weight` are initialized via `.uniform_(-bound, bound)`.
       - `self.out_proj.weight` (when present) and `self.fc2.weight` are initialized via `.zero_()`.
     - Block `x_0` mixing (`lines 309, 312`):
       - Allocates `self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))` (instead of `highway_gate`, setting `self.highway_gate = None`) and computes at the start of `forward`:
         `if self.block_lambdas is not None and x_0 is not None: x = self.block_lambdas[0].to(x.dtype) * x + self.block_lambdas[1].to(x.dtype) * x_0`
     - Value embedding mixing (`lines 267, 279–282`):
       - Allocates `self.sa_lambdas = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))` on non-skipped attention layers (setting `self.v_mix_gate = None` and `self.ve_mix_logit = None`) and computes on `(B, H, T, head_dim)` immediately after `q, k, v = self._project_qkv(h)`:
         - If `ve is not None`:
           `ve_heads = ve.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous().to(v.dtype)`
           `v = self.sa_lambdas[0].to(v.dtype) * v + self.sa_lambdas[1].to(v.dtype) * ve_heads` (NO `F.rms_norm(ve)` and NO `v_0` mixing!).
         - Else:
           `v = self.sa_lambdas[0].to(v.dtype) * v`
     - QK-Norm and Rotary order (`lines 283–284`):
       - Applies `q = F.rms_norm(q, (self.head_dim,), eps=self.eps).type_as(q)` and `k = F.rms_norm(k, (self.head_dim,), eps=self.eps).type_as(k)` **BEFORE** `apply_leaderboard_rotary(q, cos, sin)` and `apply_leaderboard_rotary(k, cos, sin)`.
     - Attention scale & mask (`line 273, 285`):
       - Uses `self.attn_scale = 0.12 if attn_scale is None else float(attn_scale)`, passed as `scale=self.attn_scale` to `F.scaled_dot_product_attention(..., scale=self.attn_scale)`.
       - Accepts an optional precomputed 4D boolean mask `attn_mask_4d: torch.Tensor | None = None` of shape `(1, 1, T, T)` or `(B, 1, T, T)` so `forward_features` can pass precomputed long/short sliding-window masks without Python-int Dynamo guard recompilations.
     - Residual additions (`lines 314–315`):
       - `x = x + self.out_proj(attn_out)` and `x = x + mlp_out` (`residual_scale = 1.0`).
3. **`ClassicalModdedGPT(..., leaderboard_exact: bool = False)`**:
   - **Arbitrary-Depth (`num_layers`) Rules & Defaults when `leaderboard_exact=True`**:
     - Forces:
       - `use_bigram_smear = False`
       - `use_fused_qkv = True`
       - `unet_skip_mode = "scalar"`
       - `num_value_embeddings = 3` (if `num_value_embeddings == 0`, else uses the passed value)
       - `skip_attn_layers = (7,)` when `num_layers >= 8` and `skip_attn_layers == ()` (for smaller unit-test models with `num_layers < 8` and `skip_attn_layers == ()`, no layer is skipped unless explicitly specified via `skip_attn_layers`).
       - `logit_softcap = 15.0`
       - `zero_init_residual_and_head = True`
       - `true_layer0_v_residual = False`
       - `depth_scaled_residual = False` (`residual_scale = 1.0`)
   - **Exact Value Embedding Layer-Index Mapping (`_ve_bank_index(layer_idx)`)**:
     - Let $k = \min(\text{num\_value\_embeddings}, \text{num\_layers} // 2)$.
     - For encoder layers $0 \le \text{layer\_idx} < k$, `_ve_bank_index` returns `layer_idx` (`0, 1, 2` for $L=12, k=3$).
     - For the last $k$ decoder layers $\text{num\_layers} - k \le \text{layer\_idx} < \text{num\_layers}$, `_ve_bank_index` returns `layer_idx - (num_layers - k)` (`0, 1, 2` for layers `9, 10, 11` when $L=12, k=3$).
     - For all middle layers $k \le \text{layer\_idx} < \text{num\_layers} - k$ (`3, 4, 5, 6, 7, 8` when $L=12, k=3$), `_ve_bank_index` returns `None`.
     - This matches line 326 of `2025-01-16_Sub3Min.txt` (`ve = [ve[0], ve[1], ve[2], None, None, None, None, None, None, ve[0], ve[1], ve[2]]`) for $L=12$ while generalizing cleanly to any $L \ge 2$ in unit tests!
   - **Embedding Dtype & Initialization (`wte` and `value_embeds`)**:
     - When `leaderboard_exact=True`, `self.wte` and `self.value_embeds` keep their standard PyTorch `nn.Embedding` `N(0, 1)` initialization (`float32` by default in `__init__` so CPU/CUDA `float32` unit tests work out of the box without dtype mismatches, and `value_embeds` is **not** overwritten with a clone of `wte.weight`!).
     - In `run_cuda_benchmark` (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`), after moving `model.to(device)`, when `leaderboard_exact_classical=True`, `model.wte.bfloat16()` and `model.value_embeds.bfloat16()` are called (matching lines 512–514 of `2025-01-16_Sub3Min.txt`), and inside `forward_features`, `wte_x = self.wte(idx)` and `ve_l = self._get_value_embed(layer_idx, idx)` are cast to the active computation dtype (`wte_x.to(torch.get_autocast_dtype("cuda"))` when CUDA autocast is enabled, or `x.dtype`).
   - **Pre-Block U-Net `skip_weights`**:
     - When `leaderboard_exact=True` and `self.use_unet_interferometry and self.num_layers >= 2`, allocates `self.skip_weights = nn.Parameter(torch.ones(self.num_layers // 2, dtype=torch.float32))` (`6` scalar weights initialized to `1.0`).
     - In `forward_features`:
       - Encoder half (`layer_idx < n_half`): runs `x = block(x, ...)` and appends `encoder_states.append(x)`.
       - Decoder half (`layer_idx >= n_half`, `dec_i = layer_idx - n_half`): adds `x = x + self.skip_weights[dec_i].to(x.dtype) * encoder_states.pop()` **BEFORE** calling `x = block(x, ...)` (`line 413` of `2025-01-16_Sub3Min.txt`).
   - **Exact Logit Softcapping Formula (`15.0 * torch.tanh(raw / 15.0)`)**:
     - Both `ClassicalModdedGPT.compute_logits(x_final)` and `direct_chunked_softcap_linear_ce(x_final, lm_head.weight, targets, logit_softcap=model.logit_softcap)` compute:
       $$\text{logits} = 15.0 \cdot \tanh(\text{raw} / 15.0)$$
       when `self.logit_softcap == 15.0`. Because $30 \sigma(z / 7.5) = 15 \tanh(z / 15) + 15$ and softmax / cross-entropy is shift-invariant to adding $+15$, `15.0 * torch.tanh(raw / 15.0)` produces **exact `0.0` logits at Step 0** (better `bfloat16` dynamic range near zero than centering at `+15.0`), identical probabilities $p_v$, identical cross-entropy loss $\ln(50304) = 10.82584$, and identical backward gradients to `30 * torch.sigmoid(raw / 7.5)`!
   - **Zero-Recompilation Progressive Long/Short Sliding-Window Schedule (`set_window_size`)**:
     - `ClassicalModdedGPT` registers a non-persistent 0D `int32` tensor buffer `self.register_buffer("_window_size_buf", torch.tensor(int(self.window_size), dtype=torch.int32), persistent=False)` and provides `set_window_size(window_size: int) -> None` which updates `self.window_size = int(window_size)` and `self._window_size_buf.fill_(int(window_size))`.
     - When `self.window_size > 0` (or `self.doc_mask_attn` is `True`) and `self.leaderboard_exact` is `True`:
       - Inside `forward_features(idx)`, `idx_t = torch.arange(T, device=x.device)` and `diff = idx_t[:, None] - idx_t[None, :]`.
       - Using the 0D tensor `w_long = self._window_size_buf` and `w_short = torch.clamp_min(w_long // 2, 128)`, we construct two 4D boolean masks on GPU without branching on Python ints:
         - `base_mask = (diff >= 0).view(1, 1, T, T)` (intersected with `doc_ids[:, None, :, None] == doc_ids[:, None, None, :]` when `self.doc_mask_attn` is `True`),
         - `long_mask_4d = base_mask & (diff < w_long).view(1, 1, T, T)`,
         - `short_mask_4d = base_mask & (diff < w_short).view(1, 1, T, T)`.
       - For any `n_half = self.num_layers // 2`, encoder layer `i in range(n_half)` uses `long_mask_4d` if `i in (0, 4)` (or `i == 0` when `n_half < 5`) and `short_mask_4d` otherwise (producing `[long, short, short, short, long, short]` for $n_{\text{half}} = 6$), and decoder layer `dec_i in range(self.num_layers - n_half)` reverses the encoder mask list (`[short, long, short, short, short, long]` for $L=12$, matching lines 406 & 411 of `2025-01-16_Sub3Min.txt`!).
       - Because `_window_size_buf` is a tensor buffer mutated in-place via `.fill_()`, `torch.compile` traces the mask construction once and executes across all 14 progressive window steps (`128 -> 1792`) with **0 Dynamo recompilations**!

### 5.4 Component 3: `--leaderboard-exact-classical` & `--progressive-window` in `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` & `src/qdlblocks/benchmarks/model_card.py`
1. Add CLI flags to `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` (and `_build_cli_arg_parser()` in `src/qdlblocks/benchmarks/model_card.py`):
   - `--leaderboard-exact-classical` (`action="store_true"`):
     - Constructs `ClassicalModdedGPT(..., leaderboard_exact=True, logit_softcap=15.0)`.
     - Casts `model.wte.bfloat16()` and (when present) `model.value_embeds.bfloat16()` on CUDA.
     - Configures `StiefelMuon` with:
       - `lr = muon_lr` (defaulting to `0.05` when `--leaderboard-exact-classical` is set and `--muon-lr` was not explicitly overridden from `0.025`),
       - `momentum = 0.95`, `momentum_warmup_steps = max(300, muon_momentum_warmup)`, `momentum_init = 0.85`,
       - `nesterov = True`, `aspect_scale = True`, `ema_momentum = True`,
       - `weight_decay = 0.0`, `decay_mode = "none"`, `stiefel_penalty = 0.0`, `batched_ns = True`.
     - Configures `opt_adamw` as `torch.optim.Adam` with **`betas=(0.8, 0.95), eps=1e-10, fused=True`**:
       - `head_group` (`lm_head`): `lr = 0.008, base_lr = 0.008, weight_decay = 0.0`,
       - `wte_group` (`wte` + `value_embeds`): `lr = 0.60, base_lr = 0.60, weight_decay = 0.0`,
       - `scalar_group` (`skip_weights`, `block_lambdas`, `sa_lambdas`, etc.): `lr = 0.04, base_lr = 0.04, weight_decay = 0.0`.
     - Sets `warmup_steps = 0` (matching `get_lr` in `2025-01-16_Sub3Min.txt` lines 533–537!) and skips `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
     - Uses sequential contiguous shard indexing (`batch_order = np.arange(steps_per_shard)`) so document boundaries and token order match `distributed_data_generator` in `2025-01-16_Sub3Min.txt`.
   - `--progressive-window` (`action="store_true"`):
     - Initializes `model` with `window_size = 128` so `_window_size_buf` is active from Step 0, and on each training step `global_step` updates `model.set_window_size(min(1792, ((int(1728 * (global_step - 1) / max(1, total_steps)) // 128) + 1) * 128))`, while keeping `evaluate_validation` strictly fixed at `block_size = 512`.

### 5.5 Component 4: Wave-11 `900M`-Token Fleet Manifest (`configs/wave11_fleet.json`)
We will queue and launch **5 `900M`-token experiments** across our 5 Colab GPUs (`3× NVIDIA RTX PRO 6000 Blackwell 96GB` + `2× NVIDIA A100-SXM4-80GB`):
1. **`w11_900m_classical_leaderboard_exact_b524k_1716st` (`gpu_g4_w6_1`, G4 96GB)**:
   - **Exact 1:1 `modded-nanogpt` Leaderboard Architecture & Optimizer at $B_{\text{tok}} = 524{,}288$ (`1,716` steps = `900M` tokens, Full Causal FlashAttention-2 `window_size=0`)**:
   - `L=12, d=768, H=6 (head_dim=128), d_ff=3072`, `--leaderboard-exact-classical`, `Muon(lr=0.05, ema_momentum=True, wd=0.0, warmup=300)`, `Adam(lr_wte=0.6, lr_scalar=0.04, lr_head=0.008, betas=(0.8, 0.95), eps=1e-10)`, `WSD(warmup=0, cooldown=0.40, min_lr_ratio=0.10)`, `T=1024, B=64, G_acc=8` ($B_{\text{tok}} = 524{,}288$).
   - **Goal**: Test exact single-GPU `modded-nanogpt` architectural and optimizer parity at the official `524k` batch size and `1,716` steps with hardware-fast causal SDPA toward `<= 3.280` Val CE!
2. **`w11_900m_classical_leaderboard_exact_b262k_3429st` (`gpu_g4_w6_2`, G4 96GB)**:
   - **Exact `modded-nanogpt` Leaderboard Architecture & Optimizer at $B_{\text{tok}} = 262{,}144$ (`3,429` steps = `900M` tokens, Full Causal FlashAttention-2 `window_size=0`)**:
   - Same exact `--leaderboard-exact-classical` architecture and optimizer (`H=6, head_dim=128, attn_scale=0.12, RoPE base=1024, un-normed VE lr=0.6, softcap=15, pre-block U-Net skip, skip_attn=7, ema_momentum=True, wd=0.0`), run at `B_tok = 262,144` (`3,429` steps, `T=1024, B=64, G_acc=4`, `muon_lr=0.04`) — the exact step budget that gave our previous best (`3.4280` in `w9_a100_1`).
   - **Goal**: Direct apples-to-apples comparison isolating the 9-point `--leaderboard-exact-classical` upgrade against `w9_900m_classical_speedrun_v2_pr360` (`3.4280`) at identical `3,429` steps!
3. **`w11_900m_classical_leaderboard_exact_progwin_docmask` (`gpu_a100_w6_1`, A100 80GB)**:
   - **Exact `modded-nanogpt` Leaderboard Architecture + Progressive Long/Short Sliding Window (`128 -> 1792`) + Document Masking (`<|endoftext|>`) at $B_{\text{tok}} = 524{,}288$ (`1,716` steps)**:
   - Adds `--progressive-window --doc-mask-attn` to `w11_900m_classical_leaderboard_exact_b524k_1716st` to test whether the progressive `128 -> 1792` long/short window curriculum + document boundary masking improves `T_val = 512` generalization even further over Run 1.
4. **`w11_900m_qborn_speedrun_v4_deep_L12_asym1024_frame4d` (`gpu_g4_w6_3`, G4 96GB)**:
   - **QuantumBornGPT Next-Gen Flagship Unification (`L=12` Deep + `Asym-1024` + `4d-Frame 4096x2`)**:
   - In Wave 10, `w10_g4_3` (`L=8 + Asym-1024 + 4d-Frame`) hit **`3.6926` Val CE** (#1 All-Time Quantum Record), while `w10_a100_1` (`L=12 + Asym-1024 + 3d-Frame + SkipAttn(6)`) hit **`3.7077` Val CE** (#2 All-Time Quantum Record).
   - On the 96GB G4 Blackwell GPU, we now unify **both** winners: **`L=12` depth + `d=1024` + `seed_dim=1024` (`Asym-1024`) + `4d-Frame` ($4096\times 2$ Overcomplete Two-Beam) + `SkipAttn(7)` + `VE=3` + `FlashPurifiedBorn(Win=384, DocMask)`**, targeting the **`< 3.65` Val CE** frontier!
5. **`w11_900m_qborn_speedrun_v4_L8_leaderboard_hybrids` (`gpu_a100_w6_2`, A100 80GB)**:
   - **QuantumBornGPT `L=8` + Leaderboard-Exact Hybrids (`H=8, head_dim=128`, `ZeroHalfRoPE`, `Muon LR=0.035`, `WSD cooldown=0.40, min_lr=0.10`)**:
   - Transfers the newly discovered `modded-nanogpt` insights (`head_dim = 128` via `num_heads = 8` on $d = 1024$, higher `muon_lr = 0.035`, and `40%` cooldown to `0.10`) onto our #1 `w10_g4_3` QuantumBornGPT architecture (`3.6926` Val CE).

---

## 6. Verification & Testing Plan

1. **Unit Test Suite (`tests/test_hypothesis_019_leaderboard_exact_classical.py`)**:
   - `test_stiefel_muon_ema_momentum_parity`: Verifies that `StiefelMuon(..., ema_momentum=True, weight_decay=0.0)` matches the exact `buf.lerp_(g, 1 - m); g = g.lerp_(buf, m)` + `zeropower_via_newtonschulz5` update of official `modded-nanogpt` to `< 1e-6` max diff across 10 steps (including slicing fused `(3*D, D)` `qkv_proj` into 3 independent `(D, D)` matrices).
   - `test_classical_modded_gpt_leaderboard_exact_architecture`: Verifies that `ClassicalModdedGPT(leaderboard_exact=True, vocab_size=50304, embed_dim=768, num_heads=6, num_layers=12, hidden_dim=3072)` has:
     - `head_dim == 128`, `skip_attn_layers == (7,)`, `logit_softcap == 15.0`,
     - `LeaderboardRotary` with `base == 1024.0` and half-chunk rotation,
     - Un-normed `ve` mixing via `sa_lambdas = [0.5, 0.5]`, pre-block `skip_weights = ones(6)`, and `block_lambdas = [1.0, 0.0]`,
     - Exact initial uniform zero logits (`0.0` via `15.0 * torch.tanh(raw / 15.0)`) with loss $\ln(50304) = 10.82584 \pm 10^{-4}$ at Step 0, and finite non-zero gradients on 100% of parameters by Step 2.
   - `test_progressive_window_zero_recompiles`: Verifies that a `torch.compile`-wrapped `ClassicalModdedGPT(leaderboard_exact=True, window_size=128, doc_mask_attn=True)` executes across multiple `set_window_size(w)` updates (`128 -> 256 -> 384 -> 512`) without triggering Dynamo recompilation (`torch._dynamo.utils.counters["frames"]["ok"]` increases by `0` after the first step).
   - `test_monitor_stale_running_fix_and_wave11_cards`: Verifies that `collect_snapshot(skip_remote_poll=True)` marks `w7_900m_qborn_earlyfix_frame4d_synergy512` and all other non-active incomplete runs as `"STOPPED"` (`0` false `"RUNNING"` runs) and validates all 5 `configs/wave11_fleet.json` Model Cards.
2. **Controlled Empirical Verification Script (`examples/29_verify_plan019_leaderboard_exact_classical.py`)**:
   - Runs the controlled H1, H2, and H3 experiments and writes `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_019.json`.
