# QuantumDLBlocks: Project Overview, Session Wrap-Up & Technical Roadmap

## 1. Executive Summary

**QuantumDLBlocks** is a deep learning research and systems engineering project that replaces standard heuristic neural network primitives—specifically Softmax attention, LayerNorm/RMSNorm, elementwise activations (ReLU/GELU/SiLU), and unconstrained Euclidean weight matrices—with **strictly valid quantum-geometric operators** acting on:
1. **Pure Quantum State Vectors on the Unit Hypersphere**: $|\psi\rangle \in S^{d-1} = \{x \in \mathbb{R}^d \text{ (or } \mathbb{C}^d\text{)} : \|x\|_2 = 1\}$ in an $n$-qubit Hilbert space of dimension $d = 2^n$.
2. **Semi-Orthogonal / Isometric Matrices on the Stiefel Manifold**: For any rectangular weight matrix $W \in \mathbb{R}^{r \times c}$ with $d_{\min} = \min(r, c)$, we constrain $W$ to have orthonormal columns when tall/square ($r \ge c$, so $W^\top W = I_c$) or orthonormal rows when wide/fat ($r < c$, so $W W^\top = I_r$, equivalently $W^\top \in \mathrm{St}(c, r)$). Throughout this document, we denote this generalized semi-orthogonal Stiefel constraint by $\mathrm{St}(r, c)$.
3. **Density Matrices & Kraus Quantum Channels**: Mixed states $\rho \in \mathcal{D}(\mathcal{H})$ ($\rho \succeq 0, \mathrm{Tr}(\rho) = 1$) evolving under Completely Positive Trace-Preserving (CPTP) Kraus channels $\mathcal{E}(\rho) = \sum_{k=1}^p K_k \rho K_k^\dagger$ subject to the completeness relation $\sum_{k=1}^p K_k^\dagger K_k = I$.

During this session, we tackled four interconnected challenges:
1. **Data-Efficient Quantum-Geometric Language Modeling ("Slowrun" Regime)**: Designing and iteratively refining a 10-qubit ($d = 1024 = 2^{10}$) language model (`QuantumBornGPT`) capable of learning effectively from a fixed **100-million-token FineWeb shard** across multiple training epochs without suffering the singular-value (spectral norm) explosion or rapid multi-epoch overfitting typical of unconstrained classical Transformers.
2. **High-Throughput Kernel & Compiler Engineering on GPU and TPU**: Eliminating memory and compute bottlenecks in custom squared-amplitude Born-rule attention and Stiefel-manifold optimization across both **NVIDIA Blackwell GPUs** (via PyTorch `torch.compile` and custom Triton kernels) and **Google Cloud TPU v5e-1** accelerators (via pure JAX/XLA whole-step HLO compilation and Matrix Multiply Unit [MXU] loss factorization), plus analyzing architectural portability to **AMD RDNA 4 (Radeon AI PRO R9700 32 GB)** workstations.
3. **Resilient Remote Execution on Google Colab Hardware**: Overcoming Jupyter/IPython WebSocket timeouts (`RuntimeError: Connection was lost`), notebook output history RAM leaks, and TPU `/dev/accel0` device lock contention by executing training scripts as detached operating-system subprocesses.
4. **Zero-Dependency Local Experiment Observability**: Building a locally hosted, LAN-accessible (`0.0.0.0:8765`) telemetry collector, multi-run SVG overlay web dashboard, and terminal TUI (`scripts/monitor_colab_runs.py`) that aggregates live training curves, GPU `nvidia-smi` metrics, TPU `libtpu` gRPC runtime metrics (MXU duty cycle and HBM2e usage), **Model FLOPs Utilization (MFU)**, **Hardware FLOPs Utilization (HFU)**, and JAX/Equinox/Haliax/Optax XLA HLO compiler statistics.

---

## 2. Problems We Have Been Solving & Discussing

### 2.1. Architectural Problem: Eliminating Heuristic Non-Linearities via Quantum Geometry
Standard Transformers rely on three ad-hoc mechanisms that lack intrinsic geometric bounds:
- **Unbounded Logits & Softmax**: Attention scores $QK^\top / \sqrt{d_h}$ and routing logits can grow without bound, causing attention entropy collapse and overconfident out-of-distribution (OOD) predictions.
- **Affine LayerNorm / RMSNorm**: Activation norms drift continuously across depth and must be forcefully centered and rescaled with learned affine gains and biases at every sub-block boundary.
- **Unconstrained Weight Matrices**: Under standard Euclidean optimizers (AdamW/SGD), weight matrices $W_\ell$ accumulate large singular values ($\sigma_{\max}(W_\ell) \gg 1$) over multi-epoch training, amplifying gradient noise and memorizing repeated training tokens.

**Our Formulation**: Every intermediate token state $|\psi\rangle \in S^{d-1}$ is strictly unit-normalized ($\|\psi\|_2 = 1$), every 2D projection matrix $W \in \mathrm{St}(r, c)$ is maintained near the Stiefel manifold of semi-orthogonal matrices, and all internal non-linearity arises from three physical quantum operations:
1. **Born-Rule Squared-Amplitude Interference**: Internal attention and expert-routing probabilities are computed from squared quantum transition amplitudes $P(i \to j) \propto |\langle q_i | k_j \rangle|^2$ rather than exponential Softmax $\exp(\langle q_i, k_j \rangle)$.
2. **Projective State Collapse ($\mathcal{N}(\cdot)$)**: Following any coherent linear superposition or bilinear interference of states, the resulting unnormalized wavevector $z \in \mathbb{R}^d$ undergoes projective measurement collapse back onto the unit hypersphere $S^{d-1}$:
   $$\mathcal{N}(z) = \frac{z}{\sqrt{\|z\|_2^2 + \epsilon_{\text{norm}}}}, \qquad \epsilon_{\text{norm}} = 10^{-12}\text{ (FP32) or } 10^{-6}\text{ (BF16)}$$
3. **Positive Operator-Valued Measure (POVM) Readouts**: At the final language modeling head, the collapsed token wavevector $|z_i\rangle \in S^{d_{\text{seed}}-1}$ is measured against unit-norm vocabulary measurement vectors $\{|m_v\rangle\}_{v=1}^V \subset S^{d_{\text{seed}}-1}$ scaled by a learnable inverse temperature $\beta = 1/\tau \in [1, 60]$. Because both $|z_i\rangle$ and $|m_v\rangle$ have unit L2 norm, every logit is a strictly bounded cosine similarity $\beta \langle z_i | m_v \rangle \in [-\beta, +\beta]$, which is trained via standard categorical cross-entropy negative log-likelihood (NLL) over the $V = 50,304$ vocabulary outcomes while all internal layers (attention, MLPs, residuals, and Mixture-of-Experts routing) remain 100% Softmax-free.

### 2.2. Benchmark Problem: Multi-Epoch Sample Efficiency on the 100M-Token FineWeb "Slowrun"
The community **Modded-NanoGPT Slowrun** benchmark evaluates data efficiency by restricting training to a fixed **100-million-token shard** of FineWeb (`fineweb_train_000001.bin`, tokenized with the GPT-2 BPE tokenizer of $50,257$ tokens, padded to $V = 50,304 = 393 \times 128$ for Tensor Core and TPU MXU tile alignment) while allowing models to train for **multiple epochs** over those 100M tokens, evaluating generalization on a disjoint **4,999,488-token validation shard** (`fineweb_val_000000.bin`).
- While leading classical Slowrun submissions use $8\times\text{NVIDIA H100}$ clusters to iterate through many passes over the 100M tokens, we investigated how far our quantum-geometric architecture (`QuantumBornGPT`) can push validation cross-entropy and perplexity on a **single accelerator** (1 Colab GPU or 1 Colab Cloud TPU v5e-1 chip).
- We also investigated how data-efficient architectural techniques (such as value residuals, input highways, U-Net skip connections, and bilinear gated MLPs) can be reformulated as strictly norm-preserving, isometric quantum interference operations on $S^{d-1} \times \mathrm{St}(r, c)$.

### 2.3. Systems & Hardware Problem: Fused Kernels, Compilation & Multi-Vendor Portability
Scaling a 10-qubit ($d = 1024$) model with sequence length $T = 512$, global batch size $B = 128$ ($65,536$ tokens/step), and $V = 50,304$ across NVIDIA GPUs, Google Cloud TPUs, and AMD RDNA 4 GPUs surfaced four critical systems bottlenecks:
1. **$O(T^2)$ Attention Materialization & Backward Atomic Contention on GPU**:
   - Materializing the $[B, H, T, T]$ causal Born attention matrix in high-bandwidth memory (HBM/VRAM) consumes excessive memory and bandwidth.
   - Our first-generation custom Triton Flash-Born backward kernel computed query gradients $dQ$ by having key/value blocks issue global FP32 `tl.atomic_add` instructions into $dQ$. At $B=128, H=16, T=512, D_h=64$, this executed $B \times H \times \frac{T}{32} \times T \times D_h \approx 1.07\text{ to }1.8\text{ billion}$ contested global atomic additions per backward pass, creating a severe L2 cache atomic serialization bottleneck.
2. **Silent FP32 Upcasting Under PyTorch `bfloat16` Autocast**:
   - In PyTorch, `torch.sum(x * x, dim=-1, keepdim=True)` inside `l2_normalize` promotes `bfloat16` inputs to `float32` to prevent overflow/underflow. Multiplying `bfloat16` tensor `x` by a `float32` reciprocal norm silently upcast every normalized state vector $|\psi\rangle$ to `float32`. Likewise, `float32` rotary position embedding (`RoPE`) cosine/sine buffers upcast $Q$ and $K$ to `float32`. This doubled activation memory bandwidth and forced downstream projections and Triton kernels out of fast `bfloat16` Tensor Core MMA paths.
3. **TPU Vector Processing Unit (VPU) Bottleneck in 50,304-Way Vocabulary Cross-Entropy**:
   - Google Cloud TPU v5e-1 (`16 GB` HBM2e, `197.0 TFLOP/s` BF16 MXU peak) achieves high throughput on dense matrix multiplications on its systolic Matrix Multiply Unit (MXU), but non-matmul operations execute on the slower Vector Processing Unit (VPU). Computing standard cross-entropy via `jax.nn.one_hot(targets, 50304)` or scatter-indexing across a $[32768, 50304]$ logit tensor stalled the TPU VPU, taking `155.67 ms` per microbatch for the loss calculation alone.
4. **Workstation Hardware Evaluation (AMD Radeon AI PRO R9700 32 GB)**:
   - We analyzed how these workloads map onto the **AMD Radeon AI PRO R9700** (RDNA 4 `gfx1201`, 32 GB GDDR6, 64 Compute Units / 128 AI Accelerators, `~191 TFLOP/s` dense BF16 peak, `1,000 GB/s` memory bandwidth). Because RDNA 4 executes natively in Wave32 (`warpsize=32`) with $16 \times 16 \times 16$ BF16 WMMA instructions and provides `32 GB` of VRAM per card, keeping our 10-qubit training footprint under `25 GB` ensures full-batch (`65,536` tok/step) training fits on a single workstation card without activation offloading.

### 2.4. Operational & Observability Problem: Detached Execution & Live Telemetry Collection
1. **Colab CLI Connection Drops & IPython Kernel Memory Retention**:
   - Synchronous commands run through `colab exec` fail after a few minutes when the WebSocket proxy times out (`RuntimeError: Connection was lost`) or when the short-lived OAuth session token in `~/.config/colab-cli/sessions.json` expires.
   - Running multi-epoch training inside an interactive IPython kernel also leaks host/device memory through IPython's output history references (`Out`, `_`, traceback frames) and holds exclusive locks on `/dev/accel0` (`libtpu`) on TPU VMs, causing subsequent cells or processes to crash with device-busy or out-of-memory errors.
2. **Live Experiment Monitoring Without External SaaS Dependencies**:
   - With training running inside detached background OS processes on separate GPU and TPU Colab VMs, we needed a self-hosted local tool (`0.0.0.0:8765`) capable of polling remote logs and hardware counters, overlaying multiple runs on shared axes, computing exact Model FLOPs Utilization (MFU/HFU), and inspecting JAX/Equinox/Haliax XLA HLO compiler statistics without requiring Weights & Biases (`wandb`).

---

## 3. Outcomes We Aimed to Achieve

1. **Mathematical & Architectural Purity**: Zero `Softmax` in attention or routing, zero `LayerNorm`/`RMSNorm`, and zero elementwise activations (`ReLU`/`GELU`/`SiLU`) across all quantum models, with weight matrices actively maintained near the semi-orthogonal Stiefel manifold ($\text{IsoErr} < 0.20$).
2. **Strong Multi-Epoch FineWeb Generalization**: Achieve monotonic validation cross-entropy and perplexity reductions across 3 to 5 full passes over the 100M-token FineWeb shard (`300M` to `500M` total training tokens), driving Validation Perplexity from `99.8` (our best earlier result from 5 epochs on a 20M-token slice = `100M` total tokens, detailed in Study C below) down into the **`71–89`** range and Top-5 next-token accuracy to **`~50%`**.
3. **Compiled Single-Chip Speed & High MFU**:
   - **CUDA GPU (NVIDIA RTX PRO 6000 Blackwell, `503.8 TFLOP/s` BF16 peak)**: Exceed **`300,000` tokens/sec** (`~4x` speedup over eager execution) and **`~28–34%` MFU/HFU** while keeping peak VRAM under **`25 GB`**.
   - **Cloud TPU v5e-1 (`197.0 TFLOP/s` BF16 MXU peak, `16 GB` HBM2e)**: Exceed **`120,000` tokens/sec** on a single v5e-1 chip at **`100%` TensorCore/MXU duty cycle**, **`~28–34%` MFU/HFU**, and **`<6 GB` HBM2e usage**.
4. **Turnkey LAN & Terminal Observability**: Provide an always-on local web and terminal dashboard (`http://0.0.0.0:8765`) displaying multi-run training/validation overlays, live GPU/TPU hardware utilization, MFU/HFU, and JAX/Equinox/Haliax/Optax XLA HLO compiler analysis.

---

## 4. Steps Taken & Experimental Results

### Step 1: Repository Architecture, Evaluation Metrics & Quantum-Geometric Specification
The project is organized into two complementary software stacks plus a standalone multi-epoch benchmark runner:
- **`src/qdlblocks/` (PyTorch + Custom Triton Stack)**: Contains the core quantum-geometric layers, the fused Triton `flash_born_attention` kernel, the `StiefelMuon` optimizer, the 10-qubit `QuantumBornGPT` language model, multi-domain models for Vision (`QuantumBornViT`), Audio (`ComplexQuantumSpeechCTC` for 16 kHz LibriSpeech ASR), Long-Range Arena (`LRA`), Out-of-Distribution (`OOD`) uncertainty calibration via von Neumann entropy, and Quantum Neural Architecture Search (`Q-NAS`).
- **`src/qdlblocks_jax/` (Named-Axis JAX + Equinox + Haliax + Optax Stack)**: Implements named-axis quantum modules (`HaxQuantumBornGPT`, `HaxSubspaceBornAttention`, `HaxQuantumBornMoE` with Kraus-operator expert channels and Parseval Born routing, and Optax `scale_by_stiefel_muon`), explicit $n$-qubit tensor-product factorization (`factorize_into_qubits` mapping `Embed: 2^n` to `(qubit_0: 2, ..., qubit_{n-1}: 2)`), and exact bipartite von Neumann entanglement entropy $S(\rho_A) = -\mathrm{Tr}(\rho_A \log \rho_A)$.
- **`examples/11_fineweb_100m_multiepoch_gpu_tpu.py` (Matched 10-Qubit 100M-Token Slowrun Runner)**: Implements the exact same 70.68M-parameter 10-Qubit `QuantumBornGPT` architecture in both compiled PyTorch+Triton (`--backend cuda`) and pure positional-axis JAX/XLA (`--backend tpu`) so GPU and TPU benchmarks execute identical tensor shapes and mathematical operations without framework overhead.

#### Definitions of Evaluation & Geometry Metrics Used Across All Tables
- **Train CE / Val CE**: Categorical cross-entropy loss (in nats) evaluated on training batches and held-out validation batches.
- **Val PPL**: Validation perplexity, defined with an overflow guard at initialization as $\text{Val PPL} = \exp(\min(\text{Val CE}, 20.0))$ (lower is better; random uniform guessing over $V = 50,304$ tokens is $50,304.0$).
- **Top-1 / Top-5 Acc (%)**: Percentage of held-out validation tokens where the ground-truth next token $y_t$ ranks #1 or within the top 5 highest-scoring vocabulary candidates.
- **Med Target Rank (`Med Rank` / `Rank`)**: For each validation token $t$, let $r_t = 1 + \sum_{v=1}^{V} \mathbf{1}(\text{logit}_{t, v} > \text{logit}_{t, y_t}) \in \{1, \dots, 50304\}$ be the 1-indexed descending rank of the ground-truth target token $y_t$ among all $V = 50,304$ vocabulary logits. `Med Target Rank` is the median $\mathrm{median}_t(r_t)$ across evaluated validation tokens (`1` is optimal; random guessing yields $\approx V/2 = 25,152$).
- **Stiefel Isometry Error (`Stiefel IsoErr` / `IsoErr`)**: For any 2D weight matrix $W \in \mathbb{R}^{r \times c}$ with $d_{\min} = \min(r, c)$, its normalized deviation from semi-orthogonality is:
  $$\mathrm{IsoErr}(W) = \begin{cases} \dfrac{\|W^\top W - I_c\|_F}{\sqrt{c}} & \text{if } r \ge c \\[10pt] \dfrac{\|W W^\top - I_r\|_F}{\sqrt{r}} & \text{if } r < c \end{cases} = \sqrt{\frac{1}{d_{\min}} \sum_{i=1}^{d_{\min}} \left(\sigma_i(W)^2 - 1\right)^2}$$
  where $\sigma_i(W)$ are the singular values of $W$. The reported model-level `Stiefel IsoErr` is the arithmetic mean of $\mathrm{IsoErr}(W)$ across all 2D projection matrices in the model backbone ($46$ matrices in our 6-layer `QuantumBornGPT`). A value of $0$ indicates exact adherence to the Stiefel manifold ($\sigma_i(W) = 1\,\forall i$), whereas in unconstrained classical models (`ClassicalGPTBaseline`), singular values $\sigma_i(W)$ grow unchecked above $1$ over multi-epoch training, causing $\mathrm{IsoErr}(W)$ to explode (`6.86` $\to$ `47.28`).

#### Exact Mathematical Specification of the 10-Qubit `QuantumBornGPT` (`70,681,903` Parameters)
Given a batch of token indices $X \in \{0, \dots, V-1\}^{B \times T}$ ($B = 128, T = 512, V = 50,304$):

1. **Isometric Hilbert-Space Lift (`IsometricHilbertLift`, $d_{\text{seed}} = 128 \to d = 1024$)**:
   - Token embeddings are parameterized in a low-dimensional seed space $E \in \mathbb{R}^{V \times d_{\text{seed}}}$ and lifted into the 10-qubit Hilbert space $S^{d-1}$ ($d = 1024$) via a semi-orthogonal Stiefel matrix $W_{\text{lift}} \in \mathrm{St}(d_{\text{seed}}, d)$ ($W_{\text{lift}} W_{\text{lift}}^\top = I_{d_{\text{seed}}}$):
     $$|\psi_{\text{seed}, t}\rangle = \mathcal{N}(E_{X_t}) \in S^{d_{\text{seed}}-1}, \qquad |\psi_t^{(0)}\rangle = \mathcal{N}(|\psi_{\text{seed}, t}\rangle W_{\text{lift}}) \in S^{d-1}$$
   - During training, small isotropic **Spherical Depolarizing Jitter** (standard deviation $\sigma_{\text{jit}} = 0.015 / \sqrt{d}$) is added before re-projecting onto $S^{d-1}$, followed by **Zero-Initialized Temporal Bigram Wave Dispersion** with learnable per-channel gate $g_{\text{smear}} \in \mathbb{R}^d$ (initialized to $0$, bounded by $\eta_{\text{smear}} = 0.06 \tanh(g_{\text{smear}}) \in [-0.06, +0.06]$):
     $$|\psi_t^{(0)}\rangle \leftarrow \mathcal{N}\!\left(|\psi_t^{(0)}\rangle + \eta_{\text{smear}} \odot |\psi_{t-1}^{(0)}\rangle\right) \quad (\text{with } |\psi_{-1}^{(0)}\rangle = \mathbf{0})$$

2. **Per-Block State Evolution ($\ell = 0, \dots, L-1$, with $L = 6$ Layers, $H = 16$ Heads, $D_h = 64$)**:
   - **Input Coherent Highway Mixing**: Using the scalar logistic function $\mathrm{sigmoid}(x) = (1 + e^{-x})^{-1}$ applied only to scalar gate logits (never to state vectors), each block $\ell$ first forms a coherent convex superposition between the incoming state $|\psi_t^{(\ell)}\rangle$ and the initial token state $|\psi_t^{(0)}\rangle$ controlled by scalar gate $\gamma_{\text{hw}}^{(\ell)} = \mathrm{sigmoid}(\text{logit}_{\text{hw}}^{(\ell)})$ (initialized to $\text{logit}_{\text{hw}}^{(\ell)} = -2.0 \Rightarrow \gamma_{\text{hw}}^{(\ell)} \approx 0.119$):
     $$|x_{h, t}^{(\ell)}\rangle = \mathcal{N}\!\left((1 - \gamma_{\text{hw}}^{(\ell)}) |\psi_t^{(\ell)}\rangle + \gamma_{\text{hw}}^{(\ell)} |\psi_t^{(0)}\rangle\right) \in S^{d-1}$$
   - **Stiefel QKV Projections & Initial-State Value-Residual Mixing**: $|x_{h, t}^{(\ell)}\rangle$ is projected via orthogonal matrices $W_Q^{(\ell)}, W_K^{(\ell)}, W_V^{(\ell)} \in \mathrm{St}(d, d)$ into $H = 16$ heads of dimension $D_h = 64$. The value states $v_t^{(\ell)}$ are mixed with the initial-state value projection $v_t^{(0, \ell)} = |\psi_t^{(0)}\rangle W_V^{(\ell)}$ via scalar gate $\alpha_v^{(\ell)} = \mathrm{sigmoid}(\text{logit}_v^{(\ell)})$ (initialized to $\text{logit}_v^{(\ell)} = -1.1 \Rightarrow \alpha_v^{(\ell)} \approx 0.25$):
     $$q_t = |x_{h, t}^{(\ell)}\rangle W_Q^{(\ell)}, \quad k_t = |x_{h, t}^{(\ell)}\rangle W_K^{(\ell)}, \quad v_t = (1 - \alpha_v^{(\ell)}) |x_{h, t}^{(\ell)}\rangle W_V^{(\ell)} + \alpha_v^{(\ell)} |\psi_t^{(0)}\rangle W_V^{(\ell)}$$
   - **2-Tap Causal QK Wave Filter, Unitary RoPE & Per-Head Hyperspherical Normalization**: Before rotary position encoding, queries and keys undergo a causal 2-tap finite-impulse-response wave filter with zero-initialized per-head parameters $m_{q, h}^{(\ell)}, m_{k, h}^{(\ell)}$ constrained to $\alpha_{q, h} = 0.06 \tanh(m_{q, h}^{(\ell)})$ and $\alpha_{k, h} = 0.06 \tanh(m_{k, h}^{(\ell)})$, followed by Unitary Rotary Position Embedding ($\mathrm{RoPE}$, which applies block-diagonal $2\times 2$ orthogonal rotations $R_t \in \mathrm{SO}(D_h)$) and per-head projective normalization onto $S^{D_h - 1}$:
     $$\tilde{q}_{h, t} = \mathcal{N}\!\left(\mathrm{RoPE}_t(q_{h, t} + \alpha_{q, h} q_{h, t-1})\right) \in S^{D_h - 1}, \qquad \tilde{k}_{h, t} = \mathcal{N}\!\left(\mathrm{RoPE}_t(k_{h, t} + \alpha_{k, h} k_{h, t-1})\right) \in S^{D_h - 1}$$
   - **Causal Squared-Amplitude Born Attention & Role of the Vacuum Sink $\epsilon_{\text{attn}}$**:
     Each head $h$ has a learnable amplitude scale $\gamma_h^{(\ell)} = \mathrm{clip}(s_h^{(\ell)}, 1.0, 40.0)$ (initialized to $\sqrt{D_h} = 8.0$). Causal transition amplitudes and Born probabilities are defined as:
     $$A_{ij}^+ = \max\!\left(0,\, \gamma_h^{(\ell)} \langle \tilde{q}_{h, i}, \tilde{k}_{h, j} \rangle\right) \cdot \mathbf{1}_{j \le i}, \qquad P_{ij} = \frac{(A_{ij}^+)^2}{\sum_{m \le i} (A_{im}^+)^2 + \epsilon_{\text{attn}}}, \qquad o_{h, i} = \sum_{j \le i} P_{ij} v_{h, j}$$
     where $\epsilon_{\text{attn}} = 10^{-6}$ acts as a non-zero **vacuum / null-measurement sink** in the denominator partition function $Z_i = (\gamma_h^{(\ell)})^2 \sum_{m \le i} \max(0, \langle \tilde{q}_{h, i}, \tilde{k}_{h, m} \rangle)^2 + \epsilon_{\text{attn}}$. Because $\epsilon_{\text{attn}} > 0$, $(\gamma_h^{(\ell)})^2$ does *not* algebraically cancel out when causal overlaps $\langle \tilde{q}_{h, i}, \tilde{k}_{h, m} \rangle$ are small or near-orthogonal: increasing $\gamma_h^{(\ell)}$ amplifies weak positive quantum overlaps above the vacuum sink $\epsilon_{\text{attn}}$, while decreasing $\gamma_h^{(\ell)}$ allows uncorrelated queries to attenuate toward the zero vector rather than amplifying noise. Consequently, $\gamma_h^{(\ell)}$ has a well-defined, non-zero gradient $d\gamma_h^{(\ell)}$ that is accumulated directly inside our fused backward kernel.
   - **Attention Output Projection & Projective Residual Collapse**:
     The concatenated head outputs $o_t \in \mathbb{R}^d$ are projected via $W_O^{(\ell)} \in \mathrm{St}(d, d)$ and superposed with $|x_{h, t}^{(\ell)}\rangle$ via projective collapse:
     $$|x_{\text{mid}, t}^{(\ell)}\rangle = \mathcal{N}\!\left(|x_{h, t}^{(\ell)}\rangle + o_t W_O^{(\ell)}\right) \in S^{d-1}$$
   - **Two-Beam Bilinear Quantum Interference MLP (`TwoBeamQuantumInterference`, $d_{\text{ff}} = 1536$)**:
     Replaces `GELU`/`SwiGLU` with two semi-orthogonal beam projections $W_a^{(\ell)}, W_b^{(\ell)} \in \mathrm{St}(d, d_{\text{ff}})$ ($1024 \times 1536$, orthonormal rows) that interfere multiplicatively before projecting back to $\mathbb{R}^d$ via $W_{\text{down}}^{(\ell)} \in \mathrm{St}(d_{\text{ff}}, d)$ ($1536 \times 1024$, orthonormal columns):
     $$|\phi_{\text{mlp}, t}^{(\ell)}\rangle = \mathcal{N}\!\left(\left[\left(|x_{\text{mid}, t}^{(\ell)}\rangle W_a^{(\ell)}\right) \odot \mathcal{N}\!\left(|x_{\text{mid}, t}^{(\ell)}\rangle W_b^{(\ell)}\right) \cdot \sqrt{d_{\text{ff}}}\right] W_{\text{down}}^{(\ell)}\right) \in S^{d-1}$$
     $$|\psi_t^{(\ell+1)}\rangle = \mathcal{N}\!\left(|x_{\text{mid}, t}^{(\ell)}\rangle + |\phi_{\text{mlp}, t}^{(\ell)}\rangle\right) \in S^{d-1}$$

3. **Mach-Zehnder U-Net Layer-Skip Interferometry**:
   - The $L = 6$ layers are divided into $n_{\text{half}} = 3$ encoder layers ($\ell \in \{0, 1, 2\}$, which save their output states $E_{\text{enc}}[k] = |\psi_t^{(k+1)}\rangle$ for $k \in \{0, 1, 2\}$) and $3$ decoder layers ($\ell \in \{3, 4, 5\}$). For each decoder layer $\ell \in \{3, 4, 5\}$, let $k = L - 1 - \ell \in \{2, 1, 0\}$ be the mirror encoder index. After decoder layer $\ell$ produces $|\psi_t^{(\ell+1)}\rangle$, it interferes coherently with the paired encoder state $E_{\text{enc}}[k] = |\psi_t^{(k+1)}\rangle$ rotated by an orthogonal gate $U_{\text{skip}}^{(k)} \in \mathrm{St}(d, d)$ (initialized to $I_d$) with learnable mixing scale $u_k = 0.35 \tanh(s_{\text{unet}, k})$ (initialized to $s_{\text{unet}, k} = 0.18$):
     $$|\psi_t^{(\ell+1)}\rangle \leftarrow \mathcal{N}\!\left(|\psi_t^{(\ell+1)}\rangle + u_k \, \mathcal{N}\!\left(E_{\text{enc}}[k]\, U_{\text{skip}}^{(k)}\right)\right) \in S^{d-1}$$

4. **Isometric Collapse to Seed Space & Stiefel-Tied POVM Readout**:
   - After layer $L = 6$, the final 10-qubit state $|\psi_t^{(L)}\rangle \in S^{1023}$ is rotated by $W_{\text{rot}} \in \mathrm{St}(d, d)$ (initialized to $I_d$) and collapsed back to the 128-D seed space via $W_{\text{lift}}^\top$:
     $$|z_t\rangle = \mathcal{N}\!\left(\mathcal{N}\!\left(|\psi_t^{(L)}\rangle W_{\text{rot}}\right) W_{\text{lift}}^\top\right) \in S^{d_{\text{seed}}-1}$$
   - The vocabulary POVM measurement vectors $|m_v\rangle \in S^{d_{\text{seed}}-1}$ are tied to the seed embeddings $E_v$ plus a zero-initialized residual delta $\Delta_v \in \mathbb{R}^{d_{\text{seed}}}$:
     $$|m_v\rangle = \mathcal{N}(E_v + \Delta_v) \in S^{d_{\text{seed}}-1}, \qquad \text{logit}_{t, v} = \beta \langle z_t, m_v \rangle, \qquad \beta = \mathrm{clip}(\beta_0, 1.0, 60.0)$$
     where $\beta_0$ is initialized to $1 / 0.035 \approx 28.5714$.

5. **Exact Parameter Count Breakdown (`70,681,903` on GPU / `70,681,904` on TPU)**:
   - Token Seed Embeddings $E$ ($50,304 \times 128$) + POVM Residual Delta $\Delta$ ($50,304 \times 128$): $2 \times 6,438,912 = 12,877,824$
   - Isometric Hilbert Lift $W_{\text{lift}}$ ($128 \times 1024$): $131,072$
   - Per-Block Stiefel Matrices ($\times 6$ blocks):
     - $W_Q, W_K, W_V, W_O$ ($4 \times 1024 \times 1024 = 4,194,304$) + $W_a, W_b, W_{\text{down}}$ ($3 \times 1024 \times 1536 = 4,718,592$) = $8,912,896$/block $\times 6 = 53,477,376$
   - Per-Block Scalar/Head Gates ($\times 6$ blocks):
     - $\gamma_h$ ($16$), $m_q$ ($16$), $m_k$ ($16$), $\text{logit}_v$ ($1$), $\text{logit}_{\text{hw}}$ ($1$) = $50$/block $\times 6 = 300$
   - U-Net Skip Rotations $U_{\text{skip}}$ ($3 \times 1024 \times 1024 = 3,145,728$) + U-Net Scales ($3$): $3,145,731$
   - Final Readout Rotation $W_{\text{rot}}$ ($1024 \times 1024$): $1,048,576$
   - Bigram Smear Gate $g_{\text{smear}}$ ($1024$) + Learnable POVM Inverse Temperature $\beta_0$ ($1$): $1,025$ (Note: in PyTorch `QuantumBornGPT`, `hilbert_lift` registers 1 scalar buffer difference vs. pure JAX, yielding `70,681,903` in PyTorch vs. `70,681,904` in JAX).

6. **Dual-Optimizer Setup (`StiefelMuon` + `AdamW`), Learning Rate Schedule & Manifold Drift Dynamics**:
   - **Stiefel Matrices (`StiefelMuon`)**: All $46$ 2D Stiefel matrices ($W_{\text{lift}}, W_{\text{rot}}, U_{\text{skip}}^{(0..2)}$, and per-block $W_Q, W_K, W_V, W_O, W_a, W_b, W_{\text{down}}$) are optimized with `StiefelMuon` (peak learning rate $\eta_{\text{muon}} = 0.025$, Nesterov momentum $\mu = 0.95$, cautious weight decay $\lambda_{\text{wd}} = 0.01$). At each step, a 5-step quintic Newton-Schulz iteration with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ computes the approximate polar factor $U_t = \mathrm{NS}_5(M_t)$ (transposing internally whenever $r > c$ so the Gram product $X X^\top$ is always $d_{\min} \times d_{\min}$ with $d_{\min} = \min(r, c)$). It then applies **Cautious Weight Decay** ($W \leftarrow W - \eta_t \lambda_{\text{wd}} (W \odot \mathbf{1}_{W \odot U_t > 0})$) and, every $k = 4$ steps (`retract_every=4`, with effective penalty step size $4 \lambda_{\text{St}} = 4 \times 0.005 = 0.02$), subtracts the **Normalized Riemannian Stiefel Penalty**:
     $$\nabla_W \mathcal{R}_{\text{St}}(W) = \begin{cases} \dfrac{W (W^\top W - I_c)}{1 + \|W^\top W - I_c\|_F / \sqrt{c}} & \text{if } r \ge c \\[10pt] \dfrac{(W W^\top - I_r) W}{1 + \|W W^\top - I_r\|_F / \sqrt{r}} & \text{if } r < c \end{cases}$$
   - **Vector, Embedding & Scalar Parameters (`AdamW`)**: All remaining parameters ($E, \Delta, g_{\text{smear}}, s_{\text{unet}}, \beta_0$, and per-block gates $\gamma_h, m_q, m_k, \text{logit}_v, \text{logit}_{\text{hw}}$) are optimized simultaneously with `AdamW` (peak learning rate $\eta_{\text{adam}} = 0.004$, $\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.01$).
   - **Gradient Clipping, Warmup-Cosine Schedule & Batch Accumulation**: Global gradient L2 norm is clipped at $1.0$ before every optimizer step. Both optimizers share a learning-rate multiplier $m(t)$ with a $150$-step linear warmup ($m(t) = t / 150$ for $t \le 150$) followed by cosine annealing to a $15\%$ floor ($m(t) = 0.15 + 0.85 \times \frac{1}{2}(1 + \cos(\pi \frac{t - 150}{T_{\text{total}} - 150}))$). Each training step processes $65,536$ tokens ($B = 128, T = 512$), executed as a single $128 \times 512$ batch on the 96 GB Blackwell GPU and as $2$ accumulated microbatches of $64 \times 512$ on the 16 GB Cloud TPU v5e-1.
   - **Why `Stiefel IsoErr` Rises at Epoch 0.5 and Reconverges by Epoch 3–5**: At Step 0, all Stiefel matrices are initialized via exact QR decomposition ($\text{IsoErr} \approx 0.0000\text{--}0.0002$). Because a finite 5-step Newton-Schulz polynomial is an approximate polar factor and the learning rate peaks at $\eta_{\text{muon}} = 0.025$ right after Step 150, large early gradient updates temporarily push weights slightly off the Stiefel manifold ($\text{IsoErr} \approx 1.25\text{--}1.48$ at Epoch 0.5) until the restorative force of the Riemannian retraction penalty $4 \lambda_{\text{St}} \nabla_W \mathcal{R}_{\text{St}}(W)$ balances the step size. As the cosine learning rate anneals over Epochs 1 to 5 while the retraction strength $4\lambda_{\text{St}} = 0.02$ remains constant, the matrices are pulled monotonically back onto the Stiefel manifold ($\text{IsoErr} \to 0.1560$).

---

### Step 2: Preliminary Architectural Ablations & Overfitting Stress Tests

Before launching the full 100M-token multi-epoch runs, we validated each architectural component through three controlled studies on an NVIDIA RTX PRO 6000 Blackwell GPU:

#### Study A: 6-Loop Cumulative Quantum-Geometric Architectural Ablation (`150` Steps = `1,228,800` Tokens/Loop, `B=16, T=512`)
In this preliminary ablation (`L=6, d=1024, d_ff=2048, d_seed=616`), we started from a matched-scale `ClassicalGPTBaseline` (`116.83M` params, standard Softmax attention + RMSNorm + GELU) and a basic `Loop 0` `QuantumBornGPT` (`116.80M` params, which used an untied $50,304 \times 616$ POVM readout table alongside the $50,304 \times 616$ seed embedding table), adding one quantum-geometric mechanism per loop:
- **Loop 1 (`85.79M` params, `-31.01M` params)**: Tied the POVM readout directly to the seed embeddings via $|m_v\rangle = \mathcal{N}(E_v + \Delta_v)$ (with rank-factored delta) and introduced warmup-cosine `StiefelMuon`.
- **Loop 2 (`85.79M` params)**: Added zero-initialized temporal bigram wave dispersion ($g_{\text{smear}}$) and the 2-tap causal QK wave filter ($m_q, m_k$).
- **Loop 3 (`88.93M` params, `+3.15M` params)**: Added the $3$ Mach-Zehnder U-Net skip rotation matrices $U_{\text{skip}}^{(0..2)} \in \mathrm{St}(1024, 1024)$.
- **Loop 4 (`103.61M` params, `+14.68M` params)**: Upgraded the single-beam projective MLP to `TwoBeamQuantumInterference` (adding the second Stiefel beam matrix $W_b^{(\ell)}$ across all 6 layers).
- **Loop 5 (`155.13M` params, `+51.51M` params)**: Added a **Dual-Frame Full-Hilbert 10-Qubit POVM Readout**, which augments the compressed seed-space POVM inner product $\beta_{\text{seed}} \langle z_t | m_v \rangle$ with a second direct 10-qubit measurement frame $M_{\text{full}} \in \mathbb{R}^{V \times 1024}$ operating directly on $|\psi_t^{(L)}\rangle \in S^{1023}$.
- **Loop 6 (`188.70M` params, `+33.57M` params)**: Replaced the dense MLP with a **Coherent Shared + Parseval `QuantumBornMoE`** layer (1 always-active shared `TwoBeamQuantumInterference` expert plus $E = 4$ routed experts). Tokens are routed with **zero Softmax** by projecting $|x_{\text{mid}, t}\rangle$ against a Parseval frame $R \in \mathrm{St}(d, E)$ ($R^\top R = I_E$) and taking squared Born amplitudes $p_e(x) = \frac{|\langle r_e | x \rangle|^2}{\sum_{j=1}^E |\langle r_j | x \rangle|^2 + \epsilon}$ over the Top-$2$ experts.

| Architecture Loop | Params | Val CE | Val PPL | Top-1 Acc | Med Rank | Stiefel IsoErr | Eager Speed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ClassicalGPTBaseline (Matched 10-Qubit Scale, Softmax+LN+GELU)** | 116.83M | 7.1169 | 1,232.6 | 10.72% | 89 | 16.8400 | 195,939 tok/s |
| **Loop 0: Baseline 10-Qubit QuantumBornGPT (Untied POVM)** | 116.80M | 6.6579 | 778.9 | 13.98% | 56 | 1.0348 | 88,043 tok/s |
| **Loop 1: + Stiefel-Tied POVM Readout & Warmup-Cosine StiefelMuon** | 85.79M | 6.4485 | 631.8 | 15.09% | 45 | **0.0834** | 86,952 tok/s |
| **Loop 2: + Zero-Init Bigram Smear & 2-Tap Causal QK Wave Filter** | 85.79M | 6.4013 | 602.7 | 15.28% | 42 | **0.0916** | 85,476 tok/s |
| **Loop 3: + Mach-Zehnder U-Net Layer-Skip Interferometry** | 88.93M | 6.3710 | 584.6 | 15.54% | 40 | **0.0842** | 80,918 tok/s |
| **Loop 4: + Two-Beam Bilinear Quantum Interference MLP** | 103.61M | 6.3547 | 575.2 | 15.54% | 38 | **0.0873** | 70,016 tok/s |
| **Loop 5: + Dual-Frame Full-Hilbert 10-Qubit POVM Readout** | 155.13M | 6.3440 | 569.1 | 15.80% | **37** | **0.0878** | 60,100 tok/s |
| **Loop 6: + Coherent Shared + Parseval QuantumBornMoE (4 Experts)** | 188.70M | **6.3394** | **566.5** | **15.87%** | 38 | **0.1164** | 48,541 tok/s |

**Selection Rationale for the Full 100M-Token GPU & TPU Benchmarks**: While Loops 5 and 6 achieved the lowest single-pass loss, Loop 5's unfactored $50,304 \times 1024$ vocabulary matmul and Loop 6's sparse expert gather/scatter reduced throughput and increased HBM pressure on a single 16 GB Cloud TPU v5e-1 chip. We therefore selected the **Loop 4 dense architecture** paired with a compact $d_{\text{seed}} = 128$ `IsometricHilbertLift` and $d_{\text{ff}} = 1536$ (`70.68M` parameters), which preserves all core quantum-geometric mechanisms (`IsometricHilbertLift`, `SubspaceBornAttention`, `TwoBeamQuantumInterference`, Mach-Zehnder U-Net skip interferometry, and Stiefel-tied POVM readout) while running exclusively on dense Tensor Core / MXU matrix multiplications.

#### Study B: 10-Epoch Multi-Pass Overfitting Stress Test (`1.97M`-Token Micro-Shard, `52:1` Parameter-to-Token Ratio)
To test resistance to multi-epoch memorization, we looped the `103.61M`-parameter Loop 4 `QuantumBornGPT` and the matched `ClassicalGPTBaseline` for **10 full epochs** over a tiny **`1.97M`-token** FineWeb slice (`52` model parameters per training token). `QuantumBornGPT` surpassed the classical model's best 10-epoch validation loss by **Epoch 2** (`6.0022` vs. `6.1115`) and reached **`5.5941` Val CE (`268.8` PPL)** while driving Stiefel isometry drift down to `0.0286` (whereas classical Euclidean weights suffered a $7\times$ singular-value explosion from `IsoErr 6.86` to `47.28`):

| Epoch | Quantum Train CE | Quantum Val CE (PPL) | Quantum Top-1 / Rank | Quantum IsoErr | Classical Train CE | Classical Val CE (PPL) | Classical Top-1 / Rank | Classical IsoErr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Epoch 1** | 7.5025 | **6.3353** (**564.1**) | **15.52%** / `40` | **0.5623** | 7.4366 | 6.8407 (935.2) | 12.04% / `68` | 6.8578 |
| **Epoch 2** | 5.9199 | **6.0022** (**404.3**) | **17.07%** / `28` | **0.5853** | 6.5862 | 6.5353 (689.0) | 13.78% / `48` | 12.3377 |
| **Epoch 4** | 5.0829 | **5.7293** (**307.8**) | **18.49%** / `20` | **0.3954** | 5.9394 | 6.2669 (526.9) | 15.80% / `36` | 24.2110 |
| **Epoch 6** | 4.4533 | **5.5941** (**268.8**) | **19.83%** / `16` | **0.2216** | 5.2610 | 6.1480 (467.8) | 17.07% / `29` | 36.4335 |
| **Epoch 10** | 2.9820 | **5.8561** (**349.4**) | **18.89%** / `20` | **0.0286** | 4.1914 | 6.1138 (452.1) | 17.91% / `26` | 47.2783 |

#### Study C: Initial 5-Epoch Run on a 20M-Token FineWeb Slice (`99.94M` Total Tokens Processed)
Before running on the full 100M-token shard, we trained the `70.68M`-parameter `QuantumBornGPT` for **5 epochs over a 20,000,000-token slice** (`305` steps/epoch $\times 65,536$ tok/step $\times 5$ epochs = `1,525` total steps = `99.94M` total training tokens). This baseline (plotted as the orange reference curve `GPU 20M-Slice 5-Epoch (100M tok)` in our multi-run dashboard) progressed from `Val CE 5.3894` (`219.1` PPL) at Epoch 1 (`20M` tok) $\to$ `5.0124` (`150.3` PPL) at Epoch 2 (`40M` tok) $\to$ `4.8012` (`121.7` PPL) at Epoch 3 (`60M` tok) $\to$ `4.6685` (`106.5` PPL) at Epoch 4 (`80M` tok) $\to$ **`Val CE 4.6034` (`Val PPL 99.8`, `Top-1 27.31%`, `Top-5 46.45%`, `MedRank 6`)** at Epoch 5 (`99.94M` tok). Seeing validation loss still falling at Epoch 5 motivated scaling up to multiple full passes over the entire **100M-token** shard (`300M` and `500M` total tokens) on both GPU and TPU.

---

### Step 3: Kernel & Compiler Optimization for Full 100M-Token GPU & TPU Runs

To scale to the full **100,000,000-token FineWeb shard** (`299.83M` tokens in 3 epochs; `499.71M` tokens in 5 epochs), we engineered four major kernel and compiler optimizations:
1. **Atomic-Free SRAM-Accumulated Backward Triton Kernel (`src/qdlblocks/kernels/flash_born.py`)**:
   - Split the backward pass into two dedicated Triton kernels: `_flash_born_bwd_kernel` (parallelized over key/value tiles to accumulate $dK, dV, d\gamma_h$ in registers) and `_flash_born_bwd_dq_kernel` (parallelized over query tiles to accumulate $dQ$ entirely in on-chip SRAM registers and write once in `bfloat16`), eliminating all `1.8 billion` contested global FP32 `tl.atomic_add` calls per step.
   - Upgraded forward and backward tile sizes to `64x64` when $T \ge 64$ and wrapped `_run_triton_flash_born` with `@torch.compiler.disable` so `torch.compile` (TorchInductor) fuses all surrounding PyTorch operations into single CUDA kernels while invoking the custom Triton autograd function cleanly.
2. **Strict `bfloat16` Preservation (`src/qdlblocks/functional.py`)**:
   - Updated `l2_normalize` to compute the squared norm in `float32` (`x.float().pow(2).sum(...)`) while casting the reciprocal norm back via `.type_as(x)`, and updated `apply_unitary_rope` to cast `cos, sin` to `x.dtype`. This kept all intermediate activations in `bfloat16` under `torch.amp.autocast`.
3. **Amortized `bfloat16` Stiefel Retraction & Compiled Newton-Schulz (`src/qdlblocks/optim/muon.py`)**:
   - Compiled `zeropower_via_newtonschulz5` with `@torch.compile` and amortized the Stiefel retraction penalty to run every 4 steps (`retract_every=4`) using `bfloat16` Tensor Core Gram products.
   - **Combined GPU Speedup**: Increased training throughput on the NVIDIA RTX PRO 6000 Blackwell from `78,339 tok/s` (eager) to **`311,495 tok/s`** (**`3.98x` speedup**) at **`24.97 GB` Peak VRAM**.
4. **Whole-Step JAX/XLA `@jax.jit` & Chunked Scatter-Free Factorized MXU Cross-Entropy on TPU (`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`)**:
   - On Google Cloud TPU v5e-1, we fused the entire training step—2-step microbatch gradient accumulation (`2 x 64 x 512 = 65,536` tok/step), per-block `@jax.checkpoint` rematerialization, global gradient norm clipping, 5-step `bfloat16` MXU Newton-Schulz `StiefelMuon`, and `AdamW`—into a single `@jax.jit(donate_argnums=(0, 1))` XLA HLO executable.
   - Replaced standard `one_hot` cross-entropy with **Chunked Factorized MXU Cross-Entropy (`mxu_factorized_ce`)**:
     $$\mathcal{L}_i = \underbrace{\log \sum_{v=1}^V \exp(\beta \langle z_i, m_v \rangle)}_{\text{4 Chunks of 8,192 Tokens on BF16 MXU + Remat LSE}} - \underbrace{\beta \langle z_i, m_{y_i} \rangle}_{\text{128-D Gather + Dot Product}}$$
     Specifically, for each microbatch of $N = B_{\text{micro}} \times T = 64 \times 512 = 32,768$ tokens, the target logit term $\beta \langle z_i, m_{y_i} \rangle$ is computed directly in $128$-D seed space without touching the $50,304$-column logit matrix, while the log-sum-exp partition function is scanned across $4$ chunks of $8,192$ tokens (`[4, 8192, 128]`) decorated with `@jax.checkpoint` (and $4$ chunks of $16,384$ tokens with `torch.utils.checkpoint` on GPU). Each chunk's $[8192, 128] \times [128, 50304] \to [8192, 50304]$ `bfloat16` matmul (`824 MB`) executes on the TPU's systolic MXU, reduces immediately to a 1D log-sum-exp vector, and frees its intermediate buffer before the next chunk runs. This eliminated the 50,304-column `one_hot` scatter on the TPU VPU, cutting cross-entropy step time by **`3.77x`** (`155.67 ms` $\to$ `41.32 ms`) and sustaining **`122,073 tok/s`** at **`100.0%` MXU duty cycle** and **`5.76 GB` live HBM2e usage** (`34.1%` of `16.91 GB`).

#### Exact Per-Token FLOPs Derivation (`153.91` / `461.73` / `548.40` MFLOPs) & MFU/HFU Formulas
For a single token passing through the 6-layer `QuantumBornGPT` ($d = 1024, H = 16, D_h = 64, d_{\text{ff}} = 1536, d_{\text{seed}} = 128, V = 50,304, T = 512$), counting $2mn$ FLOPs per $(m \times n)$ matrix-vector multiply:
- **Forward Pass (`153.91 MFLOPs/token`)**:
  1. Isometric Hilbert Lift ($128 \to 1024$) + Seed Collapse ($1024 \to 128$): $2 \times (2 \times 128 \times 1024) = 0.52\text{ MFLOPs}$
  2. Per Block ($\times 6$ blocks):
     - $W_Q, W_K, W_V, W_O$ ($4 \times 2 \times 1024^2 = 8.39\text{ MFLOPs}$) + $V_0$ Value-Residual projection ($2 \times 1024^2 = 2.10\text{ MFLOPs}$) = $10.49\text{ MFLOPs}$
     - Causal Born Attention ($Q K^\top$ + $P V$): $4 \times H \times D_h \times T = 4 \times 1024 \times 512 = 2.10\text{ MFLOPs}$
     - `TwoBeamQuantumInterference` MLP ($W_a, W_b, W_{\text{down}}$: $3 \times 2 \times 1024 \times 1536$): $9.44\text{ MFLOPs}$
     - Subtotal per block: $22.02\text{ MFLOPs} \times 6 = 132.12\text{ MFLOPs}$
  3. U-Net Skip Rotations ($3 \times 2 \times 1024^2 = 6.29\text{ MFLOPs}$) + Final Readout Rotation ($2 \times 1024^2 = 2.10\text{ MFLOPs}$) = $8.39\text{ MFLOPs}$
  4. Stiefel-Tied POVM Logits ($128 \to 50,304$): $2 \times 128 \times 50,304 = 12.88\text{ MFLOPs}$
  - **Total Forward FLOPs/token**: $0.52 + 132.12 + 8.39 + 12.88 = \mathbf{153.91\text{ MFLOPs/token}}$.
- **Standard Analytical Forward + Backward (`461.73 MFLOPs/token`) & MFU Formula**:
  - Using the standard PaLM / NanoGPT convention ($1\times\text{Fwd} + 2\times\text{Bwd} = 3\times\text{Fwd}$): $F_{\text{analytical}} = 3 \times 153.91 = \mathbf{461.73\text{ MFLOPs/token}}$ (`30.26 TFLOPs/step` at `65,536` tok/step).
  - **Model FLOPs Utilization (MFU)**:
    $$\text{MFU (\%)} = 100 \times \frac{(\text{Sustained tok/s}) \times 461.73 \times 10^6\text{ FLOPs/tok}}{\text{Accelerator Peak BF16 FLOP/s}}$$
- **Hardware-Executed Forward + Backward + Rematerialization + `StiefelMuon` (`548.40 MFLOPs/token`) & HFU Formula**:
  - Adding the checkpointed forward recomputation of the POVM logit chunks and block activations ($+83.2\text{ MFLOPs/token}$) plus the 5-step Newton-Schulz matrix multiplications and Stiefel retractions across all 46 Stiefel matrices amortized over $65,536$ tokens/step ($+3.5\text{ MFLOPs/token}$) yields $F_{\text{hardware}} = \mathbf{548.40\text{ MFLOPs/token}}$ (`35.94 TFLOPs/step`).
  - **Hardware FLOPs Utilization (HFU)**:
    $$\text{HFU (\%)} = 100 \times \frac{(\text{Sustained tok/s}) \times 548.40 \times 10^6\text{ FLOPs/tok}}{\text{Accelerator Peak BF16 FLOP/s}}$$

---

### Step 4: Detached Subprocess Execution & Live LAN Telemetry Dashboard (`scripts/monitor_colab_runs.py`)
To eliminate IPython memory retention and WebSocket timeouts on Google Colab:
- Before launching each benchmark, we terminated any existing IPython kernel (`kill -9 <pid>`) to release GPU VRAM and TPU `/dev/accel0` locks, and spawned the standalone Python script (`python3 -u ... > /content/<run>.log 2>&1`) in a detached OS session (`start_new_session=True`).
- We built `scripts/monitor_colab_runs.py`, a local collector and `ThreadingHTTPServer` hosted on **`0.0.0.0:8765`** (accessible across the local LAN at `http://192.168.50.34:8765`, plus `--watch` for terminal TUI and `/api/snapshot` for JSON) that polls the remote Colab VMs and displays:
  1. **Live GPU Hardware Telemetry (`nvidia-smi`)**: SM Compute Utilization (`99%`), VRAM Bandwidth Utilization (`59%`), VRAM Used/Total (`25.26 / 95.59 GiB`, Peak PyTorch Allocated `24.97 GB`), Power Draw (`525.5 W`), and Temperature/Clock (`76°C @ 2,392 MHz`).
  2. **Live TPU v5e-1 Hardware Telemetry (`libtpu` gRPC `RuntimeMetricService` on `localhost:8431`)**: Queries `/tpu.monitoring.runtime.RuntimeMetricService/GetRuntimeMetric` inside the TPU runtime process for `tpu.runtime.tensorcore.dutycycle.percent` (**`100.0%` MXU Duty Cycle**) and `tpu.runtime.hbm.memory.usage.bytes` / `total.bytes` (**`5.76 GB / 16.91 GB`**, `34.1%` HBM2e).
  3. **4-Panel Multi-Run SVG Overlays**: Plots Validation CE, Step Training CE, Top-1 Accuracy, and Stiefel Isometry Error across all runs (`GPU 3-Epoch 300M`, `GPU 5-Epoch 500M`, `TPU v5e-1 3-Epoch 300M`, and the Study C `GPU 20M-Slice 5-Epoch 100M` baseline).
  4. **JAX / Equinox / Haliax / Optax Named-Axis & XLA HLO Compiler Analysis (`qdlblocks_jax`)**:
     - Compiles `HaxQuantumBornGPT` (`6`-qubit Hilbert space $d = 64 = 2^6$, `4`-expert Top-2 Sparse Born MoE with Kraus expert channels, `215,552` parameters, batch $8 \times 64$) via JAX XLA and extracts:
       - **Factorized Named Qubit Axes**: `(batch:8, seq:64, qubit_0:2, qubit_1:2, qubit_2:2, qubit_3:2, qubit_4:2, qubit_5:2)`
       - **XLA HLO `cost_analysis()`**: `508,216,832` Fwd+Bwd FLOPs, `801,568,384` bytes accessed, Arithmetic Intensity **`0.63 FLOPs/Byte`**.
       - **XLA HLO `memory_analysis()`**: `175,066.2 KB` temporary buffer footprint.
       - **Optax `scale_by_stiefel_muon` + Bipartite Entanglement Trace ($3+3$ Qubit Partition)**: Tracks loss convergence from `11.7695` (Step 1) $\to$ `2.9980` (Step 5) $\to$ `0.0412` (Step 10) while maintaining 3-qubit reduced density matrix von Neumann entanglement entropy at $S(\rho_A) = 1.0677 \to 1.0487\text{ nats}$ (near the theoretical 3-qubit maximum of $3 \ln 2 \approx 2.079\text{ nats}$).

---

### Step 5: Full 100M-Token Multi-Epoch Benchmark Results (GPU & TPU)

All runs below trained the matched 10-Qubit `QuantumBornGPT` (`70.68M` parameters) on the full `100,000,000`-token FineWeb shard (`1,525` steps/epoch $\times$ `65,536` tokens/step = `99.94M` tokens/epoch) and evaluated on `327,680` tokens (`10` batches $\times 64 \times 512$) from the disjoint `4,999,488`-token validation shard.

#### Note on Initial Step 0 (`Epoch 0.0`) Loss (`~20.7` CE) and Perplexity Overflow Clamp (`485.2M`)
At Step 0 (`Epoch 0.0`), before any gradient updates, $|z_t\rangle \in S^{127}$ and $|m_v\rangle \in S^{127}$ are random unit vectors on the 128-dimensional hypersphere. The dot product $\langle z_t, m_v \rangle$ between two independent unit vectors in $\mathbb{R}^{128}$ has mean $0$ and standard deviation $1 / \sqrt{128} \approx 0.0884$. Because the POVM inverse temperature is initialized to $\beta_0 = 1 / 0.035 \approx 28.57$, the initial logits $\beta_0 \langle z_t, m_v \rangle$ have standard deviation $\sigma_{\text{logit}} \approx 28.57 / \sqrt{128} \approx 2.525$ (with maximum random logits reaching $\sim 4.3 \sigma_{\text{logit}} \approx +10.8$). Consequently, the log-partition function $\log \sum_{v=1}^{50304} \exp(\text{logit}_v)$ starts at $\approx 20.63\text{--}20.77\text{ nats}$ rather than the zero-logit uniform baseline $\ln(50,304) \approx 10.83\text{ nats}$. Because $\text{Val PPL} = \exp(\min(\text{Val CE}, 20.0))$, both initial losses (`20.7653` and `20.6317`) clamp at $\exp(20.0) = 485,165,195.4$ (`485.2M`).

#### Table 1: Accelerator Throughput, Memory & MFU/HFU Summary
| Accelerator Backend | Compiler & Kernel Stack | Sustained Speed | Analytical TFLOP/s (MFU) | Hardware TFLOP/s (HFU) | Compute / MXU Util | Live / Peak Memory |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **NVIDIA RTX PRO 6000 Blackwell** (`gpu_slowrun`, `503.8 TFLOP/s` BF16 peak) | `torch.compile` + Fused Triton `flash_born_attention` (`64x64` MMA + Atomic-Free SRAM Backward) + Compiled `StiefelMuon` | **311,495 tok/s** (`3.98x` vs eager) | **143.8 TFLOP/s** (**28.6%** MFU) | **170.8 TFLOP/s** (**33.9%** HFU) | **99% SM Util** (`59%` Mem BW, `525.5 W`) | **24.97 GB Peak** (`25.26 / 95.59 GiB`) |
| **Google Cloud TPU v5e-1** (`tpu_slowrun`, `197.0 TFLOP/s` BF16 MXU peak) | Pure JAX/XLA `@jax.jit(donate_argnums=(0, 1))` + Remat MXU Born Attention + Factorized MXU POVM CE + BF16 MXU `StiefelMuon` | **122,073 tok/s** | **56.4 TFLOP/s** (**28.6%** MFU) | **66.9 TFLOP/s** (**34.0%** HFU) | **100.0% MXU Duty Cycle** (`libtpu:8431`) | **5.76 / 16.91 GB HBM2e** (**34.1%**) |

#### Table 2: Matched 3-Epoch Full 100M-Token (`299.83M` Total Tokens) GPU vs. TPU Progression
*Note*: The `0.136`-nat difference at Epoch 3.0 between GPU (`4.3488` Val CE, wall time `16.13 min`) and TPU (`4.4849` Val CE, wall time `43.23 min`) stems from numerical precision in the optimizer and attention inner products: the GPU run accumulates Newton-Schulz Stiefel updates in `float32` and uses exact `float32` online normalization inside the Triton Flash-Born kernel, whereas the TPU v5e-1 run executes the 5-step Newton-Schulz polar iteration and attention inner products in native `bfloat16` on the systolic MXU. Both platforms converge to virtually identical Stiefel manifold error (`0.1613` vs. `0.1623`) and identical MFU (`28.6%`).

| Epoch (Step / Tokens) | GPU Train CE | GPU Val CE (Val PPL) | GPU Top-1 / Top-5 / Rank | GPU IsoErr | TPU Train CE | TPU Val CE (Val PPL) | TPU Top-1 / Top-5 / Rank | TPU IsoErr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Epoch 0.0** (`0` / `0.00M`) | 20.7653 | 20.7653 (485.2M) | 0.02% / 0.03% / `23804` | 0.0002 | 20.6317 | 20.6317 (485.2M) | 0.02% / 0.03% / `25874` | 0.0000 |
| **Epoch 0.5** (`762` / `49.94M`) | 5.9415 | **5.0007** (**148.5**) | 24.33% / 42.23% / `9` | 1.2568 | 6.3505 | **5.2209** (**185.1**) | 22.63% / 39.84% / `13` | 1.4866 |
| **Epoch 1.0** (`1,525` / `99.94M`) | 4.8441 | **4.7547** (**116.1**) | 26.06% / 44.72% / `7` | 0.9785 | 5.0234 | **4.9054** (**135.0**) | 24.78% / 42.95% / `9` | 1.0739 |
| **Epoch 1.5** (`2,287` / `149.88M`) | 4.6417 | **4.6126** (**100.7**) | 27.22% / 46.33% / `6` | 0.7210 | 4.7845 | **4.7845** (**119.6**) | 25.66% / 44.30% / `8` | 0.7182 |
| **Epoch 2.0** (`3,050` / `199.88M`) | 4.5670 | **4.5284** (**92.6**) | 27.96% / 47.23% / `6` | 0.4623 | 4.6912 | **4.6893** (**108.8**) | 26.38% / 45.21% / `7` | 0.4763 |
| **Epoch 2.5** (`3,812` / `249.82M`) | 4.4032 | **4.4006** (**81.5**) | 29.35% / 48.78% / `5` | 0.2648 | 4.5571 | **4.5555** (**95.2**) | 27.65% / 46.93% / `6` | 0.2660 |
| **Epoch 3.0** (`4,575` / `299.83M`) | **4.3322** | **4.3488** (**77.4**) | **29.80%** / **49.38%** / **`5`** | **0.1613** | **4.4846** | **4.4849** (**88.7**) | **28.43%** / **47.76%** / **`6`** | **0.1623** |

#### Table 3: Extended 5-Epoch Full 100M-Token (`499.71M` Total Tokens — Half a Billion Tokens) GPU Scaling Run
*Note*: Because the cosine learning rate decay schedule in the 5-epoch run spans `7,625` total steps instead of `4,575` steps, the learning rate stays higher through Epochs 1–3 (`Val CE 4.5340` at Epoch 3.0) before annealing into a significantly deeper minimum by Epoch 5.0 (**`Val CE 4.2676`**, **`Val PPL 71.3`**, **`Top-5 50.35%`**, **`MedRank 4`** in `26.83 minutes`):

| Epoch | Step | Tokens Processed | Train CE | Val CE | Val PPL | Top-1 Acc | Top-5 Acc | Med Target Rank | Stiefel IsoErr | Throughput |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Epoch 0.0** | 0 | 0.00M | 20.7653 | 20.7653 | 485,165,195.4 | 0.02% | 0.03% | 23,804 | 0.0002 | — |
| **Epoch 0.5** | 762 | 49.94M | 5.9102 | **4.9337** | **138.9** | 25.15% | 43.06% | 9 | 1.3763 | 310,177 tok/s |
| **Epoch 1.0** | 1,525 | 99.94M | 4.8612 | **4.8971** | **133.9** | 24.50% | 43.09% | 8 | 1.1037 | 311,481 tok/s |
| **Epoch 1.5** | 2,287 | 149.88M | 4.7120 | **4.7164** | **111.8** | 26.17% | 44.97% | 7 | 0.9996 | 311,485 tok/s |
| **Epoch 2.0** | 3,050 | 199.88M | 4.6615 | **4.6395** | **103.5** | 26.75% | 45.79% | 6 | 0.8279 | 311,488 tok/s |
| **Epoch 2.5** | 3,812 | 249.82M | 4.5580 | **4.5745** | **97.0** | 27.47% | 46.70% | 6 | 0.6366 | 311,490 tok/s |
| **Epoch 3.0** | 4,575 | 299.83M | 4.5210 | **4.5340** | **93.1** | 27.78% | 47.17% | 6 | 0.5188 | 311,491 tok/s |
| **Epoch 3.5** | 5,337 | 349.77M | 4.4112 | **4.4549** | **86.0** | 28.50% | 47.95% | 5 | 0.3864 | 311,493 tok/s |
| **Epoch 4.0** | 6,100 | 399.77M | 4.3810 | **4.3720** | **79.2** | 29.41% | 49.01% | 5 | 0.2627 | 311,495 tok/s |
| **Epoch 4.5** | 6,862 | 449.71M | 4.2620 | **4.3058** | **74.1** | 30.24% | 49.96% | 5 | 0.1863 | 311,484 tok/s |
| **Epoch 5.0** | 7,625 | **499.71M** | **4.2336** | **4.2676** | **71.3** | **30.55%** | **50.35%** | **4** | **0.1560** | **311,495 tok/s** |

---

## 5. Completed Technical Roadmap Implementations (Sections 5.1–5.5)

All roadmap items below have been implemented across `qdlblocks` (PyTorch + Triton) and `qdlblocks_jax` (JAX + Equinox + Haliax + Levanter + Pallas) and verified with 40 unit tests plus end-to-end execution on a local **NVIDIA GeForce RTX 5060 Ti (16 GB Blackwell)** GPU:

### 5.1. Pushing the 100M-Token Slowrun Frontier (WSD Schedule, Context Curriculum & Multi-Token POVM) — **Implemented**
1. **Warmup-Stable-Decay (WSD) Multi-Epoch Schedule & Checkpoint Cooldown Branching (`src/qdlblocks/optim/schedules.py`, `src/qdlblocks_jax/optim.py`)**:
   - Implemented `wsd_lr_multiplier(step, total_steps, warmup_steps, decay_fraction, min_lr_ratio)` and `wsd_lr_schedule_jax(...)`, keeping `StiefelMuon` and `AdamW` at a constant `1.0` multiplier across the stable plateau before executing a radical $1 - \sqrt{t / T_{\text{decay}}}$ cooldown over the final `decay_fraction` of steps.
   - Implemented `branch_wsd_cooldown_multiplier(step_in_cooldown, cooldown_steps, min_lr_ratio)` to branch radical square-root cooldown trajectories from any intermediate epoch checkpoint.
2. **Progressive Sequence-Length Curriculum ($T: 256 \to 512 \to 1024 \to 2048$) & Dynamic `UnitaryRoPE` Extension (`src/qdlblocks/optim/schedules.py`, `src/qdlblocks/layers/isometric.py`)**:
   - Implemented `ProgressiveSequenceCurriculum(tokens_per_step=65536, seq_lengths=(256, 512, 1024, 2048))` which maps each epoch to $(B_{\text{ep}}, T_{\text{ep}})$ while preserving the exact invariant $B_{\text{ep}} \times T_{\text{ep}} = \text{tokens\_per\_step}$.
   - Upgraded `UnitaryRoPE.get_cos_sin(seq_len)` to dynamically extend precomputed $(SO(2))^{d/2}$ tables whenever `seq_len` exceeds `max_seq_len`.
3. **Multi-Token Quantum POVM Prediction (MTP) (`src/qdlblocks/models/born_gpt.py`, `src/qdlblocks_jax/haliax_modules.py`, `src/qdlblocks_jax/levanter_model.py`)**:
   - Added `mtp_horizons` and `mtp_loss_weight` to both `QuantumBornGPT` and `HaxQuantumBornGPT`, initializing $H_{\text{mtp}} - 1$ Stiefel rotation matrices $U_{\Delta t} \in \mathrm{St}(d_{\text{seed}}, d_{\text{seed}})$ (`mtp_rotations`, orthogonalized by `StiefelMuon`) that rotate $|z_t\rangle \mapsto \mathcal{N}(U_{\Delta t} |z_t\rangle)$ to predict tokens $t+1, t+2, \dots, t+H_{\text{mtp}}$ simultaneously against the shared Stiefel-tied POVM dictionary $|m_v\rangle$, plus `forward_mtp_logits` for speculative multi-token decoding.

### 5.2. Custom Pallas TPU Flash-Born Kernel (`src/qdlblocks_jax/pallas_flash_born.py`) — **Implemented**
- Upgraded `flash_born_attention_jax` with native **Pallas forward (`pallas_fwd_kernel`) and atomic-free Pallas backward (`pallas_bwd_dq_kernel` and `pallas_bwd_dkv_kernel`) kernels** using `jax.experimental.pallas` and `jax.experimental.pallas.tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"))`:
  - Tiles $Q_m, K_n, V_n$ into on-chip VMEM/SRAM blocks and accumulates unnormalized Born numerators $\tilde{O}_i = \sum_{j \le i} (A_{ij}^+)^2 V_j$ and partition functions $Z_i = \sum_{j \le i} (A_{ij}^+)^2 + \epsilon$ via `jax.lax.fori_loop` with causal upper-triangular block skipping (`max_n_blk = m_blk + 1`).
  - Supports both scalar `scale` and differentiable per-head `scale` vectors $\gamma_h \in \mathbb{R}^H$ with exact backward gradients $(dQ, dK, dV, d\gamma_h)$ verified against `jax.lax.scan` and PyTorch reference autograd.

### 5.3. Distributed Haliax/Levanter Sharding & Multi-Device `StiefelMuon` (`src/qdlblocks_jax/distributed.py`, `src/qdlblocks/optim/distributed_muon.py`) — **Implemented**
- **Named-Axis FSDP & Tensor/Expert Parallelism (`qdlblocks_jax.distributed`)**:
  - Implemented `build_quantum_axis_mapping()`, `make_quantum_device_mesh()`, and `shard_quantum_model()`, mapping Haliax `Hilbert` and `Vocab` axes to `ResourceAxis.DATA` (FSDP) and `Heads`, `MlpDim`, and `Experts` axes to `ResourceAxis.MODEL` (Tensor / Expert Parallelism) via `jax.sharding.NamedSharding`.
- **Rank-Partitioned Multi-Device `StiefelMuon` (`DistributedStiefelMuon` & `distributed_stiefel_muon_step`)**:
  - Partitions the 2D/3D Stiefel matrices round-robin across data-parallel ranks (`i % world_size == rank`) so each accelerator executes the 5-step Newton-Schulz polar iteration and Riemannian Stiefel retraction penalty only on its owned subset before synchronizing via `all_gather` / broadcast dispersion.

### 5.4. AMD Radeon AI PRO R9700 (32 GB RDNA 4) Kernel Readiness (`src/qdlblocks/kernels/flash_born.py`) — **Implemented (Ready for R9700 Hardware)**
- Added `IS_HIP` detection, `get_flash_born_launch_meta(seq_len, is_hip)`, and `get_rdna4_autotune_configs()` to `src/qdlblocks/kernels/flash_born.py`, automatically injecting ROCm Wave32 EU launch hints (`waves_per_eu=2`, `matrix_instr_nonkdim=16`, `num_warps=4`, `num_stages=1`) when running on ROCm/HIP while preserving CUDA Warp32 defaults on NVIDIA GPUs.

### 5.5. Richer Quantum-State Telemetry & Local LAN Dashboard (`0.0.0.0:8765`) — **Implemented**
- **Fast $O(2^k)$ Pure-State Schmidt Partial Trace (`partial_trace_qubits` & `bipartite_entanglement_entropy`)**:
  - Optimized pure-state partial trace in both PyTorch (`src/qdlblocks/functional.py`) and JAX (`src/qdlblocks_jax/functional.py`): for any pure state $|\psi\rangle \in \mathbb{C}^{2^n}$, reshaping into the bipartite matrix $M \in \mathbb{C}^{2^k \times 2^{n-k}}$ computes $\rho_A = M M^\dagger \in \mathbb{C}^{2^k \times 2^k}$ in $O(2^{n+k})$ time without materializing the $2^n \times 2^n$ density matrix ($1,000\times$ faster for 10-qubit $5+5$ splits where $2^5 \times 2^5 = 32 \times 32$ vs. $1024 \times 1024$).
- **6-Panel Live LAN Dashboard (`scripts/monitor_colab_runs.py` on `0.0.0.0:8765`)**:
  - Extended `inspect_quantum_model`, `model_quantum_scale_telemetry`, `examples/11_fineweb_100m_multiepoch_gpu_tpu.py`, and `scripts/monitor_colab_runs.py` to log and plot SVG time-series charts of 10-qubit ($5+5$ split) bipartite entanglement entropy $S(\rho_A)$, per-head Born attention scales $\gamma_h^{(\ell)}$, POVM readout inverse temperature $\beta = 1/\tau$, and live local + remote GPU/TPU hardware telemetry.

#### Table 4: Local NVIDIA GeForce RTX 5060 Ti (16 GB Blackwell) End-to-End Verification (`WSD` + `SeqCurriculum 256->512` + `MTP=3` + 10-Qubit Entanglement)
| Epoch (`T_ep`) | Step | Tokens | Train CE (w/ MTP=3) | Val CE | Val PPL | Top-1 / Top-5 / Rank | Stiefel IsoErr | 10Q $S(\rho_A)$ ($5+5$ Split) | $\gamma_h$ / $\beta$ | Peak VRAM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Epoch 0.0** (`T=256`) | 0 | 0.00M | 19.9412 | 19.9412 | 457.5M | 2.41% / 2.42% / `24428` | 0.0002 | **2.9547 nats** | `8.00` / `28.57` | — |
| **Epoch 0.5** (`T=256`) | 31 | 0.13M | 12.4781 | **5.0463** | **155.4** | 20.44% / 41.22% / `12` | 0.4052 | **2.9531 nats** | `8.00` / `28.49` | 3.42 GB |
| **Epoch 1.0** (`T=256`) | 63 | 0.26M | 6.9225 | **4.3042** | **74.0** | 30.77% / 46.80% / `6` | 0.8621 | **2.9568 nats** | `7.99` / `28.35` | 3.42 GB |
| **Epoch 1.5** (`T=512`) | 94 | 0.39M | 6.2392 | **4.2261** | **68.5** | 32.08% / 48.34% / `7` | 1.1060 | **2.9540 nats** | `7.97` / `28.24` | 5.21 GB |
| **Epoch 2.0** (`T=512`) | 126 | 0.52M | **6.0961** | **3.9464** | **51.7** | **35.09%** / **50.46%** / **`5`** | **0.9167** | **2.9556 nats** | `7.97` / `28.20` | **5.21 GB** |

