# Experiment Plan 039: Fast-Loop Multi-Class Vision Benchmarks (`CIFAR-100` $K=100, N=2^7$ & `ImageNet-1K` Scale `1,000`-Class $K=1000, N=2^{10}$) — Reproducing and Resolving the High-Top-1 / Bad-Cross-Entropy Paradox (`Wave 24`)

## 1. Context & Mathematical Motivation

### 1.1 Why `Wave 23` (Plan 038) Motivated Moving to Fast Standard Multi-Class Benchmarks (`CIFAR-100` & `1,000`-Class)
In `Wave 23` (Plan 038 on `50,257`-class FineWeb-Edu), moving the measurement operator to the last layer (`use_last_layer_measurement_operator=True`) and adding the Traceless Observable (`use_traceless_observable=True`) achieved **three major wins**:
1. Prevented the detector cone collapse of `w22` (`R_sharp = 99.78` at Step 900 vs. `18.15` in `w22`).
2. Achieved the **highest 900-step Top-1 Accuracy (`20.30%`)** across all Wavelet-Cholesky and `93.97M` `QuantumBornGPT` runs (`19.46%`) while running at `25,358 tok/s` (`2.17x` faster than `w21`).
3. Maintained tight Kraus Stiefel orthogonality (`kraus_stiefel_err = 0.0131`).

However, Validation Cross-Entropy remained elevated (`5.6850` at 900 steps; `5.1479` at 3,600 steps), and each 4-run FineWeb-Edu cycle required `~35 minutes` because causal sequence modeling, Zipfian token frequencies, tied input-output embeddings, and a `50,257`-way partition function are all entangled.

By testing our **exact Wavelet-Cholesky + Quantum Readout architecture** on standard multi-class vision benchmarks where the entire dataset fits in GPU SRAM/VRAM (`< 300 MB`, **`5–15 seconds` per full multi-epoch training run on our RTX 5060 Ti**), we can:
1. **Reproduce the exact "High Top-1 Accuracy, Bad Cross-Entropy" pathology in `< 10 seconds`** on both **`CIFAR-100` ($K = 100$ classes, $N = 2^7 = 128$)** and an **`ImageNet-1K`-Scale `1,000`-Class Natural Image Benchmark ($K = 1,000$ classes, $N = 2^{10} = 1,024$)**.
2. **Isolate the exact mathematical root causes** in the readout layer without sequence/autoregressive confounders.
3. **Resolve the Cross-Entropy gap completely** in a rapid subagent planning/experiment loop so our Wavelet-Cholesky Quantum architecture **beats or matches Classical Softmax Cross-Entropy on BOTH Top-1 Accuracy AND Cross-Entropy** across `100` classes (`CIFAR-100`), `1,000` classes (`1,000`-Class), and a fast transfer probe back on `50,257` classes (`FineWeb-Edu`).

---

### 1.2 Exact Mathematical Root Cause of "High Top-1 Accuracy, Bad Cross-Entropy" in $k$-Boson / Born Readout
Consider a unit-normalized output state $\tilde{\psi}(x) \in \mathbb{S}^{d-1}$ and unit-normalized Wavelet-Cholesky class wavefunctions $\{\psi_v^{(0)}\}_{v=1}^K \subset \mathbb{S}^{d-1}$ with cosine similarities $c_v(x) = \langle \tilde{\psi}(x) \mid \psi_v^{(0)} \rangle \in [-1, 1]$.

Why does the legacy $k$-boson readout
$$\ell_v^{\text{kboson}}(x) = k \ln\!\Big(1 + \tilde{r}\, c_v(x)\Big) + \lambda_{\text{diff}}\, c_{\text{diff}, v}^2(x), \qquad \tilde{r} = \operatorname{clamp}(\beta / k, \max=0.95)$$
achieve **high Top-1 accuracy** ($\arg\max_v \ell_v$) while suffering **bad Cross-Entropy** ($-\ln \frac{e^{\ell_y}}{\sum_{v=1}^K e^{\ell_v}}$) when $K \in \{100, 1000, 50257\}$?

1. **Root Cause 1 — Concave Peak Compression ($\ln(1 + \tilde{r} c)$ Shrinks the Target Margin & Starves Positive-Peak Gradients)**:
   - Because $f(c) = k \ln(1 + \tilde{r} c)$ is strictly monotonic increasing, $\arg\max_v f(c_v) = \arg\max_v c_v$, leaving Top-1 accuracy untouched.
   - However, $f''(c) = -\frac{k \tilde{r}^2}{(1 + \tilde{r} c)^2} < 0$ is **strictly concave**:
     - Its slope $f'(c) = \frac{k \tilde{r}}{1 + \tilde{r} c}$ is **smallest at the target peak** ($c_y \in [0.5, 0.9]$, where $1 + \tilde{r} c_y \approx 1.7 \implies f'(c_y) \approx 0.58 \beta$) and **largest for negative cosines** ($c_v < 0$, where $1 + \tilde{r} c_v \approx 0.2 \implies f'(c_v) \approx 5.0 \beta$).
     - Consequently, the logit margin $\ell_y - \ell_v$ between the true class ($c_y \approx 0.65$) and the positive Parseval runner-ups ($c_v \in [0.10, 0.25]$) is **compressed by ~30–38%** relative to a linear or convex energy!
     - Worse, during backpropagation, $\frac{\partial \ell_v}{\partial c_v} = \frac{k \tilde{r}}{1 + \tilde{r} c_v}$ gives **3x–5x larger gradients to already-negative tail classes** ($c_v < 0$, which contribute negligible mass to $Z$) than to the positive target peak ($c_y > 0$) and positive runner-ups ($c_v \in [0.05, 0.25]$)!
2. **Root Cause 2 — Spherical Inverse-Temperature Clamp ($\tilde{r} \le 0.95$) + Unscaled State Purity**:
   - In a Classical linear head $\ell_v = \langle h(x), w_v \rangle$, unambiguous examples increase $\|h(x)\|_2$ to sharpen the softmax distribution.
   - In the naive Quantum readout, $\|\tilde{\psi}(x)\|_2 \equiv 1$ and $\|\psi_v^{(0)}\|_2 \equiv 1$ are strictly unit-clamped AND $\tilde{r} = \operatorname{clamp}(\beta/k, \max=0.95)$ caps the maximum positive logit at $k \ln(1.95) \approx 0.668 k$. When $K = 1,000$ classes are packed into $d = 128$, Parseval conservation ($\sum_{v=1}^K c_v^2 \approx K/d = 7.81$) forces $\sim 450$ unrelated classes to have positive cosines $c_v \sim +1/\sqrt{d} \approx +0.088$, creating an uncompressible tail partition sum $Z_{\text{tail}} = \sum_{v \ne y} e^{\ell_v}$.
3. **Root Cause 3 — Flat $K$-Way Partition Sum vs. Multi-Scale Dyadic Wavelet Conditional Factorization**:
   - Our Wavelet-Cholesky class pyramid **already** represents the $K$ classes as a multi-scale hierarchy ($20$ coarse superclasses $\times$ $5$ fine classes on `CIFAR-100`; $32 \times 32 = 1,024$ coarse-to-fine dyadic clusters on `1,000`-Class; $256 \times 256 = 65,536$ on `FineWeb-Edu`).
   - A flat $K$-way partition function operates in the $K > d$ regime where exact orthogonality is impossible (Welch bound).
   - By contrast, at **each scale** of the Wavelet-Cholesky tree, the branching factor satisfies $B \le d$ ($20 \le 128$, $32 \le 128$, $256 \le 256$)! Thus every level of a **Dyadic Coarse-to-Fine Wavelet Readout** operates in the $B \le d$ regime where **exact orthogonality ($G = I_B$, zero Welch/Parseval floor!) is mathematically achievable!**

---

### 1.3 Two Clean Architectural Solutions Tested in Plan 039 (`Wave 24`)

1. **Solution A — Convex-Peak Traceless Born-Boltzmann Readout with Input-Adaptive Purity Temperature (`convex_traceless_born`)**:
   - Replaces concave $k \ln(1 + \tilde{r} c_v)$ with the **Convex-Peak Traceless Born Observable**:
     $$\hat{c}_v(x) = c_v(x) - \frac{1}{K}\sum_{u=1}^K c_u(x), \qquad \widehat{O}_v^{\text{convex}}(x) = \hat{c}_v(x) + \gamma_+ \!\left(\operatorname{relu}(\hat{c}_v(x))^2 - \frac{1}{K}\sum_{u=1}^K \operatorname{relu}(\hat{c}_u(x))^2\right)$$
     paired with an **Input-Adaptive Purity Inverse Temperature** $\beta(x) = \beta_{\text{base}} \cdot \operatorname{softplus}(w_\beta^\top h_{\text{pool}}(x) + b_\beta)$ and zero-mean centered diffuse Kraus fidelity:
     $$\ell_v^{\text{convex}}(x) = \beta(x)\,\widehat{O}_v^{\text{convex}}(x) + \lambda_{\text{diff}}\!\left(c_{\text{diff}, v}^2(x) - \frac{1}{K}\sum_{u=1}^K c_{\text{diff}, u}^2(x)\right)$$
   - **Why This Solves Root Causes 1 & 2**:
     - On the positive hemisphere $\hat{c}_v > 0$, $\frac{\partial \widehat{O}_v^{\text{convex}}}{\partial \hat{c}_v} = 1 + 2\gamma_+ \hat{c}_v$ is **strictly increasing (convex)**: it **expands** the target-to-runner-up logit margin $\ell_y - \ell_v$ and gives the **strongest gradient** ($1 + 2\gamma_+ \hat{c}_y \approx 2.2\times$) to the target peak and positive runner-ups rather than wasting gradient on negative tail classes!
     - Exact zero trace ($\sum_{v=1}^K \ell_v^{\text{convex}}(x) \equiv 0$) guarantees that increasing $\beta(x)$ on high-purity inputs exponentially suppresses the $K - 1$ non-target classes without hitting a concave $\ln(1.95)$ ceiling.

2. **Solution B — Multi-Scale Dyadic Wavelet Coarse-to-Fine Conditional Readout (`dyadic_hierarchical_born`)**:
   - Uses the Wavelet-Cholesky pyramid's coarse basis $C_{\text{coarse}} \in \mathbb{R}^{K_{\text{coarse}} \times d}$ ($K_{\text{coarse}} = 20$ on `CIFAR-100`; $K_{\text{coarse}} = 32$ on `1,000`-Class) and fine wavelet leaves $\Psi_{\text{fine}} \in \mathbb{R}^{K \times d}$ to factorize the log-probability via the exact Law of Total Probability:
     $$\ln P(y \mid x) = \ln P_{\text{coarse}}\big(g(y) \;\big|\; \tilde{\psi}_{\text{coarse}}(x)\big) + \ln P_{\text{fine}}\big(y \;\big|\; g(y),\, \tilde{\psi}_{\text{fine}}(x)\big)$$
     where both $P_{\text{coarse}}$ (over $K_{\text{coarse}} \le d$ coarse clusters) and $P_{\text{fine}}(\cdot \mid g)$ (over $K_{\text{fine}} \le d$ fine classes within cluster $g$) use the Convex-Peak Traceless Born readout!
   - **Why This Solves Root Cause 3**:
     - Because $K_{\text{coarse}} \le d$ and $K_{\text{fine}} \le d$, **both** levels operate in the $B \le d$ regime where `StiefelMuon` can orthogonalize the cluster wavefunctions to near-zero off-diagonal overlap ($B \le d \implies \text{Welch floor} = 0$!).
     - Even when inference requires a full $K$-way distribution $P(v \mid x) = P_{\text{coarse}}(g(v) \mid x) P_{\text{fine}}(v \mid g(v), x)$ (`sum_v P(v|x) == 1.0` exactly!), the log-partition function only sums over $K_{\text{coarse}}$ and $K_{\text{fine}}$ instead of $K = 1,000$ simultaneously!

---

## 2. Falsifiable Hypotheses (`H1`, `H2`, `H3`)

### Hypothesis 1 (`H1` — Reproducing & Resolving the High-Top-1 / Bad-CE Paradox on Genuine `CIFAR-100` [$K = 100, N = 2^7 = 128$] in `< 15s` per Run)
- **Claim**: Using the Wavelet-Cholesky Patch-Token Vision Backbone (`WaveletCholeskyVisionClassifier`, `d_model = 128`, `L = 3` Fused-Kronecker Wavelet-Cholesky blocks, `< 1.2M` params) trained for `20` epochs (`~3,900` steps, `< 15s` wall-clock on RTX 5060 Ti) on genuine `CIFAR-100` (`50,000` train / `10,000` test $32 \times 32 \times 3$ images, $K = 100$ fine classes, $20$ official coarse superclasses):
  1. **`naive_kboson_born` (Legacy Concave $k$-Boson Readout)** will reproduce the exact FineWeb pathology: achieving competitive Top-1 test accuracy (within `2.0%` of `classical_softmax`), **but suffering a severe Test Cross-Entropy penalty (`>= +0.25` nats worse than `classical_softmax`)** with high tail probability mass (`mean_tail_gt_10_prob_mass`) due to concave $\ln(1 + \tilde{r} c)$ peak compression.
  2. **`convex_traceless_born` (Solution A)** and **`dyadic_hierarchical_born` (Solution B)** will **both beat `naive_kboson_born` by `>= 0.30` nats in Test Cross-Entropy** AND **beat `classical_softmax` on BOTH Test Cross-Entropy AND Test Top-1 Accuracy**!

### Hypothesis 2 (`H2` — Reproducing & Resolving the Paradox on an `ImageNet-1K`-Scale `1,000`-Class Visual Benchmark [$K = 1,000, N = 2^{10} = 1,024 > d = 128$] in `< 18s` per Run)
- **Claim**: On the `1,000`-Class Fine-Grained Natural Image Benchmark ($K = 1,000$ classes embedded in an $N = 2^{10} = 1,024 = 32 \times 32$ dyadic Wavelet-Cholesky pyramid in Hilbert dimension $d = 128$, so $K / d = 7.81 \gg 1$ directly triggers the overcomplete Parseval packing regime):
  1. **`naive_kboson_born`** will exhibit an even larger Cross-Entropy pathology on $K = 1,000$ classes (`>= +0.50` nats worse Test CE than `classical_softmax`) despite high Top-1 accuracy, because $\sim 450$ positive-cosine Parseval tail classes inflate $Z_{\text{tail}}$ while concave $\ln(1 + \tilde{r} c)$ compresses the target peak.
  2. **`convex_traceless_born` (Solution A)** and **`dyadic_hierarchical_born` (Solution B)** will reduce the $K = 1,000$ tail partition fraction (`Z_tail_gt_10_fraction`), **outperform `naive_kboson_born` by `>= 0.60` nats**, and **beat `classical_softmax` on BOTH `1,000`-Class Test Cross-Entropy AND Test Top-1 Accuracy**!

### Hypothesis 3 (`H3` — Fast Transfer Verification of `convex_traceless_born` & `dyadic_hierarchical_born` Back to `50,257`-Class `FineWeb-Edu`)
- **Claim**: Porting the winning `convex_traceless_born` readout (removing concave $k \ln(1 + \tilde{r} c)$ compression in favor of the Convex-Peak Traceless Born-Boltzmann energy with adaptive purity temperature $\beta_t$) into `direct_chunked_ce.py` and `Radical2Pow16WaveletCholeskyGPT` on genuine `FineWeb-Edu` (`fineweb_train_000001.bin` $\to$ `fineweb_val_000000.bin`, `300`-step fast verification probe under `<= 8.0 GB` VRAM):
  1. Will maintain exact analytical chunked autograd (`FwdDiff < 1e-5`, `MaxGradDiff < 1e-4`).
  2. Will beat `w23_8gb_last_layer_meas_traceless` (`6.2989` at Step 300), `w22` (`6.2375` at Step 300), and `Classical 53.61M` (`6.5447` at Step 300) on **both Validation Cross-Entropy and Top-1 Accuracy**, proving that the root cause identified and cured in the 10-second `CIFAR-100` / `1,000`-Class loop directly transfers back to `50,257`-class language modeling!

---

## 3. Planned Architectural & Code Changes

1. **[`src/qdlblocks/models/wavelet_cholesky_vision.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_vision.py)** (New Reusable Module, `<= 600` lines, `<= 8` args per function, zero semicolons):
   - `VisionWaveletCholeskyConfig`: Frozen `@dataclass(frozen=True)` configuring `num_classes` ($100$ or $1,000$), `num_coarse_classes` ($20$ or $32$), `dyadic_dim` ($128$ or $1,024$), `d_model` ($128$), `num_layers` ($3$), `readout_mode` (`"classical_softmax"`, `"naive_kboson_born"`, `"last_layer_meas_traceless_kboson"`, `"convex_traceless_born"`, `"dyadic_hierarchical_born"`), and `traceless_gamma`.
   - `TiedDyadicWaveletClassPyramid`: Generalizes our dyadic Wavelet-Cholesky class hierarchy to arbitrary $(K, K_{\text{coarse}}, N = 2^P, d)$ with Last-Layer Measurement Operator, Kraus bath, and `StiefelMuon`-compatible orthogonal upsamplers.
   - `WaveletCholeskyVisionClassifier`: Patch-token Wavelet-Cholesky vision backbone (`WaveletCholeskyMultiscaleBranch` blocks + spatial pooling + all 5 readout modes + rank-bucketed Cross-Entropy / margin / tail-mass diagnostics).
   - Automatic downloader/loader for official `CIFAR-100` (`https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` $\to$ `.cache/datasets/cifar-100-python/`) and deterministic `1,000`-Class Fine-Grained Visual Benchmark constructor (`100` CIFAR-100 classes $\times$ `10` fine-grained visual chromatic/orientation/spatial-frequency states = `1,000` classes in a $32 \times 32 = 1,024$ dyadic tree).
2. **[`src/qdlblocks/layers/direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py)** & **[`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py)**:
   - Add support for `use_convex_traceless_born: bool = False` (removing concave $\ln(1 + \tilde{r} c)$ compression in favor of the linear+quadratic Convex-Peak Traceless Born energy $\beta_t [\hat{c}_{t,v} + \gamma (\hat{c}_{t,v,+}^2 - \tau_{t,+})] + \lambda_{\text{diff}} \hat{f}_{\text{diff}, t, v}$) with exact analytical forward and backward gradients in `direct_chunked_two_beam_kboson_ce`.
3. **Promoted CLI Diagnostic Scripts (`scripts/BACKLOG.md` $\to$ `scripts/`)**:
   - [`scripts/last_layer_measurement_adjoint_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/last_layer_measurement_adjoint_bench.py): Benchmarks FLOPs, wall-clock latency, memory traffic, and self-overlap suppression of the $(B \cdot T, d)$ Last-Layer Measurement Operator vs. $(K, d)$ Class/Vocab Detector synthesis across $K \in \{100, 1000, 50304\}$.
   - [`scripts/traceless_observable_rank_bucket_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/traceless_observable_rank_bucket_probe.py): Profiles logit margins, peak compression vs. convex amplification, and probability mass across rank buckets (`1`, `2–5`, `6–20`, `21–100`, `>100`) across `naive_kboson_born`, `convex_traceless_born`, and `dyadic_hierarchical_born`.
4. **Fast-Loop Verification Runner & Unit/Hypothesis Tests**:
   - [`examples/49_verify_plan039_cifar100_and_1000class_fast_loop.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/49_verify_plan039_cifar100_and_1000class_fast_loop.py)
   - [`tests/test_plan039_cifar100_and_1000class_wavelet_cholesky.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan039_cifar100_and_1000class_wavelet_cholesky.py)
   - [`tests/test_hypothesis_039_cifar100_and_1000class_fast_loop.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_039_cifar100_and_1000class_fast_loop.py)

---

## 4. Verification & Testing Plan

1. **AST & Best-Practices Gate**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/models/wavelet_cholesky_vision.py src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py scripts/last_layer_measurement_adjoint_bench.py scripts/traceless_observable_rank_bucket_probe.py examples/49_verify_plan039_cifar100_and_1000class_fast_loop.py
   ```
2. **Unit & Hypothesis Test Suite**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/pytest tests/test_plan039_cifar100_and_1000class_wavelet_cholesky.py tests/test_hypothesis_039_cifar100_and_1000class_fast_loop.py -v
   ```

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

1. **Existing Scripts Used**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
   - [`scripts/parseval_tail_overlap_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/parseval_tail_overlap_profiler.py)
   - [`scripts/block_cholesky_kernel_benchmark.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/block_cholesky_kernel_benchmark.py)
2. **`Build-Before-Launch` Scripts (Promoted from `scripts/BACKLOG.md`)**:
   - [`scripts/last_layer_measurement_adjoint_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/last_layer_measurement_adjoint_bench.py)
   - [`scripts/traceless_observable_rank_bucket_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/traceless_observable_rank_bucket_probe.py)
3. **`Nice-to-Have` Scripts (Logged in `scripts/BACKLOG.md`)**:
   - `scripts/multiclass_ce_margin_decomposer.py`: Decomposes multi-class Cross-Entropy into target-peak deficit vs. positive-cosine runner-up tail vs. negative-cosine tail across $K \in \{100, 1000, 50257\}$ classes.
   - `scripts/dyadic_tree_conditional_calibration_probe.py`: Audits coarse-cluster accuracy vs. conditional intra-cluster accuracy and calibration ECE across multi-scale Wavelet-Cholesky trees.
