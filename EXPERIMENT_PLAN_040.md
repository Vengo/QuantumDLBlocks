# Experiment Plan 040: 900-Step & 3,600-Step `FineWeb-Edu` Scaling of Convex-Peak Traceless Born Readout + Genuine `ILSVRC ImageNet-1K` (`1,281,167` Train / `50,000` Val / `100,000` Test, `1,000` Classes) Quantum Building-Block Ablation Study (`Wave 25`)

## 1. Context & Mathematical Motivation

### 1.1 Empirical Breakthrough from Plan 039 (`Wave 24`)
In Plan 039 (`Wave 24`), our fast-loop experiments on genuine `CIFAR-100` ($K=100, N=2^7=128$) and the `1,000`-Class hierarchical benchmark ($K=1,000, N=2^{10}=1,024 > d=128$) confirmed the exact root cause of the "High Top-1 Accuracy, Bad Cross-Entropy" paradox:
1. **Why Concave $k \ln(1 + \tilde{r} c)$ Inflates Cross-Entropy Even When Top-1 Accuracy Is High**:
   - On $K = 1,000$ classes in $d = 128$, `naive_kboson_born` achieved **`99.21%` Top-1 Accuracy** (beating `classical_softmax`'s `99.07%`), yet suffered a **`2.4688` vs. `0.0379` Test Cross-Entropy** (`+2.4309` nats penalty) because **`80.90%`** of its probability mass leaked into the `990` non-top-10 Parseval tail classes (`mean_tail_gt_10_prob = 0.8090`).
   - Even when `Wave 23` (`last_layer_meas_traceless_kboson`) applied the zero-trace quadratic observable $\widehat{O}_v = \hat{c}_v + \gamma (\hat{c}_{v,+}^2 - \tau_+)$, wrapping $\widehat{O}_v$ inside the concave logarithm $k \ln(1 + \tilde{r} \widehat{O}_v)$ divided positive-peak gradients by $z_v = 1 + \tilde{r} \widehat{O}_v \approx 1.8$, canceling the convex $1 + 2\gamma \hat{c}_{v,+}$ amplification and leaving `12.95%` tail mass (`0.2211` Test CE).
2. **Why Removing the Concave Logarithm (`convex_traceless_born` & `dyadic_hierarchical_born`) Solved the Paradox**:
   - Evaluating the un-compressed **Convex-Peak Traceless Born-Boltzmann Energy**
     $$\ell_v^{\text{convex}}(x) = \beta(x)\!\left[\hat{c}_v(x) + \gamma_+ \!\left(\operatorname{relu}(\hat{c}_v(x))^2 - \frac{1}{K}\sum_{u=1}^K \operatorname{relu}(\hat{c}_u(x))^2\right)\right] + \beta(x)\lambda_{\text{diff}}\,\hat{f}_{\text{diff}, v}(x)$$
     eliminated the `80.90%` tail leak (`0.0003–0.0004` tail mass), beat `classical_softmax` on both `CIFAR-100` (`1.6854` vs. `1.6916` CE; `55.14%` vs. `54.01%` Top-1) and the `1,000`-Class benchmark (`0.0146` vs. `0.0379` CE; `99.58%` vs. `99.07%` Top-1), and transferred directly to `50,257`-class `FineWeb-Edu` at Step 300 (**`6.1552` Val CE**, beating `w23` `6.2989`, `w22` `6.2375`, `w21` `6.1668`, `w20` `6.1868`, and `Classical` `6.5447`).

---

### 1.2 Objectives for Plan 040 (`Wave 25`)
Per the user's request, Plan 040 executes two major experimental investigations:

1. **900-Step (`7.37M` Tokens) & 3,600-Step (`29.49M` Tokens) `FineWeb-Edu` Scaling of `Convex-Peak Traceless Born` (`H1` & `H2`)**:
   - Scale `Radical2Pow16WaveletCholeskyGPT` (`10.40M–10.64M` params, `use_convex_traceless_born=True`, `use_kraus_retraction_and_lambda_anneal=True`, `use_fast_cholesky_and_compile=True`) to **900 steps** (`w25_8gb_convex_traceless_born_900steps` and `w25_8gb_last_layer_meas_convex_traceless_900steps`) and **3,600 steps** (`w25_8gb_convex_traceless_born_3600steps`) on genuine `FineWeb-Edu` (`fineweb_train_000001.bin` $\to$ `fineweb_val_000000.bin`).
   - Compare full trajectory checkpoints (`0, 300, 900, 1800, 2700, 3600`), rank-bucketed probability mass, Top-1/5/10/50 accuracies, $\mathcal{R}_{\text{sharp}}$, and throughput (`tok/s`) against `w23` (`5.1479`), `w22` (`5.0688`), `w21` (`5.0211`), `w20` (`5.0858`), `38.28M` Parity-Split `QuantumBornGPT` (`5.1157`), and `Classical` (`5.5202`).

2. **Genuine `ILSVRC ImageNet-1K` (`1,281,167` Train / `50,000` Val / `100,000` Test Images, `1,000` Classes) Quantum Building-Block & Readout Ablation Study (`H3`)**:
   - Download and decode the **genuine ILSVRC ImageNet-1K dataset** (`benjamin-paine/imagenet-1k-32x32`: `1,281,167` training images across all `1,000` official WordNet synsets, `50,000` validation images (`50` per class), and `100,000` test images) into a local `uint8` tensor cache (`.cache/datasets/imagenet1k_32x32/imagenet1k_32x32_uint8.pt`).
   - Execute a systematic **7-Arm Quantum Building-Block & Readout Ablation Study** on genuine `1,000`-class `ImageNet-1K` (streaming from the `1,281,167` training images and evaluating on the official `50,000`-image validation split):
     1. **`imagenet1k_classical_softmax`**: Classical Conv/Patch backbone + Euclidean SwiGLU MLP + unconstrained Linear Softmax readout.
     2. **`imagenet1k_naive_kboson_born` (Readout Ablation 1)**: Full Quantum backbone + Tied $N=2^{10}=1,024$ Wavelet Class Pyramid + legacy concave $k \ln(1 + \tilde{r} c_v)$ readout.
     3. **`imagenet1k_w23_traceless_kboson` (Readout Ablation 2)**: Full Quantum backbone + Last-Layer Measurement + Traceless Observable wrapped in concave $k \ln(1 + \tilde{r} \widehat{O}_v)$.
     4. **`imagenet1k_ablate_no_wavelet_cholesky` (Building-Block Ablation 1 — No Multi-Scale Wavelet-Cholesky V-Cycle)**: Replaces `WaveletCholeskyMultiscaleBranch` (`BlockCholeskyLinear` SPD scale operators + Sweldens lifting wavelets) in the backbone with a parameter-matched Euclidean linear layer while keeping `convex_traceless_born` readout.
     5. **`imagenet1k_ablate_no_mach_zehnder` (Building-Block Ablation 2 — No Mach-Zehnder Two-Beam Interferometer)**: Removes the `MachZehnder` phase-beam interference gate from the quantum blocks while keeping `WaveletCholeskyMultiscaleBranch` and `convex_traceless_born`.
     6. **`imagenet1k_ablate_no_kraus_bath` (Building-Block Ablation 3 — Pure-State Readout Without Stiefel Kraus Bath / Diffuse Beam)**: Sets $\lambda_{\text{diff}} = 0$ (disabling the $R=8$ Stiefel Kraus diffuse measurement channel) to quantify how much multi-class calibration and Top-1/Top-5 accuracy on real ImageNet-1K come from the mixed-state Kraus bath.
     7. **`imagenet1k_full_convex_traceless_born` (Full Quantum Architecture + Solution A)** & **`imagenet1k_full_dyadic_hierarchical_born` (Full Quantum Architecture + Solution B)**: Complete Quantum Wavelet-Cholesky + Mach-Zehnder Interferometer + Stiefel Kraus Bath + Tied $N=2^{10}=1,024$ Wavelet Class Pyramid with `convex_traceless_born` and `dyadic_hierarchical_born`.

---

## 2. Falsifiable Hypotheses (`H1`, `H2`, `H3`)

### Hypothesis 1 (`H1` — 900-Step / `7.37M`-Token `FineWeb-Edu` Ablation of `Convex-Peak Traceless Born`)
- **Claim**: Over `900` steps (`7.37M` tokens, $B=16, T=512$, `<= 8.0 GB` VRAM) on genuine `FineWeb-Edu` (`fineweb_train_000001.bin` $\to$ `fineweb_val_000000.bin`):
  1. **`w25_8gb_convex_traceless_born_900steps` (`use_convex_traceless_born=True`)** and **`w25_8gb_last_layer_meas_convex_traceless_900steps` (`use_last_layer_measurement_operator=True, use_convex_traceless_born=True`)** will **both beat Plan 038's `w23` (`5.6818` Val CE)** and **Plan 037's `w22` (`5.5613` Val CE)** by significant margins while running at **`>= 21,000 tok/s`**.
  2. The best `w25` 900-step arm will beat **`w20` (`5.5253`)**, **`40.78M` `QuantumBornGPT` (`5.6092`)**, and **`53.54M` `ClassicalModdedGPT` (`5.9152`)**, and match or beat **`w21` (`5.4734`)** at **`~2x` higher token throughput**!

### Hypothesis 2 (`H2` — 3,600-Step / `29.49M`-Token `FineWeb-Edu` Scaling Run)
- **Claim**: Over `3,600` steps (`29.49M` tokens, evaluated at steps `0, 300, 900, 1800, 2700, 3600`) on genuine `FineWeb-Edu` under the `<= 8.0 GB` VRAM cap:
  - **`w25_8gb_convex_traceless_born_3600steps` (`10.40M` params)** will achieve **`< 5.0211` Val CE** — beating **`w23` (`5.1479`)**, **`w22` (`5.0688`)**, **`w20` (`5.0858`)**, **`w21` (`5.0211`)**, **`38.28M` Parity-Split `QuantumBornGPT` (`5.1157`)**, and **`30.95M` `ClassicalModdedGPT` (`5.5202`)**, while sustaining **`>= 21,000 tok/s`** and `kraus_stiefel_err <= 0.05`.

### Hypothesis 3 (`H3` — Genuine `ILSVRC ImageNet-1K` [`1,281,167` Train / `50,000` Val / `100,000` Test, `1,000` Classes] Quantum Building-Block & Readout Ablation Study)
- **Claim**: On the genuine `ILSVRC ImageNet-1K` benchmark (`1,281,167` training images, evaluated on the official `50,000`-image validation split across all `1,000` classes):
  1. **Dataset Integrity Gate**: The cached dataset will contain the exact `1,281,167` training images, `50,000` validation images, and `100,000` test images across `1,000` classes (`0..999`).
  2. **Readout Ablation on Real ImageNet-1K**: `imagenet1k_naive_kboson_born` will exhibit severe Cross-Entropy degradation (`>= +0.40` nats worse Val CE than `imagenet1k_classical_softmax`) with massive tail probability leakage into `Ranks > 50`, whereas `imagenet1k_full_convex_traceless_born` and `imagenet1k_full_dyadic_hierarchical_born` will **eliminate the tail leak and beat `imagenet1k_classical_softmax` on both Validation Cross-Entropy and Top-1/Top-5 Accuracy**!
  3. **Quantum Building-Block Ablations on Real ImageNet-1K**: Ablating `WaveletCholeskyMultiscaleBranch` (`no_wavelet_cholesky`), `MachZehnder` two-beam interference (`no_mach_zehnder`), or the Stiefel Kraus diffuse bath (`no_kraus_bath`) will each degrade Validation Cross-Entropy and Top-1 Accuracy relative to `imagenet1k_full_convex_traceless_born`, proving that **each quantum building block contributes positively** on genuine `1,000`-class ImageNet-1K!

---

## 3. Planned Architectural & Code Changes

1. **[`src/qdlblocks/models/wavelet_cholesky_vision.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_vision.py)** (`<= 650` lines, `<= 8` args per function, zero semicolons):
   - Add `load_or_download_imagenet1k_32x32(cache_dir, max_train_samples, max_val_samples, device)` to load the genuine `1,281,167`-train / `50,000`-val / `100,000`-test ILSVRC ImageNet-1K dataset from `.cache/datasets/imagenet1k_32x32/imagenet1k_32x32_uint8.pt` (or decode from the 5 HuggingFace parquet shards if not yet built).
   - Add quantum building-block ablation flags to `VisionWaveletCholeskyConfig`:
     - `use_wavelet_cholesky_branch: bool = True` (vs. parameter-matched Euclidean linear branch when `False`).
     - `use_mach_zehnder_interferometer: bool = True` (adds unitary Mach-Zehnder phase-beam interference gate in `WaveletCholeskyVisionBlock`, disabled when `False`).
     - `use_kraus_diffuse_bath: bool = True` (enables the $R=8$ Stiefel Kraus diffuse measurement channel, disabled when `False`).
2. **[`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py)** (`<= 1,000` lines, `<= 8` args):
   - Ensure `use_convex_traceless_born=True` works seamlessly both standalone (`w25_8gb_convex_traceless_born_*`) and combined with `use_last_layer_measurement_operator=True` (`w25_8gb_last_layer_meas_convex_traceless_900steps`).
3. **Promoted CLI Diagnostic Scripts (`scripts/BACKLOG.md` $\to$ `scripts/`)**:
   - [`scripts/multiclass_ce_margin_decomposer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multiclass_ce_margin_decomposer.py): Decomposes multi-class Cross-Entropy $-\ln P(y \mid x) = \ln(1 + Z_{\text{pos\_runnerup}}/e^{\ell_y} + Z_{\text{neg\_tail}}/e^{\ell_y})$ into target-peak deficit vs. positive-cosine runner-up partition tax vs. negative-cosine tail tax across $K \in \{100, 1000, 50257\}$ classes.
   - [`scripts/dyadic_tree_conditional_calibration_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/dyadic_tree_conditional_calibration_probe.py): Audits coarse-cluster NLL & Top-1 accuracy vs. conditional intra-cluster NLL & Top-1 accuracy and Expected Calibration Error (`ECE`) across multi-scale Wavelet-Cholesky trees.
4. **Verification Script & Unit/Hypothesis Tests**:
   - [`examples/50_verify_plan040_fineweb_3600_and_imagenet1k_ablations.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/50_verify_plan040_fineweb_3600_and_imagenet1k_ablations.py)
   - [`tests/test_plan040_imagenet1k_and_convex_traceless_scaling.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan040_imagenet1k_and_convex_traceless_scaling.py)
   - [`tests/test_hypothesis_040_fineweb_3600_and_imagenet1k_ablations.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_040_fineweb_3600_and_imagenet1k_ablations.py)

---

## 4. Verification & Testing Plan

1. **AST & Best-Practices Gate**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/models/wavelet_cholesky_vision.py src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py scripts/multiclass_ce_margin_decomposer.py scripts/dyadic_tree_conditional_calibration_probe.py examples/50_verify_plan040_fineweb_3600_and_imagenet1k_ablations.py
   ```
2. **Unit & Hypothesis Test Suite**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/pytest tests/test_plan040_imagenet1k_and_convex_traceless_scaling.py tests/test_hypothesis_040_fineweb_3600_and_imagenet1k_ablations.py -v
   ```

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

1. **Existing Scripts Used**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
   - [`scripts/last_layer_measurement_adjoint_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/last_layer_measurement_adjoint_bench.py)
   - [`scripts/traceless_observable_rank_bucket_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/traceless_observable_rank_bucket_probe.py)
2. **`Build-Before-Launch` Scripts (Promoted from `scripts/BACKLOG.md`)**:
   - [`scripts/multiclass_ce_margin_decomposer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/multiclass_ce_margin_decomposer.py)
   - [`scripts/dyadic_tree_conditional_calibration_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/dyadic_tree_conditional_calibration_probe.py)
3. **`Nice-to-Have` Scripts (Logged in `scripts/BACKLOG.md` & `scripts/TOOLING_BACKLOG.md`)**:
   - `scripts/imagenet1k_quantum_block_ablation_profiler.py`: Standalone CLI profiler that loads trained `WaveletCholeskyVisionClassifier` checkpoints on ImageNet-1K and CIFAR-100 and exports a component-level ablation scorecard (parameter cost, throughput impact, and NLL delta per quantum block).
   - `scripts/fineweb_convex_vs_concave_slope_tracker.py`: Tracks empirical positive-peak slope $\partial \ell_{t,y_t}/\partial c_{t,y_t}$ vs. negative-tail slope across training checkpoints (`0..3600`) on FineWeb-Edu.
