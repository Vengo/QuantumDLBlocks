# Experiment Plan 036: Fused Kronecker-U-Net $N = 2^{16}$ Wavelet-Cholesky Architecture, Dyadic Predecessor Bigram Coupling, & Spectral/Cholesky Diagnostic Tooling

## 1. Context & Motivation

1. **Lessons from Plan 035 (`8.99M` Standalone $N = 2^{16}$ Architectures) & Wave 19 Post-Mortem**:
   - In [`EXPERIMENT_PLAN_035.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_035.md), our two standalone $N = 2^{16} = 65,536$ Wavelet-Cholesky architectures (`multiscale_unet` and `consistent_kronecker`, each only **`8.99M` parameters** with zero separate $V \times d$ input/output tables) both beat the `40.78M`-parameter `QuantumBornGPT` (`5.6092`) and `53.54M`-parameter `ClassicalModdedGPT` (`5.9152`) at `900` steps (`7.37M` tokens), reaching **`5.5253` Val CE (`21.44%` Top-1)** and **`5.5426` Val CE (`21.31%` Top-1)** in only `3.52–3.58 GB` VRAM.
   - Over `3,600` steps (`29.49M` tokens), `multiscale_unet` (`8.99M` params) reached **`5.0858` Val CE (`24.82%` Top-1)** — beating the `30.95M` `ClassicalModdedGPT` (`5.5202`) by **`-0.4344 nats`** and beating the `38.28M` Parity-Split `QuantumBornGPT` (`5.1157`), while trailing the `40.78M`–`40.82M` Full-Ambient `QuantumBornGPT` (`4.9565` / `4.9378`) by `0.129–0.148 nats`.
   - Meanwhile, the `98.3M`-token slice of Wave 19 (`w19_900m_qborn_d1024_wavelet_cholesky_mach_zehnder_ON_L8`) confirmed that Wavelet-Cholesky + Mach-Zehnder beats standard `4x` TwoBeam on Top-1 accuracy (`35.15%` vs. `34.78%` at `75M` tokens) and subsystem entanglement ($S(\rho_A) = 2.8783$ vs. $2.8305$) while saving `172.7M` parameters.
2. **Root Causes of the Remaining `0.13 nats` Gap Between `8.99M` Standalone $N = 2^{16}$ and `40.82M` `QuantumBornGPT`**:
   - **Gap 1 — Uninitialized $J_{\text{anti}}$ Readout vs. Directed Predecessor Bigram Alignment**: In Plan 035, `synthesize_vocab_detectors()` applied a pure symplectic swap $J_{\text{anti}}(a, b) = (b, -a)$ at Step 0 (`coarse_transition = 0`), giving `0.0000` self-overlap but zero Step-0 alignment between predecessor token $\psi_{\text{in}}(u)$ and successor detector $m_{\text{sharp}}(v)$ for empirical bigrams $(u \to v)$. Initializing the $N = 2^{16}$ pyramid's coarse (`256 x 256`) and mid (`2048 x d`) readout transition operators from empirical predecessor bigram statistics (orthogonalized against $J_{\text{anti}}\phi_{\text{leaf}}(v)$ to preserve low self-overlap) directly injects directed bigram transition geometry into the tied $N = 2^{16}$ pyramid at Step 0.
   - **Gap 2 — Missing Tied $N = 2^{16}$ Dyadic Bigram Phase Gate & Tied Value-Residual Anchor**: `QuantumBornGPT` (`40.82M`) uses a `65,536`-slot N-gram hash table (`8.39M` params) and `2` dense `value_embeds` tables (`25.76M` params). In our $N = 2^{16}$ architecture, the ambient space **is already $65,536$-dimensional**! We can hash adjacent 16-bit dyadic leaf addresses $(p_{t-1}, p_t) \pmod{2^{16}}$ directly into the **already-allocated** `wav_65536` (`65,536 x 64`) and `wav_8192` (`8,192 x 128`) wavelet detail tables via a compact `BlockCholeskyLinear` adapter, and synthesize token-tied attention value anchors $v_{\text{anchor}}(idx)$ from the same tied $N = 2^{16}$ pyramid with **zero** new $V \times d$ tables!
   - **Gap 3 — Separate Arch 1 (`multiscale_unet`) vs. Arch 2 (`consistent_kronecker`) Instead of Fused Kronecker-U-Net**: Arch 1 had fast causal subband token-shift mixing in `d/8` and `d/4`, while Arch 2 had Coarse-Half $\leftrightarrow$ Fine-Half Mach-Zehnder $SO(2)$ beamsplitters and multiplicative Kronecker cross-modulation. Fusing both into **`FusedKroneckerWaveletUNetBlock` (`arch_mode="fused_kronecker_unet"`)** with outer U-Net encoder-to-decoder spherical skip connections (`Layer 0 -> Layer 3`, `Layer 1 -> Layer 2`) and `2x` Mach-Zehnder + Wavelet-Cholesky interference keeps the total model under **`11.5M` parameters** (`< 4.0 GB` VRAM) while combining multi-scale causal subband memory with Coarse$\leftrightarrow$Fine Kronecker entanglement.

---

## 2. Falsifiable Hypotheses (`H1`, `H2`, `H3`)

### Hypothesis 1 (`H1` — Tied $N = 2^{16}$ Predecessor-Bigram Readout Alignment, Tied Dyadic Bigram Phase Gate, & `< 12.5M` Parameter Budget)
- **Claim**: Upgrading [`Tied2Pow16WaveletCholeskyPyramid`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py) and [`Radical2Pow16WaveletCholeskyGPT`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py) with:
  1. **Step-0 Empirical Predecessor-Bigram Wavelet Readout Alignment** (`coarse_transition` + `mid_predecessor_bias` initialized from FineWeb bigram transitions $(u \to v)$ and projected orthogonal to $\psi_{\text{in}}(v)$),
  2. **Tied $N = 2^{16}$ Dyadic Bigram Wavelet Phase Gate** (`embed_dyadic_bigram_phase`, reusing `wav_65536` and `wav_8192` without adding any $V \times d$ table),
  3. **Tied $N = 2^{16}$ Value-Residual Cholesky Bridge** (`val_chol_bridge`), and
  4. **`FusedKroneckerWaveletUNetBlock` (`arch_mode="fused_kronecker_unet"`)** with outer U-Net spherical skip connections
  will:
  - Keep total trainable parameters strictly **`< 12,500,000` (`< 12.5M`, `> 3.3x` smaller than `40.78M` `QuantumBornGPT`)**,
  - Maintain Step-0 self-overlap $\max_v |\langle \psi_{\text{in}}(v) \mid m_{\text{sharp}}(v)\rangle| \le 0.05$,
  - Boost Step-0 **Directed Bigram Overlap Ratio** $\mathbb{E}_{(u,v)\in\text{Bigram}}[|\langle \psi_{\text{in}}(u) \mid m_{\text{sharp}}(v)\rangle|^2] / \mathbb{E}_{\text{rand}}[|\langle \psi_{\text{in}}(u) \mid m_{\text{sharp}}(v)\rangle|^2]$ to **`>= 1.80x`** (vs. `~1.00x` for unaligned $J_{\text{anti}}$), and
  - Satisfy exact causal masking (`max_future_leakage == 0.0`) and finite gradients across all parameters.

### Hypothesis 2 (`H2` — 900-Step / `7.37M`-Token Local `8 GB` Ablation: `fused_kronecker_unet` vs. Plan 035 Arch 1 & Arch 2 and `40.78M` `QuantumBornGPT`)
- **Claim**: Over `900` steps (`7.37M` tokens, $B=16, T=512$, `<= 8.0 GB` VRAM) on genuine FineWeb-Edu (`fineweb_train_000001.bin` $\to$ `fineweb_val_000000.bin`):
  - **`w21_8gb_arch3_fused_kronecker_unet_2pow16_900steps` (`arch_mode="fused_kronecker_unet"`, `< 12.5M` params)** will achieve a lower (better) validation cross-entropy than **both** Plan 035 standalone architectures (`w20_8gb_arch1_2pow16_multiscale_wavelet_cholesky_unet_900steps` at `5.5253` and `w20_8gb_arch2_consistent_2pow16_kronecker_wavelet_cholesky_900steps` at `5.5426`), as well as `40.78M` `QuantumBornGPT` (`5.6092`) and `53.54M` `ClassicalModdedGPT` (`5.9152`), reaching **`<= 5.45` Val CE** at Step 900.

### Hypothesis 3 (`H3` — 3,600-Step / `29.49M`-Token Extended Scaling Run: Breaking the `5.00` Val CE Barrier Under `< 12.5M` Parameters)
- **Claim**: Over `3,600` steps (`29.49M` tokens, evaluated at steps `0, 300, 900, 1800, 2700, 3600`) on genuine FineWeb-Edu under the `<= 8.0 GB` VRAM cap:
  - **`w21_8gb_fused_kronecker_unet_2pow16_3600steps` (`< 12.5M` params)** will break the **`5.00` Val CE barrier (`< 5.00` Val CE at Step 3600)** — improving over Plan 035's `8.99M` `multiscale_unet` (`5.0858` Val CE) at every checkpoint (`300, 900, 1800, 2700, 3600`), beating the `30.95M` `ClassicalModdedGPT` (`5.5202` Val CE) by **`> 0.52 nats`**, and closing to within `< 0.05 nats` (or beating!) the `40.78M`–`40.82M` `QuantumBornGPT` (`4.9565` / `4.9378` Val CE) while using **`> 3.5x` fewer parameters** and **`< 4.5 GB` VRAM**.

---

## 3. Planned Architectural & Code Changes

1. **[`src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py)**:
   - Extend `Radical2Pow16Config.arch_mode` to `Literal["multiscale_unet", "consistent_kronecker", "fused_kronecker_unet"]` and add optional configuration fields (`use_dyadic_bigram_gate: bool = True`, `use_tied_value_anchor: bool = True`, `use_outer_unet_skips: bool = True`, `directed_predecessor_mix: float = 0.38`, `interferometer_expansion: float = 2.0` for `fused_kronecker_unet`).
   - Add **`Tied2Pow16WaveletCholeskyPyramid.embed_dyadic_bigram_phase(idx)`** and **`synthesize_value_anchors(idx)`** reusing the tied $N = 2^{16}$ wavelet tables (`wav_65536`, `wav_8192`, `wav_2048`, `wav_256`) with compact `BlockCholeskyLinear` adapters (`bigram_chol_64`, `val_chol_bridge`).
   - Add **`FusedKroneckerWaveletUNetBlock`** combining 3-level wavelet downsampling, causal sequence token-shift Cholesky mixing (`coarse_causal_chol`, `mid_causal_chol`), Coarse-Half $\leftrightarrow$ Fine-Half Mach-Zehnder $SO(2)$ beamsplitters across scales, multiplicative Kronecker cross-modulation, and `OvercompleteTwoBeamInterference` (`frame_expansion=cfg.interferometer_expansion`, `use_mach_zehnder_beamsplitter=True`, `use_wavelet_cholesky_branch=True`).
   - Add outer U-Net encoder-to-decoder spherical skip connections in `Radical2Pow16WaveletCholeskyGPT.forward` and empirical predecessor-bigram readout alignment in `initialize_radical_2pow16_with_ppmi_svd`.
2. **Promoted CLI Diagnostic Scripts (`scripts/`)**:
   - [`scripts/dyadic_spectral_tree_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/dyadic_spectral_tree_inspector.py): Audits multi-resolution intra-cluster cosine similarity across all dyadic scales (`2, 4, 8, 16, 32, 64, 128, 256` leaves per node), inspects top semantic sibling pairs `(2k, 2k+1)` and coarse `256`-cluster neighborhoods, and exports a JSON/Markdown report.
   - [`scripts/cholesky_band_sparsity_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/cholesky_band_sparsity_profiler.py): Audits all `BlockCholeskyLinear` operators across `Radical2Pow16WaveletCholeskyGPT` (`chol_256`, `chol_2048`, `chol_8192`, `chol_65536`, `readout_chol_bridge`, subband Cholesky blocks) for diagonal positivity, SPD condition number $\kappa(L_s L_s^\top)$, off-diagonal Frobenius sparsity/energy, and GPU/CPU throughput.
3. **Verification Script & Tests**:
   - [`examples/46_verify_plan036_fused_kronecker_unet_2pow16.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/46_verify_plan036_fused_kronecker_unet_2pow16.py): Executes `H1`, `H2` (`900` steps), and `H3` (`3,600` steps) on genuine FineWeb-Edu (`<= 8.0 GB` VRAM), saves `.cache/hypothesis_experiments_036.json`, `.cache/live_logs/w21_*.json`, `.cache/model_cards/w21_*.json`, and updates `.cache/benchmarks.sqlite`.
   - [`tests/test_plan036_fused_kronecker_unet_and_tooling.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_plan036_fused_kronecker_unet_and_tooling.py) & [`tests/test_hypothesis_036_fused_kronecker_unet_2pow16.py`](file:///home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_036_fused_kronecker_unet_2pow16.py).

---

## 4. Verification & Testing Plan

1. **AST & Best-Practices Gate**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py scripts/dyadic_spectral_tree_inspector.py scripts/cholesky_band_sparsity_profiler.py examples/46_verify_plan036_fused_kronecker_unet_2pow16.py
   ```
2. **Unit & Hypothesis Test Suite**:
   ```bash
   /home/festivity/agy/QuantumDLBlocks/.venv/bin/pytest tests/test_plan036_fused_kronecker_unet_and_tooling.py tests/test_hypothesis_036_fused_kronecker_unet_2pow16.py -v
   ```

---

## 5. Operational Tooling & Script Opportunities (`AGENTS.md` Section 3 Gate)

1. **Existing Scripts Used**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
   - [`scripts/vocab_dark_ancilla_utilization_audit.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/vocab_dark_ancilla_utilization_audit.py)
   - [`scripts/mera_entanglement_cone_visualizer.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/mera_entanglement_cone_visualizer.py)
2. **`Build-Before-Launch` Scripts (Promoted from `scripts/BACKLOG.md`)**:
   - [`scripts/dyadic_spectral_tree_inspector.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/dyadic_spectral_tree_inspector.py)
   - [`scripts/cholesky_band_sparsity_profiler.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/cholesky_band_sparsity_profiler.py)
3. **`Nice-to-Have` Scripts (Logged in `scripts/BACKLOG.md`)**:
   - `scripts/dyadic_bigram_transition_heatmap.py`: Exports the $256 \times 256$ coarse dyadic cluster transition matrix $T_{\text{coarse}}$ and top predecessor-to-successor semantic cluster flows from a trained `Tied2Pow16WaveletCholeskyPyramid`.
   - `scripts/wavelet_scale_ablation_probe.py`: Evaluates zero-shot validation CE sensitivity when selectively zeroing individual wavelet scales (`256`, `2048`, `8192`, `65536`) at inference time to quantify how much NLL reduction comes from each dyadic resolution level.
