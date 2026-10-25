# QuantumDLBlocks Repository & Script Engineering Best Practices (`docs/SCRIPT_BEST_PRACTICES.md`)

## 1. Motivation & Root-Cause Postmortem (Plan 033)

An engineering audit of the repository identified three anti-patterns that accumulated across iterative research waves and directly degraded agent reliability, code readability, and local-to-cloud reproducibility:

1. **Monolithic Multi-Thousand-Line Scripts**:
   - [`examples/11_fineweb_100m_multiepoch_gpu_tpu.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py) grew to **`3,851` lines**, with a single `2,100`-line function body mixing CLI parsing, distributed shard streaming, model construction, optimizer routing, step-level telemetry, and SQLite/Model-Card serialization.
   - Per-plan verification scripts (`examples/35_*.py` through [`examples/42_verify_plan032_clean_full_ambient_and_gentle_qubit_lr.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/42_verify_plan032_clean_full_ambient_and_gentle_qubit_lr.py)) averaged **`1,300–1,657` lines** each because they duplicated helper routines instead of importing shared builders from `src/qdlblocks/benchmarks/`.
2. **100-Argument Function Signatures & Boolean Flag Jungles**:
   - [`run_cuda_benchmark`](file:///home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py#L752-L854) accumulated **`101` named arguments**, and [`QuantumBornGPT.__init__`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/born_gpt.py#L47-L141) accumulated **`93` arguments**.
   - Passing dozens of flat boolean and scalar flags (`use_overcomplete_frame`, `use_overcomplete_two_beam`, `use_sparse_64qubit`, `use_hierarchical_16qubit`, `use_last_layer_softmax`, `use_two_beam_kboson_readout`, `use_bipolar_kboson`, etc.) forces callers and agents to track implicit mutual exclusions across thousands of lines.
3. **Silent Local-vs-Cloud Drift from Copy-Pasted Builders**:
   - Because `examples/42_*.py` (local `8 GB` RTX 5060 Ti verification) and `examples/11_*.py` (remote Colab G4 runner) maintained separate copy-pasted optimizer and initialization blocks, three critical discrepancies slipped into Wave 17:
     1. `examples/42_*.py` routed `value_embeds`, `unigram_qubits`, and `higher_states` into the **Gentle `2x` Zero-Weight-Decay Qubit Embedding `AdamW` group (`lr = 0.006, weight_decay = 0.0`)**, whereas `examples/11_*.py` omitted `value_embeds` (letting it fall into `weight_decay = 0.01`) and set `sparse_lr = 0.010` (`2.5 * 0.004`).
     2. `examples/42_*.py` initialized `model.povm_log_prior` via **Bose-saturated empirical log-unigram frequencies (`alpha = 0.62`, top-16 knee saturation)**, whereas `examples/11_*.py` did not call `initialize_povm_log_prior`.

To prevent these failure modes permanently, all new and refactored code in `QuantumDLBlocks` MUST obey **Rules 1–8** and the **Graceful Experiment Wrap-Up Protocol** below.

---

## 2. The Eight Mandatory Engineering Rules

### Rule 1: Comment-Free Code-Length Budget (`<= 1,000` SLOC Soft Warning Limit, `<= 2,500` SLOC Hard Limit, Comments & Docstrings Are Free)
- **Comments & Docstrings Do NOT Count Against Line Limits**: [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) (`count_code_and_doc_lines`) strips all `#` comments, module/class/function docstrings (`"""..."""`), and blank lines before evaluating file and nested-function line limits. **Never skimp on comments or docstrings to save lines.**
- **Soft Limit (`<= 1,000` executable code lines / SLOC)**: Every script or module in `examples/`, `scripts/`, and `src/qdlblocks/` should aim for **`<= 1,000` executable code lines (SLOC)**. Exceeding `1,000` SLOC emits a soft-limit warning in `scripts/ast_signature_and_length_linter.py` (status `WARN (PASS)`).
- **Hard Limit (`<= 2,500` executable code lines / SLOC)**: No Python file in `examples/`, `scripts/`, or `src/qdlblocks/` may exceed **`2,500` SLOC** (`FAIL` in `scripts/ast_signature_and_length_linter.py`).
- **No Artificial Line-Packing**: Never pack multiple statements onto a single line with semicolons (`a = 1; b = 2`).
- **How to Comply**: Keep experiment modules focused (Rule 6) and extract reusable domain logic, data loaders, math kernels, or telemetry formatters into dedicated modules under `src/qdlblocks/benchmarks/` or `src/qdlblocks/layers/`.

### Rule 2: Function Signature Ceiling (`<= 8` Arguments) via Frozen `@dataclass` Configs
- **Rule**: No new function, method, or CLI entrypoint may accept more than **`8` positional or keyword arguments** (excluding `self`/`cls`).
- **How to Comply**: Group coherent domain parameters into frozen `@dataclass(frozen=True)` configuration objects defined in [`src/qdlblocks/benchmarks/config_models.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py):
  - [`BackboneConfig`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L25-L39): `num_layers`, `embed_dim`, `seed_dim`, `num_heads`, `frame_expansion`, `block_size`, `window_size`, `num_value_embeddings`, `ngram_vocab_size`, `ngram_inject_scale`.
  - [`QubitHierarchyConfig`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L41-L52): `qubit_registers`, `qubit_amplitudes`, `num_higher_states`, `use_ortho_decouple`, `decouple_self_in_full_ambient`, `directed_predecessor_mix`, `ppmi_alpha`.
  - [`KBosonReadoutConfig`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L54-L70): `diffuse_rank`, `diffuse_schedule`, `lambda_diff_start`, `lambda_diff_end`, `lambda_diff_min`, `lambda_diff_max`, `contextual_purity_gain`, `quadratic_diffuse_fidelity`, `kraus_spectral_floor`, `use_stereographic_lift`, `use_cosine_phase_beam_gate`, `povm_temperature`.
  - [`OptimizerConfig`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L72-L83): `muon_lr`, `body_adam_lr`, `qubit_embed_lr`, `head_lr`, `ngram_lr`, `stiefel_penalty`, `retract_every`.
  - [`QuantumExperimentSpec`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L85-L98): Composes `BackboneConfig`, `QubitHierarchyConfig`, `KBosonReadoutConfig`, and `OptimizerConfig` with run metadata (`key`, `label`, `short_label`, `steps`, `color`).

### Rule 3: Strategy / Polymorphic Dispatch Over Boolean Flag Jungles
- **Rule**: Never add mutually exclusive boolean flags (`use_mode_a=False, use_mode_b=False, use_mode_c=True`) that select `if / elif / else` branches deep inside `forward()` or a training loop.
- **How to Comply**:
  - Use explicit `nn.Module` strategy classes or `Literal` / `Enum` mode selectors paired with focused sub-config dataclasses.
  - Resolve architectural dispatch **once at construction time** inside a builder function (such as [`build_quantum_model_from_spec`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L124-L227)) rather than branching on 20 boolean attributes on every forward pass.

### Rule 4: Single Shared Builders Between Local `8 GB` Scripts & Cloud Runners
- **Rule**: Never copy-paste model initialization, log-prior calibration, optimizer parameter-group splitting, learning-rate schedules, or validation rank-bucket decomposition into `examples/XX_*.py` or `scripts/*.py`.
- **How to Comply**: Always import the canonical builders from `src/qdlblocks/benchmarks/`:
  - [`initialize_povm_log_prior`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L100-L122): Bose-saturated empirical log-unigram prior initialization.
  - [`build_quantum_model_from_spec`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L124-L227): Full `QuantumBornGPT` instantiation + Step-0 PPMI-SVD initialization + Bose-saturated log-prior + `polar_snap_weights` + Kraus Stiefel retraction.
  - [`build_quantum_born_optimizers`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L230-L328): Canonical `StiefelMuon` + 4-group `AdamW` builder (`body`, `qubit_embed` at `lr=0.006, wd=0.0` including `unigram_qubits`, `higher_states`, and `value_embeds`, `readout_head` at `lr=0.012, wd=0.0`, and `ngram` at `lr=0.025, wd=0.0`).
  - [`compute_wsd_lr_multiplier`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py#L331-L345): Warmup-Stable-Decay multiplier.
  - [`compute_calibration_rank_decomposition`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/calibration_probe.py#L36-L211) & [`compute_layerwise_dynamics_profile`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/calibration_probe.py#L260-L372).

### Rule 5: No Giant Closure-Capturing Nested Functions
- **Rule**: Do not define nested helper functions (`def _eval(): ...`, `def _step(): ...`) inside a `500+`-line outer function that implicitly capture 30 local variables from the enclosing scope.
- **How to Comply**: Keep all helper functions **top-level**, **pure** (or taking explicit state/config dataclasses with `<= 8` arguments), and **independently importable and unit-testable** via `pytest`.

### Rule 6: Clean, Flag-Free Per-Experiment Modules Over Boolean-Flag Accumulation in Legacy Models
- **Rule**: When an experiment wave introduces a simplified or structurally distinct mathematical formulation (for example, `Wave 28` replacing dynamic `PurityGate` + `kboson_raw_k` + manual temperature clamps with analytical Parseval $\mu\text{P}$ scaling $\beta_0(d) = c^*\sqrt{d}$), **do NOT thread new `use_*` / `is_*` boolean toggle flags into existing model or loss files** (such as [`wavelet_cholesky_2pow16_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py) or [`direct_chunked_ce.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/direct_chunked_ce.py)).
- **Why**: Accumulating historical `use_*` flags inside a single class or loss kernel forces readers to trace dead legacy branches, hides the true mathematical equation being tested behind nested `if / else` blocks, and pushes core files toward the `1,000`-line ceiling.
- **How to Comply (Composition of Shared Primitives + Dedicated Flag-Free Experiment Modules)**:
  1. **Share Stable Building-Block Primitives by Import**: Reusable, representation-agnostic building blocks (`BlockCholeskyLinear`, `WaveletCholeskyMultiscaleBranch`, `OvercompleteTwoBeamInterference`, `SubspaceProjectorBornAttention`, `StiefelMuon`, `build_dyadic_2pow16_spectral_tree`) live in `src/qdlblocks/layers/` and `src/qdlblocks/optim/` and are imported directly without modification.
  2. **Write a Clean, Direct, Flag-Free Module for the New Architecture**: Implement the new mathematical formulation in its own self-contained, straight-line module (e.g., `src/qdlblocks/models/parseval_mup_dyadic_gpt.py` and `src/qdlblocks/layers/parseval_mup_chunked_ce.py`, `~200–350` lines each) where `forward()` executes **only** the equation of that architecture—with zero `use_*` boolean flags, zero `if is_convex:` branches, and zero `Optional` legacy modules (`self.purity_gate = None`).
  3. **Freeze Previous Wave Models for Reproducible Baseline Comparisons**: Previous wave models remain untouched and frozen so cross-wave comparisons can import and evaluate both classes side-by-side in a clean verification script without risk of breaking historical baselines.

### Rule 7: Top-of-File Named Constants & Mandatory `EXPERIMENT_PLAN_0XX.md` Synchronization
- **Rule**: All numerical constants, scaling coefficients, damping exponents, thresholds, and empirical calibration factors MUST be defined as documented `UPPER_SNAKE_CASE` constants at the top of the file—never as unexplained inline "magic numbers" (such as `0.705 + 0.330 * progress` or `0.12 * w_c`) inside functions or methods.
- **Mandatory Plan Synchronization**: Whenever any new constant is introduced or calibrated while implementing an `EXPERIMENT_PLAN_0XX.md`:
  1. Define it at the top of the Python module with a comment stating its mathematical role and how it was derived/calibrated.
  2. **Update the corresponding `EXPERIMENT_PLAN_0XX.md`** (in a dedicated **Constants & Calibration Ledger** table) and [`docs/BUILDING_BLOCKS.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/BUILDING_BLOCKS.md) (if inside a reusable building block) so that code and specification never drift apart.

### Rule 8: Thorough Building-Block Commenting & Experiment Provenance
- **Rule**: Because comments and docstrings do **not** count against line limits (Rule 1), every reusable building block in `src/qdlblocks/{kernels,layers,models,optim,benchmarks}/` MUST be commented thoroughly enough for an independent, zero-context reviewer to understand:
  1. **When to Use It**: The specific regime or bottleneck where the building block is useful (e.g., $V \gg d$ vocabulary readout, multi-scale channel conditioning, or $M$-subspace anti-collapse).
  2. **Why It Was Built (Mathematical Rationale)**: The failure mode it resolves and the governing equation or invariance law it enforces.
  3. **Experiment Provenance (`Wave XX` / `EXPERIMENT_PLAN_0XX.md`)**: Explicit citation of the experiment plan and wave that created and validated the building block, along with its key quantitative finding.

---

## 3. Concrete Before vs. After Examples (`src/qdlblocks/benchmarks/config_models.py`)

### 3.1 Model Construction & Step-0 Initialization

#### Before (Anti-Pattern: 45+ Flat Keyword Arguments & Copy-Pasted Init Sequence)
```python
# ANTI-PATTERN: Repeated across examples/38_*.py .. examples/42_*.py (80+ lines per variant)
model = QuantumBornGPT(
    vocab_size=50304,
    embed_dim=256,
    seed_dim=256,
    num_heads=4,
    num_layers=4,
    block_size=512,
    readout_mode="asymmetric",
    povm_temperature=0.030,
    use_overcomplete_two_beam=True,
    frame_expansion=2.0,
    use_purified_attn=True,
    true_layer0_v_residual=True,
    use_gated_spherical_residual=True,
    use_dynamic_povm_beta=True,
    use_povm_log_prior=True,
    use_ortho_decouple=False,
    use_hierarchical_16qubit=True,
    qubit_registers=16,
    qubit_amplitudes=16,
    num_higher_states=128,
    use_block_hierarchy_bridge=True,
    bridge_causal_memory=False,
    use_last_layer_softmax=False,
    logit_softcap=15.0,
    use_two_beam_kboson_readout=True,
    diffuse_rank=16,
    diffuse_schedule="warmup_hold_anneal",
    use_stereographic_lift=True,
    use_cosine_phase_beam_gate=True,
    quadratic_diffuse_fidelity=0.50,
    kraus_spectral_floor=6.5,
    use_bipolar_kboson=False,
    use_unet_interferometry=True,
    unet_skip_mode="scalar",
    num_value_embeddings=2,
    zero_half_rope=True,
    window_size=256,
    use_ngram_hash=True,
    ngram_vocab_size=65536,
    ngram_inject_scale=0.08,
    use_mudd_and_gates=True,
    use_smear_gate=True,
    use_depth_scaled_residual=True,
).to(device)
# Followed by 35 lines of manual PPMI-SVD init, manual log-prior bincount, and polar_snap_weights...
```

#### After (Canonical Pattern: Composable `@dataclass` Spec + 5-Argument Shared Builder)
```python
from qdlblocks.benchmarks.config_models import (
    BackboneConfig,
    KBosonReadoutConfig,
    OptimizerConfig,
    QuantumExperimentSpec,
    QubitHierarchyConfig,
    build_quantum_born_optimizers,
    build_quantum_model_from_spec,
)

spec = QuantumExperimentSpec(
    key="w18_8gb_L4_d256_janti_soft_floor_3600steps",
    label="L=4, d=256 Full-Ambient 16x16 J_anti + Soft Spectral Floor (6.5)",
    short_label="L4-d256-Janti-Floor6.5",
    steps=3600,
    backbone=BackboneConfig(num_layers=4, embed_dim=256, seed_dim=256, num_heads=4, frame_expansion=2.0),
    hierarchy=QubitHierarchyConfig(
        qubit_registers=16,
        qubit_amplitudes=16,
        num_higher_states=128,
        use_ortho_decouple=False,
        decouple_self_in_full_ambient=True,
    ),
    readout=KBosonReadoutConfig(quadratic_diffuse_fidelity=0.50, kraus_spectral_floor=6.5),
    optimizer=OptimizerConfig(qubit_embed_lr=0.006),
)

model, ppmi_stats = build_quantum_model_from_spec(
    spec=spec,
    ppmi_embeddings=ppmi_emb_256,
    init_tokens=train_tokens[:2_000_000],
    prior_tokens=prior_tokens_t,
    device=device,
)
muon_opt, adam_opt, base_muon_lr, base_adam_lr = build_quantum_born_optimizers(model, spec.optimizer)
```

### 3.2 Clean, Flag-Free Per-Experiment Modules vs. Boolean-Flag Accumulation (Rule 6)

#### Before (Anti-Pattern: Threading New `use_*` Flags Into Legacy Model & Loss Files)
```python
# ANTI-PATTERN: Adding boolean toggles and conditional dead-code branches to legacy modules
class Radical2Pow16WaveletCholeskyGPT(nn.Module):
    def __init__(self, cfg: Radical2Pow16Config) -> None:
        if cfg.use_analytical_parseval_beta:
            self.purity_gate = None
            self.kboson_raw_k = None
        else:
            self.purity_gate = ContextualBlochPurityGate(cfg.embed_dim)
            self.kboson_raw_k = nn.Parameter(torch.tensor(3.4657))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> ...:
        if self.cfg.use_analytical_parseval_beta:
            beta_t = self.cfg.parseval_beta_scale * math.sqrt(self.cfg.embed_dim)
            k_copies = 1.0
        else:
            beta_t = self.purity_gate(psi_t)
            if self.cfg.embed_dim > 256:
                beta_t = torch.clamp(beta_t, max=44.0)
            else:
                beta_t = torch.clamp(beta_t, max=26.8)
            k_copies = 2.0 + F.softplus(self.kboson_raw_k)
```

#### After (Canonical Rule 6 Pattern: Clean, Flag-Free Per-Wave Module Importing Shared Primitives)
```python
# CANONICAL PATTERN: src/qdlblocks/models/parseval_mup_dyadic_gpt.py
# Imports shared building-block primitives; contains ZERO legacy modules and ZERO boolean flags.
from qdlblocks.layers.parseval_mup_chunked_ce import parseval_mup_dyadic_chunked_ce
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    RadicalWaveletCholeskyBlock,
    Tied2Pow16WaveletCholeskyPyramid,
    WaveletCholeskyUNetBlock,
)

class ParsevalMuPDyadicGPT(nn.Module):
    """Flag-free Wave 28 GPT with analytical Parseval muP scaling beta_0(d) = c^* sqrt(d)."""

    def __init__(self, cfg: ParsevalMuPDyadicConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.beta_0: float = cfg.mup_c_star * math.sqrt(float(cfg.embed_dim))
        self.pyramid = Tied2Pow16WaveletCholeskyPyramid(...)
        # Straight-line forward pass with no PurityGate, no kboson_raw_k, and no manual clamps.
```

---

## 4. Automated AST Verification in CI / Pytest

Every new module and CLI utility is verified via [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) and `pytest` to enforce:
1. Executable code line count (SLOC, excluding comments and docstrings) `<= 1,000` soft limit (`<= 2,500` hard ceiling) with zero semicolon statement packing.
2. Every `ast.FunctionDef` and `ast.AsyncFunctionDef` has `<= 8` parameters (excluding `self`/`cls`).
3. New architectural waves are implemented in clean, flag-free modules (Rule 6) with documented top-of-file constants (Rule 7) and thorough provenance docstrings (Rule 8).

---

## 5. Implementer Context-Budget & Graceful Experiment Wrap-Up Protocol

When implementing and evaluating an `EXPERIMENT_PLAN_0XX.md`, agents and subagents must balance **autonomous error-fixing** against **context-window discipline**:

1. **Code Implementation Success $\ne$ Mandatory Full Training Evaluation Pass**:
   - Successfully implementing the planned architecture, passing unit tests (`pytest`), and passing `scripts/ast_signature_and_length_linter.py` is a complete engineering deliverable even if a full training evaluation does not immediately beat every empirical baseline.
2. **Proactive Debugging Within Bounded Context (`<= 2–3` Training Retries)**:
   - Agents are encouraged to fix runtime errors, shape mismatches, or initialization bugs autonomously without stopping prematurely to ask for instructions.
   - However, if diagnosing or tuning a full training evaluation begins to consume a large fraction of the agent's context window (or requires more than `2–3` GPU training retries), **do NOT loop endlessly on hyperparameter/training retries until context exhaustion**.
3. **Graceful Experiment Wrap-Up & Post-Mortem Documentation**:
   - Instead of exhausting the context window on repeated training runs, wrap up the experiment cleanly by adding a **Section 6: Experiment Outcomes, Constants Ledger, & Post-Mortem Analysis** to `EXPERIMENT_PLAN_0XX.md` (and syncing logs to the WebUI via `scripts/sync_waves_to_webui.py`) that records:
     - **What Succeeded**: Analytical/CPU probes, mathematical invariants, unit tests, and throughput measurements.
     - **Empirical Training Results Observed**: Exact metrics (`val_ce`, `top1_acc`, `R_sharp`, `kraus_stiefel_err`) from the runs executed.
     - **Root-Cause Diagnosis & Next-Step Hypotheses**: Concrete mathematical or architectural hypotheses explaining why the training run behaved as it did and what specific change the next experiment should test.

