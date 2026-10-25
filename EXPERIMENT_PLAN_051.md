# Experiment Plan 051 (`Wave 36`): Pure-Neural Head-to-Head Ablation — Differentiable Surface N-Gram Wavefunction (`Option 1`) vs. Pure Wavelet-Cholesky + Multi-Rate Conv (`Option 2`) Without Static Kneser-Ney Lookup (`_kn_state = None`)

## 1. Context & Motivation
In `Wave 35` ([`EXPERIMENT_PLAN_050.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_050.md)), cancelling the `900M`-token local run at step `10,200` (`83.56M` tokens) exposed a critical architectural confounder in [`HashedSurfaceNGramProjector`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py#L183-L281):
- `HashedSurfaceNGramProjector` bundled two very different mechanisms:
  1. **Mechanism A — Differentiable Wavefunction N-Gram Embedding (`HashedSurfaceNGramProjector.forward`)**: Three `(65536, 64)` embedding tables (`unigram_table`, `bigram_table`, `trigram_table`, `12.65M` params) lifted onto $\mathbb{S}^{d-1}$ via a Stiefel matrix `u_ngram` (`(64, d)`) and added as `in_delta` and `readout_delta` during both training and evaluation.
  2. **Mechanism B — Static Non-Differentiable CSR Kneser-Ney Lookup Table (`_kn_state` / `apply_surface_logit_lift`)**: Precomputed over `85M` training tokens (`~5–6 min` CPU build, `2.4 GB` state) and applied **only during evaluation (`return_logits=True`)** to overwrite up to `80%` (`w_eff = 0.80`) of the neural network's logits (`logits = (1 - w_eff) * z_cal + w_eff * log_kn`).
- Because `_kn_state` was never used in the training loss (`subspace_coincidence_dyadic_chunked_ce`), it created a train/eval objective mismatch, masked true width scaling ($d = 256 \to 1024$), and pinned long-horizon validation CE near the static lookup table's floor (`~4.81–4.84`).

`Wave 36` executes **Option 3** — a clean, apples-to-apples, **100% pure-neural (`_kn_state = None`) head-to-head comparison** across both **`d = 256, L = 6`** and **`d = 1024, L = 6` Mode B**:
- **Option 1 (`Differentiable N-Gram Wavefunction Only`, `disable_kn_lookup()`)**: Keeps `HashedSurfaceNGramProjector.forward` (`in_delta` + `readout_delta` on $\mathbb{S}^{d-1}$) while completely disabling `_kn_state` / `apply_surface_logit_lift`.
- **Option 2 (`100% Pure Wavelet-Cholesky + Multi-Rate Conv + Subspace Coincidence`, `disable_projector()`)**: Disables and freezes `HashedSurfaceNGramProjector` entirely (`-12.65M` fewer AdamW parameters), relying strictly on `MultiRateWavefunctionConvBank` `(4, 2, 1, 1)` + `SubspaceProjectorBornAttention` + `Tied2Pow16WaveletCholeskyPyramid`.

---

## 2. Observable Hypotheses & Quantitative Acceptance Criteria

### Hypothesis 1 (`H1` — Exact Train/Eval Objective Parity & Zero Startup Bottleneck Without `_kn_state`)
- **Claim**: Disabling `_kn_state` (`disable_kn_lookup()` in Option 1 and `disable_projector()` in Option 2) guarantees exact mathematical parity between the chunked training loss `model(x, targets=y)[1]` and the full evaluation cross-entropy `F.cross_entropy(model(x, return_logits=True)[0], y)` (`loss_diff < 1e-4` under `bfloat16` autocast at any $\tau \in [0, 1]$), and reduces model initialization time from `~300s` to **`< 15s`**.

### Hypothesis 2 (`H2` — Pure-Neural Option 1 vs. Option 2 Head-to-Head at `d = 256, L = 6` and `d = 1024, L = 6` Mode B)
- **Claim**: Across `300`-step (`2.46M` tokens, `B_eff = 16, T = 512`) pure-neural runs on `FineWeb-Edu`:
  1. Both Option 1 and Option 2 exhibit monotonic, unmasked neural validation CE descent (`val_ce` tracking `train_ce` within `< 0.25` nats throughout training, with zero $\tau$-ramp distortion),
  2. Comparing **Option 1** (`with differentiable ngram_proj.forward`) vs. **Option 2** (`without ngram_proj`, saving `12.65M` AdamW parameters) at both `d = 256` and `d = 1024` quantifies the exact pure-neural contribution of `HashedSurfaceNGramProjector.forward` (`\Delta \text{Val CE} = \text{CE}_{\text{Opt2}} - \text{CE}_{\text{Opt1}}`),
  3. Comparing **`d = 1024, L = 6` Mode B** against **`d = 256, L = 6`** without the `80%` static Kneser-Ney override reveals the true width scaling gain ($\Delta \text{CE}_{d=256 \to 1024}$) of the `107M–120M`-parameter quantum backbone.

---

## 3. Planned Changes & Rule 7 Constants Ledger

| Constant Name | File | Value | Rationale |
| :--- | :--- | :---: | :--- |
| `DEFAULT_ABLATION_STEPS` | `examples/61_verify_plan051_pure_neural_ngram_ablation.py` | `300` | Iso-token `2.46M`-token head-to-head horizon per variant |
| `PURE_NEURAL_INIT_TOKENS` | `examples/61_verify_plan051_pure_neural_ngram_ablation.py` | `3_500_000` | Token slice for fast `< 15s` PPMI-SVD + differentiable `ngram_proj` SVD init (no `85M` CSR build) |
| `TRAIN_EVAL_PARITY_TOL` | `examples/61_verify_plan051_pure_neural_ngram_ablation.py` | `5e-3` | Maximum allowed `|chunked_train_ce - eval_logit_ce|` under `bfloat16` autocast |

---

## 4. Testing & Verification Plan
1. **AST Signature & SLOC Linter**:
   - Run [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py) on [`src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/parseval_mup_ngram_speedrun_gpt.py) and [`examples/61_verify_plan051_pure_neural_ngram_ablation.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/61_verify_plan051_pure_neural_ngram_ablation.py).
2. **Live 4-Run GPU Execution (`Wave 36`)**:
   - Run [`examples/61_verify_plan051_pure_neural_ngram_ablation.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/61_verify_plan051_pure_neural_ngram_ablation.py) executing:
     - `w36_opt1_diff_ngram_d256_L6_300steps` (`d=256, L=6`, Option 1)
     - `w36_opt2_no_ngram_d256_L6_300steps` (`d=256, L=6`, Option 2)
     - `w36_opt1_diff_ngram_d1024_L6_mode_b_300steps` (`d=1024, L=6` Mode B, Option 1)
     - `w36_opt2_no_ngram_d1024_L6_mode_b_300steps` (`d=1024, L=6` Mode B, Option 2)
   - Sync results to `.cache/benchmarks.db` and the `:8765` WebUI via [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py).

---

## 5. Operational Tooling & Script Opportunities
1. **Existing Scripts Used**:
   - [`scripts/ast_signature_and_length_linter.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/ast_signature_and_length_linter.py)
   - [`scripts/subspace_energy_and_rank_drift_tracker.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/subspace_energy_and_rank_drift_tracker.py)
   - [`scripts/sync_waves_to_webui.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/sync_waves_to_webui.py)
2. **Verification Runner Built**:
   - [`examples/61_verify_plan051_pure_neural_ngram_ablation.py`](file:///home/festivity/agy/QuantumDLBlocks/examples/61_verify_plan051_pure_neural_ngram_ablation.py)
