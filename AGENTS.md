# QuantumDLBlocks Agent Guidelines (`AGENTS.md`)

## 1. Live Colab GPU/TPU Experiment Monitoring (`monitor_colab_runs.py`)

- The background daemon `scripts/monitor_colab_runs.py --serve --host 0.0.0.0 --port 8765 --interval 20` is the **single source of truth** that automatically polls remote Colab sessions every 20 seconds and syncs their logs into `.cache/live_logs/`.
- **Do not spawn competing `colab exec` commands** just to check training progress while `--serve` is running. Instead:
  - Run `curl -s http://127.0.0.1:8765/api/report` or `/home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/monitor_colab_runs.py --once` (which reads directly from `http://127.0.0.1:8765/api/snapshot` in `<10ms`).
  - Or inspect the synced log files directly in `/home/festivity/agy/QuantumDLBlocks/.cache/live_logs/`.
- See `.agents/skills/colab-experiment-monitor/SKILL.md` for full details on launching detached runs and comparing against the FineWeb 100M Slowrun (`3.183` Val CE) and Modded-NanoGPT Speedrun (`3.280` Val CE in `0.665` min) records on the genuine `4,999,488`-token validation shard.

## 2. First-Class Operational Scripts (Use Scripts Instead of Ad-Hoc Commands)

Agents MUST use the repository's reusable CLI tools in `scripts/` instead of writing ad-hoc `python -c` or multi-line `colab exec` bash snippets:

1. **Colab Fleet Code Sync, Launch, Status, & Harvesting (`scripts/colab_fleet.py`)**:
   - Push `src/` + `examples/11_*.py` and launch a wave from a declarative manifest (`configs/wave6_fleet.json`):
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py launch --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave6_fleet.json
     ```
   - Check live GPU & PID health or harvest completed `.log` / `_results.json` artifacts:
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py status --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave6_fleet.json
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/colab_fleet.py harvest --manifest /home/festivity/agy/QuantumDLBlocks/configs/wave6_fleet.json
     ```
2. **Model Parameter Breakdown & Synthetic Smoke Testing (`scripts/inspect_model.py`)**:
   - Inspect module-group parameter budgets and run a 2-step CUDA/CPU forward+backward smoke test:
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/inspect_model.py --seed-dim 512 --overcomplete-two-beam --frame-expansion 4.0 --two-beam-seed-adapter --dynamic-povm-beta --smoke-steps 2
     ```
3. **Karpathy Memorization Capacity & Logit-Sharpness Probe (`scripts/overfit_probe.py`)**:
   - Measure overfitting accuracy (`top1_acc`), output entropy (`mean_output_entropy_nats`), and logit L2 norm variation (`logit_norm_cv`) across $N \in \{1, 8, 32\}$ FineWeb records:
     ```bash
     /home/festivity/agy/QuantumDLBlocks/.venv/bin/python /home/festivity/agy/QuantumDLBlocks/scripts/overfit_probe.py --variants q_adapter384_2beam,q_adapter384_2beam_dynbeta,classical_modded --records 1,8,32
     ```

## 3. Standard Planning & Execution Workflow: Mandatory Tooling & "Nice-to-Have" Script Gate

Every new `EXPERIMENT_PLAN_0XX.md` and subagent workflow MUST include an explicit **Operational Tooling & Script Identification** gate:

1. **In Every `EXPERIMENT_PLAN_0XX.md` (Required Section 5: Operational Tooling & Script Opportunities)**:
   - **Existing Scripts Used**: Specify exact `scripts/inspect_model.py`, `scripts/overfit_probe.py`, `scripts/colab_fleet.py`, and `scripts/monitor_colab_runs.py` invocations that will validate, launch, and monitor the experiment.
   - **`Build-Before-Launch` Scripts**: Any repetitive command needed to execute or launch the plan that is not yet scripted must be added to `scripts/` as part of the plan implementation before fleet launch.
   - **`Nice-to-Have` Scripts (Surfaced to User at Launch)**: Identify 1–3 candidate scripts that would make launching, monitoring, or analyzing this class of task even easier in the future. Log them in `scripts/TOOLING_BACKLOG.md` and highlight them to the user when presenting the plan/launch so we can decide together whether to build them immediately.
2. **In Blind Plan Reviews (`.agents/agents/blind-plan-reviewer/agent.md`)**:
   - The blind reviewer audits Section 5 alongside Hypotheses, Planned Changes, and Testing to verify that launch/verification steps rely on deterministic scripts rather than fragile ad-hoc commands, and that `Nice-to-Have` script opportunities are clearly articulated.
3. **In Implementer Subagent Handoffs (Friction & Ad-Hoc Command Audit)**:
   - Every implementer subagent must include a short **Tooling Friction Audit** in its completion report listing any ad-hoc one-liners it ran more than once so they can be promoted into `scripts/`.

## 4. Script & Module Engineering Best Practices (`docs/SCRIPT_BEST_PRACTICES.md`)

All agents and subagents writing or modifying Python files in `examples/`, `scripts/`, and `src/qdlblocks/` MUST enforce the five rules in [`docs/SCRIPT_BEST_PRACTICES.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SCRIPT_BEST_PRACTICES.md):

1. **Rule 1 — File-Length Budget (`<= 1,000` Lines Target, `<= 1,200` Hard Ceiling, No Semicolon Line-Packing)**:
   - Keep new scripts and modules in `examples/`, `scripts/`, and `src/qdlblocks/` `<= 1,000` lines target with an absolute hard ceiling of `1,200` lines, and never pack multiple statements onto one line with semicolons (`a; b`). Extract reusable logic into `src/qdlblocks/benchmarks/` or `src/qdlblocks/layers/`.
2. **Rule 2 — Function Signature Ceiling (`<= 8` Arguments)**:
   - No new function or method may accept more than `8` positional/keyword arguments (excluding `self`/`cls`). Group domain hyperparameters into frozen `@dataclass(frozen=True)` objects (`BackboneConfig`, `QubitHierarchyConfig`, `KBosonReadoutConfig`, `OptimizerConfig`, `QuantumExperimentSpec` in [`src/qdlblocks/benchmarks/config_models.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py)).
3. **Rule 3 — Strategy / Polymorphic Dispatch Over Boolean Flag Jungles**:
   - Replace mutually exclusive boolean flags selecting `if / elif / else` branches with explicit strategy classes, enums, or composable sub-configs resolved once at construction time.
4. **Rule 4 — Single Shared Builders Between Local `8 GB` Scripts & Cloud Runners**:
   - Never copy-paste `initialize_povm_log_prior`, `build_quantum_born_optimizers`, `build_quantum_model_from_spec`, or validation loops into per-plan scripts; always import them from [`src/qdlblocks/benchmarks/config_models.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/config_models.py) and [`src/qdlblocks/benchmarks/calibration_probe.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/calibration_probe.py).
5. **Rule 5 — No Giant Closure-Capturing Nested Functions**:
   - Keep helper functions top-level, pure, and independently unit-testable via `pytest`.

