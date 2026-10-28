"""Unit and CLI smoke tests for Track 7 Tooling Backlog Scripts & Sub-Agent Surfacing.

When to Use:
- Run via `pytest tests/test_track7_tooling_backlog_scripts.py -q` to verify:
  1. `src/qdlblocks/benchmarks/kronecker_diagnostic_utils.py` shared builders,
     10-decile Zipf partitioner, Gram spectrum, branch overlap, and JSON writer.
  2. All 14 Set A Wave 44 probe scripts and all 14 Set B `TOOLING_BACKLOG.md`
     scripts execute `--help` and `--json-out` smoke runs with exit code 0 and
     valid finite metrics.
  3. Sub-agent context files (`.agents/skills/qdlblocks-tooling-and-invariants/SKILL.md`,
     `docs/designs/SITEMAP.md`, `scripts/TOOLING_BACKLOG.md`, `AGENTS.md`,
     `docs/BUILDING_BLOCKS.md`, `docs/SCRIPT_BEST_PRACTICES.md`) are synchronized.

Experiment Provenance:
- Created in Wave 44 (`EXPERIMENT_PLAN_059.md` / Track 7).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch

from qdlblocks.benchmarks.kronecker_diagnostic_utils import (
    FINEWEB_GPT2_VOCAB_SIZE,
    PADDED_KRONECKER_VOCAB_SIZE,
    DiagnosticModelSpec,
    KroneckerProbeConfig,
    build_or_load_kronecker_probe_model,
    compute_complex_factor_gram_spectrum,
    compute_fineweb_10_frequency_deciles,
    compute_pairwise_complex_branch_overlap,
    load_fineweb_eval_slice,
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
EXPECTED_WAVE_ID: str = "W44"
EXPECTED_PLAN_ID: str = "EXPERIMENT_PLAN_059"
EXPECTED_NUM_DECILES: int = 10
TEST_SEQ_LEN: int = 32
TEST_BATCH_SIZE: int = 2
SUBPROCESS_TIMEOUT_S: int = 60

SET_A_SCRIPTS: tuple[str, ...] = (
    "ngram_helmert_carrier_calibration_inspector.py",
    "trigram_context_smearing_ablation_probe.py",
    "multibranch_polysemy_entropy_and_subspace_angle_probe.py",
    "kronecker_coincidence_gradient_snr_and_rank_probe.py",
    "calibrated_kronecker_vocab_spectrum_and_povm_drift_probe.py",
    "context_adaptive_branch_beta_and_self_echo_probe.py",
    "born_exponent_and_holographic_coincidence_sweep.py",
    "multilag_window_and_coincidence_schedule_probe.py",
    "multihead_gram_krylov_vs_intrahead_ablation_probe.py",
    "quantum_superposition_purity_and_gram_spectrum_probe.py",
    "complex_stiefel_orthogonality_and_large_batch_probe.py",
    "self_echo_repetition_penalty_and_nll_breakdown_probe.py",
    "wave_preflight_hardware_and_throughput_gate.py",
    "colab_fleet_watchdog_and_auto_requeue.py",
)

SET_B_SCRIPTS: tuple[str, ...] = (
    "multihead_von_neumann_entropy_and_coherence_inspector.py",
    "polysemy_router_token_sense_clustering_cli.py",
    "polysemy_branch_sense_disambiguation_inspector.py",
    "kronecker_factor_entanglement_and_mutual_info_probe.py",
    "povm_scale_drift_and_rarity_decile_analyzer.py",
    "rare_token_gradient_starvation_tracker.py",
    "ngram_quantum_operator_vs_attention_gate_tracker.py",
    "modded_nanogpt_10m_per_decile_ce_breakdown.py",
    "born_povm_boson_power_and_floor_calibrator.py",
    "random_init_attention_carrier_spectrum_probe.py",
    "grad_accum_effective_batch_scaling_profiler.py",
    "wall_clock_chinchilla_pareto_predictor.py",
    "verify_plan.py",
    "wave_report.py",
)

ALL_TRACK7_SCRIPTS: tuple[str, ...] = SET_A_SCRIPTS + SET_B_SCRIPTS

PARALLEL_TRACK_SCRIPTS_ALLOWED_MISSING: frozenset[str] = frozenset(
    {
        "scripts/custom_autograd_efficiency_linter.py",
        "scripts/multihead_mixed_state_and_eval_speed_profiler.py",
    }
)


def _assert_all_numbers_finite(obj: object, path: str = "root") -> None:
    """Recursively verify that every int/float inside a JSON payload is finite."""
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_numbers_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_all_numbers_finite(v, f"{path}[{i}]")


def test_kronecker_diagnostic_utils_builders_and_deciles(tmp_path: Path) -> None:
    """Test shared Kronecker diagnostic builders, deciles, Gram spectrum, and JSON writer."""
    probe_cfg = KroneckerProbeConfig(
        device="cpu",
        seq_len=TEST_SEQ_LEN,
        num_layers=2,
        num_meanings=2,
    )
    model, meta = build_or_load_kronecker_probe_model(probe_cfg)
    assert not hasattr(model, "prior_weights")
    assert model.unigram_probs.numel() >= FINEWEB_GPT2_VOCAB_SIZE
    assert "arch" in meta
    assert meta["checkpoint_mode"] == "calibrated_fresh"

    # Test JSON telemetry checkpoint fallback in load_or_build_calibrated_kronecker_model
    dummy_json_ckpt = tmp_path / "w44_dummy_results.json"
    dummy_json_ckpt.write_text(
        json.dumps({"run_key": "w44_dummy", "val_ce": 5.65}),
        encoding="utf-8",
    )
    spec = DiagnosticModelSpec(
        checkpoint_path=str(dummy_json_ckpt),
        device="cpu",
        seq_len=TEST_SEQ_LEN,
        n_layer=2,
        meaning_rank=2,
    )
    model_b, meta_b = load_or_build_calibrated_kronecker_model(spec)
    assert not hasattr(model_b, "prior_weights")
    assert meta_b["checkpoint_mode"] == "json_telemetry"
    assert meta_b["checkpoint_json"]["run_key"] == "w44_dummy"
    assert model_b.unigram_probs.numel() >= FINEWEB_GPT2_VOCAB_SIZE

    # Test 10-decile Zipfian frequency partitioner
    decile_of_token = compute_fineweb_10_frequency_deciles(
        model.unigram_probs,
        active_vocab_size=FINEWEB_GPT2_VOCAB_SIZE,
        num_deciles=EXPECTED_NUM_DECILES,
    )
    assert decile_of_token.numel() >= FINEWEB_GPT2_VOCAB_SIZE
    for d in range(EXPECTED_NUM_DECILES):
        count_in_d = int((decile_of_token == d).sum().item())
        assert count_in_d > 0

    # Test evaluation slice and batch loaders
    x_batch, y_batch = load_kronecker_probe_batch(
        batch_size=TEST_BATCH_SIZE,
        seq_len=TEST_SEQ_LEN,
        device="cpu",
    )
    assert x_batch.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN)
    assert y_batch.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN)
    flat_slice = load_fineweb_eval_slice(max_tokens=TEST_BATCH_SIZE * TEST_SEQ_LEN)
    assert flat_slice.numel() == TEST_BATCH_SIZE * TEST_SEQ_LEN

    # Test complex factor Gram spectrum and pairwise branch overlap
    with torch.no_grad():
        _, loss_t = model(x_batch, targets=y_batch, return_logits=False)
        assert loss_t is not None and math.isfinite(float(loss_t.item()))
    gram_metrics = compute_complex_factor_gram_spectrum(
        model.embed_re[:256],
        model.embed_im[:256],
    )
    assert gram_metrics["mean_participation_ratio_real32"] > 1.0
    assert 0.0 < gram_metrics["mean_purity"] <= 1.0

    overlap_metrics = compute_pairwise_complex_branch_overlap(
        model.embed_re[:256],
        model.embed_im[:256],
        max_tokens=256,
    )
    assert overlap_metrics["num_meanings"] == 2
    assert 0.0 <= overlap_metrics["mean_kronecker_branch_overlap"] <= 1.0
    assert PADDED_KRONECKER_VOCAB_SIZE >= FINEWEB_GPT2_VOCAB_SIZE

    out_json = tmp_path / "probe_out.json"
    payload = write_standardized_probe_json(
        json_out=out_json,
        script_name="test_script.py",
        metrics={"gram": gram_metrics, "overlap": overlap_metrics},
    )
    assert out_json.exists()
    assert payload["wave_id"] == EXPECTED_WAVE_ID
    assert payload["plan_id"] == EXPECTED_PLAN_ID
    assert payload["status"] == "ok"


@pytest.mark.parametrize("script_name", ALL_TRACK7_SCRIPTS)
def test_track7_cli_script_help_and_json_smoke(
    script_name: str,
    tmp_path: Path,
) -> None:
    """Verify `--help` and `--json-out` execution for every Track 7 CLI script."""
    script_path = SCRIPTS_DIR / script_name
    assert script_path.exists(), f"Missing script: {script_path}"

    # 1. Verify --help
    help_proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    assert help_proc.returncode == 0, (
        f"--help failed for {script_name}:\n{help_proc.stderr}"
    )

    # 2. Verify execution with --json-out
    json_out = tmp_path / f"{script_path.stem}.json"
    cmd = [sys.executable, str(script_path), "--json-out", str(json_out)]
    if script_name == "colab_fleet_watchdog_and_auto_requeue.py":
        cmd.append("--dry-run")
    elif script_name == "verify_plan.py":
        cmd.extend(["--plan-num", "59", "--skip-pytest"])

    run_proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    assert run_proc.returncode == 0, (
        f"Execution failed for {script_name}:\nSTDOUT:\n{run_proc.stdout}\nSTDERR:\n{run_proc.stderr}"
    )
    assert json_out.exists(), f"{script_name} did not write {json_out}"

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["script"] == script_name
    assert data["wave_id"] == EXPECTED_WAVE_ID
    assert data["plan_id"] == EXPECTED_PLAN_ID
    assert data["status"] == "ok"
    assert isinstance(data.get("metrics"), dict)
    assert len(data["metrics"]) > 0
    _assert_all_numbers_finite(data["metrics"])


def test_subagent_surfacing_and_sitemap_integrity() -> None:
    """Verify SKILL.md, SITEMAP.md, TOOLING_BACKLOG.md, AGENTS.md, and docs synchronization."""
    skill_path = REPO_ROOT / ".agents/skills/qdlblocks-tooling-and-invariants/SKILL.md"
    colab_skill_path = REPO_ROOT / ".agents/skills/colab-experiment-monitor/SKILL.md"
    sitemap_path = REPO_ROOT / "docs/designs/SITEMAP.md"
    backlog_path = REPO_ROOT / "scripts/TOOLING_BACKLOG.md"
    best_practices_path = REPO_ROOT / "docs/SCRIPT_BEST_PRACTICES.md"
    building_blocks_path = REPO_ROOT / "docs/BUILDING_BLOCKS.md"
    agents_path = REPO_ROOT / "AGENTS.md"
    plan059_path = REPO_ROOT / "EXPERIMENT_PLAN_059.md"

    for p in (
        skill_path,
        colab_skill_path,
        sitemap_path,
        backlog_path,
        best_practices_path,
        building_blocks_path,
        agents_path,
        plan059_path,
    ):
        assert p.exists(), f"Missing required documentation file: {p}"

    skill_text = skill_path.read_text(encoding="utf-8")
    assert "name: qdlblocks-tooling-and-invariants" in skill_text
    assert "10,485,760" in skill_text

    colab_skill_text = colab_skill_path.read_text(encoding="utf-8")
    assert "10,485,760" in colab_skill_text
    assert "4,999,488" not in colab_skill_text

    best_practices_text = best_practices_path.read_text(encoding="utf-8")
    assert "Rule 9" in best_practices_text

    backlog_text = backlog_path.read_text(encoding="utf-8")
    assert backlog_text.count("[x] Implemented (Wave 44 / Plan 059)") >= 14

    # Verify every scripts/*.py and docs/designs/egm_tracks/*.md path referenced in SKILL.md & SITEMAP.md exists
    combined_doc_text = "\n".join(
        [
            skill_text,
            sitemap_path.read_text(encoding="utf-8"),
            backlog_text,
            agents_path.read_text(encoding="utf-8"),
        ]
    )
    script_refs = set(re.findall(r"scripts/[a-zA-Z0-9_]+\.py", combined_doc_text))
    for rel_script in sorted(script_refs):
        if rel_script in PARALLEL_TRACK_SCRIPTS_ALLOWED_MISSING:
            continue
        assert (REPO_ROOT / rel_script).exists(), f"Referenced script missing: {rel_script}"

    track_refs = set(
        re.findall(r"docs/designs/egm_tracks/[a-zA-Z0-9_]+\.md", combined_doc_text)
    )
    assert len(track_refs) == 7
    for rel_track in sorted(track_refs):
        assert (REPO_ROOT / rel_track).exists(), f"Referenced track doc missing: {rel_track}"
