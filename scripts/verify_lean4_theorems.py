#!/usr/bin/env python3
"""Verify the Lean 4 Quantum Geometry & Measurement formalization package (`lean/`).

Executes a clean `lake build` of `QuantumDLBlocksLean`, audits all `.lean` source files
(after stripping comments) for zero `sorry`, `admit`, or `axiom` declarations, parses
`#print axioms` compiler diagnostics for all 5 theorem modules, and writes a structured
JSON verification report to `.cache/lean4_theorems_verification.json`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_LEAN_DIR: Path = REPO_ROOT / "lean"
DEFAULT_JSON_OUT: Path = REPO_ROOT / ".cache" / "lean4_theorems_verification.json"
DEFAULT_ELAN_LAKE_BIN: Path = Path.home() / ".elan" / "bin" / "lake"
BUILD_TIMEOUT_SECONDS: int = 60

FORBIDDEN_LEAN_TOKENS: tuple[str, ...] = ("sorry", "admit", "axiom")
ALLOWED_KERNEL_AXIOMS: frozenset[str] = frozenset({"propext", "Quot.sound", "Classical.choice"})

EXPECTED_THEOREMS: tuple[str, ...] = (
    "QuantumDLBlocksLean.theorem1a_chain_rule_horizontal_completeness",
    "QuantumDLBlocksLean.theorem1b_false_symmetry_vertical_leak_and_norm_loss",
    "QuantumDLBlocksLean.theorem2a_degenerate_spectrum_synchronous_spin",
    "QuantumDLBlocksLean.theorem2b_split_spectrum_zero_vertical_spin",
    "QuantumDLBlocksLean.theorem2c_positive_split_spectrum_physical_spin",
    "QuantumDLBlocksLean.theorem3a_kraus_unitary_channel_and_tp_invariance",
    "QuantumDLBlocksLean.theorem3a_two_branch_direct_expansion",
    "QuantumDLBlocksLean.theorem3b_kraus_sylvester_horizontal_iff",
    "QuantumDLBlocksLean.theorem4a_newton_schulz_horizontal_leakage",
    "QuantumDLBlocksLean.theorem4b_post_ns_horizontal_projection_restores_horizontality",
    "QuantumDLBlocksLean.theorem5a_luders_joint_nonneg",
    "QuantumDLBlocksLean.theorem5b_luders_exact_bipartite_normalization",
    "QuantumDLBlocksLean.theorem5b_luders_scaled_bipartite_normalization",
    "QuantumDLBlocksLean.theorem5c_complementary_subspace_parseval_norm_preservation",
    "QuantumDLBlocksLean.theorem6a_hilbert_schmidt_born_nonneg_and_vacuum_norm",
    "QuantumDLBlocksLean.theorem6b_isometric_outer_product_flattening",
    "QuantumDLBlocksLean.theorem6c_right_unitary_gauge_invariance",
    "QuantumDLBlocksLean.theorem7a_quantum_or_convex_superposition_closure",
    "QuantumDLBlocksLean.schur_khatri_rao_entry_factorization",
    "QuantumDLBlocksLean.theorem7b_quantum_and_schur_product_theorem",
    "QuantumDLBlocksLean.theorem8a_online_ema_stream_induction",
    "QuantumDLBlocksLean.theorem8b_multi_window_stateful_summary_invariance",
    "QuantumDLBlocksLean.factored_kraus_associative_parenthesization_eq",
    "QuantumDLBlocksLean.kraus_commutes_with_causal_attention_sum",
    "QuantumDLBlocksLean.isometric_packed_routing_linearity",
    "QuantumDLBlocksLean.factored_cross_head_kraus_cptp_closure",
    "QuantumDLBlocksLean.representational_power_strict_separation_and_rank_amplification",
    "QuantumDLBlocksLean.momentum_buffer_horizontal_invariance",
    "QuantumDLBlocksLean.opposing_momentum_cleaning_descent_and_energy_bound",
    "QuantumDLBlocksLean.uncleaned_momentum_ascent_witness",
    "QuantumDLBlocksLean.isometric_parallel_transport_preserves_horizontality_and_norm",
    "QuantumDLBlocksLean.unanchored_channel_state_collision",
    "QuantumDLBlocksLean.dual_unitary_sensory_anchored_distinguishability",
    "QuantumDLBlocksLean.dual_unitary_sensory_anchored_cptp",
    "QuantumDLBlocksLean.bipartite_register_trace_and_decoupled_frame_self_copy_free",
    "QuantumDLBlocksLean.four_qubit_segre_norm_and_intra_head_entanglement",
    "QuantumDLBlocksLean.mean_field_single_branch_ghz_collapse_vs_multibranch_exactness",
    "QuantumDLBlocksLean.factorized_65536d_hilbert_schmidt_born_coincidence_identity",
)


@dataclass(frozen=True)
class LeanFileAuditResult:
    """Static token audit summary for a single `.lean` source file (`<= 8` fields)."""

    file_path: str
    total_lines: int
    code_lines: int
    forbidden_hits: dict[str, int]
    passed: bool


def strip_lean_comments(source: str) -> str:
    """Remove Lean 4 block comments (`/- ... -/`) and line comments (`-- ...`)."""
    without_block = re.sub(r"/-.*?-/", "", source, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", without_block)


def audit_lean_file(lean_file: Path, repo_root: Path) -> LeanFileAuditResult:
    """Scan a single `.lean` file for forbidden tokens (`sorry`, `admit`, `axiom`)."""
    raw_text = lean_file.read_text(encoding="utf-8")
    clean_text = strip_lean_comments(raw_text)
    hits: dict[str, int] = {}
    for token in FORBIDDEN_LEAN_TOKENS:
        matches = re.findall(rf"\b{re.escape(token)}\b", clean_text)
        hits[token] = len(matches)
    code_lines = sum(1 for line in clean_text.splitlines() if line.strip())
    rel_path = str(lean_file.relative_to(repo_root))
    passed = all(count == 0 for count in hits.values())
    return LeanFileAuditResult(
        file_path=rel_path,
        total_lines=len(raw_text.splitlines()),
        code_lines=code_lines,
        forbidden_hits=hits,
        passed=passed,
    )


def resolve_lake_binary() -> str:
    """Locate the `lake` build binary from `~/.elan/bin/lake` or `PATH`."""
    if DEFAULT_ELAN_LAKE_BIN.is_file():
        return str(DEFAULT_ELAN_LAKE_BIN)
    which_lake = shutil.which("lake")
    if which_lake is not None:
        return which_lake
    raise FileNotFoundError("Could not find `lake` binary in ~/.elan/bin/lake or PATH.")


def parse_printed_axioms(build_output: str) -> dict[str, list[str]]:
    """Extract theorem axiom dependencies from `#print axioms` compiler info output."""
    normalized = re.sub(r"\n\s+", " ", build_output)
    axioms_map: dict[str, list[str]] = {}
    no_ax_pattern = re.compile(r"'([^']+)' does not depend on any axioms")
    for match in no_ax_pattern.finditer(normalized):
        axioms_map[match.group(1)] = []
    dep_pattern = re.compile(r"'([^']+)' depends on axioms:\s*\[([^\]]+)\]")
    for match in dep_pattern.finditer(normalized):
        thm_name = match.group(1)
        axioms = [ax.strip() for ax in match.group(2).split(",") if ax.strip()]
        axioms_map[thm_name] = axioms
    return axioms_map


def run_lake_build(lean_dir: Path, clean_first: bool = True) -> dict[str, Any]:
    """Run `lake clean` (optional) and `lake build`, returning timing and `#print axioms` map."""
    lake_bin = resolve_lake_binary()
    if clean_first:
        subprocess.run(
            [lake_bin, "clean"],
            cwd=str(lean_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
    t0 = time.perf_counter()
    proc = subprocess.run(
        [lake_bin, "build"],
        cwd=str(lean_dir),
        check=False,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    elapsed_sec = time.perf_counter() - t0
    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    axioms_map = parse_printed_axioms(combined_output)
    has_warnings = "warning:" in combined_output
    return {
        "lake_binary": lake_bin,
        "exit_code": proc.returncode,
        "build_time_seconds": round(elapsed_sec, 4),
        "has_warnings": has_warnings,
        "printed_axioms": axioms_map,
        "raw_output": combined_output.strip(),
    }


def verify_lean4_package(
    lean_dir: Path = DEFAULT_LEAN_DIR,
    json_out: Path = DEFAULT_JSON_OUT,
    clean_first: bool = True,
) -> dict[str, Any]:
    """Run full static + compiler verification of `lean/` and write JSON summary (`<= 8` args)."""
    lean_files = sorted(
        p for p in lean_dir.rglob("*.lean") if ".lake" not in p.parts
    )
    file_audits = [audit_lean_file(f, REPO_ROOT) for f in lean_files]
    static_passed = len(file_audits) >= 6 and all(a.passed for a in file_audits)

    build_info = run_lake_build(lean_dir=lean_dir, clean_first=clean_first)
    printed_axioms: dict[str, list[str]] = build_info["printed_axioms"]

    missing_theorems = [t for t in EXPECTED_THEOREMS if t not in printed_axioms]
    disallowed_axioms: dict[str, list[str]] = {}
    for thm, ax_list in printed_axioms.items():
        bad = [ax for ax in ax_list if ax not in ALLOWED_KERNEL_AXIOMS]
        if bad:
            disallowed_axioms[thm] = bad

    overall_passed = (
        static_passed
        and build_info["exit_code"] == 0
        and not build_info["has_warnings"]
        and len(missing_theorems) == 0
        and len(disallowed_axioms) == 0
    )

    report: dict[str, Any] = {
        "overall_passed": overall_passed,
        "lean_dir": str(lean_dir.relative_to(REPO_ROOT)),
        "num_lean_files_audited": len(file_audits),
        "static_zero_sorry_admit_axiom_passed": static_passed,
        "file_audits": [asdict(a) for a in file_audits],
        "lake_build_exit_code": build_info["exit_code"],
        "lake_build_time_seconds": build_info["build_time_seconds"],
        "lake_build_has_warnings": build_info["has_warnings"],
        "expected_theorems_count": len(EXPECTED_THEOREMS),
        "verified_theorems_count": len(printed_axioms),
        "missing_theorems": missing_theorems,
        "disallowed_axioms": disallowed_axioms,
        "theorem_axioms": printed_axioms,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    """CLI entrypoint for `scripts/verify_lean4_theorems.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lean-dir",
        type=Path,
        default=DEFAULT_LEAN_DIR,
        help="Path to the Lean 4 Lake project directory.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON_OUT,
        help="Output JSON verification report path.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip `lake clean` before `lake build`.",
    )
    args = parser.parse_args()
    report = verify_lean4_package(
        lean_dir=args.lean_dir.resolve(),
        json_out=args.json_out.resolve(),
        clean_first=not args.no_clean,
    )
    print(
        f"[verify_lean4_theorems] overall_passed={report['overall_passed']} "
        f"files={report['num_lean_files_audited']} "
        f"theorems={report['verified_theorems_count']}/{report['expected_theorems_count']} "
        f"build_time={report['lake_build_time_seconds']}s "
        f"json_out={args.json_out}"
    )
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
