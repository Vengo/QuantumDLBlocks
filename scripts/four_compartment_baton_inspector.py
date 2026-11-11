#!/usr/bin/env python3
"""Wave 73 (`EXPERIMENT_PLAN_093.md`) 5-Regime Quantum Statistical Mechanics Baton Inspector CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qdlblocks.benchmarks.four_compartment_baton_lm_lab import (
    REPO_ROOT,
    run_five_regime_baton_probe,
)

DEFAULT_JSON_OUT: Path = REPO_ROOT / ".cache" / "wave73_four_compartment_inspector.json"


def main() -> int:
    """CLI entrypoint for `scripts/four_compartment_baton_inspector.py` (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON_OUT,
        help="Path to write the 5-regime diagnostic JSON report.",
    )
    args = parser.parse_args()

    report = run_five_regime_baton_probe()
    out_path = args.json_out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"[four_compartment_baton_inspector] overall_passed={report['overall_passed']} "
        f"r1_bath={report['regime1_cold_bos_and_online_unigram_bath']['passed']} "
        f"r2_mps_codon={report['regime2_mps_stop_codon_destructive_cancellation']['passed']} "
        f"r3_lindblad={report['regime3_in_context_lindblad_transition_induction']['passed']} "
        f"r4_schur_hairpin={report['regime4_coding_hairpin_schur_vs_additive']['passed']} "
        f"r5_token_id={report['regime5_zero_shot_walsh_hadamard_token_id_reversal']['passed']} "
        f"json_out={out_path}"
    )
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
