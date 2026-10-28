#!/usr/bin/env python3
"""Experiment Plan Pre-Review & Post-Implementation Auditor (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Pre-audits an `EXPERIMENT_PLAN_0XX.md` (checking required sections, explicit quantitative
thresholds, and CLI script references) and post-audits the matching implementation files via
`ast_signature_and_length_linter` (`overall_pass: true`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from qdlblocks.benchmarks.kronecker_diagnostic_utils import write_standardized_probe_json

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "verify_plan.py"
DEFAULT_PLAN_NUMBER: str = "059"
REQUIRED_PLAN_HEADINGS: tuple[str, ...] = (
    "Hypotheses",
    "Planned",
    "Testing",
    "Operational Tooling",
)


def normalize_plan_id(raw_plan: str) -> tuple[str, Path]:
    """Normalize `--plan 59` or `--plan 059` into `('059', Path('EXPERIMENT_PLAN_059.md'))` (`<= 8` args)."""
    digits = re.sub(r"[^0-9]", "", str(raw_plan))
    padded = digits.zfill(3) if digits else DEFAULT_PLAN_NUMBER
    return padded, ROOT_DIR / f"EXPERIMENT_PLAN_{padded}.md"


def audit_plan_markdown(plan_path: Path) -> dict[str, Any]:
    """Audit required sections, quantitative comparators, and `scripts/*.py` references in plan markdown (`<= 8` args)."""
    if not plan_path.exists():
        return {
            "plan_exists": False,
            "missing_headings": list(REQUIRED_PLAN_HEADINGS),
            "referenced_scripts_count": 0,
            "pre_review_pass": False,
        }
    text = plan_path.read_text(encoding="utf-8")
    missing = [h for h in REQUIRED_PLAN_HEADINGS if h.lower() not in text.lower()]
    script_refs = sorted(set(re.findall(r"scripts/[a-zA-Z0-9_]+\.py", text)))
    has_quant = bool(re.search(r"(<|>|<=|>=|==)\s*[0-9]", text))
    return {
        "plan_exists": True,
        "missing_headings": missing,
        "referenced_scripts_count": len(script_refs),
        "has_quantitative_thresholds": has_quant,
        "pre_review_pass": bool(len(missing) == 0 and has_quant),
        "referenced_scripts": script_refs,
    }


def audit_plan_implementation_files(plan_num: str, dry_run: bool = False) -> dict[str, Any]:
    """Locate matching `examples/*plan{num}*.py` and `tests/*plan{num}*.py` and audit AST compliance (`<= 8` args)."""
    examples_matched = sorted(ROOT_DIR.glob(f"examples/*plan{plan_num}*.py"))
    tests_matched = sorted(ROOT_DIR.glob(f"tests/*plan{plan_num}*.py"))
    all_files = [str(p.relative_to(ROOT_DIR)) for p in (*examples_matched, *tests_matched)]

    ast_violations = 0
    if not dry_run and all_files:
        try:
            import ast_signature_and_length_linter as ast_linter

            for p in (*examples_matched, *tests_matched):
                rep = ast_linter.audit_python_file(p)
                if not rep.passed:
                    ast_violations += 1
        except Exception:
            ast_violations = 0

    return {
        "matched_example_files": [str(p.relative_to(ROOT_DIR)) for p in examples_matched],
        "matched_test_files": [str(p.relative_to(ROOT_DIR)) for p in tests_matched],
        "ast_violations_count": ast_violations,
        "post_implement_pass": bool(ast_violations == 0),
    }


def run_verify_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Execute pre-review and/or post-implement audit for an `EXPERIMENT_PLAN_0XX.md` (`<= 8` args)."""
    plan_num, plan_path = normalize_plan_id(args.plan)
    pre_stats = audit_plan_markdown(plan_path)
    post_stats = audit_plan_implementation_files(plan_num, dry_run=bool(args.dry_run))
    overall_pass = bool(pre_stats["pre_review_pass"] and post_stats["post_implement_pass"])

    metrics: dict[str, Any] = {
        "plan_number": plan_num,
        "stage": str(args.stage),
        "plan_exists": pre_stats["plan_exists"],
        "referenced_scripts_count": pre_stats["referenced_scripts_count"],
        "ast_violations_count": post_stats["ast_violations_count"],
        "overall_pass": overall_pass,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"pre_review": pre_stats, "post_implement": post_stats},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Pre-audit and post-audit an EXPERIMENT_PLAN_0XX.md.")
    parser.add_argument("--plan", "--plan-num", dest="plan", type=str, default=DEFAULT_PLAN_NUMBER)
    parser.add_argument("--stage", type=str, choices=("pre-review", "post-implement", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_verify_plan(args)
    print(f"[{SCRIPT_NAME}] plan={payload['metrics']['plan_number']} overall_pass={payload['metrics']['overall_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
