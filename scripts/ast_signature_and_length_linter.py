#!/usr/bin/env python3
"""AST Signature, File-Length, and Nested-Closure Linter (`docs/SCRIPT_BEST_PRACTICES.md`).

Promoted from `scripts/BACKLOG.md` (identified in Plan 033, implemented in Plan 034).
Audits target Python files via `ast.parse` for:
- Rule 1: File line count (`<= max_lines` target, `<= hard_ceiling_lines` ceiling)
- Rule 2: Function/method argument count (`<= max_args` excluding `self`/`cls`)
- Rule 5: Nested function closure length (`<= max_nested_lines`)
Prints a Markdown compliance table and exits with code 0 when all target files pass.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import sys
import tokenize
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGET_PATHS: tuple[str, ...] = (
    "src/qdlblocks/layers/wavelet_cholesky.py",
    "src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py",
    "src/qdlblocks/benchmarks/config_models.py",
    "scripts/full_ambient_step0_calibration_audit.py",
    "scripts/vram_envelope_autotuner.py",
    "scripts/ast_signature_and_length_linter.py",
    "scripts/wavelet_cholesky_spectrum_probe.py",
    "scripts/vocab_dark_ancilla_utilization_audit.py",
    "scripts/mera_entanglement_cone_visualizer.py",
    "examples/43_verify_plan033_scaling_and_soft_spectral_floor.py",
    "examples/45_verify_plan035_radical_2pow16_wavelet_cholesky.py",
)


@dataclass(frozen=True)
class LinterConfig:
    """Configuration for AST signature, file-length, and nested-closure linting."""

    paths: tuple[str, ...] = DEFAULT_TARGET_PATHS
    max_lines: int = 1000
    hard_ceiling_lines: int = 2500
    max_args: int = 8
    max_nested_lines: int = 40
    strict: bool = False
    out_json: str = ""


def count_function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count positional, keyword-only, and variadic arguments excluding `self` and `cls`."""
    pos_names = [a.arg for a in (*node.args.posonlyargs, *node.args.args)]
    if pos_names and pos_names[0] in ("self", "cls"):
        pos_names = pos_names[1:]
    total = len(pos_names) + len(node.args.kwonlyargs)
    if node.args.vararg is not None:
        total += 1
    if node.args.kwarg is not None:
        total += 1
    return total


def inspect_ast_functions(
    tree: ast.AST,
    cfg: LinterConfig,
) -> dict[str, Any]:
    """Traverse AST to collect function argument counts and nested closure lengths."""
    num_functions = 0
    max_fn_args = 0
    max_arg_fn_name = "-"
    max_nested_fn_lines = 0
    max_nested_fn_name = "-"
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            num_functions += 1
            n_args = count_function_args(node)
            if n_args > max_fn_args:
                max_fn_args = n_args
                max_arg_fn_name = node.name
            if n_args > cfg.max_args:
                violations.append(f"Rule2: `{node.name}` has {n_args} args (> {cfg.max_args})")

            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end_line = int(getattr(child, "end_lineno", None) or child.lineno)
                    span = max(1, end_line - int(child.lineno) + 1)
                    nested_label = f"{node.name}.{child.name}"
                    if span > max_nested_fn_lines:
                        max_nested_fn_lines = span
                        max_nested_fn_name = nested_label
                    if span > cfg.max_nested_lines:
                        violations.append(
                            f"Rule5: nested `{nested_label}` is {span} lines (> {cfg.max_nested_lines})"
                        )

    return {
        "num_functions": num_functions,
        "max_fn_args": max_fn_args,
        "max_arg_fn_name": max_arg_fn_name,
        "max_nested_fn_lines": max_nested_fn_lines,
        "max_nested_fn_name": max_nested_fn_name,
        "ast_violations": violations,
    }


def audit_python_file(rel_or_abs_path: str, cfg: LinterConfig) -> dict[str, Any]:
    """Audit a single Python file against Rule 1 (1,000 soft warning, 2,500 hard limit), Rule 2, and Rule 5."""
    p = Path(rel_or_abs_path)
    full_path = p if p.is_absolute() else (ROOT / p)
    rel_display = str(full_path.relative_to(ROOT)) if full_path.is_relative_to(ROOT) else str(full_path)

    source = full_path.read_text(encoding="utf-8")
    line_count = len(source.splitlines())
    tree = ast.parse(source, filename=str(full_path))
    fn_stats = inspect_ast_functions(tree, cfg)

    violations = list(fn_stats["ast_violations"])
    warnings: list[str] = []
    if line_count > cfg.hard_ceiling_lines:
        violations.append(f"Rule1(HardLimit): {line_count} lines > {cfg.hard_ceiling_lines}")
    elif line_count > cfg.max_lines:
        if cfg.strict:
            violations.append(f"Rule1(SoftLimitStrict): {line_count} lines > {cfg.max_lines}")
        else:
            warnings.append(
                f"Rule1(SoftLimitWarning): {line_count} lines > {cfg.max_lines} soft limit (hard limit {cfg.hard_ceiling_lines})"
            )

    semicolon_lines: list[int] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.OP and tok.string == ";":
            semicolon_lines.append(int(tok.start[0]))
    if semicolon_lines:
        violations.append(f"Rule1(NoSemicolonPacking): semicolon ';' on line(s) {sorted(set(semicolon_lines))}")

    passed = len(violations) == 0
    if not passed:
        status = "FAIL"
    elif warnings:
        status = "WARN (PASS)"
    else:
        status = "PASS"
    return {
        "path": rel_display,
        "line_count": line_count,
        "max_lines_target": cfg.max_lines,
        "hard_ceiling_lines": cfg.hard_ceiling_lines,
        "num_functions": fn_stats["num_functions"],
        "max_fn_args": fn_stats["max_fn_args"],
        "max_arg_fn_name": fn_stats["max_arg_fn_name"],
        "max_nested_fn_lines": fn_stats["max_nested_fn_lines"],
        "max_nested_fn_name": fn_stats["max_nested_fn_name"],
        "passed": passed,
        "status": status,
        "warnings": warnings,
        "violations": violations,
    }


def run_linter(cfg: LinterConfig) -> dict[str, Any]:
    """Run the AST linter across all configured `cfg.paths` and print a Markdown table."""
    rows = [audit_python_file(p, cfg) for p in cfg.paths]
    all_passed = all(bool(r["passed"]) for r in rows)

    print(
        f"\n### AST Best-Practices Compliance Table (`soft_limit={cfg.max_lines}`, "
        f"`hard_limit={cfg.hard_ceiling_lines}`, `max_args={cfg.max_args}`, `max_nested={cfg.max_nested_lines}`)"
    )
    print("| File | Lines (`Soft` / `Hard`) | Functions | Max Args (`fn`) | Max Nested Lines | Status |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        fn_tag = f"`{r['max_fn_args']}` (`{r['max_arg_fn_name']}`)"
        nest_tag = (
            "`0`"
            if r["max_nested_fn_lines"] == 0
            else f"`{r['max_nested_fn_lines']}` (`{r['max_nested_fn_name']}`)"
        )
        print(
            f"| `{r['path']}` | `{r['line_count']}` (`{cfg.max_lines}` / `{cfg.hard_ceiling_lines}`) | "
            f"`{r['num_functions']}` | {fn_tag} | {nest_tag} | **{r['status']}** |"
        )
        for w in r.get("warnings", []):
            print(f"  - Warning in `{r['path']}`: {w}")
        for v in r["violations"]:
            print(f"  - Violation in `{r['path']}`: {v}")

    report = {
        "config": asdict(cfg),
        "all_passed": all_passed,
        "num_files": len(rows),
        "rows": rows,
    }
    if cfg.out_json:
        out_p = Path(cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> LinterConfig:
    """Parse CLI arguments into a frozen `LinterConfig`."""
    p = argparse.ArgumentParser(
        description="Audit Python files for line count, function signature args, and nested closures."
    )
    p.add_argument("paths", nargs="*", default=list(DEFAULT_TARGET_PATHS))
    p.add_argument("--max-lines", type=int, default=1000, help="Soft limit carrying a warning (default: 1000)")
    p.add_argument("--hard-ceiling-lines", type=int, default=2500, help="Hard limit failing the linter (default: 2500)")
    p.add_argument("--max-args", type=int, default=8)
    p.add_argument("--max-nested-lines", type=int, default=40)
    p.add_argument("--strict", action="store_true", help="Treat soft limit (> max-lines) as a hard error")
    p.add_argument("--no-strict", action="store_true")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    return LinterConfig(
        paths=tuple(ns.paths) if ns.paths else DEFAULT_TARGET_PATHS,
        max_lines=ns.max_lines,
        hard_ceiling_lines=ns.hard_ceiling_lines,
        max_args=ns.max_args,
        max_nested_lines=ns.max_nested_lines,
        strict=bool(ns.strict) and not bool(ns.no_strict),
        out_json=ns.out_json,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning 0 if all audited files pass, or 1 on any violation."""
    report = run_linter(parse_args(argv))
    return 0 if bool(report["all_passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
