#!/usr/bin/env python3
"""AST Signature, Comment-Free Code-Length (SLOC), and Nested-Closure Linter (`docs/SCRIPT_BEST_PRACTICES.md`).

Promoted from `scripts/BACKLOG.md` (identified in Plan 033, implemented in Plan 034, upgraded in Plan 046).
Audits target Python files via `ast.parse` and `tokenize` for:
- Rule 1: Executable code line count (`code_lines <= max_lines` soft limit, `<= hard_ceiling_lines` hard limit).
  Comments (`# ...`), module/class/function docstrings (`\"\"\"...\"\"\"`), and blank lines do NOT count
  against line limits, encouraging thorough documentation of building blocks and experiment provenance.
- Rule 2: Function/method argument count (`<= max_args` excluding `self`/`cls`).
- Rule 5: Nested function closure executable code length (`<= max_nested_lines` SLOC).
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

# Top-of-file named constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
DEFAULT_SOFT_MAX_CODE_LINES: int = 1000
DEFAULT_HARD_CEILING_CODE_LINES: int = 2500
DEFAULT_MAX_FUNCTION_ARGS: int = 8
DEFAULT_MAX_NESTED_CODE_LINES: int = 40

NON_CODE_TOKEN_TYPES: frozenset[int] = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)

DEFAULT_TARGET_PATHS: tuple[str, ...] = (
    "src/qdlblocks/layers/wavelet_cholesky.py",
    "src/qdlblocks/layers/subspace_coincidence_born.py",
    "src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py",
    "src/qdlblocks/models/parseval_mup_dyadic_gpt.py",
    "src/qdlblocks/models/parseval_mup_subspace_coincidence_gpt.py",
    "src/qdlblocks/benchmarks/config_models.py",
    "scripts/full_ambient_step0_calibration_audit.py",
    "scripts/vram_envelope_autotuner.py",
    "scripts/ast_signature_and_length_linter.py",
    "scripts/multi_subspace_coincidence_margin_probe.py",
    "examples/56_verify_plan046_multi_subspace_coincidence_born.py",
)


@dataclass(frozen=True)
class LinterConfig:
    """Configuration for AST signature, comment-free SLOC length, and nested-closure linting."""

    paths: tuple[str, ...] = DEFAULT_TARGET_PATHS
    max_lines: int = DEFAULT_SOFT_MAX_CODE_LINES
    hard_ceiling_lines: int = DEFAULT_HARD_CEILING_CODE_LINES
    max_args: int = DEFAULT_MAX_FUNCTION_ARGS
    max_nested_lines: int = DEFAULT_MAX_NESTED_CODE_LINES
    strict: bool = False
    out_json: str = ""


def collect_docstring_lines(tree: ast.AST) -> set[int]:
    """Return 1-indexed line numbers spanned by module, class, and function docstrings."""
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.body:
                continue
            first_stmt = node.body[0]
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                start_ln = int(first_stmt.lineno)
                end_ln = int(getattr(first_stmt, "end_lineno", None) or start_ln)
                doc_lines.update(range(start_ln, end_ln + 1))
    return doc_lines


def count_code_and_doc_lines(source: str, tree: ast.AST) -> tuple[int, int, int, set[int]]:
    """Count executable code lines (SLOC) vs. docstring/comment lines.

    Comments (`# ...`), docstrings (`\"\"\"...\"\"\"`), and blank lines do NOT count toward
    `code_lines`, ensuring thorough comments never penalize a file's line budget.
    """
    total_lines = len(source.splitlines())
    doc_lines = collect_docstring_lines(tree)
    comment_lines: set[int] = set()
    code_line_set: set[int] = set()

    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            comment_lines.add(int(tok.start[0]))
            continue
        if tok.type in NON_CODE_TOKEN_TYPES:
            continue
        start_ln, end_ln = int(tok.start[0]), int(tok.end[0])
        if tok.type == tokenize.STRING and start_ln in doc_lines and end_ln in doc_lines:
            continue
        for ln in range(start_ln, end_ln + 1):
            if ln not in doc_lines:
                code_line_set.add(ln)

    doc_and_comment_lines = len((doc_lines | comment_lines) - code_line_set)
    return len(code_line_set), doc_and_comment_lines, total_lines, code_line_set


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


def _is_triton_jit_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if `node` is decorated with `@triton.jit` (or `@jit`), which requires flat pointer/stride args."""
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "jit":
            return True
        if isinstance(target, ast.Name) and target.id == "jit":
            return True
    return False


def inspect_ast_functions(
    tree: ast.AST,
    cfg: LinterConfig,
    code_line_set: set[int],
) -> dict[str, Any]:
    """Traverse AST to collect function argument counts and nested closure executable SLOC."""
    num_functions = 0
    max_fn_args = 0
    max_arg_fn_name = "-"
    max_nested_fn_lines = 0
    max_nested_fn_name = "-"
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            num_functions += 1
            if not _is_triton_jit_function(node):
                n_args = count_function_args(node)
                if n_args > max_fn_args:
                    max_fn_args = n_args
                    max_arg_fn_name = node.name
                if n_args > cfg.max_args:
                    violations.append(f"Rule2: `{node.name}` has {n_args} args (> {cfg.max_args})")

            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_ln = int(child.lineno)
                    end_ln = int(getattr(child, "end_lineno", None) or start_ln)
                    span = sum(1 for ln in range(start_ln, end_ln + 1) if ln in code_line_set)
                    nested_label = f"{node.name}.{child.name}"
                    if span > max_nested_fn_lines:
                        max_nested_fn_lines = span
                        max_nested_fn_name = nested_label
                    if span > cfg.max_nested_lines:
                        violations.append(
                            f"Rule5: nested `{nested_label}` is {span} code lines (> {cfg.max_nested_lines})"
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
    """Audit a Python file against comment-free Rule 1 (1,000 soft, 2,500 hard SLOC), Rule 2, and Rule 5."""
    p = Path(rel_or_abs_path)
    full_path = p if p.is_absolute() else (ROOT / p)
    rel_display = str(full_path.relative_to(ROOT)) if full_path.is_relative_to(ROOT) else str(full_path)

    source = full_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(full_path))
    code_lines, doc_comment_lines, total_lines, code_line_set = count_code_and_doc_lines(source, tree)
    fn_stats = inspect_ast_functions(tree, cfg, code_line_set)

    violations = list(fn_stats["ast_violations"])
    warnings: list[str] = []
    if code_lines > cfg.hard_ceiling_lines:
        violations.append(f"Rule1(HardLimit): {code_lines} code lines (SLOC) > {cfg.hard_ceiling_lines}")
    elif code_lines > cfg.max_lines:
        if cfg.strict:
            violations.append(f"Rule1(SoftLimitStrict): {code_lines} code lines (SLOC) > {cfg.max_lines}")
        else:
            warnings.append(
                f"Rule1(SoftLimitWarning): {code_lines} code lines (SLOC) > {cfg.max_lines} soft limit "
                f"(hard limit {cfg.hard_ceiling_lines})"
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
        "line_count": code_lines,
        "code_lines": code_lines,
        "doc_comment_lines": doc_comment_lines,
        "total_lines": total_lines,
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
        f"\n### AST Best-Practices Compliance Table (`soft_sloc={cfg.max_lines}`, "
        f"`hard_sloc={cfg.hard_ceiling_lines}`, `max_args={cfg.max_args}`, `max_nested_sloc={cfg.max_nested_lines}`)"
    )
    print(
        "| File | Code Lines / SLOC (`Soft` / `Hard`) | Doc & Comment Lines (`Total`) | "
        "Functions | Max Args (`fn`) | Max Nested SLOC | Status |"
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        fn_tag = f"`{r['max_fn_args']}` (`{r['max_arg_fn_name']}`)"
        nest_tag = (
            "`0`"
            if r["max_nested_fn_lines"] == 0
            else f"`{r['max_nested_fn_lines']}` (`{r['max_nested_fn_name']}`)"
        )
        print(
            f"| `{r['path']}` | `{r['code_lines']}` (`{cfg.max_lines}` / `{cfg.hard_ceiling_lines}`) | "
            f"`{r['doc_comment_lines']}` (`{r['total_lines']}`) | "
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
        description="Audit Python files for comment-free code line count (SLOC), function args, and nested closures."
    )
    p.add_argument("paths", nargs="*", default=list(DEFAULT_TARGET_PATHS))
    p.add_argument("--max-lines", type=int, default=DEFAULT_SOFT_MAX_CODE_LINES, help="Soft SLOC limit (default: 1000)")
    p.add_argument(
        "--hard-ceiling-lines",
        type=int,
        default=DEFAULT_HARD_CEILING_CODE_LINES,
        help="Hard SLOC limit failing the linter (default: 2500)",
    )
    p.add_argument("--max-args", type=int, default=DEFAULT_MAX_FUNCTION_ARGS)
    p.add_argument("--max-nested-lines", type=int, default=DEFAULT_MAX_NESTED_CODE_LINES)
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

