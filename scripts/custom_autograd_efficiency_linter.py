#!/usr/bin/env python3
"""Static AST Custom Autograd Efficiency Linter (`Track 5` / `Wave 44`).

Scans `src/qdlblocks/models` and `src/qdlblocks/layers` for all `torch.autograd.Function`
subclasses and enforces four structural invariants:
1. `R1 (DuplicateChunkStepCall)`: No `backward()` (or module-level helper it calls) may invoke
   multiple chunk-step / intensity / grad-step functions on the same chunk overlap tensor `z_cat`
   in the same execution block (preventing the `Wave 42–43` double-pass `_kronecker_chunk_povm_grad_step` bug).
2. `R2 (RedundantChunkLoops)`: No single statement block inside `forward()` or `backward()` may
   contain `>= 2` sibling `for` loops over vocabulary chunks (`range(0, ..., v_chunk)`).
3. `R3 (ForbiddenExponentialOp)`: No polynomial Born `torch.autograd.Function` (`*Kronecker*` / `*Born*`)
   or its helper functions may call `exp`, `softmax`, `log_softmax`, or `logsumexp`.
4. `R4 (OversizedSaveForBackward)`: `ctx.save_for_backward(...)` must save `<= 10` tensors.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_SCAN_PATHS: tuple[str, ...] = ("src/qdlblocks/models", "src/qdlblocks/layers")
MAX_SAVED_TENSORS_FOR_BACKWARD: int = 10
MAX_SIBLING_CHUNK_LOOPS_PER_BLOCK: int = 1
MAX_CHUNK_STEP_CALLS_PER_TENSOR: int = 1

FORBIDDEN_POLYNOMIAL_CALL_NAMES: tuple[str, ...] = (
    "exp",
    "softmax",
    "log_softmax",
    "logsumexp",
)

CHUNK_STEP_FN_SUBSTRINGS: tuple[str, ...] = (
    "chunk_bwd_step",
    "chunk_povm_grad_step",
    "chunk_grad_step",
    "chunk_intensity",
)


@dataclass(frozen=True)
class AutogradClassAudit:
    """Static AST audit record for a single `torch.autograd.Function` subclass (`<= 8` fields)."""

    file_path: str
    class_name: str
    line_no: int
    saved_tensor_count: int
    max_sibling_chunk_loops: int
    duplicate_step_calls: tuple[str, ...]
    forbidden_exp_calls: tuple[str, ...]
    violations: tuple[str, ...]


def _is_autograd_function_base(base: ast.expr) -> bool:
    """Return True if `base` AST node references `torch.autograd.Function` or `Function`."""
    if isinstance(base, ast.Name):
        return base.id == "Function"
    if isinstance(base, ast.Attribute):
        return base.attr == "Function"
    return False


def _get_call_func_name(call_node: ast.Call) -> str:
    """Extract simple function or attribute name from an `ast.Call` node."""
    if isinstance(call_node.func, ast.Name):
        return call_node.func.id
    if isinstance(call_node.func, ast.Attribute):
        return call_node.func.attr
    return ""


def _is_chunk_range_for_loop(stmt: ast.stmt) -> bool:
    """Return True if `stmt` is a `for ... in range(0, ..., chunk)` vocabulary/token chunk loop."""
    if not isinstance(stmt, ast.For):
        return False
    if not isinstance(stmt.iter, ast.Call):
        return False
    if _get_call_func_name(stmt.iter) != "range":
        return False
    return len(stmt.iter.args) >= 3


def _max_sibling_chunk_loops_in_stmts(stmts: list[ast.stmt]) -> int:
    """Compute maximum sibling chunk `for` loops appearing in the same statement list."""
    direct_count = sum(1 for s in stmts if _is_chunk_range_for_loop(s))
    max_count = direct_count
    for s in stmts:
        for attr in ("body", "orelse", "finalbody"):
            sub = getattr(s, attr, None)
            if isinstance(sub, list) and len(sub) > 0:
                max_count = max(max_count, _max_sibling_chunk_loops_in_stmts(sub))
    return max_count


def _find_duplicate_chunk_step_calls_in_block(stmts: list[ast.stmt]) -> list[str]:
    """Detect `>= 2` chunk step / intensity calls on the same tensor argument within one block."""
    duplicates: list[str] = []
    tensor_to_calls: dict[str, list[str]] = {}
    step_calls_in_block: list[str] = []

    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                fn_name = _get_call_func_name(node)
                if any(sub in fn_name for sub in CHUNK_STEP_FN_SUBSTRINGS):
                    step_calls_in_block.append(fn_name)
                    first_arg = node.args[0] if len(node.args) > 0 else None
                    if isinstance(first_arg, ast.Name):
                        tensor_to_calls.setdefault(first_arg.id, []).append(fn_name)
        for attr in ("body", "orelse"):
            sub = getattr(stmt, attr, None)
            if isinstance(sub, list) and len(sub) > 0:
                duplicates.extend(_find_duplicate_chunk_step_calls_in_block(sub))

    for arg_name, fn_list in tensor_to_calls.items():
        if len(fn_list) > MAX_CHUNK_STEP_CALLS_PER_TENSOR:
            duplicates.append(f"{arg_name}:{'+'.join(fn_list)}")
    if len(step_calls_in_block) >= 2 and any("povm_grad_step" in c for c in step_calls_in_block):
        tag = "+".join(step_calls_in_block)
        if tag not in duplicates:
            duplicates.append(tag)
    return duplicates


def _collect_forbidden_exp_calls(node: ast.AST) -> list[str]:
    """Collect any calls to `exp`, `softmax`, `log_softmax`, or `logsumexp` inside `node`."""
    found: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn_name = _get_call_func_name(sub)
            if fn_name in FORBIDDEN_POLYNOMIAL_CALL_NAMES:
                found.append(f"{fn_name}@L{getattr(sub, 'lineno', 0)}")
    return found


def _audit_single_autograd_class(
    cls_node: ast.ClassDef,
    module_funcs: dict[str, ast.FunctionDef],
    file_path: str,
    is_polynomial_born: bool,
) -> AutogradClassAudit:
    """Audit a single `torch.autograd.Function` class AST node (`<= 8` args)."""
    saved_count = 0
    max_loops = 0
    dup_steps: list[str] = []
    forbidden_exps: list[str] = []
    poly_check = bool(is_polynomial_born or "Born" in cls_node.name or "Kronecker" in cls_node.name)

    for item in cls_node.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        if item.name in ("forward", "backward"):
            max_loops = max(max_loops, _max_sibling_chunk_loops_in_stmts(item.body))
            if poly_check:
                forbidden_exps.extend(_collect_forbidden_exp_calls(item))
        if item.name == "forward":
            for sub in ast.walk(item):
                if isinstance(sub, ast.Call) and _get_call_func_name(sub) == "save_for_backward":
                    saved_count = max(saved_count, len(sub.args))
        if item.name == "backward":
            dup_steps.extend(_find_duplicate_chunk_step_calls_in_block(item.body))
            for sub in ast.walk(item):
                if isinstance(sub, ast.Call):
                    callee = _get_call_func_name(sub)
                    helper_fn = module_funcs.get(callee)
                    if helper_fn is not None:
                        dup_steps.extend(_find_duplicate_chunk_step_calls_in_block(helper_fn.body))
                        if poly_check:
                            forbidden_exps.extend(_collect_forbidden_exp_calls(helper_fn))

    dedup_steps = tuple(dict.fromkeys(dup_steps))
    dedup_exps = tuple(dict.fromkeys(forbidden_exps))
    violations: list[str] = []
    if len(dedup_steps) > 0:
        violations.append(f"R1(DuplicateChunkStepCall): {', '.join(dedup_steps)}")
    if max_loops > MAX_SIBLING_CHUNK_LOOPS_PER_BLOCK:
        violations.append(
            f"R2(RedundantChunkLoops): {max_loops} sibling chunk loops > {MAX_SIBLING_CHUNK_LOOPS_PER_BLOCK}"
        )
    if len(dedup_exps) > 0:
        violations.append(f"R3(ForbiddenExponentialOp): {', '.join(dedup_exps)}")
    if saved_count > MAX_SAVED_TENSORS_FOR_BACKWARD:
        violations.append(
            f"R4(OversizedSaveForBackward): {saved_count} saved tensors > {MAX_SAVED_TENSORS_FOR_BACKWARD}"
        )

    return AutogradClassAudit(
        file_path=file_path,
        class_name=cls_node.name,
        line_no=int(cls_node.lineno),
        saved_tensor_count=int(saved_count),
        max_sibling_chunk_loops=int(max_loops),
        duplicate_step_calls=dedup_steps,
        forbidden_exp_calls=dedup_exps,
        violations=tuple(violations),
    )


def lint_autograd_source(
    source_code: str,
    file_path: str = "<memory>",
    is_polynomial_born: bool | None = None,
) -> tuple[list[AutogradClassAudit], list[str]]:
    """Parse `source_code` and audit all `torch.autograd.Function` classes (`<= 8` args)."""
    tree = ast.parse(source_code, filename=file_path)
    module_funcs: dict[str, ast.FunctionDef] = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    poly_file = (
        ("kronecker" in file_path.lower())
        if is_polynomial_born is None
        else bool(is_polynomial_born)
    )
    audits: list[AutogradClassAudit] = []
    all_violations: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(_is_autograd_function_base(b) for b in node.bases):
            audit = _audit_single_autograd_class(node, module_funcs, file_path, poly_file)
            audits.append(audit)
            for v in audit.violations:
                all_violations.append(f"{file_path}:{node.lineno} ({node.name}): {v}")
    return audits, all_violations


def scan_paths_for_autograd_functions(
    paths: list[str | Path],
) -> tuple[list[AutogradClassAudit], list[str]]:
    """Recursively scan Python files in `paths` for `torch.autograd.Function` classes (`<= 8` args)."""
    py_files: list[Path] = []
    for p_raw in paths:
        p = Path(p_raw)
        if p.is_file() and p.suffix == ".py":
            py_files.append(p)
        elif p.is_dir():
            py_files.extend(sorted(p.rglob("*.py")))

    all_audits: list[AutogradClassAudit] = []
    all_violations: list[str] = []
    seen: set[Path] = set()
    for py_file in py_files:
        resolved = py_file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        src = py_file.read_text(encoding="utf-8")
        audits, violations = lint_autograd_source(src, file_path=str(py_file))
        all_audits.extend(audits)
        all_violations.extend(violations)
    return all_audits, all_violations


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `scripts/custom_autograd_efficiency_linter.py`."""
    parser = argparse.ArgumentParser(description="Static AST Custom Autograd Efficiency Linter")
    parser.add_argument(
        "pos_paths",
        nargs="*",
        help="Optional positional file or directory paths to scan.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="File or directory paths to scan (default: src/qdlblocks/models src/qdlblocks/layers).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write JSON audit report.",
    )
    args = parser.parse_args(argv)

    target_paths: list[str] = []
    if args.paths:
        target_paths.extend(args.paths)
    if args.pos_paths:
        target_paths.extend(args.pos_paths)
    if len(target_paths) == 0:
        target_paths = list(DEFAULT_SCAN_PATHS)

    audits, violations = scan_paths_for_autograd_functions(target_paths)

    print("\n### Custom Autograd Efficiency AST Audit Table")
    print(
        "| File | Autograd Class (`Line`) | Saved Tensors (`<=10`) | Sibling Chunk Loops (`<=1`) | Duplicate Step Calls | Forbidden Exp Calls | Status |"
    )
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for a in audits:
        status = "**PASS**" if len(a.violations) == 0 else "**FAIL**"
        dup_str = ", ".join(a.duplicate_step_calls) if a.duplicate_step_calls else "0"
        exp_str = ", ".join(a.forbidden_exp_calls) if a.forbidden_exp_calls else "0"
        print(
            f"| `{a.file_path}` | `{a.class_name}` (`L{a.line_no}`) | `{a.saved_tensor_count}` | "
            f"`{a.max_sibling_chunk_loops}` | `{dup_str}` | `{exp_str}` | {status} |"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scanned_paths": target_paths,
            "num_autograd_classes": len(audits),
            "num_violations": len(violations),
            "passed": len(violations) == 0,
            "audits": [asdict(a) for a in audits],
            "violations": violations,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if len(violations) > 0:
        print("\nViolations detected:")
        for v in violations:
            print(f"  - {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
