"""Runtime `TorchDispatchMode` Autograd Efficiency Auditor (`Track 5` / `Wave 44`).

When to Use This Building Block:
- Use `OpCountingDispatchMode`, `AutogradEfficiencyReport`, and `audit_chunked_nll_autograd_efficiency`
  to mechanically verify that custom `torch.autograd.Function` chunked readouts execute a **single**
  backward chunk pass (`bwd_to_fwd_pow_ratio <= 1.05`, `bwd_to_fwd_bmm_ratio <= 3.05`) and zero
  forbidden exponential operations (`aten.exp`, `aten._softmax`, `aten._log_softmax`, `aten.logsumexp`).

Why It Was Built & Mathematical Provenance (`Track 5` / `Wave 44`):
- Prior to Track 5, `CalibratedKroneckerBornChunkedNLLFn.backward` called `_kronecker_chunk_bwd_step`
  followed by `_kronecker_chunk_povm_grad_step` on the same chunk overlap `z_cat`, evaluating
  `_compute_kronecker_chunk_intensity` and `torch.pow(i_tv, boson_p - 1.0)` twice per vocabulary chunk.
- `OpCountingDispatchMode` intercepts ATen operator dispatches during `forward()` and `backward()` separately,
  turning any duplicate chunk pass or accidental exponential op into an immediate unit-test failure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_MAX_BWD_TO_FWD_BMM_RATIO: float = 3.05
DEFAULT_MAX_BWD_TO_FWD_POW_RATIO: float = 1.05
DEFAULT_MAX_BWD_TO_FWD_EXP_RATIO: float = 1.05

TRACKED_ATEN_OPS: tuple[str, ...] = (
    "aten.bmm",
    "aten.mm",
    "aten.addmm",
    "aten.pow",
    "aten.exp",
    "aten.log",
    "aten._softmax",
    "aten._log_softmax",
    "aten.logsumexp",
)

DEFAULT_FORBIDDEN_POLYNOMIAL_OPS: tuple[str, ...] = (
    "aten.exp",
    "aten._softmax",
    "aten._log_softmax",
    "aten.logsumexp",
)


@dataclass(frozen=True)
class AutogradEfficiencyReport:
    """Summary of forward vs backward ATen operator dispatches for a chunked autograd function (`<= 8` fields)."""

    fn_name: str
    num_chunks: int
    fwd_op_counts: dict[str, int]
    bwd_op_counts: dict[str, int]
    bwd_to_fwd_bmm_ratio: float
    bwd_to_fwd_pow_ratio: float
    forbidden_ops_detected: tuple[str, ...]
    passed: bool


def _normalize_aten_op_name(func: object, args: tuple[object, ...] = ()) -> str:
    """Normalize a PyTorch dispatcher OpOverload (`aten.pow.Tensor_Scalar`) to base packet (`aten.pow`).

    Maps quadratic `.square()` calls (`aten.pow` with scalar exponent `2.0`) to `"aten.square"` so
    `"aten.pow"` specifically tracks non-quadratic polynomial Born exponents (`torch.pow(I, p)`).
    """
    schema_name = getattr(getattr(func, "_schema", None), "name", "")
    if schema_name:
        base = schema_name.split("::")[-1].split(".")[0]
        pkt = f"aten.{base}"
    else:
        raw = str(getattr(func, "overloadpacket", func))
        if "::" in raw:
            raw = raw.split("::")[-1]
        parts = raw.split(".")
        pkt = f"aten.{parts[1]}" if (len(parts) >= 2 and parts[0] == "aten") else raw
    if pkt == "aten.pow" and len(args) >= 2 and isinstance(args[1], (int, float)) and float(args[1]) == 2.0:
        return "aten.square"
    return pkt


class OpCountingDispatchMode(TorchDispatchMode):
    """`TorchDispatchMode` that counts ATen tensor operations during `forward` or `backward` execution."""

    def __init__(self) -> None:
        super().__init__()
        self.op_counts: dict[str, int] = {op: 0 for op in TRACKED_ATEN_OPS}

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        op_name = _normalize_aten_op_name(func, args)
        self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
        return func(*args, **(kwargs or {}))


def audit_chunked_nll_autograd_efficiency(
    loss_callable: Callable[[], torch.Tensor],
    *,
    fn_name: str = "ChunkedNLLFn",
    num_chunks: int = 1,
    max_bwd_to_fwd_bmm_ratio: float = DEFAULT_MAX_BWD_TO_FWD_BMM_RATIO,
    max_bwd_to_fwd_pow_ratio: float = DEFAULT_MAX_BWD_TO_FWD_POW_RATIO,
    forbid_exponential_ops: bool = True,
) -> AutogradEfficiencyReport:
    """Profile `loss = loss_callable()` and `loss.backward()` under `OpCountingDispatchMode` (`<= 8` args)."""
    with OpCountingDispatchMode() as fwd_mode:
        loss = loss_callable()
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        raise ValueError(f"{fn_name}: loss_callable() must return a scalar Tensor with requires_grad=True.")

    with OpCountingDispatchMode() as bwd_mode:
        loss.backward()

    fwd_counts = dict(fwd_mode.op_counts)
    bwd_counts = dict(bwd_mode.op_counts)

    fwd_bmm = max(1, int(fwd_counts.get("aten.bmm", 0)))
    bwd_bmm = int(bwd_counts.get("aten.bmm", 0))
    bmm_ratio = float(bwd_bmm) / float(fwd_bmm)

    fwd_pow = max(1, int(fwd_counts.get("aten.pow", 0)))
    bwd_pow = int(bwd_counts.get("aten.pow", 0))
    pow_ratio = float(bwd_pow) / float(fwd_pow)

    forbidden_found: list[str] = []
    if forbid_exponential_ops:
        for op in DEFAULT_FORBIDDEN_POLYNOMIAL_OPS:
            if int(fwd_counts.get(op, 0)) > 0 or int(bwd_counts.get(op, 0)) > 0:
                forbidden_found.append(op)

    passed = (
        (bmm_ratio <= float(max_bwd_to_fwd_bmm_ratio))
        and (pow_ratio <= float(max_bwd_to_fwd_pow_ratio))
        and (len(forbidden_found) == 0)
        and (int(fwd_counts.get("aten.bmm", 0)) >= int(num_chunks))
    )
    return AutogradEfficiencyReport(
        fn_name=str(fn_name),
        num_chunks=int(num_chunks),
        fwd_op_counts=fwd_counts,
        bwd_op_counts=bwd_counts,
        bwd_to_fwd_bmm_ratio=round(bmm_ratio, 4),
        bwd_to_fwd_pow_ratio=round(pow_ratio, 4),
        forbidden_ops_detected=tuple(forbidden_found),
        passed=bool(passed),
    )
