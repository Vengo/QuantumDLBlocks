"""Dynamic Per-Tensor FP8 (`E4M3`) Linear Wrapper with Safe BF16 Fallback (Experiment Plan 021).

Provides `FP8LinearWrapper`, `is_fp8_scaled_mm_supported`, and `convert_linear_layers_to_fp8`.
During training on CUDA architectures supporting FP8 Tensor Cores (SM89+ Ada, Hopper, Blackwell)
when matrix dimensions are multiples of 16, dispatches forward matmuls through `torch._scaled_mm`
with dynamic per-tensor `amax / 448.0` scaling, falling back transparently to `F.linear` during
`eval()` or on devices/shapes that do not satisfy `torch._scaled_mm` alignment constraints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_fp8_scaled_mm_supported(device: torch.device | str | None = None) -> bool:
    """Return True if the target device supports `torch._scaled_mm` FP8 (`e4m3fn`) matmuls."""
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return False
    if device is not None:
        dev = torch.device(device)
        if dev.type != "cuda":
            return False
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
    else:
        idx = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(idx)
    return (major, minor) >= (8, 9)


class _FP8ScaledMMFunction(torch.autograd.Function):
    """Autograd function executing FP8 E4M3 `torch._scaled_mm` in forward and exact GEMM backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        in_features = x.size(-1)
        out_features = weight.size(0)
        x_2d = x.reshape(-1, in_features).contiguous()
        m_rows = x_2d.size(0)

        out_dtype = (
            torch.bfloat16
            if (x.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16)
            else (torch.float16 if x.dtype == torch.float16 else torch.bfloat16)
        )

        # Dynamic per-tensor E4M3 quantization (max representable value in float8_e4m3fn is 448.0)
        f8_max = 448.0
        amax_x = x_2d.float().abs().amax().clamp(min=1e-12)
        amax_w = weight.float().abs().amax().clamp(min=1e-12)
        scale_x = (amax_x / f8_max).to(torch.float32)
        scale_w = (amax_w / f8_max).to(torch.float32)

        x_fp8 = (x_2d.float() / scale_x).clamp(-f8_max, f8_max).to(torch.float8_e4m3fn)
        w_fp8 = (weight.contiguous().float() / scale_w).clamp(-f8_max, f8_max).to(torch.float8_e4m3fn)

        # torch._scaled_mm expects row-major A (M, K) and column-major B (K, N) = w_fp8.t()
        try:
            res = torch._scaled_mm(
                x_fp8,
                w_fp8.t(),
                scale_a=scale_x,
                scale_b=scale_w,
                out_dtype=out_dtype,
            )
            out_2d = res[0] if isinstance(res, tuple) else res
            if bias is not None:
                out_2d = out_2d + bias.to(out_2d.dtype)
            out_2d = out_2d.to(x.dtype)
        except Exception:
            w_work = weight.to(x_2d.dtype)
            b_work = bias.to(x_2d.dtype) if bias is not None else None
            out_2d = F.linear(x_2d, w_work, b_work)

        ctx.save_for_backward(x_2d, weight, bias if bias is not None else torch.empty(0, device=x.device))
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        return out_2d.reshape(*orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x_2d, weight, bias_saved = ctx.saved_tensors
        in_features = x_2d.size(-1)
        out_features = weight.size(0)
        grad_2d = grad_output.reshape(-1, out_features)

        work_dtype = (
            torch.get_autocast_gpu_dtype()
            if (x_2d.is_cuda and torch.is_autocast_enabled())
            else x_2d.dtype
        )
        g_work = grad_2d.to(work_dtype)
        x_work = x_2d.to(work_dtype)
        w_work = weight.to(work_dtype)

        grad_x = (
            torch.matmul(g_work, w_work).to(x_2d.dtype).reshape(ctx.orig_shape)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_w = (
            torch.matmul(g_work.t(), x_work).to(weight.dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        grad_b = (
            grad_2d.float().sum(dim=0).to(bias_saved.dtype)
            if (ctx.has_bias and ctx.needs_input_grad[2])
            else None
        )
        return grad_x, grad_w, grad_b


class FP8LinearWrapper(nn.Module):
    """Drop-in FP8 wrapper around `nn.Linear` preserving `.weight` and `.bias` parameter paths."""

    def __init__(
        self,
        linear_or_in_features: nn.Linear | int,
        out_features: int | None = None,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        if isinstance(linear_or_in_features, nn.Linear):
            self.in_features = int(linear_or_in_features.in_features)
            self.out_features = int(linear_or_in_features.out_features)
            self.weight = linear_or_in_features.weight
            self.bias = linear_or_in_features.bias
        else:
            if out_features is None:
                raise ValueError("`out_features` must be provided when initializing FP8LinearWrapper from ints.")
            self.in_features = int(linear_or_in_features)
            self.out_features = int(out_features)
            lin = nn.Linear(self.in_features, self.out_features, bias=bias, device=device, dtype=dtype)
            self.weight = lin.weight
            self.bias = lin.bias

    def _can_use_fp8(self, x: torch.Tensor) -> bool:
        if not self.enabled or not self.training or not x.is_cuda:
            return False
        m_rows = x.numel() // max(1, self.in_features)
        if (m_rows % 16 != 0) or (self.in_features % 16 != 0) or (self.out_features % 16 != 0):
            return False
        return is_fp8_scaled_mm_supported(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._can_use_fp8(x):
            return _FP8ScaledMMFunction.apply(x, self.weight, self.bias)
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        return F.linear(x, w, b)


def convert_linear_layers_to_fp8(
    model: nn.Module,
    skip_names: tuple[str, ...] = ("lm_head", "wte", "povm", "gate", "mudd"),
    enabled: bool = True,
) -> nn.Module:
    """Recursively wrap eligible `nn.Linear` submodules in `FP8LinearWrapper` in place."""
    if not enabled:
        return model
    for name, child in list(model.named_children()):
        if any(s in name for s in skip_names):
            continue
        if isinstance(child, nn.Linear) and not isinstance(child, FP8LinearWrapper):
            if child.in_features % 16 == 0 and child.out_features % 16 == 0 and min(child.in_features, child.out_features) >= 64:
                setattr(model, name, FP8LinearWrapper(child, enabled=enabled))
        else:
            convert_linear_layers_to_fp8(child, skip_names=skip_names, enabled=enabled)
    return model


__all__ = [
    "is_fp8_scaled_mm_supported",
    "FP8LinearWrapper",
    "convert_linear_layers_to_fp8",
]
