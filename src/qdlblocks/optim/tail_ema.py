"""
Late-Training Exponential Moving Average (EMA) with Frobenius/Spherical Decontraction (Plan 021).

Implements Section 4.2.1 of EXPERIMENT_PLAN_021.md (`KellerJordan/modded-nanogpt`
`track_1_short/tail_average.py`).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn

from ..functional import l2_normalize


class TailWeightAverager:
    """Late-training exponential moving average (EMA) with Frobenius/spherical decontraction."""

    def __init__(
        self,
        model: nn.Module,
        start_frac: float = 0.75,
        ema_rate: float = 4.0 / 53.0,
        blend_alpha: float = 0.65,
        update_every: int = 1,
        decontract_frobenius: bool = True,
        spherical_param_substrings: tuple[str, ...] = ("dm_seed", "effects"),
        *,
        beta: float | None = None,
        decay: float | None = None,
        start_fraction: float | None = None,
        start_step: int | None = None,
        every_n_steps: int | None = None,
        preserve_frobenius_norm: bool | None = None,
        spherical_param_patterns: tuple[str, ...] | None = None,
    ) -> None:
        self.model = model
        self.start_frac = float(start_fraction if start_fraction is not None else start_frac)
        if beta is not None:
            self.ema_rate = 1.0 - float(beta)
        elif decay is not None:
            self.ema_rate = 1.0 - float(decay)
        else:
            self.ema_rate = float(ema_rate)
        self.blend_alpha = float(blend_alpha)
        self.update_every = max(1, int(every_n_steps if every_n_steps is not None else update_every))
        self.start_step_override: int | None = int(start_step) if start_step is not None else None
        self.decontract_frobenius = (
            bool(preserve_frobenius_norm)
            if preserve_frobenius_norm is not None
            else bool(decontract_frobenius)
        )
        self.spherical_param_substrings = tuple(
            spherical_param_patterns
            if spherical_param_patterns is not None
            else spherical_param_substrings
        )

        self.shadow: dict[str, torch.Tensor] = {}
        self.num_updates: int = 0

    @property
    def is_seeded(self) -> bool:
        return len(self.shadow) > 0

    @property
    def active(self) -> bool:
        return len(self.shadow) > 0

    def _named_trainable_params(
        self, model: nn.Module | None = None
    ) -> list[tuple[str, nn.Parameter]]:
        target = model if model is not None else self.model
        seen: set[int] = set()
        out: list[tuple[str, nn.Parameter]] = []
        for name, param in target.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            out.append((name, param))
        return out

    @torch.no_grad()
    def update(self, step: int, total_steps: int = 0) -> bool:
        """Update EMA shadow weights when `step >= int(start_frac * total_steps)`."""
        s = int(step)
        tot = int(total_steps)
        min_step = (
            self.start_step_override
            if self.start_step_override is not None
            else int(self.start_frac * max(0, tot))
        )
        if s < min_step:
            return False
        if (s % self.update_every != 0) and (tot <= 0 or s != tot):
            return False

        params = self._named_trainable_params()
        if not self.is_seeded:
            for name, param in params:
                self.shadow[name] = param.detach().float().clone()
            self.num_updates = 1
            return True

        for name, param in params:
            p_fp32 = param.detach().float()
            if name not in self.shadow:
                self.shadow[name] = p_fp32.clone()
            else:
                self.shadow[name].lerp_(p_fp32, self.ema_rate)
        self.num_updates += 1
        return True

    @torch.no_grad()
    def step(
        self,
        step_or_model: int | nn.Module,
        total_steps: int = 0,
        *,
        current_step: int | None = None,
    ) -> bool:
        """Alias for `update(global_step, total_steps)` (also accepts `(model, current_step=...)`)."""
        if isinstance(step_or_model, nn.Module):
            self.model = step_or_model
            eff_step = int(current_step if current_step is not None else 0)
        else:
            eff_step = int(step_or_model)
        return self.update(eff_step, total_steps=total_steps)

    @contextmanager
    def average_parameters(self, model: nn.Module | None = None) -> Iterator[None]:
        """Temporarily swap live parameters with Frobenius/spherical decontracted EMA blend."""
        if not self.is_seeded:
            yield
            return

        params = self._named_trainable_params(model=model)
        orig_data: dict[str, torch.Tensor] = {}
        alpha = self.blend_alpha
        with torch.no_grad():
            for name, param in params:
                if name not in self.shadow:
                    continue
                orig_data[name] = param.data.detach().clone()
                w_cur = param.data.detach().float()
                w_ema = self.shadow[name].to(device=w_cur.device, dtype=torch.float32)
                w_blend = (1.0 - alpha) * w_cur + alpha * w_ema

                if (
                    w_blend.ndim == 2
                    and any(sub in name for sub in self.spherical_param_substrings)
                ):
                    w_ship = l2_normalize(w_blend, dim=-1, eps=1e-12)
                elif (
                    w_blend.ndim == 3
                    and any(
                        sub in name
                        for sub in ("unigram_qubits", "higher_states", "povm_odd_qubits")
                    )
                ):
                    cur_sub_norm = torch.linalg.norm(w_cur, dim=-1, keepdim=True)
                    w_ship = l2_normalize(w_blend, dim=-1, eps=1e-12) * cur_sub_norm
                elif (
                    self.decontract_frobenius
                    and w_blend.ndim == 2
                    and any(
                        sub in name
                        for sub in (
                            "wte",
                            "value_embeds",
                            "ngram",
                            "unigram_sparse_vals",
                            "povm_odd_delta",
                            "vocab_higher_affinity",
                        )
                    )
                ):
                    cur_row_norm = torch.linalg.norm(w_cur, dim=-1, keepdim=True)
                    blend_row_norm = torch.linalg.norm(w_blend, dim=-1, keepdim=True)
                    w_ship = w_blend * (cur_row_norm / (blend_row_norm + 1e-12))
                elif self.decontract_frobenius and w_blend.ndim >= 2:
                    cur_norm = torch.linalg.norm(w_cur)
                    blend_norm = torch.linalg.norm(w_blend)
                    w_ship = w_blend * (cur_norm / (blend_norm + 1e-12))
                else:
                    w_ship = w_blend

                param.data.copy_(w_ship.to(param.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in params:
                    if name in orig_data:
                        param.data.copy_(orig_data[name])

    @contextmanager
    def swap_to_ema(self, model: nn.Module | None = None) -> Iterator[None]:
        """Alias for `average_parameters`."""
        with self.average_parameters(model=model):
            yield


__all__ = ["TailWeightAverager"]
