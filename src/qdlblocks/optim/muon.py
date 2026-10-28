"""Stiefel-Manifold Muon Optimizers with Real & Joint-Complex Polar Express, NorMuon, and Parameter Classification."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
import torch.nn as nn

from ..functional import (
    COMPLEX_POLAR_EXPRESS_QUINTIC_MAPS,
    STATIONARY_NEWTON_SCHULZ_QUINTIC,
    complex_polar_express_orth,
    normuon_equalize,
    normuon_equalize_complex,
    retract_complex_stiefel_pair,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
    zeropower_via_polarexpress5,
    zeropower_via_polarexpress5_batched,
)


# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_059.md (Wave 44 Tracks 1 & 2)
# ==============================================================================
DEFAULT_MUON_REFERENCE_BATCH_TOKENS: int = 8192
REFERENCE_MUON_BATCH_TOKENS: int = DEFAULT_MUON_REFERENCE_BATCH_TOKENS
MIN_MUON_LR_SCALE: float = 0.25
MAX_MUON_LR_SCALE: float = 8.0
DEFAULT_MUON_BATCH_LR_EXPONENT: float = 0.50
MAX_MUON_BATCH_LR_MULTIPLIER: float = 3.0
MIN_COMPLEX_MUON_MATRIX_DIM: int = 2
MIN_KRONECKER_MUON_OPERATOR_DIM: int = MIN_COMPLEX_MUON_MATRIX_DIM
DEFAULT_KRONECKER_STIEFEL_PENALTY: float = 0.01
DEFAULT_KRONECKER_MUON_STIEFEL_PENALTY: float = DEFAULT_KRONECKER_STIEFEL_PENALTY
_POLAR_EXPRESS_COEFFS_5: tuple[tuple[float, float, float], ...] = COMPLEX_POLAR_EXPRESS_QUINTIC_MAPS

LOOKUP_TABLE_PARAM_SUFFIXES: tuple[str, ...] = (
    "embed_re",
    "embed_im",
    "bi_ctx_wave_re",
    "bi_ctx_wave_im",
    "tri_ctx_wave_re",
    "tri_ctx_wave_im",
    "fourgram_ctx_wave_re",
    "fourgram_ctx_wave_im",
)
POVM_SCALE_PARAM_SUFFIXES: tuple[str, ...] = ("povm_scale_logits",)
NON_OPERATOR_GATE_PARAM_TOKENS: tuple[str, ...] = (
    "branch_logits",
    "ngram_branch_gates",
    "router_w_re",
    "router_w_im",
    "router_kets_re",
    "router_kets_im",
    "branch_router_gain",
    "branch_router_bias",
    "echo_gate_logits",
    "gram_eta",
    "eta_g",
    "krylov_eta",
    "purity_gamma",
    "norm_gain",
    "resid_scale",
    "v_embed_mix",
    "v_mix_logit",
    "smear_gate_logit",
    "carrier_logit_bias",
    "phase_gain",
    "attn_gate",
    "ffn_gate",
    "mixer_gate",
    "psi0_readout_gate",
    "context_branch_bias",
    "lag_gate",
    "fourgram_gate",
)
COMPLEX_MATRIX_OPERATOR_BASENAMES: tuple[str, ...] = (
    "w_q",
    "w_k",
    "w_v",
    "w_out",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "beam1",
    "beam2",
    "proj_down",
    "meas_proj",
    "factor_u",
    "factor_mix",
    "meaning_mix",
    "out_factor_u",
    "bi_factor_u",
    "tri_factor_u",
    "lag2_factor_u",
    "lag3_factor_u",
    "fourgram_factor_u",
    "smear_u",
    "stinespring_w",
)


DecayMode = Literal["smooth", "cautious", "none"]
PolarMethod = Literal["newton_schulz", "polar_express", "polarexpress"]


@dataclass(frozen=True)
class StiefelMuonConfig:
    """Grouped configuration for `StiefelMuon` (`<= 8` fields, Rule 2 of `docs/SCRIPT_BEST_PRACTICES.md`)."""

    nesterov: bool = True
    ns_steps: int = 5
    polar_method: PolarMethod = "newton_schulz"
    stiefel_penalty: float = 0.0
    retract_every: int = 1
    use_normuon: bool = False
    normuon_beta2: float = 0.95
    use_twin_rail: bool = False


@dataclass(frozen=True)
class _ComplexPairUpdateSpec:
    """Per-slice update metadata for Joint-Complex `StiefelMuon` (`<= 8` fields, Rule 2)."""

    p_re: torch.Tensor
    p_im: torch.Tensor
    g_re: torch.Tensor
    g_im: torch.Tensor
    state_re: dict[str, Any]
    lane_key: str
    step_num: int
    group: dict[str, Any]


@dataclass(frozen=True)
class KroneckerParameterPartition:
    """Deterministic 4-group parameter partition for complex Kronecker GPT models (`Wave 44` Track 2)."""

    muon_complex_pairs: list[nn.Parameter]
    embed_params: list[nn.Parameter]
    povm_params: list[nn.Parameter]
    gate_params: list[nn.Parameter]

    @property
    def muon_pair_tuples(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return paired `(p_re, p_im)` 2-tuples from `muon_complex_pairs`."""
        return [
            (self.muon_complex_pairs[2 * i], self.muon_complex_pairs[2 * i + 1])
            for i in range(len(self.muon_complex_pairs) // 2)
        ]

    @property
    def muon_flat_params(self) -> list[nn.Parameter]:
        """Return flat interleaved `[w1_re, w1_im, ...]` list for `group['params']`."""
        return list(self.muon_complex_pairs)


def _validate_complex_pair_group(group: dict[str, Any]) -> None:
    """Validate that a complex pair parameter group has even length and matching 2D/3D `(p_re, p_im)` shapes."""
    params = list(group.get("params", []))
    if len(params) % 2 != 0:
        raise ValueError(
            f"Complex pair group in StiefelMuon requires an even number of parameters, got {len(params)}"
        )
    for idx in range(0, len(params), 2):
        w_re = params[idx]
        w_im = params[idx + 1]
        if w_re.shape != w_im.shape or w_re.dtype != w_im.dtype:
            raise ValueError(
                f"Mismatched complex pair at index {idx // 2}: {tuple(w_re.shape)} ({w_re.dtype}) vs {tuple(w_im.shape)} ({w_im.dtype})"
            )
        if w_re.ndim not in (2, 3) or min(w_re.size(-2), w_re.size(-1)) < MIN_COMPLEX_MUON_MATRIX_DIM:
            raise ValueError(
                f"Complex pair at index {idx // 2} must be 2D or 3D with min(m, n) >= {MIN_COMPLEX_MUON_MATRIX_DIM}, got {tuple(w_re.shape)}"
            )


class StiefelMuon(torch.optim.Optimizer):
    """Stiefel-Manifold Muon Optimizer for Real & Joint-Complex 2D/3D Isometric & Unitary Weight Matrices.

    Supports both standard real parameter groups (`is_complex_pair_group=False`) and paired
    complex matrix operator groups (`is_complex_pair_group=True`, `Wave 44` Track 1) where
    consecutive `(W_re, W_im)` tensors are orthogonalized, NorMuon-equalized, weight-decayed,
    and Stiefel-retracted jointly as `W = W_re + i W_im`.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        decay_mode: DecayMode = "smooth",
        aspect_scale: bool = False,
        config: StiefelMuonConfig | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = config if config is not None else StiefelMuonConfig()
        nesterov = bool(kwargs.pop("nesterov", cfg.nesterov))
        ns_steps = int(kwargs.pop("ns_steps", cfg.ns_steps))
        polar_method = kwargs.pop("polar_method", cfg.polar_method)
        stiefel_penalty = float(kwargs.pop("stiefel_penalty", cfg.stiefel_penalty))
        retract_every = int(kwargs.pop("retract_every", cfg.retract_every))
        use_normuon = bool(kwargs.pop("use_normuon", cfg.use_normuon))
        normuon_beta2 = float(kwargs.pop("normuon_beta2", cfg.normuon_beta2))
        use_twin_rail = bool(kwargs.pop("use_twin_rail", cfg.use_twin_rail))

        momentum_warmup_steps = int(kwargs.pop("momentum_warmup_steps", 0))
        momentum_init = float(kwargs.pop("momentum_init", 0.85))
        batched_ns = bool(kwargs.pop("batched_ns", True))
        ema_momentum = bool(kwargs.pop("ema_momentum", False))
        rail_fast_beta = float(kwargs.pop("rail_fast_beta", 0.85))
        slow_momentum = kwargs.pop("slow_momentum", 0.98)
        rail_slow_beta = float(kwargs.pop("rail_slow_beta", slow_momentum))
        slow_blend = kwargs.pop("slow_blend", None)
        default_fast_w = (1.0 - float(slow_blend)) if slow_blend is not None else 0.4385
        rail_fast_weight = float(kwargs.pop("rail_fast_weight", default_fast_w))
        rail_engage_step = int(kwargs.pop("rail_engage_step", 64))
        stiefel_strength = kwargs.pop("stiefel_strength", None)
        cautious_wd = kwargs.pop("cautious_wd", None)
        is_complex_pair_group = bool(kwargs.pop("is_complex_pair_group", False))
        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s) for StiefelMuon: {sorted(kwargs.keys())}")

        if stiefel_strength is not None:
            stiefel_penalty = float(stiefel_strength)
        if cautious_wd is not None:
            decay_mode = "cautious" if cautious_wd else decay_mode
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if not (0.0 <= momentum_init < 1.0):
            raise ValueError(f"Invalid momentum_init: {momentum_init}")
        if decay_mode not in ("smooth", "cautious", "none"):
            raise ValueError(f"Invalid decay_mode: {decay_mode!r}")
        if polar_method not in ("newton_schulz", "polar_express", "polarexpress"):
            raise ValueError(f"Invalid polar_method: {polar_method!r}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            decay_mode=decay_mode,
            stiefel_penalty=stiefel_penalty,
            ns_steps=ns_steps,
            aspect_scale=aspect_scale,
            retract_every=max(1, retract_every),
            momentum_warmup_steps=max(0, momentum_warmup_steps),
            momentum_init=momentum_init,
            batched_ns=batched_ns,
            ema_momentum=ema_momentum,
            polar_method=polar_method,
            use_normuon=use_normuon,
            normuon_beta2=normuon_beta2,
            use_twin_rail=use_twin_rail,
            rail_fast_beta=rail_fast_beta,
            rail_slow_beta=rail_slow_beta,
            rail_fast_weight=rail_fast_weight,
            rail_engage_step=max(0, rail_engage_step),
            is_complex_pair_group=is_complex_pair_group,
        )
        params_list = list(params)
        if params_list and isinstance(params_list[0], (tuple, list)):
            flat_params = [p for pair in params_list for p in pair]
            params_list = [{"params": flat_params, "is_complex_pair_group": True}]
        super().__init__(params_list, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Normalize complex pair parameter groups (`(w_re, w_im)` tuples or flat list) before registration."""
        raw_params = param_group.get("params", [])
        if not isinstance(raw_params, torch.Tensor):
            raw_list = list(raw_params)
            if raw_list and isinstance(raw_list[0], (tuple, list)):
                param_group["complex_pairs"] = [( pair[0], pair[1] ) for pair in raw_list]
                param_group["params"] = [p for pair in raw_list for p in pair]
                param_group["is_complex_pair_group"] = True
            else:
                param_group["params"] = raw_list
        super().add_param_group(param_group)
        if param_group.get("is_complex_pair_group", False) and len(param_group["params"]) > 0:
            _validate_complex_pair_group(param_group)

    def get_effective_momentum(self, step_num: int, group: dict[str, Any] | None = None) -> float:
        """Compute effective Nesterov momentum at 1-indexed `step_num`."""
        grp = group if group is not None else self.param_groups[0]
        momentum: float = float(grp["momentum"])
        warmup_steps: int = int(grp.get("momentum_warmup_steps", 0))
        if warmup_steps <= 0:
            return momentum
        momentum_init: float = float(grp.get("momentum_init", 0.85))
        frac = min(1.0, max(0.0, float(step_num - 1) / float(max(1, warmup_steps - 1))))
        return momentum_init + (momentum - momentum_init) * frac

    def _compute_momentum_update(
        self,
        g: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
    ) -> tuple[torch.Tensor, int]:
        """Update momentum state (`momentum_buffer`, `slow_momentum_buffer`, `step`) and return `(g_in, step_num)`."""
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)
            state["step"] = 0
        state["step"] = int(state.get("step", 0)) + 1
        step_num: int = state["step"]
        eff_momentum = self.get_effective_momentum(step_num, group)
        state["effective_momentum"] = eff_momentum
        group["last_effective_momentum"] = eff_momentum

        nesterov = bool(group.get("nesterov", True))
        use_twin_rail = bool(group.get("use_twin_rail", False))
        ema_momentum = bool(group.get("ema_momentum", False))
        buf: torch.Tensor = state["momentum_buffer"]

        if use_twin_rail:
            if "slow_momentum_buffer" not in state:
                state["slow_momentum_buffer"] = torch.zeros_like(g)
            slow_buf: torch.Tensor = state["slow_momentum_buffer"]
            rail_fast_beta = float(group.get("rail_fast_beta", 0.85))
            rail_slow_beta = float(group.get("rail_slow_beta", 0.98))
            rail_fast_weight = float(group.get("rail_fast_weight", 0.4385))
            rail_engage_step = int(group.get("rail_engage_step", 64))
            buf.lerp_(g, 1.0 - rail_fast_beta)
            slow_buf.lerp_(g, 1.0 - rail_slow_beta)
            w_rail = 1.0 if step_num < rail_engage_step else rail_fast_weight
            g_blend = w_rail * buf + (1.0 - w_rail) * slow_buf
            g_in = g.lerp(g_blend, eff_momentum) if nesterov else g_blend
        elif ema_momentum:
            buf.lerp_(g, 1.0 - eff_momentum)
            g_in = g.lerp(buf, eff_momentum) if nesterov else buf
        else:
            buf.mul_(eff_momentum).add_(g)
            g_in = g.add(buf, alpha=eff_momentum) if nesterov else buf
        return g_in, step_num

    def _polar_2d(self, g_in: torch.Tensor, ns_steps: int, polar_method: str) -> torch.Tensor:
        if polar_method in ("polar_express", "polarexpress"):
            return zeropower_via_polarexpress5(g_in, steps=ns_steps)
        return zeropower_via_newtonschulz5(g_in, steps=ns_steps)

    def _polar_batched(self, g_stack: torch.Tensor, ns_steps: int, polar_method: str) -> torch.Tensor:
        if polar_method in ("polar_express", "polarexpress"):
            return zeropower_via_polarexpress5_batched(g_stack, steps=ns_steps)
        return zeropower_via_newtonschulz5_batched(g_stack, steps=ns_steps)

    def _polar_complex_2d(
        self,
        g_re: torch.Tensor,
        g_im: torch.Tensor,
        ns_steps: int,
        polar_method: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sched = (
            COMPLEX_POLAR_EXPRESS_QUINTIC_MAPS
            if polar_method in ("polar_express", "polarexpress")
            else (STATIONARY_NEWTON_SCHULZ_QUINTIC,) * max(1, int(ns_steps))
        )
        return complex_polar_express_orth(g_re, g_im, steps=ns_steps, quintic_schedule=sched)

    def _polar_complex_batched(
        self,
        g_re_b: torch.Tensor,
        g_im_b: torch.Tensor,
        ns_steps: int,
        polar_method: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sched = (
            COMPLEX_POLAR_EXPRESS_QUINTIC_MAPS
            if polar_method in ("polar_express", "polarexpress")
            else (STATIONARY_NEWTON_SCHULZ_QUINTIC,) * max(1, int(ns_steps))
        )
        return complex_polar_express_orth(g_re_b, g_im_b, steps=ns_steps, quintic_schedule=sched)

    def _maybe_normuon(
        self,
        u: torch.Tensor,
        state: dict[str, Any],
        state_key: str,
        use_normuon: bool,
        normuon_beta2: float,
    ) -> torch.Tensor:
        if not use_normuon:
            return u
        u_eq, v_next = normuon_equalize(u, state.get(state_key), beta2=normuon_beta2)
        state[state_key] = v_next
        return u_eq

    def _apply_param_update(
        self,
        p: torch.Tensor,
        u: torch.Tensor,
        group: dict[str, Any],
        step_num: int,
    ) -> None:
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        decay_mode: DecayMode = group["decay_mode"]
        stiefel_penalty = float(group["stiefel_penalty"])
        aspect_scale = bool(group["aspect_scale"])
        retract_every = max(1, int(group.get("retract_every", 1)))

        if aspect_scale:
            scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            update = u * scale
        else:
            update = u

        if wd > 0.0:
            if decay_mode == "smooth":
                p.mul_(1.0 - lr * wd)
            elif decay_mode == "cautious":
                cwd_mask = (p * update) > 0.0
                p.add_(p * cwd_mask, alpha=-lr * wd)

        if stiefel_penalty > 0.0 and (step_num % retract_every == 0):
            work_dtype = torch.bfloat16 if p.is_cuda else torch.float32
            w_f = p.to(work_dtype)
            if w_f.size(0) >= w_f.size(1):
                gram = w_f.t() @ w_f
                ident = torch.eye(gram.size(0), device=p.device, dtype=work_dtype)
                diff = gram - ident
                denom = 1.0 + (torch.norm(diff.float(), p="fro") / math.sqrt(diff.size(0)))
                stiefel_grad = (w_f @ (diff / denom)).type_as(p)
            else:
                gram = w_f @ w_f.t()
                ident = torch.eye(gram.size(0), device=p.device, dtype=work_dtype)
                diff = gram - ident
                denom = 1.0 + (torch.norm(diff.float(), p="fro") / math.sqrt(diff.size(0)))
                stiefel_grad = ((diff / denom) @ w_f).type_as(p)
            p.add_(stiefel_grad, alpha=-stiefel_penalty * retract_every)

        p.add_(update, alpha=-lr)

    def _apply_complex_pair_orth_step(
        self,
        spec: _ComplexPairUpdateSpec,
        q_re: torch.Tensor,
        q_im: torch.Tensor,
    ) -> None:
        """Apply Joint-Complex NorMuon equalization, weight decay, step, and complex Stiefel retraction."""
        if bool(spec.group.get("use_normuon", False)):
            beta2 = float(spec.group.get("normuon_beta2", 0.95))
            q_re, q_im, v_joint = normuon_equalize_complex(
                q_re,
                q_im,
                spec.state_re.get(spec.lane_key),
                beta2=beta2,
            )
            spec.state_re[spec.lane_key] = v_joint
            if spec.lane_key == "lane_energy":
                spec.state_re["v_joint"] = v_joint

        self._apply_complex_param_update(
            spec.p_re,
            spec.p_im,
            q_re,
            q_im,
            spec.group,
            spec.step_num,
        )

    def _apply_complex_param_update(
        self,
        w_re: torch.Tensor,
        w_im: torch.Tensor,
        u_re: torch.Tensor,
        u_im: torch.Tensor,
        group: dict[str, Any],
        step_num: int,
    ) -> None:
        """Apply weight decay, orthogonalized step, and complex Stiefel retraction on 2D pair `(w_re, w_im)`."""
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        decay_mode = group["decay_mode"]
        aspect_scale = bool(group.get("aspect_scale", False))
        stiefel_penalty = float(group.get("stiefel_penalty", 0.0))
        retract_every = max(1, int(group.get("retract_every", 1)))

        m, n = w_re.size(0), w_re.size(1)
        scale = math.sqrt(max(1.0, float(m) / float(n))) if aspect_scale else 1.0
        step_scale = lr * scale

        if wd > 0.0 and decay_mode != "none":
            if decay_mode == "cautious":
                mask = ((w_re * u_re + w_im * u_im) > 0.0).to(dtype=w_re.dtype)
                w_re.sub_(w_re * mask, alpha=lr * wd)
                w_im.sub_(w_im * mask, alpha=lr * wd)
            else:
                w_re.mul_(1.0 - lr * wd)
                w_im.mul_(1.0 - lr * wd)

        w_re.add_(u_re.to(dtype=w_re.dtype), alpha=-step_scale)
        w_im.add_(u_im.to(dtype=w_im.dtype), alpha=-step_scale)

        if stiefel_penalty > 0.0 and (step_num % retract_every == 0):
            retract_complex_stiefel_pair(w_re, w_im, penalty=stiefel_penalty * float(retract_every))

    def _step_joint_complex_pairs(self, group: dict[str, Any]) -> None:
        """Execute Joint-Complex `StiefelMuon` step over paired `(w_re, w_im)` 2D and 3D parameters."""
        _validate_complex_pair_group(group)
        ns_steps = int(group["ns_steps"])
        polar_method = str(group.get("polar_method", "polar_express"))
        batched_ns = bool(group.get("batched_ns", True))
        params = group["params"]

        shape_buckets: dict[
            tuple[tuple[int, ...], torch.device, torch.dtype],
            list[_ComplexPairUpdateSpec],
        ] = {}

        for idx in range(0, len(params), 2):
            p_re = params[idx]
            p_im = params[idx + 1]
            if p_re.grad is None and p_im.grad is None:
                continue
            if (p_re.grad is None) != (p_im.grad is None):
                raise ValueError("Mismatched None grad in complex pair")

            state_re = self.state[p_re]
            state_im = self.state[p_im]
            g_re, step_num = self._compute_momentum_update(p_re.grad, state_re, group)
            g_im, _ = self._compute_momentum_update(p_im.grad, state_im, group)

            if p_re.ndim == 2:
                specs = [_ComplexPairUpdateSpec(p_re, p_im, g_re, g_im, state_re, "lane_energy", step_num, group)]
            else:
                specs = [
                    _ComplexPairUpdateSpec(
                        p_re[s],
                        p_im[s],
                        g_re[s],
                        g_im[s],
                        state_re,
                        f"lane_energy_{s}",
                        step_num,
                        group,
                    )
                    for s in range(p_re.size(0))
                ]

            for spec in specs:
                if batched_ns:
                    key = (tuple(spec.p_re.shape), spec.p_re.device, spec.g_re.dtype)
                    shape_buckets.setdefault(key, []).append(spec)
                else:
                    q_re, q_im = self._polar_complex_2d(spec.g_re, spec.g_im, ns_steps, polar_method)
                    self._apply_complex_pair_orth_step(spec, q_re, q_im)

        if batched_ns:
            for bucket in shape_buckets.values():
                if len(bucket) == 1:
                    spec = bucket[0]
                    q_re, q_im = self._polar_complex_2d(spec.g_re, spec.g_im, ns_steps, polar_method)
                    self._apply_complex_pair_orth_step(spec, q_re, q_im)
                else:
                    g_re_b = torch.stack([s.g_re for s in bucket], dim=0)
                    g_im_b = torch.stack([s.g_im for s in bucket], dim=0)
                    q_re_b, q_im_b = self._polar_complex_batched(g_re_b, g_im_b, ns_steps, polar_method)
                    for idx_b, spec in enumerate(bucket):
                        self._apply_complex_pair_orth_step(spec, q_re_b[idx_b], q_im_b[idx_b])

    _step_complex_pair_group = _step_joint_complex_pairs

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("is_complex_pair_group", False):
                self._step_joint_complex_pairs(group)
                continue

            ns_steps: int = group["ns_steps"]
            batched_ns: bool = group.get("batched_ns", True)
            polar_method: str = group.get("polar_method", "newton_schulz")
            use_normuon: bool = group.get("use_normuon", False)
            normuon_beta2: float = group.get("normuon_beta2", 0.95)

            shape_buckets: dict[
                tuple[tuple[int, ...], torch.device, torch.dtype],
                list[tuple[torch.Tensor, torch.Tensor, int, dict[str, Any], str]],
            ] = {}

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim not in (2, 3):
                    raise ValueError(
                        f"StiefelMuon only supports 2D or 3D batched weight tensors, got shape {tuple(p.shape)}"
                    )

                state = self.state[p]
                g_in, step_num = self._compute_momentum_update(p.grad, state, group)

                if p.ndim == 3:
                    for c_idx in range(p.size(0)):
                        p_c = p[c_idx]
                        g_c = g_in[c_idx]
                        s_key = f"lane_energy_{c_idx}"
                        if batched_ns:
                            key = (tuple(p_c.shape), p.device, g_in.dtype)
                            shape_buckets.setdefault(key, []).append((p_c, g_c, step_num, state, s_key))
                        else:
                            u_c = self._polar_2d(g_c, ns_steps, polar_method)
                            u_c = self._maybe_normuon(u_c, state, s_key, use_normuon, normuon_beta2)
                            self._apply_param_update(p_c, u_c, group, step_num)
                elif p.size(0) == 3 * p.size(1) and p.size(1) > 0:
                    d_sq = p.size(1)
                    p_slices = p.view(3, d_sq, d_sq)
                    g_slices = g_in.view(3, d_sq, d_sq)
                    for c_idx in range(3):
                        p_c = p_slices[c_idx]
                        g_c = g_slices[c_idx]
                        s_key = f"lane_energy_{c_idx}"
                        if batched_ns:
                            key = ((d_sq, d_sq), p.device, g_in.dtype)
                            shape_buckets.setdefault(key, []).append((p_c, g_c, step_num, state, s_key))
                        else:
                            u_c = self._polar_2d(g_c, ns_steps, polar_method)
                            u_c = self._maybe_normuon(u_c, state, s_key, use_normuon, normuon_beta2)
                            self._apply_param_update(p_c, u_c, group, step_num)
                elif batched_ns:
                    key = (tuple(p.shape), p.device, g_in.dtype)
                    shape_buckets.setdefault(key, []).append((p, g_in, step_num, state, "lane_energy"))
                else:
                    u = self._polar_2d(g_in, ns_steps, polar_method)
                    u = self._maybe_normuon(u, state, "lane_energy", use_normuon, normuon_beta2)
                    self._apply_param_update(p, u, group, step_num)

            if batched_ns:
                for bucket in shape_buckets.values():
                    if len(bucket) == 1:
                        p, g_in, step_num, state, s_key = bucket[0]
                        u = self._polar_2d(g_in, ns_steps, polar_method)
                        u = self._maybe_normuon(u, state, s_key, use_normuon, normuon_beta2)
                        self._apply_param_update(p, u, group, step_num)
                    else:
                        g_stack = torch.stack([item[1] for item in bucket], dim=0)
                        u_stack = self._polar_batched(g_stack, ns_steps, polar_method)
                        for idx_b, (p, _, step_num, state, s_key) in enumerate(bucket):
                            u_b = self._maybe_normuon(
                                u_stack[idx_b], state, s_key, use_normuon, normuon_beta2
                            )
                            self._apply_param_update(p, u_b, group, step_num)

        return loss


@torch.no_grad()
def retract_complex_kraus_generators(
    k_re_or_model: torch.Tensor | nn.Module,
    k_im: torch.Tensor | None = None,
    alpha: float = 0.5,
    *,
    stiefel_penalty: float | None = None,
) -> int:
    """Retract 2D/3D complex Kraus/unitary generators `(k_re, k_im)` or all 3D pairs in `model` in-place."""
    eff_alpha = float(stiefel_penalty) if stiefel_penalty is not None else float(alpha)
    if isinstance(k_re_or_model, nn.Module):
        pairs, _ = collect_named_complex_muon_pairs(k_re_or_model)
        count = 0
        for p_re, p_im in pairs:
            if p_re.ndim == 3:
                retract_complex_stiefel_pair(p_re, p_im, penalty=eff_alpha)
                count += 1
        return count
    if k_im is None:
        raise ValueError("k_im must be provided when k_re_or_model is a Tensor")
    retract_complex_stiefel_pair(k_re_or_model, k_im, penalty=eff_alpha)
    return 1


def compute_large_batch_muon_lr_scale(
    batch_tokens: int,
    ref_batch_tokens: int = DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
    exponent: float = DEFAULT_MUON_BATCH_LR_EXPONENT,
    max_multiplier: float = MAX_MUON_LR_SCALE,
    min_multiplier: float = MIN_MUON_LR_SCALE,
) -> float:
    """Compute square-root batch-size learning-rate scale `(B_tok / B_ref)^0.50` clamped to `[0.25, 8.0]`."""
    ratio = float(max(1, int(batch_tokens))) / float(max(1, int(ref_batch_tokens)))
    raw = ratio ** float(exponent)
    return float(min(float(max_multiplier), max(float(min_multiplier), raw)))


def collect_named_complex_muon_pairs(
    model: nn.Module,
    min_dim: int = MIN_COMPLEX_MUON_MATRIX_DIM,
    max_dim: int = 4096,
    *,
    min_matrix_dim: int | None = None,
    as_tuples: bool = True,
) -> tuple[list[Any], set[int]]:
    """Collect paired `(w_re, w_im)` 2D/3D complex matrix operators and their parameter `id()` set.

    Excludes token/context lookup tables (`embed_re/im`, `bi_ctx_wave_re/im`, `tri_ctx_wave_re/im`)
    and non-operator gate/router parameters (`branch_logits`, `ngram_branch_gates`, `echo_gate_logits`, `router_w`).
    """
    eff_min_dim = int(min_matrix_dim) if min_matrix_dim is not None else int(min_dim)
    eff_max_dim = int(max_dim)
    param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
    pairs: list[Any] = []
    assigned_ids: set[int] = set()
    excluded_tokens = ("branch_logits", "ngram_branch_gates", "echo_gate_logits", "router_w", "router_kets")

    for name, p in param_dict.items():
        if not name.endswith("_re") or "embed_" in name or name.endswith(LOOKUP_TABLE_PARAM_SUFFIXES):
            continue
        low_name = name.lower()
        if any(tok in low_name for tok in excluded_tokens):
            continue
        if (
            p.ndim in (2, 3)
            and min(p.size(-2), p.size(-1)) >= eff_min_dim
            and max(p.size(-2), p.size(-1)) <= eff_max_dim
        ):
            im_name = name[:-3] + "_im"
            if im_name not in param_dict:
                raise ValueError(f"Missing _im partner for complex matrix parameter {name}")
            p_im = param_dict[im_name]
            if p.shape != p_im.shape:
                raise ValueError(
                    f"Shape mismatch for complex pair {name} {tuple(p.shape)} vs {im_name} {tuple(p_im.shape)}"
                )
            if as_tuples:
                pairs.append((p, p_im))
            else:
                pairs.extend([p, p_im])
            assigned_ids.add(id(p))
            assigned_ids.add(id(p_im))

    return pairs, assigned_ids


def classify_kronecker_model_parameters(
    model: nn.Module,
    min_operator_dim: int = MIN_COMPLEX_MUON_MATRIX_DIM,
) -> KroneckerParameterPartition:
    """Deterministically partition 100% of a Kronecker model's trainable parameters by mathematical role (`Track 2`).

    Categories:
    1. `embed_params`: Lookup tables (`embed_re/im`, `bi_ctx_wave_re/im`, `tri_ctx_wave_re/im`, `fourgram_ctx_wave_re/im`).
    2. `povm_params`: Per-token POVM scale logits (`povm_scale_logits`).
    3. `gate_params`: Non-operator gates (`branch_logits`, `ngram_branch_gates`, `router_w_re/im`, `echo_gate_logits`,
       `gram_eta`, `meaning_mix_re/im` and `stinespring_w_re/im` when `R == 1`, plus all 1D/0D parameters).
    4. `muon_complex_pairs`: Flat interleaved `[w1_re, w1_im, ...]` of paired 2D/3D complex matrix operators
       with `min(m, n) >= min_operator_dim` (`>= 2`), ready for Joint-Complex `StiefelMuon(is_complex_pair_group=True)`.
    """
    param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
    muon_complex_pairs: list[nn.Parameter] = []
    embed_params: list[nn.Parameter] = []
    povm_params: list[nn.Parameter] = []
    gate_params: list[nn.Parameter] = []
    assigned_ids: set[int] = set()

    for name, p in param_dict.items():
        if id(p) in assigned_ids:
            continue
        low_name = name.lower()
        if name.endswith(LOOKUP_TABLE_PARAM_SUFFIXES):
            embed_params.append(p)
            assigned_ids.add(id(p))
        elif name.endswith(POVM_SCALE_PARAM_SUFFIXES):
            povm_params.append(p)
            assigned_ids.add(id(p))
        elif p.ndim <= 1 or any(tok in low_name for tok in NON_OPERATOR_GATE_PARAM_TOKENS):
            gate_params.append(p)
            assigned_ids.add(id(p))
        elif name.endswith("_re") and any(base in low_name for base in COMPLEX_MATRIX_OPERATOR_BASENAMES):
            im_name = name[:-3] + "_im"
            if im_name not in param_dict:
                raise ValueError(f"Missing _im partner for {name}")
            p_im = param_dict[im_name]
            if p.shape != p_im.shape or p.ndim not in (2, 3):
                raise ValueError(
                    f"Invalid complex matrix operator pair {name} {tuple(p.shape)} vs {im_name} {tuple(p_im.shape)}"
                )
            if min(p.size(-2), p.size(-1)) >= int(min_operator_dim):
                muon_complex_pairs.extend([p, p_im])
            else:
                gate_params.extend([p, p_im])
            assigned_ids.add(id(p))
            assigned_ids.add(id(p_im))
        elif name.endswith("_im") and (name[:-3] + "_re") in param_dict:
            continue
        else:
            raise ValueError(
                f"Unclassified parameter in classify_kronecker_model_parameters: {name} shape={tuple(p.shape)}"
            )

    if len(assigned_ids) != len(param_dict) or len(muon_complex_pairs) % 2 != 0:
        raise ValueError(
            f"Incomplete Kronecker parameter partition: assigned {len(assigned_ids)} / {len(param_dict)}"
        )
    return KroneckerParameterPartition(
        muon_complex_pairs=muon_complex_pairs,
        embed_params=embed_params,
        povm_params=povm_params,
        gate_params=gate_params,
    )


@torch.no_grad()
def polar_snap_weights(
    model: nn.Module,
    steps: int = 5,
    exclude_names: tuple[str, ...] = (
        "wte",
        "lm_head",
        "povm_head",
        "embed",
        "unigram_qubits",
        "higher_states",
        "povm_odd_qubits",
        "vocab_higher_affinity",
        "hier_even_proj",
        "readout_rms_gain",
        "token_radial_scale",
        "qubit_readout_gain",
        "kraus",
        "kboson",
        "diffuse",
        "stereo",
        "phase",
        "gate",
        "mudd",
        "smear",
        "purity",
        "adapter",
        "lambdas",
        "wavelet",
        "cholesky",
        "chol",
        "beamsplitter",
    ),
    *,
    polar_method: str = "newton_schulz",
) -> None:
    """Hard polar retraction snapping 2D/3D weight matrices directly onto the Stiefel manifold."""
    use_pe = polar_method in ("polar_express", "polarexpress")
    skip_odd_qubit_snap = bool(
        getattr(model, "use_last_layer_softmax", False)
        or getattr(model, "use_two_beam_kboson", False)
        or getattr(model, "use_two_beam_kboson_readout", False)
    )
    for name, p in model.named_parameters():
        if p.ndim == 3 and any(sub in name for sub in ("unigram_qubits", "higher_states", "povm_odd_qubits")):
            if "povm_odd_qubits" in name and skip_odd_qubit_snap:
                continue
            q_cnt = max(1, int(p.size(1)))
            sub_scale = 1.0 / math.sqrt(float(q_cnt))
            p.data.copy_(torch.nn.functional.normalize(p.data.float(), p=2.0, dim=-1).to(p.dtype) * sub_scale)
            continue
        if any(excl in name for excl in exclude_names):
            continue
        if p.ndim == 2:
            if use_pe:
                p.data.copy_(zeropower_via_polarexpress5(p.data, steps=steps))
            else:
                p.data.copy_(zeropower_via_newtonschulz5(p.data, steps=steps))
        elif p.ndim == 3 and any(sub in name for sub in ("higher_interaction", "higher_sub_unitary")):
            if use_pe:
                u_3d = zeropower_via_polarexpress5_batched(p.data, steps=steps)
            else:
                u_3d = zeropower_via_newtonschulz5_batched(p.data, steps=steps)
            p.data.copy_(torch.nn.functional.normalize(u_3d.float(), p=2.0, dim=-2).to(p.dtype))
    for mod in model.modules():
        if hasattr(mod, "retract_kraus_stiefel"):
            mod.retract_kraus_stiefel()


def split_muon_adamw_params(
    model: nn.Module,
    adamw_name_patterns: tuple[str, ...] = (
        "wte",
        "lm_head",
        "povm_head",
        "embed",
        "unigram_qubits",
        "higher_states",
        "povm_odd_qubits",
        "vocab_higher_affinity",
        "hier_even_proj",
        "readout_rms_gain",
        "token_radial_scale",
        "qubit_readout_gain",
        "kraus",
        "kboson",
        "diffuse",
        "stereo",
        "phase",
        "gate",
        "mudd",
        "smear",
        "purity",
        "adapter",
        "lambdas",
        "wavelet",
        "cholesky",
        "chol",
        "beamsplitter",
    ),
    muon_3d_name_patterns: tuple[str, ...] = ("higher_interaction", "higher_sub_unitary"),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition model parameters into 2D/3D Stiefel isometries (for Muon) and other params (for AdamW)."""
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 3 and any(pat in name for pat in muon_3d_name_patterns):
            muon_params.append(p)
        elif p.ndim == 2 and not any(pat in name for pat in adamw_name_patterns):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params



