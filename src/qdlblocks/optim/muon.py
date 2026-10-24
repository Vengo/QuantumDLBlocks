"""Stiefel-Manifold Muon Optimizers with Smooth Decay, Cautious Weight Decay, and Riemannian Retraction."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Literal

import torch
import torch.nn as nn

from ..functional import (
    normuon_equalize,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
    zeropower_via_polarexpress5,
    zeropower_via_polarexpress5_batched,
)


DecayMode = Literal["smooth", "cautious", "none"]
PolarMethod = Literal["newton_schulz", "polar_express", "polarexpress"]


class StiefelMuon(torch.optim.Optimizer):
    """Stiefel-Manifold Muon Optimizer for 2D Isometric & Unitary Weight Matrices.

    Computes orthogonalized momentum updates on the Stiefel manifold via quintic
    Newton-Schulz or PolarExpress/ANVIL polar iterations, and supports:
    1. **Smooth Weight Decay (`decay_mode="smooth"`)**:
       Applies continuous multiplicative weight decay `W <- (1 - lr * wd) * W` before
       stepping along the polar direction.
    2. **Cautious Weight Decay (`decay_mode="cautious"`)**:
       Applies weight decay only to coordinates where the parameter sign aligns with
       the polar update direction `(W * U > 0)`.
    3. **Continuous Riemannian Stiefel Penalty (`stiefel_penalty > 0`)**:
       Applies a soft gradient step of the Stiefel manifold potential `1/4 ||W^T W - I||_F^2`.
    4. **Muon Momentum Warmup (`momentum_warmup_steps > 0`)**:
       Linearly ramps Nesterov momentum from `momentum_init` (e.g. `0.85`) to `momentum` (e.g. `0.95`).
    5. **Batched 3D Newton-Schulz / PolarExpress (`batched_ns=True`)**:
       Stacks same-shape 2D gradient updates into 3D `(K, M, N)` batches for single-call `torch.bmm` orthogonalization.
    6. **NorMuon Second-Moment Equalization (`use_normuon=True`)**:
       Equalizes per-lane RMS energy after polar orthogonalization.
    7. **Twin-Rail Momentum (`use_twin_rail=True`)**:
       Blends fast (`rail_fast_beta`) and slow (`rail_slow_beta`) EMA momentum rails before polar projection.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        decay_mode: DecayMode = "smooth",
        stiefel_penalty: float = 0.0,
        ns_steps: int = 5,
        aspect_scale: bool = False,
        retract_every: int = 1,
        momentum_warmup_steps: int = 0,
        momentum_init: float = 0.85,
        batched_ns: bool = True,
        ema_momentum: bool = False,
        polar_method: PolarMethod = "newton_schulz",
        use_normuon: bool = False,
        normuon_beta2: float = 0.95,
        use_twin_rail: bool = False,
        rail_fast_beta: float = 0.85,
        rail_slow_beta: float = 0.98,
        rail_fast_weight: float = 0.4385,
        rail_engage_step: int = 64,
        *,
        stiefel_strength: float | None = None,
        cautious_wd: bool | None = None,
    ) -> None:
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
            nesterov=bool(nesterov),
            weight_decay=weight_decay,
            decay_mode=decay_mode,
            stiefel_penalty=stiefel_penalty,
            ns_steps=ns_steps,
            aspect_scale=aspect_scale,
            retract_every=max(1, int(retract_every)),
            momentum_warmup_steps=max(0, int(momentum_warmup_steps)),
            momentum_init=float(momentum_init),
            batched_ns=bool(batched_ns),
            ema_momentum=bool(ema_momentum),
            polar_method=polar_method,
            use_normuon=bool(use_normuon),
            normuon_beta2=float(normuon_beta2),
            use_twin_rail=bool(use_twin_rail),
            rail_fast_beta=float(rail_fast_beta),
            rail_slow_beta=float(rail_slow_beta),
            rail_fast_weight=float(rail_fast_weight),
            rail_engage_step=max(0, int(rail_engage_step)),
        )
        super().__init__(params, defaults)

    def get_effective_momentum(self, step_num: int, group: dict | None = None) -> float:
        """Compute effective Nesterov momentum at 1-indexed `step_num`."""
        grp = group if group is not None else self.param_groups[0]
        momentum: float = float(grp["momentum"])
        warmup_steps: int = int(grp.get("momentum_warmup_steps", 0))
        if warmup_steps <= 0:
            return momentum
        momentum_init: float = float(grp.get("momentum_init", 0.85))
        frac = min(1.0, max(0.0, float(step_num - 1) / float(max(1, warmup_steps - 1))))
        return momentum_init + (momentum - momentum_init) * frac

    def _polar_2d(self, g_in: torch.Tensor, ns_steps: int, polar_method: str) -> torch.Tensor:
        if polar_method in ("polar_express", "polarexpress"):
            return zeropower_via_polarexpress5(g_in, steps=ns_steps)
        return zeropower_via_newtonschulz5(g_in, steps=ns_steps)

    def _polar_batched(self, g_stack: torch.Tensor, ns_steps: int, polar_method: str) -> torch.Tensor:
        if polar_method in ("polar_express", "polarexpress"):
            return zeropower_via_polarexpress5_batched(g_stack, steps=ns_steps)
        return zeropower_via_newtonschulz5_batched(g_stack, steps=ns_steps)

    def _maybe_normuon(
        self,
        u: torch.Tensor,
        state: dict,
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
        step_num: int,
        lr: float,
        wd: float,
        decay_mode: DecayMode,
        stiefel_penalty: float,
        aspect_scale: bool,
        retract_every: int,
    ) -> None:
        if aspect_scale:
            scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            update = u * scale
        else:
            update = u

        # 1. Weight decay (smooth or cautious)
        if wd > 0.0:
            if decay_mode == "smooth":
                p.mul_(1.0 - lr * wd)
            elif decay_mode == "cautious":
                cwd_mask = (p * update) > 0.0
                p.add_(p * cwd_mask, alpha=-lr * wd)

        # 2. Continuous Riemannian Stiefel retraction penalty
        if stiefel_penalty > 0.0 and (step_num % retract_every == 0):
            work_dtype = torch.bfloat16 if p.is_cuda else torch.float32
            W_f = p.to(work_dtype)
            if W_f.size(0) >= W_f.size(1):
                gram = W_f.t() @ W_f
                ident = torch.eye(gram.size(0), device=p.device, dtype=work_dtype)
                diff = gram - ident
                denom = 1.0 + (torch.norm(diff.float(), p="fro") / math.sqrt(diff.size(0)))
                stiefel_grad = (W_f @ (diff / denom)).type_as(p)
            else:
                gram = W_f @ W_f.t()
                ident = torch.eye(gram.size(0), device=p.device, dtype=work_dtype)
                diff = gram - ident
                denom = 1.0 + (torch.norm(diff.float(), p="fro") / math.sqrt(diff.size(0)))
                stiefel_grad = ((diff / denom) @ W_f).type_as(p)
            p.add_(stiefel_grad, alpha=-stiefel_penalty * retract_every)

        # 3. Apply Newton-Schulz / PolarExpress polar gradient step
        p.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            nesterov: bool = group.get("nesterov", True)
            wd: float = group["weight_decay"]
            decay_mode: DecayMode = group["decay_mode"]
            stiefel_penalty: float = group["stiefel_penalty"]
            ns_steps: int = group["ns_steps"]
            aspect_scale: bool = group["aspect_scale"]
            retract_every: int = group.get("retract_every", 1)
            batched_ns: bool = group.get("batched_ns", True)
            ema_momentum: bool = group.get("ema_momentum", False)
            polar_method: str = group.get("polar_method", "newton_schulz")
            use_normuon: bool = group.get("use_normuon", False)
            normuon_beta2: float = group.get("normuon_beta2", 0.95)
            use_twin_rail: bool = group.get("use_twin_rail", False)
            rail_fast_beta: float = group.get("rail_fast_beta", 0.80)
            rail_slow_beta: float = group.get("rail_slow_beta", 0.98)
            rail_fast_weight: float = group.get("rail_fast_weight", 0.35)
            rail_engage_step: int = group.get("rail_engage_step", 0)

            shape_buckets: dict[
                tuple[tuple[int, ...], torch.device, torch.dtype],
                list[tuple[torch.Tensor, torch.Tensor, int, dict, str]],
            ] = {}

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim not in (2, 3):
                    raise ValueError(
                        f"StiefelMuon only supports 2D or 3D batched weight tensors, got shape {tuple(p.shape)}"
                    )

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["step"] = 0
                state["step"] = state.get("step", 0) + 1
                step_num: int = state["step"]
                eff_momentum = self.get_effective_momentum(step_num, group)
                state["effective_momentum"] = eff_momentum
                group["last_effective_momentum"] = eff_momentum

                buf: torch.Tensor = state["momentum_buffer"]
                if use_twin_rail:
                    if "slow_momentum_buffer" not in state:
                        state["slow_momentum_buffer"] = torch.zeros_like(g)
                    slow_buf: torch.Tensor = state["slow_momentum_buffer"]
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
                            self._apply_param_update(
                                p_c, u_c, step_num, lr, wd, decay_mode, stiefel_penalty, aspect_scale, retract_every
                            )
                # If p is a fused QKV projection of shape (3 * D, D), treat its 3 (D, D)
                # Q/K/V slices as independent square Stiefel matrices so each retains
                # unit spectral norm and merges into the (K, D, D) 3D Newton-Schulz batch.
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
                            self._apply_param_update(
                                p_c, u_c, step_num, lr, wd, decay_mode, stiefel_penalty, aspect_scale, retract_every
                            )
                elif batched_ns:
                    key = (tuple(p.shape), p.device, g_in.dtype)
                    shape_buckets.setdefault(key, []).append((p, g_in, step_num, state, "lane_energy"))
                else:
                    u = self._polar_2d(g_in, ns_steps, polar_method)
                    u = self._maybe_normuon(u, state, "lane_energy", use_normuon, normuon_beta2)
                    self._apply_param_update(
                        p, u, step_num, lr, wd, decay_mode, stiefel_penalty, aspect_scale, retract_every
                    )


            if batched_ns:
                for bucket in shape_buckets.values():
                    if len(bucket) == 1:
                        p, g_in, step_num, state, s_key = bucket[0]
                        u = self._polar_2d(g_in, ns_steps, polar_method)
                        u = self._maybe_normuon(u, state, s_key, use_normuon, normuon_beta2)
                        self._apply_param_update(
                            p, u, step_num, lr, wd, decay_mode, stiefel_penalty, aspect_scale, retract_every
                        )
                    else:
                        g_stack = torch.stack([item[1] for item in bucket], dim=0)
                        u_stack = self._polar_batched(g_stack, ns_steps, polar_method)
                        for idx_b, (p, _, step_num, state, s_key) in enumerate(bucket):
                            u_b = self._maybe_normuon(
                                u_stack[idx_b], state, s_key, use_normuon, normuon_beta2
                            )
                            self._apply_param_update(
                                p,
                                u_b,
                                step_num,
                                lr,
                                wd,
                                decay_mode,
                                stiefel_penalty,
                                aspect_scale,
                                retract_every,
                            )

        return loss



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
    """Hard polar retraction snapping 2D/3D weight matrices directly onto the Stiefel manifold.

    Warning:
        Executing hard periodic polar snaps mid-training abruptly resets all singular
        values to 1.0 simultaneously, which can cause severe validation loss spikes
        (representational shock). Prefer continuous regularization via
        `StiefelMuon(..., stiefel_penalty=0.005)` during training.
    """
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
    """Partition model parameters into 2D/3D Stiefel isometries (for Muon) and other params (for AdamW).

    Args:
        model: PyTorch model to partition.
        adamw_name_patterns: Substrings identifying lookup tables or readout frames
            that should be optimized by AdamW rather than Muon.
        muon_3d_name_patterns: Substrings identifying 3D batched orthogonal operators
            (e.g. `higher_interaction` `(16, 256, 256)` and `higher_sub_unitary` `(16, 32, 32)`)
            that should be optimized by `StiefelMuon`.

    Returns:
        Tuple `(muon_params, adamw_params)`.
    """
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


