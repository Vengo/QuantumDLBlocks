"""Distributed Stiefel-Manifold Muon Optimizer with Asynchronous Rank-Partitioned Newton-Schulz."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch
import torch.distributed as dist

from ..functional import zeropower_via_newtonschulz5
from .muon import DecayMode, StiefelMuon


class DistributedStiefelMuon(StiefelMuon):
    """Multi-Device Stiefel-Manifold Muon Optimizer with Asynchronous All-Gather / Broadcast Dispersion.

    Partitions the `M` 2D Stiefel weight matrices (e.g., 46 matrices in 6-layer `QuantumBornGPT`)
    round-robin across `world_size` data-parallel ranks (`owner_rank = matrix_idx % world_size`).
    At each step:
    1. All ranks update their local momentum buffer `M_t = mu * M_{t-1} + G_t`.
    2. Only `owner_rank` executes the 5-step quintic Newton-Schulz polar orthogonalization
       `U_t = NS_5(M_t)`, Cautious Weight Decay, and Riemannian Stiefel retraction penalty
       for matrix `matrix_idx`, cutting per-accelerator Newton-Schulz FLOPs by `world_size`.
    3. When `torch.distributed` is active (`world_size > 1`), each updated Stiefel matrix is
       immediately dispersed to peer ranks via non-blocking `dist.broadcast(..., async_op=True)`
       overlapped with subsequent Newton-Schulz evaluations.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.025,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        decay_mode: DecayMode = "cautious",
        stiefel_penalty: float = 0.005,
        ns_steps: int = 5,
        aspect_scale: bool = False,
        retract_every: int = 4,
        rank: int | None = None,
        world_size: int | None = None,
        process_group: Any = None,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            decay_mode=decay_mode,
            stiefel_penalty=stiefel_penalty,
            ns_steps=ns_steps,
            aspect_scale=aspect_scale,
            retract_every=retract_every,
        )
        self.process_group = process_group
        if rank is not None and world_size is not None:
            self.rank = int(rank)
            self.world_size = max(1, int(world_size))
            self._use_dist_collectives = dist.is_available() and dist.is_initialized()
        elif dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank(group=process_group)
            self.world_size = dist.get_world_size(group=process_group)
            self._use_dist_collectives = self.world_size > 1
        else:
            self.rank = 0
            self.world_size = 1
            self._use_dist_collectives = False

    def partition_summary(self) -> dict[str, int | float | list[int]]:
        """Return rank-ownership statistics across all 2D Stiefel parameter matrices."""
        all_params = [p for g in self.param_groups for p in g["params"]]
        total = len(all_params)
        owned_indices = [idx for idx in range(total) if (idx % self.world_size) == self.rank]
        owned = len(owned_indices)
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "total_stiefel_matrices": total,
            "owned_stiefel_matrices": owned,
            "owned_indices": owned_indices,
            "ns5_flop_reduction_factor": round(float(total) / float(max(1, owned)), 2),
        }

    @staticmethod
    def _apply_single_stiefel_update(
        p: torch.Tensor,
        buf: torch.Tensor,
        step_num: int,
        lr: float,
        wd: float,
        decay_mode: DecayMode,
        stiefel_penalty: float,
        ns_steps: int,
        aspect_scale: bool,
        retract_every: int,
    ) -> None:
        u = zeropower_via_newtonschulz5(buf, steps=ns_steps)
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

        p.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        async_handles: list[Any] = []
        matrix_idx = 0

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            wd: float = group["weight_decay"]
            decay_mode: DecayMode = group["decay_mode"]
            stiefel_penalty: float = group["stiefel_penalty"]
            ns_steps: int = group["ns_steps"]
            aspect_scale: bool = group["aspect_scale"]
            retract_every: int = group.get("retract_every", 1)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"DistributedStiefelMuon only supports 2D weight matrices, got shape {tuple(p.shape)}"
                    )

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["step"] = 0
                state["step"] = state.get("step", 0) + 1
                step_num: int = state["step"]
                buf: torch.Tensor = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                owner_rank = matrix_idx % self.world_size
                matrix_idx += 1

                if not self._use_dist_collectives:
                    # Single-process or simulated multi-rank mode: execute update directly
                    self._apply_single_stiefel_update(
                        p=p,
                        buf=buf,
                        step_num=step_num,
                        lr=lr,
                        wd=wd,
                        decay_mode=decay_mode,
                        stiefel_penalty=stiefel_penalty,
                        ns_steps=ns_steps,
                        aspect_scale=aspect_scale,
                        retract_every=retract_every,
                    )
                else:
                    if owner_rank == self.rank:
                        self._apply_single_stiefel_update(
                            p=p,
                            buf=buf,
                            step_num=step_num,
                            lr=lr,
                            wd=wd,
                            decay_mode=decay_mode,
                            stiefel_penalty=stiefel_penalty,
                            ns_steps=ns_steps,
                            aspect_scale=aspect_scale,
                            retract_every=retract_every,
                        )
                    handle = dist.broadcast(
                        p.data,
                        src=owner_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                    if handle is not None:
                        async_handles.append(handle)

        for handle in async_handles:
            handle.wait()

        return loss
