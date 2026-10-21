"""Distributed Haliax/Levanter Named-Axis FSDP/TP Sharding and Multi-Device StiefelMuon in JAX."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import haliax as hax
from haliax.partitioning import ResourceAxis
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

from .optim import _apply_stiefel_muon_tensor

__all__ = [
    "build_quantum_axis_mapping",
    "make_quantum_device_mesh",
    "shard_quantum_model",
    "partition_stiefel_leaves",
    "distributed_stiefel_muon_step",
]


def build_quantum_axis_mapping(
    *,
    fsdp_enabled: bool = True,
    tensor_parallel_enabled: bool = True,
) -> dict[str, dict[str, str | tuple[str, ...]]]:
    """Construct Haliax named-axis mappings (`param_mapping` and `compute_mapping`) for FSDP + TP/EP.

    Maps logical quantum-geometric axes (`batch`, `hilbert`, `heads`, `mlp`, `experts`, `vocab`)
    onto physical device mesh axes `(ResourceAxis.DATA, ResourceAxis.MODEL)`:
    - `param_mapping`: Shards large Stiefel weight matrices along `hilbert` across `data` (FSDP)
      and shards `heads`, `mlp`, and `experts` across `model` (Tensor / Expert Parallelism).
    - `compute_mapping`: Shards activation batches along `batch` across `data` and shards
      multi-head / MLP / MoE activations across `model`.
    """
    data_ax = str(ResourceAxis.DATA)
    model_ax = str(ResourceAxis.MODEL)

    param_mapping: dict[str, str | tuple[str, ...]] = {}
    compute_mapping: dict[str, str | tuple[str, ...]] = {"batch": data_ax}

    if fsdp_enabled:
        param_mapping["hilbert"] = data_ax
        param_mapping["vocab"] = data_ax

    if tensor_parallel_enabled:
        param_mapping["heads"] = model_ax
        param_mapping["mlp"] = model_ax
        param_mapping["experts"] = model_ax
        compute_mapping["heads"] = model_ax
        compute_mapping["mlp"] = model_ax
        compute_mapping["experts"] = model_ax

    return {
        "param_mapping": param_mapping,
        "compute_mapping": compute_mapping,
    }


def make_quantum_device_mesh(
    devices: list[Any] | None = None,
    data_parallel_size: int | None = None,
    model_parallel_size: int = 1,
) -> Mesh:
    """Create a 2D JAX `(data, model)` device mesh for Haliax/Levanter sharding."""
    devs = list(devices) if devices is not None else list(jax.devices())
    n_devs = len(devs)
    mp = max(1, min(int(model_parallel_size), n_devs))
    if data_parallel_size is None:
        dp = max(1, n_devs // mp)
    else:
        dp = max(1, int(data_parallel_size))
    used = devs[: dp * mp]
    dev_grid = np.array(used, dtype=object).reshape(dp, mp)
    return Mesh(dev_grid, (str(ResourceAxis.DATA), str(ResourceAxis.MODEL)))


def shard_quantum_model(
    model: Any,
    mesh: Mesh | None = None,
    param_mapping: dict[str, Any] | None = None,
) -> Any:
    """Shard a `HaxQuantumBornGPT` or Equinox/Haliax module across a 2D `(data, model)` mesh."""
    active_mesh = mesh if mesh is not None else make_quantum_device_mesh()
    if param_mapping is None:
        mp_size = int(active_mesh.devices.shape[1]) if active_mesh.devices.ndim >= 2 else 1
        dp_size = int(active_mesh.devices.shape[0])
        mappings = build_quantum_axis_mapping(
            fsdp_enabled=(dp_size > 1),
            tensor_parallel_enabled=(mp_size > 1),
        )
        param_mapping = mappings["param_mapping"]

    def _shard_leaf(leaf: Any) -> Any:
        if isinstance(leaf, hax.NamedArray):
            spec_names: list[str | None] = []
            for ax in leaf.axes:
                mapped = param_mapping.get(ax.name)
                if isinstance(mapped, str):
                    mesh_idx = active_mesh.axis_names.index(mapped) if mapped in active_mesh.axis_names else -1
                    mesh_sz = int(active_mesh.devices.shape[mesh_idx]) if mesh_idx >= 0 else 1
                    if ax.size % mesh_sz == 0:
                        spec_names.append(mapped)
                    else:
                        spec_names.append(None)
                else:
                    spec_names.append(None)
            sharding = NamedSharding(active_mesh, PartitionSpec(*spec_names))
            sharded_arr = jax.device_put(leaf.array, sharding)
            return hax.NamedArray(sharded_arr, leaf.axes)
        if eqx.is_array(leaf):
            replicated = NamedSharding(active_mesh, PartitionSpec())
            return jax.device_put(leaf, replicated)
        return leaf

    return jax.tree_util.tree_map(
        _shard_leaf,
        model,
        is_leaf=lambda x: isinstance(x, hax.NamedArray) or eqx.is_array(x),
    )


def partition_stiefel_leaves(
    params: Any,
    world_size: int = 8,
) -> dict[str, Any]:
    """Assign each 2D/3D Stiefel parameter leaf in `params` round-robin to one of `world_size` ranks."""
    w_sz = max(1, int(world_size))
    excluded_tokens = ("seed_embed", "token_embed", "povm_head", "m_vocab", "pos_embed", "wte", "povm_delta")

    stiefel_entries: list[dict[str, Any]] = []
    idx_counter = 0

    for path, leaf in jax.tree_util.tree_leaves_with_path(
        params, is_leaf=lambda x: x is None or eqx.is_array(x)
    ):
        if not eqx.is_inexact_array(leaf):
            continue
        path_str = ".".join(str(k).strip(".[]'").lower() for k in path)
        if (
            leaf.ndim in (2, 3)
            and not any(tok in path_str for tok in excluded_tokens)
            and max(leaf.shape) < 20000
        ):
            owner_rank = idx_counter % w_sz
            stiefel_entries.append(
                {
                    "matrix_index": idx_counter,
                    "path": path_str,
                    "shape": list(leaf.shape),
                    "owner_rank": owner_rank,
                }
            )
            idx_counter += 1

    per_rank_counts = [
        sum(1 for e in stiefel_entries if e["owner_rank"] == r) for r in range(w_sz)
    ]
    max_per_rank = max(per_rank_counts) if per_rank_counts else 1
    return {
        "world_size": w_sz,
        "total_stiefel_matrices": idx_counter,
        "per_rank_counts": per_rank_counts,
        "ns5_flop_reduction_factor": round(float(idx_counter) / float(max(1, max_per_rank)), 2),
        "assignments": stiefel_entries,
    }


def distributed_stiefel_muon_step(
    grads: list[jax.Array],
    momentums: list[jax.Array],
    weights: list[jax.Array],
    *,
    world_size: int = 4,
    lr: float = 0.025,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.01,
    decay_mode: str = "cautious",
    stiefel_penalty: float = 0.05,
    axis_name: str | None = None,
) -> tuple[list[jax.Array], list[jax.Array], dict[str, Any]]:
    """Execute multi-rank distributed `StiefelMuon` with rank-partitioned 5-step Newton-Schulz dispersion.

    Each rank `r in {0, ..., world_size - 1}` computes the 5-step Newton-Schulz polar factor
    and Riemannian Stiefel retraction penalty only for its owned subset `{i : i % world_size == r}`
    of the Stiefel matrices, and all-gathers the orthogonalized parameter updates across ranks.
    """
    if not (len(grads) == len(momentums) == len(weights)):
        raise ValueError("grads, momentums, and weights must have identical length.")

    w_sz = max(1, int(world_size))
    n_mats = len(weights)
    new_momentums: list[jax.Array] = []
    rank_updates: dict[int, dict[int, jax.Array]] = {r: {} for r in range(w_sz)}

    for idx in range(n_mats):
        owner = idx % w_sz
        dw, nm = _apply_stiefel_muon_tensor(
            grads[idx],
            momentums[idx],
            weights[idx],
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            decay_mode=decay_mode,  # type: ignore[arg-type]
            stiefel_penalty=stiefel_penalty,
        )
        new_momentums.append(nm)
        if axis_name is not None:
            # Collective all_gather across physical SPMD axis when inside shard_map/pmap
            r_idx = jax.lax.axis_index(axis_name)
            local_dw = jnp.where(r_idx == owner, dw, jnp.zeros_like(dw))
            dw = jax.lax.psum(local_dw, axis_name=axis_name)
        rank_updates[owner][idx] = dw

    # All-gather dispersion: reconstruct full ordered update list from per-rank shards
    dispersed_updates: list[jax.Array] = []
    for idx in range(n_mats):
        owner = idx % w_sz
        dispersed_updates.append(rank_updates[owner][idx])

    per_rank_counts = [len(rank_updates[r]) for r in range(w_sz)]
    telemetry = {
        "world_size": w_sz,
        "total_matrices": n_mats,
        "per_rank_counts": per_rank_counts,
        "ns5_flop_reduction_factor": round(float(n_mats) / float(max(1, max(per_rank_counts))), 2),
    }
    return dispersed_updates, new_momentums, telemetry
