"""Manifold-aware optimizers for Quantum-Geometric Deep Learning."""

from .distributed_muon import DistributedStiefelMuon
from .muon import DecayMode, PolarMethod, StiefelMuon, polar_snap_weights, split_muon_adamw_params
from .schedules import (
    ProgressiveSequenceCurriculum,
    branch_wsd_cooldown_multiplier,
    cosine_lr_multiplier,
    wsd_lr_multiplier,
)
from .tail_ema import TailWeightAverager

__all__ = [
    "DecayMode",
    "PolarMethod",
    "StiefelMuon",
    "DistributedStiefelMuon",
    "polar_snap_weights",
    "split_muon_adamw_params",
    "cosine_lr_multiplier",
    "wsd_lr_multiplier",
    "branch_wsd_cooldown_multiplier",
    "ProgressiveSequenceCurriculum",
    "TailWeightAverager",
]


