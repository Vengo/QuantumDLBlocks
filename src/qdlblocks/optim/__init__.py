"""Manifold-aware optimizers for Quantum-Geometric Deep Learning."""

from .muon import DecayMode, StiefelMuon, polar_snap_weights, split_muon_adamw_params

__all__ = [
    "DecayMode",
    "StiefelMuon",
    "polar_snap_weights",
    "split_muon_adamw_params",
]
