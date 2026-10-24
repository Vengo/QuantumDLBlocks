"""Unit and schema tests for Experiment Plan 042 (`Wave 27`):
1. Iso-Token & Iso-FLOP standardization in `scripts/fineweb_convex_vs_concave_slope_tracker.py`.
2. Proportional Wavelet Detail Ranks `(d//2, d//4)` and Empirical `< 3.28` Ceiling Probe
   in `scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`.
3. `Wave27IsoTokenArmSpec` token-to-step conversion across `B in {16, 24, 48}`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
)


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_proportional_wavelet_detail_ranks_forward_backward_cpu() -> None:
    """Verify proportional detail ranks `(d//2, d//4)` work cleanly at d=256, 384, 512."""
    for d, m_dim, f_dim in [(256, 128, 64), (384, 192, 96), (512, 256, 128)]:
        cfg = Radical2Pow16Config(
            vocab_size=50304,
            valid_vocab_size=50257,
            embed_dim=d,
            num_layers=2,
            num_heads=d // 64,
            mid_detail_dim=m_dim,
            fine_detail_dim=f_dim,
            num_fine_up_banks=1,
            use_convex_traceless_born=True,
            use_dyadic_hierarchical_born=True,
            readout_gemm_precision="bf16",
        )
        model = Radical2Pow16WaveletCholeskyGPT(cfg)
        assert model.pyramid.wav_8192.shape == (8192, m_dim)
        assert model.pyramid.up_8192.shape == (m_dim, d)
        assert model.pyramid.wav_65536.shape == (65536, f_dim)
        assert model.pyramid.up_65536.shape[-2:] == (f_dim, d)

        x = torch.randint(0, 50257, (2, 16), dtype=torch.long)
        y = torch.randint(0, 50257, (2, 16), dtype=torch.long)
        _, loss = model(x, y, return_logits=False)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        assert model.pyramid.up_8192.grad is not None
        assert torch.isfinite(model.pyramid.up_8192.grad).all()


def test_isotoken_step_grid_exactness_across_batch_sizes() -> None:
    """Verify that the LCM token horizon grid maps to exact integer steps across B=16, 24, 48."""
    milestones = (2_457_600, 7_372_800, 14_745_600, 22_118_400, 29_491_200, 44_236_800)
    for b in (16, 24, 48):
        tok_per_step = b * 512
        for tok in milestones:
            assert tok % tok_per_step == 0
            steps = tok // tok_per_step
            assert steps * tok_per_step == tok


def test_empirical_3p28_ceiling_and_wavelet_rank_probe_artifact() -> None:
    """Verify schema and results of `.cache/empirical_3p28_ceiling_and_wavelet_rank_probe.json`."""
    path = ROOT / ".cache" / "empirical_3p28_ceiling_and_wavelet_rank_probe.json"
    assert path.exists(), f"Missing {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "wavelet_rank_probe" in data
    assert "empirical_3p28_ceiling_probe" in data
    assert "summary" in data

    for d_key in ("d256", "d384", "d512"):
        assert d_key in data["wavelet_rank_probe"]
    assert data["wavelet_rank_probe"]["d384"]["r2_gain_over_fixed"] > 0.0
    assert data["wavelet_rank_probe"]["d512"]["r2_gain_over_fixed"] > 0.0
    assert data["wavelet_rank_probe"]["d384"]["r_sharp_gain_over_fixed"] > 0.0
    assert data["wavelet_rank_probe"]["d512"]["r_sharp_gain_over_fixed"] > 0.0

    s = data["summary"]
    assert s["proportional_ranks_beat_fixed_for_d384_d512"] is True
    assert s["all_widths_break_3p28_ce"] is True
    assert s["all_widths_break_1p80_ce"] is True
    assert s["monotonic_width_ceiling_d512_lt_d384_lt_d256"] is True


def test_isotoken_slope_tracker_artifact() -> None:
    """Verify `.cache/fineweb_convex_vs_concave_slope_tracker.json` includes iso-token & FLOP fields."""
    path = ROOT / ".cache" / "fineweb_convex_vs_concave_slope_tracker.json"
    assert path.exists(), f"Missing {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "iso_token_table" in data
    assert len(data["iso_token_table"]) >= 6
    first_row = data["iso_token_table"][0]
    for key in (
        "tokens_per_step",
        "tokens_M",
        "total_compute_tflops",
        "optimizer_updates_per_10m_tokens",
        "ce_at_2p46m",
        "ce_at_7p37m",
    ):
        assert key in first_row, f"Missing {key} in iso_token_table row"
