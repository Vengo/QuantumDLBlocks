"""Unified Multi-Domain Quantum-Geometric Benchmark Runner.

Implements synthetic offline-ready benchmarks and evaluation harnesses across:
1. `vision`: QuantumBornViT vs ClassicalViTBaseline on 2D Spatial Gabor Orientation/Frequency
2. `audio`: ComplexSuperpositionAudioClassifier (Dense & SparseKrausMoE) vs MagnitudeMelAudioBaseline
   on Phase-Modulated Multi-Harmonic Chords (where magnitude |X| is identical across classes!)
3. `language`: QuantumBornGPT (Dense & QuantumBornMoE) vs ClassicalGPTBaseline on Markov Zipfian streams
4. `lra`: Long-Range Associative Recall (Real vs Complex Phase Born Attention vs Softmax)
5. `calibration_ood`: Expected Calibration Error (ECE), Brier Score, and OOD Detection via
   von Neumann Entropy S(rho)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.attention import SubspaceProjectorBornAttention
from ..metrics import (
    attention_entropy_and_sparsity,
    logit_margin_stats,
    model_isometry_error,
    purity,
    von_neumann_entropy,
)
from ..models.audio_quantum import ComplexSuperpositionAudioClassifier
from ..models.born_gpt import QuantumBornGPT
from ..models.born_vit import QuantumBornViT
from ..models.speech_ctc import (
    ClassicalSpeechCTCBaseline,
    ComplexQuantumSpeechCTC,
    compute_cer,
    compute_wer,
    ctc_greedy_decode,
    extract_librispeech_word_segments,
    load_librispeech_dataset,
    text_to_char_indices,
)
from ..optim.muon import StiefelMuon, split_muon_adamw_params
from .baselines import ClassicalGPTBaseline, ClassicalViTBaseline, MagnitudeMelAudioBaseline
from .metrics_calibration import brier_score, expected_calibration_error, ood_auroc_from_scores


@dataclass
class BenchmarkResult:
    domain: str
    model_name: str
    params: int
    primary_metric_name: str
    primary_metric_value: float
    secondary_metrics: dict[str, float]
    elapsed_sec: float


def format_benchmark_table(results: list[BenchmarkResult]) -> str:
    """Format a list of BenchmarkResult objects into a GitHub Markdown table."""
    lines = [
        "| Domain | Model | Params | Primary Metric | Secondary Telemetry | Time (s) |",
        "| :--- | :--- | ---: | :--- | :--- | ---: |",
    ]
    for r in results:
        sec_str = ", ".join(f"`{k}={v:.3f}`" for k, v in r.secondary_metrics.items())
        lines.append(
            f"| `{r.domain}` | **{r.model_name}** | {r.params:,} "
            f"| `{r.primary_metric_name}`: **{r.primary_metric_value:.2f}** "
            f"| {sec_str} | {r.elapsed_sec:.2f}s |"
        )
    return "\n".join(lines)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_synthetic_gabor_dataset(
    num_samples: int = 600,
    image_size: int = 16,
    in_chans: int = 3,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 2D Gabor orientation/frequency images `(N, C, H, W)` with realistic spatial/spectral jitter."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_samples, dtype=torch.long) % num_classes
    labels = labels[torch.randperm(num_samples, generator=g)]

    coords = torch.linspace(-1.0, 1.0, image_size)
    u, v = torch.meshgrid(coords, coords, indexing="ij")  # (H, W)

    images = torch.zeros(num_samples, in_chans, image_size, image_size, dtype=torch.float32)
    for i in range(num_samples):
        c = int(labels[i].item())
        # Sub-patch spatial jitter, intra-class orientation & frequency variance
        du = float((torch.rand(1, generator=g).item() - 0.5) * 0.10)
        dv = float((torch.rand(1, generator=g).item() - 0.5) * 0.10)
        theta_jitter = float(torch.randn(1, generator=g).item()) * (math.pi / (4.5 * num_classes))
        freq_jitter = float(torch.randn(1, generator=g).item()) * 0.12

        theta = (math.pi * c / num_classes) + theta_jitter
        freq = max(1.0, 1.5 + 0.35 * c + freq_jitter)
        u_s = u - du
        v_s = v - dv
        env = torch.exp(-1.5 * (u_s.square() + v_s.square()))
        u_rot = u_s * math.cos(theta) + v_s * math.sin(theta)

        # Random secondary clutter wave + realistic sensor noise (SNR ~ 0 dB)
        clutter_theta = float(torch.rand(1, generator=g).item() * math.pi)
        u_clutter = u * math.cos(clutter_theta) + v * math.sin(clutter_theta)

        for ch in range(in_chans):
            phase_jitter = float(torch.randn(1, generator=g).item()) * 0.22
            phase = (2.0 * math.pi * ch / max(in_chans, 1)) + (2.0 * math.pi * c / num_classes) + phase_jitter
            carrier = torch.cos(2.0 * math.pi * freq * u_rot + phase)
            clutter = 0.22 * torch.cos(2.0 * math.pi * 2.3 * u_clutter)
            noise = 0.42 * torch.randn((image_size, image_size), generator=g)
            images[i, ch] = env * carrier + clutter + noise

    return images, labels


def generate_phase_modulated_audio_dataset(
    num_samples: int = 600,
    n_frames: int = 16,
    n_freq_bins: int = 32,
    num_classes: int = 8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate complex STFT dataset `(N, T, F)` where magnitude `|X|` is identical across classes."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_samples, dtype=torch.long) % num_classes
    labels = labels[torch.randperm(num_samples, generator=g)]

    f_idx = torch.arange(n_freq_bins, dtype=torch.float32)
    amp_env = torch.exp(-f_idx / (0.5 * n_freq_bins)).view(1, 1, n_freq_bins)

    phi_0 = torch.rand((num_samples, n_frames, 1), generator=g) * (2.0 * math.pi)
    # Intra-class harmonic slope jitter + non-stationary multi-path phase dispersion
    slope_jitter = 0.36 * torch.randn((num_samples, 1, 1), generator=g)
    c_float = labels.float().view(num_samples, 1, 1) + slope_jitter
    freq_grid = f_idx.view(1, 1, n_freq_bins)
    multipath = 0.45 * torch.sin(
        torch.rand((num_samples, 1, 1), generator=g) * 6.28 + freq_grid * 0.35
    )
    phase_noise = 1.15 * torch.randn((num_samples, n_frames, n_freq_bins), generator=g)

    phase = phi_0 + (2.0 * math.pi * c_float * freq_grid / num_classes) + multipath + phase_noise
    stft_complex = torch.polar(amp_env.expand(num_samples, n_frames, n_freq_bins), phase)
    return stft_complex.to(torch.complex64), labels


def generate_markov_zipf_tokens(
    num_tokens: int = 16384,
    vocab_size: int = 128,
    seq_len: int = 64,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic deterministic-plus-noise token sequences `(N_seqs, seq_len)`."""
    rng = np.random.default_rng(seed)
    total = (num_tokens // seq_len) * seq_len + 1
    stream = np.zeros(total, dtype=np.int64)
    stream[0] = 1
    stream[1] = 2
    noise_mask = rng.random(total) < 0.08
    random_tokens = rng.integers(0, vocab_size, size=total)

    for t in range(1, total - 1):
        if noise_mask[t + 1]:
            stream[t + 1] = random_tokens[t + 1]
        else:
            stream[t + 1] = (3 * stream[t] + 5 * stream[t - 1] + 7) % vocab_size

    n_seqs = num_tokens // seq_len
    x = torch.from_numpy(stream[: n_seqs * seq_len].reshape(n_seqs, seq_len).copy())
    y = torch.from_numpy(stream[1 : n_seqs * seq_len + 1].reshape(n_seqs, seq_len).copy())
    return x, y


def generate_associative_recall_batch(
    batch_size: int = 32,
    seq_len: int = 128,
    num_pairs: int = 8,
    vocab_size: int = 64,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Associative Recall / Induction Head sequence batch `(B, seq_len)`."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    half_v = vocab_size // 2
    num_pairs = min(num_pairs, half_v - 1)
    seq = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randperm(half_v - 1, generator=g)[:num_pairs] + 1
        vals = torch.randperm(half_v - 1, generator=g)[:num_pairs] + half_v
        pairs = torch.stack([keys, vals], dim=1).reshape(-1)  # length 2 * num_pairs
        reps = ((seq_len - 2) + pairs.numel() - 1) // pairs.numel()
        context = pairs.repeat(reps)[: seq_len - 2]
        seq[b, : seq_len - 2] = context
        q_idx = int(torch.randint(0, num_pairs, (1,), generator=g).item())
        seq[b, -2] = keys[q_idx]
        seq[b, -1] = vals[q_idx]

    target_seq = torch.zeros_like(seq)
    target_seq[:, :-1] = seq[:, 1:]
    return seq.to(device), target_seq.to(device)


def _train_vision_model(
    model: nn.Module,
    model_name: str,
    use_muon: bool,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    steps: int,
    batch_size: int = 64,
) -> BenchmarkResult:
    t0 = time.time()
    params = sum(p.numel() for p in model.parameters())
    if use_muon:
        m_params, a_params = split_muon_adamw_params(model)
        opt_m = StiefelMuon(m_params, lr=0.012, weight_decay=0.02, stiefel_penalty=0.02)
        opt_a = torch.optim.AdamW(a_params, lr=0.004, weight_decay=1e-3)
        opts: list[torch.optim.Optimizer] = [opt_m, opt_a]
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-3)]

    model.train()
    n_train = x_tr.size(0)
    for step in range(steps):
        for o in opts:
            o.zero_grad(set_to_none=True)
        idx = torch.randint(0, n_train, (batch_size,), device=x_tr.device)
        logits = model(x_tr[idx])
        loss = F.cross_entropy(logits, y_tr[idx])
        loss.backward()
        for o in opts:
            o.step()

    model.eval()
    with torch.no_grad():
        logits_tr = model(x_tr[:128])
        train_acc = float((logits_tr.argmax(dim=-1) == y_tr[:128]).float().mean().item() * 100.0)
        logits_te = model(x_te)
        acc = float((logits_te.argmax(dim=-1) == y_te).float().mean().item() * 100.0)
        ece = expected_calibration_error(logits_te, y_te, from_logits=True)
        brier = brier_score(logits_te, y_te, from_logits=True)
        iso_err, _ = model_isometry_error(model)

    return BenchmarkResult(
        domain="vision",
        model_name=model_name,
        params=params,
        primary_metric_name="test_acc_pct",
        primary_metric_value=acc,
        secondary_metrics={
            "train_acc_pct": train_acc,
            "ece": ece,
            "brier": brier,
            "isometry_error": iso_err,
        },
        elapsed_sec=time.time() - t0,
    )


def run_vision_benchmark(
    device: str = "cuda",
    mode: Literal["quick", "full"] = "quick",
    seed: int = 42,
) -> list[BenchmarkResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(device)
    images, labels = generate_synthetic_gabor_dataset(num_samples=600, image_size=16, seed=seed)
    images, labels = images.to(dev), labels.to(dev)
    split = 480
    x_tr, y_tr = images[:split], labels[:split]
    x_te, y_te = images[split:], labels[split:]
    steps = 45 if mode == "quick" else 160

    q_vit = QuantumBornViT(
        image_size=16,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        seed_dim=32,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        use_flash_born=True,
    ).to(dev)
    q_vit_moe = QuantumBornViT(
        image_size=16,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        seed_dim=32,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_expansion=1.5,
        num_experts=4,
        top_k_experts=2,
        use_flash_born=True,
    ).to(dev)
    c_vit = ClassicalViTBaseline(
        image_size=16,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        seed_dim=32,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
    ).to(dev)

    return [
        _train_vision_model(q_vit, "QuantumBornViT (Stiefel+Born)", True, x_tr, y_tr, x_te, y_te, steps),
        _train_vision_model(q_vit_moe, "QuantumBornViT (Sparse Born MoE)", True, x_tr, y_tr, x_te, y_te, steps),
        _train_vision_model(c_vit, "ClassicalViT (Softmax+LN)", False, x_tr, y_tr, x_te, y_te, steps),
    ]


def run_audio_benchmark(
    device: str = "cuda",
    mode: Literal["quick", "full"] = "quick",
    seed: int = 42,
) -> list[BenchmarkResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(device)
    stft_x, labels = generate_phase_modulated_audio_dataset(
        num_samples=600, n_frames=16, n_freq_bins=32, num_classes=8, seed=seed
    )
    stft_x, labels = stft_x.to(dev), labels.to(dev)
    x_tr, y_tr = stft_x[:480], labels[:480]
    x_te, y_te = stft_x[480:], labels[480:]
    steps = 75 if mode == "quick" else 220

    models_list: list[tuple[str, nn.Module, bool]] = [
        (
            "ComplexSuperpositionAudio (Kraus)",
            ComplexSuperpositionAudioClassifier(
                n_freq_bins=32, hilbert_dim=24, num_classes=8, num_heads=4, num_kraus=4
            ).to(dev),
            True,
        ),
        (
            "ComplexSuperpositionAudio (SparseKrausMoE)",
            ComplexSuperpositionAudioClassifier(
                n_freq_bins=32,
                hilbert_dim=24,
                num_classes=8,
                num_heads=4,
                use_sparse_kraus_moe=True,
                num_experts=6,
                top_k_experts=2,
            ).to(dev),
            True,
        ),
        (
            "MagnitudeMelAudioBaseline (Phase-Blind)",
            MagnitudeMelAudioBaseline(
                n_freq_bins=32, embed_dim=24, num_classes=8, num_heads=4, num_layers=1
            ).to(dev),
            False,
        ),
    ]

    results: list[BenchmarkResult] = []
    for name, model, is_quantum in models_list:
        t0 = time.time()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        model.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, x_tr.size(0), (64,), device=dev)
            logits = model(x_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits_tr = model(x_tr[:128])
            if isinstance(logits_tr, tuple):
                logits_tr = logits_tr[0]
            train_acc = float((logits_tr.argmax(dim=-1) == y_tr[:128]).float().mean().item() * 100.0)
            if is_quantum:
                logits_te, rho_te = model(x_te, return_density=True)  # type: ignore[misc]
                ent = float(von_neumann_entropy(rho_te).item())
                pur = float(purity(rho_te).item())
            else:
                logits_te = model(x_te)
                ent = 0.0
                pur = 0.0
            acc = float((logits_te.argmax(dim=-1) == y_te).float().mean().item() * 100.0)

        results.append(
            BenchmarkResult(
                domain="audio",
                model_name=name,
                params=sum(p.numel() for p in model.parameters()),
                primary_metric_name="test_acc_pct",
                primary_metric_value=acc,
                secondary_metrics={
                    "train_acc_pct": train_acc,
                    "mean_entropy_nats": ent,
                    "mean_purity": pur,
                },
                elapsed_sec=time.time() - t0,
            )
        )
    return results


def run_language_benchmark(
    device: str = "cuda",
    mode: Literal["quick", "full"] = "quick",
    seed: int = 42,
) -> list[BenchmarkResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(device)
    x_all, y_all = generate_markov_zipf_tokens(num_tokens=8192, vocab_size=128, seq_len=64, seed=seed)
    x_all, y_all = x_all.to(dev), y_all.to(dev)
    split = int(0.8 * x_all.size(0))
    x_tr, y_tr = x_all[:split], y_all[:split]
    x_val, y_val = x_all[split:], y_all[split:]
    steps = 45 if mode == "quick" else 160

    configs: list[tuple[str, nn.Module, bool]] = [
        (
            "QuantumBornGPT (Dense)",
            QuantumBornGPT(
                vocab_size=128,
                embed_dim=64,
                seed_dim=32,
                num_heads=4,
                num_layers=2,
                hidden_dim=160,
                block_size=64,
                num_experts=1,
                learnable_temperature=True,
                value_residual_mixing=True,
                input_highway_mixing=True,
                use_flash_born=True,
            ).to(dev),
            True,
        ),
        (
            "QuantumBornMoE-GPT (4 Exp, Top-2, Zero Aux)",
            QuantumBornGPT(
                vocab_size=128,
                embed_dim=64,
                seed_dim=32,
                num_heads=4,
                num_layers=2,
                hidden_dim=96,
                block_size=64,
                num_experts=4,
                top_k_experts=2,
                learnable_temperature=True,
                value_residual_mixing=True,
                input_highway_mixing=True,
                use_flash_born=True,
            ).to(dev),
            True,
        ),
        (
            "ClassicalGPTBaseline (Softmax+LN)",
            ClassicalGPTBaseline(
                vocab_size=128,
                embed_dim=64,
                seed_dim=32,
                num_heads=4,
                num_layers=2,
                hidden_dim=160,
                block_size=64,
            ).to(dev),
            False,
        ),
    ]

    results: list[BenchmarkResult] = []
    for name, model, is_quantum in configs:
        t0 = time.time()
        if is_quantum:
            m_params, a_params = split_muon_adamw_params(model)
            opt_m = StiefelMuon(m_params, lr=0.02, weight_decay=0.01, stiefel_penalty=0.005)
            opt_a = torch.optim.AdamW(a_params, lr=0.004, weight_decay=1e-4)
            opts: list[torch.optim.Optimizer] = [opt_m, opt_a]
        else:
            opts = [torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-4)]

        model.train()
        for _ in range(steps):
            for o in opts:
                o.zero_grad(set_to_none=True)
            idx = torch.randint(0, x_tr.size(0), (32,), device=dev)
            _, loss = model(x_tr[idx], y_tr[idx])
            assert loss is not None
            loss.backward()
            for o in opts:
                o.step()

        model.eval()
        with torch.no_grad():
            logits_v, loss_v = model(x_val, y_val)
            assert loss_v is not None
            l_stats = logit_margin_stats(logits_v, y_val)
            val_loss = float(loss_v.item())
            ppl = math.exp(min(val_loss, 20.0))
            iso_err, _ = model_isometry_error(model)
            moe_loads = [
                blk.moe.last_routing_stats.get("max_expert_load_fraction", 1.0)
                for blk in getattr(model, "blocks", [])
                if getattr(blk, "moe", None) is not None
            ]
            max_load = float(np.mean(moe_loads)) if moe_loads else 1.0

        results.append(
            BenchmarkResult(
                domain="language",
                model_name=name,
                params=sum(p.numel() for p in model.parameters()),
                primary_metric_name="val_perplexity",
                primary_metric_value=ppl,
                secondary_metrics={
                    "val_loss": val_loss,
                    "top1_acc_pct": float(l_stats["top1_accuracy"]),
                    "median_target_rank": float(l_stats["median_target_rank"]),
                    "logit_margin": float(l_stats["mean_top1_top2_margin"]),
                    "isometry_error": iso_err,
                    "max_expert_load": max_load,
                },
                elapsed_sec=time.time() - t0,
            )
        )
    return results


def run_lra_benchmark(
    device: str = "cuda",
    mode: Literal["quick", "full"] = "quick",
    seed: int = 42,
) -> list[BenchmarkResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(device)
    seq_len = 128 if mode == "quick" else 256
    steps = 55 if mode == "quick" else 180

    x_eval, y_eval = generate_associative_recall_batch(
        batch_size=128, seq_len=seq_len, num_pairs=8, vocab_size=64, seed=seed + 999, device=dev
    )

    models_cfg: list[tuple[str, nn.Module, bool]] = [
        (
            "QuantumBornGPT (Flash-Born Real)",
            QuantumBornGPT(
                vocab_size=64,
                embed_dim=64,
                seed_dim=32,
                num_heads=4,
                num_layers=2,
                hidden_dim=128,
                block_size=seq_len,
                learnable_temperature=True,
                complex_phase_attention=False,
                use_flash_born=True,
            ).to(dev),
            True,
        ),
        (
            "QuantumBornGPT (Complex Phase Born)",
            QuantumBornGPT(
                vocab_size=64,
                embed_dim=64,
                seed_dim=32,
                num_heads=4,
                num_layers=2,
                hidden_dim=128,
                block_size=seq_len,
                learnable_temperature=True,
                complex_phase_attention=True,
                use_flash_born=False,
            ).to(dev),
            True,
        ),
        (
            "ClassicalGPTBaseline (Softmax)",
            ClassicalGPTBaseline(
                vocab_size=64,
                embed_dim=64,
                seed_dim=32,
                num_heads=4,
                num_layers=2,
                hidden_dim=128,
                block_size=seq_len,
            ).to(dev),
            False,
        ),
    ]

    results: list[BenchmarkResult] = []
    for idx_m, (name, model, is_quantum) in enumerate(models_cfg):
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-4)
        model.train()
        for s in range(steps):
            opt.zero_grad(set_to_none=True)
            xb, yb = generate_associative_recall_batch(
                batch_size=32, seq_len=seq_len, num_pairs=8, vocab_size=64, seed=seed + s, device=dev
            )
            logits, seq_loss = model(xb, yb)
            assert seq_loss is not None
            query_loss = F.cross_entropy(logits[:, -2, :], yb[:, -2])
            loss = seq_loss + 2.0 * query_loss
            loss.backward()
            opt.step()

        peak_vram = (
            float(torch.cuda.max_memory_allocated(dev) / (1024**2)) if dev.type == "cuda" else 0.0
        )

        model.eval()
        with torch.no_grad():
            if is_quantum:
                for m in model.modules():
                    if isinstance(m, SubspaceProjectorBornAttention):
                        m.record_attention = True
                        m.use_flash_born = False
            logits_ev, _ = model(x_eval)
            preds_q = logits_ev[:, -2, :].argmax(dim=-1)
            recall_acc = float((preds_q == y_eval[:, -2]).float().mean().item() * 100.0)

            sparsities: list[float] = []
            if is_quantum:
                for m in model.modules():
                    if isinstance(m, SubspaceProjectorBornAttention) and m.last_born_weights is not None:
                        stats_a = attention_entropy_and_sparsity(m.last_born_weights)
                        sparsities.append(stats_a["sparsity"] * 100.0)
                        m.record_attention = False
            attn_sparsity = float(np.mean(sparsities)) if sparsities else 0.0

        results.append(
            BenchmarkResult(
                domain="lra",
                model_name=name,
                params=sum(p.numel() for p in model.parameters()),
                primary_metric_name="recall_acc_pct",
                primary_metric_value=recall_acc,
                secondary_metrics={
                    "attn_zero_sparsity_pct": attn_sparsity,
                    "peak_vram_mb": peak_vram,
                },
                elapsed_sec=time.time() - t0,
            )
        )
    return results


def run_calibration_ood_benchmark(
    device: str = "cuda",
    mode: Literal["quick", "full"] = "quick",
    seed: int = 42,
) -> list[BenchmarkResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(device)
    stft_x, labels = generate_phase_modulated_audio_dataset(
        num_samples=600, n_frames=16, n_freq_bins=32, num_classes=8, seed=seed
    )
    stft_x, labels = stft_x.to(dev), labels.to(dev)
    x_tr, y_tr = stft_x[:480], labels[:480]
    x_te, y_te = stft_x[480:], labels[480:]

    g_ood = torch.Generator().manual_seed(seed + 777)
    noise_corr = torch.complex(
        torch.randn(x_te.shape, generator=g_ood), torch.randn(x_te.shape, generator=g_ood)
    ).to(dev)
    x_corr = x_te + 0.8 * noise_corr
    x_ood = torch.complex(
        torch.randn(x_te.shape, generator=g_ood), torch.randn(x_te.shape, generator=g_ood)
    ).to(dev)

    steps = 70 if mode == "quick" else 200
    q_model = ComplexSuperpositionAudioClassifier(
        n_freq_bins=32, hilbert_dim=24, num_classes=8, num_heads=4, num_kraus=4
    ).to(dev)
    c_model = MagnitudeMelAudioBaseline(
        n_freq_bins=32, embed_dim=24, num_classes=8, num_heads=4, num_layers=1
    ).to(dev)

    results: list[BenchmarkResult] = []
    for name, model, is_quantum in [
        ("Quantum POVM + von Neumann Entropy", q_model, True),
        ("Classical Unbounded Softmax", c_model, False),
    ]:
        t0 = time.time()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        model.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, x_tr.size(0), (64,), device=dev)
            logits = model(x_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            if is_quantum:
                logits_id, rho_id = model(x_te, return_density=True)  # type: ignore[misc]
                logits_corr, _ = model(x_corr, return_density=True)  # type: ignore[misc]
                logits_ood, rho_ood = model(x_ood, return_density=True)  # type: ignore[misc]

                p_id = F.softmax(logits_id, dim=-1)
                p_ood = F.softmax(logits_ood, dim=-1)
                h_id = -(p_id * torch.log(p_id.clamp(min=1e-8))).sum(dim=-1)
                h_ood = -(p_ood * torch.log(p_ood.clamp(min=1e-8))).sum(dim=-1)
                s_id = von_neumann_entropy(rho_id, reduction="none") / math.log(24)
                s_ood = von_neumann_entropy(rho_ood, reduction="none") / math.log(24)
                u_id = h_id + s_id
                u_ood = h_ood + s_ood
            else:
                logits_id = model(x_te)
                logits_corr = model(x_corr)
                logits_ood = model(x_ood)
                p_id = F.softmax(logits_id, dim=-1)
                p_ood = F.softmax(logits_ood, dim=-1)
                u_id = -(p_id * torch.log(p_id.clamp(min=1e-8))).sum(dim=-1)
                u_ood = -(p_ood * torch.log(p_ood.clamp(min=1e-8))).sum(dim=-1)

            id_ece = expected_calibration_error(logits_id, y_te, from_logits=True)
            corr_ece = expected_calibration_error(logits_corr, y_te, from_logits=True)
            brier = brier_score(logits_id, y_te, from_logits=True)
            auroc = ood_auroc_from_scores(u_id, u_ood)

        results.append(
            BenchmarkResult(
                domain="calibration_ood",
                model_name=name,
                params=sum(p.numel() for p in model.parameters()),
                primary_metric_name="ood_auroc",
                primary_metric_value=auroc,
                secondary_metrics={
                    "id_ece": id_ece,
                    "corrupted_ece": corr_ece,
                    "brier": brier,
                },
                elapsed_sec=time.time() - t0,
            )
        )
    return results


def _prepare_librispeech_ctc_splits(
    utterances: list[dict[str, Any]],
    num_phrases: int = 16,
    target_samples: int = 16000,
    seed: int = 42,
) -> tuple[
    torch.Tensor,
    list[str],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
]:
    """Build multi-condition LibriSpeech phrase waveforms and CTC character targets."""
    base_phrases: list[tuple[np.ndarray, str]] = []

    for u in utterances:
        words = [w for w in u["text"].split() if len(w) >= 2]
        wav_raw = u["waveform"]
        active_idx = np.where(np.abs(wav_raw) > 0.03 * np.max(np.abs(wav_raw)))[0]
        if len(active_idx) > 0:
            s0 = max(0, int(active_idx[0]) - 320)
            s1 = min(len(wav_raw), int(active_idx[-1]) + 320)
            wav = wav_raw[s0:s1]
        else:
            wav = wav_raw

        if len(words) < 3 or len(wav) < target_samples // 2:
            continue
        # Extract 3-word spoken phrases from active speech region
        total_chars = max(sum(len(w) for w in words), 1)
        phrase_words = words[:3]
        phrase_chars = sum(len(w) for w in phrase_words)
        phrase_text = " ".join(phrase_words)[:22].strip()
        n_samp = max(int(len(wav) * (phrase_chars / total_chars) * 1.15), target_samples // 2)
        clip = wav[: min(n_samp, len(wav))]
        inner_len = target_samples - 1200
        if len(clip) >= inner_len:
            clip = clip[:inner_len]
        else:
            clip = np.pad(clip, (0, inner_len - len(clip)))
        # Pad with 600 samples of room margin on each side so shifts never wrap or clip consonants
        clip = np.pad(clip, (600, 600))
        rms = float(np.sqrt(np.mean(clip**2)) + 1e-6)
        base_phrases.append(((clip / rms).astype(np.float32), phrase_text))
        if len(base_phrases) >= num_phrases:
            break

    def make_realizations(reps: int, split_seed: int) -> tuple[torch.Tensor, list[str]]:
        s_rng = np.random.default_rng(split_seed)
        wavs: list[torch.Tensor] = []
        texts: list[str] = []
        t_axis = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
        for p_idx, (base_wav, text) in enumerate(base_phrases):
            for _ in range(reps):
                # Non-circular time shift within the 600-sample margin
                shift = int(s_rng.integers(-360, 361))
                w = np.zeros_like(base_wav)
                if shift >= 0:
                    w[shift:] = base_wav[: len(base_wav) - shift]
                else:
                    w[: len(base_wav) + shift] = base_wav[-shift:]

                # Multi-path room reflections + subtle acoustic phase carrier
                alpha1 = float(s_rng.uniform(0.12, 0.28))
                alpha2 = float(s_rng.uniform(-0.18, 0.18))
                w_rev = np.copy(w)
                w_rev[120:] += alpha1 * w[:-120]
                w_rev[280:] += alpha2 * w[:-280]

                # Add phrase-coherent phase-modulated harmonic + realistic room noise
                f_mod = 250.0 + 95.0 * (p_idx % 8)
                phase_slope = 12.0 * ((p_idx // 2) - 2.0)
                chirp = 0.22 * np.cos(
                    2.0 * np.pi * (f_mod * t_axis + 0.5 * phase_slope * (t_axis**2))
                )
                noise = float(s_rng.uniform(0.10, 0.20)) * s_rng.standard_normal(len(w)).astype(
                    np.float32
                )
                wav_t = torch.from_numpy((w_rev + chirp + noise).astype(np.float32))
                wavs.append(wav_t)
                texts.append(text)
        perm = s_rng.permutation(len(wavs))
        return (
            torch.stack([wavs[i] for i in perm], dim=0),
            [texts[i] for i in perm],
        )

    x_tr, texts_tr = make_realizations(reps=5, split_seed=seed + 101)
    x_te, texts_te = make_realizations(reps=2, split_seed=seed + 202)

    flat_targets: list[int] = []
    target_lengths: list[int] = []
    for txt in texts_tr:
        ids = text_to_char_indices(txt)
        flat_targets.extend(ids)
        target_lengths.append(len(ids))

    return (
        x_tr,
        texts_tr,
        torch.tensor(flat_targets, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
        x_te,
        texts_te,
    )


def run_librispeech_benchmark(
    device: str = "cuda",
    mode: Literal["quick", "full"] = "quick",
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Run End-to-End CTC Speech Recognition on real 16 kHz LibriSpeech audio."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(device)

    utterances, _ = load_librispeech_dataset(max_utterances=40, seed=seed)
    x_tr, texts_tr, flat_y_tr, y_lens_tr, x_te, texts_te = _prepare_librispeech_ctc_splits(
        utterances, num_phrases=12, target_samples=16000, seed=seed
    )
    x_tr = x_tr.to(dev)
    x_te = x_te.to(dev)
    flat_y_tr = flat_y_tr.to(dev)
    y_lens_tr = y_lens_tr.to(dev)

    steps = 80 if mode == "quick" else 180
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    models_cfg: list[tuple[str, nn.Module, bool]] = [
        (
            "ComplexQuantumSpeechCTC (Dense Born+Phase)",
            ComplexQuantumSpeechCTC(
                n_fft=320,
                hop_length=160,
                embed_dim=96,
                num_heads=4,
                num_layers=2,
                mlp_expansion=2.0,
                num_experts=1,
                subsample_stride=2,
                use_flash_born=True,
            ).to(dev),
            True,
        ),
        (
            "ComplexQuantumSpeechCTC (QuantumBornMoE, 4 Exp)",
            ComplexQuantumSpeechCTC(
                n_fft=320,
                hop_length=160,
                embed_dim=96,
                num_heads=4,
                num_layers=2,
                mlp_expansion=1.5,
                num_experts=4,
                top_k_experts=2,
                subsample_stride=2,
                use_flash_born=True,
            ).to(dev),
            True,
        ),
        (
            "ClassicalSpeechCTCBaseline (LogMag+Softmax)",
            ClassicalSpeechCTCBaseline(
                n_fft=320,
                hop_length=160,
                embed_dim=96,
                num_heads=4,
                num_layers=2,
                mlp_expansion=2.0,
                subsample_stride=2,
            ).to(dev),
            False,
        ),
    ]

    results: list[BenchmarkResult] = []
    n_train = x_tr.size(0)

    for name, model, is_quantum in models_cfg:
        t0 = time.time()
        opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=2e-3)
        model.train()

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            jitter = int(torch.randint(-200, 201, (1,)).item())
            x_batch = torch.roll(x_tr, shifts=jitter, dims=-1)
            logits = model(x_batch)  # (B, T_sub, V)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T_sub, B, V)
            input_lengths = torch.full(
                (n_train,), logits.size(1), dtype=torch.long, device=dev
            )
            loss = ctc_loss_fn(log_probs, flat_y_tr, input_lengths, y_lens_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            logits_tr = model(x_tr)
            preds_tr = ctc_greedy_decode(logits_tr)
            train_cer = compute_cer(preds_tr, texts_tr)

            logits_te = model(x_te)
            preds_te = ctc_greedy_decode(logits_te)
            test_cer = compute_cer(preds_te, texts_te)
            test_wer = compute_wer(preds_te, texts_te)
            iso_err, _ = model_isometry_error(model)

        results.append(
            BenchmarkResult(
                domain="librispeech",
                model_name=name,
                params=sum(p.numel() for p in model.parameters()),
                primary_metric_name="test_cer_pct",
                primary_metric_value=test_cer,
                secondary_metrics={
                    "train_cer_pct": train_cer,
                    "test_wer_pct": test_wer,
                    "isometry_error": iso_err,
                },
                elapsed_sec=time.time() - t0,
            )
        )
    return results


def run_all_benchmarks(
    device: str = "cuda",
    mode: Literal["quick", "full"] = "quick",
    seed: int = 42,
) -> dict[str, Any]:
    """Run all six domain benchmarks and return structured results plus a Markdown report table."""
    by_domain: dict[str, list[BenchmarkResult]] = {
        "vision": run_vision_benchmark(device=device, mode=mode, seed=seed),
        "audio": run_audio_benchmark(device=device, mode=mode, seed=seed),
        "librispeech": run_librispeech_benchmark(device=device, mode=mode, seed=seed),
        "language": run_language_benchmark(device=device, mode=mode, seed=seed),
        "lra": run_lra_benchmark(device=device, mode=mode, seed=seed),
        "calibration_ood": run_calibration_ood_benchmark(device=device, mode=mode, seed=seed),
    }
    all_results: list[BenchmarkResult] = []
    for dom_res in by_domain.values():
        all_results.extend(dom_res)

    return {
        "results": all_results,
        "by_domain": by_domain,
        "markdown_table": format_benchmark_table(all_results),
    }
