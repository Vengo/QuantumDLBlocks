#!/usr/bin/env python3
"""Example 10: Quantum-Geometric Automatic Speech Recognition (ASR) on LibriSpeech.

Demonstrates end-to-end Speech Recognition on real 16 kHz LibriSpeech utterances
using QuantumDLBlocks:
1. ComplexPhaseAcousticStem: Computes complex STFT and extracts gauge-invariant
   instantaneous frequency (X_t * X_{t-1}^*) and spectral group delay (X_f * X_{f-1}^*)
   before projecting onto a spherical Hilbert space via ComplexLinear + Stiefel lifting.
2. QuantumBornBlock Encoder: Processes acoustic frames using squared-amplitude Born-rule
   attention (|Q K^T|^2) with Riemannian centering and optional zero-Softmax Sparse Born
   Mixture of Experts (QuantumBornMoE) for specialized phoneme subspace routing.
3. ProjectivePOVMHead + CTC Loss: Maps normalized acoustic states onto 29 character
   measurement operators (blank, space, A-Z, ') and decodes transcripts.

Also evaluates utterance-disjoint LibriSpeech spoken-word segment classification to verify
generalization across unseen utterances.
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.benchmarks import (
    MagnitudeMelAudioBaseline,
    format_benchmark_table,
    run_librispeech_benchmark,
)
from qdlblocks.benchmarks.runner import _prepare_librispeech_ctc_splits
from qdlblocks.models import ComplexSuperpositionAudioClassifier
from qdlblocks.models.speech_ctc import (
    ClassicalSpeechCTCBaseline,
    ComplexQuantumSpeechCTC,
    compute_cer,
    compute_wer,
    ctc_greedy_decode,
    extract_librispeech_word_segments,
    load_librispeech_dataset,
)


def run_transcript_showcase(device: str = "cuda", seed: int = 42) -> None:
    """Train ComplexQuantumSpeechCTC vs ClassicalSpeechCTCBaseline and print decoded transcripts."""
    torch.manual_seed(seed)
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    utterances, source_desc = load_librispeech_dataset(max_utterances=40, seed=seed)
    print(f"\nDataset Source: {source_desc}")
    x_tr, texts_tr, flat_y_tr, y_lens_tr, x_te, texts_te = _prepare_librispeech_ctc_splits(
        utterances, num_phrases=12, target_samples=16000, seed=seed
    )
    x_tr = x_tr.to(dev)
    x_te = x_te.to(dev)
    flat_y_tr = flat_y_tr.to(dev)
    y_lens_tr = y_lens_tr.to(dev)

    q_model = ComplexQuantumSpeechCTC(
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
    ).to(dev)
    c_model = ClassicalSpeechCTCBaseline(
        n_fft=320,
        hop_length=160,
        embed_dim=96,
        num_heads=4,
        num_layers=2,
        mlp_expansion=2.0,
        subsample_stride=2,
    ).to(dev)

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    n_train = x_tr.size(0)

    for label, model in [
        ("ComplexQuantumSpeechCTC (QuantumBornMoE + Complex Phase Stem)", q_model),
        ("ClassicalSpeechCTCBaseline (Phase-Blind LogMag + Softmax)", c_model),
    ]:
        opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=2e-3)
        model.train()
        for _ in range(85):
            opt.zero_grad(set_to_none=True)
            jitter = int(torch.randint(-200, 201, (1,)).item())
            x_batch = torch.roll(x_tr, shifts=jitter, dims=-1)
            logits = model(x_batch)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            in_lens = torch.full((n_train,), logits.size(1), dtype=torch.long, device=dev)
            loss = ctc_loss_fn(log_probs, flat_y_tr, in_lens, y_lens_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds_te = ctc_greedy_decode(model(x_te))

        print(f"\n--- Held-Out LibriSpeech Decoded Phrases: {label} ---")
        for i in range(min(5, len(preds_te))):
            ref = texts_te[i]
            hyp = preds_te[i]
            cer = compute_cer([hyp], [ref])
            wer = compute_wer([hyp], [ref])
            print(f"  [Test Clip {i + 1}] CER={cer:5.1f}% | WER={wer:5.1f}% | Ground Truth: {ref!r:24s} -> Decoded: {hyp!r}")


def run_word_segment_generalization(device: str = "cuda", seed: int = 42) -> None:
    """Train & evaluate utterance-disjoint LibriSpeech spoken word segment classification."""
    print("\n--- Utterance-Disjoint LibriSpeech Spoken Word Segment Classification ---")
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    utterances, _ = load_librispeech_dataset(max_utterances=40, seed=seed)
    x_train, y_train, x_test, y_test, vocab = extract_librispeech_word_segments(
        utterances,
        num_classes=8,
        segment_frames=20,
        n_fft=62,
        hop_length=32,
        seed=seed,
    )
    print(f"Target Spoken Words ({len(vocab)} classes): {vocab}")
    print(
        f"Train segments: {x_train.shape[0]} (from 75% train utterances) | "
        f"Test segments: {x_test.shape[0]} (from 25% held-out unseen utterances)"
    )

    x_tr, y_tr = x_train.to(dev), y_train.to(dev)
    x_te, y_te = x_test.to(dev), y_test.to(dev)
    n_freq = x_tr.size(-1)

    for name, model in [
        (
            "ComplexSuperpositionAudio (Dense Kraus + POVM)",
            ComplexSuperpositionAudioClassifier(
                n_freq_bins=n_freq, hilbert_dim=24, num_classes=len(vocab), num_heads=4, num_kraus=4
            ).to(dev),
        ),
        (
            "ComplexSuperpositionAudio (SparseKrausMoE + POVM)",
            ComplexSuperpositionAudioClassifier(
                n_freq_bins=n_freq,
                hilbert_dim=24,
                num_classes=len(vocab),
                num_heads=4,
                use_sparse_kraus_moe=True,
                num_experts=6,
                top_k_experts=2,
            ).to(dev),
        ),
        (
            "MagnitudeMelAudioBaseline (Phase-Blind LogMag + Softmax)",
            MagnitudeMelAudioBaseline(
                n_freq_bins=n_freq, embed_dim=24, num_classes=len(vocab), num_heads=4, num_layers=1
            ).to(dev),
        ),
    ]:
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
        model.train()
        for _ in range(80):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, x_tr.size(0), (48,), device=dev)
            logits = model(x_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_acc = (model(x_tr).argmax(dim=-1) == y_tr).float().mean().item() * 100.0
            te_acc = (model(x_te).argmax(dim=-1) == y_te).float().mean().item() * 100.0
        print(f"  {name:56s} | Train Acc: {tr_acc:6.2f}% | Unseen Utterance Test Acc: {te_acc:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="LibriSpeech Quantum-Geometric ASR Benchmark & Showcase")
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Benchmark mode ('quick' or 'full').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu').",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    print("=" * 92)
    print(f"Example 10: LibriSpeech 16 kHz Quantum-Geometric Speech Recognition | mode={args.mode} | device={args.device}")
    print("=" * 92)

    results = run_librispeech_benchmark(device=args.device, mode=args.mode, seed=args.seed)  # type: ignore[arg-type]
    print("\n" + format_benchmark_table(results))

    run_transcript_showcase(device=args.device, seed=args.seed)
    run_word_segment_generalization(device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
