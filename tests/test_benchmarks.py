"""Unit tests for Track 1 multi-domain benchmark suite, models, and calibration metrics."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks import (
    ClassicalGPTBaseline,
    ClassicalViTBaseline,
    MagnitudeMelAudioBaseline,
    brier_score,
    expected_calibration_error,
    generate_associative_recall_batch,
    generate_phase_modulated_audio_dataset,
    ood_auroc_from_scores,
    run_audio_benchmark,
    run_calibration_ood_benchmark,
    run_language_benchmark,
    run_librispeech_benchmark,
    run_lra_benchmark,
    run_vision_benchmark,
)
from qdlblocks.models import (
    ClassicalSpeechCTCBaseline,
    ComplexQuantumSpeechCTC,
    ComplexSuperpositionAudioClassifier,
    QuantumBornViT,
    compute_cer,
    compute_wer,
    ctc_greedy_decode,
    load_librispeech_dataset,
    text_to_char_indices,
)


def test_quantum_born_vit_forward():
    torch.manual_seed(42)
    model = QuantumBornViT(
        image_size=16,
        patch_size=4,
        in_chans=3,
        num_classes=5,
        seed_dim=16,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        num_experts=4,
        top_k_experts=2,
        use_flash_born=False,
    )
    x = torch.randn(4, 3, 16, 16)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(x)
    assert logits.shape == (4, 5)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()


def test_complex_superposition_audio_classifier_and_phase_dataset():
    torch.manual_seed(7)
    x_stft, y = generate_phase_modulated_audio_dataset(
        num_samples=16,
        n_frames=12,
        n_freq_bins=16,
        num_classes=4,
        seed=7,
    )
    assert x_stft.is_complex()
    assert x_stft.shape == (16, 12, 16)
    assert y.shape == (16,)

    q_audio = ComplexSuperpositionAudioClassifier(
        n_freq_bins=16,
        hilbert_dim=8,
        num_classes=4,
        num_heads=2,
        num_kraus=2,
        purification_power=2,
    )
    logits, rho = q_audio(x_stft, return_density=True)
    assert logits.shape == (16, 4)
    assert rho.shape == (16, 8, 8)
    assert torch.all(torch.isfinite(logits))

    # Verify global per-frame phase invariance of cross-frequency covariance
    # Note: q_proj/k_proj/v_proj temporal wave interference depends on relative temporal phase,
    # while a global constant phase exp(i * phi_global) across the utterance cancels everywhere!
    global_phase = torch.exp(1j * torch.rand(16, 1, 1) * 6.2831853)
    logits_shifted = q_audio(x_stft * global_phase)
    assert torch.allclose(logits, logits_shifted, atol=1e-5)

    # Verify classical magnitude baseline is completely blind to phase
    mel_base = MagnitudeMelAudioBaseline(n_freq_bins=16, embed_dim=16, num_classes=4, num_heads=2)
    logits_1 = mel_base(x_stft)
    logits_2 = mel_base(torch.abs(x_stft).to(torch.complex64))
    assert torch.allclose(logits_1, logits_2, atol=1e-6)


def test_calibration_and_ood_metrics():
    probs = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    ece = expected_calibration_error(probs, targets, num_bins=5)
    brier = brier_score(probs, targets)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 2.0

    id_scores = torch.tensor([0.1, 0.2, 0.15, 0.25])
    ood_scores = torch.tensor([0.8, 0.9, 0.75, 0.95])
    auroc = ood_auroc_from_scores(id_scores, ood_scores)
    assert abs(auroc - 1.0) < 1e-6


def test_classical_baselines_and_associative_recall():
    x_ar, y_ar = generate_associative_recall_batch(
        batch_size=8,
        seq_len=16,
        num_pairs=4,
        vocab_size=32,
        seed=42,
    )
    assert x_ar.shape == (8, 16)
    assert y_ar.shape == (8, 16)

    vit_base = ClassicalViTBaseline(
        image_size=16,
        patch_size=4,
        in_chans=3,
        num_classes=4,
        seed_dim=16,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
    )
    logits_v = vit_base(torch.randn(4, 3, 16, 16))
    assert logits_v.shape == (4, 4)

    gpt_base = ClassicalGPTBaseline(
        vocab_size=32,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=64,
        block_size=16,
    )
    logits_g, loss_g = gpt_base(x_ar, y_ar)
    assert logits_g.shape == (8, 16, 32) and loss_g is not None


def test_complex_quantum_speech_ctc_and_librispeech():
    torch.manual_seed(42)
    q_ctc = ComplexQuantumSpeechCTC(
        n_fft=160,
        hop_length=80,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        num_experts=4,
        top_k_experts=2,
        subsample_stride=2,
        use_flash_born=False,
    )
    c_ctc = ClassicalSpeechCTCBaseline(
        n_fft=160,
        hop_length=80,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        subsample_stride=2,
    )
    wav = torch.randn(3, 3200)
    logits_q = q_ctc(wav)
    logits_c = c_ctc(wav)
    assert logits_q.shape == logits_c.shape
    assert logits_q.size(0) == 3 and logits_q.size(2) == 29

    # Verify global constant complex phase shift invariance of ComplexQuantumSpeechCTC on STFT input
    stft_in = q_ctc.stem.waveform_to_stft(wav)
    global_phase = torch.exp(1j * torch.rand(3, 1, 1) * 6.2831853)
    logits_s1 = q_ctc(stft_in)
    logits_s2 = q_ctc(stft_in * global_phase)
    assert torch.allclose(logits_s1, logits_s2, atol=1e-4)

    decoded = ctc_greedy_decode(logits_q)
    assert len(decoded) == 3
    ids = text_to_char_indices("HELLO WORLD")
    assert len(ids) == 11
    assert compute_cer(["HELLO WORLD"], ["HELLO WORLD"]) == 0.0
    assert compute_wer(["HELLO WORLD"], ["HELLO WORLD"]) == 0.0
    assert compute_cer(["HELLO"], ["WORLD"]) > 0.0

    utts, desc = load_librispeech_dataset(max_utterances=4, seed=42)
    assert len(utts) == 4 and "waveform" in utts[0] and len(desc) > 0


def test_benchmark_runners_quick():
    vis_res = run_vision_benchmark(device="cpu", mode="quick", seed=42)
    assert len(vis_res) == 3
    assert "train_acc_pct" in vis_res[0].secondary_metrics

    aud_res = run_audio_benchmark(device="cpu", mode="quick", seed=42)
    assert len(aud_res) == 3
    assert "train_acc_pct" in aud_res[0].secondary_metrics
    # Quantum audio classifier must beat phase-blind Mel baseline on phase-modulated chords
    assert aud_res[0].primary_metric_value > aud_res[2].primary_metric_value

    ls_res = run_librispeech_benchmark(device="cpu", mode="quick", seed=42)
    assert len(ls_res) == 3
    # Quantum speech CTC must achieve lower Character Error Rate than phase-blind classical baseline
    assert ls_res[0].primary_metric_value < ls_res[2].primary_metric_value

    lang_res = run_language_benchmark(device="cpu", mode="quick", seed=42)
    assert len(lang_res) == 3

    lra_res = run_lra_benchmark(device="cpu", mode="quick", seed=42)
    assert len(lra_res) == 3

    cal_res = run_calibration_ood_benchmark(device="cpu", mode="quick", seed=42)
    assert len(cal_res) == 2

