"""Wave 49 (`EXPERIMENT_PLAN_064.md`) Runner: Modular Quantum Statistical-Mechanics GPT Experiments on FineWeb.

Executes the prioritized FineWeb experiments using the factored `ModularQuantumStatMechGPT` building blocks:
1. **Priority 1 & 2 (`--mode train_single`)**:
   - Rebuilt `Wave 48 v3` (`SOFTCAP_LINEAR_CE`) scaling run (`100M` to `600M` tokens) and head-to-head
     `100M`-token comparison against the **Mean-Centered Isometric Born Trace Readout**
     (`MEAN_CENTERED_ISOMETRIC_TRACE`).
2. **Priority 3 (`--mode train_single --channel-mode cptp_kraus_rank_k`)**:
   - `50M`-token Completely Positive Trace-Preserving (CPTP) Rank-$K=4$ Kraus channel benchmark
     ($\\mathcal{E}_h(\\rho) = \\sum_{k=1}^4 A_{h,k}\\rho A_{h,k}^\\dagger$, $\\sum_{k=1}^4 A_{h,k}^\\dagger A_{h,k} = I_{16}$).
3. **Priorities 4–7 (`--mode ablation_suite`)**:
   - Controlled FineWeb comparisons for Embedding Trainability (`TRAINABLE_SPHERE_MANIFOLD` vs.
     `FROZEN_RANDOM_SPHERE` vs. `FROZEN_PPMI_SVD_SPHERE`), Readout Divergence (`SOFTCAP_LINEAR_CE` vs.
     `RANK2_KRYLOV_BORN` vs. `MEAN_CENTERED_ISOMETRIC_TRACE` with rank-bucketed NLL telemetry),
     Purification (`QUADRATIC_MCWEENY` vs. `CUBIC_FERMI_DIRAC` vs. `LEARNABLE_TEMP_MCWEENY`), and
     Detached Purity / Tsallis-2 Entropy Gate (`NONE` vs. `DETACHED_TSALLIS_PURITY_GATE`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.layers.cooccurrence_init import compute_ppmi_svd_embeddings
from qdlblocks.layers.isometric_born_trace_readout import (
    VocabularyReadoutMode,
    evaluate_readout_divergence_batch,
)
from qdlblocks.layers.quantum_statmech_channels import (
    DEFAULT_KRAUS_CHANNEL_RANK,
    EntropyGateMode,
    PurificationMode,
    QuantumChannelMode,
    StiefelMomentumMode,
)
from qdlblocks.layers.wavepacket_statmech_block import (
    ClassicalQuantumFrontEndMode,
    HybridQuantumFrontEndClassicalGPT,
    build_hybrid_quantum_classical_optimizers,
)
from qdlblocks.models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
)
from qdlblocks.models.modular_quantum_statmech_gpt import (
    DEFAULT_STATMECH_EMBED_DIM,
    DEFAULT_STATMECH_NUM_HEADS,
    DEFAULT_STATMECH_NUM_LAYERS,
    DEFAULT_STATMECH_VOCAB_SIZE,
    EmbeddingTrainMode,
    ModularQuantumStatMechConfig,
    ModularQuantumStatMechGPT,
    build_modular_quantum_statmech_optimizers,
)

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_064.md)
# ---------------------------------------------------------------------------
CACHE_DIR = REPO_ROOT / ".cache"
FINEWEB_DIR = CACHE_DIR / "fineweb_100M"
DEFAULT_JSON_OUT = CACHE_DIR / "hypothesis_experiments_064.json"
PPMI_CACHE_PT = CACHE_DIR / "ppmi_svd_50304_768_shard1.pt"
SHARD_HEADER_INTS: int = 256
FAST_SCREENING_VAL_TOKENS: int = 65_536
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_SEQ_LEN: int = 1024


class MultiShardTokenStream:
    """Streams contiguous `(B, T)` token batches sequentially across all 9 local FineWeb shards."""

    def __init__(self, shard_paths: list[Path], batch_size: int, seq_len: int, vocab_size: int) -> None:
        if not shard_paths:
            raise FileNotFoundError("No FineWeb training shards found.")
        self.shard_paths = list(shard_paths)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.span = self.batch_size * (self.seq_len + 1)
        self.shard_idx = 0
        self.offset = self.span
        self._tokens = self._load_shard(self.shard_paths[0])

    @staticmethod
    def _load_shard(path: Path) -> np.ndarray:
        raw = np.memmap(path, dtype=np.uint16, mode="r")
        header = np.fromfile(path, dtype=np.int32, count=SHARD_HEADER_INTS)
        if len(header) >= 3 and int(header[0]) == 20240520:
            return raw[SHARD_HEADER_INTS * 2 :]
        return raw

    def advance_steps(self, num_steps: int) -> None:
        """Fast-forward token stream pointer by `num_steps` batches without materializing tensors."""
        for _ in range(max(0, int(num_steps))):
            if self.offset + self.span > len(self._tokens):
                self.shard_idx = (self.shard_idx + 1) % len(self.shard_paths)
                self._tokens = self._load_shard(self.shard_paths[self.shard_idx])
                self.offset = 0
            self.offset += self.span

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return next `(idx, targets)` batch of shape `(B, T)` on `device` matching Wave 48 layout."""
        if self.offset + self.span > len(self._tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.shard_paths)
            self._tokens = self._load_shard(self.shard_paths[self.shard_idx])
            self.offset = 0
        chunk = np.asarray(self._tokens[self.offset : self.offset + self.span], dtype=np.int64)
        self.offset += self.span
        t_cpu = torch.from_numpy(chunk).clamp_(0, self.vocab_size - 1).view(self.batch_size, self.seq_len + 1)
        x = t_cpu[:, :-1].contiguous().to(device, non_blocking=True)
        y = t_cpu[:, 1:].contiguous().to(device, non_blocking=True)
        return x, y


@dataclass(frozen=True)
class StatMechRunSpec:
    """Specification for a single FineWeb training run (`<= 8` fields, Rule 2)."""

    run_id: str
    cfg: ModularQuantumStatMechConfig
    embedding_mode: EmbeddingTrainMode
    steps: int
    eval_every: int
    device: torch.device
    out_json: Path
    ckpt_path: Path | None = None


def load_fineweb_shards_and_val(vocab_size: int = DEFAULT_STATMECH_VOCAB_SIZE) -> tuple[list[Path], torch.Tensor]:
    """Locate all FineWeb training shards and load the deterministic `65,536`-token validation slice."""
    shard_paths = sorted(FINEWEB_DIR.glob("fineweb_train_*.bin"))
    if not shard_paths:
        fallback = FINEWEB_DIR / "train_full_100M.bin"
        if fallback.exists():
            shard_paths = [fallback]
    val_path = FINEWEB_DIR / "fineweb_val_000000.bin"
    if not val_path.exists():
        val_path = FINEWEB_DIR / "val_slice.bin"
    raw_val = MultiShardTokenStream._load_shard(val_path)
    n_val = min(len(raw_val), 16 * DEFAULT_BATCH_SIZE * (DEFAULT_SEQ_LEN + 1))
    val_t = torch.from_numpy(np.asarray(raw_val[:n_val], dtype=np.int64)).clamp_(0, vocab_size - 1)
    return shard_paths, val_t


@torch.no_grad()
def evaluate_validation_with_divergence(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on the deterministic `65,536`-token validation prefix with rank-bucketed NLL."""
    model.eval()
    eval_bsz = min(4, batch_size)
    eval_batches = max(8, (8 * batch_size) // max(1, eval_bsz))
    span = eval_bsz * (seq_len + 1)
    accum: dict[str, float] = {}
    for b_idx in range(eval_batches):
        s = b_idx * span
        chunk = val_tokens[s : s + span].to(device, non_blocking=True).view(eval_bsz, seq_len + 1)
        idx = chunk[:, :-1].contiguous()
        targets = chunk[:, 1:].contiguous()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits, _ = model(idx, targets=None, return_logits=True)
        assert logits is not None
        batch_div = evaluate_readout_divergence_batch(logits.float(), targets)
        for k, v in batch_div.items():
            accum[k] = accum.get(k, 0.0) + float(v)
    model.train()
    return {k: float(v / eval_batches) for k, v in accum.items()}


def get_or_build_ppmi_embeddings(shard_paths: list[Path], device: torch.device) -> torch.Tensor:
    """Load or build Shifted-PPMI + Randomized SVD `50304 x 768` token embeddings from Shard 1."""
    first_arr = MultiShardTokenStream._load_shard(shard_paths[0])[:25_000_000]
    res = compute_ppmi_svd_embeddings(
        token_arrays=[first_arr],
        vocab_size=DEFAULT_STATMECH_VOCAB_SIZE,
        embed_dim=DEFAULT_STATMECH_EMBED_DIM,
        num_context_buckets=2048,
        window_size=4,
        device=device,
        cache_path=PPMI_CACHE_PT,
        seed=42,
    )
    return res["embeddings"].float()


def cosine_lr_schedule(step: int, total_steps: int) -> float:
    """Compute full-horizon cosine decay learning-rate multiplier (`1.0 -> 0.15`) over `total_steps`."""
    tau = float(step) / float(max(1, total_steps - 1))
    tau = max(0.0, min(1.0, tau))
    return float(0.5 * (1.0 + math.cos(math.pi * tau)) * 0.85 + 0.15)


def run_statmech_training_trajectory(
    spec: StatMechRunSpec,
    shard_paths: list[Path],
    val_tokens: torch.Tensor,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = 2026063,
    momentum_mode: StiefelMomentumMode = StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED,
) -> dict[str, Any]:
    """Execute a complete FineWeb training run with periodic validation divergence + manifold telemetry logging (`<= 8` args)."""
    if spec.device.type == "cuda":
        torch.cuda.set_device(spec.device)
    torch.manual_seed(seed)

    model = ModularQuantumStatMechGPT(spec.cfg, kraus_rank=DEFAULT_KRAUS_CHANNEL_RANK).to(spec.device)
    if spec.embedding_mode in (
        EmbeddingTrainMode.FROZEN_PPMI_SVD_SPHERE,
        EmbeddingTrainMode.WAVEPACKET_SPHERE_48X16,
        EmbeddingTrainMode.CAUSAL_WAVEPACKET_48X16,
        EmbeddingTrainMode.NATIVE_WAVEPACKET_48X16,
        EmbeddingTrainMode.NATIVE_CAUSAL_WAVEPACKET_48X16,
    ):
        ppmi_emb = get_or_build_ppmi_embeddings(shard_paths, spec.device)
        model.configure_embedding_mode(spec.embedding_mode, ppmi_embeddings=ppmi_emb)
    else:
        model.configure_embedding_mode(spec.embedding_mode)

    model.set_compiled_krylov(True)
    opt = build_modular_quantum_statmech_optimizers(model, momentum_mode=momentum_mode)
    stream = MultiShardTokenStream(shard_paths, batch_size=batch_size, seq_len=seq_len, vocab_size=spec.cfg.vocab_size)
    tokens_per_step = batch_size * seq_len

    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    val0 = evaluate_validation_with_divergence(model, val_tokens, batch_size, seq_len, spec.device)
    tel0 = model.compute_manifold_and_purity_telemetry()
    step0_rec = {"step": 0, "tokens": 0, "elapsed_s": 0.0, **val0, **tel0}
    history.append(step0_rec)
    print(
        f"[{spec.run_id}] step=0 tok=0.00M val_ce={val0['val_ce']:.4f} top1={val0['top1_acc']*100:.2f}% "
        f"top5={val0['top5_acc']*100:.2f}% gap={val0['gap_top5_top1']*100:.2f}% mom={momentum_mode.value}",
        flush=True,
    )

    ema_train_loss: float | None = None
    milestone_steps = {100, 250, 500, 1000, 2500, 6104, 9155, 12207, 15259, 18311, 24414, 36621, 48828, 61035, 73242}

    for step in range(1, spec.steps + 1):
        idx, targets = stream.next_batch(spec.device)
        lr_mult = cosine_lr_schedule(step - 1, spec.steps)

        should_eval = (step % spec.eval_every == 0) or (step in milestone_steps) or (step == spec.steps)
        if should_eval:
            opt.trunk_opt.collect_diagnostics = True

        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets=targets, return_logits=False)
        assert loss is not None
        loss.backward()
        model.clip_riemannian_tangent_grad_norm_(1.0)
        opt.step(idx=idx, targets=targets, lr_scale=lr_mult)

        loss_v = float(loss.detach().item())
        ema_train_loss = loss_v if ema_train_loss is None else (0.95 * ema_train_loss + 0.05 * loss_v)

        if should_eval:
            elapsed = float(time.perf_counter() - t0)
            tot_tokens = step * tokens_per_step
            tok_s = float(tot_tokens / max(1e-6, elapsed))
            val_m = evaluate_validation_with_divergence(model, val_tokens, batch_size, seq_len, spec.device)
            tel_m = model.compute_manifold_and_purity_telemetry()
            vec_diag = opt.trunk_opt.get_last_step_diagnostics()
            rec = {
                "step": int(step),
                "tokens": int(tot_tokens),
                "tokens_m": round(tot_tokens / 1e6, 3),
                "train_ce_ema": float(ema_train_loss),
                "elapsed_s": round(elapsed, 2),
                "tok_per_s": round(tok_s, 1),
                "lr_mult": round(lr_mult, 4),
                **val_m,
                **tel_m,
                **vec_diag,
            }
            history.append(rec)
            print(
                f"[{spec.run_id}] step={step}/{spec.steps} ({tot_tokens/1e6:.2f}M tok, {tok_s:.0f} tok/s) "
                f"train_ce={ema_train_loss:.4f} val_ce={val_m['val_ce']:.4f} top1={val_m['top1_acc']*100:.2f}% "
                f"top5={val_m['top5_acc']*100:.2f}% nll2_5={val_m['nll_rank2_5']:.3f} nll6_50={val_m['nll_rank6_50']:.3f} "
                f"purity={tel_m['mean_purity']:.3f} eff_rank={tel_m['mean_eff_rank']:.2f}",
                flush=True,
            )
            result_payload = {
                "run_id": spec.run_id,
                "config": {
                    "vocab_size": spec.cfg.vocab_size,
                    "n_layer": spec.cfg.n_layer,
                    "n_head": spec.cfg.n_head,
                    "n_embd": spec.cfg.n_embd,
                    "purification_mode": spec.cfg.purification_mode.value,
                    "channel_mode": spec.cfg.channel_mode.value,
                    "entropy_gate_mode": spec.cfg.entropy_gate_mode.value,
                    "readout_mode": spec.cfg.readout_mode.value,
                    "embedding_mode": spec.embedding_mode.value,
                    "momentum_mode": momentum_mode.value,
                },
                "steps": spec.steps,
                "completed_steps": step,
                "tokens_processed": tot_tokens,
                "final": rec,
                "history": history,
            }
            spec.out_json.parent.mkdir(parents=True, exist_ok=True)
            spec.out_json.write_text(json.dumps(result_payload, indent=2))
            if spec.ckpt_path is not None and (step in {6104, 9155, 12207, 15259, 18311, 24414, 36621, 48828, 61035, 73242} or step == spec.steps):
                spec.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"step": step, "tokens": tot_tokens, "config": result_payload["config"], "model_state_dict": model.state_dict()},
                    spec.ckpt_path,
                )

    return {
        "run_id": spec.run_id,
        "final": history[-1],
        "history": history,
    }


def run_ablation_suite(
    shard_paths: list[Path],
    val_tokens: torch.Tensor,
    device: torch.device,
    steps: int = 500,
    eval_every: int = 100,
    out_json: Path = DEFAULT_JSON_OUT,
    subset: str = "all",
) -> dict[str, Any]:
    """Run controlled ablations for Priorities 4 (Embeddings), 5 (Readout Divergence), 6 (Purification), and 7 (Tsallis Gate)."""
    all_arms: list[tuple[str, ModularQuantumStatMechConfig, EmbeddingTrainMode]] = [
        (
            "baseline_v3_rebuilt_quad_u16_softcap_trainable_wte",
            ModularQuantumStatMechConfig(
                purification_mode=PurificationMode.QUADRATIC_MCWEENY,
                channel_mode=QuantumChannelMode.UNITARY_U16,
                entropy_gate_mode=EntropyGateMode.NONE,
                readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
            ),
            EmbeddingTrainMode.TRAINABLE_SPHERE_MANIFOLD,
        ),
        (
            "p4_embed_frozen_random_sphere",
            ModularQuantumStatMechConfig(readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE),
            EmbeddingTrainMode.FROZEN_RANDOM_SPHERE,
        ),
        (
            "p4_embed_frozen_ppmi_svd_sphere",
            ModularQuantumStatMechConfig(readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE),
            EmbeddingTrainMode.FROZEN_PPMI_SVD_SPHERE,
        ),
        (
            "p5_readout_rank2_krylov_born",
            ModularQuantumStatMechConfig(readout_mode=VocabularyReadoutMode.RANK2_KRYLOV_BORN),
            EmbeddingTrainMode.TRAINABLE_SPHERE_MANIFOLD,
        ),
        (
            "p5_readout_mean_centered_isometric_trace",
            ModularQuantumStatMechConfig(readout_mode=VocabularyReadoutMode.MEAN_CENTERED_ISOMETRIC_TRACE),
            EmbeddingTrainMode.TRAINABLE_SPHERE_MANIFOLD,
        ),
        (
            "p6_purify_cubic_fermi_dirac",
            ModularQuantumStatMechConfig(
                purification_mode=PurificationMode.CUBIC_FERMI_DIRAC,
                readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
            ),
            EmbeddingTrainMode.TRAINABLE_SPHERE_MANIFOLD,
        ),
        (
            "p6_purify_learnable_temperature",
            ModularQuantumStatMechConfig(
                purification_mode=PurificationMode.LEARNABLE_TEMP_MCWEENY,
                readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
            ),
            EmbeddingTrainMode.TRAINABLE_SPHERE_MANIFOLD,
        ),
        (
            "p7_gate_detached_tsallis_purity",
            ModularQuantumStatMechConfig(
                entropy_gate_mode=EntropyGateMode.DETACHED_TSALLIS_PURITY_GATE,
                readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
            ),
            EmbeddingTrainMode.TRAINABLE_SPHERE_MANIFOLD,
        ),
    ]
    if subset == "gpu0_fast":
        arms = all_arms[:4]
    elif subset == "gpu1_fast":
        arms = all_arms[4:]
    else:
        arms = all_arms

    suite_results: dict[str, Any] = {"steps_per_arm": steps, "tokens_per_arm": steps * DEFAULT_BATCH_SIZE * DEFAULT_SEQ_LEN, "arms": {}}
    if out_json.exists():
        try:
            existing = json.loads(out_json.read_text())
            if isinstance(existing, dict) and "arms" in existing:
                suite_results["arms"].update(existing["arms"])
        except Exception:
            pass

    for arm_id, cfg, emb_mode in arms:
        arm_path = CACHE_DIR / f"w49_ablation_{arm_id}_{steps}steps.json"
        if arm_path.exists():
            try:
                cached_arm = json.loads(arm_path.read_text())
                if int(cached_arm.get("completed_steps", 0)) >= steps:
                    print(f"[{arm_id}] Already completed ({steps} steps), loading from {arm_path.name}", flush=True)
                    suite_results["arms"][arm_id] = cached_arm
                    out_json.write_text(json.dumps(suite_results, indent=2))
                    continue
            except Exception:
                pass
        spec = StatMechRunSpec(
            run_id=arm_id,
            cfg=cfg,
            embedding_mode=emb_mode,
            steps=steps,
            eval_every=eval_every,
            device=device,
            out_json=arm_path,
        )
        res = run_statmech_training_trajectory(
            spec,
            shard_paths,
            val_tokens,
            momentum_mode=StiefelMomentumMode(args.momentum_mode),
        )
        suite_results["arms"][arm_id] = res
        out_json.write_text(json.dumps(suite_results, indent=2))
        torch.cuda.empty_cache()

    return suite_results


def run_single_resume_arm(
    arm_id: str,
    ckpt_payload: dict[str, Any],
    shard_paths: list[Path],
    val_tokens: torch.Tensor,
    device: torch.device,
    momentum_mode: StiefelMomentumMode,
    lr_restart_peak: float | None = None,
    delta_steps: int = 586,
) -> dict[str, Any]:
    """Resume from `ckpt_payload` (`step=24414`) for `delta_steps` (`-> 25000`) with full vector diagnostics (`<= 8` args)."""
    if device.type == "cuda":
        torch.cuda.set_device(device)
    start_step = int(ckpt_payload["step"])
    cfg_d = ckpt_payload["config"]
    cfg = ModularQuantumStatMechConfig(
        vocab_size=int(cfg_d["vocab_size"]),
        n_layer=int(cfg_d["n_layer"]),
        n_head=int(cfg_d["n_head"]),
        n_embd=int(cfg_d["n_embd"]),
        purification_mode=PurificationMode(cfg_d["purification_mode"]),
        channel_mode=QuantumChannelMode(cfg_d["channel_mode"]),
        entropy_gate_mode=EntropyGateMode(cfg_d["entropy_gate_mode"]),
        readout_mode=VocabularyReadoutMode(cfg_d["readout_mode"]),
    )
    model = ModularQuantumStatMechGPT(cfg, kraus_rank=DEFAULT_KRAUS_CHANNEL_RANK).to(device)
    model.load_state_dict(ckpt_payload["model_state_dict"])
    model.configure_embedding_mode(EmbeddingTrainMode(cfg_d.get("embedding_mode", "trainable_sphere_manifold")))
    model.set_compiled_krylov(True)

    opt = build_modular_quantum_statmech_optimizers(model, momentum_mode=momentum_mode)
    stream = MultiShardTokenStream(
        shard_paths, batch_size=DEFAULT_BATCH_SIZE, seq_len=DEFAULT_SEQ_LEN, vocab_size=cfg.vocab_size
    )
    stream.advance_steps(start_step)
    tokens_per_step = DEFAULT_BATCH_SIZE * DEFAULT_SEQ_LEN

    val0 = evaluate_validation_with_divergence(model, val_tokens, DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, device)
    tel0 = model.compute_manifold_and_purity_telemetry()
    history: list[dict[str, Any]] = [
        {"step": start_step, "delta_step": 0, "tokens": start_step * tokens_per_step, **val0, **tel0}
    ]
    print(
        f"[{arm_id}] RESUME start_step={start_step} val_ce={val0['val_ce']:.4f} "
        f"top1={val0['top1_acc']*100:.2f}% top5={val0['top5_acc']*100:.2f}%",
        flush=True,
    )

    ema_train_loss: float | None = None
    t0 = time.perf_counter()
    eval_steps = {24450, 24500, 24600, 24700, 24800, 24900, start_step + delta_steps}
    diag_only_deltas = {1, 5, 10, 25, 50}
    vector_diag_trace: list[dict[str, Any]] = []

    for d_step in range(1, delta_steps + 1):
        cur_step = start_step + d_step
        idx, targets = stream.next_batch(device)
        if lr_restart_peak is not None:
            tau = float(d_step - 1) / float(max(1, delta_steps - 1))
            lr_mult = float(0.15 + (lr_restart_peak - 0.15) * 0.5 * (1.0 + math.cos(math.pi * tau)))
        else:
            lr_mult = cosine_lr_schedule(cur_step - 1, 73242)

        need_eval = (cur_step in eval_steps) or (d_step == delta_steps)
        need_diag = need_eval or (d_step in diag_only_deltas)
        if need_diag:
            opt.trunk_opt.collect_diagnostics = True

        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets=targets, return_logits=False)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(idx=idx, targets=targets, lr_scale=lr_mult)

        loss_v = float(loss.detach().item())
        ema_train_loss = loss_v if ema_train_loss is None else (0.95 * ema_train_loss + 0.05 * loss_v)

        if need_diag:
            v_diag = opt.trunk_opt.get_last_step_diagnostics()
            vector_diag_trace.append(
                {"step": cur_step, "delta_step": d_step, "lr_mult": round(lr_mult, 4), **v_diag}
            )
            if d_step in diag_only_deltas and not need_eval:
                print(
                    f"[{arm_id}] diag d_step={d_step} (step={cur_step}) "
                    f"grad_norm_f={v_diag['grad_normal_frac']:.4f} "
                    f"pre_mom_norm_f={v_diag['pre_mom_normal_frac']:.4f} "
                    f"post_mom_norm_f={v_diag['post_mom_normal_at_w_next_frac']:.4f} "
                    f"leak_N2T={v_diag['normal_to_tangent_leakage_ratio']:.4f} "
                    f"cos(gtan,mtan)={v_diag['cos_gtan_mtan']:.4f} "
                    f"cos(gtan,utan)={v_diag['cos_gtan_utan']:.4f}",
                    flush=True,
                )

        if need_eval:
            elapsed = float(time.perf_counter() - t0)
            tok_s = float((d_step * tokens_per_step) / max(1e-6, elapsed))
            val_m = evaluate_validation_with_divergence(model, val_tokens, DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, device)
            tel_m = model.compute_manifold_and_purity_telemetry()
            v_diag = opt.trunk_opt.get_last_step_diagnostics()
            rec = {
                "step": cur_step,
                "delta_step": d_step,
                "tokens": cur_step * tokens_per_step,
                "tokens_m": round((cur_step * tokens_per_step) / 1e6, 3),
                "train_ce_ema": float(ema_train_loss),
                "elapsed_s": round(elapsed, 2),
                "tok_per_s": round(tok_s, 1),
                "lr_mult": round(lr_mult, 4),
                **val_m,
                **tel_m,
                **v_diag,
            }
            history.append(rec)
            print(
                f"[{arm_id}] step={cur_step} (+{d_step}, {tok_s:.0f} tok/s, lr_m={lr_mult:.4f}) "
                f"val_ce={val_m['val_ce']:.4f} (dCE={val_m['val_ce'] - val0['val_ce']:+.4f}) "
                f"top1={val_m['top1_acc']*100:.2f}% top5={val_m['top5_acc']*100:.2f}% "
                f"pre_mom_norm={v_diag['pre_mom_normal_frac']:.4f} "
                f"leak_N2T={v_diag['normal_to_tangent_leakage_ratio']:.4f} "
                f"cos(gtan,mtan)={v_diag['cos_gtan_mtan']:.4f}",
                flush=True,
            )

    return {
        "arm_id": arm_id,
        "momentum_mode": momentum_mode.value,
        "lr_restart_peak": lr_restart_peak,
        "start_step": start_step,
        "end_step": start_step + delta_steps,
        "initial_val_ce": val0["val_ce"],
        "final": history[-1],
        "delta_val_ce": float(history[-1]["val_ce"] - val0["val_ce"]),
        "history": history,
        "vector_diag_trace": vector_diag_trace,
    }


def run_checkpoint_resume_diagnostic(
    ckpt_path: Path,
    shard_paths: list[Path],
    val_tokens: torch.Tensor,
    device: torch.device,
    delta_steps: int = 586,
    out_json: Path = CACHE_DIR / "w49_200m_resume_momentum_diagnostic.json",
) -> dict[str, Any]:
    """Compare fresh Euclidean momentum vs. fresh Riemannian tangent-transported momentum from the `200M`-token checkpoint."""
    ckpt_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arms_spec: list[tuple[str, StiefelMomentumMode, float | None]] = [
        ("arm1_fresh_euclidean_same_lr", StiefelMomentumMode.EUCLIDEAN_AMBIENT, None),
        ("arm2_fresh_riemannian_tangent_same_lr", StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED, None),
        ("arm3_fresh_riemannian_tangent_warm_restart_lr050", StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED, 0.50),
    ]
    results: dict[str, Any] = {
        "checkpoint_path": str(ckpt_path),
        "start_step": int(ckpt_payload["step"]),
        "start_tokens": int(ckpt_payload["tokens"]),
        "delta_steps": int(delta_steps),
        "original_uninterrupted_reference": {
            "step_24414_val_ce": 4.5202754735946655,
            "step_24500_val_ce": 4.517397582530975,
            "step_25000_val_ce": 4.513251304626465,
            "delta_ce_24414_to_24500": -0.0028778910636901855,
            "delta_ce_24414_to_25000": -0.007024168968200684,
        },
        "arms": {},
    }
    for arm_id, mom_mode, lr_peak in arms_spec:
        res = run_single_resume_arm(
            arm_id=arm_id,
            ckpt_payload=ckpt_payload,
            shard_paths=shard_paths,
            val_tokens=val_tokens,
            device=device,
            momentum_mode=mom_mode,
            lr_restart_peak=lr_peak,
            delta_steps=delta_steps,
        )
        results["arms"][arm_id] = res
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2))
        torch.cuda.empty_cache()
    return results


def run_classical_training_trajectory(
    run_id: str,
    steps: int,
    eval_every: int,
    shard_paths: list[Path],
    val_tokens: torch.Tensor,
    device: torch.device,
    out_json: Path,
    frontend_mode: ClassicalQuantumFrontEndMode = ClassicalQuantumFrontEndMode.PURE_CLASSICAL,
) -> dict[str, Any]:
    """Execute a matched Classical or Hybrid Quantum-Front-End Classical trajectory on the identical token stream (`<= 8` args)."""
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(2026063)

    cfg = CompiledLeaderboardClassicalConfig(
        vocab_size=DEFAULT_STATMECH_VOCAB_SIZE,
        embed_dim=DEFAULT_STATMECH_EMBED_DIM,
        num_heads=DEFAULT_STATMECH_NUM_HEADS,
        num_layers=DEFAULT_STATMECH_NUM_LAYERS,
        max_seq_len=DEFAULT_SEQ_LEN,
        num_value_embeddings=0,
        use_flex_doc_swc=False,
        bfloat16_weights=(device.type == "cuda"),
    )
    model = HybridQuantumFrontEndClassicalGPT(cfg, frontend_mode=frontend_mode).to(device)
    if device.type == "cuda":
        model.enable_compiled_backbone(dynamic=False)
    opt_bundle = build_hybrid_quantum_classical_optimizers(model)
    stream = MultiShardTokenStream(
        shard_paths, batch_size=DEFAULT_BATCH_SIZE, seq_len=DEFAULT_SEQ_LEN, vocab_size=cfg.vocab_size
    )
    tokens_per_step = DEFAULT_BATCH_SIZE * DEFAULT_SEQ_LEN

    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    val0 = evaluate_validation_with_divergence(model, val_tokens, DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, device)
    with torch.no_grad():
        head_rms0 = model.lm_head.weight.float().pow(2).mean(dim=-1).sqrt()
        head_std0 = float(head_rms0.std().item())
    q_tel0 = model.compute_quantum_frontend_telemetry()
    step0_rec = {"step": 0, "tokens": 0, "elapsed_s": 0.0, **val0, "head_row_scale_std": head_std0, **q_tel0}
    history.append(step0_rec)
    print(
        f"[{run_id}] step=0 tok=0.00M val_ce={val0['val_ce']:.4f} top1={val0['top1_acc']*100:.2f}% "
        f"top5={val0['top5_acc']*100:.2f}% gap={val0['gap_top5_top1']*100:.2f}% frontend={frontend_mode.value}",
        flush=True,
    )

    ema_train_loss: float | None = None
    milestone_steps = {100, 250, 500, 1000, 2500, 6104, 9155, 12207, 15259, 18311}

    for step in range(1, steps + 1):
        idx, targets = stream.next_batch(device)
        lr_mult = cosine_lr_schedule(step - 1, steps)

        opt_bundle.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss = model(idx, targets=targets, return_logits=False)
        assert loss is not None
        loss.backward()
        opt_bundle.step(idx=idx, lr_mult=lr_mult)

        loss_v = float(loss.detach().item())
        ema_train_loss = loss_v if ema_train_loss is None else (0.95 * ema_train_loss + 0.05 * loss_v)

        should_eval = (step % eval_every == 0) or (step in milestone_steps) or (step == steps)
        if should_eval:
            elapsed = float(time.perf_counter() - t0)
            tot_tokens = step * tokens_per_step
            tok_s = float(tot_tokens / max(1e-6, elapsed))
            val_m = evaluate_validation_with_divergence(model, val_tokens, DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, device)
            with torch.no_grad():
                head_rms = model.lm_head.weight.float().pow(2).mean(dim=-1).sqrt()
                head_std = float(head_rms.std().item())
                head_mean = float(head_rms.mean().item())
            q_tel = model.compute_quantum_frontend_telemetry()
            rec = {
                "step": int(step),
                "tokens": int(tot_tokens),
                "tokens_m": round(tot_tokens / 1e6, 3),
                "train_ce_ema": float(ema_train_loss),
                "elapsed_s": round(elapsed, 2),
                "tok_per_s": round(tok_s, 1),
                "lr_mult": round(lr_mult, 4),
                **val_m,
                "head_row_scale_std": head_std,
                "head_row_rms_mean": head_mean,
                **q_tel,
            }
            history.append(rec)
            q_str = (
                f" q_pur={q_tel.get('q_l0_purity', 0.0):.3f} q_rank={q_tel.get('q_l0_eff_rank', 0.0):.2f}"
                if "q_l0_purity" in q_tel
                else ""
            )
            print(
                f"[{run_id}] step={step}/{steps} ({tot_tokens/1e6:.2f}M tok, {tok_s:.0f} tok/s) "
                f"train_ce={ema_train_loss:.4f} val_ce={val_m['val_ce']:.4f} top1={val_m['top1_acc']*100:.2f}% "
                f"top5={val_m['top5_acc']*100:.2f}% nll1={val_m['nll_rank1']:.3f} nll2_5={val_m['nll_rank2_5']:.3f} "
                f"nll6_50={val_m['nll_rank6_50']:.3f} head_std={head_std:.4f}{q_str}",
                flush=True,
            )
            result_payload = {
                "run_id": run_id,
                "config": {
                    "arch": "HybridQuantumFrontEndClassicalGPT",
                    "frontend_mode": frontend_mode.value,
                    "vocab_size": cfg.vocab_size,
                    "n_layer": cfg.num_layers,
                    "n_head": cfg.num_heads,
                    "n_embd": cfg.embed_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "num_value_embeddings": cfg.num_value_embeddings,
                    "use_flex_doc_swc": cfg.use_flex_doc_swc,
                    "active_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
                "steps": steps,
                "completed_steps": step,
                "tokens_processed": tot_tokens,
                "final": rec,
                "history": history,
            }
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(result_payload, indent=2))

    return {"run_id": run_id, "final": history[-1], "history": history}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `74_verify_plan064_modular_quantum_statmech_experiments.py`."""
    p = argparse.ArgumentParser(description="Wave 49 (EXPERIMENT_PLAN_064.md): Modular Quantum StatMech GPT Runner")
    p.add_argument(
        "--mode",
        type=str,
        default="train_single",
        choices=["train_single", "train_classical", "ablation_suite", "resume_diagnostic"],
    )
    p.add_argument("--run-id", type=str, default="w49_modular_v3")
    p.add_argument("--purification-mode", type=str, default=PurificationMode.QUADRATIC_MCWEENY.value)
    p.add_argument("--channel-mode", type=str, default=QuantumChannelMode.UNITARY_U16.value)
    p.add_argument("--entropy-gate-mode", type=str, default=EntropyGateMode.NONE.value)
    p.add_argument("--readout-mode", type=str, default=VocabularyReadoutMode.SOFTCAP_LINEAR_CE.value)
    p.add_argument("--embedding-mode", type=str, default=EmbeddingTrainMode.TRAINABLE_SPHERE_MANIFOLD.value)
    p.add_argument("--momentum-mode", type=str, default=StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED.value)
    p.add_argument("--classical-frontend-mode", type=str, default=ClassicalQuantumFrontEndMode.PURE_CLASSICAL.value)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-json", type=str, default=str(DEFAULT_JSON_OUT))
    p.add_argument("--ckpt-path", type=str, default="")
    p.add_argument("--subset", type=str, default="all", choices=["all", "gpu0_fast", "gpu1_fast"])
    return p


def main() -> None:
    """Entrypoint for Wave 49 (`EXPERIMENT_PLAN_064.md`) experiments."""
    args = build_arg_parser().parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    shard_paths, val_tokens = load_fineweb_shards_and_val()

    intent = ExperimentIntentSpec(
        run_key=str(args.run_id),
        wave_id="W49",
        plan_id="EXPERIMENT_PLAN_064",
        task_domain="fineweb_lm",
        hypothesis_intent=(
            "Wave 49 (EXPERIMENT_PLAN_064.md): Modular Quantum Statistical-Mechanics GPT benchmarking "
            "Rebuilt Wave 48 v3, Mean-Centered Isometric Born Trace Readout, CPTP Kraus Channel, "
            "Embedding Trainability, Readout Divergence, Purification Modes, and Detached Tsallis Gate."
        ),
        architecture_summary=(
            f"ModularQuantumStatMechGPT (L=12, d=768, H=6, d_h=16, R=4, "
            f"purify={args.purification_mode}, chan={args.channel_mode}, "
            f"gate={args.entropy_gate_mode}, readout={args.readout_mode}, emb={args.embedding_mode})"
        ),
        hardware_budget=torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        batch_size=DEFAULT_BATCH_SIZE,
        seq_len=DEFAULT_SEQ_LEN,
        total_steps=int(args.steps),
        target_tokens_M=round(DEFAULT_BATCH_SIZE * DEFAULT_SEQ_LEN * int(args.steps) / 1e6, 4),
    )
    logger = StandardizedExperimentLogger(intent)
    if args.mode == "ablation_suite":
        res = run_ablation_suite(
            shard_paths=shard_paths,
            val_tokens=val_tokens,
            device=dev,
            steps=int(args.steps),
            eval_every=int(args.eval_every),
            out_json=Path(args.out_json),
            subset=str(args.subset),
        )
        logger.finalize(extra_summary={"mode": "ablation_suite", "subset": args.subset, "completed_arms": len(res["arms"])})
    elif args.mode == "resume_diagnostic":
        ckpt_file = (
            Path(args.ckpt_path)
            if args.ckpt_path
            else (CACHE_DIR / "checkpoints" / "w49_v3_rebuilt_200m_step24414.pt")
        )
        res = run_checkpoint_resume_diagnostic(
            ckpt_path=ckpt_file,
            shard_paths=shard_paths,
            val_tokens=val_tokens,
            device=dev,
            delta_steps=int(args.steps),
            out_json=Path(args.out_json),
        )
        logger.finalize(extra_summary={"mode": "resume_diagnostic", "completed_arms": len(res["arms"])})
    elif args.mode == "train_classical":
        res = run_classical_training_trajectory(
            run_id=str(args.run_id),
            steps=int(args.steps),
            eval_every=int(args.eval_every),
            shard_paths=shard_paths,
            val_tokens=val_tokens,
            device=dev,
            out_json=Path(args.out_json),
            frontend_mode=ClassicalQuantumFrontEndMode(args.classical_frontend_mode),
        )
        final_m = res["final"]
        logger.log_eval(
            int(args.steps),
            {
                "tokens_M": final_m.get("tokens_m", intent.target_tokens_M),
                "compute_tflops": 0.65,
                "tau_progress": 1.0,
                "val_ce": final_m["val_ce"],
                "val_ppl": final_m["val_ppl"],
                "top1_acc": final_m["top1_acc"],
            },
        )
        logger.finalize(extra_summary=res)
    else:
        cfg = ModularQuantumStatMechConfig(
            vocab_size=DEFAULT_STATMECH_VOCAB_SIZE,
            n_layer=DEFAULT_STATMECH_NUM_LAYERS,
            n_head=DEFAULT_STATMECH_NUM_HEADS,
            n_embd=DEFAULT_STATMECH_EMBED_DIM,
            purification_mode=PurificationMode(args.purification_mode),
            channel_mode=QuantumChannelMode(args.channel_mode),
            entropy_gate_mode=EntropyGateMode(args.entropy_gate_mode),
            readout_mode=VocabularyReadoutMode(args.readout_mode),
        )
        ckpt = Path(args.ckpt_path) if args.ckpt_path else None
        spec = StatMechRunSpec(
            run_id=str(args.run_id),
            cfg=cfg,
            embedding_mode=EmbeddingTrainMode(args.embedding_mode),
            steps=int(args.steps),
            eval_every=int(args.eval_every),
            device=dev,
            out_json=Path(args.out_json),
            ckpt_path=ckpt,
        )
        res = run_statmech_training_trajectory(
            spec,
            shard_paths,
            val_tokens,
            momentum_mode=StiefelMomentumMode(args.momentum_mode),
        )
        final_m = res["final"]
        logger.log_eval(
            int(args.steps),
            {
                "tokens_M": final_m.get("tokens_m", intent.target_tokens_M),
                "compute_tflops": 0.45,
                "tau_progress": 1.0,
                "val_ce": final_m["val_ce"],
                "val_ppl": final_m["val_ppl"],
                "top1_acc": final_m["top1_acc"],
                "sharp_eff_rank": final_m.get("mean_eff_rank", 1.5),
                "kraus_stiefel_err": final_m.get("max_stiefel_err", 0.0),
            },
        )
        logger.finalize(extra_summary=res)


if __name__ == "__main__":
    main()
