"""Warmup-Stable-Decay (WSD) Schedules, Cooldown Branching, and Progressive Sequence Curricula."""

from __future__ import annotations

from dataclasses import dataclass
import math


def cosine_lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int = 150,
    min_lr_ratio: float = 0.15,
) -> float:
    """Compute linear warmup followed by cosine decay learning-rate multiplier in `[min_lr_ratio, 1.0]`."""
    if step <= 0:
        return 0.0
    if warmup_steps > 0 and step <= warmup_steps:
        return float(step) / float(warmup_steps)
    denom = max(1, total_steps - max(0, warmup_steps))
    prog = min(1.0, max(0.0, float(step - max(0, warmup_steps)) / float(denom)))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def wsd_lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int = 150,
    decay_fraction: float = 0.20,
    min_lr_ratio: float = 0.10,
) -> float:
    """Compute Warmup-Stable-Decay (WSD) learning-rate multiplier with radical `1 - sqrt(t / T_decay)` cooldown.

    Phases:
    1. **Warmup (`1 <= step <= warmup_steps`)**:
       Linear ramp `step / warmup_steps` from `0` to `1.0`.
    2. **Stable (`warmup_steps < step <= stable_end`)**:
       Constant multiplier `1.0` across the main multi-epoch training horizon (e.g., Epochs 1-8
       of a 10-epoch run when `decay_fraction=0.20`), enabling checkpoint branching at any epoch
       without premature cosine learning-rate slowdown.
    3. **Radical Square-Root Decay (`stable_end < step <= total_steps`)**:
       Sharp `1 - sqrt((step - stable_end) / decay_steps)` cooldown down to `min_lr_ratio`.
    """
    if step <= 0:
        return 0.0
    if warmup_steps > 0 and step <= warmup_steps:
        return float(step) / float(warmup_steps)

    decay_steps = max(1, int(round(total_steps * max(0.0, min(0.95, decay_fraction)))))
    stable_end = max(warmup_steps, total_steps - decay_steps)
    if step <= stable_end:
        return 1.0

    actual_decay = max(1, total_steps - stable_end)
    tau = min(1.0, max(0.0, float(step - stable_end) / float(actual_decay)))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - math.sqrt(tau)))


def branch_wsd_cooldown_multiplier(
    step_in_cooldown: int,
    cooldown_steps: int,
    min_lr_ratio: float = 0.05,
) -> float:
    """Compute radical `1 - sqrt(tau)` cooldown multiplier for a branched WSD checkpoint."""
    if step_in_cooldown <= 0:
        return 1.0
    tau = min(1.0, max(0.0, float(step_in_cooldown) / float(max(1, cooldown_steps))))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - math.sqrt(tau)))


@dataclass(frozen=True)
class ProgressiveSequenceCurriculum:
    """Progressive sequence-length curriculum (`T: 256 -> 512 -> 1024 -> 2048`) with invariant `B * T`.

    Trains early epochs at shorter sequence lengths (where local bigram/trigram statistics
    dominate and attention FLOPs are lower) and progressively expands the context window in
    later epochs while keeping `tokens_per_step = batch_size * seq_len` strictly constant.
    """

    tokens_per_step: int = 65536
    seq_lengths: tuple[int, ...] = (256, 512, 1024, 2048)

    def __post_init__(self) -> None:
        if len(self.seq_lengths) == 0:
            raise ValueError("seq_lengths must contain at least one sequence length.")
        for seq_len in self.seq_lengths:
            if seq_len <= 0 or (seq_len & (seq_len - 1)) != 0:
                raise ValueError(f"Each seq_len must be a positive power of 2, got {seq_len}.")
            if self.tokens_per_step % seq_len != 0:
                raise ValueError(
                    f"tokens_per_step ({self.tokens_per_step}) must be divisible by seq_len ({seq_len})."
                )

    def for_epoch(self, epoch_idx: int, num_epochs: int) -> tuple[int, int]:
        """Return `(batch_size, seq_len)` for 0-indexed `epoch_idx` out of `num_epochs`."""
        n_stages = len(self.seq_lengths)
        if num_epochs <= 1:
            seq_len = self.seq_lengths[-1]
        else:
            stage_idx = min(
                n_stages - 1,
                max(0, int(math.floor(epoch_idx * n_stages / float(num_epochs)))),
            )
            seq_len = self.seq_lengths[stage_idx]
        batch_size = self.tokens_per_step // seq_len
        return batch_size, seq_len

    def stage_schedule(self, num_epochs: int) -> list[dict[str, int]]:
        """Return the full per-epoch `(epoch, batch_size, seq_len, tokens_per_step)` curriculum."""
        schedule: list[dict[str, int]] = []
        for ep in range(max(1, num_epochs)):
            b_sz, t_sz = self.for_epoch(ep, max(1, num_epochs))
            schedule.append(
                {
                    "epoch": ep + 1,
                    "batch_size": b_sz,
                    "seq_len": t_sz,
                    "tokens_per_step": b_sz * t_sz,
                }
            )
        return schedule
