"""Multi-Shard FineWeb10B Streaming Loader & Memorization Diagnostics (`qdlblocks.benchmarks.fineweb_stream`).

Implements Section 3.1 of EXPERIMENT_PLAN_011.md:
- `FINEWEB10B_BASE_URL`, `FINEWEB_SHARD_HEADER_BYTES`, `FINEWEB_TRAIN_SHARD_BYTES`, `FINEWEB_VAL_SHARD_BYTES`
- `generate_contextual_zipf_shard`: deterministic contextual Markov-Zipf shard generator with a
  shared ground-truth transition law (`base_seed`) and distinct per-shard realizations (`shard_idx`).
- `FineWebMultiShardStream`: zero-stall asynchronous multi-shard streamer with legacy Wave 1-3
  `train_full_100M.bin` fallback for Shard 1.
- `compute_memorization_gap_metrics`: generalization gap and pass-boundary drop diagnostics.
- `compute_empirical_bigram_coverage`: unigram and bigram coverage across token shards.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import time
import urllib.request

import numpy as np

FINEWEB10B_BASE_URL = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main"
FINEWEB_SHARD_HEADER_BYTES = 1024  # 256 int32 header (magic, version, n_tokens, ...)
FINEWEB_TRAIN_SHARD_BYTES = 200_001_024  # 100,000,000 uint16 tokens + 1024B header
FINEWEB_VAL_TOKENS = 10_485_760  # Official KellerJordan/modded-nanogpt val_tokens (10 * 1024 * 1024)
FINEWEB_GPT2_VOCAB_SIZE = 50257  # Exact GPT-2 vocabulary size (0..50256, where 50256 is EOT/BOS)
FINEWEB_VAL_SHARD_BYTES = 21_103_616  # 1024B header + (10,485,760 + 65,536) * 2B uint16 tokens


def generate_contextual_zipf_shard(
    num_tokens: int,
    vocab_size: int = 50257,
    shard_idx: int = 1,
    base_seed: int = 42,
) -> np.ndarray:
    """Generate a deterministic `np.uint16` token shard from a shared contextual Markov-Zipf law.

    All shards (`shard_idx = 1..K` and validation `shard_idx = 0`) share the exact same
    underlying contextual transition table governed by `base_seed`, while each `shard_idx`
    samples an independent stochastic token realization via `base_seed + 10007 * shard_idx`.
    """
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")

    eff_vocab = min(2048, int(vocab_size))
    num_classes = min(16, eff_vocab)
    k_fav = min(16, max(1, eff_vocab // num_classes))

    rng_law = np.random.default_rng(int(base_seed))
    ranks = np.arange(1, eff_vocab + 1, dtype=np.float64)
    probs = 1.0 / (ranks ** 1.08)
    probs /= probs.sum()

    # Deterministic class-specific favored transition table shared across all shards
    favored_tokens = np.zeros((num_classes, k_fav), dtype=np.int64)
    for c in range(num_classes):
        c_next = (5 * c + 3) % num_classes
        class_candidates = np.arange(c_next, eff_vocab, num_classes, dtype=np.int64)
        cand_probs = probs[class_candidates].copy()
        cand_probs /= cand_probs.sum()
        replace_flag = len(class_candidates) < k_fav
        favored_tokens[c] = rng_law.choice(
            class_candidates, size=k_fav, replace=replace_flag, p=cand_probs
        )

    # Secondary trigram/context offset table shared across all shards
    context_shift = rng_law.integers(0, k_fav, size=(num_classes, num_classes), dtype=np.int64)

    rng_shard = np.random.default_rng(int(base_seed) + 10007 * int(shard_idx))

    stream_len = min(256, max(1, num_tokens))
    n_streams = (num_tokens + stream_len - 1) // stream_len
    total_gen = n_streams * stream_len

    bg_tokens = rng_shard.choice(eff_vocab, size=(n_streams, stream_len), replace=True, p=probs)
    u_branch = rng_shard.random(size=(n_streams, stream_len))
    rand_fav_k = rng_shard.integers(0, k_fav, size=(n_streams, stream_len))

    streams = np.empty((n_streams, stream_len), dtype=np.int64)
    streams[:, 0] = bg_tokens[:, 0]
    if stream_len > 1:
        c0 = streams[:, 0] % num_classes
        s0 = (streams[:, 0] // num_classes) % k_fav
        fav0 = favored_tokens[c0, s0]
        streams[:, 1] = np.where(u_branch[:, 1] < 0.65, fav0, bg_tokens[:, 1])

    for t in range(2, stream_len):
        prev1 = streams[:, t - 1]
        prev2 = streams[:, t - 2]
        c1 = prev1 % num_classes
        c2 = prev2 % num_classes
        s1 = (prev1 // num_classes) % k_fav
        det_k = (s1 + context_shift[c1, c2]) % k_fav
        k_choice = np.where(u_branch[:, t] < 0.52, det_k, rand_fav_k[:, t])
        fav_tok = favored_tokens[c1, k_choice]
        streams[:, t] = np.where(u_branch[:, t] < 0.65, fav_tok, bg_tokens[:, t])

    out = streams.reshape(-1)[:num_tokens].astype(np.uint16)
    return out


class FineWebMultiShardStream:
    """Multi-shard token streamer with zero-stall background prefetching and Wave 1-3 cache reuse."""

    def __init__(
        self,
        data_dir: Path | str,
        num_shards: int = 9,
        prefetch_async: bool = True,
        synthetic_tokens_per_shard: int = 0,
        vocab_size: int = 50257,
        seed: int = 42,
    ) -> None:
        if not (1 <= int(num_shards) <= 100):
            raise ValueError(f"num_shards must be in [1, 100], got {num_shards}")
        self.data_dir = Path(data_dir)
        self.num_shards = int(num_shards)
        self.prefetch_async = bool(prefetch_async)
        self.synthetic_tokens_per_shard = int(synthetic_tokens_per_shard)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1) if self.prefetch_async else None
        )
        self._prefetch_futures: dict[int, Future[np.ndarray]] = {}

    def shard_path(self, shard_idx: int) -> Path:
        """Resolve local path for shard `shard_idx`, reusing `train_full_100M.bin` for shard 1 if present."""
        if shard_idx < 1:
            raise ValueError(f"shard_idx must be >= 1, got {shard_idx}")
        canonical = self.data_dir / f"fineweb_train_{shard_idx:06d}.bin"
        if shard_idx == 1 and not canonical.exists():
            legacy = self.data_dir / "train_full_100M.bin"
            if legacy.exists() and legacy.stat().st_size >= 200_000_000:
                return legacy
            if legacy.exists() and legacy.stat().st_size > FINEWEB_SHARD_HEADER_BYTES:
                return legacy
        return canonical

    def ensure_val_shard(self) -> np.ndarray:
        """Load or download the held-out validation token shard (`shard_idx = 0`).

        Downloads at least `FINEWEB_VAL_TOKENS = 10,485,760` (`10 * 1024 * 1024`) tokens from
        `fineweb_val_000000.bin` to match `KellerJordan/modded-nanogpt` (`val_tokens = 10485760`),
        and automatically upgrades legacy `10,000,000`-byte (`4.99M`-token) `val_slice.bin` files.
        """
        if self.synthetic_tokens_per_shard > 0:
            val_len = max(16385, self.synthetic_tokens_per_shard // 2)
            return generate_contextual_zipf_shard(
                val_len,
                vocab_size=self.vocab_size,
                shard_idx=0,
                base_seed=self.seed,
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        val_path = self.data_dir / "val_slice.bin"
        needs_download = (
            (not val_path.exists())
            or (val_path.stat().st_size <= FINEWEB_SHARD_HEADER_BYTES)
            or (val_path.stat().st_size == 10_000_000)
        )
        if needs_download:
            url = f"{FINEWEB10B_BASE_URL}/fineweb_val_000000.bin"
            tmp_path = val_path.with_suffix(".bin.part")
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes=0-{FINEWEB_VAL_SHARD_BYTES - 1}", "User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp_path.replace(val_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                if not val_path.exists() or val_path.stat().st_size <= FINEWEB_SHARD_HEADER_BYTES:
                    raise
        return np.fromfile(val_path, dtype=np.uint16, offset=FINEWEB_SHARD_HEADER_BYTES)

    def _load_or_download_shard_sync(self, shard_idx: int) -> np.ndarray:
        """Synchronously generate, memory-map, or download training shard `shard_idx`."""
        if self.synthetic_tokens_per_shard > 0:
            return generate_contextual_zipf_shard(
                self.synthetic_tokens_per_shard,
                vocab_size=self.vocab_size,
                shard_idx=shard_idx,
                base_seed=self.seed,
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.shard_path(shard_idx)
        if not path.exists() or path.stat().st_size <= FINEWEB_SHARD_HEADER_BYTES:
            url = f"{FINEWEB10B_BASE_URL}/fineweb_train_{shard_idx:06d}.bin"
            tmp_path = path.with_suffix(".bin.part")
            req = urllib.request.Request(
                url,
                headers={"Range": f"bytes=0-{FINEWEB_TRAIN_SHARD_BYTES - 1}", "User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp_path.replace(path)
        return np.memmap(path, dtype=np.uint16, mode="r", offset=FINEWEB_SHARD_HEADER_BYTES)

    def prefetch_shard(self, shard_idx: int) -> None:
        """Submit background prefetch job for `shard_idx` if `prefetch_async` is enabled."""
        if (
            self.prefetch_async
            and self._executor is not None
            and 1 <= shard_idx <= self.num_shards
            and shard_idx not in self._prefetch_futures
        ):
            self._prefetch_futures[shard_idx] = self._executor.submit(
                self._load_or_download_shard_sync, shard_idx
            )

    def get_shard(self, shard_idx: int) -> tuple[np.ndarray, float]:
        """Return `(tokens_array, wait_ms)` for `shard_idx` and trigger prefetch for `shard_idx + 1`."""
        t0 = time.perf_counter()
        fut = self._prefetch_futures.pop(shard_idx, None)
        if fut is not None:
            tokens = fut.result()
        else:
            tokens = self._load_or_download_shard_sync(shard_idx)
        wait_ms = (time.perf_counter() - t0) * 1000.0
        if shard_idx + 1 <= self.num_shards:
            self.prefetch_shard(shard_idx + 1)
        return tokens, float(wait_ms)

    def iter_shards(self, num_passes: int = 1) -> Iterator[tuple[int, int, np.ndarray, float]]:
        """Yield `(pass_idx, shard_idx, tokens, wait_ms)` across `num_passes` passes of `1..num_shards`."""
        if num_passes < 1:
            raise ValueError(f"num_passes must be >= 1, got {num_passes}")
        for pass_idx in range(num_passes):
            if self.prefetch_async:
                self.prefetch_shard(1)
            for shard_idx in range(1, self.num_shards + 1):
                tokens, wait_ms = self.get_shard(shard_idx)
                yield pass_idx, shard_idx, tokens, wait_ms

    def close(self) -> None:
        """Shut down the background prefetch executor."""
        self._prefetch_futures.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def compute_memorization_gap_metrics(
    train_ce_at_evals: list[float],
    val_ce_at_evals: list[float],
    boundary_pre_train_ce: list[float] | None = None,
    boundary_post_train_ce: list[float] | None = None,
) -> dict[str, float]:
    """Compute generalization/memorization gap metrics and pass-boundary train-CE drops."""
    if not train_ce_at_evals or not val_ce_at_evals:
        raise ValueError("train_ce_at_evals and val_ce_at_evals must be non-empty")
    if len(train_ce_at_evals) != len(val_ce_at_evals):
        raise ValueError("train_ce_at_evals and val_ce_at_evals must have equal length")

    gaps = [float(v) - float(t) for t, v in zip(train_ce_at_evals, val_ce_at_evals)]
    mean_boundary_drop = 0.0
    if boundary_pre_train_ce and boundary_post_train_ce:
        drops = [
            float(pre) - float(post)
            for pre, post in zip(boundary_pre_train_ce, boundary_post_train_ce)
        ]
        if drops:
            mean_boundary_drop = float(np.mean(drops))

    return {
        "final_train_ce": float(train_ce_at_evals[-1]),
        "final_val_ce": float(val_ce_at_evals[-1]),
        "final_memorization_gap": float(gaps[-1]),
        "max_memorization_gap": float(max(gaps)),
        "mean_boundary_drop": float(mean_boundary_drop),
    }


def compute_empirical_bigram_coverage(
    shards: list[np.ndarray],
    max_vocab: int = 2048,
) -> dict[str, float]:
    """Compute distinct unigram and consecutive bigram counts across a list of token shards."""
    if max_vocab < 1:
        raise ValueError(f"max_vocab must be >= 1, got {max_vocab}")
    if not shards:
        return {
            "unique_unigrams": 0.0,
            "unique_bigrams": 0.0,
            "bigram_coverage_ratio": 0.0,
        }

    unigram_parts: list[np.ndarray] = []
    bigram_parts: list[np.ndarray] = []
    mv = int(max_vocab)
    for shard in shards:
        arr = np.asarray(shard, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            continue
        mod_arr = arr % mv
        unigram_parts.append(np.unique(mod_arr))
        if mod_arr.size >= 2:
            bi_keys = mod_arr[:-1] * mv + mod_arr[1:]
            bigram_parts.append(np.unique(bi_keys))

    n_uni = int(np.unique(np.concatenate(unigram_parts)).size) if unigram_parts else 0
    n_bi = int(np.unique(np.concatenate(bigram_parts)).size) if bigram_parts else 0
    return {
        "unique_unigrams": float(n_uni),
        "unique_bigrams": float(n_bi),
        "bigram_coverage_ratio": float(n_bi / max(1, mv * mv)),
    }


def find_bos_aligned_starts(
    tokens: np.ndarray | object,
    seq_len: int,
    batch_size: int,
    eot_token_id: int = 50256,
    rng: np.random.Generator | object | None = None,
) -> np.ndarray | object:
    """Find `batch_size` start positions `pos` in `[0, len(tokens) - seq_len - 1)` where `pos == 0` or `tokens[pos - 1] == eot_token_id`."""
    import torch

    is_tensor = isinstance(tokens, torch.Tensor)
    if is_tensor:
        t_tok: torch.Tensor = tokens.reshape(-1)
        n_total = int(t_tok.numel())
        max_start = max(1, n_total - int(seq_len) - 1)
        dev = t_tok.device
        if max_start <= 1:
            return torch.zeros((batch_size,), dtype=torch.int64, device=dev)
        eot_pos = (t_tok[: max_start - 1] == int(eot_token_id)).nonzero(as_tuple=False).reshape(-1) + 1
        zero_pos = torch.zeros((1,), dtype=torch.int64, device=dev)
        valid = torch.cat([zero_pos, eot_pos], dim=0)
        gen = rng if isinstance(rng, torch.Generator) else None
        if valid.numel() >= 1:
            pick = torch.randint(0, valid.numel(), (int(batch_size),), generator=gen, device=dev)
            return valid[pick]
        return torch.randint(0, max_start, (int(batch_size),), generator=gen, device=dev)

    arr = np.asarray(tokens).reshape(-1)
    n_total = int(arr.size)
    max_start = max(1, n_total - int(seq_len) - 1)
    np_rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(42)
    if max_start <= 1:
        return np.zeros((int(batch_size),), dtype=np.int64)
    eot_hits = np.flatnonzero(arr[: max_start - 1] == int(eot_token_id)) + 1
    valid_np = np.concatenate([np.array([0], dtype=np.int64), eot_hits.astype(np.int64)])
    if valid_np.size >= 1:
        replace_flag = valid_np.size < int(batch_size)
        return np_rng.choice(valid_np, size=int(batch_size), replace=replace_flag).astype(np.int64)
    return np_rng.integers(0, max_start, size=int(batch_size), dtype=np.int64)


def compute_tapered_mtp_weight(
    step: int,
    total_steps: int,
    mtp_weight_init: float,
    taper_end_frac: float = 0.65,
) -> float:
    """Linearly taper `mtp_weight` from `mtp_weight_init` at step 0 to `0.0` at `taper_end_frac * total_steps`."""
    s_taper = max(1, int(float(taper_end_frac) * int(total_steps)))
    if int(step) >= s_taper:
        return 0.0
    if int(step) <= 0:
        return float(mtp_weight_init)
    return float(mtp_weight_init) * (1.0 - float(step) / float(s_taper))


def compute_modded_nanogpt_val_steps(
    num_avail_tokens: int,
    batch_size: int = 64,
    seq_len: int = 512,
    val_tokens_budget: int = FINEWEB_VAL_TOKENS,
) -> int:
    """Compute exact `val_steps = val_tokens // (batch_size * seq_len)` matching KellerJordan/modded-nanogpt."""
    stride = max(1, int(batch_size) * int(seq_len))
    eff_tokens = min(int(val_tokens_budget), max(0, int(num_avail_tokens) - 1))
    if int(num_avail_tokens) == int(val_tokens_budget) and int(val_tokens_budget) >= stride:
        eff_tokens = int(val_tokens_budget)
    return max(1, eff_tokens // stride)


def iter_modded_nanogpt_val_batches(
    val_tokens: np.ndarray,
    batch_size: int = 64,
    seq_len: int = 512,
    val_tokens_budget: int = FINEWEB_VAL_TOKENS,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield `(b_idx, x_np, y_np)` sequentially from `pos = 0` (`align_to_bos=False`) matching KellerJordan/modded-nanogpt.

    Evaluates `val_steps = val_tokens_budget // (batch_size * seq_len)` contiguous batches across
    `fineweb_val_000000.bin` (`val_tokens = 10,485,760` = `10 * 1024 * 1024` tokens).
    """
    b_sz = max(1, int(batch_size))
    t_sz = max(1, int(seq_len))
    stride = b_sz * t_sz
    n_avail = int(len(val_tokens))
    val_steps = compute_modded_nanogpt_val_steps(
        n_avail, batch_size=b_sz, seq_len=t_sz, val_tokens_budget=val_tokens_budget
    )
    for b_idx in range(val_steps):
        base = b_idx * stride
        end = base + stride + 1
        if end <= n_avail:
            chunk = val_tokens[base:end].astype(np.int64)
        else:
            body = val_tokens[base : min(base + stride, n_avail)].astype(np.int64)
            pad_len = (stride + 1) - int(body.size)
            tail = np.full((pad_len,), int(val_tokens[0]) if n_avail > 0 else 50256, dtype=np.int64)
            chunk = np.concatenate([body, tail], axis=0)
        x_np = chunk[:-1].reshape(b_sz, t_sz)
        y_np = chunk[1:].reshape(b_sz, t_sz)
        yield b_idx, x_np, y_np


__all__ = [
    "FINEWEB10B_BASE_URL",
    "FINEWEB_SHARD_HEADER_BYTES",
    "FINEWEB_TRAIN_SHARD_BYTES",
    "FINEWEB_VAL_TOKENS",
    "FINEWEB_GPT2_VOCAB_SIZE",
    "FINEWEB_VAL_SHARD_BYTES",
    "generate_contextual_zipf_shard",
    "FineWebMultiShardStream",
    "compute_memorization_gap_metrics",
    "compute_empirical_bigram_coverage",
    "find_bos_aligned_starts",
    "compute_tapered_mtp_weight",
    "compute_modded_nanogpt_val_steps",
    "iter_modded_nanogpt_val_batches",
]

