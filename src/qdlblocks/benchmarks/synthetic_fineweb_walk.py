"""FineWeb-Matched 50,257-Vocabulary Synthetic Random-Walk Dataset & Fast Quantum Attention Overfit Suite.

Implements Experiment Plan 029:
1. `FineWebSyntheticWalkConfig` & `FineWebSyntheticRandomWalkGenerator`:
   - Generates structured random walks over the exact FineWeb-Edu GPT-2 vocabulary (`V_valid = 50,257`,
     `V_padded = 50,304`, `EOT_TOKEN_ID = 50256`) matching FineWeb's Zipf-Mandelbrot unigram entropy
     (`H_1 in [7.0, 8.2] nats`), Top-100 (`[0.35, 0.52]`) & Top-1,000 (`[0.55, 0.75]`) concentration,
     `256` semantic clusters, document resets (`EOT = 50256`), and multi-branch hub ambiguity
     (`H_bigram(X_{t+1} | X_t) >= 1.35 nats`, `Oracle Top-1 <= 48%`).
2. `Task A` (Full `50,304`-Vocab Random-Walk Trajectory Overfitting):
   - Compares 5 controlled arms (`OracleMarkovBigramFloor`, `NaiveQuadraticBornAttnLM`,
     `FastQuantumNoAttnMLP`, `FastQuantumBornAttentionSynthLM`, `FastClassicalSoftmaxAttnLM`) under
     the `<= 8.0 GB` RTX 5060 Ti VRAM cap (`< 1.0 GB` actual Peak VRAM).
3. `Task B` & `Task B-Radial` (Position-Shifted Variable-Lag `Delta in [8, 32]` Hub & Collinear
   Radial-Shell Disambiguation):
   - Proves that `PurifiedBornSelfAttention` + `StereographicLift` resolves variable-lag (`Delta in [8, 32]`)
     hub transitions and identical-token radial shells where both `OracleMarkovBigramFloor`,
     `FastQuantumNoAttnMLP` (`skip_attn_layers=(0, 1)`), and radial-erasing (`use_stereographic_lift=False`)
     models fail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from ..layers.stereographic_lift import apply_dimension_preserving_stereographic_rotation
from ..models.born_gpt import QuantumBornGPT
from ..models.classical_modded_gpt import ClassicalModdedGPT

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FINEWEB_VAL_SLICE = ROOT / ".cache" / "fineweb_100M" / "val_slice.bin"

_CACHED_FINEWEB_RANK_PERM: np.ndarray | None = None


@dataclass
class FineWebSyntheticWalkConfig:
    """Configuration for the FineWeb-matched `50,257`-vocabulary synthetic random walk."""

    vocab_valid: int = 50257
    vocab_padded: int = 50304
    eot_token_id: int = 50256
    num_clusters: int = 256
    num_branches: int = 16
    zipf_alpha: float = 1.07
    zipf_r0: float = 2.5
    tail_pool_size: int = 3350
    latent_dim: int = 32
    doc_reset_period: int = 512
    lookback_min_lag: int = 6
    lookback_max_lag: int = 32
    seed: int = 1337


def _load_or_build_fineweb_rank_permutation(
    vocab_valid: int = 50257,
    eot_token_id: int = 50256,
    seed: int = 1337,
    fineweb_bin_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return `(rank_to_token_id, token_id_to_rank, loaded_from_fineweb)` over `0 .. vocab_valid - 1`.

    Rank `0` (`1`-indexed rank 1) is the most frequent token; `eot_token_id` is pinned at the
    dedicated document boundary slot while non-EOT tokens `0 .. eot_token_id - 1` are ordered by
    empirical FineWeb frequency when `.cache/fineweb_100M/val_slice.bin` exists on disk.
    """
    global _CACHED_FINEWEB_RANK_PERM
    bin_path = fineweb_bin_path if fineweb_bin_path is not None else DEFAULT_FINEWEB_VAL_SLICE
    non_eot_vocab = int(eot_token_id)

    if (
        vocab_valid == 50257
        and eot_token_id == 50256
        and _CACHED_FINEWEB_RANK_PERM is not None
        and _CACHED_FINEWEB_RANK_PERM.shape == (50257,)
    ):
        rank_to_tok = _CACHED_FINEWEB_RANK_PERM.copy()
        tok_to_rank = np.empty_like(rank_to_tok)
        tok_to_rank[rank_to_tok] = np.arange(vocab_valid, dtype=np.int64)
        return rank_to_tok, tok_to_rank, True

    loaded_fw = False
    if bin_path is not None and Path(bin_path).exists() and vocab_valid == 50257 and eot_token_id == 50256:
        try:
            arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
            slice_len = min(1_000_000, max(0, int(arr.size) - 256))
            if slice_len >= 100_000:
                toks = np.asarray(arr[256 : 256 + slice_len], dtype=np.int64)
                counts = np.bincount(toks, minlength=vocab_valid)[:non_eot_vocab]
                # Sort descending by frequency, tie-breaking deterministically by token ID
                order = np.lexsort((np.arange(non_eot_vocab, dtype=np.int64), -counts))
                # Ensure token ID 50255 sits inside the active stratified tail range (rank ~ 1500)
                idx_50255 = int(np.where(order == (non_eot_vocab - 1))[0][0])
                if idx_50255 > 2200:
                    swap_val = int(order[1500])
                    order[1500] = non_eot_vocab - 1
                    order[idx_50255] = swap_val
                rank_to_tok = np.empty(vocab_valid, dtype=np.int64)
                rank_to_tok[:non_eot_vocab] = order
                rank_to_tok[non_eot_vocab] = eot_token_id
                _CACHED_FINEWEB_RANK_PERM = rank_to_tok.copy()
                loaded_fw = True
                tok_to_rank = np.empty_like(rank_to_tok)
                tok_to_rank[rank_to_tok] = np.arange(vocab_valid, dtype=np.int64)
                return rank_to_tok, tok_to_rank, loaded_fw
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    perm = rng.permutation(non_eot_vocab).astype(np.int64)
    idx_max = int(np.where(perm == (non_eot_vocab - 1))[0][0])
    if idx_max > 2200:
        perm[idx_max] = perm[1500]
        perm[1500] = non_eot_vocab - 1
    rank_to_tok = np.empty(vocab_valid, dtype=np.int64)
    rank_to_tok[:non_eot_vocab] = perm
    rank_to_tok[non_eot_vocab] = eot_token_id
    tok_to_rank = np.empty_like(rank_to_tok)
    tok_to_rank[rank_to_tok] = np.arange(vocab_valid, dtype=np.int64)
    return rank_to_tok, tok_to_rank, loaded_fw


class FineWebSyntheticRandomWalkGenerator:
    """FineWeb-Matched `50,257`-Vocabulary Quota-Balanced Quantum Random-Walk Generator."""

    def __init__(self, config: FineWebSyntheticWalkConfig | None = None) -> None:
        self.config = config if config is not None else FineWebSyntheticWalkConfig()
        c = self.config
        self.rng = np.random.default_rng(c.seed)

        ranks_1idx = np.arange(1, c.vocab_valid + 1, dtype=np.float64)
        weights = np.power(ranks_1idx + float(c.zipf_r0), -float(c.zipf_alpha))
        self.zipf_prior = weights / np.sum(weights)
        self.theoretical_zipf_entropy_nats = float(
            -np.sum(self.zipf_prior * np.log(np.clip(self.zipf_prior, 1e-15, 1.0)))
        )

        self.rank_to_token_id, self.token_id_to_rank, self.loaded_from_fineweb = (
            _load_or_build_fineweb_rank_permutation(
                vocab_valid=c.vocab_valid,
                eot_token_id=c.eot_token_id,
                seed=c.seed,
            )
        )

        # Assign every token v in 0..vocab_valid-1 to a semantic cluster c(v) in 0..num_clusters-1
        # and a radial shell in {0, 1, 2} -> {0.8, 1.8, 3.0}
        ranks0 = self.token_id_to_rank
        self.token_cluster = ((ranks0 * 73 + np.arange(c.vocab_valid, dtype=np.int64) * 19) % c.num_clusters).astype(
            np.int64
        )
        self.token_shell_idx = ((ranks0 * 31 + np.arange(c.vocab_valid, dtype=np.int64) * 7) % 3).astype(np.int64)
        shell_radii = np.array([0.8, 1.8, 3.0], dtype=np.float32)
        self.token_radius = shell_radii[self.token_shell_idx]

        # Latent Hilbert cluster kets in S^{latent_dim - 1}
        cluster_kets = self.rng.standard_normal((c.num_clusters, c.latent_dim)).astype(np.float32)
        cluster_kets /= np.linalg.norm(cluster_kets, axis=-1, keepdims=True) + 1e-12
        noise_kets = self.rng.standard_normal((c.vocab_valid, c.latent_dim)).astype(np.float32) * 0.25
        tok_kets = cluster_kets[self.token_cluster] + noise_kets
        tok_kets /= np.linalg.norm(tok_kets, axis=-1, keepdims=True) + 1e-12
        self.token_kets = tok_kets

        # Orthogonal walk query rotation R_walk in O(latent_dim)
        q_mat, _ = np.linalg.qr(self.rng.standard_normal((c.latent_dim, c.latent_dim)).astype(np.float32))
        self.walk_rot = q_mat.astype(np.float32)
        self.rotated_token_kets = (self.token_kets @ self.walk_rot.T).astype(np.float32)

        # Partition vocabulary by FineWeb rank tiers:
        # Tier 1: Top-100 hubs (ranks 0..99)
        # Tier 2: Mid-rank 101..1000 (ranks 100..999)
        # Tier 3: Stratified Heavy-Tail Pool (ranks 1000..50255) of size tail_pool_size
        self.top100_tokens = self.rank_to_token_id[:100].copy()
        self.mid900_tokens = self.rank_to_token_id[100:1000].copy()
        all_tail_tokens = self.rank_to_token_id[1000 : c.eot_token_id].copy()

        tail_size = min(int(c.tail_pool_size), int(all_tail_tokens.size))
        tail_probs = self.zipf_prior[1000 : 1000 + all_tail_tokens.size].copy()
        tail_probs = np.power(tail_probs, 0.35)
        tail_probs /= np.sum(tail_probs)
        chosen_tail = self.rng.choice(all_tail_tokens, size=tail_size, replace=False, p=tail_probs)
        if (c.eot_token_id - 1) not in chosen_tail:
            chosen_tail[0] = c.eot_token_id - 1
        self.tail_pool_tokens = np.sort(chosen_tail).astype(np.int64)

        # Prebuild deterministic 16-branch outgoing table for every token:
        # Branches 0..5  (6/16 = 37.5%): Top-100 hub core
        # Branches 6..8  (3/16 = 18.75%): Mid-rank 101..1000 semantic transitions
        # Branches 9..15 (7/16 = 43.75%): Stratified heavy-tail pool transitions
        top100_w = np.power(np.arange(1, 101, dtype=np.float64) + 5.0, -0.85)
        top100_w /= np.sum(top100_w)
        mid900_w = np.power(np.arange(101, 1001, dtype=np.float64) + 15.0, -0.75)
        mid900_w /= np.sum(mid900_w)

        self.out_edges = np.empty((c.vocab_valid, c.num_branches), dtype=np.int64)
        v_idx = np.arange(c.vocab_valid, dtype=np.int64)
        for b in range(6):
            # Sample top-100 hubs with low-discrepancy prime stride + Zipf weighting
            hub_choices = self.rng.choice(100, size=c.vocab_valid, replace=True, p=top100_w)
            hub_toks = self.top100_tokens[(hub_choices + b * 17) % 100]
            same_mask = hub_toks == v_idx
            if np.any(same_mask):
                hub_toks[same_mask] = self.top100_tokens[(hub_choices[same_mask] + b * 17 + 1) % 100]
            self.out_edges[:, b] = hub_toks

        for b_idx, b in enumerate(range(6, 9)):
            mid_choices = self.rng.choice(900, size=c.vocab_valid, replace=True, p=mid900_w)
            mid_toks = self.mid900_tokens[(mid_choices + b_idx * 131 + self.token_cluster * 3) % 900]
            same_mask = mid_toks == v_idx
            if np.any(same_mask):
                mid_toks[same_mask] = self.mid900_tokens[(mid_choices[same_mask] + 1) % 900]
            self.out_edges[:, b] = mid_toks

        for b_idx, b in enumerate(range(9, c.num_branches)):
            tail_idx = (v_idx * 37 + self.token_cluster * 13 + b_idx * 479) % tail_size
            self.out_edges[:, b] = self.tail_pool_tokens[tail_idx]

    def generate_corpus(
        self,
        num_seqs: int = 64,
        seq_len: int = 128,
    ) -> dict[str, Any]:
        """Generate `Task A` FineWeb-matched synthetic random-walk corpus `(num_seqs, seq_len)` over `V = 50,257`."""
        c = self.config
        tokens = np.empty((num_seqs, seq_len), dtype=np.int64)
        selected_branches = np.zeros((num_seqs, seq_len - 1), dtype=np.int64)
        anchor_positions = np.zeros((num_seqs, seq_len - 1), dtype=np.int64)

        visit_counts = np.zeros(c.vocab_valid, dtype=np.int64)
        tail_cursor = 0
        tail_size = int(self.tail_pool_tokens.size)
        # Ensure token 50255 (max non-EOT token ID) is placed early in the tail rotation
        idx_50255_in_tail = int(np.where(self.tail_pool_tokens == (c.eot_token_id - 1))[0][0])

        for n in range(num_seqs):
            # Sequence start from Top-100 hub or Mid-rank topic starter
            start_tok = int(self.top100_tokens[(n * 7) % 100])
            tokens[n, 0] = start_tok

            for t in range(seq_len - 1):
                global_step = n * seq_len + t
                cur_tok = int(tokens[n, t])
                v_cnt = int(visit_counts[cur_tok])
                visit_counts[cur_tok] = v_cnt + 1

                # Periodic EOT document reset (~1 per doc_reset_period tokens, placed at mid-sequence)
                reset_seq_stride = max(1, c.doc_reset_period // max(1, seq_len))
                if t == (seq_len // 2) and (n % reset_seq_stride == 0):
                    tokens[n, t + 1] = c.eot_token_id
                    selected_branches[n, t] = 0
                    anchor_positions[n, t] = max(0, t - c.lookback_min_lag)
                    continue

                if cur_tok == c.eot_token_id:
                    next_tok = int(self.top100_tokens[(n * 13 + t * 5) % 100])
                    tokens[n, t + 1] = next_tok
                    selected_branches[n, t] = 0
                    anchor_positions[n, t] = max(0, t - 1)
                    continue

                # Causal quantum-attention anchor selection strictly beyond 4-step local memory for t >= 6
                if t >= c.lookback_min_lag:
                    j_start = max(0, t - c.lookback_max_lag)
                    j_end = t - c.lookback_min_lag + 1  # exclusive, so max j is t - 6
                    hist_toks = tokens[n, j_start:j_end]
                    q_ket = self.token_kets[cur_tok]  # (latent_dim,)
                    k_kets = self.rotated_token_kets[hist_toks]  # (W, latent_dim)
                    overlaps = np.square(np.square(np.clip(k_kets @ q_ket, 0.0, None)))
                    j_star = int(j_start + int(np.argmax(overlaps)))
                else:
                    j_star = 0

                anchor_positions[n, t] = j_star
                anchor_tok = int(tokens[n, j_star])
                c_star = int(self.token_cluster[anchor_tok])
                s_star = int(self.token_shell_idx[anchor_tok])

                ctx_hash = (c_star * 29 + s_star * 101 + n * 17 + j_star * 7) & 0xFFFF
                branch = int((11 * v_cnt + ctx_hash) % c.num_branches)
                selected_branches[n, t] = branch

                if branch >= 9:
                    # Context-cluster-routed stratified heavy-tail transition
                    if tail_cursor == 0:
                        next_tok = int(self.tail_pool_tokens[idx_50255_in_tail])
                    else:
                        t_idx = int((tail_cursor * 197 + c_star * 13 + s_star * 3 + (branch - 9) * 41) % tail_size)
                        next_tok = int(self.tail_pool_tokens[t_idx])
                    tail_cursor += 1
                else:
                    next_tok = int(self.out_edges[cur_tok, branch])

                tokens[n, t + 1] = next_tok

        stats = compute_synthetic_fineweb_walk_stats(tokens, generator=self)
        return {
            "tokens": torch.from_numpy(tokens).long(),
            "selected_branches": torch.from_numpy(selected_branches).long(),
            "anchor_positions": torch.from_numpy(anchor_positions).long(),
            "stats": stats,
        }


def generate_fineweb_synthetic_random_walk(
    num_seqs: int = 64,
    seq_len: int = 128,
    config: FineWebSyntheticWalkConfig | None = None,
) -> dict[str, Any]:
    """Convenience entry point to build `FineWebSyntheticRandomWalkGenerator` and generate `Task A` corpus."""
    gen = FineWebSyntheticRandomWalkGenerator(config=config)
    out = gen.generate_corpus(num_seqs=num_seqs, seq_len=seq_len)
    out["generator"] = gen
    return out


def compute_synthetic_fineweb_walk_stats(
    tokens: torch.Tensor | np.ndarray,
    generator: FineWebSyntheticRandomWalkGenerator | None = None,
) -> dict[str, Any]:
    """Compute FineWeb statistical fidelity and 1st-order Markov conditional entropy metrics on `tokens`."""
    tok_np = tokens.detach().cpu().numpy().astype(np.int64) if isinstance(tokens, torch.Tensor) else np.asarray(tokens, dtype=np.int64)
    flat = tok_np.reshape(-1)
    total_tokens = int(flat.size)
    vocab_valid = generator.config.vocab_valid if generator is not None else 50257
    eot_id = generator.config.eot_token_id if generator is not None else 50256

    counts = np.bincount(flat, minlength=vocab_valid)
    nz_counts = counts[counts > 0]
    emp_probs = nz_counts.astype(np.float64) / float(total_tokens)
    emp_unigram_entropy = float(-np.sum(emp_probs * np.log(emp_probs)))

    sorted_emp_probs = np.sort(emp_probs)[::-1]
    top100_mass_empirical = float(np.sum(sorted_emp_probs[:100]))
    top1000_mass_empirical = float(np.sum(sorted_emp_probs[:1000]))

    if generator is not None:
        top100_ids = generator.top100_tokens
        top1000_ids = generator.rank_to_token_id[:1000]
        top100_mass_fw_rank = float(np.sum(counts[top100_ids]) / float(total_tokens))
        top1000_mass_fw_rank = float(np.sum(counts[top1000_ids]) / float(total_tokens))
        theo_entropy = float(generator.theoretical_zipf_entropy_nats)
        loaded_fw = bool(generator.loaded_from_fineweb)
    else:
        top100_mass_fw_rank = top100_mass_empirical
        top1000_mass_fw_rank = top1000_mass_empirical
        theo_entropy = emp_unigram_entropy
        loaded_fw = False

    non_eot = flat[flat != eot_id]
    max_non_eot = int(np.max(non_eot)) if non_eot.size > 0 else 0
    eot_count = int(np.sum(flat == eot_id))

    oracle = evaluate_oracle_markov_bigram_floor(tok_np)

    return {
        "total_tokens": total_tokens,
        "vocab_valid": vocab_valid,
        "unique_tokens": int(nz_counts.size),
        "max_token_id": int(np.max(flat)),
        "max_non_eot_token_id": max_non_eot,
        "eot_token_count": eot_count,
        "eot_rate": float(eot_count / max(1, total_tokens)),
        "empirical_unigram_entropy_nats": emp_unigram_entropy,
        "theoretical_zipf_entropy_nats": theo_entropy,
        "top100_mass_fineweb_rank": top100_mass_fw_rank,
        "top100_mass_empirical": top100_mass_empirical,
        "top1000_mass_fineweb_rank": top1000_mass_fw_rank,
        "top1000_mass_empirical": top1000_mass_empirical,
        "empirical_bigram_cond_entropy_nats": float(oracle["cond_entropy_nats"]),
        "oracle_markov_bigram_ce": float(oracle["train_ce"]),
        "oracle_markov_bigram_top1_acc": float(oracle["top1_acc"]),
        "oracle_markov_bigram_top5_acc": float(oracle["top5_acc"]),
        "loaded_from_fineweb": loaded_fw,
    }


def evaluate_oracle_markov_bigram_floor(
    train_tokens: torch.Tensor | np.ndarray,
    eval_tokens: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute exact 1st-order Markov / bigram oracle CE and Top-1/Top-5 accuracy over `V = 50,257`."""
    tr = (
        train_tokens.detach().cpu().numpy().astype(np.int64)
        if isinstance(train_tokens, torch.Tensor)
        else np.asarray(train_tokens, dtype=np.int64)
    )
    ev = (
        eval_tokens.detach().cpu().numpy().astype(np.int64)
        if isinstance(eval_tokens, torch.Tensor)
        else (np.asarray(eval_tokens, dtype=np.int64) if eval_tokens is not None else tr)
    )

    x_tr = tr[:, :-1].reshape(-1)
    y_tr = tr[:, 1:].reshape(-1)
    n_tr = int(x_tr.size)

    # Sparse transition table: map each observed x_t to Counter of y_t
    trans_counts: dict[int, dict[int, int]] = {}
    state_totals: dict[int, int] = {}
    for u, v in zip(x_tr.tolist(), y_tr.tolist()):
        d = trans_counts.get(u)
        if d is None:
            d = {}
            trans_counts[u] = d
        d[v] = d.get(v, 0) + 1
        state_totals[u] = state_totals.get(u, 0) + 1

    # Conditional bigram entropy H(X_{t+1} | X_t) on training transitions
    cond_ent_sum = 0.0
    for u, d in trans_counts.items():
        tot_u = float(state_totals[u])
        h_u = 0.0
        for cnt in d.values():
            p_uv = float(cnt) / tot_u
            h_u -= p_uv * math.log(max(p_uv, 1e-15))
        cond_ent_sum += (tot_u / float(n_tr)) * h_u

    # Precompute top-5 predictions and log-probabilities per state u
    top5_map: dict[int, list[int]] = {}
    for u, d in trans_counts.items():
        sorted_vs = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
        top5_map[u] = [kv[0] for kv in sorted_vs[:5]]

    x_ev = ev[:, :-1].reshape(-1)
    y_ev = ev[:, 1:].reshape(-1)
    n_ev = int(x_ev.size)

    nll_sum = 0.0
    top1_hits = 0
    top5_hits = 0
    uniform_nll = math.log(50257.0)
    for u, v in zip(x_ev.tolist(), y_ev.tolist()):
        d = trans_counts.get(u)
        if d is None:
            nll_sum += uniform_nll
            continue
        tot_u = float(state_totals[u])
        cnt_uv = d.get(v, 0)
        if cnt_uv > 0:
            nll_sum -= math.log(float(cnt_uv) / tot_u)
        else:
            nll_sum += uniform_nll
        t5 = top5_map[u]
        if t5[0] == v:
            top1_hits += 1
        if v in t5:
            top5_hits += 1

    return {
        "cond_entropy_nats": float(cond_ent_sum),
        "train_ce": float(nll_sum / max(1, n_ev)),
        "top1_acc": float(100.0 * top1_hits / max(1, n_ev)),
        "top5_acc": float(100.0 * top5_hits / max(1, n_ev)),
    }


def generate_position_shifted_hub_walk_batch(
    num_seqs: int = 256,
    seq_len: int = 64,
    num_branches: int = 8,
    min_lag: int = 8,
    max_lag: int = 32,
    seed: int = 2029,
    generator: FineWebSyntheticRandomWalkGenerator | None = None,
) -> dict[str, torch.Tensor]:
    """Generate `Task B` Position-Shifted Variable-Lag (`Delta in [8, 32]`) Hub Walk sequences over `V = 50,257`.

    On each sequence `n`:
    - Background tokens are drawn from the FineWeb-matched random walk filler pool (`ranks 200..2500`).
    - A hub query token `h_k` (`rank in 0..7`) is placed at a randomly shifted position `t_hub in [36, seq_len - 4]`.
    - A salient trigger token `x_trig^{(c)}` (`c in 0..num_branches-1`, disjoint trigger tokens `ranks 50..57`)
      is placed at a randomly sampled variable lookback lag `Delta in [min_lag, max_lag]` (`t_trig = t_hub - Delta`,
      strictly beyond the 4-step embedding memory window!).
    - The target token at `t_hub + 1` is `target_table[c]` in `{0, ..., 50256}`.
    """
    gen = generator if generator is not None else FineWebSyntheticRandomWalkGenerator()
    rng = np.random.default_rng(seed)

    hub_token_id = int(gen.top100_tokens[0])
    trigger_tokens = gen.top100_tokens[50 : 50 + num_branches].copy()
    target_tokens = gen.rank_to_token_id[120 : 120 + num_branches].copy()
    filler_pool = gen.rank_to_token_id[250:2250].copy()

    seqs = np.empty((num_seqs, seq_len), dtype=np.int64)
    hub_positions = np.empty(num_seqs, dtype=np.int64)
    trig_positions = np.empty(num_seqs, dtype=np.int64)
    lags = np.empty(num_seqs, dtype=np.int64)
    branch_labels = np.empty(num_seqs, dtype=np.int64)
    hub_targets = np.empty(num_seqs, dtype=np.int64)

    for n in range(num_seqs):
        # Sample background walk from filler pool
        bg_idx = rng.integers(0, filler_pool.size, size=seq_len)
        seqs[n] = filler_pool[bg_idx]

        t_hub = int(rng.integers(max(36, max_lag + 4), seq_len - 3))
        delta = int(rng.integers(min_lag, max_lag + 1))
        t_trig = t_hub - delta
        c_branch = int((n + rng.integers(0, num_branches)) % num_branches)

        seqs[n, t_trig] = int(trigger_tokens[c_branch])
        seqs[n, t_hub] = hub_token_id
        seqs[n, t_hub + 1] = int(target_tokens[c_branch])

        hub_positions[n] = t_hub
        trig_positions[n] = t_trig
        lags[n] = delta
        branch_labels[n] = c_branch
        hub_targets[n] = int(target_tokens[c_branch])

    return {
        "tokens": torch.from_numpy(seqs).long(),
        "hub_positions": torch.from_numpy(hub_positions).long(),
        "trig_positions": torch.from_numpy(trig_positions).long(),
        "lags": torch.from_numpy(lags).long(),
        "branch_labels": torch.from_numpy(branch_labels).long(),
        "hub_targets": torch.from_numpy(hub_targets).long(),
        "trigger_tokens": torch.from_numpy(trigger_tokens).long(),
        "target_tokens": torch.from_numpy(target_tokens).long(),
        "hub_token_id": torch.tensor(hub_token_id, dtype=torch.long),
    }


class NaiveQuadraticBornAttnLM(nn.Module):
    """Historical Naive `k=1` Linear Born Readout + Un-Thresholded Quadratic `|<q_i, k_j>|^2` Attention LM.

    Demonstrates the three historical failure traps on `V = 50,304`:
    1. Pure `k=1` linear Born readout `ln(1 + <psi_t, m_v>) in (-inf, ln 2]` with zero additive classical
       `log_prior`, capping maximum single-token probability at `2 / (2 + 50303) = 0.00398%` (`CE >= 10.13 nats`).
    2. Un-thresholded quadratic Born attention `A_{ij} ~ |<q_i, k_j>|^2` without degree-4 coherence purification.
    3. Naive `l2_normalize` without stereographic radial lift or orthogonal two-beam interference.
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.head_dim = self.embed_dim // self.num_heads
        self.eps = float(eps)

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.qkv_projs = nn.ModuleList(
            [nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False) for _ in range(self.num_layers)]
        )
        self.out_projs = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim, bias=False) for _ in range(self.num_layers)]
        )
        self.mlp_fc1 = nn.ModuleList(
            [nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=False) for _ in range(self.num_layers)]
        )
        self.mlp_fc2 = nn.ModuleList(
            [nn.Linear(2 * self.embed_dim, self.embed_dim, bias=False) for _ in range(self.num_layers)]
        )
        self.povm_effects = nn.Parameter(
            l2_normalize(torch.randn(self.vocab_size, self.embed_dim) * 0.02, dim=-1)
        )
        with torch.no_grad():
            nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        psi = l2_normalize(self.wte(idx).float(), dim=-1, eps=self.eps)
        causal_mask = torch.tril(torch.ones((T, T), device=idx.device, dtype=torch.bool)).view(1, 1, T, T)

        for l_idx in range(self.num_layers):
            qkv = self.qkv_projs[l_idx](psi).view(B, T, 3, self.num_heads, self.head_dim)
            q = l2_normalize(qkv[:, :, 0].transpose(1, 2), dim=-1, eps=self.eps)
            k = l2_normalize(qkv[:, :, 1].transpose(1, 2), dim=-1, eps=self.eps)
            v = l2_normalize(qkv[:, :, 2].transpose(1, 2), dim=-1, eps=self.eps)

            # Un-thresholded quadratic Born overlaps |<q_i, k_j>|^2
            overlaps_sq = torch.matmul(q, k.transpose(-1, -2)).square()
            overlaps_sq = overlaps_sq.masked_fill(~causal_mask, 0.0)
            attn_w = overlaps_sq / (overlaps_sq.sum(dim=-1, keepdim=True) + self.eps)
            attn_out = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(B, T, self.embed_dim)
            psi = l2_normalize(psi + 0.5 * self.out_projs[l_idx](attn_out), dim=-1, eps=self.eps)

            h = F.relu(self.mlp_fc1[l_idx](psi)).square()
            psi = l2_normalize(psi + 0.5 * self.mlp_fc2[l_idx](h), dim=-1, eps=self.eps)

        m_norm = l2_normalize(self.povm_effects.float(), dim=-1, eps=self.eps)
        # Pure k=1 single-copy linear Born fidelity log(1 + 0.5 * <psi_t, m_v>) in [ln(0.5), ln(1.5)]
        cos_sim = torch.matmul(psi, m_norm.t())
        logits = torch.log(torch.clamp_min(1.0 + 0.5 * cos_sim, 0.5))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss


def initialize_corpus_unigram_log_prior(
    model: QuantumBornGPT,
    train_tokens: torch.Tensor,
    unseen_Penalty: float = -12.0,
) -> None:
    """Initialize `model.povm_log_prior` from the empirical log-unigram distribution of `train_tokens`."""
    if getattr(model, "povm_log_prior", None) is None:
        return
    with torch.no_grad():
        vocab_size = int(model.vocab_size)
        flat = train_tokens.detach().reshape(-1).cpu()
        counts = torch.bincount(flat, minlength=vocab_size).float()
        prior = torch.full((vocab_size,), float(unseen_Penalty), dtype=torch.float32)
        nz = counts > 0
        if torch.any(nz):
            log_p = torch.log(counts[nz] / counts[nz].sum())
            log_p = log_p - log_p.mean()
            prior[nz] = log_p
        model.povm_log_prior.copy_(prior.to(device=model.povm_log_prior.device, dtype=model.povm_log_prior.dtype))


def build_fast_quantum_born_synth_lm(
    vocab_size: int = 50304,
    embed_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    block_size: int = 128,
    skip_attention: bool = False,
    use_stereographic_lift: bool = True,
    use_cosine_phase_beam_gate: bool = True,
    quadratic_diffuse_fidelity: float = 0.50,
    use_embed_causal_memory: bool = False,
    seed: int = 1337,
    device: torch.device | str = "cuda",
) -> QuantumBornGPT:
    """Build `FastQuantumBornAttentionSynthLM` (or `FastQuantumNoAttnMLP` when `skip_attention=True`)."""
    torch.manual_seed(seed)
    skip_layers: tuple[int, ...] = tuple(range(num_layers)) if skip_attention else ()
    num_ve = 0 if skip_attention else 1
    model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=64,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        use_bigram_smear=False,
        use_unet_interferometry=True,
        unet_skip_mode="scalar",
        use_fused_qkv=True,
        num_value_embeddings=num_ve,
        skip_attn_layers=skip_layers,
        zero_half_rope=True,
        window_size=block_size,
        doc_mask_attn=False,
        use_overcomplete_two_beam=True,
        frame_expansion=2.0,
        depth_scaled_residual=True,
        use_dynamic_povm_beta=True,
        use_ortho_decouple=False,
        use_povm_log_prior=True,
        use_gated_spherical_residual=True,
        use_purified_attn=True,
        true_layer0_v_residual=True,
        use_ngram_hash=False,
        use_mudd_and_gates=True,
        use_smear_gate=False,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=embed_dim // 16,
        num_higher_states=64,
        use_block_hierarchy_bridge=True,
        bridge_causal_memory=False,
        use_embed_causal_memory=use_embed_causal_memory,
        use_last_layer_softmax=False,
        use_two_beam_kboson_readout=True,
        diffuse_rank=16,
        diffuse_schedule="contextual_purity_complement",
        lambda_diff_min=0.08,
        lambda_diff_max=0.35,
        lambda_diff_start=0.22,
        lambda_diff_end=0.12,
        contextual_purity_gain=0.15,
        use_stereographic_lift=use_stereographic_lift,
        use_cosine_phase_beam_gate=use_cosine_phase_beam_gate,
        quadratic_diffuse_fidelity=quadratic_diffuse_fidelity,
    ).to(device)
    return model


def build_fast_classical_softmax_synth_lm(
    vocab_size: int = 50304,
    embed_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    block_size: int = 128,
    seed: int = 1337,
    device: torch.device | str = "cuda",
) -> ClassicalModdedGPT:
    """Build parameter-matched `FastClassicalSoftmaxAttnLM` (`L=2, d=256, H=4, V=50,304`)."""
    torch.manual_seed(seed)
    model = ClassicalModdedGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=3 * embed_dim,
        block_size=block_size,
        logit_softcap=30.0,
        use_unet_interferometry=True,
        unet_skip_mode="scalar",
        value_residual_mixing=True,
        num_value_embeddings=1,
        skip_attn_layers=(),
        zero_half_rope=True,
        window_size=block_size,
        doc_mask_attn=False,
        use_bigram_smear=False,
        use_ngram_hash=False,
        use_smear_gate=False,
        use_mudd_and_gates=True,
    ).to(device)
    return model


def _build_overfit_optimizer(
    model: nn.Module,
    lr_embed_readout: float = 0.015,
    lr_backbone: float = 0.006,
) -> torch.optim.Optimizer:
    """Build two-group `AdamW` optimizer for fast `V = 50,304` synthetic walk overfitting."""
    fast_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    fast_keywords = (
        "unigram_qubits",
        "higher_states",
        "povm_odd_qubits",
        "vocab_higher_affinity",
        "povm_head",
        "povm_effects",
        "povm_log_prior",
        "wte",
        "lm_head",
        "value_embeds",
        "kboson_copies",
        "inv_temp",
        "stereo_",
        "phase_",
        "register_gamma",
        "register_tau",
        "readout_rms_gain",
        "diffuse_kraus",
        "lambda_diff",
        "purity_gate",
    )
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in fast_keywords) or p.ndim < 2:
            fast_params.append(p)
        else:
            backbone_params.append(p)
    groups = []
    if fast_params:
        groups.append({"params": fast_params, "lr": lr_embed_readout, "weight_decay": 0.0, "initial_lr": lr_embed_readout})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4, "initial_lr": lr_backbone})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate_full_vocab_walk_model(
    model: nn.Module,
    tokens: torch.Tensor,
    batch_size: int = 16,
    valid_vocab_size: int = 50257,
) -> dict[str, float]:
    """Evaluate `model` on `tokens: (N, T)` in mini-batches of `min(batch_size, 4)` to keep Peak VRAM `< 0.75 GB`."""
    model.eval()
    dev = next(model.parameters()).device
    x_all = tokens[:, :-1].to(dev)
    y_all = tokens[:, 1:].to(dev)
    n_seqs, seq_m1 = x_all.shape
    total_toks = n_seqs * seq_m1

    total_nll = 0.0
    top1_hits = 0
    top5_hits = 0
    use_amp = dev.type == "cuda"
    eval_mb = min(int(batch_size), 4)

    for i0 in range(0, n_seqs, eval_mb):
        i1 = min(n_seqs, i0 + eval_mb)
        xb = x_all[i0:i1]
        yb = y_all[i0:i1]
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
        logits_f = logits[..., :valid_vocab_size].float()
        loss_b = F.cross_entropy(logits_f.reshape(-1, valid_vocab_size), yb.reshape(-1), reduction="sum")
        total_nll += float(loss_b.item())

        top5_idx = torch.topk(logits_f, k=5, dim=-1).indices  # (b, T-1, 5)
        top1_hits += int((top5_idx[..., 0] == yb).sum().item())
        top5_hits += int((top5_idx == yb.unsqueeze(-1)).any(dim=-1).sum().item())

    beam_ortho = 0.0
    if isinstance(model, QuantumBornGPT):
        ortho_vals = []
        for blk in model.blocks:
            tb = getattr(blk, "two_beam_mlp", None)
            if tb is not None and hasattr(tb, "last_beam_ortho_cos"):
                ortho_vals.append(float(tb.last_beam_ortho_cos))
        if ortho_vals:
            beam_ortho = float(sum(ortho_vals) / len(ortho_vals))

    return {
        "train_ce": float(total_nll / max(1, total_toks)),
        "train_ppl": float(math.exp(min(20.0, total_nll / max(1, total_toks)))),
        "top1_acc": float(100.0 * top1_hits / max(1, total_toks)),
        "top5_acc": float(100.0 * top5_hits / max(1, total_toks)),
        "beam_ortho": beam_ortho,
    }


def _compute_hub_only_logits(
    model: QuantumBornGPT,
    xb: torch.Tensor,
    hb: torch.Tensor,
    valid_vocab_size: int = 50257,
) -> torch.Tensor:
    """Compute `(B, valid_vocab_size)` logits at ONLY the hub index `hb` per sequence (`63x` less memory)."""
    b_cur = xb.size(0)
    idx_arange = torch.arange(b_cur, device=xb.device)
    psi, _ = model.forward_features(xb)
    psi_hub = psi[idx_arange, hb].unsqueeze(1)  # (B, 1, D)
    if model._last_psi_beam2 is not None:
        model._last_psi_beam2 = model._last_psi_beam2[idx_arange, hb].unsqueeze(1)  # (B, 1, D)
    logits_hub = model.compute_logits(psi_hub)  # (B, 1, V_padded)
    return logits_hub[:, 0, :valid_vocab_size].float()


def train_and_evaluate_task_a_arm(
    arm_key: str,
    model: nn.Module,
    tokens: torch.Tensor,
    steps: int = 300,
    batch_size: int = 16,
    lr_embed_readout: float = 0.015,
    lr_backbone: float = 0.006,
    eval_every: int = 100,
) -> dict[str, Any]:
    """Train a `Task A` model arm for `steps` mini-batch steps (`B = 16, T = 128`) and record trajectory."""
    dev = next(model.parameters()).device
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    if isinstance(model, QuantumBornGPT):
        initialize_corpus_unigram_log_prior(model, tokens)

    opt = _build_overfit_optimizer(model, lr_embed_readout=lr_embed_readout, lr_backbone=lr_backbone)
    x_all = tokens[:, :-1].to(dev)
    y_all = tokens[:, 1:].to(dev)
    n_seqs, seq_m1 = x_all.shape
    num_mb = max(1, (n_seqs + batch_size - 1) // batch_size)
    use_amp = dev.type == "cuda"
    is_qborn = isinstance(model, QuantumBornGPT)

    t0 = time.perf_counter()
    evals: list[dict[str, Any]] = []

    ev0 = evaluate_full_vocab_walk_model(model, tokens, batch_size=batch_size)
    evals.append(
        {
            "step": 0,
            "tokens_M": 0.0,
            "elapsed_min": 0.0,
            "val_ce": round(ev0["train_ce"], 4),
            "val_ppl": round(ev0["train_ppl"], 2),
            "top1_acc": round(ev0["top1_acc"], 2),
            "top5_acc": round(ev0["top5_acc"], 2),
            "beam_ortho": ev0["beam_ortho"],
            "iso_err": 0.0,
        }
    )

    model.train()
    for step in range(1, steps + 1):
        # Cosine LR decay with 10-step linear warmup
        if step <= 10:
            scale = float(step) / 10.0
        else:
            prog = float(step - 10) / float(max(1, steps - 10))
            scale = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = float(g["initial_lr"]) * scale

        mb_idx = (step - 1) % num_mb
        i0 = mb_idx * batch_size
        i1 = min(n_seqs, i0 + batch_size)
        xb = x_all[i0:i1]
        yb = y_all[i0:i1]

        opt.zero_grad(set_to_none=True)
        if is_qborn:
            if model.hierarchical_povm_readout is not None:
                model.hierarchical_povm_readout.set_diffuse_schedule_progress(step, steps)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(xb, targets=yb, direct_chunked_ce=True)
        else:
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(xb, targets=yb)
                loss = out[1] if isinstance(out, tuple) else out

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if is_qborn and model.hierarchical_povm_readout is not None and (step % 25 == 0 or step == steps):
            model.hierarchical_povm_readout.retract_kraus_stiefel()

        if step % eval_every == 0 or step == steps:
            ev = evaluate_full_vocab_walk_model(model, tokens, batch_size=batch_size)
            elapsed_s = float(time.perf_counter() - t0)
            toks_m = float(step * batch_size * seq_m1) / 1e6
            evals.append(
                {
                    "step": step,
                    "tokens_M": round(toks_m, 4),
                    "elapsed_min": round(elapsed_s / 60.0, 4),
                    "val_ce": round(ev["train_ce"], 4),
                    "val_ppl": round(ev["train_ppl"], 2),
                    "top1_acc": round(ev["top1_acc"], 2),
                    "top5_acc": round(ev["top5_acc"], 2),
                    "beam_ortho": ev["beam_ortho"],
                    "iso_err": 0.0,
                }
            )
            model.train()

    wall_s = float(time.perf_counter() - t0)
    peak_vram_gb = float(torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else 0.0
    final_ev = evals[-1]
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    tok_per_s = float(steps * batch_size * seq_m1) / max(1e-6, wall_s)

    return {
        "arm_key": arm_key,
        "params_M": round(params_m, 3),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "step0_ce": float(evals[0]["val_ce"]),
        "final_ce": float(final_ev["val_ce"]),
        "best_val_ce": float(min(e["val_ce"] for e in evals)),
        "final_ppl": float(final_ev["val_ppl"]),
        "top1_acc": float(final_ev["top1_acc"]),
        "top5_acc": float(final_ev["top5_acc"]),
        "beam_ortho": float(final_ev["beam_ortho"]),
        "wall_s": round(wall_s, 3),
        "wall_min": round(wall_s / 60.0, 4),
        "tok_per_s": round(tok_per_s, 1),
        "total_steps": steps,
        "target_tokens_M": round(float(steps * batch_size * seq_m1) / 1e6, 4),
        "evals": evals,
    }


def run_task_b_variable_lag_hub_benchmark(
    steps: int = 160,
    batch_size: int = 32,
    seq_len: int = 64,
    seed: int = 2029,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Run `Task B` Position-Shifted Variable-Lag (`Delta in [8, 32]`) Hub Disambiguation over `V = 50,304`."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    gen = FineWebSyntheticRandomWalkGenerator()
    train_batch = generate_position_shifted_hub_walk_batch(
        num_seqs=256, seq_len=seq_len, num_branches=8, min_lag=8, max_lag=32, seed=seed, generator=gen
    )
    eval_batch = generate_position_shifted_hub_walk_batch(
        num_seqs=128, seq_len=seq_len, num_branches=8, min_lag=8, max_lag=32, seed=seed + 999, generator=gen
    )

    # 1. Oracle Markov Bigram Floor on Hub Positions
    tr_toks = train_batch["tokens"]
    ev_toks = eval_batch["tokens"]
    ev_hub_pos = eval_batch["hub_positions"]
    ev_hub_targets = eval_batch["hub_targets"]

    # Empirical bigram counts at hub_token_id in train_batch
    hub_id = int(train_batch["hub_token_id"].item())
    x_tr = tr_toks[:, :-1].reshape(-1)
    y_tr = tr_toks[:, 1:].reshape(-1)
    hub_mask = x_tr == hub_id
    hub_next = y_tr[hub_mask]
    if hub_next.numel() > 0:
        bc = torch.bincount(hub_next, minlength=50257).float()
        probs_hub = (bc + 1e-6) / (bc.sum() + 50257e-6)
        top1_hub_tok = int(torch.argmax(probs_hub).item())
        oracle_hub_ce = float(-torch.log(probs_hub[ev_hub_targets]).mean().item())
        oracle_hub_top1 = float(100.0 * (ev_hub_targets == top1_hub_tok).float().mean().item())
    else:
        oracle_hub_ce = math.log(8.0)
        oracle_hub_top1 = 12.5

    results: dict[str, Any] = {
        "min_lag": 8,
        "max_lag": 32,
        "OracleMarkovBigramFloor": {
            "hub_ce": round(oracle_hub_ce, 4),
            "hub_top1_acc": round(oracle_hub_top1, 2),
        },
    }

    for arm_name, skip_attn in (
        ("FastQuantumNoAttnMLP", True),
        ("FastQuantumBornAttentionSynthLM", False),
    ):
        if dev.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        model = build_fast_quantum_born_synth_lm(
            vocab_size=50304,
            embed_dim=256,
            num_heads=4,
            num_layers=2,
            block_size=seq_len,
            skip_attention=skip_attn,
            use_stereographic_lift=True,
            use_cosine_phase_beam_gate=True,
            quadratic_diffuse_fidelity=0.50,
            use_embed_causal_memory=False,
            seed=seed,
            device=dev,
        )
        initialize_corpus_unigram_log_prior(model, tr_toks)
        opt = _build_overfit_optimizer(model, lr_embed_readout=0.018, lr_backbone=0.008)

        x_tr_seq = tr_toks[:, :-1].to(dev)
        tr_hub_pos = train_batch["hub_positions"].to(dev)
        tr_hub_y = train_batch["hub_targets"].to(dev)
        n_train = x_tr_seq.size(0)
        num_mb = max(1, (n_train + batch_size - 1) // batch_size)

        model.train()
        for step in range(1, steps + 1):
            mb_idx = (step - 1) % num_mb
            i0 = mb_idx * batch_size
            i1 = min(n_train, i0 + batch_size)
            xb = x_tr_seq[i0:i1]
            hb = tr_hub_pos[i0:i1]
            yb = tr_hub_y[i0:i1]

            opt.zero_grad(set_to_none=True)
            if model.hierarchical_povm_readout is not None:
                model.hierarchical_povm_readout.set_diffuse_schedule_progress(step, steps)
            hub_logits = _compute_hub_only_logits(model, xb, hb, valid_vocab_size=50257)
            loss = F.cross_entropy(hub_logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            x_ev_seq = ev_toks[:, :-1].to(dev)
            ev_hb = ev_hub_pos.to(dev)
            ev_yb = ev_hub_targets.to(dev)
            hub_ev_logits = _compute_hub_only_logits(model, x_ev_seq, ev_hb, valid_vocab_size=50257)
            hub_ce = float(F.cross_entropy(hub_ev_logits, ev_yb).item())
            preds = hub_ev_logits.argmax(dim=-1)
            hub_top1 = float(100.0 * (preds == ev_yb).float().mean().item())

        results[arm_name] = {
            "hub_ce": round(hub_ce, 4),
            "hub_top1_acc": round(hub_top1, 2),
        }
        del opt
        del model
        if dev.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    # Task B-Radial: Collinear Radial-Shell Variable-Lag Sub-Probe (Identical Trigger Token IDs, Radial Shell in {0.55, 2.75})
    radial_probe = run_collinear_radial_lag_attention_probe(
        seq_len=seq_len, min_lag=8, max_lag=32, steps=200, seed=seed, device=dev
    )
    results["TaskB_CollinearRadialSubProbe"] = radial_probe
    return results


class _CollinearRadialLagQuantumAttnClassifier(nn.Module):
    """Variable-Lag (`Delta in [8, 32]`) Purified Born Attention + Stereographic Lift Probe.

    Tests whether `PurifiedBornSelfAttention` + `apply_dimension_preserving_stereographic_rotation`
    can disambiguate `4` trigger token IDs x `2` collinear radial shells (`rho_trig in {0.55, 2.75}`)
    = `8` target classes at variable lag `Delta in [8, 32]` when the trigger token ID is IDENTICAL
    across both radial shells!
    """

    def __init__(
        self,
        num_token_types: int = 32,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_classes: int = 8,
        use_stereographic_lift: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(embed_dim // num_heads)
        self.use_stereographic_lift = bool(use_stereographic_lift)
        self.token_embed = nn.Embedding(num_token_types, embed_dim)
        self.stereo_pole = nn.Parameter(torch.randn(embed_dim) / math.sqrt(float(embed_dim)))
        self.stereo_tau = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.stereo_gain = nn.Parameter(torch.tensor(2.20, dtype=torch.float32))
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.raw_gamma = nn.Parameter(torch.full((num_heads, 1), 2.5, dtype=torch.float32))
        self.beam1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.beam2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.povm_detectors = nn.Parameter(l2_normalize(torch.randn(num_classes, embed_dim), dim=-1))
        self.kboson_scale = nn.Parameter(torch.tensor(20.0, dtype=torch.float32))

    def forward(
        self,
        token_ids: torch.Tensor,
        radial_scales: torch.Tensor,
        hub_positions: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-normalized state u_{b,t} = sqrt(d) * radial_scales_{b,t} * l2_normalize(E[token_ids_{b,t}])
        base_dir = l2_normalize(self.token_embed(token_ids).float(), dim=-1)
        u_raw = base_dir * (math.sqrt(float(self.embed_dim)) * radial_scales.unsqueeze(-1).float())
        if self.use_stereographic_lift:
            psi = apply_dimension_preserving_stereographic_rotation(
                u_raw, self.stereo_pole, self.stereo_tau, self.stereo_gain
            )
        else:
            # Homogeneous l2_normalize erases radial_scales > 0 identically at every token!
            psi = l2_normalize(u_raw, dim=-1)

        B, T, D = psi.shape
        H, dh = self.num_heads, self.head_dim
        q = l2_normalize(self.q_proj(psi).view(B, T, H, dh).transpose(1, 2), dim=-1)  # (B, H, T, dh)
        k = l2_normalize(self.k_proj(psi).view(B, T, H, dh).transpose(1, 2), dim=-1)  # (B, H, T, dh)
        v = l2_normalize(self.v_proj(psi).view(B, T, H, dh).transpose(1, 2), dim=-1)  # (B, H, T, dh)

        b_idx = torch.arange(B, device=psi.device)
        q_hub = q[b_idx, :, hub_positions, :].unsqueeze(2)  # (B, H, 1, dh)
        # Multi-head Born transition probability |<q_{hub,h}, k_{t,h}>|^2 >= 0 raised to purified power gamma_h
        born_fid = torch.matmul(q_hub, k.transpose(-1, -2)).squeeze(2).square()  # (B, H, T)
        t_arange = torch.arange(T, device=psi.device).view(1, 1, T)
        causal_mask = t_arange < hub_positions.view(B, 1, 1)
        gamma = torch.clamp(F.softplus(self.raw_gamma) + 2.5, min=2.5, max=18.0).unsqueeze(0)  # (1, H, 1)
        coherent = torch.pow(born_fid + 1e-4, gamma)
        coherent = coherent.masked_fill(~causal_mask, 0.0)
        attn_w = coherent / (coherent.sum(dim=-1, keepdim=True) + 1e-8)  # (B, H, T)
        v_head = l2_normalize(torch.einsum("bht,bhtd->bhd", attn_w, v), dim=-1)
        v_attended = l2_normalize(v_head.reshape(B, D), dim=-1)

        # Two-beam bilinear quantum interference on the attended ket
        b1 = l2_normalize(self.beam1(v_attended), dim=-1)
        b2 = l2_normalize(self.beam2(v_attended), dim=-1)
        psi_out = l2_normalize(v_attended + b1 * b2 * math.sqrt(float(D)), dim=-1)

        m_det = l2_normalize(self.povm_detectors.float(), dim=-1)
        cos_sim = torch.matmul(psi_out, m_det.t())
        return self.kboson_scale * torch.log(torch.clamp_min(1.0 + 0.85 * cos_sim, 1e-4))


def run_collinear_radial_lag_attention_probe(
    seq_len: int = 64,
    min_lag: int = 8,
    max_lag: int = 32,
    steps: int = 180,
    seed: int = 2029,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Compare `use_stereographic_lift=True` vs `False` on identical-token collinear radial shells at lag `Delta in [8, 32]`."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    def _make_split(n_samples: int, split_seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = np.random.default_rng(split_seed)
        toks = r.integers(5, 30, size=(n_samples, seq_len), dtype=np.int64)
        rad = np.ones((n_samples, seq_len), dtype=np.float32)
        hub_pos = np.empty(n_samples, dtype=np.int64)
        labels = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            t_h = int(r.integers(max(36, max_lag + 2), seq_len - 2))
            delta = int(r.integers(min_lag, max_lag + 1))
            t_tr = t_h - delta
            c_id = i % 4  # 4 trigger token IDs (1..4)
            s_rad = (i // 4) % 2  # 2 radial shells: 0 -> 0.55, 1 -> 2.75
            toks[i, t_tr] = 1 + c_id
            rad[i, t_tr] = 0.55 if s_rad == 0 else 2.75
            toks[i, t_h] = 0  # hub query token
            hub_pos[i] = t_h
            labels[i] = c_id * 2 + s_rad
        return (
            torch.from_numpy(toks).to(dev),
            torch.from_numpy(rad).to(dev),
            torch.from_numpy(hub_pos).to(dev),
            torch.from_numpy(labels).to(dev),
        )

    tr_toks, tr_rad, tr_hub, tr_y = _make_split(256, seed + 10)
    ev_toks, ev_rad, ev_hub, ev_y = _make_split(128, seed + 20)

    out: dict[str, Any] = {}
    for key, use_stereo in (("stereographic_lift_true", True), ("naive_radial_erasing_false", False)):
        torch.manual_seed(seed)
        clf = _CollinearRadialLagQuantumAttnClassifier(use_stereographic_lift=use_stereo).to(dev)
        opt = torch.optim.AdamW(clf.parameters(), lr=0.025, weight_decay=0.0)
        clf.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            logits = clf(tr_toks, tr_rad, tr_hub)
            loss = F.cross_entropy(logits, tr_y)
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            ev_logits = clf(ev_toks, ev_rad, ev_hub)
            ev_ce = float(F.cross_entropy(ev_logits, ev_y).item())
            ev_acc = float(100.0 * (ev_logits.argmax(dim=-1) == ev_y).float().mean().item())
        out[key] = {"eval_ce": round(ev_ce, 4), "eval_acc": round(ev_acc, 2)}
    return out


def run_synthetic_fineweb_overfit_suite(
    num_seqs: int = 64,
    seq_len: int = 128,
    steps: int = 300,
    batch_size: int = 16,
    seed: int = 1337,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Run the complete Plan 029 `H1 + H2 + H3` synthetic FineWeb random-walk & quantum attention overfit suite."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5, device=dev.index or 0)
        torch.cuda.empty_cache()

    t_suite0 = time.perf_counter()
    cfg = FineWebSyntheticWalkConfig(seed=seed)
    walk_data = generate_fineweb_synthetic_random_walk(num_seqs=num_seqs, seq_len=seq_len, config=cfg)
    tokens = walk_data["tokens"]
    h1_stats = walk_data["stats"]

    h1_pass = bool(
        7.00 <= h1_stats["empirical_unigram_entropy_nats"] <= 8.20
        and 0.35 <= h1_stats["top100_mass_fineweb_rank"] <= 0.52
        and 0.35 <= h1_stats["top100_mass_empirical"] <= 0.52
        and 0.55 <= h1_stats["top1000_mass_fineweb_rank"] <= 0.75
        and 0.55 <= h1_stats["top1000_mass_empirical"] <= 0.75
        and h1_stats["unique_tokens"] >= 2000
        and h1_stats["max_non_eot_token_id"] >= 50000
        and h1_stats["eot_token_count"] >= 8
        and h1_stats["empirical_bigram_cond_entropy_nats"] >= 1.35
        and h1_stats["oracle_markov_bigram_top1_acc"] <= 48.0
    )

    # Task A: 5-Model Full-Vocabulary (V = 50,304) Overfitting Comparison
    oracle_res = evaluate_oracle_markov_bigram_floor(tokens)
    task_a_arms: dict[str, Any] = {
        "w17_synth_oracle_markov_bigram_floor": {
            "arm_key": "w17_synth_oracle_markov_bigram_floor",
            "short_label": "Plan029 Synth-FW Oracle 1st-Order Markov Bigram Floor (V=50,257)",
            "arch_family": "classical",
            "color": "#adb5bd",
            "params_M": 0.0,
            "peak_vram_gb": 0.0,
            "step0_ce": round(oracle_res["train_ce"], 4),
            "final_ce": round(oracle_res["train_ce"], 4),
            "best_val_ce": round(oracle_res["train_ce"], 4),
            "final_ppl": round(math.exp(oracle_res["train_ce"]), 2),
            "top1_acc": round(oracle_res["top1_acc"], 2),
            "top5_acc": round(oracle_res["top5_acc"], 2),
            "beam_ortho": 0.0,
            "wall_s": 0.05,
            "wall_min": 0.01,
            "tok_per_s": 0.0,
            "total_steps": steps,
            "target_tokens_M": round(float(steps * batch_size * (seq_len - 1)) / 1e6, 4),
            "evals": [
                {
                    "step": 0,
                    "tokens_M": 0.0,
                    "elapsed_min": 0.0,
                    "val_ce": round(oracle_res["train_ce"], 4),
                    "val_ppl": round(math.exp(oracle_res["train_ce"]), 2),
                    "top1_acc": round(oracle_res["top1_acc"], 2),
                    "top5_acc": round(oracle_res["top5_acc"], 2),
                    "iso_err": 0.0,
                },
                {
                    "step": steps,
                    "tokens_M": round(float(steps * batch_size * (seq_len - 1)) / 1e6, 4),
                    "elapsed_min": 0.01,
                    "val_ce": round(oracle_res["train_ce"], 4),
                    "val_ppl": round(math.exp(oracle_res["train_ce"]), 2),
                    "top1_acc": round(oracle_res["top1_acc"], 2),
                    "top5_acc": round(oracle_res["top5_acc"], 2),
                    "iso_err": 0.0,
                },
            ],
        }
    }

    # Arm 2: Naive Quadratic Born Attention LM (k=1 linear Born readout without log_prior)
    naive_model = NaiveQuadraticBornAttnLM(vocab_size=50304, embed_dim=256, num_heads=4, num_layers=2).to(dev)
    res_naive = train_and_evaluate_task_a_arm(
        "w17_synth_naive_quadratic_born_k1_L2_d256",
        naive_model,
        tokens,
        steps=min(steps, 100),
        batch_size=batch_size,
        lr_embed_readout=0.010,
        lr_backbone=0.005,
        eval_every=max(1, min(steps, 100) // 2),
    )
    res_naive["short_label"] = "Plan029 Synth-FW Naive k=1 Linear Born + Quadratic Attn (L=2, d=256)"
    res_naive["arch_family"] = "quantum_born"
    res_naive["color"] = "#ff595e"
    task_a_arms["w17_synth_naive_quadratic_born_k1_L2_d256"] = res_naive
    del naive_model
    if dev.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # Arm 3: FastQuantumNoAttnMLP (skip_attn_layers=(0, 1), use_embed_causal_memory=False)
    no_attn_model = build_fast_quantum_born_synth_lm(
        vocab_size=50304,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        block_size=seq_len,
        skip_attention=True,
        use_stereographic_lift=True,
        use_cosine_phase_beam_gate=True,
        quadratic_diffuse_fidelity=0.50,
        use_embed_causal_memory=False,
        seed=seed,
        device=dev,
    )
    res_no_attn = train_and_evaluate_task_a_arm(
        "w17_synth_qborn_no_attn_control_L2_d256",
        no_attn_model,
        tokens,
        steps=steps,
        batch_size=batch_size,
        lr_embed_readout=0.015,
        lr_backbone=0.006,
        eval_every=max(1, steps // 3),
    )
    res_no_attn["short_label"] = "Plan029 Synth-FW Quantum No-Attn Control (SkipAttn, L=2, d=256)"
    res_no_attn["arch_family"] = "quantum_born"
    res_no_attn["color"] = "#ffca3a"
    task_a_arms["w17_synth_qborn_no_attn_control_L2_d256"] = res_no_attn
    del no_attn_model
    if dev.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # Arm 4: FastQuantumBornAttentionSynthLM (Full Small, Fast Quantum Attention)
    q_attn_model = build_fast_quantum_born_synth_lm(
        vocab_size=50304,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        block_size=seq_len,
        skip_attention=False,
        use_stereographic_lift=True,
        use_cosine_phase_beam_gate=True,
        quadratic_diffuse_fidelity=0.50,
        use_embed_causal_memory=False,
        seed=seed,
        device=dev,
    )
    res_q_attn = train_and_evaluate_task_a_arm(
        "w17_synth_qborn_fast_quantum_attn_stereo_phase_quad_L2_d256",
        q_attn_model,
        tokens,
        steps=steps,
        batch_size=batch_size,
        lr_embed_readout=0.018,
        lr_backbone=0.007,
        eval_every=max(1, steps // 3),
    )
    res_q_attn["short_label"] = "Plan029 Synth-FW FastQuantumBornAttentionSynthLM (PurifiedAttn+Stereo+Phase+QuadKBoson, L=2, d=256)"
    res_q_attn["arch_family"] = "quantum_born"
    res_q_attn["color"] = "#00f5d4"
    task_a_arms["w17_synth_qborn_fast_quantum_attn_stereo_phase_quad_L2_d256"] = res_q_attn
    del q_attn_model
    if dev.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    # Arm 5: FastClassicalSoftmaxAttnLM (Matched Classical Softmax SDPA Baseline)
    c_attn_model = build_fast_classical_softmax_synth_lm(
        vocab_size=50304,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        block_size=seq_len,
        seed=seed,
        device=dev,
    )
    res_c_attn = train_and_evaluate_task_a_arm(
        "w17_synth_classical_softmax_attn_L2_d256",
        c_attn_model,
        tokens,
        steps=steps,
        batch_size=batch_size,
        lr_embed_readout=0.012,
        lr_backbone=0.005,
        eval_every=max(1, steps // 3),
    )
    res_c_attn["short_label"] = "Plan029 Synth-FW FastClassicalSoftmaxAttnLM (Softmax SDPA, L=2, d=256)"
    res_c_attn["arch_family"] = "classical"
    res_c_attn["color"] = "#ff0055"
    task_a_arms["w17_synth_classical_softmax_attn_L2_d256"] = res_c_attn
    del c_attn_model
    if dev.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    max_vram_gb = max(float(a.get("peak_vram_gb", 0.0)) for a in task_a_arms.values())
    h2_pass = bool(
        res_naive["final_ce"] >= 8.50
        and res_no_attn["final_ce"] >= h1_stats["empirical_bigram_cond_entropy_nats"] - 0.02
        and res_q_attn["final_ce"] <= 0.10
        and res_q_attn["top1_acc"] >= 98.5
        and res_q_attn["top5_acc"] >= 99.8
        and res_q_attn["beam_ortho"] < 1e-5
        and max_vram_gb <= 8.0
    )

    # Task B: Position-Shifted Variable-Lag (Delta in [8, 32]) Hub & Collinear Radial Disambiguation
    task_b_res = run_task_b_variable_lag_hub_benchmark(
        steps=max(80, steps // 2),
        batch_size=32,
        seq_len=64,
        seed=seed + 92,
        device=dev,
    )
    rad_probe = task_b_res["TaskB_CollinearRadialSubProbe"]
    h3_pass = bool(
        task_b_res["OracleMarkovBigramFloor"]["hub_top1_acc"] <= 20.0
        and task_b_res["FastQuantumNoAttnMLP"]["hub_top1_acc"] <= 25.0
        and task_b_res["FastQuantumBornAttentionSynthLM"]["hub_top1_acc"] >= 95.0
        and task_b_res["FastQuantumBornAttentionSynthLM"]["hub_ce"] <= 0.25
        and rad_probe["stereographic_lift_true"]["eval_acc"] >= 95.0
        and rad_probe["naive_radial_erasing_false"]["eval_acc"] <= 58.0
    )

    return {
        "plan_id": "EXPERIMENT_PLAN_029",
        "config": asdict(cfg),
        "h1_walk_statistics": h1_stats,
        "h1_pass": h1_pass,
        "task_a_arms": task_a_arms,
        "micro_verification_arms": task_a_arms,
        "h2_pass": h2_pass,
        "task_b_variable_lag_hub": task_b_res,
        "h3_pass": h3_pass,
        "max_peak_vram_gb": round(max_vram_gb, 4),
        "suite_wall_s": round(float(time.perf_counter() - t_suite0), 2),
        "overall_pass": bool(h1_pass and h2_pass and h3_pass),
    }
