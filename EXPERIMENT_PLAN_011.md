# Experiment Plan 011: Single-Pass 900M-Token FineWeb Multi-Shard Streaming vs. Multi-Epoch 100M Repetition

**Status:** DRAFT (Pending Blind Sub-Agent Review)
**Target Architecture:** `QuantumBornGPT` (`d = 1024`, $L = 6$, $H = 16$, Two-Beam `1536x2` / `2048x2` Interference, `D_seed = 384 / 512`, `OvercompleteSeedReadoutAdapter`)
**Primary Reference Candidate:** `w3_gpu_adapter384_2beam` (`D_seed = 384`, `1536x2` Two-Beam Bilinear MLP, Plan 007 `OvercompleteSeedReadoutAdapter`, `99.58M` parameters, `300,748 tok/s` on Blackwell G4)

---

## 1. Motivation & Empirical Observation from Wave 3

In Wave 3 (`EXPERIMENT_PLAN_008` and `EXPERIMENT_PLAN_009`), restoring the Two-Beam `1536x2` interference backbone alongside the Plan 007 `OvercompleteSeedReadoutAdapter` at $D_{\text{seed}} = 384$ (`w3_gpu_adapter384_2beam`) achieved **`4.1531` Val CE (`63.6` PPL, `30.98%` Top-1, `51.39%` Top-5)** in **`16.72 minutes` (`300,748 tok/s`)** on a single 96GB Blackwell G4 GPU—matching the `165.94M`-parameter $D_{\text{seed}} = 1024$ Wave-2 record (`4.1527` CE) within `0.0004` nats at $1.75\times$ higher throughput and $66.3\text{M}$ fewer parameters.

However, every experiment from Wave 1 through Wave 3 trained by **looping multiple epochs ($K \in \{3, 5\}$ passes) over a single 100M-token shard** (`fineweb_train_000001.bin`, `100,000,000` `uint16` tokens). Inspecting the step-by-step trajectory of `w3_gpu_adapter384_2beam` across its 3 passes (`300M` total tokens) exposes a pronounced **multi-epoch repetition / memorization drag**:

| Training Checkpoint | Pass # on Shard 1 | Cumulative Tokens | Rolling `Train CE` | Held-Out `Val CE` | Generalization Gap $\Delta_{\text{mem}} = \text{Val CE} - \text{Train CE}$ | Boundary Train-CE Drop $\Delta_{\text{boundary}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Step 762 (`Ep 0.50`) | Pass 1 | `49.94M` | `5.5968` | `4.8514` | `-0.7454` nats *(fresh tokens)* | — |
| Step 1525 (`Ep 1.00`) | Pass 1 | `99.94M` | `4.6934` | `4.5937` | `-0.0997` nats *(fresh tokens)* | — |
| Step 1650 (`Ep 1.08`) | **Pass 2** | `108.13M` | `4.5114` | — | — | **`-0.1820` nats** *(pass-2 repeat drop)* |
| Step 3050 (`Ep 2.00`) | **Pass 2** | `199.88M` | `4.3591` | `4.3266` | `-0.0325` nats | — |
| Step 3150 (`Ep 2.07`) | **Pass 3** | `206.44M` | `4.1885` | — | — | **`-0.1706` nats** *(pass-3 repeat drop)* |
| Step 3812 (`Ep 2.50`) | **Pass 3** | `249.82M` | `4.1734` | `4.2209` | **`+0.0475` nats (`Train < Val`!)** | — |
| Step 4575 (`Ep 3.00`) | **Pass 3** | `299.83M` | `4.1255` | `4.1531` | **`+0.0276` nats (`Train < Val`!)** | — |

By Pass 3 (`200M–300M` tokens), `Train CE` drops below held-out `Val CE`, and at each epoch boundary (`100M` and `200M` tokens) `Train CE` exhibits a discontinuous `-0.17` to `-0.18` nat step-down because the model recognizes exact 512-token contexts from Pass 1. In our earlier 5-epoch (`500M`-token) runs (`gpu_5ep`, `gpu_wsd_mtp`, `w2_gpu_adapter_wsd`), Passes 4 and 5 suffered even stronger diminishing returns, and Warmup-Stable-Decay (`WSD`) stalled after 100M tokens because maintaining peak learning rate during Passes 2–4 aggressively memorized Shard 1.

Meanwhile, the standard **NanoGPT Speedrun** (`kjj0/fineweb10B-gpt2`) streams **9 distinct 100M-token shards** (`fineweb_train_000001.bin` through `fineweb_train_000009.bin` = **`900,000,000` unique tokens**) in a **single pass (`1` epoch, zero token repetition)**.

---

## 2. Observable Hypotheses & Quantitative Predictions

### Hypothesis 1 (`H1` — Multi-Shard Unique-Token Streaming Eliminates Memorization Drag and Accelerates Validation Scaling)

* **Mathematical Mechanism**:
  Let $\mathcal{P}$ be a stationary language distribution over vocabulary $\mathcal{V}$ with Zipfian marginal frequencies $p(v) \propto v^{-\alpha}$ and Markov/contextual transition law $P(y \mid x_{1:t})$.
  - In the **Repeated Single-Shard Regime (`1xK`)**, a single finite sample $\mathcal{D}_1 \sim \mathcal{P}^{N_1}$ of $N_1$ tokens is traversed $K$ times ($N_{\text{total}} = K N_1$). Because the empirical conditional distribution $\hat{P}_{\mathcal{D}_1}(y \mid x_{1:t})$ has finite-sample variance $\propto \frac{1}{\text{count}_{\mathcal{D}_1}(x_{1:t})}$ and zero support on $n$-grams absent from $\mathcal{D}_1$, optimizing on $\mathcal{D}_1$ for $K \ge 3$ passes drives the Born readout amplitudes toward $\hat{P}_{\mathcal{D}_1}$ rather than $\P$. This manifests as:
    1. Large negative pass-boundary train-loss discontinuities:
       $$\Delta_{\text{boundary}}^{(k)} = \mathcal{L}_{\text{train}}(\text{step}_{k, \text{first}}) - \mathcal{L}_{\text{train}}(\text{step}_{k-1, \text{last}}) \ll 0,$$
    2. A positive **memorization gap** between held-out validation loss and training loss:
       $$\Delta_{\text{mem}}(t) = \mathcal{L}_{\text{val}}(t) - \mathcal{L}_{\text{train}}(t) > 0,$$
    3. Lower unique token/bigram coverage across the training trajectory.
  - In the **Multi-Shard Single-Pass Streaming Regime (`Kx1`)**, the model traverses $K$ independent shards $\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_K \stackrel{\text{i.i.d.}}{\sim} \mathcal{P}^{N_1}$ once each ($N_{\text{total}} = K N_1$ unique tokens). Every gradient step is an unbiased stochastic estimate of $\nabla \mathbb{E}_{\mathcal{P}}[\mathcal{L}]$, eliminating pass-boundary memorization drops and keeping $\mathcal{L}_{\text{train}}(t) \approx \mathcal{L}_{\text{val}}(t)$ throughout all $K N_1$ tokens.

* **Quantitative Pass/Fail Predictions (`H1`)**:
  In a controlled experiment (`examples/21_hypothesis_900m_multishard_streaming.py`) comparing a `QuantumBornGPT` (`vocab_size = 512`, `embed_dim = 64`, `seed_dim = 24`, `num_layers = 2`, `num_heads = 4`, `hidden_dim = 96`, `use_two_beam_mlp = True`, `use_overcomplete_seed_adapter = True`) trained for $N_{\text{total}} = 9 \times N_{\text{shard}}$ tokens under **Regime A (`1x9` Repeated Single Shard: $1$ shard of $N_{\text{shard}} = 16,384$ tokens looped for $9$ passes)** vs. **Regime B (`9x1` Multi-Shard Unique Stream: $9$ distinct shards of $N_{\text{shard}} = 16,384$ tokens each drawn from the same ground-truth contextual Markov-Zipf generator, $1$ pass each)** with identical initial weights, batch size ($B = 16, T = 64$), and cosine LR schedule ($144$ total steps):
  1. **Memorization Gap Reduction**: The final generalization gap $\Delta_{\text{mem}} = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$ in Regime B (`9x1` unique stream) is at least **`3.0x` smaller** than in Regime A (`1x9` repeated shard):
     $$\Delta_{\text{mem}}^{\text{9x1}} \le \frac{1}{3.0}\,\Delta_{\text{mem}}^{\text{1x9}} \quad \text{and} \quad \Delta_{\text{mem}}^{\text{1x9}} - \Delta_{\text{mem}}^{\text{9x1}} \ge 0.20\text{ nats}.$$
  2. **Validation Cross-Entropy Improvement**: Regime B (`9x1` unique stream) achieves a final held-out validation cross-entropy at least **`0.12` nats lower** than Regime A (`1x9` repeated shard):
     $$\mathcal{L}_{\text{val}}^{\text{9x1}} \le \mathcal{L}_{\text{val}}^{\text{1x9}} - 0.12\text{ nats}.$$
  3. **Empirical Context Coverage**: The number of unique token bigrams $(x_t, x_{t+1})$ encountered across the 9-shard stream is at least **`2.5x` greater** than in the single repeated shard.

---

### Hypothesis 2 (`H2` — WSD Schedule Recovery on Unique Shards & Zero-Stall Asynchronous Shard Prefetching)

* **Mathematical Mechanism**:
  1. **Why WSD Failed on Repeated Shards but Succeeds on Unique Shards**:
     Under Warmup-Stable-Decay (`WSD`, where learning rate remains at $1.0 \times \text{lr}_{\max}$ for $80\%$ of training steps before a $20\%$ $1 - \sqrt{\tau}$ cooldown), traversing a single repeated shard (`1x9`) at peak learning rate during passes $2 \dots 7$ causes severe overfitting to $\mathcal{D}_1$ before cooldown begins. Conversely, when streaming $9$ unique shards (`9x1`), the $80\%$ stable phase explores fresh data from shards $1 \dots 7$ without memorization, and the $20\%$ $1 - \sqrt{\tau}$ cooldown on shards $8 \dots 9$ anneals stochastic gradient noise on unseen tokens.
  2. **Zero-Stall Asynchronous Prefetching**:
     By overlapping background preparation/downloading of shard $s+1$ while the accelerator executes training steps on shard $s$ via `FineWebMultiShardStream`, host memory only holds the active shard plus at most one prefetched buffer, and shard-boundary transition latency is bounded to millisecond pointer swap overhead.

* **Quantitative Pass/Fail Predictions (`H2`)**:
  1. **WSD Schedule Recovery**: In the controlled 9-shard benchmark (`examples/21_hypothesis_900m_multishard_streaming.py`), training with `--schedule wsd` (`decay_fraction = 0.20`) on the `9x1` unique-shard stream reduces final validation cross-entropy by at least **`0.15` nats** compared to `--schedule wsd` on the `1x9` repeated shard ($\mathcal{L}_{\text{val}}^{\text{9x1, WSD}} \le \mathcal{L}_{\text{val}}^{\text{1x9, WSD}} - 0.15$), and achieves final validation CE within `0.03` nats of (or better than) `9x1` Cosine ($\mathcal{L}_{\text{val}}^{\text{9x1, WSD}} \le \mathcal{L}_{\text{val}}^{\text{9x1, Cosine}} + 0.03$).
  2. **Asynchronous Prefetching Verification**: `FineWebMultiShardStream` with `prefetch_async=True` overlaps background shard preparation with step execution such that shard transition wait time on pre-cached/prefetched shards is `< 25.0 ms` per shard boundary and all 9 shards yield bit-exact deterministic token arrays matching sequential loading.

---

## 3. Planned Code & Architectural Changes

### 3.1 Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/fineweb_stream.py`

Create a self-contained multi-shard streaming and memorization-diagnostics module containing:

1. **`FINEWEB10B_BASE_URL`**:
   ```python
   FINEWEB10B_BASE_URL = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main"
   FINEWEB_SHARD_HEADER_BYTES = 1024  # 256 int32 header (magic, version, n_tokens, ...)
   FINEWEB_TRAIN_SHARD_BYTES = 200_001_024  # 100,000,000 uint16 tokens + 1024B header
   FINEWEB_VAL_SHARD_BYTES = 10_000_000     # ~4,999,488 uint16 tokens + 1024B header
   ```

2. **`generate_contextual_zipf_shard(num_tokens: int, vocab_size: int = 50257, shard_idx: int = 1, base_seed: int = 42) -> np.ndarray`**:
   - Generates a deterministic `np.uint16` token array of length `num_tokens` governed by a **shared ground-truth contextual Markov-Zipf transition mechanism** (dependently seeded by `base_seed` for the true underlying language transition rules, and by `base_seed + 10007 * shard_idx` for the stochastic realization of shard `shard_idx`):
     - Specifically:
       - A global RNG `rng_law = np.random.default_rng(base_seed)` defines a fixed Zipfian base distribution $p_0(v) \propto (v + 1)^{-1.08}$ over `eff_vocab = min(2048, vocab_size)` and a deterministic latent cluster/topic transition structure: each token $v \in \{0, \dots, \text{eff\_vocab}-1\}$ maps to one of $C = 16$ syntactic classes $c(v) = v \bmod 16$, and each class $c$ has a preferred successor class $c_{\text{next}}(c) = (5c + 3) \bmod 16$ and a class-specific favored token table `favored_tokens[c, k]` of size $K_{\text{fav}} = 16$ sampled once via `rng_law`.
       - A shard-specific RNG `rng_shard = np.random.default_rng(base_seed + 10007 * int(shard_idx))` samples the token stream of length `num_tokens`: at each position $t$, with probability $0.65$ the next token is drawn from `favored_tokens[c(x_{t-1})]` (contextual transition law shared across all shards and validation), and with probability $0.35$ it is drawn from the Zipfian background $p_0$.
     - Because `favored_tokens` depends only on `base_seed` while the sequence realization depends on `shard_idx`, all training shards `shard_idx = 1..K` and the held-out validation shard (`shard_idx = 0`) follow the **exact same ground-truth conditional distribution** $P(x_t \mid x_{t-1})$ while containing distinct stochastic token realizations!

3. **`FineWebMultiShardStream`**:
   - `__init__(self, data_dir: Path, num_shards: int = 9, prefetch_async: bool = True, synthetic_tokens_per_shard: int = 0, vocab_size: int = 50257, seed: int = 42) -> None`:
     - Validates `1 <= num_shards <= 100`.
     - Stores configuration and initializes an internal thread-safe prefetch cache `_prefetch_futures: dict[int, concurrent.futures.Future[np.ndarray]]` using a single-worker `ThreadPoolExecutor(max_workers=1)`.
   - `shard_path(self, shard_idx: int) -> Path`:
     - Returns `self.data_dir / f"fineweb_train_{shard_idx:06d}.bin"`.
     - For `shard_idx == 1`, if `fineweb_train_000001.bin` does not exist on disk but `self.data_dir / "train_full_100M.bin"` exists with size $\ge 200,000,000$ bytes, returns `self.data_dir / "train_full_100M.bin"` so existing Wave 1–3 caches are reused with zero re-download.
   - `ensure_val_shard(self) -> np.ndarray`:
     - If `self.synthetic_tokens_per_shard > 0`, returns `generate_contextual_zipf_shard(max(16385, self.synthetic_tokens_per_shard // 2), vocab_size=self.vocab_size, shard_idx=0, base_seed=self.seed)`.
     - Otherwise loads or downloads `self.data_dir / "val_slice.bin"` from `f"{FINEWEB10B_BASE_URL}/fineweb_val_000000.bin"` (`10,000,000` bytes) and returns `np.fromfile(val_path, dtype=np.uint16, offset=FINEWEB_SHARD_HEADER_BYTES)`.
   - `_load_or_download_shard_sync(self, shard_idx: int) -> np.ndarray`:
     - If `self.synthetic_tokens_per_shard > 0`, returns `generate_contextual_zipf_shard(self.synthetic_tokens_per_shard, vocab_size=self.vocab_size, shard_idx=shard_idx, base_seed=self.seed)`.
     - Otherwise resolves `path = self.shard_path(shard_idx)`. If `not path.exists() or path.stat().st_size < FINEWEB_TRAIN_SHARD_BYTES`, downloads `f"{FINEWEB10B_BASE_URL}/fineweb_train_{shard_idx:06d}.bin"` (`200,001,024` bytes) to a `.part` temporary file and atomically renames it to `path`.
     - Returns `np.memmap(path, dtype=np.uint16, mode="r", offset=FINEWEB_SHARD_HEADER_BYTES)` (zero-copy read-only memory map).
   - `prefetch_shard(self, shard_idx: int) -> None`:
     - If `self.prefetch_async` and `1 <= shard_idx <= self.num_shards` and `shard_idx not in self._prefetch_futures`: submits `self._load_or_download_shard_sync(shard_idx)` to the background executor.
   - `get_shard(self, shard_idx: int) -> tuple[np.ndarray, float]`:
     - Measures wait time `t0 = time.perf_counter()`. If `shard_idx` is in `self._prefetch_futures`, pops and awaits the future; otherwise calls `self._load_or_download_shard_sync(shard_idx)`.
     - Automatically calls `self.prefetch_shard(shard_idx + 1)` if `shard_idx + 1 <= self.num_shards`.
     - Returns `(tokens_array, wait_ms)`.
   - `iter_shards(self, num_passes: int = 1) -> Iterator[tuple[int, int, np.ndarray, float]]`:
     - If `self.prefetch_async`: calls `self.prefetch_shard(1)` before starting.
     - Loops `pass_idx` in `range(num_passes)` and `shard_idx` in `range(1, self.num_shards + 1)`, yielding `(pass_idx, shard_idx, tokens, wait_ms)`.
   - `close(self) -> None`: shuts down the background executor cleanly.

4. **`compute_memorization_gap_metrics(...) -> dict[str, float]`**:
   ```python
   def compute_memorization_gap_metrics(
       train_ce_at_evals: list[float],
       val_ce_at_evals: list[float],
       boundary_pre_train_ce: list[float] | None = None,
       boundary_post_train_ce: list[float] | None = None,
   ) -> dict[str, float]:
   ```
   - Computes:
     - `final_train_ce`: `float(train_ce_at_evals[-1])`
     - `final_val_ce`: `float(val_ce_at_evals[-1])`
     - `final_memorization_gap`: `float(val_ce_at_evals[-1] - train_ce_at_evals[-1])` ($\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$; positive indicates `Train CE < Val CE` memorization)
     - `max_memorization_gap`: `float(max(v - t for t, v in zip(train_ce_at_evals, val_ce_at_evals)))`
     - `mean_boundary_drop`: mean of `(pre - post)` across shard/pass boundaries (positive when `post < pre` drops at pass boundary), or `0.0` if empty.

5. **`compute_empirical_bigram_coverage(shards: list[np.ndarray], max_vocab: int = 2048) -> dict[str, float]`**:
   - Computes the number of distinct token unigrams and distinct consecutive token bigrams `(int(x[t]) % max_vocab, int(x[t+1]) % max_vocab)` observed across the list of shard arrays, returning `{"unique_unigrams": float(n_uni), "unique_bigrams": float(n_bi), "bigram_coverage_ratio": float(n_bi / max(1, max_vocab * max_vocab))}`.

---

### 3.2 Create `/home/festivity/agy/QuantumDLBlocks/src/qdlblocks_jax/fineweb_stream.py`

Create the JAX companion module containing:
- `compute_memorization_gap_metrics_jax(train_ce_arr: jax.Array, val_ce_arr: jax.Array) -> dict[str, jax.Array]`: pure JAX computation of `final_memorization_gap` and `max_memorization_gap`.
- `extract_shard_batch_jax(shard_tokens: np.ndarray, chunk_idx: int, batch_size: int, seq_len: int) -> tuple[jax.Array, jax.Array]`: slices `buf = shard_tokens[start : start + B*T + 1]` and returns `(jnp.asarray(buf[:-1].reshape(B, T), dtype=jnp.int32), jnp.asarray(buf[1:].reshape(B, T), dtype=jnp.int32))`.

---

### 3.3 Update `/home/festivity/agy/QuantumDLBlocks/examples/11_fineweb_100m_multiepoch_gpu_tpu.py`

Extend `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` while preserving 100% backward compatibility when `--num-shards 1` (the default):
1. Add CLI arguments:
   - `--num-shards`: `int`, default `1`, help `"Number of distinct 100M-token FineWeb shards (fineweb_train_000001..N.bin) to stream per epoch (default: 1; set --num-shards 9 --epochs 1 for 900M unique tokens)."`
   - `--eval-every-tokens-m`: `float`, default `50.0`, help `"Validation evaluation interval in millions of tokens (default: 50.0M tokens = 762 steps at 65,536 tok/step)."`
2. In `run_cuda_benchmark` and `run_tpu_benchmark`:
   - When `num_shards == 1`, behavior is unchanged (`1` shard looped for `num_epochs` epochs, `steps_per_epoch = 1525`, `total_steps = 1525 * num_epochs`).
   - When `num_shards > 1`:
     - Instantiate `FineWebMultiShardStream(data_dir=data_dir, num_shards=num_shards, prefetch_async=True, synthetic_tokens_per_shard=synthetic_tokens)`.
     - Load `val_tokens = stream.ensure_val_shard()` and Shard 1 `first_shard, _ = stream.get_shard(1)` to determine `steps_per_shard = max(1, (len(first_shard) - max(4, mtp_horizons + 1)) // tok_per_step)`.
     - Set `steps_per_epoch = steps_per_shard * num_shards` and `total_steps = steps_per_epoch * num_epochs` (so for `--num-shards 9 --epochs 1`, `steps_per_shard = 1,525`, `total_steps = 13,725` steps = `899,481,600` unique tokens!).
     - Set `eval_every = max(1, int(round((eval_every_tokens_m * 1e6) / tok_per_step)))` (for `50.0M` tokens, `eval_every = 762` steps, evaluating twice per 100M-token shard = 18 validation checkpoints across 900M tokens!).
     - Stream shards sequentially (`shard_idx = 1 .. num_shards`), prefetching `shard_idx + 1` in the background thread while training on `shard_idx`, and logging `[Epoch X.XX | Step Y | Z.ZZM tok | T=512]` in the exact same telemetry format so `monitor_colab_runs.py` parses every step and validation checkpoint seamlessly!

---

### 3.4 Create `/home/festivity/agy/QuantumDLBlocks/examples/21_hypothesis_900m_multishard_streaming.py`

Create a standalone verification script that executes:
1. **Experiment 1 (`H1` — `1x9` Single-Shard Repetition vs. `9x1` Multi-Shard Unique Streaming with Cosine Schedule)**:
   - Trains two identically initialized `QuantumBornGPT` models (`vocab_size=512, embed_dim=64, seed_dim=24, num_layers=2, num_heads=4, hidden_dim=96, use_two_beam_mlp=True, use_overcomplete_seed_adapter=True`) for $144$ steps ($9 \times 16$ steps/shard, $B=16, T=64$, $1,024$ tokens/step, $147,456$ total training tokens):
     - **Arm A (`repeated_1x9_cosine`)**: 1 shard (`shard_idx=1`, $N_{\text{shard}} = 16,385$ tokens) repeated for $9$ passes.
     - **Arm B (`streaming_9x1_cosine`)**: 9 distinct shards (`shard_idx=1..9`, $N_{\text{shard}} = 16,385$ tokens each) streamed in $1$ pass.
   - Evaluates on the held-out validation shard (`shard_idx=0`) at every shard boundary and records `final_val_ce`, `final_train_ce`, `final_memorization_gap`, `mean_boundary_drop`, and `unique_bigrams`.
2. **Experiment 2 (`H2` — `1x9` vs. `9x1` under WSD Schedule + Async Prefetch Latency Benchmark)**:
   - Trains the same two data regimes under `--schedule wsd` (`decay_fraction = 0.20`):
     - **Arm C (`repeated_1x9_wsd`)** vs. **Arm D (`streaming_9x1_wsd`)**.
   - Benchmarks `FineWebMultiShardStream(prefetch_async=True)` across 9 shards and verifies `max_prefetch_wait_ms < 25.0`.
3. Writes all results and pass/fail assertions to `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_011.json`.

---

## 4. Verification & Unit Testing Plan

Create `/home/festivity/agy/QuantumDLBlocks/tests/test_hypothesis_011_multishard_streaming.py` with 5 unit tests:
1. `test_contextual_zipf_shards_shared_distribution_distinct_realizations`:
   - Verifies that `generate_contextual_zipf_shard` produces distinct token sequences for `shard_idx=1..9` (`not np.array_equal(s1, s2)`), while `compute_empirical_bigram_coverage` across 9 shards yields $\ge 2.5\times$ more unique bigrams than 9 repetitions of Shard 1.
2. `test_fineweb_multishard_stream_fallback_and_async_prefetch`:
   - Creates a temporary directory with a mock `train_full_100M.bin` and `val_slice.bin`, verifies `FineWebMultiShardStream.shard_path(1)` reuses `train_full_100M.bin`, and verifies `iter_shards(num_passes=1)` with `prefetch_async=True` yields exact deterministic arrays for all shards with `wait_ms < 25.0`.
3. `test_memorization_gap_metrics_torch_and_jax_parity`:
   - Verifies `compute_memorization_gap_metrics` and `compute_memorization_gap_metrics_jax` agree to `< 1e-6` on synthetic train/val trajectories.
4. `test_h1_multishard_streaming_eliminates_memorization_drag`:
   - Runs a fast controlled comparison (`1x9` repeated shard vs `9x1` unique shards) and asserts $\Delta_{\text{mem}}^{\text{9x1}} \le \frac{1}{3.0}\Delta_{\text{mem}}^{\text{1x9}}$ and $\mathcal{L}_{\text{val}}^{\text{9x1}} < \mathcal{L}_{\text{val}}^{\text{1x9}}$.
5. `test_examples_11_cli_accepts_num_shards_argument`:
   - Verifies `examples/11_fineweb_100m_multiepoch_gpu_tpu.py` parses `--num-shards 9 --epochs 1 --eval-every-tokens-m 50.0` cleanly.
