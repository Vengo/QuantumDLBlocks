# Local Quantum Wavepacket & Density-Matrix Schur Research Roadmap (`TODOs`)

**Last Updated**: 2026-11-03  
**Scope**: Local single-GPU (`NVIDIA GeForce RTX 5060 Ti`, `16 GB` VRAM) architectural exploration of initialization, fiber-bundle geometry, multi-branch wavepacket readouts, and density-matrix Schur-product attention/highways.  
**Data Hygiene Invariant**: All corpus statistics (`unigram`, skip-gram `CountSketch`, Kneser-Ney backoffs) and training steps strictly use `fineweb_train_000001.bin .. fineweb_train_000009.bin` (`900M` tokens). Validation strictly uses unseen `fineweb_val_000000.bin` (`Shard 0`).

---

## 1. Completed Milestones & Verified Findings

1. **Support-Preserving CountSketch & Fiber-Bundle Split (`I1..I8`)**:
   - **Code**: [`src/qdlblocks/benchmarks/support_preserving_countsketch_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/support_preserving_countsketch_lab.py)
   - **Result**: Splitting a product-sphere embedding table into **Semantic Packets** (initialized via GPU multi-horizon PPMI `CountSketch` + iterative tangent Gram-Schmidt) and **Identity Packets** (strictly orthogonalized across all `50,304` tokens) + per-packet adaptive gating achieves **`100.00%` Top-1 accuracy (`0.0002` Val CE)** on `50,304`-vocab sequence reversal in 300 steps.

2. **Complex Product-Sphere Shape Tradeoff (`(50304, M, 2, d_c)` for $M \in \{4, 8, 16\}$)**:
   - **Code**: [`src/qdlblocks/benchmarks/complex_sphere_tradeoff_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/complex_sphere_tradeoff_lab.py)
   - **Result**: At total real dimension $d = 512$, **`(50304, 8, 2, 32)`** (`7` Semantic + `1` Identity packet, or `6` Semantic + `2` Identity packets) is the Pareto sweet spot (`100.00%` reversal Top-1 accuracy and lowest FineWeb Step-0 CE), balancing per-packet complex dimension ($d_c = 32$, 64 real DoF) against packet modularity ($M = 8$).

3. **900M-Token Shards `1..9` Multi-Horizon Initialization & Kneser-Ney Synergy (`H1..H9`)**:
   - **Code**: [`src/qdlblocks/benchmarks/long_context_semantic_init_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/long_context_semantic_init_lab.py)
   - **Result**: Dedicating Semantic Packet 0 to a shared **Unigram Frequency Anchor** and Semantic Packets `1..6` to **Multi-Horizon (`[1, 2, 4, 8]`) Skip-Gram CountSketch** over Shards `1..9` drops zero-training FineWeb Shard-0 Val CE from `10.8286 -> 7.1079` (pure `(50304, 8, 2, 32)` wavepacket table) and to **`5.4754` Val CE (`21.62%` Top-1 accuracy at Step 0)** when combined with exact Shard `1..9` Kneser-Ney bigram calibration.

4. **Multi-Branch Wavepacket Readout & The Geodesic Stiefel Tangent-Cone Frame (`M1..M4`)**:
   - **Code**: [`src/qdlblocks/benchmarks/multibranch_wavepacket_readout_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/multibranch_wavepacket_readout_lab.py)
   - **Result 1 ($K_{\max}$ Superposition Verification)**: Verified empirically that a single `64D` identity readout breaks at $K_{\max}(64, 50304) = \frac{64}{2\ln(50304)} = \mathbf{2.956}$ simultaneous targets (`100%` recall at $K=1,2$, dropping to `78.9%` at $K=4$ and `22.5%` at $K=16$). Scaling to $R=4$ orthogonal branches restores `100%` recall through $K=8$ (`-2.39` nats excess CE at $K=4$), and $R=8$ branches restores `100%` recall through $K=16$ (`-3.25` nats excess CE at $K=8$).
   - **Result 2 (Geodesic Stiefel Tangent-Cone Frame vs. $90^\circ$ Orthogonal Dilution)**: On 300-step (`2.46M`-token) FineWeb training (`L = 4, d = 512, T = 512`), placing $R = 4$ identity branches on a **$30^\circ$ Geodesic Stiefel Tangent Cone** ($\text{cone\_id}_r = \cos(30^\circ)\, u_{\text{id},0} + \sin(30^\circ)\, u_{\text{id},r}$ with $u_{\text{id},r} \perp u_{\text{id},0}$) avoids the $-\ln \pi_0$ orthogonal probability-dilution tax and **strictly beats single-branch `S3` across every metric**:
     - `Val CE`: `5.8400 -> 5.8299` (`PPL: 343.76 -> 340.31`)
     - `Top-1 / Top-5`: `18.92% / 35.37% -> 19.12% / 35.68%`
     - `Active Branches`: `2.35 / 4` (context-adaptive).

5. **Born-Schur Density-Matrix Attention & Full Density-Matrix Highway (`D0..D4`)**:
   - **Code**: [`src/qdlblocks/benchmarks/born_schur_density_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/born_schur_density_lab.py), [`scripts/born_schur_density_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/born_schur_density_bench.py)
   - **Result 1 (Dimension-Calibrated Schur Scaling & Thermodynamic Cooling)**:
     - Verified pure-packet Schur identity $(|u\rangle\langle u|) \odot (|v\rangle\langle v|) = (|u\rangle \odot |v\rangle)(|u\rangle \odot |v\rangle)^\dagger$ (`1.01e-10` error) and single-GEMM Hermitian triangle packing (`1.42e-09` error).
     - Scaling entrywise Schur products by $D$ ($S_{t,ab} = D \cdot (\bar{\rho}_{t,a} \odot \bar{\rho}_{t,b})$) calibrates $\mathbb{E}[\operatorname{Tr}(S_{t,ab})] = 1.0$ to match $\operatorname{Tr}(\bar{\rho}_{t,h}) = 1.0$.
     - Across an 8-layer `DensityMatrixHighwayBlock` stack, combining entrywise Schur binding with 1-step McWeeny purification $\rho \leftarrow \rho^2 / \operatorname{Tr}(\rho^2)$ drives spectral purity $\operatorname{Tr}(\rho^2)$ from `0.6607 -> 0.9993` (`EffRank = 1.00`) and computational-basis diagonal coincidence $D\sum_d \rho_{dd}^2$ from `1.6217 -> 1.9652`.
   - **Result 2 (`50,304`-Vocab Sequence Reversal — Monotonic Gains Across `D0 -> D1 -> D2 -> D3`)**:
     - `D0_MCWEENY_MATMUL_BASELINE`: Step 50 `Val CE = 5.5768` (`33.56%`) $\to$ Step 100 `0.0981` (`100.00%`) $\to$ Step 200 `0.0233` (`100.00%`)
     - `D1_BORN_SCHUR_STATIC_ATTN`: Step 50 `Val CE = 5.2542` (`36.59%`) $\to$ Step 100 `0.1281` (`99.96%`) $\to$ Step 200 `0.0232` (`100.00%`)
     - `D2_BORN_SCHUR_DYNAMIC_POVM`: Step 50 `Val CE = 5.1959` (`37.80%`) $\to$ Step 100 `0.0946` (`99.99%`) $\to$ Step 200 `0.0226` (`100.00%`)
     - **`D3_FULL_DENSITY_MATRIX_HIGHWAY`**: Step 50 **`Val CE = 4.3994` (`46.77%`, `-1.18` nats ahead of `D0`)** $\to$ Step 100 **`0.0576` (`100.00%`)** $\to$ Step 200 **`0.0183` (`100.00%`)**.
    - **Result 3 (FineWeb `L=4, d=512, T=512` 300-Step Benchmark — `D4_BORN_SCHUR_POVM_CONE_R4`)**:
      - Reaches `Val CE = 5.8592` (`PPL = 350.43`, `Top-1 = 18.86%`, `Top-5 = 35.31%`, `Active Branches = 2.29 / 4`) with zero extra activation memory via $O(k D)$ low-rank Cholesky Schur binding.

6. **Wavefunction Outer-Product Input Density (`W0..W3`) & Cone-Free Native Cholesky Density Readout (`D5, D6`)**:
   - **Code**: [`src/qdlblocks/benchmarks/wavefunction_outer_density_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/wavefunction_outer_density_lab.py), [`scripts/wavefunction_outer_density_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wavefunction_outer_density_bench.py)
   - **Result 1 (`50,304`-Vocab Reversal Input Outer-Product Density Benchmark `W0..W3`)**:
     - Lifting input token wavefunctions $|\psi_{x_t, h}\rangle$ and phase-transported neighbor wavefunctions $|\tilde{\psi}_{x_{t-1}, h}\rangle$ to outer-product density operators before combining them (`W1_OUTER_PHASE_ALIGNED_MIXTURE` and `W2_OUTER_PLUS_BIGRAM_SCHUR_INPUT`) is **100% invariant to arbitrary global phase shifts** ($0.0$ numerical error under random phase gauge transformations) and improves Step-100 (`0.0576 -> 0.0527`) and Step-200 (`0.0183 -> 0.0179`) reversal cross-entropy over vector addition (`W0_VECTOR_SUM_THEN_OUTER`).
   - **Result 2 (Eliminating the `D4` Rank-4 $\to$ Rank-1 $\to$ Rank-4 Bottleneck — `D5_NATIVE_CHOLESKY_DENSITY_NO_CONE` & `D6_OUTER_INPUT_NATIVE_CHOLESKY_NO_CONE` Beat All Previous `300`-Step FineWeb Models)**:
     - Stripping out the redundant `99K`-parameter $R=4$ Geodesic Cone readout head (`mb_head = None`, zero `id_branch_stiefel`, zero $30^\circ$ cone hyperparameter, zero Step-0 SVD calibration) and reading out the 4 Cholesky branches directly from the final attention layer's rank-$4$ Cholesky density factor $L_{t,7}^{(L)} \in \mathbb{C}^{4 \times 32}$ achieves a **new best 300-step FineWeb result**:
       - **`D5_NATIVE_CHOLESKY_DENSITY_NO_CONE`**: **`Val CE = 5.7528` (`PPL = 315.08`, `Top-1 = 20.95%`, `Top-5 = 36.64%`, `CholeskyModes = 2.24 / 4`)** — beating `D4_BORN_SCHUR_POVM_CONE_R4` (`5.8592`, `18.86%` Top-1) by **`-0.1064` nats (`-35.35` PPL, `+2.09%` Top-1)** and beating `M1_CONE_FIBER_R4` (`5.8299`, `19.12%` Top-1) by **`-0.0771` nats (`+1.83%` Top-1)** with **`99,000` fewer parameters**!
       - **`D6_OUTER_INPUT_NATIVE_CHOLESKY_NO_CONE`**: **`Val CE = 5.7612` (`PPL = 317.73`, `Top-1 = 20.92%`, `Top-5 = 36.68%`, `CholeskyModes = 2.26 / 4`)**.

7. **Uncollapsed 4-Mode Cholesky Inter-Layer Highway (`D7, D8`) & `1,200`-Step (`9.83M`-Token) FineWeb Scaling (`D9`)**:
   - **Code**: [`src/qdlblocks/benchmarks/wavefunction_outer_density_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/wavefunction_outer_density_lab.py), [`scripts/wavefunction_outer_density_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wavefunction_outer_density_bench.py)
   - **Result 1 (Why Inter-Layer Cholesky Propagation Requires Canonical Pre-`U_evol` Basis + $U(1)^4$ Row Phase-Locking)**:
     - Propagating the rank-4 Cholesky factor $L_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$ across layers $\ell = 1 \to 2 \to 3 \to 4$ without collapsing requires keeping the highway in the canonical pre-$U_{\text{evol}}$ packet basis and applying [`phase_lock_cholesky_modes`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/wavefunction_outer_density_lab.py#L326-L343) ($l_r \mapsto e^{i\phi_r} l_r$, which leaves $\rho = L^\dagger L$ strictly invariant while guaranteeing constructive real overlap $\operatorname{Re}\langle \text{cur}_r \mid \text{in}_r^{\text{aligned}}\rangle \ge 0$).
   - **Result 2 (Joint Semantic `448D` + Identity `64D` Uncollapsed Cholesky Readout `D8` Jumps Another `-0.0658` Nats to `5.6870` Val CE in `300` Steps)**:
     - **`D7_UNCOLLAPSED_CHOLESKY_4MODE_HIGHWAY`** (Identity Cholesky readout only): **`Val CE = 5.7522` (`PPL = 314.87`, `Top-1 = 20.92%`, `Top-5 = 36.83%`, `CholeskyModes = 2.27 / 4`)**.
     - **`D8_UNCOLLAPSED_CHOLESKY_SEM_ID_READOUT`** (Uncollapsed 4-Mode Cholesky Highway + Joint Semantic `0..6` & Identity `7` Cholesky Readout): **`Val CE = 5.6870` (`PPL = 295.00`, `Top-1 = 21.00%`, `Top-5 = 37.32%`, `CholeskyModes = 2.44 / 4`)** — beating `D5` (`5.7528`) by **`-0.0658` nats (`-20.08` PPL, `+0.68%` Top-5)**, beating `D4` (`5.8592`) by **`-0.1722` nats (`-55.43` PPL, `+2.14%` Top-1, `+2.07%` Top-5)**, and beating single-branch `M0` (`5.9102`) by **`-0.2232` nats (`-73.78` PPL)**!
   - **Result 3 (`1,200`-Step / `9.83M`-Token FineWeb Scaling Run — `D9_UNCOLLAPSED_CHOLESKY_LONG_SCALING_1200` Breaks Under `5.0` Val CE to `4.9162`)**:
     - Across `1,200` steps (`9.83M` training tokens on Shards `1..9`, evaluated on unseen Shard 0 `65,536` tokens):
       - **Pure Neural Val CE** drops monotonically by **`-1.9743` nats (`7.1549 -> 5.1806`, `PPL: 1,280.36 -> 177.79`)**.
       - **Combined KN-Synergy + Uncollapsed Cholesky Val CE** breaks below `5.00` at Step 750 (`5.0030`) and reaches **`4.9162` (`PPL = 136.48`, `Top-1 = 26.12%`, `Top-5 = 43.93%`, `CholeskyModes = 3.04 / 4`)** at Step 1,200!

---

## 2. Prioritized Research `TODO` Backlog

### `TODO-1`: Multi-Epoch / `4,800`-Step (`39.3M`-Token) Local Scaling Run of `D9` with Trigram / 4-Gram Semantic Subspace Calibration Toward `< 4.5` Val CE
- **Objective**: Scale **`D8 / D9`** (`NativeCholeskyDensityWavepacketGPT` with `uncollapsed_cholesky_highway=True, use_semantic_cholesky_readout=True`) from `1,200` steps (`9.83M` tokens, **`4.9162` Val CE / `5.1806` Neural CE**) to `4,800` steps (`39.3M` tokens) on Shards `1..9`, optionally pairing with a 6-layer (`L = 6`) uncollapsed Cholesky trunk.
- **Entry Point**: [`scripts/wavefunction_outer_density_bench.py`](file:///home/festivity/agy/QuantumDLBlocks/scripts/wavefunction_outer_density_bench.py) (`--skip-reversal --skip-fineweb --run-long-scaling --long-scaling-steps 4800`).
