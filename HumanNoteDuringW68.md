Interesting side conversation again while the script runs. 

I haven't closely inspected the network architecture in a bit, but I'm getting an impression that we're not sticking to the spirit of what I want to try. It sounds like we're collapsing things to classical probabilities in spots we might not need to. Generating positive weights that sum to one is clearly important for mixing density matrices, but I want the network to generally focus on operations involving wave functions and density matrices. I think the ideal network block might be one that takes in a batch of density matrices, projects them to Q/K/V spaces, mixes the Vs together based on quantum similarity from the Q/K spaces, and then adds the resulting mixed density matrix deltas back to the input density matrix. There's significant design space in there for multi-head architecture and efficient state representation.

Path A Semantic BPE Clustering 
This sounds like something we can't do in the spirit of the NanoGPT speedrun. If there's a good way to build up information about observed bytes, byte pairs and longer sequences of bytes, I'm open to it. The structure must follow the spirit of the NanoGPT speedrun though, and I'd like it to match with the spirit of the rest of the quantum structures we're using. I can imagine there's some way that we could store this information as a bunch of density matrices that would get updated with an EMA based on what a new token reveals about ngram frequency.

The Path B True Gauge-Invariant Grassmanian/Hilbert-Schmidt Causal Attention on the Cholesky Factors
I'm not sure what "the real vector stream" means here. I would expect token-to-token attention is happening in a wave function or density matrix space. Exponential decay channels definitely sound like they could be a problem. Using a learned horizontal Stiefel probe sounds appealing. Would flattening the hermition outer product be expressive enough? I'm definitely interested in this.

The Path C Cross-Head Quantum Entanglement/Subspace Braiding
I definitely want there to be some sort of cross-head mixing. At one point in an earlier wave, I think we discussed a process where additive mixing acts like an "OR" and Schur products act like "AND" operations, and together they allow the network to represent complex logic. I'm definitely interested in proposals to fix the problem in Path C, and it doesn't have to bring back Schur products. Whatever operation makes the most sense to mix the heads' output, I'm interested.

The Path D Full-Matrix horizontal muon preconditioning and fisher scaling
I'm open to addressing this, but is the Mach–Zehnder interferometer the right approach to begin with? This might collide with other changes, but what other operations could apply here? If Mach-Zehnder interferometer is the right choice, let's keep it, but I'm curious what else could fit.

The Path E Tanh logit softcap removal
I'm comfortable this change, and an unconstrained exact Born log-likelihood sounds great.

The Path F Stateful document-level carry over
This sounds really cool! I've wondered about exploration directions that would try to learn incredibly long-range dependencies, maybe even storing millions of previous tokens and then implementing a version of our quantum attention mechanism that can attend to that enormous history buffer. Passing a density matrix from one training window to the next sounds related to me.

I think there's a lot here to address that can be pursued. Some of it would probably benefit from our new theorem proving process.

Side Question: what are the most fruitful paths available for driving error lower?

Now that horizontal gradient completeness (`SelectiveExactQuotientMuon`) and the `150M+`-token plateau LR starvation bug (`(step - 200) / 300.0`) are fixed, the remaining validation cross-entropy (CE) gap is governed by **architectural expressivity bottlenecks**, **readout geometry**, and **cross-token/cross-head quantum transport**.

Here are the most fruitful paths available for driving validation error lower, ranked by expected impact per FLOP:

---

### 1. Already in Flight in Wave 68 (`W68_L2` & `W68_L3`)
Before looking at new changes, the three changes currently running in `W68_L2` (`4L`) and `W68_L3` (`8L`) target the largest immediate bottlenecks from `W67_S3`:
1. **Horizon-Proportional Plateau Schedule**: Fixing the hardcoded `/ 300.0` decay denominator keeps the learning rate in `[0.707x, 1.000x]` across the entire `85%` plateau (`0 -> 127.5M` tokens) instead of starving the optimizer down to `0.128x` mid-run.
2. **Eliminating the 50% Tomographic Readout Bottleneck (`256D -> 512D`)**: `W67_S3` projected the `512D` two-basis $\mathfrak{su}(16)$ MUB measurement down to a single `256D` subspace (`U_out`), discarding half of the measured quantum state norm before scoring tokens. The complementary dual-subspace readout $[U_{\text{coarse}}, U_{\text{fine}}] \in \operatorname{O}(512)$ preserves **100% of the Parseval tomographic norm** ($\|U_{\text{coarse}}^\top u\|^2 + \|U_{\text{fine}}^\top u\|^2 = \|u\|^2$).
3. **Scaling Trunk Depth ($L = 4 \to 8$) with $\sqrt{4/L}$ Geodesic Trust Regions (`W68_L3`)**: `W67_S3` only had `~18.3M` parameters in its 4-layer quantum trunk (`12.9M` were in the vocabulary boundary). Doubling trunk depth to $L=8$ (`~31.1M` trunk parameters) while scaling residual-writing Stiefel/Kraus geodesic steps by $\sqrt{4/L}$ adds hierarchical composition without blowing up highway transport variance.

---

### 2. Next Highest-Leverage Paths Beyond Wave 68

#### Path A: Semantic BPE Clustering for the $256_{\text{coarse}} \times 256_{\text{fine}}$ Lüders Readout (Zero Extra FLOPs, High Gain)
- **The Bottleneck**: In our bipartite $256 \times 256$ readout, token $v \in \{0, \dots, 50256\}$ is currently mapped to coarse cluster $a = \lfloor v / 256 \rfloor$ and fine offset $b = v \bmod 256$ using raw tokenizer ID order. Because BPE token IDs are ordered by merge history rather than semantics or syntax, a single coarse bucket $a$ mixes completely unrelated tokens (e.g., punctuation, verb suffixes, and rare nouns). That forces the coarse `256D` subspace $U_{\text{coarse}}$ to learn an unnatural disjunctive partition.
- **The Fix**: Introduce a fixed permutation buffer `token_to_cluster_perm[v]` (computed once offline or at Step 0 via $k$-means / spectral clustering on token unigram/bigram co-occurrence or initial embeddings) so that each of the 256 coarse outcomes corresponds to a coherent syntactic/semantic cluster.
- **Why it is high-leverage**: It requires **zero extra parameters and zero extra runtime FLOPs** (just a 1D index lookup on `targets`), while making the two-stage Sequential Lüders measurement ($P(a, b) = P_{\text{coarse}}(a) P_{\text{fine}}(b \mid a)$) align with natural language hierarchy.

#### Path B: True Gauge-Invariant Grassmannian / Hilbert–Schmidt Causal Attention on the Cholesky Factors $L_t \in \mathbb{C}^{16 \times R}$
- **The Bottleneck**: Right now, the density-matrix highway $\rho_t^{(h)} = L_t^{(h)} L_t^{(h)\dagger}$ evolves across sequence positions primarily via exponential/Lindblad recurrence and local convolutions, while token-to-token attention happens in the real vector stream $x_t$. Exponential decay channels have fixed memory horizons and struggle with sharp, content-addressed retrieval 100–500 tokens back.
- **The Fix**: Because we already maintain the Rank-$R$ ($R=4$) Cholesky factor $L_t^{(h)} \in \mathbb{C}^{16 \times 4}$ of each head's density matrix $\rho_t^{(h)}$, the exact quantum Born / Hilbert–Schmidt overlap between positions $t$ and $s$ is:
  $$\operatorname{Tr}\!\left(\rho_t^{(h)} \rho_s^{(h)}\right) = \left\| L_t^{(h)\dagger} L_s^{(h)} \right\|_F^2 = \operatorname{Tr}\!\left(\left(L_t^{(h)} L_t^{(h)\dagger}\right) \left(L_s^{(h)} L_s^{(h)\dagger}\right)\right)$$
  Flattening the Hermitian outer product $L_t^{(h)} L_t^{(h)\dagger}$ (or using a learned horizontal Stiefel probe $W_Q, W_K$ on $L_t^{(h)}$) allows **linear-cost or chunked causal quantum attention** directly on the density-matrix highway.
- **Why it is high-leverage**: It is **strictly $U(R)$-gauge invariant** by construction ($L_t \mapsto L_t U_t$ cancels identically in $L_t L_t^\dagger$)—realizing the exact *"Extrinsic Density-Matrix Highway / Convex Projective Attention"* from your earlier geometric analysis while giving the quantum highway long-range, content-selective retrieval.

#### Path C: Cross-Head Quantum Entanglement / Subspace Braiding Between Layers
- **The Bottleneck**: Each of the $H=16$ heads evolves in an isolated $\mathbb{C}^{16}$ Hilbert space: both the CPTP Kraus channel $\sum_{r=1}^R K_r^{(h)} \rho^{(h)} K_r^{(h)\dagger}$ and the $\mathfrak{su}(16)$ MUB measurement act **block-diagonally per head**. Cross-head interaction only occurs after collapsing $\rho^{(h)}$ to classical MUB probabilities and projecting through `obs_proj`.
- **The Fix**: Before the per-head Kraus evolution in each layer, apply a lightweight **unitary cross-head braiding / Stiefel mixing** across the $H=16$ head channels (e.g., an orthogonal $(16 \times 16)$ head-mixing rotation on the Cholesky factors $L^{(h)}$ followed by trace normalization).
- **Why it is high-leverage**: It allows quantum phase coherence to propagate *across* heads across layers $1 \dots L$ without collapsing to classical probabilities at every layer boundary, effectively expanding the coherent state space from $\bigoplus_{h=1}^{16} \mathbb{C}^{16}$ into an entangled multi-head system at negligible compute cost.

#### Path D: Full-Matrix Horizontal Muon Preconditioning for the Mach–Zehnder MLP (`up_gate_proj`) & Fisher Scaling for Singular Spectra ($\sigma$)
- **The Bottleneck**: In [`ParallelTransportSelectiveQuotientMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/complementary_512d_lindblad_8l_lab.py), the Stiefel factors $(U, V)$ and Kraus stacks $\mathbf{K}$ receive 5-step Newton–Schulz polar orthogonalization, which equalizes curvature across all tangent directions. However:
  1. The largest parameter block in each layer—the Mach–Zehnder interferometer MLP's `up_gate_proj` (`512 -> 3072`, accounting for `~50%` of trunk parameters)—is updated without Stiefel/polar orthogonalization if treated as an unconstrained Euclidean weight.
  2. In `BiorthogonalSVDStiefelProjection` ($M = U \operatorname{diag}(\sigma) V^\top$), a perturbation $d\Omega$ to $(U, V)$ induces a physical operator change scaled by $\sigma_i$, whereas $d\sigma$ acts with unit metric.
- **The Fix**:
  1. Apply standard 5-step Newton–Schulz orthogonalization (or semi-orthogonal Stiefel retraction) to `mz_mlp.up_gate_proj` so 100% of the dense trunk matrices benefit from spectral-norm steps.
  2. Precondition the rotational step $(dU, dV)$ vs. singular-value step $d\sigma$ by the local Riemannian metric of the SVD map (verified in Lean 4 first via [`SplitSpectrumBiorthogonalSVD.lean`](file:///home/festivity/agy/QuantumDLBlocks/lean/QuantumDLBlocksLean/SplitSpectrumBiorthogonalSVD.lean)).

#### Path E: Removing or Relaxing the `tanh` Logit Softcap (`cap = 30.0`) Under Parseval-Bounded Readout
- **The Bottleneck**: Early architectures needed `30.0 * tanh(z / 30.0)` in the cross-entropy readout to prevent Euclidean logit explosions from destabilizing training. However, now that the input to the readout is L2/Parseval-normalized on the $\mathfrak{su}(16)$ MUB sphere and projected through an exact orthogonal matrix $[U_{\text{coarse}}, U_{\text{fine}}] \in \operatorname{O}(512)$ with analytical temperature $\beta$, logits are geometrically bounded. As validation CE drops below `~4.6`, the `tanh` softcap artificially squashes high-certainty token predictions (like deterministic BPE continuations and closing brackets/quotes), creating an artificial loss floor.
- **The Fix**: Replace the hard `cap = 30.0` `tanh` squash with either an unconstrained exact Born log-likelihood or a much gentler/higher analytical bound once the Stiefel readout norm is locked.

#### Path F: Stateful Document-Level Quantum Density Carry-Over Across Batch Chunks
- **The Bottleneck**: At `seq_len = 512`, every training window resets $\rho_0$ at token position 0 to a fresh initial state, blinding the first 50–100 tokens of every sequence to the preceding paragraph.
- **The Fix**: Because our packed density-matrix highway state is compact (`L_packed` of shape `(B, H, 16, 4, 2)` = only `8 KB` per batch stream), carrying the detached final quantum state $\rho_{T}^{(h)}$ across consecutive document segments gives the model infinite-horizon quantum memory at zero extra FLOPs and near-zero VRAM cost.
