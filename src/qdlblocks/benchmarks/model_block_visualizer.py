"""Interactive Model Block-Graph Introspector & WebUI Visualizer (`Wave 44–45`).

When to Use This Building Block:
- Use `ModelBlockGraphSpec`, `build_w44_architecture_presets`, `load_w44_empirical_telemetry_summary`,
  and `build_architecture_visualizer_payload` when inspecting or visualizing how QuantumDLBlocks models—
  specifically `w44_compact_r2_d128c_L2_b32` (`MultiHeadMixedStateKroneckerGPT` with `R=2, L=2,
  d_complex=128, M=4, d_f=16, K=2, B=32, T=512`) and its companion architectures—string together
  their embedding, causal smearing, multi-head Stinespring Gram attention, two-beam unitary FFN,
  3-stage Kronecker `16^4` unitary meaning mixer, carrier-orthogonal complex self-echo, 900M N-gram
  channels, phase-sensitive polysemy router, and fused Triton mixed-state Born readout blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_PRESET_KEY: str = "w44_compact_r2_d128c_L2_b32"
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_SEQ_LEN: int = 512
DEFAULT_VOCAB_SIZE: int = 50304
DEFAULT_VALID_VOCAB_SIZE: int = 50257
DEFAULT_NUM_MEANINGS: int = 2
DEFAULT_NUM_FACTORS: int = 4
DEFAULT_FACTOR_DIM: int = 16
DEFAULT_SUBHEADS_PER_FACTOR: int = 2
DEFAULT_NUM_LAYERS_COMPACT: int = 2
DEFAULT_NUM_LAYERS_FLAGSHIP: int = 6
DEFAULT_VOCAB_CHUNK_SIZE: int = 4096
DEFAULT_BOSON_P: float = 11.0
DEFAULT_ATTN_BOSON_P: int = 2
DEFAULT_BIGRAM_SLOTS: int = 8192
DEFAULT_TRIGRAM_BUCKETS: int = 16384
DEFAULT_FFN_EXPANSION: int = 2
KRONECKER_HILBERT_DIM: int = 65536  # 16^4 = 65,536


@dataclass(frozen=True)
class ParameterTensorSpec:
    """Metadata for a single learnable parameter or paired complex tensor inside a block (`<= 8` fields)."""

    name: str
    shape: str
    params: int
    dtype_str: str
    optimizer_group: str
    init_rule: str
    role: str


@dataclass(frozen=True)
class SubBlockNodeSpec:
    """Specification of a micro sub-block strung inside a macro stage (`<= 8` fields)."""

    id: str
    title: str
    subtitle: str
    category: str
    in_shape: str
    out_shape: str
    hilbert_space: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class MacroStageSpec:
    """Specification of a top-level macro stage in the strung-together model pipeline (`<= 8` fields)."""

    stage_id: str
    stage_num: int
    title: str
    subtitle: str
    total_params: int
    pct_of_model: float
    tensor_ribbon: str
    sub_blocks: list[SubBlockNodeSpec]


@dataclass(frozen=True)
class SignalHighwaySpec:
    """Specification of an end-to-end signal or skip highway across blocks (`<= 8` fields)."""

    id: str
    label: str
    color: str
    description: str
    node_ids: list[str]
    edge_notes: list[str]


@dataclass(frozen=True)
class ModelBlockGraphSpec:
    """Complete introspected block-graph specification for a model preset (`<= 8` fields)."""

    preset_key: str
    title: str
    model_class: str
    summary_badge: str
    total_params: int
    hyperparams: dict[str, Any]
    stages: list[MacroStageSpec]
    highways: list[SignalHighwaySpec]


def _pt(name: str, shape: str, params: int, dtype_str: str, opt: str, init: str, role: str) -> dict[str, Any]:
    """Build serialized `ParameterTensorSpec` dictionary (`<= 8` args)."""
    return asdict(ParameterTensorSpec(name, shape, params, dtype_str, opt, init, role))


def _build_stage0_embedding_and_smearing(
    b_sz: int, seq_t: int, r_num: int, m_num: int, d_f: int, total_params: int,
) -> MacroStageSpec:
    """Construct Stage 0: Input Token Lookup, Tied `16^4` Wavefunction & Causal 1-Token Smearing (`<= 8` args)."""
    d_c, k_tot = r_num * m_num * d_f, r_num * m_num
    embed_params = DEFAULT_VOCAB_SIZE * r_num * m_num * d_f * 2
    branch_logit_params = DEFAULT_VOCAB_SIZE * r_num
    smear_params = k_tot * d_f * d_f * 2 + 1
    stage_params = embed_params + branch_logit_params + smear_params

    sub_blocks = [
        SubBlockNodeSpec(
            id="s0_tokens", title="0.1 Input Token Stream & Lag-1 Shift",
            subtitle="FineWeb GPT-2 BPE Token IDs + Causal Lag-1 Index Shift", category="input",
            in_shape=f"idx: ({b_sz}, {seq_t}) int64",
            out_shape=f"idx_t, idx_{{t-1}}: ({b_sz}, {seq_t}) in [0, {DEFAULT_VALID_VOCAB_SIZE - 1}]",
            hilbert_space="Discrete Token Index Space V = 50,257 (padded to 50,304)",
            meta={
                "params": 0, "optimizer": "None (Input Indices)",
                "kernel": "Clamped index gather + 1-token causal right-shift",
                "source_ref": "src/qdlblocks/models/ngram_calibrated_kronecker_born_gpt.py#L816-L844",
                "equation": r"x_t \in \{0,\dots,50256\}, \quad x_{\text{prev},t} = x_0 \text{ if } t=0 \text{ else } x_{t-1}",
                "why_built": "Feeds three parallel pathways simultaneously: (1) current token wavefunction |ψ_{0,t}⟩, (2) lag-1 causal context smearing |ψ_{0,t-1}⟩, and (3) 900M-token calibrated Bigram (x_t) & Hashed Trigram (x_{t-1}, x_t) readout channels.",
                "highways": ["primary", "anchor_psi0", "ngram_hperp", "tied_vocab"], "param_tensors": [],
            },
        ),
        SubBlockNodeSpec(
            id="s0_tied_embed", title=f"0.2 Tied Rank-R={r_num} 16^4 Complex Wavefunction Table",
            subtitle=f"embed_re / embed_im ({DEFAULT_VOCAB_SIZE}, R={r_num}, M={m_num}, d_f={d_f}) + branch_logits",
            category="embedding", in_shape=f"idx: ({b_sz}, {seq_t})",
            out_shape=f"|ψ_0⟩: ({b_sz}, {seq_t}, R={r_num}, M={m_num}, d_f={d_f}) complex -> ({b_sz}, {seq_t}, {d_c})",
            hilbert_space=f"Product of Complex Spheres (CS^15)^{k_tot} subset C^(16^4 = 65,536)",
            meta={
                "params": embed_params + branch_logit_params,
                "optimizer": "AdamW (embed_lr = 0.045 * scale, weight_decay = 0.0)",
                "kernel": "Step-0 900M-Token Polar-ZCA Helmert H_perp Calibration + Algebraic Branch Alpha",
                "source_ref": "src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py#L646-L648",
                "equation": r"|w_{v,r,m}\rangle = \rho_v |c_{\text{ref}}\rangle + \sqrt{1-\rho_v^2}\,|u_{\perp,v,r,m}\rangle \in \mathbb{C}\mathbb{S}^{15}, \quad |\psi_{0,t,r,m}\rangle = \sqrt{R}\,\alpha_{x_t,r}\,|w_{x_t,r,m}\rangle",
                "why_built": "Ties the input token representation directly to the 16^4 = 65,536 output Born measurement kets. In Wave 44 (Track 4), secondary polysemy branches (r >= 1) are initialized at full spectral strength (secondary_branch_scale = 1.0, alpha_spec = 0.88) via carrier-preserving 15D Helmert orthogonal rotations branch_ortho_rot[r,m], eliminating the legacy 0.22x (98.1% noise) branch collapse.",
                "highways": ["primary", "anchor_psi0", "tied_vocab", "adamw_opt"],
                "param_tensors": [
                    _pt("embed_re + embed_im", f"2 x ({DEFAULT_VOCAB_SIZE}, {r_num}, {m_num}, {d_f})", embed_params, "float32 / bfloat16", "AdamW (embed_params)", "900M Polar-ZCA 30D spectral coords -> 15D H_perp + carrier tilt |c_ref⟩", "Tied Rank-R 4-factor complex token & vocabulary measurement kets"),
                    _pt("branch_logits", f"({DEFAULT_VOCAB_SIZE}, {r_num})", branch_logit_params, "float32", "AdamW (gate_params)", "Zeros -> uniform Step-0 alpha_{v,r} = 1 / sqrt(R)", "100% exp-free algebraic vocabulary polysemy branch weights alpha_{v,r}"),
                ],
            },
        ),
        SubBlockNodeSpec(
            id="s0_causal_smear", title="0.3 Causal 1-Token Unitary Context Smearing",
            subtitle=f"smear_u_re/im ({k_tot}, {d_f}, {d_f}) + smear_gate_logit + Cloned |ψ_0⟩ Anchor",
            category="unitary", in_shape=f"raw |ψ_0⟩: ({b_sz}, {seq_t}, d_c={d_c}) complex",
            out_shape=f"smeared |ψ^(0)⟩: ({b_sz}, {seq_t}, {d_c}) + anchor |ψ_0⟩: ({b_sz}, {seq_t}, {d_c})",
            hilbert_space=f"U({d_f})^{k_tot} Unitary Transport + Complex Sphere Normalization",
            meta={
                "params": smear_params,
                "optimizer": "Joint-Complex Polar Express StiefelMuon (smear_u) + AdamW (smear_gate_logit)",
                "kernel": "_apply_factor_unitary_bank + complex_l2_normalize",
                "source_ref": "src/qdlblocks/models/ngram_calibrated_kronecker_born_gpt.py#L816-L844",
                "equation": r"|\psi^{(0)}_t\rangle = \text{Normalize}_{\mathbb{C}}\!\Big(|\psi_{0,t}\rangle + 0.25\tanh(\theta_{\text{smear}})\big(\bigoplus_{k=1}^{8} U_{\text{smear},k}\big)|\psi_{0,t-1}\rangle\Big)",
                "why_built": "Injects immediate causal 1-token look-back context (x_{t-1} -> x_t) via learnable 16x16 unitary rotations before Layer 0 begins, while cloning the un-smeared |ψ_0⟩ anchor so downstream self-echo removal and phase routing have an exact reference for x_t.",
                "highways": ["primary", "anchor_psi0", "stiefel_muon", "adamw_opt"],
                "param_tensors": [
                    _pt("smear_u_re + smear_u_im", f"2 x ({k_tot}, {d_f}, {d_f})", k_tot * d_f * d_f * 2, "float32 (complex unitary bank)", "StiefelMuon (is_complex_pair_group=True)", "Haar random complex unitary U(16) per (r, m) factor", "Rotates lag-1 token factor kets before blending into current token"),
                    _pt("smear_gate_logit", "(1,)", 1, "float32", "AdamW (gate_params)", "DEFAULT_SMEAR_GATE_INIT = -0.85", "Controls causal lag-1 unitary smearing amplitude"),
                ],
            },
        ),
    ]
    return MacroStageSpec(
        stage_id="stage_0_embed", stage_num=0,
        title="Stage 0 — Tied 16^4 Wavefunction Lookup & Causal 1-Token Unitary Smearing",
        subtitle=f"Maps token IDs (B={b_sz}, T={seq_t}) into R={r_num} x M={m_num} x C^{d_f} complex kets (d_c={d_c}) + |ψ_0⟩ anchor",
        total_params=stage_params, pct_of_model=round(100.0 * stage_params / float(max(1, total_params)), 2),
        tensor_ribbon=f"idx ({b_sz}, {seq_t}) ➔ |ψ^(0)⟩ & |ψ_0⟩ ({b_sz}, {seq_t}, {d_c} complex)",
        sub_blocks=sub_blocks,
    )


def _build_transformer_layer_stage(
    layer_idx: int, b_sz: int, seq_t: int, r_num: int, m_num: int, d_f: int, k_sub: int, total_params: int,
) -> MacroStageSpec:
    """Construct Stage `layer_idx + 1`: Quantum Transformer Block + `16^4` Unitary Meaning Mixer (`<= 8` args)."""
    d_c, h_tot = r_num * m_num * d_f, m_num * k_sub
    d_attn, d_ffn = h_tot * d_f, d_c * DEFAULT_FFN_EXPANSION
    qkv_params, o_proj_params = 3 * (d_c * d_attn * 2), d_attn * d_c * 2
    stinespring_params, gram_eta_params = m_num * k_sub * r_num * 2, m_num
    v0_mix_params, attn_gate_params = (h_tot if layer_idx > 0 else 0), d_c
    attn_total = qkv_params + o_proj_params + stinespring_params + gram_eta_params + v0_mix_params + attn_gate_params
    ffn_linear_params, ffn_misc_params = 3 * (d_c * d_ffn * 2), d_ffn + 2 + d_c
    ffn_total = ffn_linear_params + ffn_misc_params
    mixer_u16, mixer_u4, mixer_ur = r_num * m_num * d_f * d_f * 2, r_num * m_num * m_num * 2, r_num * r_num * 2
    mixer_misc = (r_num * m_num * d_f) + 1
    mixer_total = mixer_u16 + mixer_u4 + mixer_ur + mixer_misc
    stage_params, stage_num = attn_total + ffn_total + mixer_total, layer_idx + 1
    v0_note = f"Active in Layer {layer_idx}: blends Layer-0 V^(0) anchor via v0_subhead_mix_logit ({h_tot},)" if layer_idx > 0 else "Layer 0 captures raw value kets V^(0) and exports them to downstream layers!"

    qkv_tensors = [_pt(f"blocks.{layer_idx}.attn.{{q,k,v}}_proj.weight_{{re,im}}", f"3 x 2 x ({d_attn}, {d_c})", qkv_params, "float32 (3 complex pairs)", "StiefelMuon (is_complex_pair_group=True)", "Haar complex unitary U(128) initialization", "Projects 128D complex state into M=4 x K=2 = 8 sub-heads of C^16")]
    if v0_mix_params > 0:
        qkv_tensors.append(_pt(f"blocks.{layer_idx}.attn.v0_subhead_mix_logit", f"({h_tot},)", v0_mix_params, "float32", "AdamW (gate_params)", "INIT_V0_SUBHEAD_MIX_LOGIT = -1.1", "Algebraic root-sigmoid gate mixing Layer-0 V^(0) into Layer-l V^(l)"))

    sub_blocks = [
        SubBlockNodeSpec(
            id=f"l{layer_idx}_qkv_rope", title=f"{stage_num}.1 Complex QKV Projections + V^(0) Residual + Fused Norm-RoPE",
            subtitle=f"q_proj, k_proj, v_proj ComplexLinear({d_c}->{d_attn}) -> H={h_tot} sub-heads (M={m_num}xK={k_sub}, d_f={d_f})",
            category="attention", in_shape=f"|ψ^({layer_idx})⟩: ({b_sz}, {seq_t}, {d_c}) complex",
            out_shape=f"Q, K, V: ({b_sz}, H={h_tot}, {seq_t}, d_f={d_f}) on CS^15",
            hilbert_space=f"H = M*K = {m_num}*{k_sub} = {h_tot} Complex Register Sub-Heads of C^{d_f}",
            meta={
                "params": qkv_params + v0_mix_params, "optimizer": "StiefelMuon (q/k/v_proj pairs) + AdamW (v0_subhead_mix_logit)",
                "kernel": "fused_complex_norm_rope_triton (@triton.jit L2 norm + e^{i t omega} rotary phase)",
                "source_ref": "src/qdlblocks/layers/multihead_superposition_mixed_state_attn.py#L465-L515",
                "equation": r"Q, K, V = \text{Reshape}_{(B, M\cdot K, T, 16)}\big(W_Q|\psi\rangle, W_K|\psi\rangle, W_V|\psi\rangle\big), \quad \hat{q}_{t,h} = e^{i t \boldsymbol{\omega}} q_{t,h}/\|q_{t,h}\|_2",
                "why_built": f"Track 2 & Track 6: Enforces d_register = {d_f} across H = M*K = {h_tot} sub-heads so each sub-head operates natively in the C^16 Kronecker factor dimension. {v0_note}",
                "highways": ["primary", "anchor_psi0", "stiefel_muon", "triton_kernels"], "param_tensors": qkv_tensors,
            },
        ),
        SubBlockNodeSpec(
            id=f"l{layer_idx}_born_attn", title=f"{stage_num}.2 Fused Flash Causal p=2 Complex Born Attention",
            subtitle="Single-Pass 3-Matmul / SRAM-Tiled 64x64 Triton Born Attention (11 -> 3 T^2 GEMMs)",
            category="attention", in_shape=f"Q, K, V: ({b_sz}, H={h_tot}, {seq_t}, d_f={d_f}) complex",
            out_shape=f"Φ = (phi_re, phi_im): ({b_sz}, {seq_t}, M={m_num}, K={k_sub}, d_f={d_f}) on CS^15",
            hilbert_space=f"Causal p={DEFAULT_ATTN_BOSON_P} Born Probability Simplex (100% exp/softmax-free)",
            meta={
                "params": 0, "optimizer": "Parameter-Free Quantum Measurement Rule",
                "kernel": "flash_causal_pboson_complex_attn_triton / fused_causal_pboson_born_triton",
                "source_ref": "src/qdlblocks/layers/multihead_superposition_mixed_state_attn.py#L160-L196",
                "equation": r"W_{t,s} = \frac{|\langle \hat{k}_s \mid \hat{q}_t \rangle_{\mathbb{C}}|^4\,\mathbb{I}(s \le t)}{\sum_{j \le t} |\langle \hat{k}_j \mid \hat{q}_t \rangle_{\mathbb{C}}|^4 + \varepsilon}, \qquad |\phi_{t,m,k}\rangle = \text{Normalize}_{\mathbb{C}}\!\Big(\sum_{s \le t} W_{t,s}|\hat{v}_{s,m,k}\rangle\Big)",
                "why_built": "Track 6 (Issue 6): Replaces the legacy 11 (T x T) intra-head Krylov power-iteration GEMMs per layer with a single-pass causal p=2 Born attention kernel, cutting quadratic T^2 matmuls by 72.7% (11 -> 3) and letting the K=2 sub-heads per factor carry distinct contextual modes.",
                "highways": ["primary", "triton_kernels"], "param_tensors": [],
            },
        ),
        SubBlockNodeSpec(
            id=f"l{layer_idx}_gram_stinespring", title=f"{stage_num}.3 Cross-Head Hermitian Gram Density & Stinespring Coupling",
            subtitle=f"Trace-1 KxK ({k_sub}x{k_sub}) Gram G_hat + Stinespring W_m (M={m_num}, K={k_sub}, R={r_num}) + o_proj",
            category="attention", in_shape=f"Φ: ({b_sz}, {seq_t}, M={m_num}, K={k_sub}, d_f={d_f}) complex",
            out_shape=f"|ψ_mid^({layer_idx})⟩: ({b_sz}, {seq_t}, {d_c}) complex on CS^{d_c - 1}",
            hilbert_space=f"Hermitian Density Matrix G_hat in C^({k_sub}x{k_sub}) (Tr(G_hat)=1) -> R={r_num} Meaning Branches",
            meta={
                "params": gram_eta_params + stinespring_params + o_proj_params + attn_gate_params,
                "optimizer": "StiefelMuon (stinespring_w, o_proj) + AdamW (gram_eta, attn_gate)",
                "kernel": "fused_cross_head_gram_stinespring_step (@torch.compile fused Gram + Stinespring)",
                "source_ref": "src/qdlblocks/layers/multihead_superposition_mixed_state_attn.py#L198-L319",
                "equation": r"\hat{G}_{t,m} = \frac{1}{K}\Phi_{t,m}\Phi_{t,m}^H \in \mathbb{C}^{K\times K}, \quad \tilde{\Phi}_{t,m} = \Phi_{t,m} + \eta_{G,m}\hat{G}_{t,m}\Phi_{t,m}, \quad |\psi^{\text{attn}}_{t,r,m}\rangle = \sum_{k=0}^{K-1} W_{m,k,r}|\tilde{\phi}_{t,m,k}\rangle",
                "why_built": "By the Schmidt decomposition, the K x K = 2 x 2 cross-head Gram matrix G_hat_{t,m} shares the exact same non-zero eigenvalues and Bloch purity Tr(G_hat^2) in [1/K, 1] as the 16 x 16 outer density matrix rho_{t,m}, in O(K^2 d_f) time with zero ad-hoc 1.5*rho - 0.5*rho^2 polynomial or gamma gate.",
                "highways": ["primary", "stiefel_muon", "adamw_opt", "triton_kernels"],
                "param_tensors": [
                    _pt(f"blocks.{layer_idx}.attn.gram_eta", f"({m_num},)", gram_eta_params, "float32", "AdamW (gate_params)", "DEFAULT_GRAM_COUPLING_ETA_INIT = 0.15", "Per-factor Hermitian Gram transport coupling strength eta_{G,m}"),
                    _pt(f"blocks.{layer_idx}.attn.stinespring_w_{{re,im}}", f"2 x ({m_num}, {k_sub}, {r_num})", stinespring_params, "float32 (3D complex semi-unitary)", "StiefelMuon (is_complex_pair_group=True)", "Haar semi-unitary W_m^H W_m = I_R", "Stinespring projection mapping K=2 sub-heads into R=2 meaning branches"),
                    _pt(f"blocks.{layer_idx}.attn.o_proj.weight_{{re,im}}", f"2 x ({d_c}, {d_attn})", o_proj_params, "float32 (complex pair)", "StiefelMuon (is_complex_pair_group=True)", "Haar complex unitary U(128)", "Output unitary projection across packed (R, M, d_f) registers"),
                    _pt(f"blocks.{layer_idx}.attn_gate", f"({d_c},)", attn_gate_params, "float32", "AdamW (gate_params)", f"0.35 / sqrt({layer_idx + 1})", "Tanh-bounded residual gate before sphere re-normalization"),
                ],
            },
        ),
        SubBlockNodeSpec(
            id=f"l{layer_idx}_unitary_ffn", title=f"{stage_num}.4 Two-Beam ComplexUnitaryInterferenceFFN",
            subtitle=f"Mach-Zehnder Two-Beam Interferometer (up_a, up_b: {d_c}->{d_ffn}, beam_phase, down: {d_ffn}->{d_c})",
            category="ffn", in_shape=f"|ψ_mid^({layer_idx})⟩: ({b_sz}, {seq_t}, {d_c}) complex",
            out_shape=f"|ψ_ffn^({layer_idx})⟩: ({b_sz}, {seq_t}, {d_c}) complex on CS^{d_c - 1}",
            hilbert_space=f"Overcomplete Two-Beam Interference Space C^{d_ffn} -> CS^{d_c - 1}",
            meta={
                "params": ffn_total, "optimizer": "StiefelMuon (up_a, up_b, down complex pairs) + AdamW (beam_phase, bias, gates)",
                "kernel": "ComplexLinear + Mach-Zehnder e^{i*phi} interference + algebraic modulus gate",
                "source_ref": "src/qdlblocks/models/gemini_packed_complex_rope_gpt.py#L142-L195",
                "equation": r"|u_a\rangle = W_a|\psi\rangle, \quad |u_b\rangle = e^{i\boldsymbol{\phi}}\odot(W_b|\psi\rangle), \quad |m\rangle = \frac{1}{\sqrt{2}}(|u_a\rangle + |u_b\rangle)\odot \frac{\sqrt{|u_a|^2+|u_b|^2}}{\sqrt{|u_a|^2+|u_b|^2}+\sigma(b)}",
                "why_built": "Replaces classical ReLU/SwiGLU MLPs with a quantum Mach-Zehnder two-beam phase interferometer that creates constructive and destructive wave interference between two unitary projections.",
                "highways": ["primary", "stiefel_muon", "adamw_opt"],
                "param_tensors": [
                    _pt(f"blocks.{layer_idx}.ffn.{{up_a,up_b,down}}.weight_{{re,im}}", f"2x({d_ffn},{d_c}) + ({d_c},{d_ffn}) complex pairs", ffn_linear_params, "float32 (3 complex pairs)", "StiefelMuon (is_complex_pair_group=True)", "Semi-unitary ComplexLinear initialization", f"Two-beam expansion ({d_c} -> {d_ffn}) and contraction ({d_ffn} -> {d_c})"),
                    _pt(f"blocks.{layer_idx}.ffn.{{beam_phase,bias,gate}} + ffn_gate", f"({d_ffn},) + (1,) + (1,) + ({d_c},)", ffn_misc_params, "float32", "AdamW (gate_params)", "Phase 0.0, bias 0.25, residual 0.35/sqrt(l+1)", "Relative interferometer phase phi, modulus threshold bias, and residual gate"),
                ],
            },
        ),
        SubBlockNodeSpec(
            id=f"l{layer_idx}_kron_mixer", title=f"{stage_num}.5 Kronecker16x4UnitaryMeaningMixer (3-Stage Unitary Circuit)",
            subtitle=f"Stage 1: Intra-Factor U(16) -> Stage 2: Cross-Factor U(4) -> Stage 3: Cross-Branch U({r_num})",
            category="mixer", in_shape=f"|ψ_ffn^({layer_idx})⟩: ({b_sz}, {seq_t}, {d_c}) -> ({b_sz}, {seq_t}, R={r_num}, M={m_num}, d_f={d_f})",
            out_shape=f"|ψ^({layer_idx + 1})⟩: ({b_sz}, {seq_t}, {d_c}) complex (per-factor unit norm on (CS^15)^8)",
            hilbert_space=f"Structured Kronecker Tensor Product (C^16)^(x4) x C^{r_num} = C^65,536 x {r_num}",
            meta={
                "params": mixer_total, "optimizer": "StiefelMuon (factor_u, cross_factor_u, branch_mixer_u) + AdamW (mod_bias, res_scale)",
                "kernel": "fused_kronecker_16x4_unitary_mixer_step (@torch.compile fused 3-stage unitary circuit)",
                "source_ref": "src/qdlblocks/models/kronecker_multimeaning_16x4_gpt.py#L430-L520",
                "equation": r"\text{(1) } |\psi'_{r,m}\rangle = U^{(16)}_{r,m}|\psi_{r,m}\rangle, \quad \text{(2) } |\psi''_{r,m}\rangle = \phi_{\text{mod}}\!\Big(\sum_{m'=1}^4 U^{(4)}_{r,m,m'}|\psi'_{r,m'}\rangle\Big), \quad \text{(3) } |\psi'''_{r,m}\rangle = \sum_{s=1}^R U^{(R)}_{r,s}|\psi''_{s,m}\rangle",
                "why_built": "Explicitly respects the 3-level hierarchy of the model: (1) rotates coordinates inside each C^16 factor, (2) entangles the M=4 Kronecker factors within each meaning branch, and (3) couples the R=2 polysemy meaning branches before re-normalizing each C^16 register onto CS^15.",
                "highways": ["primary", "stiefel_muon", "adamw_opt", "triton_kernels"],
                "param_tensors": [
                    _pt(f"kron_mixers.{layer_idx}.factor_u_{{re,im}}", f"2 x ({r_num}, {m_num}, {d_f}, {d_f})", mixer_u16, "float32 (complex unitary bank)", "StiefelMuon / Unitary Penalty", "Haar random U(16) per (r, m)", "Stage 1: Intra-factor U(16) basis rotation"),
                    _pt(f"kron_mixers.{layer_idx}.cross_factor_u_{{re,im}} + branch_mixer_u_{{re,im}}", f"2x({r_num},{m_num},{m_num}) + 2x({r_num},{r_num})", mixer_u4 + mixer_ur, "float32 (complex unitaries)", "StiefelMuon (is_complex_pair_group=True)", "Haar U(4) and near-identity U(R)", "Stage 2 cross-factor entanglement U(4) and Stage 3 cross-branch mixing U(R)"),
                    _pt(f"kron_mixers.{layer_idx}.{{mod_bias,res_scale}}", f"({r_num}, {m_num}, {d_f}) + (1,)", mixer_misc, "float32", "AdamW (gate_params)", "mod_bias = -1.0, res_scale = 0.20", "Phase-preserving complex modulus gate threshold and mixer residual scale"),
                ],
            },
        ),
    ]
    return MacroStageSpec(
        stage_id=f"stage_{stage_num}_layer_{layer_idx}", stage_num=stage_num,
        title=f"Stage {stage_num} — Quantum Transformer Layer {layer_idx} (blocks[{layer_idx}] + kron_mixers[{layer_idx}])",
        subtitle=f"Multi-Head Stinespring Gram Attention (M={m_num}, K={k_sub}) ➔ Two-Beam Unitary FFN ➔ 3-Stage 16^4 Unitary Meaning Mixer",
        total_params=stage_params, pct_of_model=round(100.0 * stage_params / float(max(1, total_params)), 2),
        tensor_ribbon=f"|ψ^({layer_idx})⟩ ({b_sz}, {seq_t}, {d_c}) ➔ |ψ^({layer_idx + 1})⟩ ({b_sz}, {seq_t}, {d_c} complex)",
        sub_blocks=sub_blocks,
    )


def _build_stage_readout_prep(
    stage_num: int, b_sz: int, seq_t: int, r_num: int, m_num: int, d_f: int, total_params: int,
) -> MacroStageSpec:
    """Construct Calibrated Multi-Channel Readout State Preparation stage (`<= 8` args)."""
    n_tok, d_c, k_tot = b_sz * seq_t, r_num * m_num * d_f, r_num * m_num
    out_u_params, echo_params = k_tot * d_f * d_f * 2 + 1, r_num * m_num
    bi_table_params = (DEFAULT_BIGRAM_SLOTS + 1) * m_num * d_f * 2
    tri_table_params = DEFAULT_TRIGRAM_BUCKETS * m_num * d_f * 2
    bi_tri_u_params, ngram_gate_params = 2 * (k_tot * d_f * d_f * 2), r_num * 3
    ngram_total = bi_table_params + tri_table_params + bi_tri_u_params + ngram_gate_params
    router_params = (r_num * m_num * d_f * 2) + (r_num * 3)
    stage_params = out_u_params + echo_params + ngram_total + router_params

    sub_blocks = [
        SubBlockNodeSpec(
            id="s_prep_out_u", title=f"{stage_num}.1 Conjugate |ψ_0*⟩ Skip Blend & Procrustes Readout Unitary Bank",
            subtitle=f"psi0_readout_gate + out_factor_u_re/im ({k_tot}, {d_f}, {d_f})", category="readout_prep",
            in_shape=f"|ψ^(L)⟩ & |ψ_0⟩: ({b_sz}, {seq_t}, {d_c}) complex",
            out_shape=f"op_4d: (N={n_tok}, R={r_num}, M={m_num}, d_f={d_f}) complex",
            hilbert_space=f"U({d_f})^{k_tot} Procrustes-Aligned Vocabulary Measurement Frame",
            meta={
                "params": out_u_params, "optimizer": "StiefelMuon (out_factor_u complex pair) + AdamW (psi0_readout_gate)",
                "kernel": "_apply_factor_unitary_bank (batched 8 x 16 x 16 complex bmm)",
                "source_ref": "src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py#L862-L871",
                "equation": r"|\psi_{\text{mix}}\rangle = \text{Normalize}_{\mathbb{C}}\!\big(|\psi^{(L)}\rangle + \tanh(g_0)|\psi_0^*\rangle\big), \qquad |\text{op}_{t,r,m}\rangle = U_{\text{out},r,m}|\psi_{\text{mix},t,r,m}\rangle",
                "why_built": "Rotates the backbone output factors into the tied vocabulary measurement frame. At Step 0, out_factor_u is Procrustes-aligned so the empirical mean backbone state maps directly onto the canonical phase-balanced carrier |c_ref⟩ = ((1+i)/(4*sqrt(2))) * 1_16.",
                "highways": ["primary", "anchor_psi0", "stiefel_muon"],
                "param_tensors": [
                    _pt("out_factor_u_re + out_factor_u_im", f"2 x ({k_tot}, {d_f}, {d_f})", k_tot * d_f * d_f * 2, "float32 (complex unitary bank)", "StiefelMuon (is_complex_pair_group=True)", "Step-0 Procrustes unitary alignment to |c_ref⟩", "Maps backbone factor kets into vocabulary measurement coordinates"),
                    _pt("psi0_readout_gate", "(1,)", 1, "float32", "AdamW (gate_params)", "DEFAULT_PSI0_GATE_INIT = 0.20", "Conjugate Layer-0 skip anchor blend weight before readout unitary"),
                ],
            },
        ),
        SubBlockNodeSpec(
            id="s_prep_self_echo", title=f"{stage_num}.2 Track 3 Carrier-Orthogonal Complex Self-Echo Projection",
            subtitle=f"apply_carrier_orthogonal_complex_self_echo in 15D H_perp + echo_gate_logits ({r_num}, {m_num})",
            category="readout_prep", in_shape=f"op_4d & p0_4d: (N={n_tok}, R={r_num}, M={m_num}, d_f={d_f}) complex",
            out_shape=f"echo_4d: (N={n_tok}, R={r_num}, M={m_num}, d_f={d_f}) complex (<c_ref|echo> == <c_ref|op>)",
            hilbert_space="15D Zero-Sum Carrier-Orthogonal Subspace H_perp = {|u⟩ in C^16 : <c_ref | u⟩ = 0}",
            meta={
                "params": echo_params, "optimizer": "AdamW (gate_params)",
                "kernel": "Zero-mean H_perp projection + full complex Hermitian inner product removal",
                "source_ref": "src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py#L119-L175",
                "equation": r"|\hat{u}_{0,\perp}\rangle = \frac{P_{\mathcal{H}_\perp}|\psi_0\rangle}{\|P_{\mathcal{H}_\perp}|\psi_0\rangle\|_2}, \quad |\text{echo}\rangle = |\text{op}\rangle - \lambda_{r,m}\langle \hat{u}_{0,\perp}\mid \text{op}\rangle_{\mathbb{C}}|\hat{u}_{0,\perp}\rangle",
                "why_built": "Track 3 (Issue 3): In Wave 43, ambient real-part self-echo subtraction stripped the unigram carrier |c_ref⟩ component whenever x_t was a frequent token and ignored imaginary phase overlap i*ov_im. Projecting into the 15D zero-sum subspace H_perp removes 100% of complex self-repetition while leaving unigram carrier alignment <c_ref | echo>_C strictly invariant!",
                "highways": ["primary", "anchor_psi0", "adamw_opt"],
                "param_tensors": [
                    _pt("echo_gate_logits", f"({r_num}, {m_num})", echo_params, "float32", "AdamW (gate_params)", "u_0 = 0.0 -> lambda_{r,m} = 0.50 via algebraic root-sigmoid", "Per-(branch, factor) attenuation gate in (0, 1) for H_perp self-echo"),
                ],
            },
        ),
        SubBlockNodeSpec(
            id="s_prep_ngram", title=f"{stage_num}.3 Track 4 R-Invariant Norm & 900M-Token Bigram + Trigram H_perp Injection",
            subtitle=f"bi_ctx_wave ({DEFAULT_BIGRAM_SLOTS + 1}, 4, 16) + tri_ctx_wave ({DEFAULT_TRIGRAM_BUCKETS}, 4, 16) + bi/tri_factor_u",
            category="readout_prep", in_shape=f"echo_4d: (N={n_tok}, R={r_num}, M={m_num}, d_f={d_f}) + idx ({b_sz}, {seq_t})",
            out_shape=f"phi_re, phi_im: (N={n_tok}, R={r_num}, M={m_num}, d_f={d_f}) on CS^15",
            hilbert_space="R-Invariant Unit Factor Spheres + 15D H_perp Conditional N-Gram Subspace",
            meta={
                "params": ngram_total,
                "optimizer": "AdamW (bi_ctx_wave, tri_ctx_wave, ngram_branch_gates) + StiefelMuon (bi_factor_u, tri_factor_u)",
                "kernel": "normalize_raw_phi_per_factor_r_invariant + _compute_ngram_carrier_ortho_delta",
                "source_ref": "src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py#L879-L886",
                "equation": r"|\hat{\phi}^{\text{raw}}_{r,m}\rangle = \frac{|\text{echo}_{r,m}\rangle}{\||\text{echo}_{r,m}\rangle\|_2}, \quad |\phi_{r,m}\rangle = \text{Normalize}_{\mathbb{C}}\!\Big(|\hat{\phi}^{\text{raw}}_{r,m}\rangle + \sqrt{8}\big(g_{\text{bi},r} U_{\text{bi},r,m}|b_{x_t,m}\rangle + g_{\text{tri},r} U_{\text{tri},r,m}|\tau_{x_{t-1},x_t,m}\rangle\big)\Big)",
                "why_built": "Combines 900M-token calibrated direct Bigram (8,192 top tokens) and Multiplicative Hashed Trigram (16,384 buckets) kets in H_perp with the deep quantum backbone state. Normalizing raw_phi per C^16 factor (Track 4) ensures the N-gram-to-backbone ratio is invariant to R.",
                "highways": ["primary", "ngram_hperp", "stiefel_muon", "adamw_opt"],
                "param_tensors": [
                    _pt("bi_ctx_wave_re + bi_ctx_wave_im", f"2 x ({DEFAULT_BIGRAM_SLOTS + 1}, {m_num}, {d_f})", bi_table_params, "float32", "AdamW (embed_params)", "900M-token conditional bigram expectations in H_perp", "Direct causal bigram (x_t -> x_{t+1}) context wavefunctions"),
                    _pt("tri_ctx_wave_re + tri_ctx_wave_im", f"2 x ({DEFAULT_TRIGRAM_BUCKETS}, {m_num}, {d_f})", tri_table_params, "float32", "AdamW (embed_params)", "900M-token hashed trigram residuals in H_perp", "Hashed causal trigram ((x_{t-1}, x_t) -> x_{t+1}) context wavefunctions"),
                    _pt("bi_factor_u_{re,im} + tri_factor_u_{re,im}", f"2 x 2 x ({k_tot}, {d_f}, {d_f})", bi_tri_u_params, "float32 (complex unitary banks)", "StiefelMuon (is_complex_pair_group=True)", "Co-rotated branch_ortho_rot[r,m] unitaries in H_perp", "Branch-specific unitary rotations for bigram and trigram factor kets"),
                    _pt("ngram_branch_gates", f"({r_num}, 3)", ngram_gate_params, "float32", "AdamW (gate_params)", "[1.45, 0.90, 0.24] across all R branches (secondary_branch_scale = 1.0)", "Per-branch channel gains [g_bi, g_tri, g_op]"),
                ],
            },
        ),
        SubBlockNodeSpec(
            id="s_prep_router", title=f"{stage_num}.4 Track 4 Phase-Sensitive Context-Adaptive Polysemy Router",
            subtitle=f"compute_phase_sensitive_branch_beta -> branch_beta (N={n_tok}, R={r_num}) with sum_r beta_r^2 = 1",
            category="readout_prep", in_shape=f"phi_re/im & p0_unit_re/im: (N={n_tok}, R={r_num}, M={m_num}, d_f={d_f})",
            out_shape=f"phi_factors: (N={n_tok}, R={r_num}, M={m_num}, 32) + branch_beta: (N={n_tok}, R={r_num})",
            hilbert_space=f"Real Positive Sphere S_+^{r_num - 1} (Mixed-State Branch Probabilities pi_{{t,r}} = beta_{{t,r}}^2)",
            meta={
                "params": router_params, "optimizer": "AdamW (gate_params)",
                "kernel": "compute_phase_sensitive_branch_beta (100% exp-free algebraic positivity map)",
                "source_ref": "src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py#L198-L254",
                "equation": r"f_{t,r} = \hat{\phi}_{t,r} + \hat{\phi}_{t,r}\odot \hat{\psi}_{0,t,r}^*, \quad u_{t,r} = \frac{\text{Re}\langle w^{\text{eff}}_r \mid f_{t,r}\rangle}{\sqrt{64}} + b_r, \quad \beta_{t,r} \propto \frac{1}{2}\big(u_{t,r} + \sqrt{u_{t,r}^2 + 1}\big)",
                "why_built": "Track 4 (Issue 4): The legacy norm-only router depended on ||comb_r||_2 after per-factor L2 normalization, where ||phi_r||_2 == 2 is identically constant! The phase-sensitive router reads both complex state direction and relative phase shift phi * conj(psi_0), while starting at exact uniform beta_{t,r}^{(0)} = 1 / sqrt(R) at Step 0.",
                "highways": ["primary", "anchor_psi0", "adamw_opt"],
                "param_tensors": [
                    _pt("router_w_re + router_w_im", f"2 x ({r_num}, {m_num}, {d_f})", r_num * m_num * d_f * 2, "float32", "AdamW (gate_params)", "Exact zeros (guarantees uniform Step-0 beta = 1/sqrt(R))", "Phase-sensitive complex directional routing weights per branch"),
                    _pt("branch_router_gain + branch_router_bias + context_branch_bias", f"3 x ({r_num},)", r_num * 3, "float32", "AdamW (gate_params)", "gain = 1.0, bias = 0.0", "Per-branch router gain and bias before algebraic positivity map"),
                ],
            },
        ),
    ]
    return MacroStageSpec(
        stage_id="stage_readout_prep", stage_num=stage_num,
        title=f"Stage {stage_num} — Calibrated Readout Unitary, H_perp Self-Echo, 900M N-Gram & Phase Router",
        subtitle="Prepares unit-norm context factor kets phi_factors (N, R, 4, 32) and mixed-state weights branch_beta (N, R)",
        total_params=stage_params, pct_of_model=round(100.0 * stage_params / float(max(1, total_params)), 2),
        tensor_ribbon=f"|ψ^(L)⟩ ({b_sz}, {seq_t}, {d_c}) ➔ phi_factors (N={n_tok}, R={r_num}, M={m_num}, 32) + β (N={n_tok}, R={r_num})",
        sub_blocks=sub_blocks,
    )


def _build_stage_born_readout(
    stage_num: int, b_sz: int, seq_t: int, r_num: int, m_num: int, d_f: int, total_params: int,
) -> MacroStageSpec:
    """Construct Final Stage: Fused Triton Mixed-State `16^4 = 65,536` `p=11` Boson Born Readout (`<= 8` args)."""
    n_tok, k_tot, d_2, povm_params = b_sz * seq_t, r_num * m_num, 2 * d_f, DEFAULT_VOCAB_SIZE
    sub_blocks = [
        SubBlockNodeSpec(
            id="s_readout_bmm", title=f"{stage_num}.1 Vocabulary-Chunked Batched Factor Overlap GEMM (torch.bmm)",
            subtitle=f"Contracts u_batched ({k_tot}, {2 * n_tok}, {d_2}) @ w_chunk ({k_tot}, {d_2}, {DEFAULT_VOCAB_CHUNK_SIZE}) over 13 chunks",
            category="readout_born",
            in_shape=f"phi_factors: ({n_tok}, {r_num}, {m_num}, {d_2}) + tied psi_factors: ({DEFAULT_VALID_VOCAB_SIZE}, {r_num}, {m_num}, {d_2})",
            out_shape=f"z_cat = (x_ov, y_ov): ({k_tot}, {2 * n_tok}, {DEFAULT_VOCAB_CHUNK_SIZE}) per chunk",
            hilbert_space=f"R*M = {k_tot} Parallel C^{d_f} Hermitian Inner Products <phi_{{t,r,m}} | w_{{v,r,m}}>_C",
            meta={
                "params": 0, "optimizer": "Reuses Tied embed_re / embed_im (12.88M params from Stage 0.2)",
                "kernel": "Single bfloat16 Tensor-Core torch.bmm per 4,096-vocab chunk (zero (N, V) logit materialization)",
                "source_ref": "src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py#L465-L485",
                "equation": r"x_{r,m,t,v} = \text{Re}\langle \hat{\phi}_{t,r,m}\mid \hat{w}_{v,r,m}\rangle_{\mathbb{C}}, \qquad y_{r,m,t,v} = \text{Im}\langle \hat{\phi}_{t,r,m}\mid \hat{w}_{v,r,m}\rangle_{\mathbb{C}}",
                "why_built": "Packing [phi_re, phi_im] and skew-symmetric [-phi_im, phi_re] along the token dimension allows a single batched Tensor-Core BMM of batch size K_tot = R*M = 8 to compute both the real and imaginary parts of all 4 Kronecker factor overlaps simultaneously.",
                "highways": ["primary", "tied_vocab"], "param_tensors": [],
            },
        ),
        SubBlockNodeSpec(
            id="s_readout_triton_tree", title=f"{stage_num}.2 Fused Triton 16^4 Binary Product Tree & Trace-1 Mixed-State Intensity",
            subtitle="fused_mixed_state_kronecker_chunk_fwd_triton: (z_0*z_1)*(z_2*z_3) in C^65,536 + Holographic + Coincidence",
            category="readout_born",
            in_shape=f"x_ov, y_ov: ({r_num}, {m_num}, {n_tok}, {DEFAULT_VOCAB_CHUNK_SIZE}) + β ({n_tok}, {r_num}) + α ({DEFAULT_VOCAB_CHUNK_SIZE}, {r_num})",
            out_shape=f"I_tv^mixed: ({n_tok}, {DEFAULT_VOCAB_CHUNK_SIZE}) positive Born intensity per chunk",
            hilbert_space=f"Rank-R={r_num} Mixed-State Density Operator on (C^16)^(x4) = C^65,536",
            meta={
                "params": 0, "optimizer": "Gradients flow through single-pass analytical Triton backward kernel",
                "kernel": "fused_mixed_state_kronecker_chunk_fwd_triton & fused_mixed_state_kronecker_chunk_bwd_triton",
                "source_ref": "src/qdlblocks/kernels/fused_kronecker_mixed_state_kernels.py#L60-L310",
                "equation": r"K_{r,t,v} = \prod_{m=0}^3 (x_{r,m,t,v} + i y_{r,m,t,v}), \quad I_{r,t,v} = |s_0 + \alpha_{\text{holo}} a_{r,v} M_{r,t,v}|^2 + \gamma_{\text{coinc}} a_{r,v}^2 |K_{r,t,v}|^2, \quad I^{\text{mixed}}_{t,v} = \sum_{r=0}^{R-1} \beta_{t,r}^2 I_{r,t,v} + \varepsilon",
                "why_built": "Track 6 & Wave 44 Fused Kernels: Evaluates the 65,536-dimensional Kronecker tensor product overlap in O(4) registers via a 2-level binary complex product tree (z_0*z_1)*(z_2*z_3) inside SRAM without materializing intermediate (R, 4, N, V_chunk) float32 tensors in HBM.",
                "highways": ["primary", "tied_vocab", "triton_kernels"], "param_tensors": [],
            },
        ),
        SubBlockNodeSpec(
            id="s_readout_born_nll", title=f"{stage_num}.3 Polynomial p=11 Boson Born Rule, POVM Scales & Single-Pass Backward",
            subtitle="P(v|x_<=t) = w_povm[v] * (I_tv^mixed)^11 / Z_t (100% exp/softmax/logsumexp-free)",
            category="readout_born",
            in_shape=f"I_tv^mixed: ({n_tok}, {DEFAULT_VOCAB_CHUNK_SIZE}) + povm_scale_logits: ({DEFAULT_VOCAB_SIZE},) + targets: ({n_tok},)",
            out_shape="Scalar Quantum Born NLL Loss L_NLL + Single-Pass Analytical Pullback (bwd_pow / fwd_pow == 1.0)",
            hilbert_space="50,257-Outcome Polynomial p=11 Boson POVM Measurement Simplex",
            meta={
                "params": povm_params, "optimizer": "AdamW (povm_params, povm_lr = 0.008)",
                "kernel": "MixedStateKroneckerBornChunkedNLLFn (Rule 9 Verified: 1.0x pow ratio, 2.0x bmm ratio)",
                "source_ref": "src/qdlblocks/models/multihead_mixed_state_kronecker_gpt.py#L421-L550",
                "equation": r"w_{\text{povm},v} = \frac{1}{2}\big(\theta_v + \sqrt{\theta_v^2 + 1}\big) + \varepsilon, \qquad P(v \mid x_{\le t}) = \frac{w_{\text{povm},v}\,\big(I^{\text{mixed}}_{t,v}\big)^{11}}{\sum_{u=0}^{V-1} w_{\text{povm},u}\,\big(I^{\text{mixed}}_{t,u}\big)^{11}}, \qquad \mathcal{L}_{\text{NLL}} = -\frac{1}{N}\sum_{t=1}^N \ln P(y_t \mid x_{\le t})",
                "why_built": "Track 5 (Rule 9): Computes the exact polynomial p=11 Born probability distribution with zero exp/softmax/logsumexp operations, and computes all 5 input gradients (d_phi, d_psi, d_beta, d_alpha, d_povm) inside a single vocabulary-chunk backward loop.",
                "highways": ["primary", "tied_vocab", "adamw_opt", "triton_kernels"],
                "param_tensors": [
                    _pt("povm_scale_logits", f"({DEFAULT_VOCAB_SIZE},)", povm_params, "float32", "AdamW (povm_params)", "Step-0 exact unigram calibration theta_v so P_bar_Step0 == P_target_uni", "Learnable positive POVM measurement operator weights w_{povm,v}"),
                ],
            },
        ),
    ]
    return MacroStageSpec(
        stage_id="stage_born_readout", stage_num=stage_num,
        title=f"Stage {stage_num} — Fused Triton Mixed-State 16^4 = 65,536 p=11 Boson Born Readout",
        subtitle=f"13 Vocabulary Chunks of {DEFAULT_VOCAB_CHUNK_SIZE} ➔ Batched BMM ➔ 4-Factor Product Tree ➔ p=11 Born NLL",
        total_params=povm_params, pct_of_model=round(100.0 * povm_params / float(max(1, total_params)), 2),
        tensor_ribbon=f"phi_factors ({n_tok}, {r_num}, {m_num}, {d_2}) × psi_factors ({DEFAULT_VALID_VOCAB_SIZE}, {r_num}, {m_num}, {d_2}) ➔ Scalar Born NLL",
        sub_blocks=sub_blocks,
    )


def _build_signal_highways(num_layers: int) -> list[SignalHighwaySpec]:
    """Build the interactive signal & skip highway definitions across all blocks (`<= 8` args)."""
    layer_all, layer_muon, layer_triton = [], [], []
    layer_anchor = ["s0_tokens", "s0_tied_embed", "s0_causal_smear", "l0_qkv_rope"]
    for l_idx in range(num_layers):
        layer_all.extend([f"l{l_idx}_qkv_rope", f"l{l_idx}_born_attn", f"l{l_idx}_gram_stinespring", f"l{l_idx}_unitary_ffn", f"l{l_idx}_kron_mixer"])
        layer_muon.extend([f"l{l_idx}_qkv_rope", f"l{l_idx}_gram_stinespring", f"l{l_idx}_unitary_ffn", f"l{l_idx}_kron_mixer"])
        layer_triton.extend([f"l{l_idx}_qkv_rope", f"l{l_idx}_born_attn", f"l{l_idx}_gram_stinespring", f"l{l_idx}_kron_mixer"])
        if l_idx > 0:
            layer_anchor.append(f"l{l_idx}_qkv_rope")

    return [
        SignalHighwaySpec("all", "🌐 All Strung-Together Blocks", "#38bdf8", "Complete end-to-end quantum circuit from input token IDs to chunked p=11 Boson Born NLL.", ["s0_tokens", "s0_tied_embed", "s0_causal_smear", *layer_all, "s_prep_out_u", "s_prep_self_echo", "s_prep_ngram", "s_prep_router", "s_readout_bmm", "s_readout_triton_tree", "s_readout_born_nll"], ["Every intermediate state lives on a normalized complex sphere or trace-1 Hermitian density manifold.", "Zero exp(), softmax(), or logsumexp() calls anywhere in attention, routing, or readout."]),
        SignalHighwaySpec("anchor_psi0", "⚓ Layer-0 Anchor Skip Highway (|ψ_0⟩ & V^(0))", "#f59e0b", "Traces how the un-smeared input token wavefunction |ψ_0⟩ and Layer-0 value sub-heads V^(0) bypass intermediate layers to anchor four distinct downstream blocks.", [*layer_anchor, "s_prep_out_u", "s_prep_self_echo", "s_prep_router"], ["Skip 1 (V^(0) Value Residual): Layer 0 exports raw value kets V^(0) to Layer 1..L-1 via v0_subhead_mix_logit.", "Skip 2 (Conjugate Readout Skip): Stage 3.1 blends |ψ^(L)⟩ + tanh(psi0_readout_gate) |ψ_0*⟩ before out_factor_u.", "Skip 3 (Track 3 H_perp Self-Echo): Stage 3.2 projects |ψ_0⟩ onto 15D H_perp and subtracts <u_{0,perp}|op>_C.", "Skip 4 (Track 4 Phase Router): Stage 3.4 computes relative Hermitian phase shift phi * conj(psi_0) for branch routing."]),
        SignalHighwaySpec("ngram_hperp", "📚 900M-Token N-Gram H_perp Highway", "#a855f7", "Traces how input token indices (x_{t-1}, x_t) index directly into the 900M-token calibrated Bigram (8,192 slots) and Hashed Trigram (16,384 buckets) carrier-orthogonal tables.", ["s0_tokens", "s_prep_ngram", "s_prep_router", "s_readout_bmm", "s_readout_triton_tree"], ["Bigram & Trigram tables live strictly inside the 15D zero-sum subspace H_perp subset C^16 (<c_ref | ng> == 0).", "Rotated by branch-specific unitaries bi_factor_u and tri_factor_u and weighted by ngram_branch_gates = [1.45, 0.90, 0.24]."]),
        SignalHighwaySpec("tied_vocab", "🔁 Tied 16^4 Wavefunction Loop (Input ↔ Readout)", "#10b981", "Highlights how embed_re / embed_im (50304, R, 4, 16) serves both as the Stage 0 input token lookup table and as the 16^4 = 65,536 output Born measurement kets in Stage 4.", ["s0_tied_embed", "s_readout_bmm", "s_readout_triton_tree", "s_readout_born_nll"], ["12,877,824 parameters (76.4% of w44_compact_r2_d128c_L2_b32) are shared between Stage 0.2 and Stage 4.1.", "Each token v has R=2 meaning branches of M=4 factors in C^16, spanning C^(16^4) = C^65,536 without a dense 50304x65536 matrix."]),
        SignalHighwaySpec("stiefel_muon", "🧭 Joint-Complex Polar Express StiefelMuon Routing", "#ec4899", "Highlights all blocks containing paired (weight_re, weight_im) complex unitary matrices optimized on U(n) by Joint-Complex Polar Express StiefelMuon.", ["s0_causal_smear", *layer_muon, "s_prep_out_u", "s_prep_ngram"], ["Track 1: Joint-Complex Polar Express orthogonalizes M = G_re + i G_im via complex A = M M^H in C^(m x m), preserving U(1) phase equivariance.", "Covers smear_u, q/k/v/o_proj, stinespring_w, FFN up_a/up_b/down, Kronecker mixer U(16)/U(4)/U(2), out_factor_u, and bi/tri_factor_u."]),
        SignalHighwaySpec("triton_kernels", "⚡ Fused Triton & Compiled CUDA Kernels", "#06b6d4", "Highlights the 6 custom Triton / torch.compile fused GPU kernels that accelerate w44_compact_r2_d128c_L2_b32 to >20,200 tok/s on RTX 5060 Ti.", [*layer_triton, "s_readout_triton_tree", "s_readout_born_nll"], ["1. fused_complex_norm_rope_triton: Fused C^16 L2 normalization + complex rotary phase.", "2. flash_causal_pboson_complex_attn_triton: 64x64 SRAM-tiled causal p=2 Born attention.", "3. fused_cross_head_gram_stinespring_step: Compiled trace-1 KxK Gram + Stinespring projection.", "4. fused_kronecker_16x4_unitary_mixer_step: Compiled 3-stage U(16)->U(4)->U(R) Kronecker mixer.", "5 & 6. fused_mixed_state_kronecker_chunk_{fwd,bwd}_triton: Fused 16^4 binary product tree + single-pass backward."]),
    ]


def build_w44_preset_graph(
    preset_key: str = DEFAULT_PRESET_KEY, num_meanings: int = DEFAULT_NUM_MEANINGS,
    num_layers: int = DEFAULT_NUM_LAYERS_COMPACT, batch_size: int = DEFAULT_BATCH_SIZE, seq_len: int = DEFAULT_SEQ_LEN,
) -> ModelBlockGraphSpec:
    """Build a complete `ModelBlockGraphSpec` for a `MultiHeadMixedStateKroneckerGPT` configuration (`<= 8` args)."""
    m_num, d_f, k_sub = DEFAULT_NUM_FACTORS, DEFAULT_FACTOR_DIM, DEFAULT_SUBHEADS_PER_FACTOR
    d_c = num_meanings * m_num * d_f
    s0_d = _build_stage0_embedding_and_smearing(batch_size, seq_len, num_meanings, m_num, d_f, 1)
    l0_d = _build_transformer_layer_stage(0, batch_size, seq_len, num_meanings, m_num, d_f, k_sub, 1)
    l1_d = _build_transformer_layer_stage(1, batch_size, seq_len, num_meanings, m_num, d_f, k_sub, 1)
    pr_d = _build_stage_readout_prep(num_layers + 1, batch_size, seq_len, num_meanings, m_num, d_f, 1)
    rd_d = _build_stage_born_readout(num_layers + 2, batch_size, seq_len, num_meanings, m_num, d_f, 1)
    total_params = s0_d.total_params + l0_d.total_params + max(0, num_layers - 1) * l1_d.total_params + pr_d.total_params + rd_d.total_params

    stages = [
        _build_stage0_embedding_and_smearing(batch_size, seq_len, num_meanings, m_num, d_f, total_params),
        *[_build_transformer_layer_stage(l_idx, batch_size, seq_len, num_meanings, m_num, d_f, k_sub, total_params) for l_idx in range(num_layers)],
        _build_stage_readout_prep(num_layers + 1, batch_size, seq_len, num_meanings, m_num, d_f, total_params),
        _build_stage_born_readout(num_layers + 2, batch_size, seq_len, num_meanings, m_num, d_f, total_params),
    ]
    title_map = {
        "w44_compact_r2_d128c_L2_b32": "w44_compact_r2_d128c_L2_b32 — Compact 2-Layer Rank-2 Multi-Head Mixed-State Kronecker GPT",
        "w44_compact_r1_d64c_L2_b32": "w44_compact_r1_d64c_L2_b32 — Compact 2-Layer Rank-1 (Single-Meaning) Kronecker GPT",
        "w44_flagship_r2_d128c_L6_b32": "w44_flagship_r2_d128c_L6_b32 — 6-Layer Flagship Multi-Head Mixed-State Kronecker GPT",
    }
    return ModelBlockGraphSpec(
        preset_key=preset_key, title=title_map.get(preset_key, preset_key), model_class="MultiHeadMixedStateKroneckerGPT",
        summary_badge=f"R={num_meanings} Meanings × M={m_num} Factors of C^{d_f} (C^65,536) • d_complex={d_c} ({2 * d_c}D real) • L={num_layers} Layers • K={k_sub} Sub-Heads/Factor • B={batch_size}, T={seq_len} ({batch_size * seq_len:,} tok/step) • {total_params / 1e6:.2f}M Params",
        total_params=total_params,
        hyperparams={
            "num_meanings_R": num_meanings, "num_factors_M": m_num, "d_register_df": d_f,
            "kronecker_hilbert_dim": f"16^4 = {KRONECKER_HILBERT_DIM:,} complex dims",
            "d_complex_backbone": d_c, "real_equiv_dim": 2 * d_c, "num_layers_L": num_layers,
            "subheads_per_factor_K": k_sub, "total_subheads_H": m_num * k_sub,
            "batch_size_B": batch_size, "seq_len_T": seq_len, "tokens_per_step_N": batch_size * seq_len,
            "attn_evolution_mode": "multihead_gram (3 T^2 GEMMs + KxK Gram + Stinespring)",
            "self_echo_mode": "ortho_complex_learned (15D H_perp)", "secondary_branch_scale": 1.0,
            "branch_router_mode": "phase_sensitive", "readout_density_mode": f"mixed_density (p={int(DEFAULT_BOSON_P)} Boson Born)",
        },
        stages=stages, highways=_build_signal_highways(num_layers),
    )


def build_w44_architecture_presets() -> dict[str, ModelBlockGraphSpec]:
    """Return dictionary of pre-built `ModelBlockGraphSpec` presets for the WebUI (`<= 8` args)."""
    return {
        "w44_compact_r2_d128c_L2_b32": build_w44_preset_graph("w44_compact_r2_d128c_L2_b32", num_meanings=2, num_layers=2, batch_size=32, seq_len=512),
        "w44_compact_r1_d64c_L2_b32": build_w44_preset_graph("w44_compact_r1_d64c_L2_b32", num_meanings=1, num_layers=2, batch_size=32, seq_len=512),
        "w44_flagship_r2_d128c_L6_b32": build_w44_preset_graph("w44_flagship_r2_d128c_L6_b32", num_meanings=2, num_layers=6, batch_size=32, seq_len=512),
    }


def load_w44_empirical_telemetry_summary(live_logs_dir: Path) -> dict[str, Any]:
    """Load real empirical LR-sweep and 1,200-step convergence telemetry for `w44_compact_r2_d128c_L2_b32` (`<= 8` args)."""
    lr_sweep_keys = [
        ("0.15x", "w44_compact_r2_l2_b32_constant_lr_0p15x"), ("0.25x", "w44_compact_r2_l2_b32_constant_lr_0p25x"),
        ("0.35x", "w44_compact_r2_l2_b32_constant_lr_0p35x"), ("0.50x", "w44_compact_r2_l2_b32_constant_lr_0p5x"),
        ("0.70x", "w44_compact_r2_l2_b32_constant_lr_0p7x"), ("1.00x", "w44_compact_r2_l2_b32_constant_lr_1p0x"),
        ("1.50x", "w44_compact_r2_l2_b32_constant_lr_1p5x"), ("2.00x", "w44_compact_r2_l2_b32_constant_lr_2p0x"),
        ("3.00x", "w44_compact_r2_l2_b32_constant_lr_3p0x"),
    ]
    lr_sweep_rows: list[dict[str, Any]] = []
    for mult_lbl, rk in lr_sweep_keys:
        p = live_logs_dir / f"{rk}_results.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                lr_sweep_rows.append({"lr_mult": mult_lbl, "run_key": rk, "best_val_ce": d.get("best_val_ce"), "final_val_ce": d.get("final_val_ce"), "top1_acc": d.get("final_top1_acc"), "tok_per_s": d.get("avg_tok_per_s"), "steps": d.get("current_step", d.get("total_steps", 300))})
            except Exception:
                pass

    flagship_runs: list[dict[str, Any]] = []
    for rk in ("w44_compact_r2_d128c_L2_b32_const_lr_1200steps", "w44_compact_r2_d128c_L2_b32_converge", "w44_compact_r1_d64c_L2_b32_converge", "w44_flagship_fused_kernels_b32_r2_d128c_L6_600s", "w45_compiled_classical_l12_d768_b65k_3hr_sub3p27"):
        p = live_logs_dir / f"{rk}_results.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                evals = d.get("evals", [])
                flagship_runs.append({"run_key": rk, "arch_summary": d.get("architecture_summary", ""), "params_M": d.get("params_M"), "current_step": d.get("current_step"), "total_steps": d.get("total_steps"), "live_tokens_M": d.get("live_tokens_M"), "best_val_ce": d.get("best_val_ce"), "final_val_ce": d.get("final_val_ce"), "top1_acc": d.get("final_top1_acc") or (evals[-1].get("top1_acc") if evals else None), "avg_tok_per_s": d.get("avg_tok_per_s"), "run_state": d.get("run_state", "COMPLETED")})
            except Exception:
                pass
    return {"lr_sweep_300steps": lr_sweep_rows, "comparison_runs": flagship_runs}


def build_architecture_visualizer_payload(live_logs_dir: Path | None = None) -> dict[str, Any]:
    """Assemble full JSON payload for the WebUI Block Architecture Visualizer (`<= 8` args)."""
    presets = build_w44_architecture_presets()
    resolved_logs = live_logs_dir if live_logs_dir is not None else Path(".cache/live_logs")
    return {"default_preset": DEFAULT_PRESET_KEY, "presets": {k: asdict(v) for k, v in presets.items()}, "telemetry": load_w44_empirical_telemetry_summary(resolved_logs)}
