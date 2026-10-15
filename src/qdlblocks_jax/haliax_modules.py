"""Haliax named-axis quantum-geometric modules and HaxQuantumBornGPT."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax._src.numpy.einsum as _je
import jax._src.numpy.lax_numpy as _lax_numpy
import jax._src.tree_util as _jtu
import jax.numpy as jnp

if not hasattr(_jtu, "BuiltInKeyEntry"):
    _jtu.BuiltInKeyEntry = getattr(_jtu, "KeyEntry", object)  # type: ignore[attr-defined]
if not hasattr(jax.core, "is_constant_dim"):
    setattr(jax.core, "is_constant_dim", lambda d: isinstance(d, int))
for _attr in ("_einsum", "_poly_einsum_handlers", "_default_poly_einsum_handler"):
    if not hasattr(_lax_numpy, _attr) and hasattr(_je, _attr):
        setattr(_lax_numpy, _attr, getattr(_je, _attr))

import haliax as hax

from .functional import apply_unitary_rope, l2_normalize, precompute_rope_cos_sin
from .pallas_flash_born import flash_born_attention_jax

__all__ = [
    "hax_l2_normalize",
    "orthogonal_init",
    "HaxIsometricHilbertLift",
    "HaxSubspaceBornAttention",
    "HaxQuantumBornMoE",
    "HaxQuantumBornBlock",
    "HaxProjectivePOVMHead",
    "HaxQuantumBornGPT",
]


def hax_l2_normalize(
    x: hax.NamedArray,
    axis: hax.AxisSpec,
    eps: float = 1e-12,
) -> hax.NamedArray:
    """Project a Haliax NamedArray onto the unit hypersphere along `axis`."""
    sq = hax.sum(x * x, axis=axis)
    return x / hax.sqrt(hax.maximum(sq, eps * eps))


def _qr_stiefel_2d(mat_2d: jax.Array) -> jax.Array:
    if mat_2d.shape[0] >= mat_2d.shape[1]:
        q, _ = jnp.linalg.qr(mat_2d)
        return q
    q_t, _ = jnp.linalg.qr(mat_2d.T)
    return q_t.T


def orthogonal_init(
    axes: tuple[hax.Axis, ...],
    key: jax.Array,
) -> hax.NamedArray:
    """Initialize a 2D or 3D Haliax NamedArray on the Stiefel manifold via QR decomposition."""
    shape = tuple(ax.size for ax in axes)
    raw = jax.random.normal(key, shape, dtype=jnp.float32)
    if len(axes) == 2:
        q = _qr_stiefel_2d(raw)
    elif len(axes) == 3:
        if shape[1] * shape[2] == shape[0]:
            q = _qr_stiefel_2d(raw.reshape(shape[0], -1)).reshape(shape)
        elif shape[0] * shape[1] == shape[2]:
            q = _qr_stiefel_2d(raw.reshape(-1, shape[2])).reshape(shape)
        else:
            q = jax.vmap(_qr_stiefel_2d)(raw)
    else:
        raise ValueError(f"orthogonal_init expects 2 or 3 axes, got {axes}.")
    return hax.NamedArray(q, axes)


class HaxIsometricHilbertLift(eqx.Module):
    """Two-stage isometric Hilbert-space token lift with zero-mean Riemannian centering."""

    Vocab: hax.Axis = eqx.field(static=True)
    Seed: hax.Axis = eqx.field(static=True)
    Hilbert: hax.Axis = eqx.field(static=True)
    seed_embed: hax.NamedArray
    w_lift: hax.NamedArray

    def __init__(
        self,
        Vocab: hax.Axis,
        Seed: hax.Axis,
        Hilbert: hax.Axis,
        *,
        key: jax.Array,
    ):
        self.Vocab = Vocab
        self.Seed = Seed
        self.Hilbert = Hilbert
        k_seed, k_lift = jax.random.split(key)
        raw_seed = jax.random.normal(k_seed, (Vocab.size, Seed.size), dtype=jnp.float32)
        centered = raw_seed - jnp.mean(raw_seed, axis=-1, keepdims=True)
        normed = l2_normalize(centered, axis=-1)
        self.seed_embed = hax.NamedArray(normed, (Vocab, Seed))
        self.w_lift = orthogonal_init((Hilbert, Seed), k_lift)

    def __call__(self, token_ids: hax.NamedArray) -> hax.NamedArray:
        s_raw = self.seed_embed.take(self.Vocab, token_ids)
        s_centered = s_raw - hax.mean(s_raw, axis=self.Seed)
        s_norm = hax_l2_normalize(s_centered, axis=self.Seed)
        psi = hax.dot(s_norm, self.w_lift, axis=self.Seed)
        return hax_l2_normalize(psi, axis=self.Hilbert)


class HaxSubspaceBornAttention(eqx.Module):
    """Named-axis multi-head subspace Born attention with unitary RoPE and value-state mixing."""

    Hilbert: hax.Axis = eqx.field(static=True)
    Heads: hax.Axis = eqx.field(static=True)
    HeadDim: hax.Axis = eqx.field(static=True)
    is_causal: bool = eqx.field(static=True)
    use_flash_born: bool = eqx.field(static=True)
    w_q: hax.NamedArray
    w_k: hax.NamedArray
    w_v: hax.NamedArray
    w_o: hax.NamedArray
    v_mix_logit: jax.Array

    def __init__(
        self,
        Hilbert: hax.Axis,
        Heads: hax.Axis,
        HeadDim: hax.Axis,
        *,
        is_causal: bool = True,
        use_flash_born: bool = True,
        key: jax.Array,
    ):
        self.Hilbert = Hilbert
        self.Heads = Heads
        self.HeadDim = HeadDim
        self.is_causal = is_causal
        self.use_flash_born = use_flash_born
        k_q, k_k, k_v, k_o = jax.random.split(key, 4)
        self.w_q = orthogonal_init((Hilbert, Heads, HeadDim), k_q)
        self.w_k = orthogonal_init((Hilbert, Heads, HeadDim), k_k)
        self.w_v = orthogonal_init((Hilbert, Heads, HeadDim), k_v)
        self.w_o = orthogonal_init((Heads, HeadDim, Hilbert), k_o)
        self.v_mix_logit = jnp.array(0.0, dtype=jnp.float32)

    def __call__(self, x: hax.NamedArray, Seq: hax.Axis) -> hax.NamedArray:
        q = hax_l2_normalize(hax.dot(x, self.w_q, axis=self.Hilbert), axis=self.HeadDim)
        k = hax_l2_normalize(hax.dot(x, self.w_k, axis=self.Hilbert), axis=self.HeadDim)
        v_proj = hax.dot(x, self.w_v, axis=self.Hilbert)

        alpha_v = hax.NamedArray(jax.nn.sigmoid(self.v_mix_logit), ())
        one_minus_alpha = hax.NamedArray(1.0 - jax.nn.sigmoid(self.v_mix_logit), ())
        x_heads = x.unflatten_axis(self.Hilbert, (self.Heads, self.HeadDim))
        v = hax_l2_normalize(
            alpha_v * v_proj + one_minus_alpha * x_heads,
            axis=self.HeadDim,
        )

        # Rearrange to (..., Heads, Seq, HeadDim) for RoPE and Flash-Born kernel
        leading_axes = tuple(
            ax for ax in q.axes if ax not in (self.Heads, Seq, self.HeadDim)
        )
        ordered_axes = leading_axes + (self.Heads, Seq, self.HeadDim)
        q_ord = q.rearrange(ordered_axes).array
        k_ord = k.rearrange(ordered_axes).array
        v_ord = v.rearrange(ordered_axes).array

        orig_leading_shape = q_ord.shape[:-3]
        B_flat = math.prod(orig_leading_shape) if len(orig_leading_shape) > 0 else 1
        H_sz, T_sz, D_sz = self.Heads.size, Seq.size, self.HeadDim.size

        q_4d = q_ord.reshape(B_flat, H_sz, T_sz, D_sz)
        k_4d = k_ord.reshape(B_flat, H_sz, T_sz, D_sz)
        v_4d = v_ord.reshape(B_flat, H_sz, T_sz, D_sz)

        cos, sin = precompute_rope_cos_sin(T_sz, D_sz, dtype=q_4d.dtype)
        q_4d = apply_unitary_rope(q_4d, cos, sin)
        k_4d = apply_unitary_rope(k_4d, cos, sin)

        scale = math.sqrt(float(D_sz))
        if self.use_flash_born:
            attn_4d = flash_born_attention_jax(
                q_4d,
                k_4d,
                v_4d,
                scale=scale,
                is_causal=self.is_causal,
                eps=1e-6,
            )
        else:
            scores = (
                jnp.einsum(
                    "bhid,bhjd->bhij",
                    q_4d,
                    k_4d,
                    precision=jax.lax.Precision.HIGHEST,
                )
                * scale
            )
            if self.is_causal:
                idx = jnp.arange(T_sz)
                mask = idx[:, None] >= idx[None, :]
                scores = jnp.where(mask[None, None, :, :], scores, -1.0)
            p = jnp.maximum(scores, 0.0) ** 2
            w = p / (jnp.sum(p, axis=-1, keepdims=True) + 1e-6)
            attn_4d = jnp.einsum(
                "bhij,bhjd->bhid",
                w,
                v_4d,
                precision=jax.lax.Precision.HIGHEST,
            )

        attn_arr = attn_4d.reshape(orig_leading_shape + (H_sz, T_sz, D_sz))
        attn_named = hax.NamedArray(attn_arr, ordered_axes).rearrange(q.axes)
        attn_norm = hax_l2_normalize(attn_named, axis=self.HeadDim)
        out = hax.dot(attn_norm, self.w_o, axis=(self.Heads, self.HeadDim))
        return hax_l2_normalize(out, axis=self.Hilbert)


class HaxQuantumBornMoE(eqx.Module):
    """Zero-Softmax positive-cone Born Mixture of Experts with Stiefel routing axes."""

    Hilbert: hax.Axis = eqx.field(static=True)
    MlpDim: hax.Axis = eqx.field(static=True)
    Experts: hax.Axis = eqx.field(static=True)
    top_k: int = eqx.field(static=True)
    router: hax.NamedArray
    w_up: hax.NamedArray
    w_down: hax.NamedArray

    def __init__(
        self,
        Hilbert: hax.Axis,
        MlpDim: hax.Axis,
        Experts: hax.Axis,
        top_k: int = 2,
        *,
        key: jax.Array,
    ):
        self.Hilbert = Hilbert
        self.MlpDim = MlpDim
        self.Experts = Experts
        self.top_k = min(top_k, Experts.size)
        k_r, k_u, k_d = jax.random.split(key, 3)
        self.router = orthogonal_init((Hilbert, Experts), k_r)
        self.w_up = orthogonal_init((Experts, MlpDim, Hilbert), k_u)
        self.w_down = orthogonal_init((Experts, Hilbert, MlpDim), k_d)

    def __call__(self, x: hax.NamedArray) -> hax.NamedArray:
        leading_axes = tuple(ax for ax in x.axes if ax != self.Hilbert)
        ordered_axes = leading_axes + (self.Hilbert,)
        x_ord = x.rearrange(ordered_axes).array
        orig_shape = x_ord.shape
        d = self.Hilbert.size
        x_2d = x_ord.reshape(-1, d)

        # Positive-cone Born routing: A_e = max(0, <psi|r_e>)^2
        r_mat = self.router.rearrange((self.Hilbert, self.Experts)).array  # (d, E)
        proj = jnp.matmul(x_2d, r_mat, precision=jax.lax.Precision.HIGHEST)
        a_pos = jnp.maximum(proj, 0.0) ** 2  # (N_tok, E)

        top_vals, top_idx = jax.lax.top_k(a_pos, self.top_k)  # (N_tok, k)
        weights = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-6)

        w_up_arr = self.w_up.rearrange((self.Experts, self.MlpDim, self.Hilbert)).array
        w_down_arr = self.w_down.rearrange((self.Experts, self.Hilbert, self.MlpDim)).array
        scale_up = math.sqrt(2.0 / float(d))

        def eval_token(tok_vec: jax.Array, idx_k: jax.Array, w_k: jax.Array) -> jax.Array:
            up_k = w_up_arr[idx_k]  # (k, M, d)
            down_k = w_down_arr[idx_k]  # (k, d, M)
            h = jnp.einsum("kmd,d->km", up_k, tok_vec, precision=jax.lax.Precision.HIGHEST) * scale_up
            act = jnp.maximum(h, 0.0) ** 2
            y_k = jnp.einsum("kdm,km->kd", down_k, act, precision=jax.lax.Precision.HIGHEST)
            y_k = l2_normalize(y_k, axis=-1)
            combined = jnp.einsum("k,kd->d", w_k, y_k, precision=jax.lax.Precision.HIGHEST)
            return l2_normalize(combined, axis=-1)

        out_2d = jax.vmap(eval_token)(x_2d, top_idx, weights)
        out_named = hax.NamedArray(out_2d.reshape(orig_shape), ordered_axes).rearrange(x.axes)
        return out_named


class HaxQuantumBornBlock(eqx.Module):
    """Geodesic-gated Quantum Born Transformer Block in Haliax."""

    Hilbert: hax.Axis = eqx.field(static=True)
    MlpDim: hax.Axis = eqx.field(static=True)
    use_moe: bool = eqx.field(static=True)
    attn: HaxSubspaceBornAttention
    w_up: hax.NamedArray | None
    w_down: hax.NamedArray | None
    moe: HaxQuantumBornMoE | None
    alpha_attn: jax.Array
    alpha_mlp: jax.Array

    def __init__(
        self,
        Hilbert: hax.Axis,
        Heads: hax.Axis,
        HeadDim: hax.Axis,
        MlpDim: hax.Axis,
        *,
        Experts: hax.Axis | None = None,
        top_k: int = 2,
        is_causal: bool = True,
        use_flash_born: bool = True,
        key: jax.Array,
    ):
        self.Hilbert = Hilbert
        self.MlpDim = MlpDim
        k_attn, k_mlp1, k_mlp2 = jax.random.split(key, 3)
        self.attn = HaxSubspaceBornAttention(
            Hilbert,
            Heads,
            HeadDim,
            is_causal=is_causal,
            use_flash_born=use_flash_born,
            key=k_attn,
        )
        if Experts is not None and Experts.size > 1:
            self.use_moe = True
            self.moe = HaxQuantumBornMoE(
                Hilbert, MlpDim, Experts, top_k=top_k, key=k_mlp1
            )
            self.w_up = None
            self.w_down = None
        else:
            self.use_moe = False
            self.moe = None
            self.w_up = orthogonal_init((MlpDim, Hilbert), k_mlp1)
            self.w_down = orthogonal_init((Hilbert, MlpDim), k_mlp2)

        self.alpha_attn = jnp.full((Hilbert.size,), 0.1, dtype=jnp.float32)
        self.alpha_mlp = jnp.full((Hilbert.size,), 0.1, dtype=jnp.float32)

    def __call__(self, x: hax.NamedArray, Seq: hax.Axis) -> hax.NamedArray:
        a_attn = hax.NamedArray(self.alpha_attn, (self.Hilbert,))
        a_mlp = hax.NamedArray(self.alpha_mlp, (self.Hilbert,))

        attn_delta = self.attn(x, Seq=Seq)
        x = hax_l2_normalize(x + a_attn * attn_delta, axis=self.Hilbert)

        if self.use_moe and self.moe is not None:
            mlp_delta = self.moe(x)
        else:
            assert self.w_up is not None and self.w_down is not None
            h = hax.dot(x, self.w_up, axis=self.Hilbert) * math.sqrt(
                2.0 / float(self.Hilbert.size)
            )
            act = hax.maximum(h, 0.0) ** 2
            mlp_delta = hax_l2_normalize(
                hax.dot(act, self.w_down, axis=self.MlpDim),
                axis=self.Hilbert,
            )

        x = hax_l2_normalize(x + a_mlp * mlp_delta, axis=self.Hilbert)
        return x


class HaxProjectivePOVMHead(eqx.Module):
    """Rank-1 Projective POVM vocabulary head on the unit hypersphere."""

    Hilbert: hax.Axis = eqx.field(static=True)
    Vocab: hax.Axis = eqx.field(static=True)
    m_vocab: hax.NamedArray
    inv_temp: jax.Array

    def __init__(
        self,
        Hilbert: hax.Axis,
        Vocab: hax.Axis,
        *,
        temperature: float = 0.035,
        key: jax.Array,
    ):
        self.Hilbert = Hilbert
        self.Vocab = Vocab
        raw_m = jax.random.normal(key, (Vocab.size, Hilbert.size), dtype=jnp.float32)
        self.m_vocab = hax_l2_normalize(
            hax.NamedArray(raw_m, (Vocab, Hilbert)),
            axis=Hilbert,
        )
        self.inv_temp = jnp.array(1.0 / temperature, dtype=jnp.float32)

    def __call__(self, x: hax.NamedArray) -> hax.NamedArray:
        m_norm = hax_l2_normalize(self.m_vocab, axis=self.Hilbert)
        beta = hax.NamedArray(jnp.clip(self.inv_temp, 10.0, 100.0), ())
        return hax.dot(x, m_norm, axis=self.Hilbert) * beta


class HaxQuantumBornGPT(eqx.Module):
    """Named-Axis Quantum Born Autoregressive Language Model in Haliax + Equinox."""

    Vocab: hax.Axis = eqx.field(static=True)
    Seq: hax.Axis = eqx.field(static=True)
    Seed: hax.Axis = eqx.field(static=True)
    Hilbert: hax.Axis = eqx.field(static=True)
    Heads: hax.Axis = eqx.field(static=True)
    HeadDim: hax.Axis = eqx.field(static=True)
    MlpDim: hax.Axis = eqx.field(static=True)
    lift: HaxIsometricHilbertLift
    blocks: tuple[HaxQuantumBornBlock, ...]
    povm_head: HaxProjectivePOVMHead

    def __init__(
        self,
        Vocab: hax.Axis,
        Seq: hax.Axis,
        Seed: hax.Axis,
        Hilbert: hax.Axis,
        Heads: hax.Axis,
        HeadDim: hax.Axis,
        MlpDim: hax.Axis,
        num_layers: int = 4,
        *,
        Experts: hax.Axis | None = None,
        top_k_experts: int = 2,
        povm_temperature: float = 0.035,
        use_flash_born: bool = True,
        key: jax.Array,
    ):
        self.Vocab = Vocab
        self.Seq = Seq
        self.Seed = Seed
        self.Hilbert = Hilbert
        self.Heads = Heads
        self.HeadDim = HeadDim
        self.MlpDim = MlpDim

        keys = jax.random.split(key, num_layers + 2)
        self.lift = HaxIsometricHilbertLift(Vocab, Seed, Hilbert, key=keys[0])
        self.blocks = tuple(
            HaxQuantumBornBlock(
                Hilbert,
                Heads,
                HeadDim,
                MlpDim,
                Experts=Experts,
                top_k=top_k_experts,
                is_causal=True,
                use_flash_born=use_flash_born,
                key=keys[i + 1],
            )
            for i in range(num_layers)
        )
        self.povm_head = HaxProjectivePOVMHead(
            Hilbert,
            Vocab,
            temperature=povm_temperature,
            key=keys[-1],
        )

    def encode(self, token_ids: hax.NamedArray) -> hax.NamedArray:
        """Encode token IDs into normalized Hilbert-space state vectors |psi>."""
        psi = self.lift(token_ids)
        for blk in self.blocks:
            psi = blk(psi, Seq=self.Seq)
        return psi

    def factorize_into_qubits(self, psi: hax.NamedArray) -> hax.NamedArray:
        """Unflatten the Hilbert axis (size 2^n) into explicit named qubit axes (qubit_0, ..., qubit_{n-1})."""
        d = self.Hilbert.size
        n_qubits = int(round(math.log2(d)))
        if 2**n_qubits != d:
            raise ValueError(
                f"Hilbert.size must be a power of 2 to factorize into qubits, got {d}."
            )
        qubit_axes = tuple(hax.Axis(f"qubit_{i}", 2) for i in range(n_qubits))
        return psi.unflatten_axis(self.Hilbert, qubit_axes)

    def __call__(
        self,
        token_ids: hax.NamedArray,
        targets: hax.NamedArray | None = None,
    ) -> tuple[hax.NamedArray, jax.Array | None]:
        psi = self.encode(token_ids)
        logits = self.povm_head(psi)
        if targets is None:
            return logits, None

        # Cross-entropy loss over Vocab axis
        leading_axes = tuple(ax for ax in logits.axes if ax != self.Vocab)
        logits_ord = logits.rearrange(leading_axes + (self.Vocab,)).array
        targets_ord = targets.rearrange(leading_axes).array
        log_probs = jax.nn.log_softmax(logits_ord, axis=-1)
        nll = -jnp.take_along_axis(log_probs, targets_ord[..., None], axis=-1).squeeze(-1)
        loss = jnp.mean(nll)
        return logits, loss
