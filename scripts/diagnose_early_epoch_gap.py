#!/usr/bin/env python3
"""Diagnostic CLI for the Early-Epoch Validation Loss Gap between QuantumBornGPT and ClassicalModdedGPT.

Downloads/loads the genuine FineWeb validation shard (`fineweb_val_000000.bin`) via
`FineWebMultiShardStream` and tests four concrete mathematical mechanisms responsible
for the early-epoch (0–300M token) validation Cross-Entropy lag of `QuantumBornGPT`
relative to `ClassicalModdedGPT`:

1. **Mechanism 1 — The Spherical Identity-Loop / Self-Repetition Trap (`+ortho_decouple`)**:
   - In `QuantumBornGPT`, `hilbert_lift` lifts seed vector `e_{x_t}` via `W_lift`, `unet_gates`
     are initialized to `I`, `readout_rot` is initialized to `I` (`nn.init.eye_`), `collapse_to_seed`
     projects by `W_lift^T`, and `M_seed` is initialized to `wte` (`povm_head_seed_delta = 0`).
   - Because `W_lift^T @ I @ W_lift = I`, the residual highway maps the current token `x_t` directly
     onto its own output POVM detector `m_{x_t}` with cosine similarity ~ +1.0 (`logit = +28.57`),
     predicting that every token repeats itself (`x_{t+1} = x_t`)!
   - To escape this toxic self-repetition loop in early steps, `StiefelMuon` has to blow up sublayer
     weights (`IsoErr` spiking to `1.35` at 50M tokens) just to drown out the residual highway `psi_0`.
   - Fix (`+ortho_decouple`): Initialize `readout_rot` and `unet_gates` with random orthogonal Stiefel
     matrices (`nn.init.orthogonal_`) instead of `torch.eye`, breaking the `x_{t+1} = x_t` loop at Step 0.

2. **Mechanism 2 — Missing Rank-1 POVM Measure Weights & High Initial Beta (`+povm_prior`)**:
   - Unit-norm POVM rows `||m_v||_2 = 1` have zero radial freedom to represent the 5-order-of-magnitude
     Zipf token-frequency distribution without rotating 50,304 unit vectors into an anisotropic cone.
   - Fix (`+povm_prior`): General rank-1 POVM measure weights `E_v = exp(b_v) |m_v><m_v|`
     (`logits_v = beta * <phi, m_v> + b_v`) + calibrated initial `beta_0 = 14.0`.

3. **Mechanism 3 — Unbounded Sublayer Residual Overwriting (`+gated_res`)**:
   - Once the identity highway is decoupled from self-repetition (`+ortho_decouple`), learnable
     sublayer residual gates (`alpha_attn, alpha_mlp = 0.25`) prevent untrained unit-norm MLP
     frames (`||h||_2 = 1.0`) from scrambling `psi_0` by 45 degrees per layer at Step 0.

4. **Mechanism 4 — Homogeneous Born Attention Scale Cancellation & O(T/d_h) Floor (`+purified_attn`)**:
   - In `ReLU(gamma_h * <q, k>)^2 / sum`, `gamma_h^2` cancels out between numerator and denominator,
     and ~T/2 positive random keys create an `O(T / (2 d_h))` background noise floor.
   - Fix (`+purified_attn`): Coherence-thresholded degree-4 Born purification `ReLU(<q, k> - tau_h)^4`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.fineweb_stream import FineWebMultiShardStream
from qdlblocks.functional import apply_unitary_rope, l2_normalize
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params


def load_genuine_fineweb_tokens(
    num_train_tokens: int = 2_000_000,
    num_val_tokens: int = 262_144,
) -> tuple[torch.Tensor, torch.Tensor]:
    stream = FineWebMultiShardStream(
        data_dir=REPO_ROOT / ".cache" / "fineweb10B",
        num_shards=1,
        prefetch_async=False,
    )
    val_np = stream.ensure_val_shard()
    train_slice = torch.from_numpy(np.array(val_np[:num_train_tokens], dtype=np.int64))
    val_slice = torch.from_numpy(
        np.array(val_np[num_train_tokens : num_train_tokens + num_val_tokens], dtype=np.int64)
    )
    return train_slice, val_slice


class EarlyEpochQuantumAblationWrapper(nn.Module):
    """Wraps QuantumBornGPT to test incremental fixes for the early-epoch validation gap."""

    def __init__(
        self,
        base_model: QuantumBornGPT,
        use_ortho_decouple: bool = False,
        use_povm_prior: bool = False,
        use_gated_res: bool = False,
        use_purified_attn: bool = False,
    ) -> None:
        super().__init__()
        self.m = base_model
        self.use_ortho_decouple = use_ortho_decouple
        self.use_povm_prior = use_povm_prior
        self.use_gated_res = use_gated_res
        self.use_purified_attn = use_purified_attn

        if self.use_ortho_decouple:
            with torch.no_grad():
                if self.m.readout_rot is not None:
                    nn.init.orthogonal_(self.m.readout_rot.weight)
                if self.m.unet_gates is not None:
                    for ug in self.m.unet_gates:
                        nn.init.orthogonal_(ug)

        if self.use_povm_prior:
            self.povm_log_prior = nn.Parameter(torch.zeros(self.m.vocab_size))
            if self.m.tied_inv_temp is not None:
                with torch.no_grad():
                    self.m.tied_inv_temp.fill_(14.0)
        else:
            self.register_parameter("povm_log_prior", None)

        if self.use_gated_res:
            self.attn_res_gates = nn.Parameter(torch.full((self.m.num_layers,), 0.25))
            self.mlp_res_gates = nn.Parameter(torch.full((self.m.num_layers,), 0.25))
        else:
            self.register_parameter("attn_res_gates", None)
            self.register_parameter("mlp_res_gates", None)

        if self.use_purified_attn:
            init_tau = 0.5 / math.sqrt(float(self.m.head_dim))
            self.attn_coherence_tau = nn.Parameter(
                torch.full((self.m.num_layers, 1, self.m.num_heads, 1, 1), init_tau)
            )
        else:
            self.register_parameter("attn_coherence_tau", None)

    def _purified_attn_forward(
        self,
        layer_idx: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        psi_0: torch.Tensor,
    ) -> torch.Tensor:
        attn = self.m.blocks[layer_idx].attn
        B, T, D = x.shape
        psi = l2_normalize(x, dim=-1, eps=attn.eps)
        Q = attn.q_proj(psi).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)
        K = attn.k_proj(psi).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)
        V = attn.v_proj(psi).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)

        if attn.value_residual_mixing and attn.v_mix_logit is not None:
            V_0 = attn.v_proj(l2_normalize(psi_0, dim=-1, eps=attn.eps))
            V_0 = V_0.view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)
            alpha = torch.sigmoid(attn.v_mix_logit).to(V.dtype)
            V = (1.0 - alpha) * V + alpha * V_0

        Q = apply_unitary_rope(Q, cos, sin)
        K = apply_unitary_rope(K, cos, sin)
        Q = l2_normalize(Q, dim=-1, eps=attn.eps)
        K = l2_normalize(K, dim=-1, eps=attn.eps)

        raw_overlap = torch.matmul(Q, K.transpose(-1, -2))
        assert self.attn_coherence_tau is not None
        tau = torch.clamp(self.attn_coherence_tau[layer_idx], min=0.0, max=0.35).to(raw_overlap.dtype)
        excess = F.relu(raw_overlap - tau)
        prob = excess.square().square() + 0.05 * F.relu(raw_overlap).square()
        mask = attn.causal_mask[:, :, :T, :T]
        prob = prob * mask
        born_weights = prob / (torch.sum(prob, dim=-1, keepdim=True) + attn.eps)
        out_heads = torch.matmul(born_weights, V)
        out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, D)
        return attn.out_proj(out_concat)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        psi_0 = self.m.hilbert_lift(idx)
        if self.m.use_bigram_smear and T > 1 and self.m.smear_embed_gate is not None:
            psi_prev = F.pad(psi_0[:, :-1, :], (0, 0, 1, 0))
            eta = (0.06 * torch.tanh(self.m.smear_embed_gate)).to(psi_0.dtype)
            psi_0 = l2_normalize(psi_0 + eta * psi_prev, dim=-1, eps=self.m.eps)

        psi = psi_0
        encoder_states: list[torch.Tensor] = []
        n_half = self.m.num_layers // 2
        cos, sin = self.m.rope.get_cos_sin(T)

        for l_idx, block in enumerate(self.m.blocks):
            if block.input_highway_mixing and block.highway_logit is not None:
                gamma = torch.sigmoid(block.highway_logit).to(psi.dtype)
                psi = l2_normalize((1.0 - gamma) * psi + gamma * psi_0, dim=-1, eps=self.m.eps)

            if self.use_purified_attn:
                a = self._purified_attn_forward(l_idx, psi, cos, sin, psi_0)
            else:
                a = block.attn(psi, cos=cos, sin=sin, psi_0=psi_0)

            g_a = (
                self.attn_res_gates[l_idx].to(psi.dtype)
                if self.attn_res_gates is not None
                else block.residual_scale
            )
            psi = l2_normalize(psi + g_a * a, dim=-1, eps=self.m.eps)

            h = block._channel_forward(psi)
            g_m = (
                self.mlp_res_gates[l_idx].to(psi.dtype)
                if self.mlp_res_gates is not None
                else block.residual_scale
            )
            psi = l2_normalize(psi + g_m * h, dim=-1, eps=self.m.eps)

            if self.m.use_unet_interferometry and self.m.unet_gates is not None and self.m.unet_scales is not None:
                if l_idx < n_half:
                    encoder_states.append(psi)
                elif l_idx >= self.m.num_layers - n_half:
                    enc_idx = self.m.num_layers - 1 - l_idx
                    skip_psi = encoder_states[enc_idx]
                    u_scale = (0.35 * torch.tanh(self.m.unet_scales[enc_idx])).to(psi.dtype)
                    skip_rot = F.linear(skip_psi, self.m.unet_gates[enc_idx].to(psi.dtype))
                    psi = l2_normalize(
                        psi + u_scale * l2_normalize(skip_rot, dim=-1, eps=self.m.eps),
                        dim=-1,
                        eps=self.m.eps,
                    )

        assert self.m.readout_rot is not None and self.m.tied_inv_temp is not None
        psi_final = l2_normalize(self.m.readout_rot(psi), dim=-1, eps=self.m.eps)
        phi_seed = self.m.hilbert_lift.collapse_to_seed(psi_final)
        if self.m.seed_adapter is not None:
            phi_seed = self.m.seed_adapter(psi_final, phi_seed)

        if self.m.povm_head_seed_delta is not None:
            M_seed = l2_normalize(self.m.wte.weight + self.m.povm_head_seed_delta, dim=1, eps=self.m.eps).to(phi_seed.dtype)
        else:
            M_seed = l2_normalize(self.m.wte.weight, dim=1, eps=self.m.eps).to(phi_seed.dtype)

        inv_t = torch.clamp(self.m.tied_inv_temp, min=1.0, max=60.0)
        if self.m.purity_gate is not None:
            inv_t = self.m.purity_gate(psi, inv_t)

        logits = F.linear(phi_seed, M_seed) * inv_t.to(phi_seed.dtype)
        if self.povm_log_prior is not None:
            logits = logits + self.povm_log_prior.to(logits.dtype)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    stride = batch_size * seq_len
    n_batches = min(8, (len(val_tokens) - 1) // stride)
    losses = []
    top1_hits = 0
    total_toks = 0
    self_rep_probs = []
    for i in range(n_batches):
        chunk = val_tokens[i * stride : (i + 1) * stride + 1].to(device)
        x = chunk[:-1].reshape(batch_size, seq_len)
        y = chunk[1:].reshape(batch_size, seq_len)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits, loss = model(x, targets=y)
            if i == 0:
                probs = F.softmax(logits.float(), dim=-1)
                p_self = probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
                self_rep_probs.append(float(p_self.mean().item()))
        losses.append(float(loss.item()))
        top1_hits += int((logits.argmax(dim=-1) == y).sum().item())
        total_toks += int(y.numel())
    model.train()
    return {
        "val_ce": round(float(np.mean(losses)), 4),
        "top1_acc": round(100.0 * top1_hits / max(1, total_toks), 2),
        "self_repeat_prob": round(100.0 * float(np.mean(self_rep_probs)), 2) if self_rep_probs else 0.0,
    }


def run_ablation(
    name: str,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    steps: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(42)
    vocab_size = 50304
    embed_dim = 384
    num_heads = 6
    num_layers = 4
    hidden_dim = 768
    seed_dim = 192

    if name == "classical_modded":
        model: nn.Module = ClassicalModdedGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            block_size=seq_len,
        ).to(device)
    else:
        q_base = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=seed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            block_size=seq_len,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            value_residual_mixing=True,
            input_highway_mixing=True,
            use_bigram_smear=True,
            use_unet_interferometry=True,
            use_two_beam_mlp=True,
            use_overcomplete_two_beam=True,
            frame_expansion=2.0,
            use_two_beam_seed_adapter=True,
            use_dynamic_povm_beta=True,
            use_flash_born=False,
        ).to(device)
        model = EarlyEpochQuantumAblationWrapper(
            q_base,
            use_ortho_decouple=("ortho" in name or "all_fixes" in name),
            use_povm_prior=("povm_prior" in name or "all_fixes" in name),
            use_gated_res=("gated_res" in name or "all_fixes" in name),
            use_purified_attn=("purified_attn" in name or "all_fixes" in name),
        ).to(device)

    from qdlblocks.optim.schedules import cosine_lr_multiplier

    use_muon_for_mlp = ("muon_mlp" in name) or ("all_fixes" in name)
    adamw_pats = (
        "wte",
        "povm_head",
        "lm_head",
        "purity_gate",
        "highway",
        "v_mix",
        "embed",
        "unet_scales",
        "smear",
        "inv_temp",
        "kraus_readout" if use_muon_for_mlp else "beam_",
        "seed_gate",
        "povm_log_prior",
        "res_gates",
        "coherence_tau",
    )
    muon_params, adamw_params = split_muon_adamw_params(
        model,
        adamw_name_patterns=adamw_pats,
    )
    base_muon_lr = 0.025
    base_adamw_lr = 0.004
    opt_muon = StiefelMuon(
        muon_params,
        lr=base_muon_lr,
        momentum=0.95,
        weight_decay=0.01,
        decay_mode="cautious",
        stiefel_penalty=0.0 if name == "classical_modded" else 0.005,
        retract_every=1_000_000 if name == "classical_modded" else 4,
    )
    opt_adamw = torch.optim.AdamW(adamw_params, lr=base_adamw_lr, betas=(0.9, 0.95), weight_decay=0.01)

    stride = batch_size * seq_len
    max_batches = (len(train_tokens) - 1) // stride
    checkpoints: list[dict[str, float]] = []

    val0 = evaluate_val(model, val_tokens, batch_size, seq_len, device)
    checkpoints.append({"step": 0, **val0})

    warmup_steps = max(1, steps // 6)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        lr_mult = cosine_lr_multiplier(step, steps, warmup_steps=warmup_steps, min_lr_ratio=0.15)
        for g in opt_muon.param_groups:
            g["lr"] = base_muon_lr * lr_mult
        for g in opt_adamw.param_groups:
            g["lr"] = base_adamw_lr * lr_mult

        b_idx = (step - 1) % max_batches
        chunk = train_tokens[b_idx * stride : (b_idx + 1) * stride + 1].to(device)
        x = chunk[:-1].reshape(batch_size, seq_len)
        y = chunk[1:].reshape(batch_size, seq_len)

        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss = model(x, targets=y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adamw.step()

        if step in (50, 100, 200, steps):
            vk = evaluate_val(model, val_tokens, batch_size, seq_len, device)
            checkpoints.append({"step": step, **vk})

    return {
        "name": name,
        "muon_params": sum(p.numel() for p in muon_params),
        "adamw_params": sum(p.numel() for p in adamw_params),
        "checkpoints": checkpoints,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose early-epoch validation CE gap on genuine FineWeb tokens.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--out-json", type=Path, default=REPO_ROOT / ".cache" / "early_epoch_gap_diagnosis.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tokens, val_tokens = load_genuine_fineweb_tokens()

    variants = [
        "quantum_baseline",
        "quantum_+muon_mlp",
        "quantum_+ortho+povm_prior",
        "quantum_+ortho+gated_res",
        "quantum_+all_fixes",
        "classical_modded",
    ]

    print(
        f"{'Variant':<28} | {'Step 0 CE':<10} | {'SelfRep0':<9} | {'Step 50 CE':<10} | "
        f"{'Step 100 CE':<11} | {'Step 300 CE':<11} | {'Top-1'}"
    )
    print("-" * 104)
    results = []
    for var in variants:
        res = run_ablation(var, train_tokens, val_tokens, args.steps, args.batch_size, args.seq_len, device)
        results.append(res)
        ck = {c["step"]: c for c in res["checkpoints"]}
        print(
            f"{var:<28} | {ck[0]['val_ce']:<10.4f} | {ck[0]['self_repeat_prob']:>7.2f}% | "
            f"{ck[50]['val_ce']:<10.4f} | {ck[100]['val_ce']:<11.4f} | "
            f"{ck[args.steps]['val_ce']:<11.4f} | {ck[args.steps]['top1_acc']:>5.2f}%"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
