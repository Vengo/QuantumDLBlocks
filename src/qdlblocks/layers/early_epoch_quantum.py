"""Early-Epoch Quantum Convergence Fixes & Diagnostics (Experiment Plan 014).

Provides:
1. `compute_self_repetition_metrics`: Measures Step-0 / early-epoch token self-repetition
   probability P(x_{t+1} = x_t), top-1 self-repeat rate, and predictive Shannon entropy.
2. `purified_born_attention_weights`: Canonical degree-4 coherence-thresholded Born attention
   with non-canceling per-head gain `gamma_h`:
       prob_{ij} = ReLU(gamma_h * (s_{ij} - tau_h))^4 + eps_base * ReLU(s_{ij})^2
3. `compute_born_attention_snr_metrics`: Evaluates target-key attention concentration and
   signal-to-noise ratio (SNR) gain at long context lengths (e.g. T=512).
4. `apply_orthogonal_highway_decoupling`: Re-initializes `readout_rot` and `unet_gates` with
   deterministic orthogonal Stiefel matrices so W_lift^T R_readout W_lift has zero diagonal
   self-repetition bias at Step 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from .dynamic_povm_purity import ContextualBlochPurityGate
from .overcomplete_two_beam import OvercompleteTwoBeamInterference


@torch.no_grad()
def compute_self_repetition_metrics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute token self-repetition probability, top-1 self-repeat rate, and output entropy.

    Args:
        logits: Predictive logits of shape `(B, T, V)`.
        input_ids: Input token indices of shape `(B, T)`.
        eps: Numerical stability floor for Shannon entropy.

    Returns:
        Dictionary with `mean_self_repeat_prob`, `top1_self_repeat_rate`, and `mean_entropy_nats`.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits (B, T, V), got shape {tuple(logits.shape)}")
    if input_ids.ndim != 2 or input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"Expected 2D input_ids matching logits[:2] {tuple(logits.shape[:2])}, got {tuple(input_ids.shape)}"
        )

    logits_f = logits.detach().float()
    idx = input_ids.detach().long()
    probs = F.softmax(logits_f, dim=-1)

    self_probs = probs.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    mean_self_repeat_prob = float(self_probs.mean().item())

    top1_preds = logits_f.argmax(dim=-1)
    top1_self_repeat_rate = float((top1_preds == idx).float().mean().item())

    entropies = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    mean_entropy_nats = float(entropies.mean().item())

    return {
        "mean_self_repeat_prob": mean_self_repeat_prob,
        "top1_self_repeat_rate": top1_self_repeat_rate,
        "mean_entropy_nats": mean_entropy_nats,
    }


def purified_born_attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    gamma: torch.Tensor | float = 8.0,
    tau: torch.Tensor | float = 0.0625,
    eps_base: float = 0.05,
    causal: bool = True,
    eps: float = 1e-8,
    window_size: int = 0,
    doc_ids: torch.Tensor | None = None,
    *,
    gamma_h: torch.Tensor | float | None = None,
    tau_h: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Compute canonical coherence-thresholded degree-4 purified Born attention weights.

    Canonical formula:
        s_{ij} = <q_i, k_j>
        prob_{ij} = (ReLU(gamma * (s_{ij} - tau)))^4 + eps_base * ReLU(s_{ij})^2
        A_{ij} = prob_{ij} / (sum_{m <= i} prob_{im} + eps)
    """
    if gamma_h is not None:
        gamma = gamma_h
    if tau_h is not None:
        tau = tau_h
    orig_dtype = q.dtype
    s = torch.matmul(q.float(), k.float().transpose(-1, -2))
    if isinstance(gamma, torch.Tensor):
        gamma_t: torch.Tensor | float = gamma.to(dtype=torch.float32, device=s.device)
        if gamma_t.ndim == 1:
            gamma_t = gamma_t.view(1, -1, 1, 1)
    else:
        gamma_t = float(gamma)
    if isinstance(tau, torch.Tensor):
        tau_t: torch.Tensor | float = tau.to(dtype=torch.float32, device=s.device)
        if tau_t.ndim == 1:
            tau_t = tau_t.view(1, -1, 1, 1)
    else:
        tau_t = float(tau)

    coherent = F.relu(gamma_t * (s - tau_t))
    prob = coherent.square().square() + float(eps_base) * F.relu(s).square()

    if causal or window_size > 0 or doc_ids is not None:
        T = s.size(-1)
        idx = torch.arange(T, device=s.device)
        diff = idx[:, None] - idx[None, :]
        mask_bool = (diff >= 0) if causal else torch.ones((T, T), device=s.device, dtype=torch.bool)
        if window_size > 0:
            mask_bool = mask_bool & (diff < int(window_size))
        mask_4d = mask_bool.view(1, 1, T, T)
        if doc_ids is not None:
            mask_4d = mask_4d & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
        prob = prob * mask_4d.to(prob.dtype)

    weights = prob / (torch.sum(prob, dim=-1, keepdim=True) + eps)
    return weights.to(orig_dtype)


@torch.no_grad()
def compute_born_attention_snr_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    gamma_h: float = 8.0,
    tau_h: float = 0.0625,
    eps_base: float = 0.05,
    target_overlap: float | None = None,
) -> dict[str, float]:
    """Compare target-key (`j=0`) attention concentration between standard and purified Born attention.

    Args:
        q: Unit-normalized query tensor `(B, H, T, d_h)`.
        k: Unit-normalized key tensor `(B, H, T, d_h)` where `j=0` is the planted target key.
        gamma_h: Per-head sharpness gain.
        tau_h: Per-head coherence threshold.
        eps_base: Quadratic background floor coefficient.
        target_overlap: Optional overlap in `(0, 1)` to plant between queries `i >= T // 2` and key `j=0`.

    Returns:
        Dictionary with `std_target_weight`, `purified_target_weight`, `snr_gain_ratio`,
        `std_entropy_nats`, and `purified_entropy_nats` averaged over queries `i >= T // 2`.
    """
    q_u = l2_normalize(q.detach().float(), dim=-1).clone()
    k_u = l2_normalize(k.detach().float(), dim=-1)
    T = q_u.size(-2)
    half_t = T // 2

    if target_overlap is not None:
        alpha = float(target_overlap)
        k0 = k_u[:, :, 0:1, :]
        q_sub = q_u[:, :, half_t:, :]
        proj = (q_sub * k0).sum(dim=-1, keepdim=True) * k0
        ortho = l2_normalize(q_sub - proj, dim=-1)
        q_u[:, :, half_t:, :] = alpha * k0 + ((1.0 - alpha * alpha) ** 0.5) * ortho

    s = torch.matmul(q_u, k_u.transpose(-1, -2))
    prob_std = F.relu(float(gamma_h) * s).square()
    w_std = prob_std / (torch.sum(prob_std, dim=-1, keepdim=True) + 1e-8)

    w_pur = purified_born_attention_weights(
        q_u,
        k_u,
        gamma=gamma_h,
        tau=tau_h,
        eps_base=eps_base,
        causal=True,
        eps=1e-8,
    )

    w_std_sub = w_std[:, :, half_t:, :]
    w_pur_sub = w_pur[:, :, half_t:, :]

    std_target = float(w_std_sub[..., 0].mean().item())
    pur_target = float(w_pur_sub[..., 0].mean().item())
    snr_gain = pur_target / max(std_target, 1e-8)

    std_ent = float((-torch.sum(w_std_sub * torch.log(w_std_sub + 1e-8), dim=-1)).mean().item())
    pur_ent = float((-torch.sum(w_pur_sub * torch.log(w_pur_sub + 1e-8), dim=-1)).mean().item())

    return {
        "std_target_weight": std_target,
        "purified_target_weight": pur_target,
        "snr_gain_ratio": snr_gain,
        "std_entropy_nats": std_ent,
        "purified_entropy_nats": pur_ent,
    }


@torch.no_grad()
def apply_orthogonal_highway_decoupling(
    model: nn.Module,
    seed: int = 1337,
) -> int:
    """Re-initialize `readout_rot` and `unet_gates` with orthogonal Stiefel matrices.

    Breaks the Step-0 identity self-repetition loop `W_lift^T @ I @ W_lift = I` without
    altering the outer random number generator state.

    Args:
        model: `QuantumBornGPT` (or compatible module) to decouple in-place.
        seed: Deterministic seed for orthogonal Stiefel matrix generation.

    Returns:
        Number of 2D Stiefel matrices re-initialized.
    """
    count = 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        readout_rot = getattr(model, "readout_rot", None)
        if readout_rot is not None and hasattr(readout_rot, "weight"):
            w = readout_rot.weight
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                tmp = torch.empty(w.shape, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(tmp)
                w.data.copy_(tmp.to(device=w.device, dtype=w.dtype))
                count += 1

        unet_gates = getattr(model, "unet_gates", None)
        if unet_gates is not None:
            for ug in unet_gates:
                if isinstance(ug, torch.Tensor) and ug.ndim == 2:
                    tmp = torch.empty(ug.shape, dtype=torch.float32, device="cpu")
                    nn.init.orthogonal_(tmp)
                    ug.data.copy_(tmp.to(device=ug.device, dtype=ug.dtype))
                    count += 1

    return count
