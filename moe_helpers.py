import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """
    logits: [..., K]
    mask:   broadcastable to logits, 1 for valid, 0 for invalid
    """
    if mask is None:
        return F.softmax(logits, dim=dim)
    logits = logits.masked_fill(mask == 0, -1e9)
    return F.softmax(logits, dim=dim)


def topk_mask(probs: torch.Tensor, k: int) -> torch.Tensor:
    """
    probs: [B, T, H]
    returns bool mask [B, T, H]
    """
    if k >= probs.shape[-1]:
        return torch.ones_like(probs, dtype=torch.bool)
    topk_idx = torch.topk(probs, k=k, dim=-1).indices
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    return mask


def compute_p_m_e_given_t(
    gate_probs: torch.Tensor,          # [B, T, H]
    modality_ids: torch.Tensor,        # [B, T]
    task_ids: torch.Tensor,            # [B]
    num_modalities: int,
    num_tasks: int,
    token_mask: Optional[torch.Tensor] = None,   # [B, T], 1 valid, 0 invalid
    eps: float = 1e-9,
):
    """
    Build:
      p(m,e|t), p(m|t), p(e|t), p(t)

    where experts e are attention heads.
    """
    device = gate_probs.device
    dtype = gate_probs.dtype
    B, T, H = gate_probs.shape
    M = num_modalities
    P = num_tasks

    if token_mask is None:
        token_mask = torch.ones(B, T, device=device, dtype=dtype)
    else:
        token_mask = token_mask.to(dtype)

    mass = torch.zeros(P, M, H, device=device, dtype=dtype)
    task_mass = torch.zeros(P, device=device, dtype=dtype)

    for b in range(B):
        t = int(task_ids[b].item())
        mb = modality_ids[b]      # [T]
        wb = token_mask[b]        # [T]
        pb = gate_probs[b]        # [T, H]

        for m in range(M):
            mask_m = (mb == m).to(dtype) * wb
            if mask_m.sum() > 0:
                # sum over tokens in this modality
                mass[t, m] += torch.einsum("t,th->h", mask_m, pb)

        task_mass[t] += wb.sum()

    p_t = task_mass / task_mass.sum().clamp_min(eps)                         # [P]
    p_me_t = mass / mass.sum(dim=(1, 2), keepdim=True).clamp_min(eps)       # [P, M, H]
    p_m_t = p_me_t.sum(dim=-1)                                               # [P, M]
    p_e_t = p_me_t.sum(dim=-2)                                               # [P, H]

    return p_me_t, p_m_t, p_e_t, p_t


def mutual_information_m_e_given_t(
    p_me_t: torch.Tensor,   # [P, M, H]
    p_m_t: torch.Tensor,    # [P, M]
    p_e_t: torch.Tensor,    # [P, H]
    p_t: torch.Tensor,      # [P]
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    I(M;E|T) = sum_t p(t) sum_{m,e} p(m,e|t) log [ p(m,e|t) / (p(m|t)p(e|t)) ]
    """
    denom = (p_m_t.unsqueeze(-1) * p_e_t.unsqueeze(-2)).clamp_min(eps)      # [P, M, H]
    numer = p_me_t.clamp_min(eps)
    mi_per_task = (numer * (numer.log() - denom.log())).sum(dim=(1, 2))      # [P]
    return (p_t * mi_per_task).sum()


def expert_load_balance_loss(
    gate_probs: torch.Tensor,          # [B, T, H]
    task_ids: torch.Tensor,            # [B]
    num_tasks: int,
    token_mask: Optional[torch.Tensor] = None,   # [B, T]
    eps: float = 1e-9,
):
    """
    Task-conditional KL-to-uniform:
      p(e|t) = average routing mass over tokens from task t
      LB = sum_t p(t) KL(p(e|t) || Uniform)

    Lower is better.
    """
    device = gate_probs.device
    dtype = gate_probs.dtype
    B, T, H = gate_probs.shape
    P = num_tasks

    if token_mask is None:
        token_mask = torch.ones(B, T, device=device, dtype=dtype)
    else:
        token_mask = token_mask.to(dtype)

    expert_mass = torch.zeros(P, H, device=device, dtype=dtype)
    total_mass = torch.zeros(P, device=device, dtype=dtype)

    for b in range(B):
        t = int(task_ids[b].item())
        wb = token_mask[b]     # [T]
        pb = gate_probs[b]     # [T, H]
        expert_mass[t] += torch.einsum("t,th->h", wb, pb)
        total_mass[t] += wb.sum()

    p_e_t = expert_mass / expert_mass.sum(dim=-1, keepdim=True).clamp_min(eps)   # [P, H]

    uniform = torch.full_like(p_e_t, 1.0 / H)
    kl_per_task = (p_e_t.clamp_min(eps) * (p_e_t.clamp_min(eps).log() - uniform.log())).sum(dim=-1)
    active_task_weights = total_mass / total_mass.sum().clamp_min(eps)
    loss = (active_task_weights * kl_per_task).sum()

    return loss, p_e_t


def attn_mi_loss(
    gate_probs_soft: torch.Tensor,     # [B, T, H],
    modality_ids: torch.Tensor,        # [B, T]
    task_ids: torch.Tensor,            # [B]
    num_modalities: int,
    num_tasks: int,
    token_mask: Optional[torch.Tensor] = None,   # [B, T], 1 for valid, 0 for pad
    mi_weight: float = 0.0,
    eps: float = 1e-9,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    
    aux_attn_loss: Dict[str, torch.Tensor] = {}
    info: Dict[str, torch.Tensor] = {}

    if mi_weight <= 0:
        return aux_attn_loss, info

    device = gate_probs_soft.device
    dtype = gate_probs_soft.dtype

    B, T, H = gate_probs_soft.shape
    assert modality_ids.shape == (B, T)
    assert task_ids.shape == (B,)

    if token_mask is None:
        token_mask = torch.ones(B, T, device=device, dtype=dtype)
    else:
        token_mask = token_mask.to(device=device, dtype=dtype)

    modality_ids = modality_ids.to(device=device, dtype=torch.long)
    task_ids = task_ids.to(device=device, dtype=torch.long)

    # Expand task ids from [B] -> [B, T]
    task_ids_bt = task_ids.unsqueeze(1).expand(B, T)

    # Flatten token-level tensors
    flat_gate = gate_probs_soft.reshape(B * T, H)  # [N, H]
    flat_mod = modality_ids.reshape(B * T)   # [N]
    flat_task = task_ids_bt.reshape(B * T)  # [N]
    flat_mask = token_mask.reshape(B * T)   # [N]

    # Keep only valid tokens
    valid = flat_mask > 0
    if valid.sum() == 0:
        zero = gate_probs_soft.new_tensor(0.0)
        aux_attn_loss["mi"] = zero
        info["mi_e_tm"] = zero.detach()
        info["H_e"] = zero.detach()
        info["H_e_given_tm"] = zero.detach()
        return aux_attn_loss, info

    flat_gate = flat_gate[valid]    # [N_valid, H]
    flat_mod = flat_mod[valid]     # [N_valid]
    flat_task = flat_task[valid]   # [N_valid]
    flat_mask = flat_mask[valid]     # [N_valid], > 0

    # Optional: if mask may contain fractional weights, keep them.
    weights = flat_mask   # [N_valid]
    total_weight = weights.sum().clamp_min(eps)

    # --- 1. marginal expert usage q(e) = p(E=e)

    # Weighted average over all valid tokens
    q_e = (flat_gate * weights.unsqueeze(-1)).sum(dim=0) / total_weight   # [H]
    q_e = q_e / q_e.sum().clamp_min(eps)  # normalize defensively

    H_e = -(q_e * torch.log(q_e.clamp_min(eps))).sum()

    # -- 2. (t,m) expert usage q(e | t, m) and pi(t, m)
    # group index g = t * num_modalities + m
    group_ids = flat_task * num_modalities + flat_mod                      # [N_valid]
    num_groups = num_tasks * num_modalities

    # pi_g = p(T=t, M=m)
    pi_g = torch.zeros(num_groups, device=device, dtype=dtype)             # [G]
    pi_g.scatter_add_(0, group_ids, weights)
    pi_g = pi_g / total_weight   # [G]

    # group_expert_mass[g, h] = sum over tokens in group g of weight * p(E=h|token)
    group_expert_mass = torch.zeros(num_groups, H, device=device, dtype=dtype)  # [G, H]
    group_expert_mass.index_add_(
        0,
        group_ids,
        flat_gate * weights.unsqueeze(-1),
    )

    # group_weight[g] = total token weight in group g
    group_weight = torch.zeros(num_groups, device=device, dtype=dtype)     # [G]
    group_weight.scatter_add_(0, group_ids, weights)

    # q(e | g)
    q_e_given_g = group_expert_mass / group_weight.unsqueeze(-1).clamp_min(eps)  # [G, H]

    # For empty groups, q_e_given_g will be all zeros; make them harmless.
    # Since pi_g = 0 for empty groups, they will not contribute anyway.
    q_e_given_g = q_e_given_g / q_e_given_g.sum(dim=-1, keepdim=True).clamp_min(eps)

    # H(E | T,M) = sum_g pi_g * H(q(.|g))
    H_e_given_tm_per_group = -(q_e_given_g * torch.log(q_e_given_g.clamp_min(eps))).sum(dim=-1)  # [G]
    H_e_given_tm = (pi_g * H_e_given_tm_per_group).sum()

    # -- 3) Mutual information and loss
    mi = H_e - H_e_given_tm
    mi_loss = mi_weight * (H_e_given_tm - H_e)

    aux_attn_loss["mi"] = mi_loss

    # Optional reshaped views for inspection
    info["mi_e_tm"] = mi.detach()
    info["H_e"] = H_e.detach()
    info["H_e_given_tm"] = H_e_given_tm.detach()
    info["p_e"] = q_e.detach()  # [H]
    info["p_tm"] = pi_g.view(num_tasks, num_modalities).detach()  # [num_tasks, num_modalities]
    info["p_e_given_tm"] = q_e_given_g.view(num_tasks, num_modalities, H).detach()  # [T, M, H]

    return aux_attn_loss, info


from typing import Optional, Tuple, Dict
import torch


def attn_mi_loss_conditional(
    gate_probs_soft: torch.Tensor,     # [B, T, H], H = number of experts
    modality_ids: torch.Tensor,        # [B, T], values in [0, num_modalities-1]
    task_ids: torch.Tensor,            # [B], one task id per sample; batch may be single-task
    num_modalities: int,
    num_tasks: int,                    # kept for interface compatibility; not used below
    token_mask: Optional[torch.Tensor] = None,   # [B, T], 1 for valid, 0 for pad
    mi_weight: float = 0.0,
    eps: float = 1e-9,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Conditional mutual-information routing loss:

        I(M; E | T) = H(E | T) - H(E | M, T)

    In the common case where each batch contains a single task t:
        I(M; E | T=t) = H(E | T=t) - H(E | M, T=t)

    We minimize:
        L_mi = H(E | M, T) - H(E | T)

    Intuition:
        - H(E | T): expert diversity for the current task
        - H(E | M, T): expert uncertainty after also knowing modality
        Maximizing MI encourages different modalities within the same task
        to prefer different experts.

    gate_probs_soft[b, t, h] is treated as p(E=h | token_{b,t}).
    """
    aux_attn_loss: Dict[str, torch.Tensor] = {}
    info: Dict[str, torch.Tensor] = {}

    if mi_weight <= 0:
        return aux_attn_loss, info

    device = gate_probs_soft.device
    dtype = gate_probs_soft.dtype

    B, T, H = gate_probs_soft.shape
    assert modality_ids.shape == (B, T)
    assert task_ids.shape == (B,)

    if token_mask is None:
        token_mask = torch.ones(B, T, device=device, dtype=dtype)
    else:
        token_mask = token_mask.to(device=device, dtype=dtype)

    modality_ids = modality_ids.to(device=device, dtype=torch.long)
    task_ids = task_ids.to(device=device, dtype=torch.long)

    # Expand task ids from [B] -> [B, T]
    task_ids_bt = task_ids.unsqueeze(1).expand(B, T)  # [B, T]

    # Flatten to token level
    flat_gate = gate_probs_soft.reshape(B * T, H)      # [N, H]
    flat_mod = modality_ids.reshape(B * T)             # [N]
    flat_task = task_ids_bt.reshape(B * T)             # [N]
    flat_mask = token_mask.reshape(B * T)              # [N]

    # Keep valid tokens
    valid = flat_mask > 0
    if valid.sum() == 0:
        zero = gate_probs_soft.new_tensor(0.0)
        aux_attn_loss["mi"] = zero
        info["mi_m_e_given_t"] = zero.detach()
        info["H_e_given_t"] = zero.detach()
        info["H_e_given_mt"] = zero.detach()
        return aux_attn_loss, info

    flat_gate = flat_gate[valid]     # [N_valid, H]
    flat_mod = flat_mod[valid]       # [N_valid]
    flat_task = flat_task[valid]   # [N_valid]
    weights = flat_mask[valid]      # [N_valid], can be fractional
    total_weight = weights.sum().clamp_min(eps)

    # --- 1) p(t)
    p_t = torch.zeros(num_tasks, device=device, dtype=dtype)   # [num_tasks]
    p_t.scatter_add_(0, flat_task, weights)
    p_t = p_t / total_weight

    # --- 2) q(e | t) and H(E | T)
    task_expert_mass = torch.zeros(num_tasks, H, device=device, dtype=dtype)  # [num_tasks, H]
    task_expert_mass.index_add_(
        0,
        flat_task,
        flat_gate * weights.unsqueeze(-1),
    )

    task_weight = torch.zeros(num_tasks, device=device, dtype=dtype)  # [num_tasks]
    task_weight.scatter_add_(0, flat_task, weights)

    q_e_given_t = task_expert_mass / task_weight.unsqueeze(-1).clamp_min(eps)  # [num_tasks, H]
    q_e_given_t = q_e_given_t / q_e_given_t.sum(dim=-1, keepdim=True).clamp_min(eps)

    H_e_given_t_per_task = -(q_e_given_t * torch.log(q_e_given_t.clamp_min(eps))).sum(dim=-1)  # [num_tasks]
    H_e_given_t = (p_t * H_e_given_t_per_task).sum()

    # --- 3) q(e | m, t) and H(E | M, T)
    num_groups = num_tasks * num_modalities
    group_ids = flat_task * num_modalities + flat_mod  # group = (t, m)

    p_mt = torch.zeros(num_groups, device=device, dtype=dtype)  # [num_tasks * num_modalities]
    p_mt.scatter_add_(0, group_ids, weights)
    p_mt = p_mt / total_weight

    group_expert_mass = torch.zeros(num_groups, H, device=device, dtype=dtype)  # [G, H]
    group_expert_mass.index_add_(
        0,
        group_ids,
        flat_gate * weights.unsqueeze(-1),
    )

    group_weight = torch.zeros(num_groups, device=device, dtype=dtype)  # [G]
    group_weight.scatter_add_(0, group_ids, weights)

    q_e_given_mt = group_expert_mass / group_weight.unsqueeze(-1).clamp_min(eps)  # [G, H]
    q_e_given_mt = q_e_given_mt / q_e_given_mt.sum(dim=-1, keepdim=True).clamp_min(eps)

    H_e_given_mt_per_group = -(q_e_given_mt * torch.log(q_e_given_mt.clamp_min(eps))).sum(dim=-1)  # [G]
    H_e_given_mt = (p_mt * H_e_given_mt_per_group).sum()

    # --- 4) I(M; E | T) = H(E | T) - H(E | M, T)
    mi = H_e_given_t - H_e_given_mt
    mi_loss = mi_weight * (H_e_given_mt - H_e_given_t)

    aux_attn_loss["mi"] = mi_loss

    info["mi_m_e_given_t"] = mi.detach()
    info["H_e_given_t"] = H_e_given_t.detach()
    info["H_e_given_mt"] = H_e_given_mt.detach()
    info["p_t"] = p_t.detach()  # [num_tasks]
    info["p_mt"] = p_mt.view(num_tasks, num_modalities).detach()  # [num_tasks, num_modalities]
    info["p_e_given_t"] = q_e_given_t.detach()  # [num_tasks, H]
    info["p_e_given_mt"] = q_e_given_mt.view(num_tasks, num_modalities, H).detach()  # [num_tasks, num_modalities, H]

    return aux_attn_loss, info