from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Tuple, List
from config import ModelConfig

from dataset import multitask_batcher, batch_to_inputs, StepBasedMultiTaskBatcher

from utils import final_loss_scaler


def train_step(
    model: nn.Module,
    task_name: str,
    batch,
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    device,
) -> Dict[str, float]:
    
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Move task_id to device and check it's single-task
    task_id = batch["task_id"].to(device)  # (B,)

    if not torch.all(task_id == task_id[0]):
        raise ValueError("Mixed task_ids in a single batch are not supported.")
    # tid = int(task_id[0].item())

    vision_tokens = batch.get("vision_tokens", None)
    text_tokens   = batch.get("text_tokens", None)
    attn_mask     = batch.get("attention_mask", None)

    out = model(
        vision_tokens=vision_tokens,
        text_tokens=text_tokens,
        task_id=task_id,
        # attn_mask=attn_mask,
        attn_mask=None,
    )

    logits = out["logits"] 
    aux_loss = out["aux_loss_attn_ffn"] 

    # img_cls: standard classification loss
    if task_name == "img_cls":
        # labels: (B,) of class indices
        labels = batch["labels"].to(device)  # e.g. CIFAR10 labels
        # logits: (B, num_classes=10)
        loss_task = F.cross_entropy(logits, labels)

    # img_rec: reconstruction loss in pixel space
    elif task_name == "img_rec":
        # logits: (B, 3, H, W) from ImageReconstructionHead
        recon = logits
        # target: same shape as recon
        # adapt key name if your batch uses something else
        target = batch["labels"].to(device)
        loss_task = F.mse_loss(recon, target)

    # txt_cls: standard classification loss
    elif task_name == "txt_cls":
        labels = batch["labels"].to(device)  # (B,)
        # logits: (B, num_classes=2)
        loss_task = F.cross_entropy(logits, labels)

    # txt_rec: 
    elif task_name == "txt_rec":
        # logits: (B, L, vocab_size)
        logits_txt = logits
        B, L_model, V = logits_txt.shape
        
        target_ids = batch["labels"].to(device)  # (B, L)

        # ---- Align label length with model output length
        L_target = target_ids.size(1)

        if L_target == L_model + 1:
            target_ids = target_ids[:, 1:]  # drop CLS
        elif L_target > L_model:
            target_ids = target_ids[:, :L_model]
        elif L_target < L_model:
            raise ValueError(f"Target seq len ({L_target}) < model seq len ({L_model})")

        # Flatten for cross_entropy: (B*L, V) vs (B*L)
        logits_flat  = logits_txt.reshape(-1, V)
        targets_flat = target_ids.reshape(-1)

        loss_task = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=0,  # ignore padded positions
        )

    # vqa_cls: answer classification
    elif task_name == "vqa":
        # logits: (B, num_answers)
        # labels: (B,) answer class indices
        labels = batch["labels"].to(device)
        loss_task = F.cross_entropy(logits, labels)

        # print('aux loss vqa:', aux_loss)

    else:
        raise ValueError(f"Unexpected task_name {task_name} in training step.")

    # ------ total loss
    # dict keys: attn_mi, attn_lb, ffn_text, ffn_vision, ffn_speech
    # loss = loss_task + cfg.attn_lb_weight * aux_loss["attn_lb"] + cfg.attn_mi_weight * aux_loss["attn_mi"] + cfg.ffn_lb_weight * (aux_loss.get("ffn_text", 0) + aux_loss.get("ffn_vision", 0) + aux_loss.get("ffn_speech", 0))

    loss = loss_task + cfg.attn_lb_weight * aux_loss.get("attn_lb", 0) + cfg.ffn_mi_weight * aux_loss.get("ffn_mi", 0)

    loss = final_loss_scaler(task_name, loss)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    aux_loss = {k: v.detach().cpu().item() for k, v in aux_loss.items()}

    return {
        "aux_loss": aux_loss,
        "total_loss": float(loss.detach().cpu()),
        "loss_task": float(loss_task.detach().cpu()),
        "task_name": task_name,
    }


@torch.no_grad()
def eval_step(model, task_name, input_batch, cfg, device):
    model.eval()

    out = model(
        vision_tokens=input_batch.get("vision_tokens", None),
        text_tokens=input_batch.get("text_tokens", None),
        task_id=input_batch["task_id"],
        attn_mask=input_batch.get("attn_mask", None),
    )

    logits = out["logits"]
    aux_loss = out["aux_loss_attn_ffn"]

    # task_id = input_batch["task_id"]
    # tid = int(task_id[0].item())

    if task_name == "img_cls":
        labels = input_batch["labels"]
        loss_task = F.cross_entropy(logits, labels)

        # accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.numel()

    elif task_name == "img_rec":
        recon  = logits
        target = input_batch["labels"]
        loss_task = F.mse_loss(recon, target)
        correct, total = 0, 0   # no "accuracy" for rec

    elif task_name == "txt_cls":
        labels = input_batch["labels"]
        loss_task = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.numel()

    elif task_name == "txt_rec":
        #to-do: change to BLEU score for txt_rec evaluation 

        logits_txt = logits
        B, L_model, V = logits_txt.shape

        target_ids = input_batch["labels"]
        L_target = target_ids.size(1)

        if L_target == L_model + 1:
            target_ids = target_ids[:, 1:]
        elif L_target > L_model:
            target_ids = target_ids[:, :L_model]
        elif L_target < L_model:
            raise ValueError(
                f"Target seq len ({L_target}) < model seq len ({L_model}); "
                "check your text pipeline."
            )

        logits_flat  = logits_txt.reshape(-1, V)
        targets_flat = target_ids.reshape(-1)

        ignore_index = -100  # or tokenizer.pad_token_id if your labels use pads

        bad = (targets_flat < 0) | (targets_flat >= V)
        if bad.any():
            targets_flat = targets_flat.clone()
            targets_flat[bad] = ignore_index

        loss_task = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
        )
        correct, total = 0, 0

    elif task_name == "vqa":
        # logits: (B, num_answers)
        # labels: (B,) answer class indices
        labels = input_batch["labels"].to(device)
        loss_task = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.numel()

    else:
        raise ValueError(f"Unexpected task_name {task_name} in eval_step.")

    # loss = loss_task + cfg.attn_lb_weight * aux_loss["attn_lb"] + cfg.attn_mi_weight * aux_loss["attn_mi"] + cfg.ffn_lb_weight * (aux_loss.get("ffn_text", 0) + aux_loss.get("ffn_vision", 0) + aux_loss.get("ffn_speech", 0))

    loss = loss_task + cfg.attn_lb_weight * aux_loss.get("attn_lb", 0) + cfg.ffn_mi_weight * aux_loss.get("ffn_mi", 0)

    # aux_loss = {k: v.detach().cpu().item() for k, v in aux_loss.items()}

    return {
        "loss": float(loss.detach().cpu()),
        # "loss_task": float(loss_task.detach().cpu()),
        # "loss_aux": aux_loss,
        "correct": correct,
        "total": total,
        "task_name": task_name,
    }


def eval_epoch(model, test_loaders, cfg, device, max_batches=None, eval_steps=100):
    model.eval()
    stats_agg = {}  # per task_name
    batcher = StepBasedMultiTaskBatcher(test_loaders)

    with torch.no_grad():
        for step in range(eval_steps):
            task_name, batch = batcher.next()

            if max_batches is not None and step >= max_batches:
                break

            # print(f"Evaluating batch {step} for task {task_name}...")

            input_batch = batch_to_inputs(batch)
            input_batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in input_batch.items()
            }

            out_stats = eval_step(model, task_name, input_batch, cfg, device)

            if task_name not in stats_agg:
                stats_agg[task_name] = {
                    "loss_sum": 0.0,
                    # "loss_task_sum": 0.0,
                    # "aux_sum": 0.0,
                    "count": 0,
                    "correct": 0,
                    "total": 0,
                }

            s = stats_agg[task_name]
            s["loss_sum"]      += out_stats["loss"]
            # s["loss_task_sum"] += out_stats["loss_task"]
            # s["aux_sum"]       += out_stats["aux"]
            s["count"]         += 1
            s["correct"]       += out_stats["correct"]
            s["total"]         += out_stats["total"]

    # Compute averages
    results = {}
    for task_name, s in stats_agg.items():
        avg_loss      = s["loss_sum"] / max(s["count"], 1)
        # avg_loss_task = s["loss_task_sum"] / max(s["count"], 1)
        # avg_aux       = s["aux_sum"] / max(s["count"], 1)
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
        else:
            acc = None  # e.g. reconstruction tasks

        results[task_name] = {
            "loss": avg_loss,
            # "loss_task": avg_loss_task,
            # "aux": avg_aux,
            "accuracy": acc,
        }

    return results