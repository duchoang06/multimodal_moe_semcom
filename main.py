from __future__ import annotations

from dataclasses import dataclass
import datetime
import itertools
from typing import Dict, Optional, Tuple, List

import math, os
print(f"Process ID: {os.getpid()}", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import topk_mask, build_causal_mask, fix_seed
from dataset import multitask_batcher, batch_to_inputs, StepBasedMultiTaskBatcher

from base_models import TextEmbedder, VisionEmbedder
from semcom_models import MoAMoH_SemCom

from config import ModelConfig, RunConfig
from dataset import CIFAR10ImageTask, SST2TextTask, collate_img, collate_text, VQAv2Task, collate_vqa
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from train_test_utils import train_step, eval_epoch

if __name__ == "__main__":
    # rand_seed = 2028
    # fix_seed(rand_seed)

    cfg = ModelConfig()
    run_cfg = RunConfig()

    print("--- Model Configuration:")
    for field_name, field_value in cfg.__dict__.items():
        print(f"  {field_name}: {field_value}")
    
    print("Run Configuration:")
    for field_name, field_value in run_cfg.__dict__.items():
        print(f"  {field_name}: {field_value}")

    task_selection = run_cfg.task_selection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'img_rec' in task_selection or 'img_cls' in task_selection or 'vqa' in task_selection:
    #to-do: may remove cls token from vision embedder
        vision_embedder = VisionEmbedder(output_dim=cfg.d_model) # always 197 tokens

    if 'txt_rec' in task_selection or 'txt_cls' in task_selection or 'vqa' in task_selection:
        text_embedder = TextEmbedder(output_dim=cfg.d_model)

    if 'spc' in task_selection:
        speech_embedder = None
        # speech_embedder = SpeechEmbedder(output_dim=cfg.d_model)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size


    # Dataset setup
    train_ds = {}
    test_ds = {}
    top_k_vqa = cfg.task_output_dims['vqa']  
    for task in task_selection:
        if task == 'img_cls':
            train_ds[task] = CIFAR10ImageTask("./data", train=True, task="img_cls", patch_size=16)
            test_ds[task] = CIFAR10ImageTask("./data", train=False, task="img_cls", patch_size=16)

        elif task == 'img_rec':
            train_ds[task] = CIFAR10ImageTask("./data", train=True, task="img_rec", patch_size=16)
            test_ds[task] = CIFAR10ImageTask("./data", train=False, task="img_rec", patch_size=16)

        elif task == 'txt_cls':
            train_ds[task] = SST2TextTask("train", tokenizer, task="txt_cls", max_len=128)
            test_ds[task] = SST2TextTask("validation", tokenizer, task="txt_cls", max_len=128)

        elif task == 'txt_rec':
            train_ds[task] = SST2TextTask("train", tokenizer, task="txt_rec", max_len=128)
            test_ds[task] = SST2TextTask("validation", tokenizer, task="txt_rec", max_len=128)

        elif task == 'vqa':
            train_ds[task] = VQAv2Task(root="./data/vqav2", split="train", tokenizer=tokenizer, task="vqa", max_len=32, patch_size=16, top_k=top_k_vqa)
            test_ds[task] = VQAv2Task(root="./data/vqav2", split="validation", tokenizer=tokenizer, task="vqa", max_len=32, patch_size=16, top_k=top_k_vqa,
            )

        else:
            raise Exception(f"Unknown task {task} in dataset setup")


    # Dataset loaders setup
    train_batch_size = run_cfg.train_batch_size
    test_batch_size = run_cfg.test_batch_size

    train_loaders = {}
    test_loaders = {}
    
    for task in task_selection:
        if task == 'img_cls':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=collate_img, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=collate_img, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        elif task == 'img_rec':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=collate_img, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=collate_img, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        elif task == 'txt_cls':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=lambda b: collate_text(b, pad_id=tokenizer.pad_token_id), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=lambda b: collate_text(b, pad_id=tokenizer.pad_token_id), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        elif task == 'txt_rec':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=lambda b: collate_text(b, pad_id=tokenizer.pad_token_id), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=lambda b: collate_text(b, pad_id=tokenizer.pad_token_id), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        elif task == 'vqa':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=lambda b: collate_vqa(b, pad_id=tokenizer.pad_token_id), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=lambda b: collate_vqa(b, pad_id=tokenizer.pad_token_id), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        else:
            raise Exception(f"Unknown task {task} in dataloader setup")


    model = MoAMoH_SemCom(cfg, run_cfg, vision_embedder, text_embedder, speech_embedder=None).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Do not decay biases, layernorms, or router networks
        if any(nd in name for nd in ['bias', 'LayerNorm.weight', 'ln', 'router']):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=2e-4, betas=(0.9, 0.95))

    # optimizer = torch.optim.AdamW(
    #     model.parameters(), 
    #     lr=2e-4,          # Peak learning rate
    #     betas=(0.9, 0.95), # Crucial for from-scratch training
    #     weight_decay=0.1
    # )    
    num_epochs = run_cfg.num_epochs
    steps_per_epoch = run_cfg.steps_per_epoch

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    min_lr_ratio = 0.1  

    def get_lr_multiplier(step):
        # Phase 1: Linear Warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # Phase 2: Cosine Decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between 10% (min_lr_ratio) and 100% of peak LR
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay 
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    batcher = StepBasedMultiTaskBatcher(train_loaders, run_cfg.task_selection, run_cfg.sample_task_probs)

    log_every = 100
    eval_every = 5
    start_time = datetime.datetime.now()

    start_temp = 5.0
    end_temp = 0.1

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        model.train()
        current_temp = start_temp * (end_temp / start_temp) ** (epoch / max(1, num_epochs - 1))

        for step in range(steps_per_epoch):
            task_name, batch = batcher.next()
            input_batch = batch_to_inputs(batch)

            input_batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in input_batch.items()
            }

            stats = train_step(model, task_name, input_batch, optimizer, cfg, device, scheduler, current_temp)

            if step % log_every == 0:
                print(
                    f"[train] step {step} task={task_name} "
                    f"loss={stats['total_loss']:.4f} "
                    f"loss_task={stats['loss_task']:.4f} "
                    f"aux_loss={stats['aux_loss']}"
                )
        
        elapsed = datetime.datetime.now() - start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Elapsed: {hours}h {minutes}m {seconds}s")

        
        if (epoch + 1) % eval_every == 0:
            eval_results = eval_epoch(model, test_loaders, cfg, device, max_batches=None, eval_steps=run_cfg.eval_steps)

            print("Eval results:")
            for task_name, r in eval_results.items():
                acc_str = f", acc={r['accuracy']:.4f}" if r["accuracy"] is not None else ""

                print(
                    f"  {task_name}: "
                    f"loss = {r['loss']:.4f}, "
                    f"task_loss = {r['loss_task']:.4f}, "
                    # f"aux_loss = {r['aux_loss']}"
                    f"{acc_str}"
                )

        
    # save the final model
    os.makedirs("./checkpoints", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # torch.save(model.state_dict(), f"./checkpoints/semcom_{timestamp}.pth")


# nohup python -u main.py > ./log/semcom_huge_update_withtask_$(date +%Y%m%d_%H%M%S).log 2>&1 & 
