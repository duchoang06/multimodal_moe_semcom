from __future__ import annotations

from dataclasses import dataclass
import datetime
import itertools
from typing import Dict, Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import topk_mask, build_causal_mask, fix_seed

from dataset import multitask_batcher, batch_to_inputs, StepBasedMultiTaskBatcher

from base_models import MultiModalMultiTaskMoHMoE, TextEmbedder, VisionEmbedder, SpeechEmbedder
from config import ModelConfig, RunConfig

from dataset import CIFAR10ImageTask, SST2TextTask, collate_img, collate_text, VQAv2Task, collate_vqa
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from train_test_utils import train_step, eval_epoch

if __name__ == "__main__":
    rand_seed = 2026
    fix_seed(rand_seed)

    cfg = ModelConfig(
        d_model=768,
        n_layers=4,
        n_tasks=5,
        max_seq_len=512,

        attn_n_heads=12,
        attn_head_dim=64,
    )

    run_cfg = RunConfig()

    task_selection = run_cfg.task_selection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'img_rec' in task_selection or 'img_cls' in task_selection:
    #to-do: may remove cls token from vision embedder
        vision_embedder = VisionEmbedder(output_dim=cfg.d_model) # always 197 tokens

    if 'txt_rec' in task_selection or 'txt_cls' in task_selection:
        text_embedder = TextEmbedder(output_dim=cfg.d_model)

    if 'spc' in task_selection:
        speech_embedder = None
        # speech_embedder = SpeechEmbedder(output_dim=cfg.d_model)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size


    # Dataset setup
    train_ds = {}
    test_ds = {}
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
            train_ds[task] = VQAv2Task(root="./data/vqav2", split="train", tokenizer=tokenizer, task="vqa", max_len=32, patch_size=16, top_k=3000)
            test_ds[task] = VQAv2Task(root="./data/vqav2", split="validation", tokenizer=tokenizer, task="vqa", max_len=32, patch_size=16, top_k=3000,
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
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=collate_img)
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=collate_img)

        elif task == 'img_rec':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=collate_img)
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=collate_img)

        elif task == 'txt_cls':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=lambda b: collate_text(b, pad_id=tokenizer.pad_token_id))
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=lambda b: collate_text(b, pad_id=tokenizer.pad_token_id))

        elif task == 'txt_rec':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=lambda b: collate_text(b, pad_id=tokenizer.pad_token_id))
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=lambda b: collate_text(b, pad_id=tokenizer.pad_token_id))

        elif task == 'vqa':
            train_loaders[task] = DataLoader(train_ds[task], batch_size=train_batch_size, shuffle=True, collate_fn=lambda b: collate_vqa(b, pad_id=tokenizer.pad_token_id))
            test_loaders[task] = DataLoader(test_ds[task], batch_size=test_batch_size, shuffle=False, collate_fn=lambda b: collate_vqa(b, pad_id=tokenizer.pad_token_id))

        else:
            raise Exception(f"Unknown task {task} in dataloader setup")


    task_output_dims = [
        10,           # img_cls
        None,         # img_rec (handled by reconstruction head)
        2,            # txt_cls
        vocab_size,   # txt_rec
        3000,         # vqa (assuming 3000 possible answers)
    ]

    model = MultiModalMultiTaskMoHMoE(cfg, vision_embedder, text_embedder, task_output_dims).to(device)
    
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
    
    num_epochs = run_cfg.num_epochs
    steps_per_epoch = run_cfg.steps_per_epoch

    batcher = StepBasedMultiTaskBatcher(train_loaders)

    log_every = 100
    eval_every = 10 
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        model.train()
        for step in range(steps_per_epoch):

            task_name, batch = batcher.next()
            input_batch = batch_to_inputs(batch)

            if task_name == 'vqa':
                print('test vqa:')

            input_batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in input_batch.items()
            }

            stats = train_step(model, task_name, input_batch, optim, cfg, device)

            if step % log_every == 0:
                print(
                    f"[train] step {step} task={task_name} "
                    f"loss={stats['total_loss']:.4f} "
                    f"loss_task={stats['loss_task']:.4f} "
                    f"aux_loss={stats['aux_loss']}"
                )
        
        if (epoch + 1) % eval_every == 0:
            eval_results = eval_epoch(model, test_loaders, cfg, device, max_batches=None, eval_steps=run_cfg.eval_steps)

            print("Eval results:")
            for task_name, r in eval_results.items():
                acc_str = f", acc={r['accuracy']:.4f}" if r["accuracy"] is not None else ""
                print(
                    f"  {task_name}: "
                    f"loss = {r['loss']:.4f}, "
                    # f"task_loss = {r['loss_task']:.4f}, "
                    # f"aux_loss = {r['aux_loss']}"
                    f"{acc_str}"
                )

        
    # save the final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"./checkpoints/final_model_{timestamp}.pth")


# nohup python -u main.py > ./log/main_nomi_loss_$(date +%Y%m%d_%H%M%S).log 2>&1 & 

