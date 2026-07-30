import torch
import torch.nn as nn
import numpy as np 
import math
from transformers import BertModel, BertTokenizer, ViTImageProcessor, ViTModel, Wav2Vec2Model, Wav2Vec2Processor
from typing import List, Optional, Tuple, Dict
from torch.nn import functional as F
from utils import attn_load_balance, loss_parser
from config import ModelConfig, RunConfig

# ----------------------------
# Scalable Embedders
# ----------------------------
class TextEmbedder(nn.Module):
    def __init__(self, output_dim=768, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.projection = nn.Linear(self.bert_model.config.hidden_size, output_dim)

    def forward(self, text_list, task_name=None):
        device = next(self.parameters()).device
        with torch.no_grad():
            outputs = self.bert_model(text_list)
        return self.projection(outputs.last_hidden_state)

class VisionEmbedder(nn.Module):
    def __init__(self, output_dim=768, vit_model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.vit_model = ViTModel.from_pretrained(vit_model_name)
        self.processor = ViTImageProcessor.from_pretrained(vit_model_name)
        for p in self.vit_model.parameters():
            p.requires_grad = False
        self.projection = nn.Linear(self.vit_model.config.hidden_size, output_dim)

    def forward(self, images, task_name=None):
        device = next(self.parameters()).device
        if isinstance(images, (list, tuple)):
            encoded = self.processor(images=images, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)
        elif torch.is_tensor(images):
            pixel_values = images.to(device).float()
        else:
            raise TypeError("Images must be list of PIL or Tensor.")

        with torch.no_grad():
            outputs = self.vit_model(pixel_values=pixel_values)

        projected = self.projection(outputs.last_hidden_state)

        if task_name != None and 'vqa' in task_name:
            return projected[:, 1: , :]
        else:
            return projected


# ----------------------------
# Task heads
# ----------------------------
class ImageClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        cls_token: (B, D)
        returns:   (B, num_classes)
        """
        return self.fc(cls_token)
    
class ImageReconstructionHead(nn.Module):
    def __init__(self, d_model: int, img_size: int = 224, patch_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches_per_side = img_size // patch_size  # 14 for 224/16
        self.num_patches = self.num_patches_per_side ** 2    # 196
        self.patch_dim = 3 * patch_size * patch_size         # 3*16*16 = 768

        # Map each token (D) → flattened patch pixels (3 * P * P)
        self.proj = nn.Linear(d_model, self.patch_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: (B, L, D), where L = num_patches (=196 for 224/16)
        returns:      (B, 3, img_size, img_size)
        """
        B, L, D = patch_tokens.shape
        assert L == self.num_patches, f"Expected {self.num_patches} patch tokens, got {L}"

        # (B, L, D) -> (B, L, patch_dim)
        patches = self.proj(patch_tokens)  # (B, L, 3*P*P)

        # (B, L, 3*P*P) -> (B, L, 3, P, P)
        patches = patches.view(B, L, 3, self.patch_size, self.patch_size)

        # (B, L, 3, P, P) -> (B, H_p, W_p, 3, P, P)
        H_p = W_p = self.num_patches_per_side
        patches = patches.view(B, H_p, W_p, 3, self.patch_size, self.patch_size)

        # rearrange into full image: (B, 3, H_p*P, W_p*P) = (B, 3, img_size, img_size)
        recon = (
            patches
            .permute(0, 3, 1, 4, 2, 5)    # (B, 3, H_p, P, W_p, P)
            .contiguous()
            .view(B, 3, self.img_size, self.img_size)
        )

        return recon
    
class TextClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        cls_token: (B, D)
        returns:   (B, num_classes)
        """
        return self.fc(cls_token)
    
class TextReconstructionHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        """
        token_features: (B, L, D) -- all non-CLS tokens
        returns: (B, L, vocab_size)
        """
        return self.proj(token_features)
    
class VQAHead(nn.Module):
    def __init__(self, d_model: int, num_answers: int):
        super().__init__()
        # self.proj = nn.Linear(d_model, num_answers)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_answers),
        )

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        """
        token_features: (B, L, D)   -- all tokens
        returns:        (B, num_answers)
        """
        # Simple approach: mean pool over all tokens and project to answers
        #to-do: try other pooling strats
        pooled = token_features.mean(dim=1)  # (B, D)
        return self.proj(pooled)             # (B, num_answers)

# ----------------------------
# Mixture-of-Head Attention (MoH)
# ----------------------------
class MoHAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        self.hs = cfg.attn_shared_heads
        self.hr = cfg.attn_n_heads - self.hs

        self.Ws = nn.Linear(cfg.d_model, self.hs, bias=False) if self.hs > 0 else None
        self.Wr = nn.Linear(cfg.d_model, self.hr, bias=False) if self.hr > 0 else None
        self.Wh = nn.Linear(cfg.d_model, 2, bias=False) if self.hr > 0 and self.hs > 0 else None

    def forward(
            self, x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            pad_mask: Optional[torch.Tensor] = None,
            modality_ids: Optional[torch.Tensor] = None,
    ):
        
        B, T, D = x.shape
        H, Dh = self.cfg.attn_n_heads, self.cfg.attn_head_dim
        _x = x.reshape(B * T, D)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        if attn_mask is not None:
            scores = scores + attn_mask 

        attn = F.softmax(scores, dim=-1)
        
        # Calculate mean attention matrix across all heads to pass to FFN
        avg_attn_matrix = attn.mean(dim=1) # [B, T, T]
        
        attn_drop = self.dropout(attn)
        head_out = torch.matmul(attn_drop, v).transpose(1, 2)

        aux_loss = {}

        # shared head gates
        if self.hs > 1:
            shared_gates = F.softmax(self.Ws(_x), dim=-1).reshape(B, T, self.hs) * self.hs
        elif self.hs == 1:
            # Single shared head always gets weight 1.0
            shared_gates = torch.ones(B, T, self.hs, device=x.device, dtype=x.dtype)
        else:
            shared_gates = None
        
        # routed head gates 
        if self.hr > 0:
            logits = self.Wr(_x)
            gates = F.softmax(logits, dim=-1)

            _, indices = torch.topk(gates, k=self.cfg.attn_topk, dim=-1)
            mask = F.one_hot(indices, num_classes=self.hr).sum(dim=1).float()

            flat_pad = pad_mask.reshape(-1) if pad_mask is not None else None
            
            # --- Dynamic Modality Load Balancing ---
            flat_mod_ids = modality_ids.reshape(-1)
            unique_mods = torch.unique(flat_mod_ids)
            
            total_lb_loss = 0.0
            for mod_id in unique_mods:
                mod_mask = (flat_mod_ids == mod_id)
                total_lb_loss = total_lb_loss + attn_load_balance(
                    gates, 
                    mask, 
                    modality_mask=mod_mask, 
                    pad_mask=flat_pad
                )
                
            # Automatically scale by the number of distinct modalities present
            aux_loss['lb'] = total_lb_loss / len(unique_mods)

            # Renormalize within selected heads, then scale by topk
            routed_gates = gates * mask
            denom = routed_gates.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(routed_gates.dtype).eps)
            routed_gates = (routed_gates / denom) * self.cfg.attn_topk  # [B*T, hr]
            routed_gates = routed_gates.reshape(B, T, self.hr)

        # --- Two-stage alpha balancing (only when both groups exist) ---
        if self.hr > 0 and self.hs > 0:
            alphas = F.softmax(self.Wh(_x), dim=-1).reshape(B, T, 2) * 2
            shared_gates = shared_gates * alphas[..., 0:1]
            routed_gates = routed_gates * alphas[..., 1:2]
            masked_gates = torch.cat([shared_gates, routed_gates], dim=-1)
        elif self.hr > 0:
            masked_gates = routed_gates
        else:
            masked_gates = shared_gates

        gated = torch.einsum("bte,bted->bted", masked_gates, head_out).reshape(B, T, H * Dh)
        y = self.out(gated)
        
        return y, aux_loss, avg_attn_matrix

# ----------------------------
# FFN Experts & Modality Blocks
# ----------------------------
class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ModalityBlock(nn.Module):
    def __init__(self, d_model, hidden, dropout, n_experts, topk):
        super().__init__()
        self.topk = topk
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, hidden, dropout) for _ in range(n_experts)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.numel() == 0: return x, 0.0
        B, T, D = x.shape
        probs = F.softmax(self.router(x), dim=-1)
        
        topk_idx = torch.topk(probs, k=self.topk, dim=-1).indices
        mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, topk_idx, True)
        
        probs_sel = probs * mask.float()
        weights = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        
        lb_loss = (probs.mean(dim=(0, 1)) * mask.float().mean(dim=(0, 1))).sum() * probs.size(-1)

        y = x.new_zeros(B, T, D)
        for e, expert in enumerate(self.experts):
            if not mask[..., e].any(): continue
            y += weights[..., e].unsqueeze(-1) * expert(x)
        return self.dropout(y), lb_loss

class MSoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleDict({
            modality: ModalityBlock(
                cfg.d_model, cfg.msoe_modality_blk_hidden, cfg.dropout, 
                cfg.msoe_n_experts, cfg.msoe_topk
            ) for modality in cfg.active_modalities
        })

    def forward(self, x, modality_masks):
        y = torch.zeros_like(x)
        # aux_loss = {}
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for mod_name, mask in modality_masks.items():
            if mod_name in self.blocks and mask.any():
                idx = torch.nonzero(mask, as_tuple=True)
                if len(idx[0]) == 0: continue
                
                # Extract specific modality tokens
                x_sub = x[idx[0], idx[1]].unsqueeze(1) # [N, 1, D]
                y_sub, l_sub = self.blocks[mod_name](x_sub)
                
                y[idx[0], idx[1]] += y_sub.squeeze(1)
                # aux_loss[mod_name] = l_sub
                aux_loss += l_sub

        return y, {'mod_lb': aux_loss}

class CrossModalExpert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tau = cfg.crossmodal_threshold # 0.5 by default
        self.router = nn.Linear(cfg.d_model, 1) # Outputs P(Collision)
        
        # Heavyweight Expert
        self.collision_expert = ExpertFFN(cfg.d_model, cfg.mtoe_expert_hidden, cfg.dropout)

    def forward(self, x, a_cross):
        # 1. Independent Routing Probability
        p_collision = torch.sigmoid(self.router(x)).squeeze(-1) # [B, T]
        
        # 2. FLOP-Weighted Sparsity Loss (No-Op = 0, Collision = 4 FLOP units)
        l_compute = (p_collision * 4.0).mean()
        
        # 3. Attention-Guided Alignment Loss
        l_align = F.mse_loss(p_collision, a_cross)
        
        # 4. Thresholding & Execution
        mask = (p_collision > self.tau).float()
        w = p_collision * mask 
        
        # Only tokens passing the threshold get computed; others return 0 (No-Op equivalent relative to residual)
        y = self.collision_expert(x) * w.unsqueeze(-1)
        
        return y, {'compute': l_compute, 'align': l_align}

class MoEFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.msoe = MSoE(cfg)
        self.cross_modal = CrossModalExpert(cfg)
        self.linear_proj = nn.Linear(cfg.d_model * 2, cfg.d_model)

    def forward(self, x, modality_masks, a_cross):
        msoe_out, msoe_loss = self.msoe(x, modality_masks) # single modality experts specific to each modality  
        cm_out, cm_loss = self.cross_modal(x, a_cross) # compute cross modal token complexity
        
        y = self.linear_proj(torch.cat([msoe_out, cm_out], dim=-1))
        return y, {**msoe_loss, **cm_loss}

# ----------------------------
# Transformer Block
# ----------------------------
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MoHAttention(cfg)
        self.drop1 = nn.Dropout(cfg.dropout)
        
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MoEFFN(cfg)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x, pad_mask, modality_masks_dict):
        # vision = 0, text = 1, etc.
        B, T, D = x.shape
        mod_ids = torch.full((B, T), -1, device=x.device, dtype=torch.long)
        for i, (mod_name, mask) in enumerate(modality_masks_dict.items()):
            mod_ids[mask] = i

        # 1. Attention
        a, attn_loss, avg_attn_matrix = self.attn(self.ln1(x), pad_mask=pad_mask, modality_ids=mod_ids)
        x = x + self.drop1(a)
        
        # Calculate A_{i -> cross}
        cross_modality_mask = mod_ids.unsqueeze(2) != mod_ids.unsqueeze(1)
        a_cross = (avg_attn_matrix * cross_modality_mask).sum(dim=-1)

        # 2. FFN
        f, ffn_loss = self.ffn(self.ln2(x), modality_masks_dict, a_cross)
        x = x + self.drop2(f)
        
        return x, {'attn': attn_loss, 'ffn': ffn_loss}

# ----------------------------
# Scalable Master Model
# ----------------------------
class MultiModalMultiTaskMoHMoE(nn.Module):
    def __init__(self,
                cfg: ModelConfig,
                run_cfg: RunConfig,
                vision_embedder: nn.Module,
                text_embedder: nn.Module,
                speech_embedder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.run_cfg = run_cfg
        self.active_tasks = run_cfg.task_selection
        task_output_dims = cfg.task_output_dims 
        
        self.embedders = nn.ModuleDict()

        for modality in self.cfg.active_modalities:
            if modality == 'vision':
                self.embedders[modality] = vision_embedder
            elif modality == 'text':
                self.embedders[modality] = text_embedder
            elif modality == 'speech' and speech_embedder is not None:
                self.embedders[modality] = speech_embedder
            else:
                raise ValueError(f"Unknown or unsupported modality: {modality}")

        self.task_heads = nn.ModuleDict({
            'txt_cls': ImageClassificationHead(cfg.d_model, task_output_dims['txt_cls']),
            'txt_rec': TextReconstructionHead(cfg.d_model, task_output_dims['txt_rec']),
            'img_cls': ImageClassificationHead(cfg.d_model, task_output_dims['img_cls']),
            'img_rec': ImageReconstructionHead(cfg.d_model, cfg.img_size, cfg.patch_size),
            'vqa': VQAHead(cfg.d_model, task_output_dims['vqa'])
        })


        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.task_emb = nn.Embedding(cfg.n_tasks, cfg.d_model)
        self.modality_emb = nn.Embedding(cfg.n_modalities, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.fusion_ln = nn.LayerNorm(cfg.d_model)

    def forward(
            self,
            vision_tokens,
            text_tokens,
            speech_tokens,
            task_name: str,
            attn_mask: Optional[torch.Tensor] = None,
        ):
 
        device = next(self.parameters()).device
        B = len(vision_tokens) if vision_tokens is not None else len(text_tokens)

        input_dict = {}
        for modality in self.cfg.active_modalities:
            if modality == 'vision' and vision_tokens is not None:
                input_dict['vision'] = vision_tokens
            elif modality == 'text' and text_tokens is not None:
                input_dict['text'] = text_tokens
            elif modality == 'speech' and speech_tokens is not None:
                input_dict['speech'] = speech_tokens
        
        parts = []
        modality_bounds = {}

        current_len = 0
        for mod_name, raw_data in input_dict.items():
            emb = self.embedders[mod_name](raw_data, task_name=task_name)
            parts.append(emb)
            T_mod = emb.size(1)
            modality_bounds[mod_name] = (current_len, current_len + T_mod)
                
            current_len += T_mod

        total_len = current_len

        # Building required masks based on modality
        modality_masks_dict = {}
        pad_mask = torch.zeros((B, total_len), device=device) 

        for mod_name, (start, end) in modality_bounds.items():
            mask = torch.zeros((B, total_len), device=device, dtype=torch.bool)
            mask[:, start:end] = True
            modality_masks_dict[mod_name] = mask

            if mod_name == 'vision':
                pad_mask[:, start:end] = 1
            elif mod_name == 'text':
                text_mask = (text_tokens != 0).long() 
                pad_mask[:, start:end] = text_mask
            else:
                raise ValueError(f"Unknown modality: {mod_name}")

        x = torch.cat(parts, dim=1)
        x = self.fusion_ln(x)
        T = x.size(1)

        pos_emb_tensor = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        modality_ids = torch.zeros((B, T), device=device, dtype=torch.long) # 1 for vision, 0 for text

        for mod_name, mask in modality_masks_dict.items():
            mod_idx = self.cfg.active_modalities.index(mod_name)
            modality_ids[mask] = mod_idx
        
        task_emb_tensor = self.task_emb(
            torch.tensor(self.run_cfg.task_selection.index(task_name), device=device)
        ).view(1, 1, -1).expand(B, T, -1)  

        x = x + self.pos_emb(pos_emb_tensor) + self.modality_emb(modality_ids) + task_emb_tensor

        # 3. Transformer Blocks
        aux_loss_list = []        
        for blk in self.blocks:
            x, aux_blk = blk(x, pad_mask, modality_masks_dict)
            aux_loss_list.append(aux_blk)

        x = self.ln_f(x)
        
        # 4. Task Head Routing
        head = self.task_heads[task_name]
        
        # Determine if task requires pooling (just task token) or full sequence, eg: classification uses task token (x[:, 0, :]), Reconstruction uses data (x[:, 1:, :])
        if "cls" in task_name:
            features = x[:, 0, :]
        else:
            features = x[:, 1:, :]
            
        logits = head(features)
        parsed_loss_dict = loss_parser(aux_loss_list, device=device)


        return {"logits": logits, "aux_losses": parsed_loss_dict}

class SharedSNRTaskRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = getattr(cfg, 'num_channel_experts', 4)
        
        # The output dimensions (analog symbols) of each expert level
        # e.g., [128, 256, 512, 1024]
        self.expert_dims = torch.tensor([
            cfg.d_model // 4,  
            cfg.d_model // 2,  
            cfg.d_model,       
            cfg.d_model * 2    
        ], dtype=torch.float32)
        
        self.router_mlp = nn.Sequential(
            nn.Linear(1 + cfg.d_model, 64),
            nn.GELU(),
            nn.Linear(64, self.num_experts)
        )
        
    def forward(self, snr, task_emb, temperature=1.0, hard=True):
            device = task_emb.device
            self.expert_dims = self.expert_dims.to(device)
            
            snr_tensor = torch.tensor([[snr]], dtype=torch.float32, device=device)
            router_input = torch.cat([snr_tensor, task_emb], dim=-1)
            
            logits = self.router_mlp(router_input)
                        
            if self.training:
                soft_gumbel_probs = F.gumbel_softmax(logits, tau=temperature, hard=False)
                expected_bw_cost = torch.sum(soft_gumbel_probs * self.expert_dims, dim=-1).mean()
                
                if hard:
                    idx = soft_gumbel_probs.argmax(dim=-1, keepdim=True)
                    y_hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
                    route_weights = y_hard - soft_gumbel_probs.detach() + soft_gumbel_probs
                else:
                    route_weights = soft_gumbel_probs
                    
            else:
                idx = logits.argmax(dim=-1)
                route_weights = F.one_hot(idx, num_classes=self.num_experts).float()
                expected_bw_cost = self.expert_dims[idx].mean() # Actual discrete cost
                
            return route_weights, expected_bw_cost