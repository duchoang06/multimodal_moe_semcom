import torch
import torch.nn as nn
import numpy as np 
import math
from transformers import BertModel, BertTokenizer, ViTForImageClassification, ViTImageProcessor, ViTModel

from config import ModelConfig
from utils import topk_mask, l2_normalize, attn_load_balance, loss_parser
from typing import List, Optional, Tuple, Dict
from torch.nn import functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from moe_helpers import attn_mi_loss_conditional


import os 

# Modality Embedding
class TextEmbedder(nn.Module):
    '''
        Input: raw text list
        Output: text embedding [B, L, d_model]
    '''
    def __init__(self, output_dim=768, bert_model_name='bert-base-uncased', max_seq_len=128):
        super(TextEmbedder, self).__init__()
        
        # Load pretrained BERT model and tokenizer
        self.bert_model = BertModel.from_pretrained(bert_model_name)

        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name) # detault embedding size is 768
        
        self.projection = nn.Linear(self.bert_model.config.hidden_size, output_dim) # 768 to output_dim=256
        self.vocab_size = self.tokenizer.vocab_size
        self.max_seq_len = max_seq_len

    def forward(self, text_list):
        device = next(self.parameters()).device
        # encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=self.max_seq_len).to(device)

        assert torch.is_tensor(text_list)
        encoded_input = text_list

        with torch.no_grad():
            outputs = self.bert_model(encoded_input)
        
        # cls_embedding = outputs.last_hidden_state[:, 0, :]
        text_embedding = outputs.last_hidden_state  

        text_embedding = self.projection(text_embedding)  # Shape: (batch_size, embed_dim) 

        # return text_embedding, encoded_input['input_ids'], encoded_input['attention_mask'] # encoded_input['input_ids'] is for reconstruction 
        return text_embedding  


class VisionEmbedder(nn.Module):
    """
    Input:
      - images: list[PIL.Image] OR torch.Tensor of shape [B, C, H, W] (float in [0,1] or [0,255])
    Output:
      - vision embedding: [B, L, output_dim]
        where L = 1 + num_patches (includes CLS token), matching ViT's token sequence.
    """
    def __init__(self,
                 output_dim=768,
                 vit_model_name="google/vit-base-patch16-224-in21k",
                 freeze_backbone=True):
        super().__init__()

        # Pretrained ViT + processor (handles resize/normalize)
        # self.vit_model = ViTModel.from_pretrained(vit_model_name)

        self.vit_model = ViTModel.from_pretrained(vit_model_name)
        self.processor = ViTImageProcessor.from_pretrained(vit_model_name)

        if freeze_backbone:
            for p in self.vit_model.parameters():
                p.requires_grad = False

        self.projection = nn.Linear(self.vit_model.config.hidden_size, output_dim)

    def forward(self, images):
        device = next(self.parameters()).device

        # Prepare inputs for ViT
        # - If list of PIL images: processor will resize/normalize and return pixel_values
        # - If tensor: we pass it directly as pixel_values (but ensure correct dtype/device)
        if isinstance(images, (list, tuple)):
            encoded = self.processor(images=images, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)

        elif torch.is_tensor(images):
            pixel_values = images.to(device)
            # ViT expects float pixel_values; if your tensor is uint8, convert:
            if pixel_values.dtype == torch.uint8:
                pixel_values = pixel_values.float()
            # If your tensor is in [0,255], the model expects normalized values.
            # Best practice: use processor even for tensors, but that requires converting
            # to PIL or implementing normalization yourself. See note below.
        else:
            raise TypeError("images must be a list of PIL images or a torch.Tensor [B,C,H,W].")

        # Backbone forward (frozen like your BERT block)
        if any(p.requires_grad for p in self.vit_model.parameters()):
            outputs = self.vit_model(pixel_values=pixel_values)
        else:
            with torch.no_grad():
                outputs = self.vit_model(pixel_values=pixel_values)

        # Token embeddings: [B, 1 + num_patches, hidden]
        vision_tokens = outputs.last_hidden_state

        # Project to desired output_dim -> [B, L, output_dim]
        vision_embedding = self.projection(vision_tokens)
        return vision_embedding


    from transformers import Wav2Vec2Model, Wav2Vec2Processor

class SpeechEmbedder(nn.Module):
    """
    Input:
      - audio_list: list of 1D float arrays/tensors (raw waveform), OR a float tensor [B, T]
        Assumes 16kHz, mono audio by default (most wav2vec2 checkpoints expect 16k).
    Output:
      - speech_embedding: [B, L, output_dim]  (frame-level embeddings)
        where L is the encoder time steps after feature extractor stride.
    """
    def __init__(
        self,
        output_dim=768,
        model_name="facebook/wav2vec2-base",  # lightweight pretrained backbone
        sampling_rate=16000,
        max_seconds=None,  # optional truncation in seconds
        freeze_backbone=True,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.max_seconds = max_seconds

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Linear(self.backbone.config.hidden_size, output_dim)

    def forward(self, audio):
        """
        audio:
          - list of 1D waveforms (torch/numpy) OR tensor [B, T]
        """
        device = next(self.parameters()).device

        # Optional truncation (keeps “vibe” similar to max_seq_len in text)
        if self.max_seconds is not None:
            max_len = int(self.max_seconds * self.sampling_rate)
        else:
            max_len = None

        if isinstance(audio, (list, tuple)):
            # processor handles padding + attention_mask
            proc = self.processor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=(max_len is not None),
                max_length=max_len,
            )
            input_values = proc["input_values"].to(device)            # [B, T]
            attention_mask = proc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        elif torch.is_tensor(audio):
            input_values = audio.to(device)  # [B, T]
            if max_len is not None and input_values.shape[1] > max_len:
                input_values = input_values[:, :max_len]
            attention_mask = None
        else:
            raise TypeError("audio must be a list of 1D waveforms or a torch.Tensor [B, T].")

        # Frozen backbone forward, like your BERT code
        if any(p.requires_grad for p in self.backbone.parameters()):
            outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state  # [B, L, hidden_size]
        speech_embedding = self.projection(hidden)  # [B, L, output_dim]
        return speech_embedding

# ----------------------------
# Mixture-of-Head Attention (MoH)
# ----------------------------
class MoHAttention_mi_loss(nn.Module): # OLD
    """
    Clean MoE-style mixture of attention heads.

    - Heads are the experts
    - x is assumed to ALREADY contain modality/task bias embeddings
    - routing is standard per-token gating over heads
    - top-k routing is applied after soft probabilities are computed
    - output is still a weighted combination of attention heads
    - auxiliary losses use the SOFT routing probabilities before top-k masking

    Inputs:
      x:            [B, T, D]
      modality_ids: [B, T]   integer modality labels
      task_ids:     [B]      integer task labels
      attn_mask:    optional additive mask broadcastable to [B, 1, T, T]
      token_mask:   optional [B, T], 1 valid, 0 padding

    Returns:
      y:        [B, T, D]
      aux_loss: scalar
      aux_info: dict (optional)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model == cfg.attn_n_heads * cfg.attn_head_dim, "d_model must equal attn_n_heads * attn_head_dim"

        self.cfg = cfg
        self.num_modalities = cfg.n_modalities
        self.num_tasks = cfg.n_tasks

        D = cfg.d_model
        H = cfg.attn_n_heads
        hidden = D // 2

        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # One clean router over all heads
        self.router = nn.Sequential(
            nn.Linear(D, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(cfg.moh_router_dropout),
            nn.Linear(hidden, H, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,                     # [B, T, D], already contains bias embeddings
        modality_ids: torch.Tensor,         # [B, T]
        task_ids: torch.Tensor,             # [B]
        attn_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        return_aux_info: bool = False,
    ):
        B, T, D = x.shape
        H, Dh = self.cfg.attn_n_heads, self.cfg.attn_head_dim

        # --------- 1) Standard dense multi-head attention 
        qkv = self.qkv(x)                                      # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, H, Dh).transpose(1, 2)               # [B, H, T, Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)   # [B, H, T, T]
        if attn_mask is not None:
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        head_out = torch.matmul(attn, v)                       # [B, H, T, Dh]
        head_out = head_out.transpose(1, 2)                    # [B, T, H, Dh]

        # ------- 2) Per-token routing over heads
        gate_logits = self.router(x)                           # [B, T, H]

        if token_mask is not None:
            gate_logits = gate_logits.masked_fill(token_mask.unsqueeze(-1) == 0, -1e9)

        # soft routing probs used for aux losses
        gate_probs_soft = F.softmax(gate_logits, dim=-1)       # [B, T, H]

        # top-k sparse routing used for actual mixture
        if self.cfg.moh_topk is not None and self.cfg.moh_topk < H:
            mask = topk_mask(gate_probs_soft, self.cfg.moh_topk)        # [B, T, H]
            gate_probs = gate_probs_soft * mask.float()
            gate_probs = gate_probs / gate_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            gate_probs = gate_probs_soft

        if token_mask is not None:
            gate_probs = gate_probs * token_mask.unsqueeze(-1)
            gate_probs_soft = gate_probs_soft * token_mask.unsqueeze(-1)

        # -------- 3) Weighted sum of attention heads
        gated_heads = head_out * gate_probs.unsqueeze(-1)      # [B, T, H, Dh]
        y = gated_heads.reshape(B, T, H * Dh)                  # [B, T, D]
        y = self.out(y)                                        # [B, T, D]

        if token_mask is not None:
            y = y * token_mask.unsqueeze(-1)

        # ---- 4) Auxiliary losses from soft routing probs
        aux_attn_loss_dict, aux_info = attn_mi_loss_conditional(
            gate_probs_soft=gate_probs_soft,
            modality_ids=modality_ids,
            task_ids=task_ids,
            num_modalities=self.num_modalities,
            num_tasks=self.num_tasks,
            token_mask=token_mask,
            mi_weight=self.cfg.attn_mi_weight,
        )

        if return_aux_info:
            aux_info = {
                **aux_info,
                "gate_probs_soft": gate_probs_soft.detach(),
                "gate_probs_topk": gate_probs.detach(),
                "gate_logits": gate_logits.detach(),
            }
            return y, aux_attn_loss_dict, aux_info

        return y, aux_attn_loss_dict


class MoHAttention(nn.Module):
    """
    - compute all heads (dense) in a standard MHA-style way
    - per-token routing produces weights g_i(x_t) over heads
    - shared heads always active, routed heads are top-k
    - output is weighted sum over head outputs
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model == cfg.attn_n_heads * cfg.attn_head_dim, "d_model must equal attn_n_heads * attn_head_dim"
        self.cfg = cfg

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        self.hs = cfg.attn_shared_heads
        self.hr = cfg.attn_n_heads - self.hs

        # Routers (two-stage like paper): separate projections for shared vs routed
        self.Ws = nn.Linear(cfg.d_model, self.hs, bias=False) if self.hs > 0 else None
        self.Wr = nn.Linear(cfg.d_model, self.hr, bias=False) if self.hr > 0 else None

        self.Wh = nn.Linear(cfg.d_model, 2, bias=False) if self.hr > 0 and self.hs > 0 else None

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, d_model]
        attn_mask: optional additive mask broadcastable to [B, 1, T, T]
                   (e.g., causal mask: -inf above diagonal)

        returns:
          y: [B, T, d_model]
          aux_loss: loss dict
        """
        B, T, D = x.shape
        H, Dh = self.cfg.attn_n_heads, self.cfg.attn_head_dim
        _x = x.reshape(B * T, D) # flatten x for per-token processig

        qkv = self.qkv(x)    # [B,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,T,T]
        if attn_mask is not None:
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        head_out = torch.matmul(attn, v).transpose(1, 2)  # [B,T,H,Dh]

        # Build per-token gating g_i(x_t)
        alphas = F.softmax(self.Wh(x), dim=-1) # [B,T,2] -> [alpha_shared, alpha_routed]

        # aux_loss = x.new_zeros(())
        aux_loss = {}
        # --- Shared head gates ---
        if self.hs > 1:
            # Soft distribution over shared heads, scaled by hs so they sum to hs
            shared_gates = F.softmax(self.Ws(_x), dim=-1).reshape(B, T, self.hs) * self.hs
        else:
            # Single shared head always gets weight 1.0
            shared_gates = torch.ones(B, T, self.hs, device=x.device, dtype=x.dtype)

        # --- Routed head gates ---
        if self.hr > 0:
            logits = self.Wr(_x)   # [B*T, hr]
            gates = F.softmax(logits, dim=-1)     # [B*T, hr]

            _, indices = torch.topk(gates, k=self.cfg.attn_topk, dim=-1)
            mask = F.one_hot(indices, num_classes=self.hr).sum(dim=1).float()  # [B*T, hr]

            aux_loss['lb'] = attn_load_balance(gates, mask)

            # Renormalize within selected heads, then scale by topk
            routed_gates = gates * mask
            denom = routed_gates.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(routed_gates.dtype).eps)
            routed_gates = (routed_gates / denom) * self.cfg.attn_topk  # [B*T, hr]
            routed_gates = routed_gates.reshape(B, T, self.hr)

        # --- Two-stage alpha balancing (only when both groups exist) ---
        if self.hr > 0 and self.hs > 0:
            # Scale by 2 so alphas average to 1 rather than sum to 1
            alphas = F.softmax(self.Wh(_x), dim=-1).reshape(B, T, 2) * 2
            shared_gates = shared_gates * alphas[..., 0:1]
            routed_gates = routed_gates * alphas[..., 1:2]
            masked_gates = torch.cat([shared_gates, routed_gates], dim=-1)  # [B,T,H]

        elif self.hr > 0:
            masked_gates = routed_gates
        else:
            masked_gates = shared_gates

        # Apply gates and project
        gated = torch.einsum("bte,bted->bted", masked_gates, head_out)  # [B,T,H,Dh]
        gated = gated.reshape(B, T, H * Dh)    # [B,T,D]
        y = self.out(gated)
        return y, aux_loss 

# ----------------------------
# Mixture-of-Experts FFN
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

def topk_mask(probs: torch.Tensor, k: int) -> torch.Tensor:
    """
    probs: [B, T, E]
    returns: bool mask [B, T, E]
    """
    k = min(k, probs.size(-1))
    topk_idx = torch.topk(probs, k=k, dim=-1).indices
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    return mask


def load_balance_loss_from_scores(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Simple auxiliary load-balancing loss.
    probs: [B, T, E]
    mask:  [B, T, E] bool
    """
    E = probs.size(-1)
    importance = probs.mean(dim=(0, 1))          # [E]
    load = mask.float().mean(dim=(0, 1))         # [E]
    return E * torch.sum(importance * load)


class ModalityBlock(nn.Module):
    """
    Conventional token-level Top-K MoE FFN block.

    Input:
      x: [B, T, D]

    Output:
      y: [B, T, D]
      aux_loss: 
    """
    def __init__(self, d_model: int, hidden: int, dropout: float, n_experts: int, topk: int):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, hidden, dropout) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            return x, x.new_zeros(())

        B, T, D = x.shape

        logits = self.router(x)                   # [B, T, E]
        probs = F.softmax(logits, dim=-1)     # [B, T, E]

        mask = topk_mask(probs, self.topk)   # [B, T, E]
        probs_sel = probs * mask.float()
        denom = probs_sel.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = probs_sel / denom   # [B, T, E]

        lb_loss = load_balance_loss_from_scores(probs, mask)

        y = x.new_zeros(B, T, D)
        for e, expert in enumerate(self.experts):
            if not mask[..., e].any():
                continue
            w = weights[..., e].unsqueeze(-1)     # [B, T, 1]
            y = y + w * expert(x)

        y = self.dropout(y)
        return y, lb_loss


class MoEFFN(nn.Module):
    """
    modality ids:
      0 = text
      1 = vision
      2 = speech

    forward returns:
      y: [B, T+1, D] (Since we concatenate MToE and MSoE outputs)
      aux_loss: scalar
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.msoe = MSoE(cfg)
        self.mtoe = MToE(cfg)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor):
        # 1. Extract shared/task-dependent features via MToE

        data_tokens = x[:, 1:, :]

        mtoe_out, cmi_loss = self.mtoe(data_tokens, modality_ids)
        
        new_task_token = mtoe_out.mean(dim=1, keepdim=True)

        # 2. Extract modality-specific features via MSoE
        msoe_out, msoe_loss_dict = self.msoe(data_tokens, modality_ids)
        #TO-DO: maybe load balancing for MSoE blocks
        
        # 3. Concat task and modality-specific embeddings
        y = torch.cat([new_task_token, msoe_out], dim=1)
        
        # 4. Aggregate Auxiliary Losses
        # aux_loss = - (self.cfg.cmi_alpha * cmi_loss) 
        # for aux_val in msoe_loss_dict.values():
        #     aux_loss = aux_loss + aux_val

        mi_loss_dict = {'mi': -cmi_loss}

        return y, mi_loss_dict

class MSoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg 

        self.text_block = ModalityBlock(
            d_model=cfg.d_model,
            hidden=cfg.msoe_modality_blk_hidden,
            dropout=cfg.dropout,
            n_experts=cfg.msoe_n_experts_text,
            topk=cfg.msoe_topk,
        )

        self.vision_block = ModalityBlock(
            d_model=cfg.d_model,
            hidden=cfg.msoe_modality_blk_hidden,
            dropout=cfg.dropout,
            n_experts=cfg.msoe_n_experts_vision,
            topk=cfg.msoe_topk,
        )

        self.speech_block = ModalityBlock(
            d_model=cfg.d_model,
            hidden=cfg.msoe_modality_blk_hidden,
            dropout=cfg.dropout,
            n_experts=cfg.msoe_n_experts_speech,
            topk=cfg.msoe_topk,
        )

    def _apply_subset_block(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        block: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply block only on masked token positions.

        Args:
          x: [B, T, D]
          token_mask: [B, T] bool

        Returns:
          out: [B, T, D], zero outside masked positions
          aux_loss: scalar
        """
        B, T, D = x.shape
        out = x.new_zeros(B, T, D)
        total_aux = x.new_zeros(())

        for b in range(B):
            idx = torch.nonzero(token_mask[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            x_b = x[b:b+1, idx, :]         # [1, N_mod, D]
            y_b, aux_b = block(x_b)        # [1, N_mod, D], scalar
            out[b, idx, :] = y_b[0]
            total_aux = total_aux + aux_b

        return out, total_aux

    def forward(self, x, modality_ids):
        vision_mask = (modality_ids == 0)
        text_mask = (modality_ids == 1)
        speech_mask = (modality_ids == 2)

        has_text = bool(text_mask.any().item())
        has_vision = bool(vision_mask.any().item())
        has_speech = bool(speech_mask.any().item())

        y = torch.zeros_like(x)
        msoe_loss_dict = {}

        if has_text:
            y_text, aux_text = self._apply_subset_block(x, text_mask, self.text_block)
            y = y + y_text
            msoe_loss_dict['text'] = aux_text

        if has_vision:
            y_vision, aux_vision = self._apply_subset_block(x, vision_mask, self.vision_block)
            y = y + y_vision
            msoe_loss_dict['vision'] = aux_vision

        if has_speech:
            y_speech, aux_speech = self._apply_subset_block(x, speech_mask, self.speech_block)
            y = y + y_speech
            msoe_loss_dict['speech'] = aux_speech   

        return y, msoe_loss_dict
        

class MToE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.experts = nn.ModuleList([
            ExpertFFN(cfg.d_model, cfg.mtoe_expert_hidden, cfg.dropout) 
            for _ in range(cfg.mtoe_n_experts)
        ])

        self.router = nn.Linear(cfg.d_model, cfg.mtoe_n_experts, bias=False)
        self.dropout = nn.Dropout(cfg.mtoe_dropout)

    def compute_cmi_loss(self, probs: torch.Tensor, modality_ids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Computes the Mutual Information I(M; E) for the current batch.
        Assumes the batch belongs to a single task (T_k), effectively giving I(M; E | T_k).
        """
        B, T, E = probs.shape
        num_modalities = self.cfg.n_modalities # 0: text, 1: vision, 2: speech
        
        # 1. Compute Joint Probability P(M_i, E_j)
        joint_probs = torch.zeros(num_modalities, E, device=probs.device)
        
        for i in range(num_modalities):
            mask = (modality_ids == i)
            if mask.any():
                # Sum the routing probabilities for all tokens of this modality
                joint_probs[i] = probs[mask].sum(dim=0)
                
        # Normalize over the batch to make it a valid probability distribution
        joint_probs = joint_probs / (joint_probs.sum() + eps)
        
        # 2. Compute Marginals P(M_i) and P(E_j)
        p_m = joint_probs.sum(dim=1, keepdim=True) # Shape: [3, 1]
        p_e = joint_probs.sum(dim=0, keepdim=True) # Shape: [1, E]
        
        # 3. Compute Mutual Information: sum( P(M,E) * log(P(M,E) / (P(M)*P(E))) )
        numerator = joint_probs + eps
        denominator = (p_m * p_e) + eps
        
        # Mutual Information scalar
        mi = (joint_probs * torch.log(numerator / denominator)).sum()

        return mi

    def forward(self, x, modality_ids):
        B, T, D = x.shape
        
        # 1. Routing
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1) # Full distribution needed for MI Loss
        
        # Calculate CMI using the full probability distribution
        cmi_loss = self.compute_cmi_loss(probs, modality_ids)
        
        # Sparse Top-K Selection
        top_k_probs, top_k_indices = torch.topk(probs, self.cfg.mtoe_topk, dim=-1)
        
        # Normalize top-k weights (optional but recommended for stability in sparse MoE)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 2. Dispatch to Experts
        y = torch.zeros_like(x)
        
        # Standard sparse MoE iteration
        for k in range(self.cfg.mtoe_topk):
            # Get the indices and weights for the k-th choice of every token
            indices_k = top_k_indices[..., k] # [B, T]
            weights_k = top_k_probs[..., k]   # [B, T]
            
            for i, expert in enumerate(self.experts):
                expert_mask = (indices_k == i)
                
                if expert_mask.any():
                    # Extract tokens assigned to expert i
                    expert_input = x[expert_mask]
                    
                    # Process and weigh output
                    expert_out = expert(expert_input)
                    expert_out = expert_out * weights_k[expert_mask].unsqueeze(-1)
                    
                    # Accumulate back into the output tensor
                    y[expert_mask] += self.dropout(expert_out)

        return y, cmi_loss



class OmniSLOAFFN_old(nn.Module):
    """
    Omni-style FFN replacement with:
      - one multimodal branch
      - one text branch
      - one vision branch
      - one speech branch
      - top-k routing inside each branch
      - plain sum across active branches

    modality ids:
      0 = text
      1 = vision
      2 = speech

    Behavior when a modality is absent:
      - its modality-specific branch is not activated
      - multimodal branch still runs on all present tokens

    forward returns:
      y: [B, T, D]
      aux_loss: scalar
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.mm_block = ModalityBlock(
            d_model=cfg.d_model,
            hidden=cfg.ffn_hidden,
            dropout=cfg.dropout,
            n_experts=cfg.smola_n_experts_mm,
            topk=cfg.moe_topk,
        )
        self.text_block = ModalityBlock(
            d_model=cfg.d_model,
            hidden=cfg.ffn_hidden,
            dropout=cfg.dropout,
            n_experts=cfg.smola_n_experts_text,
            topk=cfg.moe_topk,
        )
        self.vision_block = ModalityBlock(
            d_model=cfg.d_model,
            hidden=cfg.ffn_hidden,
            dropout=cfg.dropout,
            n_experts=cfg.smola_n_experts_vision,
            topk=cfg.moe_topk,
        )
        self.speech_block = ModalityBlock(
            d_model=cfg.d_model,
            hidden=cfg.ffn_hidden,
            dropout=cfg.dropout,
            n_experts=cfg.smola_n_experts_speech,
            topk=cfg.moe_topk,
        )

        self.out_dropout = nn.Dropout(cfg.dropout)

    def _apply_subset_block(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        block: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply block only on masked token positions.

        Args:
          x: [B, T, D]
          token_mask: [B, T] bool

        Returns:
          out: [B, T, D], zero outside masked positions
          aux_loss: scalar
        """
        B, T, D = x.shape
        out = x.new_zeros(B, T, D)
        total_aux = x.new_zeros(())

        for b in range(B):
            idx = torch.nonzero(token_mask[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            x_b = x[b:b+1, idx, :]         # [1, N_mod, D]
            y_b, aux_b = block(x_b)        # [1, N_mod, D], scalar
            out[b, idx, :] = y_b[0]
            total_aux = total_aux + aux_b

        return out, total_aux

    def forward(
        self,
        x: torch.Tensor,
        mod: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        """
        Args:
          x:   [B, T, D]
          mod: [B, T], with values:
               0=text, 1=vision, 2=speech

        Returns:
          y: [B, T, D]
          aux_loss: scalar
        """
        B, T, D = x.shape
        aux_ffn_loss_dict = {}


        # multimodal branch always runs on all tokens
        y_mm, aux_mm = self.mm_block(x)

        # If modality ids are not provided, fall back to multimodal-only
        if mod is None:
            y = self.out_dropout(y_mm)
            aux_ffn_loss_dict = {"mm": aux_mm}

            if not return_stats:
                return y, aux_ffn_loss_dict

            stats = {
                "present_text": False,
                "present_vision": False,
                "present_speech": False,
                "used_mm": True,
                "used_text": False,
                "used_vision": False,
                "used_speech": False,
            }
            return y, aux_ffn_loss_dict, stats

        vision_mask = (mod == 0)
        text_mask = (mod == 1)
        speech_mask = (mod == 2)

        # Since each batch has the same input type, checking batch[0] is enough.
        # But using .any() is safer and still cheap.
        has_text = bool(text_mask.any().item())
        has_vision = bool(vision_mask.any().item())
        has_speech = bool(speech_mask.any().item())

        y = y_mm

        if has_text:
            y_text, aux_text = self._apply_subset_block(x, text_mask, self.text_block)
            y = y + y_text
            # aux_loss = aux_loss + aux_text
            aux_ffn_loss_dict['text'] = aux_text

        if has_vision:
            y_vision, aux_vision = self._apply_subset_block(x, vision_mask, self.vision_block)
            y = y + y_vision
            aux_ffn_loss_dict['vision'] = aux_vision

        if has_speech:
            y_speech, aux_speech = self._apply_subset_block(x, speech_mask, self.speech_block)
            y = y + y_speech
            aux_ffn_loss_dict['speech'] = aux_speech

        y = self.out_dropout(y)

        if not return_stats:
            return y, aux_ffn_loss_dict

        stats = {
            "present_text": has_text,
            "present_vision": has_vision,
            "present_speech": has_speech,
            "used_mm": True,
            "used_text": has_text,
            "used_vision": has_vision,
            "used_speech": has_speech,
            "n_text_tokens": int(text_mask.sum().item()),
            "n_vision_tokens": int(vision_mask.sum().item()),
            "n_speech_tokens": int(speech_mask.sum().item()),
        }
        return y, aux_ffn_loss_dict, stats



# class MoEFFN_naive(nn.Module):
#     """
#     Token-level Top-K MoE FFN.
#     - Router produces probs over experts per token.
#     - Select top-k experts per token and combine their outputs weighted by normalized probs.
#     - Includes load balance aux loss.
#     """

#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.cfg = cfg
#         self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
#         self.experts = nn.ModuleList([ExpertFFN(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
#                                       for _ in range(cfg.n_experts)])
#         self.dropout = nn.Dropout(cfg.dropout)

#     def forward(self, x: torch.Tensor, mod: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x: [B,T,D]
#         returns:
#           y: [B,T,D]
#           aux_loss: scalar
#         """
#         # mod: [B, T] modality ids for concated tokens. this is used to identify which tokens belong to which modality, 0 for text, 1 for vision, 2 for speech 

#         B, T, D = x.shape
#         logits = self.router(x)                          # [B,T,E]
#         probs = F.softmax(logits, dim=-1)                # [B,T,E]

#         mask = topk_mask(probs, self.cfg.moe_topk)       # [B,T,E]
#         probs_sel = probs * mask.float()
#         denom = probs_sel.sum(dim=-1, keepdim=True).clamp_min(1e-9)
#         weights = probs_sel / denom                      # [B,T,E] renorm

#         aux_loss = load_balance_loss_from_scores(probs, mask)

#         # Compute expert outputs; combine
#         y = x.new_zeros((B, T, D))
#         for e, expert in enumerate(self.experts):
#             w = weights[..., e].unsqueeze(-1)            # [B,T,1]
#             if torch.count_nonzero(w).item() == 0:
#                 continue
#             y = y + w * expert(x)

#         y = self.dropout(y)
#         return y, aux_loss

    
# ----------------------------
# Transformer block: MoH + MoE
# MoHA init paras: cfg, forward paras: x, modality_ids, task_ids, attn_mask, token_mask, return_aux_info
# ----------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MoHAttention(cfg)

        self.drop1 = nn.Dropout(cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        # self.ffn = MoEFFN(cfg)

        self.ffn = MoEFFN(cfg)
        
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, mod: Optional[torch.Tensor] = None, task_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_loss_block = {}

        # x: [B, T+1, D]
        a, aux_attn_dict = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.drop1(a)

        aux_loss_block['attn'] = aux_attn_dict

        f, aux_ffn_dict = self.ffn(self.ln2(x), modality_ids=mod)
        # aux_ffn_dict keys: text, vision, speech 

        x = x + self.drop2(f)
        aux_loss_block['ffn'] = aux_ffn_dict

        return x, aux_loss_block # dict of two dicts: attn aux losses and ffn aux losses

    
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
        token_features: (B, L, D)   -- all non-CLS tokens
        returns:        (B, L, vocab_size)
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
# Multi-modal, multi-task model
# ----------------------------
class MultiModalMultiTaskMoHMoE(nn.Module):
    def __init__(self, cfg: ModelConfig,
                vision_embedder: nn.Module,
                text_embedder: nn.Module,
                #  speech_embedder: nn.Module,
                task_output_dims: List[int],
                ):
        """
        task_output_dims: list length n_tasks, each is output dimension for that task head (e.g., num_classes or regression dim)
        """
        super().__init__()
        self.cfg = cfg

        self.vision_embedder = vision_embedder
        self.text_embedder = text_embedder
        # # self.speech_embedder = speech_embedder

        # Optionally freeze embedders outside this class:
        for p in self.vision_embedder.parameters():
            p.requires_grad = False

        self.modality_emb = nn.Embedding(cfg.n_modalities, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.task_emb = nn.Embedding(cfg.n_tasks, cfg.task_emb_dim)
        self.task_to_model = nn.Linear(cfg.task_emb_dim, cfg.d_model, bias=False)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        assert len(task_output_dims) == cfg.n_tasks
        img_cls_dim, img_rec_dim, txt_cls_dim, txt_rec_dim, vqa_dim = task_output_dims

        self.task_heads = nn.ModuleList([
            # task 0: img_cls  (CLS token -> class logits)
            ImageClassificationHead(cfg.d_model, img_cls_dim),

            # task 1: img_rec  (patch tokens -> reconstructed image)
            ImageReconstructionHead(
                d_model=cfg.d_model,
                img_size=cfg.img_size,       # e.g. 224
                patch_size=cfg.patch_size,   # e.g. 16
            ),

            # task 2: txt_cls (CLS token -> class logits)
            TextClassificationHead(cfg.d_model, txt_cls_dim),

            # task 3: txt_rec (token features -> vocab logits)
            TextReconstructionHead(cfg.d_model, txt_rec_dim),  # txt_rec_dim = vocab_size

            # task 4: vqa (token features -> answer logits)
            VQAHead(cfg.d_model, vqa_dim),
        ])

        self.dropout = nn.Dropout(cfg.dropout)
        self.task_token = nn.Parameter(torch.randn(1, 1, cfg.d_model)) 

    def forward(
        self,
        vision_tokens,  # [B, Tv, in_v] or already embedded [B,Tv,d_model]
        text_tokens,    # [B, Tt, in_t]
        # speech_tokens: Optional[torch.Tensor],  # [B, Ts, in_s]
        task_id: torch.Tensor,                  # [B] long
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          logits: [B, task_output_dim(task_id)]
          aux_loss: scalar
        """

        B = task_id.size(0) # batch size
        device = task_id.device

        parts = []
        modality_ids = []
        Tv = Tt = Ts = 0

        # Embed each modality into d_model token sequences
        if vision_tokens is not None:
            xv = self.vision_embedder(vision_tokens)    # [B,Tv,D]
            parts.append(xv)
            modality_ids.append(torch.full((B, xv.size(1)), 0, device=device, dtype=torch.long)) #append [B,Tv] of 0s
            # Tv length

        if text_tokens is not None:
            xt = self.text_embedder(text_tokens)         # [B,Tt,D]
            parts.append(xt)
            modality_ids.append(torch.full((B, xt.size(1)), 1, device=device, dtype=torch.long))

        # if speech_tokens is not None:
        #     xs = self.speech_embedder(speech_tokens)     # [B,Ts,D]
        #     parts.append(xs)
        #     modality_ids.append(torch.full((B, xs.size(1)), 2, device=device, dtype=torch.long))
        #     Ts = xs.size(1)

    
        if len(parts) == 0:
            raise ValueError("At least one modality must be provided.")

        x = torch.cat(parts, dim=1)                      # [B,T,D]
        mod = torch.cat(modality_ids, dim=1)             # [B,T]

        T = x.size(1)
        if T > self.cfg.max_seq_len: # max = 2048
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        # Add modality + position embeddings
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B,T]

        x = x + self.modality_emb(mod) + self.pos_emb(pos)

        # Inject task embedding as an additive bias on all tokens (simple and effective)
        tvec = self.task_to_model(self.task_emb(task_id))               # [B,D]
        x = x + tvec.unsqueeze(1)

        x = self.dropout(x)

        # aux = x.new_zeros(())
        aux_loss_list = [] # list of dicts from each block, to be summed later
        # aux_loss_list[i] = {'attn': {...}, 'ffn': {...}} for block i

        # Pass through MoH + MoE blocks, accumulating aux losses
        x = torch.cat([self.task_token.expand(B, -1, -1), x], dim=1)

        for blk in self.blocks:
            x, aux_blk = blk(x, attn_mask=attn_mask, mod=mod, task_ids=task_id)
            aux_loss_list.append(aux_blk)

        final_loss_aux_dict = loss_parser(aux_loss_list, device=device)

        x = self.ln_f(x)
        tid = int(task_id[0].item())

        if tid == 0:  # task img_cls
            features = x[:, 1, :]  
        elif tid == 1:  # task img_rec
            features = x[:, 2:, :]  
        elif tid == 2:  # task txt_cls
            features = x[:, 1, :]
        elif tid == 3:  # task txt_rec
            features = x[:, 2:, :]
        elif tid == 4:  # task vqa
            features = x  # use all tokens for VQA
        else:
            raise ValueError(f"Unexpected task_id {tid}.")

        # task_heads[tid] should know whether it expects (B, D) or (B, L, D)
        # depending on task type (cls vs rec).
        logits = self.task_heads[tid](features)

        return {
            "logits": logits,
            "aux_loss_attn_ffn": final_loss_aux_dict,  # dict keys: attn_lb, ffn_mi
        }
