import torch 
import torch.nn as nn
from config import ModelConfig, RunConfig
from base_models import Block as TransformerBlock
from base_models import ImageClassificationHead, ImageReconstructionHead, TextClassificationHead, TextReconstructionHead, VQAHead

from wireless_utils import ComplexWirelessChannel


class SemanticEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 

        # embedding biases
        self.modality_emb = nn.Embedding(cfg.n_modalities, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.task_emb = nn.Embedding(cfg.n_tasks, cfg.task_emb_dim)
        self.task_to_model = nn.Linear(cfg.task_emb_dim, cfg.d_model, bias=False)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # img_cls_dim, img_rec_dim, txt_cls_dim, txt_rec_dim, vqa_dim = self.cfg.task_output_dims

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, modality_ids=None, task_id=None):
        B = task_id.size(0)
        device = task_id.device

        mod = torch.cat(modality_ids, dim=1)
        T = x.size(1)

        if T > self.cfg.max_seq_len: # max = 2048
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T) # (B, T)

        x = x + self.modality_emb(mod) + self.pos_emb(pos) + self.task_to_model(self.task_emb(task_id))

        x = self.dropout(x)

        aux_loss_list = []
        for blk in self.blocks:
            x, aux_blk = blk(x, attn_mask=None, mod=mod, task_ids=task_id)
            aux_loss_list.append(aux_blk)

        final_loss_aux_dict = {}
        for aux_blk in aux_loss_list:
            for key, val in aux_blk['attn'].items():
                final_loss_aux_dict[f'attn_{key}'] = final_loss_aux_dict.get(f'attn_{key}', 0) + val
            for key, val in aux_blk['ffn'].items():
                final_loss_aux_dict[f'ffn_{key}'] = final_loss_aux_dict.get(f'ffn_{key}', 0) + val

        x = self.ln_f(x)

        tid = int(task_id[0].item())
        if tid == 0:  # task img_cls
            features = x[:, 0, :]  
        elif tid == 1:  # task img_rec
            features = x[:, 1:, :]  
        elif tid == 2:  # task txt_cls
            features = x[:, 0, :]
        elif tid == 3:  # task txt_rec
            features = x[:, 1:, :]
        elif tid == 4:  # task vqa
            features = x  # use all tokens for VQA

        return {
            'features': features,
            'sem_encd_loss': final_loss_aux_dict
        }

class ChannelEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.channel_hidden),
            nn.ReLU(),
            nn.Linear(cfg.channel_hidden, cfg.transmit_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ChannelDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(cfg.transmit_dim, cfg.channel_hidden),
            nn.ReLU(),
            nn.Linear(cfg.channel_hidden, cfg.d_model)
        )

    def forward(self, x):
        return self.decoder(x)

# temp decoder, including task heads only for tasks
class SemanticDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        img_cls_dim, img_rec_dim, txt_cls_dim, txt_rec_dim, vqa_dim = cfg.task_output_dims  
        self.heads = nn.ModuleDict({
            'img_cls': ImageClassificationHead(cfg.d_model, img_cls_dim),
            'img_rec': ImageReconstructionHead(d_model=cfg.d_model, img_size=cfg.img_size, patch_size=cfg.patch_size),
            'txt_cls': TextClassificationHead(cfg.d_model, txt_cls_dim),
            'txt_rec': TextReconstructionHead(cfg.d_model, txt_rec_dim),
            'vqa': VQAHead(cfg.d_model, vqa_dim)
        })


    def forward(self, features, task_id):
        tid = int(task_id[0].item())
        if tid == 0:  # task img_cls
            out = self.heads['img_cls'](features)
        elif tid == 1:  # task img_rec
            out = self.heads['img_rec'](features)
        elif tid == 2:  # task txt_cls
            out = self.heads['txt_cls'](features)
        elif tid == 3:  # task txt_rec
            out = self.heads['txt_rec'](features)
        elif tid == 4:  # task vqa
            out = self.heads['vqa'](features)

        return out

class MoAMoH_SemCom(nn.Module):
    def __init__(self, cfg, run_cfg, vision_embedder, text_embedder, speech_embedder=None, snr_ware=True):
        # Blocks: frozen embedders -> semantic encoder -> channel encoder -> wireless -> channel decoder -> semantic decoder -> task-specific heads

        super().__init__()
        self.cfg = cfg
        self.run_cfg = run_cfg

        self.vision_embedder = vision_embedder
        self.text_embedder = text_embedder
        self.speech_embedder = speech_embedder

        for p in self.vision_embedder.parameters():
            p.requires_grad = False

        self.semantic_encoder = SemanticEncoder(cfg)
        self.channel_encoder = ChannelEncoder(cfg)
        self.channel_decoder = ChannelDecoder(cfg)
        self.semantic_decoder = SemanticDecoder(cfg)

        self.physical_channel = ComplexWirelessChannel(snr_dB=run_cfg.snr_dB, fading=run_cfg.fading, rician_k=run_cfg.rician_k)


    def forward(self, vision_tokens, text_tokens, speech_tokens=None, task_id=None, attn_mask=None):
        B = task_id.size(0)
        device = task_id.device

        parts = []
        modality_ids = []

        if vision_tokens is not None:
            xv = self.vision_embedder(vision_tokens)
            parts.append(xv)
            modality_ids.append(torch.full((B, xv.size(1)), 0, dtype=torch.long, device=device))  # modality_id 0 for vision

        if text_tokens is not None:
            xt = self.text_embedder(text_tokens)
            parts.append(xt)
            modality_ids.append(torch.full((B, xt.size(1)), 1, dtype=torch.long, device=device))  # modality_id 1 for text

        if speech_tokens is not None:
            xs = self.speech_embedder(speech_tokens)
            parts.append(xs)
            modality_ids.append(torch.full((B, xs.size(1)), 2, dtype=torch.long, device=device))  # modality_id 2 for speech

        x = torch.cat(parts, dim=1)
        # mod = torch.cat(modality_ids, dim=1)

        T = x.size(1)

        # semantic_out: dict{features, loss_dict}
        semantic_out = self.semantic_encoder(x, modality_ids=modality_ids, task_id=task_id)

        semantic_encoded = semantic_out['features']
        sem_encd_loss = semantic_out['sem_encd_loss']

        channel_encoded = self.channel_encoder(semantic_encoded)

        rx_signal, x_complex, y_noisy = self.physical_channel(channel_encoded, snr=self.run_cfg.snr_dB, fading=self.run_cfg.fading, rician_k=self.run_cfg.rician_k)

        channel_decoded = self.channel_decoder(rx_signal)

        semantic_decoded = self.semantic_decoder(channel_decoded, task_id)

        return semantic_decoded, sem_encd_loss, x_complex, y_noisy



        








        