import torch 
import torch.nn as nn
from config import ModelConfig, RunConfig

from wireless_utils import ComplexWirelessChannel
from base_models import ImageClassificationHead, ImageReconstructionHead, TextClassificationHead, TextReconstructionHead, VQAHead

from base_models import MoHAttention, ExpertFFN, ModalityBlock, MSoE, CrossModalExpert, MoEFFN, Block, SharedSNRTaskRouter

from utils import loss_parser

class SemanticEncoder(nn.Module):
    def __init__(self, cfg, run_cfg, vision_embedder, text_embedder, speech_embedder=None):
        super().__init__()
        self.cfg = cfg
        self.run_cfg = run_cfg 
        self.active_tasks = run_cfg.task_selection
        self.task_output_dims = cfg.task_output_dims

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

        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.task_emb = nn.Embedding(cfg.n_tasks, cfg.d_model)
        self.modality_emb = nn.Embedding(cfg.n_modalities, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.fusion_ln = nn.LayerNorm(cfg.d_model)


    def forward(self, vision_tokens, text_tokens, speech_tokens, task_name, attn_mask=None):
        B = len(vision_tokens) if vision_tokens is not None else len(text_tokens)

        device = next(self.parameters()).device

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
            emb = self.embedders[mod_name](raw_data, task_name)
            parts.append(emb)
            T_mod = emb.size(1)
            modality_bounds[mod_name] = (current_len, current_len + T_mod)
                
            current_len += T_mod

        total_len = current_len

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
        modality_ids = torch.zeros((B, T), device=device, dtype=torch.long) # 1 for text, 0 for vision

        for mod_name, mask in modality_masks_dict.items():
            mod_idx = self.cfg.active_modalities.index(mod_name)
            modality_ids[mask] = mod_idx
        
        task_emb_tensor = self.task_emb(
            torch.tensor(self.run_cfg.task_selection.index(task_name), device=device)
        ).view(1, 1, -1).expand(B, T, -1)  

        x = x + self.pos_emb(pos_emb_tensor) + self.modality_emb(modality_ids) + task_emb_tensor
        # x = x + self.pos_emb(pos_emb_tensor) + self.modality_emb(modality_ids)

        # 3. Transformer Blocks
        attn_mask = (1.0 - pad_mask) * torch.finfo(x.dtype).min
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        aux_loss_list = []        
        for blk in self.blocks:
            x, aux_blk = blk(x, pad_mask, modality_masks_dict, attn_mask)
            aux_loss_list.append(aux_blk)

        x = self.ln_f(x)

        if task_name in ['img_cls', 'txt_cls',]: # 'vqa'
            features = x[:, 0, :]
        elif task_name in ['vqa']:
            features = x     
        else:
            features = x[:, 1:, :]

        # parsed_loss_dict = loss_parser(aux_loss_list, device=device)

        return {
            'features': features,
            'sem_encd_loss': aux_loss_list
        }

class ChannelEncoder(nn.Module):
    def __init__(self, cfg, run_cfg):
        super().__init__()
        self.d_model = cfg.d_model
        
        self.expert_dims = [ # output dimensions for each expert
            self.d_model // 8,  
            self.d_model // 6,  
            self.d_model // 4,  
            self.d_model // 2,  
        ]
        
        # We want the total parameters for each expert to be roughly the same.
        # Target params for the 'standard' expert (d_model -> d_model -> d_model) is 2 * d_model^2 from (d_model * hidden) + (hidden * out_dim) = 2 * d_model^2
        # hidden * (d_model + out_dim) = 2 * d_model^2
        # hidden = (2 * d_model^2) // (d_model + out_dim)
        # this design beats single mlp as a channel encoder
        
        self.experts = nn.ModuleList()
        for out_dim in self.expert_dims:
            # Calculate a hidden dim that balances the parameter count
            target_params = 2 * (self.d_model ** 2)
            hidden_dim = target_params // (self.d_model + out_dim)
            
            self.experts.append(
                nn.Sequential(
                    nn.Linear(self.d_model, hidden_dim), 
                    nn.GELU(), 
                    nn.Linear(hidden_dim, out_dim)
                )
            )

    def forward(self, x, route_weights):
        out = None
        for i in range(len(self.experts)):
            if route_weights[0, i] == 1.0:
                out = self.experts[i](x) * route_weights[0, i]
                break
        return out

class SimpleChannelEncoder(nn.Module):
    def __init__(self, cfg, run_cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.hidden_dim = self.d_model * 4  # Example hidden dimension
        self.output_dim = self.d_model // 2  # Example output dimension
        
        self.encoder = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    
class ChannelDecoder(nn.Module):
    def __init__(self, cfg, run_cfg):
        super().__init__()
        self.d_model = cfg.d_model
        
        self.expert_dims = [
            self.d_model // 8,  
            self.d_model // 6,  
            self.d_model // 4,  
            self.d_model // 2,   
        ]
        
        self.experts = nn.ModuleList()
        
        for in_dim in self.expert_dims:
            target_params = 2 * (self.d_model ** 2)
            hidden_dim = target_params // (in_dim + self.d_model)
            
            self.experts.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.d_model)
                )
            )
        
    def forward(self, x, route_weights):
        out = None
        for i in range(len(self.experts)):
            if route_weights[0, i] == 1.0:
                out = self.experts[i](x) * route_weights[0, i]
                break
        return out

class SimpleChannelDecoder(nn.Module):
    def __init__(self, cfg, run_cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.hidden_dim = self.d_model * 4  # Example hidden dimension
        self.input_dim = self.d_model // 2  # Example input dimension
        
        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.d_model)
        )

    def forward(self, x):
        return self.decoder(x)

#to-do: temporary, including task heads only for tasks
class SemanticDecoder(nn.Module):
    def __init__(self, cfg, run_cfg):
        super().__init__()
        
        self.cfg = cfg
        self.run_cfg = run_cfg
        
        self.task_output_dims = cfg.task_output_dims  
        self.task_heads = nn.ModuleDict({
            'img_cls': ImageClassificationHead(cfg.d_model, self.task_output_dims['img_cls']),
            'img_rec': ImageReconstructionHead(d_model=cfg.d_model, img_size=cfg.img_size, patch_size=cfg.patch_size),
            'txt_cls': TextClassificationHead(cfg.d_model, self.task_output_dims['txt_cls']),
            'txt_rec': TextReconstructionHead(cfg.d_model, self.task_output_dims['txt_rec']),
            'vqa': VQAHead(cfg.d_model, self.task_output_dims['vqa'])
        })

    def forward(self, features, task_name):
        if task_name in self.task_heads:
            return self.task_heads[task_name](features)
        else:
            raise ValueError(f"Task {task_name} not found in task heads.")

class MoAMoH_SemCom(nn.Module):
    def __init__(self, cfg, run_cfg, vision_embedder, text_embedder, speech_embedder=None, snr_ware=True):
        super().__init__()
        self.cfg = cfg
        self.run_cfg = run_cfg

        self.semantic_encoder = SemanticEncoder(cfg, run_cfg, vision_embedder, text_embedder, speech_embedder)
        
        self.channel_encoder = ChannelEncoder(cfg, run_cfg)
        self.channel_decoder = ChannelDecoder(cfg, run_cfg)

        self.channel_router = SharedSNRTaskRouter(cfg)
        
        self.semantic_decoder = SemanticDecoder(cfg, run_cfg)
        self.physical_channel = ComplexWirelessChannel(snr_dB=run_cfg.snr_dB, fading=run_cfg.fading, rician_k=run_cfg.rician_k)

        # self.tx_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)

    def forward(self, vision_tokens, text_tokens, speech_tokens=None, task_name=None, attn_mask=None, temperature=1.0):
        device = next(self.parameters()).device

        aux_loss_dict = {}

        # 1. Semantic Encoding
        semantic_out = self.semantic_encoder(vision_tokens, text_tokens, speech_tokens, task_name=task_name)
        # semantic_encoded = self.tx_norm(semantic_out['features'])

        semantic_encoded = semantic_out['features']
        aux_loss_dict['semantic_encoder'] = semantic_out['sem_encd_loss']

        task_idx = self.run_cfg.task_selection.index(task_name)
        task_emb = self.semantic_encoder.task_emb(torch.tensor([task_idx], device=device))

        # route_weights, expected_bw_cost = self.channel_router(
        #     snr=self.run_cfg.snr_dB, 
        #     task_emb=task_emb, 
        #     temperature=temperature
        # )
        # aux_loss_dict['channel_encoder'] = {'router_lb': expected_bw_cost}

        # 2. Variable-Dimension Channel Encoding (Adaptive SNR-Task Router)
        # We pass task_name and temperature to enable the differentiable HMoE
        # channel_encoded = self.channel_encoder(semantic_encoded, route_weights)
        
        # 3. Analog Transmission
        # rx_signal = self.physical_channel(
        #     channel_encoded, 
        #     snr=self.run_cfg.snr_dB, 
        #     fading=self.run_cfg.fading, 
        #     rician_k=self.run_cfg.rician_k
        # )

        # rx_signal = self.physical_channel(
        #     semantic_encoded, 
        #     snr=self.run_cfg.snr_dB, 
        #     fading=self.run_cfg.fading, 
        #     rician_k=self.run_cfg.rician_k
        # )

        # channel_decoded = self.channel_decoder(rx_signal, route_weights)

        # 5. Final Semantic Decoding
        semantic_decoded = self.semantic_decoder(semantic_encoded, task_name)
        # semantic_decoded = self.semantic_decoder(rx_signal, task_name) # rx_signal
        # semantic_decoded = self.semantic_decoder(channel_decoded, task_name)


        # this step should return single-level dict 
        parsed_loss = loss_parser(aux_loss_dict, device=device)

        return {
            "logits": semantic_decoded,
            "aux_losses": parsed_loss,
            # "x_complex": x_complex,
            # "y_noisy": y_noisy
        }



        








        