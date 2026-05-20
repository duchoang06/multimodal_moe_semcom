from dataclasses import dataclass, field

@dataclass
class RunConfig:
    task_selection: list = field(default_factory=lambda: ['img_rec', 'img_cls', 'vqa', 'txt_rec', 'txt_cls']) # 'spc'
    train_batch_size: int = 64
    test_batch_size: int = 64

    steps_per_epoch: int = 1000

    eval_steps: int = 500

    num_epochs: int = 50 

    vocab_size: int = 30522
    vqa_answer_vocab_size: int = 3000
    task_output_dims: list = field(init=False) 

    snr_dB: float = 15.0
    fading: str = 'none'  # options: 'none', 'rayleigh', 'rician'
    rician_k: float = 3.0 

    def __post_init__(self):
        self.task_output_dims = [10, None, 2, self.vocab_size, self.vqa_answer_vocab_size]


@dataclass
class ModelConfig:
    d_model: int = 512
    n_layers: int = 4
    n_modalities = 2

    # image size
    img_size: int = 224
    patch_size: int = 16

    #--- Semantic encoder: Attention
    attn_n_heads: int = 8
    attn_head_dim: int = 64          # usually d_model = n_heads * head_dim
    attn_topk: int = 2          # routed heads active per token
    attn_shared_heads: int = 2   # always-on shared heads (first hs heads)
    attn_router_dropout: float = 0.1

    #--- Semantic encoder: FFN 
    msoe_modality_blk_hidden: int = 4 * d_model
    msoe_topk: int = 2

    msoe_n_experts_text: int = 4
    msoe_n_experts_vision: int = 4
    msoe_n_experts_speech: int = 4

    mtoe_n_experts: int = 8
    mtoe_expert_hidden: int = 4 * d_model
    mtoe_dropout: float = 0.1
    mtoe_topk: int = 2

    cmi_alpha: float = 1.0

    # Embeddings
    max_seq_len: int = 512
    n_modalities: int = 3       # v,t,s
    n_tasks: int = 4
    task_emb_dim: int = 256
    dropout: float = 0.1

    # Channel encoder/decoder
    channel_hidden: int = 0
    transmit_dim: int = 0

    # Loss weights
    attn_lb_weight: float = 0.001 # set to 0.0 to remove from aux loss dict
    # attn_mi_weight: float = 0.0

    ffn_mi_weight: float = 2.0 


    # Final loss weights for each task
    img_rec_loss_weight: float = 1.0
    img_cls_loss_weight: float = 1.0
    vqa_loss_weight: float = 1.0
    txt_rec_loss_weight: float = 1.0
    txt_cls_loss_weight: float = 1.0
    spc_loss_weight: float = 1.0

    # smola_alpha_init: float = 1.0
    # smola_router_eps: float = 1e-6

    def __post_init__(self):
        self.channel_hidden = self.d_model * 4
        self.transmit_dim = self.d_model // 4
