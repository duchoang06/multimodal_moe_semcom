from dataclasses import dataclass, field

@dataclass
class RunConfig:
    task_selection: list = field(default_factory=lambda: ['img_rec', 'img_cls', 'vqa', 'txt_rec', 'txt_cls']) # 'spc'
    sample_task_probs: list = field(default_factory=lambda: [0.15, 0.05, 0.6, 0.15, 0.05])

    # task_selection: list = field(default_factory=lambda: ['vqa'])
    # sample_task_probs: list = field(default_factory=lambda: [1.0])

    train_batch_size: int = 64
    test_batch_size: int = 64

    steps_per_epoch: int = 1000

    eval_steps: int = 500

    num_epochs: int = 25

    vocab_size: int = 30522
    # task_output_dims: list = field(init=False) 

    snr_dB: float = 20.0
    fading: str = 'none'  # options: 'none', 'rayleigh', 'rician'
    rician_k: float = 3.0 

    def __post_init__(self):
        pass 


@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 8
    n_modalities: int = 2
    active_modalities: list = field(default_factory=lambda: ['vision', 'text']) # modality indexes do matter here

    task_output_dims: dict[str, int] = field(default_factory=lambda: {
        'img_cls': 10,  # img_cls
        'img_rec': None,  # img_rec (handled by reconstruction head)
        'txt_cls': 2,  # txt_cls
        'txt_rec': 30522,  # txt_rec
        'vqa': 3129,  # vqa (assuming 3129 possible answers)
    })

    n_tasks: int = 5

    max_seq_len: int = 512

    # image size
    img_size: int = 224
    patch_size: int = 16

    #--- Semantic encoder: Attention
    attn_n_heads: int = 12
    attn_head_dim: int = 64          # d_model = n_heads * head_dim
    attn_topk: int = 2          # routed heads active per token
    attn_shared_heads: int = 2   # always-on shared heads
    attn_router_dropout: float = 0.1

    #--- Semantic encoder: FFN 
    msoe_modality_blk_hidden: int = 2 * d_model
    msoe_topk: int = 1

    # msoe_n_experts_text: int = 4
    # msoe_n_experts_vision: int = 4
    # msoe_n_experts_speech: int = 4
    msoe_n_experts: int = 4

    mtoe_expert_hidden: int = 4 * d_model
    crossmodal_threshold: float = 0.5
    # mtoe_dropout: float = 0.1

    # Embeddings
    # task_emb_dim: int = 256
    dropout: float = 0.1

    # Channel encoder/decoder
    channel_hidden: int = 0
    transmit_dim: int = 0

    # Loss weights
    se_attn_lb_weight: float = 1.0 # set to 0.0 to remove from total loss
    se_ffn_mod_lb_weight: float = 1.0
    se_ffn_compute_weight: float = 1.0
    se_ffn_align_weight: float = 50.0

    ce_router_lb_weight : float = 1e-2


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
