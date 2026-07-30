from typing import Optional
import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random, os
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Utils
# ----------------------------
def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    scores: [B, T, N]
    returns mask: [B, T, N] boolean where top-k are True
    """
    if k >= scores.size(-1):
        return torch.ones_like(scores, dtype=torch.bool)
    topk_idx = torch.topk(scores, k, dim=-1).indices  # [B,T,k]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    return mask

# def load_balance_loss_from_scores(
#     probs: torch.Tensor,  # softmax router probs [B,T,N]
#     selected_mask: torch.Tensor,  # hard selection mask [B,T,N] bool
# ) -> torch.Tensor:
#     """
#     A common MoE-style load balance loss:
#       L = N * sum_i (P_i * f_i)
#     where:
#       P_i = mean_t probs[..., i]
#       f_i = mean_t 1[selected]
#     """
#     B, T, N = probs.shape
#     P_i = probs.mean(dim=(0, 1))                           # [N]
#     f_i = selected_mask.float().mean(dim=(0, 1))           # [N]
#     return (N * (P_i * f_i).sum())


# ----------------------------
# Causal mask helper (optional)
# ----------------------------

def build_causal_mask(B: int, T: int, device: torch.device) -> torch.Tensor:
    """
    returns additive mask [B, 1, T, T] with -inf above diagonal
    """
    m = torch.full((T, T), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m.view(1, 1, T, T).expand(B, 1, T, T)



class QQPPromptDataset(Dataset):
    def __init__(self, hf_dataset_split):
        self.q1 = hf_dataset_split["question1"]
        self.q2 = hf_dataset_split["question2"]
        self.labels = hf_dataset_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Embed the questions in a prompt-like sentence
        prompt = f"Are these questions asking the same thing? Question1: {self.q1[idx]} Question2: {self.q2[idx]}"
        label = self.labels[idx]
        return prompt, label

def text_loss(outputs, labels, task_id, input_ids=None, input_lengths=None):
    loss = 0.0
    if task_id == 0:
        logit = outputs
        label = labels
        loss = F.cross_entropy(logit, label)

    elif task_id == 1:
        target_len = outputs.size(1)

        pred_logits = outputs.transpose(1, 2)  # [batch, vocab_size, seq_len]
        target_tokens = input_ids[:, :target_len] 

        loss = F.cross_entropy(pred_logits, target_tokens, ignore_index=0)

    else:
        raise ValueError(f"Unknown task_id: {task_id}")
                
    return loss

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            self.linear(input_dim, hidden_dim),
            nn.ReLU(),
            self.linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.linear(hidden_dim, 1),
        )

    def linear(self, in_dim, out_dim, bias=True):
        lin = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.normal_(lin.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(lin.bias)
        return lin

    def forward(self, inputs):
        return self.net(inputs)

def mutual_information_loss(tx_signal, rx_signal, critic):
    joint, marginal = sample_batch(tx_signal, rx_signal)

    t = critic(joint)
    et = torch.exp(critic(marginal)) 

    mi_loss = torch.mean(t) - torch.log(torch.mean(et) + 1e-8)

    return -mi_loss

def sample_batch(rec, noise):
    rec = torch.reshape(rec, shape=(-1, 1))
    noise = torch.reshape(noise, shape=(-1, 1))
    rec_sample1, rec_sample2 = torch.split(rec, int(rec.shape[0]/2), dim=0)
    noise_sample1, noise_sample2 = torch.split(noise, int(noise.shape[0]/2), dim=0)
    joint = torch.cat((rec_sample1, noise_sample1), 1)
    marg = torch.cat((rec_sample1, noise_sample2), 1)
    return joint, marg


def moe_balancing_loss_p_penalty(all_gate_scores, all_expert_masks, expert_sizes):
    '''
        parameter-penalty loss adapted from "HMoE: Heterogeneous Mixture of Experts for Language Modeling"
    '''
    # all_expert_mask: (num_layers, num_tokens, num_experts)
    loss = 0.0
    N = expert_sizes.shape[0]

    for gate_scores, expert_mask in zip(all_gate_scores, all_expert_masks):
        T, N_layer = gate_scores.shape
        assert N_layer == N, "Mismatch in number of experts"

        # P = torch.softmax(gate_scores, dim=-1)  # (T, N)
        P = gate_scores # already softmaxed in the model

        P_hat = P.mean(dim=0)  # (N,)
        M = (expert_mask.float() * expert_sizes.view(1, -1)).mean(dim=0)  # (N,)
        loss += N * torch.sum(M * P_hat)

    return loss #to-do: may return 0.0 here
    # return torch.tensor(0.0)

def moe_balancing_loss(all_gate_scores, all_expert_masks, expert_sizes):
    '''
    Balancing loss for HMoE: encourages routing proportional to expert sizes.
    '''
    loss = 0.0
    ideal_load = expert_sizes / expert_sizes.sum()  # (N,)

    for gate_scores in all_gate_scores:  # gate_scores: (T, N), softmaxed
        actual_load = gate_scores.mean(dim=0)  # (N,)
        # Small constant for numerical stability
        actual_load = actual_load + 1e-8
        loss += torch.nn.functional.kl_div(actual_load.log(), ideal_load, reduction='batchmean')

    return loss


def snr_loss(semantic_decoded, semantic_encoded, snr_tensor):
    """
    Encourages the decoded semantic features to remain close to the
    original (pre-channel) semantic representation, especially at high SNR.
    Args:
        semantic_decoded: [batch, dim] - output of the decoder
        semantic_encoded: [batch, dim] - output before the channel (clean)
        snr_tensor:       [batch, 1]   - SNR in dB for each sample

    Returns:
        scalar loss (mean over batch)
    """
    with torch.no_grad():
        clean_feat = semantic_encoded.detach()

    snr_linear = 10 ** (snr_tensor / 10)  # convert dB to linear scale
    weight = 1 / (snr_linear + 1e-6)      # lower SNR -> weaker penalty

    loss = ((semantic_decoded - clean_feat) ** 2).mean(dim=1) * weight.squeeze()
    return loss.mean()



def fix_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def sample_mixed_task_batch(data, batch_size=8):
    batch = random.sample(list(data), batch_size)

    batch_dicts = []
    for i, sample in enumerate(batch):
        task_id = 0 if i < batch_size // 2 else 1  # First half: classification, second: reconstruction
        batch_dicts.append({
            'text': sample['sentence'],
            'label': sample['label'],   # Only used for classification
            'task_id': task_id
        })

    # Shuffle to mix task types
    random.shuffle(batch_dicts)

    # Convert to lists/tensors
    texts = [item['text'] for item in batch_dicts]
    task_ids = torch.tensor([item['task_id'] for item in batch_dicts])
    labels = torch.tensor([item['label'] for item in batch_dicts])

    return texts, labels, task_ids

def sample_single_task_batch(task_id, data, batch_size=8):
    batch = random.sample(list(data), batch_size)

    batch_dicts = []
    for sample in batch:
        batch_dicts.append({
            'text': sample['sentence'],
            'label': sample['label'],  # Used only for classification
            'task_id': task_id
        })

    texts = [item['text'] for item in batch_dicts]
    task_ids = torch.tensor([item['task_id'] for item in batch_dicts])
    labels = torch.tensor([item['label'] for item in batch_dicts])

    return texts, labels, task_ids


def estimate_transformer_flops(num_blocks, layers_per_block, d_model, d_ff, seq_len, batch_size= 1):
    # Attention FLOPs per layer (Q, K, V, attn score, softmax, context proj)
    flops_attention = 4 * seq_len * d_model**2 + 2 * seq_len**2 * d_model

    # Feedforward FLOPs per layer
    flops_ffn = 4 * seq_len * d_model * d_ff

    # FLOPs per transformer encoder layer
    flops_per_layer = flops_attention + flops_ffn

    # Total layers
    total_layers = num_blocks * layers_per_block

    # Multiply by total layers and batch
    total_flops = flops_per_layer * total_layers * batch_size

    return total_flops

def l2_normalize(x: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

def final_loss_scaler(task_name: str, base_loss: torch.Tensor) -> torch.Tensor:
    'scaling the final loss value based on the task'
    if task_name == "img_cls":
        return base_loss  # no scaling

    elif task_name == "img_rec":
        return base_loss  

    elif task_name == "txt_cls":
        return base_loss  

    elif task_name == "txt_rec":
        return base_loss 

    elif task_name == "vqa":
        return base_loss * 2.0

    else:
        raise ValueError(f"Unknown task name: {task_name}")


def attn_load_balance(
    gates: torch.Tensor,   # [B*T, hr]
    mask: torch.Tensor,    # [B*T, hr]
    modality_mask: torch.Tensor, # [B*T]
    pad_mask: Optional[torch.Tensor] = None, 
) -> torch.Tensor:
    '''
        encouraging uniform routing across experts
    '''
    num_tokens, num_routed_heads = gates.shape

    active_tokens = modality_mask.float() 

    if pad_mask is not None:
        active_tokens = active_tokens * pad_mask.float()

    valid_count = active_tokens.sum() + 1e-5

    P = (gates * active_tokens.unsqueeze(-1)).sum(dim=0) / valid_count
    f = (mask * active_tokens.unsqueeze(-1)).sum(dim=0) / valid_count

    loss = torch.sum(P * f) * num_routed_heads
    return loss

def loss_parser_nonsemcom(losses, device):
    num_blks = len(losses)
    
    attn_lb_loss = torch.tensor(0.0, device=device)
    ffn_compute_loss = torch.tensor(0.0, device=device)
    ffn_align_loss = torch.tensor(0.0, device=device)
    ffn_mod_lb_loss = torch.tensor(0.0, device=device)
    shared_router_lb_loss = torch.tensor(0.0, device=device)

    for i in range(num_blks):
        blk_loss = losses[i]
        attn_lb_loss += blk_loss.get('attn', {}).get('lb', 0.0)
        # ffn_mi_loss += blk_loss.get('ffn', {}).get('mi', 0.0)

        # print(blk_loss['ffn']['mi'])
        ffn_compute_loss += blk_loss.get('ffn', {}).get('compute', 0.0)
        ffn_align_loss += blk_loss.get('ffn', {}).get('align', 0.0)
        ffn_mod_lb_loss += blk_loss.get('ffn', {}).get('mod_lb', 0.0)

    #single-level dict only 
    loss_dict = {
        'attn_lb': attn_lb_loss / num_blks,
        'ffn_compute': ffn_compute_loss / num_blks,
        'ffn_align': ffn_align_loss / num_blks,
        'ffn_mod_lb': ffn_mod_lb_loss / num_blks,
    }

    return loss_dict

def loss_parser(losses, device):
    se_attn_lb_loss = torch.tensor(0.0, device=device)
    se_ffn_compute_loss = torch.tensor(0.0, device=device)
    se_ffn_align_loss = torch.tensor(0.0, device=device)
    se_ffn_mod_lb_loss = torch.tensor(0.0, device=device)
    ce_router_lb_loss = torch.tensor(0.0, device=device)

    # semantic encoder parsing
    semantic_encoder_loss = losses.get('semantic_encoder', None)
    num_blks_encoder = len(semantic_encoder_loss)
    for blk_loss in semantic_encoder_loss:
        se_attn_lb_loss += blk_loss.get('attn', {}).get('lb', 0.0)
        # ffn_mi_loss += blk_loss.get('ffn', {}).get('mi', 0.0)

        # print(blk_loss['ffn']['mi'])
        se_ffn_compute_loss += blk_loss.get('ffn', {}).get('compute', 0.0)
        se_ffn_align_loss += blk_loss.get('ffn', {}).get('align', 0.0)
        se_ffn_mod_lb_loss += blk_loss.get('ffn', {}).get('mod_lb', 0.0)

    # channel encoder parsing
    channel_encoder_loss = losses.get('channel_encoder', None)
    ce_router_lb_loss = channel_encoder_loss.get('router_lb', 0.0)


    #single-level dict only 
    loss_dict = {
        'se_attn_lb': se_attn_lb_loss / num_blks_encoder,
        'se_ffn_compute': se_ffn_compute_loss / num_blks_encoder,
        'se_ffn_align': se_ffn_align_loss / num_blks_encoder,
        'se_ffn_mod_lb': se_ffn_mod_lb_loss / num_blks_encoder,
        'ce_router_lb': ce_router_lb_loss,
    }
    
    return loss_dict



# @torch.no_grad()
# def evaluate_vqa(task, model, dataloader, device, print_freq=500):
#     model.eval()
#     dataset = dataloader.dataset
#     qid_list = [ques['question_id'] for ques in dataset.ques_list]
#     ans_ix_list = []
#     i = 0

#     for batch_idx, (imgs, texts, targets) in enumerate(dataloader):
#         imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
#         batch_size = imgs.shape[0]
#         i += batch_size
#         outputs = model(img=imgs, text=texts, task=task)
#         pred_np = outputs.cpu().data.numpy()
#         pred_argmax = np.argmax(pred_np, axis=1)
#         if pred_argmax.shape[0] != dataset.configs.eval_batch_size:
#             pred_argmax = np.pad(
#                 pred_argmax,(0, dataset.configs.eval_batch_size - pred_argmax.shape[0]),
#                 mode='constant',constant_values=-1)
#         ans_ix_list.append(pred_argmax)
#         if batch_idx % print_freq == 0:
#             print('Test %d/%d:' %(batch_idx*batch_size, 
#                         len(dataloader.dataset)))
        
#     ans_ix_list = np.array(ans_ix_list).reshape(-1)
#     result = [{
#         'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
#         'question_id': int(qid_list[qix])}for qix in range(qid_list.__len__())]

#     result_eval_file = 'vqaeval_result/result_run_' + dataset.configs.version + '.json'
#     print('Save the result to file: {}'.format(result_eval_file))
#     json.dump(result, open(result_eval_file, 'w'))

#     # create vqa object and vqaRes object
#     ques_file_path = dataset.configs.question_path['val']
#     ans_file_path = dataset.configs.answer_path['val']
#     vqa = VQA_Tool(ans_file_path, ques_file_path)
#     vqaRes = vqa.loadRes(result_eval_file, ques_file_path)
#     vqaEval = VQA_Eval(vqa, vqaRes, n=2)  
#     vqaEval.evaluate()

#     return vqaEval.accuracy