import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import json, os
from PIL import Image
from datasets import load_dataset
from transformers import ViTImageProcessor
from collections import Counter
import random
import numpy as np

# ----------------------------
# Helpers
def patchify(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    img: [C, H, W] with H=W and divisible by patch_size
    returns: [L, patch_dim] where L = (H/P)*(W/P), patch_dim = C*P*P
    """
    C, H, W = img.shape
    P = patch_size
    assert H % P == 0 and W % P == 0
    h = H // P
    w = W // P
    # [C, h, P, w, P] -> [h, w, C, P, P] -> [L, C*P*P]
    patches = img.reshape(C, h, P, w, P).permute(1, 3, 0, 2, 4).reshape(h*w, C*P*P)
    return patches

def collate_img(batch):
    task = batch[0]["task"]
    images = torch.stack([b["vision"]["image"] for b in batch], dim=0)  # [B,C,H,W]

    # print('--- debug collate_img ---')
    # print('task:', task)
    # print(batch[0].keys())
    
    # print('type:', type(batch[0]['labels']['class']))
    # print('value:', batch[0]['labels']['class'])

    # try:
    #     print('shape:', batch[0]['labels']['class'].shape)
    # except Exception as e:
    #     pass

    # labels = torch.tensor([b["labels"]["class"] for b in batch])
    if task == "img_cls":
        labels = torch.tensor([b["labels"]["class"] for b in batch], dtype=torch.long)
    elif task == "img_rec":
        labels = torch.stack([b["labels"]["class"] for b in batch])
    else:
        raise ValueError(f"Unknown image task: {task}")

    patch_size = batch[0]["vision"]["patch_size"]
    
    return {"task": task, "vision": {"image": images, "patch_size": patch_size}, "labels": {"class": labels}}

    # # img_rec: pad patches to max L (though CIFAR with fixed patch_size has fixed L)
    # patches = [b["vision"]["image"] for b in batch]  # list of [L, D]
    # patches = torch.stack(patches, dim=0)  # [B,L,D]
    # target = torch.stack([b["labels"]["target_patches"] for b in batch], dim=0)
    # patch_size = batch[0]["vision"]["patch_size"]
    # image_shape = batch[0]["vision"]["image_shape"]
    # return {"task": task, "vision": {"image": patches, "patch_size": patch_size, "image_shape": image_shape},
    #         "labels": {"target_patches": target}}

def collate_text(batch, pad_id: int = 0):
    task = batch[0]["task"]
    input_ids = [b["text"]["input_ids"] for b in batch]
    attn = [b["text"]["attention_mask"] for b in batch]

    max_len = max(x.size(0) for x in input_ids)
    B = len(batch)

    input_pad = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_pad = torch.zeros((B, max_len), dtype=torch.long)

    for i in range(B):
        L = input_ids[i].size(0)
        input_pad[i, :L] = input_ids[i]
        attn_pad[i, :L] = attn[i]

    out = {"task": task, "text": {"input_ids": input_pad, "attention_mask": attn_pad}}

    if task == "txt_cls":
        labels = torch.tensor([b["labels"]["class"] for b in batch], dtype=torch.long)
        out["labels"] = {"class": labels}
    elif task == "txt_rec":
        target = input_pad.clone()
        target[target == pad_id] = 0
        out["labels"] = {"target_ids": target}
    else:
        raise ValueError(f"Unknown text task: {task}")

    return out

# Modality-specific datasets
class CIFAR10ImageTask(Dataset):
    def __init__(self, root, train: bool, task: str, patch_size: int):
        assert task in ["img_cls", "img_rec"]
        self.task = task
        self.patch_size = patch_size

        # Use a simple transform. If you use ViTImageProcessor, you can align with ViT normalization.
        self.transform = T.Compose([
            T.ToTensor(),  # [0,1]
        ])

        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.processor = ViTImageProcessor.from_pretrained("nateraw/vit-base-patch16-224-cifar10")

        self.ds = CIFAR10(root=root, train=train, download=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil_img, label = self.ds[idx]
        # img = self.transform(pil_img)  # [C,32,32]
        img = self.processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)  # [C,H,W], already normalized

        if self.task == "img_cls":
            return {
                "task": "img_cls",
                "vision": {
                    "image": img,
                    "patch_size": self.patch_size,
                },
                "labels": {
                    "class": int(label),
                },
            }

        # img_rec
        # patches = patchify(img, self.patch_size)  # patch=4 per paper
        elif self.task == "img_rec":
            return {
                "task": "img_rec",
                "vision": {
                    "image": img,      # [L, patch_dim]
                    "patch_size": self.patch_size,
                    # "image_shape": img.shape
                },
                "labels": {
                    "class": img.clone(),  # class is the original img
                },
            }
        else:
            raise ValueError(f"Unknown task: {self.task}")


class SST2TextTask(Dataset):
    def __init__(self, split: str, tokenizer, task: str, max_len: int = 128):
        assert task in ["txt_cls", "txt_rec"]
        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer

        ds = load_dataset("glue", "sst2", split=split)
        self.texts = ds["sentence"]
        self.labels = ds["label"]  # 0/1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            padding=False, # no padding here because it is single sentence; padding will be done in collate_text
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )
        item = {
            "task": self.task,
            "text": {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            }
        }
        if self.task == "txt_cls":
            item["labels"] = {"class": int(self.labels[idx])}
        elif self.task == "txt_rec":
            # reconstruct the same tokens; loss should ignore padding after collate
            item["labels"] = {"target_ids": item["text"]["input_ids"].clone()}
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return item


def collate_vqa(batch, pad_id: int = 0):
    task = batch[0]["task"]
    assert task == "vqa", f"Expected vqa, got {task}"

    images = torch.stack([b["vision"]["image"] for b in batch], dim=0)  # [B,C,H,W]

    input_ids = [b["text"]["input_ids"] for b in batch]
    attn = [b["text"]["attention_mask"] for b in batch]

    max_len = max(x.size(0) for x in input_ids)
    B = len(batch)

    input_pad = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_pad = torch.zeros((B, max_len), dtype=torch.long)

    for i in range(B):
        L = input_ids[i].size(0)
        input_pad[i, :L] = input_ids[i]
        attn_pad[i, :L] = attn[i]

    labels = torch.tensor([b["labels"]["class"] for b in batch], dtype=torch.long)
    scores = torch.stack([b["labels"]["scores"] for b in batch], dim=0)


    patch_size = batch[0]["vision"]["patch_size"]

    return {
        "task": task,
        "vision": {
            "image": images,
            "patch_size": patch_size,
        },
        "text": {
            "input_ids": input_pad,
            "attention_mask": attn_pad,
        },
        "labels": {
            "class": labels,
            "scores": scores,
        },
    }

# class VQAv2Task(Dataset): # this class assumes soft scores are precomputed
#     def __init__(
#         self,
#         root: str,
#         split: str,
#         tokenizer,
#         task: str = "vqa",
#         max_len: int = 32,
#         patch_size: int = 16,
#         top_k: int = 3000,
#         image_processor_name: str = "google/vit-base-patch16-224-in21k",
#     ):
#         assert task == "vqa"
#         assert split in ["train", "validation"]

#         self.root = root
#         self.split = split
#         self.task = task
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.patch_size = patch_size
#         self.processor = ViTImageProcessor.from_pretrained(image_processor_name)

#         cache_dir = os.path.join(root, "cache")

#         with open(os.path.join(cache_dir, f"ans2id_top{top_k}.json"), "r") as f:
#             self.ans2id = json.load(f)

#         with open(os.path.join(cache_dir, f"id2ans_top{top_k}.json"), "r") as f:
#             raw_id2ans = json.load(f)
#             self.id2ans = {int(k): v for k, v in raw_id2ans.items()}

#         samples_file = f"{split}_samples_top{top_k}.json"
#         with open(os.path.join(cache_dir, samples_file), "r") as f:
#             self.samples = json.load(f)

#         self.img_dir = os.path.join(root, "train2014" if split == "train" else "val2014")

#     def __len__(self):
#         return len(self.samples)

#     def _image_path(self, image_id: int) -> str:
#         split_name = "train2014" if self.split == "train" else "val2014"
#         fname = f"COCO_{split_name}_{image_id:012d}.jpg"
#         return os.path.join(self.img_dir, fname)

#     def __getitem__(self, idx):
#         ex = self.samples[idx]

#         image_path = self._image_path(ex["image_id"])
#         img = Image.open(image_path).convert("RGB")
#         img = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

#         enc = self.tokenizer(
#             ex["question"],
#             padding=False,
#             truncation=True,
#             max_length=self.max_len,
#             return_tensors=None,
#         )

#         scores = torch.zeros(len(self.ans2id), dtype=torch.float32)
#         if self.split == "train":
#             scores = torch.tensor(ex.get("answer_scores", scores.tolist()), dtype=torch.float32)
            
#         return {
#             "task": "vqa",
#             "vision": {
#                 "image": img,
#                 "patch_size": self.patch_size,
#             },
#             "text": {
#                 "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
#                 "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
#             },
#             "labels": {
#                 "class": int(ex["answer_id"]),
#                 "scores": scores
#             },
#             "meta": {
#                 "question_id": ex["question_id"],
#                 "image_id": ex["image_id"],
#                 "answer_text": ex["answer"],
#                 "question": ex["question"],
#             },
#         }

class VQAv2Task(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        tokenizer,
        task: str = "vqa",
        max_len: int = 32,
        patch_size: int = 16,
        top_k: int = 3129,
        image_processor_name: str = "google/vit-base-patch16-224-in21k",
    ):
        assert task == "vqa"
        assert split in ["train", "validation"]

        self.root = root
        self.split = split
        self.task = task
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.patch_size = patch_size
        self.processor = ViTImageProcessor.from_pretrained(image_processor_name)

        cache_dir = os.path.join(root, "cache")

        with open(os.path.join(cache_dir, f"ans2id_top{top_k}.json"), "r") as f:
            self.ans2id = json.load(f)

        with open(os.path.join(cache_dir, f"id2ans_top{top_k}.json"), "r") as f:
            raw_id2ans = json.load(f)
            self.id2ans = {int(k): v for k, v in raw_id2ans.items()}

        samples_file = f"{split}_samples_top{top_k}.json"
        with open(os.path.join(cache_dir, samples_file), "r") as f:
            self.samples = json.load(f)

        self.img_dir = os.path.join(root, "train2014" if split == "train" else "val2014")

    def __len__(self):
        return len(self.samples)

    def _image_path(self, image_id: int) -> str:
        split_name = "train2014" if self.split == "train" else "val2014"
        fname = f"COCO_{split_name}_{image_id:012d}.jpg"
        return os.path.join(self.img_dir, fname)

    def __getitem__(self, idx):
        ex = self.samples[idx]

        image_path = self._image_path(ex["image_id"])
        img = Image.open(image_path).convert("RGB")
        img = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        enc = self.tokenizer(
            ex["question"],
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )

        scores = torch.zeros(len(self.ans2id), dtype=torch.float32)
        
        # if self.split == "train":
            # ex["answers"] contains the 10 pre-normalized human answers
        # counts = Counter(ex["answers"])
        
        # for ans, occur in counts.items():
        #     ans_id = self.ans2id.get(ans)
        #     if ans_id is not None:
        #         if occur == 1:
        #             score = 0.3
        #         elif occur == 2:
        #             score = 0.6
        #         elif occur == 3:
        #             score = 0.9
        #         elif occur >= 4:
        #             score = 1.0
        #         else:
        #             continue 
                
        #         scores[ans_id] = score
            
        return {
            "task": "vqa",
            "vision": {
                "image": img,
                "patch_size": self.patch_size,
            },
            "text": {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            },
            "labels": {
                "class": int(ex["answer_id"]),
                "scores": scores
            },
            "meta": {
                "question_id": ex["question_id"],
                "image_id": ex["image_id"],
                "answer_text": ex["answer"],
                "question": ex["question"],
            },
        }

class MOSEIFeatureDataset(Dataset):
    """
    Expects a manifest list where each item points to saved arrays:
      - face_feat_path: npy with shape [T, F_face]
      - audio_feat_path: npy with shape [T, F_audio] (COVAREP-like) OR wav2vec2 embeddings
      - transcript: string (optional)
      - label: 0/1 (binary sentiment)
    """
    def __init__(self, manifest_path, tokenizer=None, max_txt_len=128):
        with open(manifest_path, "r") as f:
            self.items = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        face = np.load(it["face_feat_path"])   # [T, F_face]
        aud = np.load(it["audio_feat_path"])   # [T, F_audio]
        y = int(it["label"])

        sample = {
            "task": "video_sent",
            "vision": {"face_feats": torch.tensor(face, dtype=torch.float32)},
            "audio": {"audio_feats": torch.tensor(aud, dtype=torch.float32)},
            "labels": {"class": y},
        }

        if self.tokenizer is not None and "transcript" in it:
            enc = self.tokenizer(
                it["transcript"],
                padding=False,
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors=None,
            )
            sample["text"] = {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            }

        return sample

def pad_time_series(seqs, pad_value=0.0):
    # seqs: list of [T, F]
    B = len(seqs)
    F = seqs[0].shape[1]
    T_max = max(s.shape[0] for s in seqs)
    out = torch.full((B, T_max, F), float(pad_value), dtype=seqs[0].dtype)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0]] = s
    return out, lengths

def collate_mosei(batch, pad_id=0):
    face_list = [b["vision"]["face_feats"] for b in batch]
    aud_list  = [b["audio"]["audio_feats"] for b in batch]

    face, face_len = pad_time_series(face_list, pad_value=0.0)
    aud,  aud_len  = pad_time_series(aud_list,  pad_value=0.0)

    labels = torch.tensor([b["labels"]["class"] for b in batch], dtype=torch.long)

    out = {
        "task": "video_sent",
        "vision": {"face_feats": face, "face_len": face_len},
        "audio": {"audio_feats": aud, "audio_len": aud_len},
        "labels": {"class": labels},
    }

    if "text" in batch[0]:
        # reuse your text collate
        out_text = collate_text([{**b, "task": "txt_dummy"} for b in batch], pad_id=pad_id)["text"]
        out["text"] = out_text

    return out

class StepBasedMultiTaskBatcher:
    def __init__(self, loaders: dict, task_names=None, task_probs=None):
        self.loaders = loaders
        self.task_names = task_names or list(loaders.keys())

        if task_probs is None:
            self.task_probs = None
        else:
            s = sum(task_probs)
            self.task_probs = [p / s for p in task_probs]

        self.iters = {k: iter(v) for k, v in loaders.items()}

    def _next_batch(self, task_name):
        try:
            return next(self.iters[task_name])
        except StopIteration:
            self.iters[task_name] = iter(self.loaders[task_name])
            return next(self.iters[task_name])

    def sample_task(self):
        if self.task_probs is None:
            return random.choice(self.task_names)
        return random.choices(self.task_names, weights=self.task_probs, k=1)[0]

    def next(self):
        task_name = self.sample_task()
        batch = self._next_batch(task_name)
        return task_name, batch


def multitask_batcher(loaders: dict):
    """
    One epoch: iterate over all loaders exactly once.
    At each step, randomly choose a task that still has data.
    """
    iters = {k: iter(v) for k, v in loaders.items()}
    active = list(loaders.keys())  # tasks that still have data

    while active:
        # pick a random task from those that are not exhausted
        k = random.choice(active)
        try:
            batch = next(iters[k])
            yield k, batch
        except StopIteration:
            # this task is done for this epoch
            active.remove(k)

def batch_to_inputs(batch):
    inputs = {}
    if batch["task"] == "img_cls":
        inputs["vision_tokens"] = batch["vision"]["image"]
        inputs["labels"] = batch["labels"]["class"]
        inputs["task_id"] = torch.zeros(inputs["labels"].shape[0], dtype=torch.long)  # task_id 0 for img_cls
        inputs['patch_size'] = batch['vision']['patch_size']

    elif batch["task"] == "img_rec":
        inputs["vision_tokens"] = batch["vision"]["image"]
        inputs["labels"] = batch["labels"]["class"]
        inputs["task_id"] = torch.ones(inputs["labels"].shape[0], dtype=torch.long)  # task_id 1 for img_rec
        inputs['patch_size'] = batch['vision']['patch_size']

    elif batch["task"] in ["txt_cls", "txt_rec"]:
        inputs["text_tokens"] = batch["text"]["input_ids"]
        inputs["labels"] = batch["labels"]["class"] if batch["task"] == "txt_cls" else batch["labels"]["target_ids"]
        
        inputs["task_id"] = torch.full((inputs["labels"].shape[0],), 2 if batch["task"] == "txt_cls" else 3, dtype=torch.long)  # task_id 2 for txt_cls, 3 for txt_rec
        inputs['attention_mask'] = batch['text']['attention_mask']

    elif batch["task"] == "vqa":
        inputs["vision_tokens"] = batch["vision"]["image"]
        inputs["text_tokens"] = batch["text"]["input_ids"]
        inputs["scores"] = batch["labels"]["scores"].to(torch.float32)
        inputs["labels"] = batch["labels"]["class"]
        inputs["task_id"] = torch.full((inputs["labels"].shape[0],), 4, dtype=torch.long)  # task_id 4 for vqa
        inputs['attention_mask'] = batch['text']['attention_mask']
        inputs['patch_size'] = batch['vision']['patch_size']    

    else:
        raise ValueError(f"Unknown task type: {batch['task']}")

    return inputs
