from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_modeling.mask_transformer.multi_frame_token_dataset import VQMultiFrameIndexDataset


@dataclass
class QwenVQLoraConfig:
    token_root: str
    output_dir: str
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    segment_len: int = 4
    stride: int = 1
    num_codebook_vectors: int = 128
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 3
    learning_rate: float = 1e-4
    max_train_steps: int | None = None
    max_val_steps: int | None = None
    max_segments: int | None = None
    max_shards: int | None = None
    max_length: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    save_every: int = 0
    hf_cache_dir: str | None = None


def vq_tokens(n: int) -> list[str]:
    return [f"<vq_{i:03d}>" for i in range(n)]


def structure_tokens(segment_len: int) -> list[str]:
    toks = ["<vq4d>", "</vq4d>"]
    for i in range(segment_len):
        toks.extend([f"<f{i}>", f"</f{i}>"])
    return toks


class MultiFrameVQSpecialTokenDataset(Dataset):
    """Represent each VQ index as one atomic special token."""

    def __init__(self, root: str, split: str, tokenizer, cfg: QwenVQLoraConfig):
        self.base = VQMultiFrameIndexDataset(
            root,
            split,
            segment_len=cfg.segment_len,
            stride=cfg.stride,
            flatten=False,
            max_segments=cfg.max_segments,
            max_shards=cfg.max_shards,
        )
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.vq_id_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(vq_tokens(cfg.num_codebook_vectors)), dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        arr, info = self.base[idx]
        pieces = ["<vq4d>"]
        for frame_i, frame in enumerate(arr):
            pieces.append(f"<f{frame_i}>")
            pieces.extend(f"<vq_{int(x):03d}>" for x in frame.reshape(-1).numpy().tolist())
            pieces.append(f"</f{frame_i}>")
        pieces.append("</vq4d>")
        enc = self.tokenizer(" ".join(pieces), truncation=True, max_length=self.cfg.max_length, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        labels = input_ids.clone()
        is_vq = torch.isin(input_ids, self.vq_id_tensor)
        labels[(attn == 0) | (~is_vq)] = -100
        meta = {k: v for k, v in info.items() if k != "frames"}
        return input_ids, attn, labels, json.dumps(meta, ensure_ascii=False)


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if not bool(mask.any()):
        return float("nan")
    pred = shifted_logits.argmax(dim=-1)
    return float((pred[mask] == shifted_labels[mask]).float().mean().item())


def evaluate(model, loader: DataLoader, cfg: QwenVQLoraConfig):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if cfg.max_val_steps is not None and step >= cfg.max_val_steps:
                break
            input_ids, attn, labels, _info = batch
            input_ids = input_ids.to(cfg.device)
            attn = attn.to(cfg.device)
            labels = labels.to(cfg.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            losses.append(float(out.loss.item()))
            accs.append(token_accuracy(out.logits, labels))
    model.train()
    return float(np.mean(losses)) if losses else float("nan"), float(np.nanmean(accs)) if accs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--segment-len", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num-codebook-vectors", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--max-segments", type=int, default=None)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--hf-cache-dir", default=None)
    args = p.parse_args()
    cfg = QwenVQLoraConfig(**vars(args))

    out = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.hf_cache_dir, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": structure_tokens(cfg.segment_len) + vq_tokens(cfg.num_codebook_vectors)})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(out / "tokenizer")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        cache_dir=cfg.hf_cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_cfg).to(cfg.device)

    train_ds = MultiFrameVQSpecialTokenDataset(cfg.token_root, "train", tokenizer, cfg)
    test_ds = MultiFrameVQSpecialTokenDataset(cfg.token_root, "test", tokenizer, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"qwen-vq-lora-epoch{epoch}", dynamic_ncols=True)):
            if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
                break
            input_ids, attn, labels, _info = batch
            out_obj = model(input_ids=input_ids.to(cfg.device), attention_mask=attn.to(cfg.device), labels=labels.to(cfg.device))
            opt.zero_grad(set_to_none=True)
            out_obj.loss.backward()
            opt.step()
        val_loss, val_acc = evaluate(model, test_loader, cfg)
        row = {"epoch": epoch, "val_loss": val_loss, "val_vq_acc": val_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            model.save_pretrained(out / "checks" / f"qwen_vq_lora_epoch_{epoch}")
    (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
