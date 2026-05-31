from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train_qwen_vq_lora import MultiFrameVQSpecialTokenDataset, QwenVQLoraConfig, vq_tokens


@dataclass
class FreeRunConfig:
    run_dir: str
    adapter_dir: str
    output_dir: str
    split: str = "test"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mask_ratio: float = 0.4
    max_samples: int = 32
    max_shards: int | None = None
    max_segments: int | None = None
    hf_cache_dir: str | None = None
    dtype: str = "bf16"


def load_train_config(run_dir: Path) -> QwenVQLoraConfig:
    return QwenVQLoraConfig(**json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")))


def load_model_and_tokenizer(eval_cfg: FreeRunConfig, train_cfg: QwenVQLoraConfig):
    run_dir = Path(eval_cfg.run_dir)
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if eval_cfg.dtype == "bf16" and torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(train_cfg.model_name, cache_dir=eval_cfg.hf_cache_dir, trust_remote_code=True, torch_dtype=torch_dtype)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, eval_cfg.adapter_dir).to(eval_cfg.device)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def free_run_suffix_vq_accuracy(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, vq_ids: list[int], mask_ratio: float, device: str):
    full_ids = input_ids[attention_mask.bool()].tolist()
    vq_id_set = set(vq_ids)
    vq_positions = [i for i, tok in enumerate(full_ids) if tok in vq_id_set]
    if not vq_positions:
        return None

    keep_count = max(1, int(round(len(vq_positions) * (1.0 - mask_ratio))))
    keep_count = min(keep_count, len(vq_positions) - 1)
    target_positions = set(vq_positions[keep_count:])
    first_target_pos = min(target_positions)

    prefix = torch.tensor([full_ids[:first_target_pos]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=prefix, use_cache=True)
        past = out.past_key_values
        logits = out.logits
        correct = 0
        total = 0
        pred_counts: dict[int, int] = {}
        for pos in range(first_target_pos, len(full_ids)):
            true_id = full_ids[pos]
            if pos in target_positions:
                vq_logits = logits[:, -1, vq_ids]
                pred_id = int(vq_ids[int(vq_logits.argmax(dim=-1).item())])
                total += 1
                correct += int(pred_id == true_id)
                pred_counts[pred_id] = pred_counts.get(pred_id, 0) + 1
                next_id = pred_id
            else:
                next_id = true_id
            out = model(input_ids=torch.tensor([[next_id]], dtype=torch.long, device=device), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits

    return {
        "acc": correct / total if total else float("nan"),
        "correct": correct,
        "total": total,
        "n_vq_tokens": len(vq_positions),
        "keep_vq_tokens": keep_count,
        "generated_vq_tokens": total,
        "unique_pred_vq": len(pred_counts),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--max-segments", type=int, default=None)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()
    eval_cfg = FreeRunConfig(**vars(args))

    out_dir = Path(eval_cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_config.json").write_text(json.dumps(asdict(eval_cfg), indent=2), encoding="utf-8")

    train_cfg = load_train_config(Path(eval_cfg.run_dir))
    train_cfg.device = eval_cfg.device
    train_cfg.max_shards = eval_cfg.max_shards
    train_cfg.max_segments = eval_cfg.max_segments
    model, tokenizer = load_model_and_tokenizer(eval_cfg, train_cfg)
    vq_ids = tokenizer.convert_tokens_to_ids(vq_tokens(train_cfg.num_codebook_vectors))
    ds = MultiFrameVQSpecialTokenDataset(train_cfg.token_root, eval_cfg.split, tokenizer, train_cfg)

    rows = []
    for idx in tqdm(range(min(eval_cfg.max_samples, len(ds))), desc="qwen-vq-freerun", dynamic_ncols=True):
        input_ids, attn, _labels, info_json = ds[idx]
        result = free_run_suffix_vq_accuracy(model, input_ids, attn, vq_ids, eval_cfg.mask_ratio, eval_cfg.device)
        if result is None:
            continue
        result.update({"idx": idx, "info": json.loads(info_json)})
        rows.append(result)

    total_correct = sum(r["correct"] for r in rows)
    total_tokens = sum(r["total"] for r in rows)
    summary = {
        "split": eval_cfg.split,
        "mask_ratio": eval_cfg.mask_ratio,
        "samples": len(rows),
        "micro_acc": total_correct / total_tokens if total_tokens else float("nan"),
        "macro_acc": float(np.mean([r["acc"] for r in rows])) if rows else float("nan"),
        "total_generated_vq_tokens": total_tokens,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "per_sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
