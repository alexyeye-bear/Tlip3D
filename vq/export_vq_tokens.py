from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def write_sharded_indices(indices: torch.Tensor, metas: list[dict], output_root: Path, split: str, shard_size: int = 4096) -> None:
    """Write VQ indices in the `sharded_pt_v1` format used by token models."""

    output_dir = output_root / "indices_shards" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for shard_id, start in enumerate(range(0, int(indices.shape[0]), shard_size)):
        end = min(start + shard_size, int(indices.shape[0]))
        shard = {
            "format": "sharded_pt_v1",
            "indices": indices[start:end].short().cpu(),
            "meta": metas[start:end],
            "num_samples": end - start,
        }
        shard_name = f"shard_{shard_id:05d}.pt"
        torch.save(shard, output_dir / shard_name)
        manifest_rows.append({"split": split, "shard": shard_name, "num_samples": end - start, "start": start, "end": end})

    manifest_path = output_root / f"manifest_{split}.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "shard", "num_samples", "start", "end"])
        writer.writeheader()
        writer.writerows(manifest_rows)


def make_synthetic_export(output_root: Path, num_train: int, num_test: int, token_shape: tuple[int, int, int], num_codes: int) -> None:
    """Create a tiny synthetic export for smoke tests and documentation."""

    output_root.mkdir(parents=True, exist_ok=True)
    for split, count in [("train", num_train), ("test", num_test)]:
        indices = torch.randint(0, num_codes, (count, *token_shape), dtype=torch.long)
        metas = [
            {
                "group_id": f"synthetic-{i // 8:03d}",
                "task": "synthetic",
                "condition": "demo",
                "run": "run-01",
                "t": i % 8,
            }
            for i in range(count)
        ]
        write_sharded_indices(indices, metas, output_root, split, shard_size=max(1, min(16, count)))
    (output_root / "export_config.json").write_text(
        json.dumps({"format": "sharded_pt_v1", "token_shape": token_shape, "num_codebook_vectors": num_codes}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilities for public VQ-index sharded exports")
    parser.add_argument("--make-synthetic", action="store_true", help="Create a synthetic token export for smoke tests")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-train", type=int, default=32)
    parser.add_argument("--num-test", type=int, default=16)
    parser.add_argument("--token-shape", default="8,8,8")
    parser.add_argument("--num-codebook-vectors", type=int, default=128)
    args = parser.parse_args()
    token_shape = tuple(int(x) for x in args.token_shape.split(","))
    if args.make_synthetic:
        make_synthetic_export(Path(args.output_root), args.num_train, args.num_test, token_shape, args.num_codebook_vectors)
    else:
        raise SystemExit("This public utility currently provides --make-synthetic. Plug your VQ encoder before write_sharded_indices().")


if __name__ == "__main__":
    main()
