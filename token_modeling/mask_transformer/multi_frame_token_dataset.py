from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class VQMultiFrameIndexDataset(Dataset):
    """Build contiguous multi-frame VQ-index segments from sharded exports.

    Expected shard schema:

    ```python
    {
        "indices": LongTensor[N, D, H, W],
        "meta": [{"group_id": "...", "condition": "...", "run": "...", "t": 0}, ...],
        "num_samples": N,
    }
    ```

    `subject_id` and `path` are supported for backward compatibility but are not
    returned by default, so public examples do not leak private identifiers.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        segment_len: int = 4,
        stride: int = 1,
        flatten: bool = False,
        require_consecutive: bool = True,
        max_segments: int | None = None,
        max_shards: int | None = None,
        cache_size: int = 4,
        expose_private_meta: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.segment_len = int(segment_len)
        self.stride = int(stride)
        self.flatten = bool(flatten)
        self.require_consecutive = bool(require_consecutive)
        self.max_segments = max_segments
        self.max_shards = max_shards
        self.cache_size = int(cache_size)
        self.expose_private_meta = bool(expose_private_meta)
        if self.segment_len <= 0:
            raise ValueError("segment_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.shard_paths = sorted((self.root / "indices_shards" / split).glob("shard_*.pt"))
        if self.max_shards is not None:
            self.shard_paths = self.shard_paths[: int(self.max_shards)]
        if not self.shard_paths:
            raise RuntimeError(f"No sharded VQ indices found under {self.root}/indices_shards/{split}")

        self.records: list[dict[str, Any]] = []
        self.segments: list[list[int]] = []
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self._build_index()

    def _build_index(self) -> None:
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for shard_path in self.shard_paths:
            payload = torch.load(shard_path, map_location="cpu")
            metas = payload.get("meta")
            if metas is None:
                raise RuntimeError(f"Shard lacks meta rows: {shard_path}")
            num_samples = int(payload.get("num_samples", len(metas)))
            if len(metas) != num_samples:
                raise RuntimeError(f"Meta/sample mismatch in {shard_path}: meta={len(metas)} num_samples={num_samples}")
            for offset, meta in enumerate(metas):
                t_value = int(meta["t"])
                group_id = meta.get("group_id") or meta.get("subject_id") or "group"
                record_idx = len(self.records)
                record = {
                    "shard_path": shard_path,
                    "offset": offset,
                    "group_id": group_id,
                    "task": meta.get("task"),
                    "condition": meta.get("condition"),
                    "run": meta.get("run"),
                    "t": t_value,
                }
                if self.expose_private_meta:
                    record["subject_id"] = meta.get("subject_id")
                    record["source_path"] = meta.get("path")
                self.records.append(record)
                groups[(record["task"], record["group_id"], record["condition"], record["run"])].append(record_idx)

        for _key, idxs in groups.items():
            idxs = sorted(idxs, key=lambda i: self.records[i]["t"])
            if len(idxs) < self.segment_len:
                continue
            for start in range(0, len(idxs) - self.segment_len + 1, self.stride):
                window = idxs[start : start + self.segment_len]
                if self.require_consecutive:
                    ts = [self.records[i]["t"] for i in window]
                    if any((b - a) != 1 for a, b in zip(ts, ts[1:])):
                        continue
                self.segments.append(window)
                if self.max_segments is not None and len(self.segments) >= self.max_segments:
                    return

    def __len__(self) -> int:
        return len(self.segments)

    def _load_shard(self, path: Path) -> dict[str, Any]:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        payload = torch.load(path, map_location="cpu")
        self._cache[path] = payload
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def __getitem__(self, idx: int):
        rec_indices = self.segments[idx]
        tensors = []
        metas = []
        for rec_idx in rec_indices:
            rec = self.records[rec_idx]
            payload = self._load_shard(rec["shard_path"])
            tensors.append(payload["indices"][rec["offset"]].long())
            metas.append({k: v for k, v in rec.items() if k not in {"shard_path", "offset"}})
        segment = torch.stack(tensors, dim=0)
        if self.flatten:
            segment = segment.reshape(-1)
        first = metas[0]
        last = metas[-1]
        info = {
            "group_id": first["group_id"],
            "task": first["task"],
            "condition": first["condition"],
            "run": first["run"],
            "t_start": first["t"],
            "t_end": last["t"],
            "segment_len": self.segment_len,
            "shape": tuple(segment.shape),
            "frames": metas,
        }
        return segment, info
