# Sharded VQ Index Data Format

Token-modeling code consumes precomputed VQ indices in `sharded_pt_v1` format.
The format intentionally separates expensive 3D VQ encoding from downstream
language-model experiments.

```text
token_root/
  export_config.json
  manifest_train.csv
  manifest_test.csv
  indices_shards/
    train/
      shard_00000.pt
    test/
      shard_00000.pt
```

Each shard is a `torch.save` payload:

```python
{
    "format": "sharded_pt_v1",
    "indices": LongTensor[N, D, H, W],
    "meta": [
        {
            "group_id": "subject-or-sequence-id",
            "task": "task-name",
            "condition": "condition-name",
            "run": "run-01",
            "t": 0,
        },
    ],
    "num_samples": N,
}
```

`group_id`, `condition`, `run`, and integer `t` are used to build consecutive
temporal windows. If you work with sensitive data, keep private subject IDs and
source file paths out of public shards; use anonymized group IDs instead.

Create a synthetic smoke dataset:

```bash
python -m vq.export_vq_tokens --make-synthetic --output-root examples/tiny_tokens
```
