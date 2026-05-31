# Desensitization Notes

The public extraction follows these rules:

- Do not commit raw fMRI volumes, VQ token shards from real participants,
  checkpoints, adapters, TensorBoard logs, or run directories.
- Replace absolute local paths with `$DATA_ROOT`, `$OUTPUT_ROOT`, and
  `$HF_HOME` placeholders.
- Do not expose private subject IDs, scanner/session paths, NAS mounts, process identifiers,
  GPU allocations, or shell history.
- Store only aggregate metrics in `results/`.
- Use synthetic token exports for smoke tests and examples.

Before publishing changes, scan staged files for private paths and credentials:

```bash
grep -R -nE 'PRIVATE_PATH_PATTERN|CREDENTIAL_PATTERN' .
find . -type f -size +10M
```
