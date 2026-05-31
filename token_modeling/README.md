# Token Modeling

This folder contains two downstream models for short sequences of VQ indices.

## Mask Transformer

`token_modeling.mask_transformer.train_multiframe_mask_transformer` is a
single-pass bidirectional encoder. It randomly masks VQ tokens and predicts the
masked positions in parallel.

## Qwen VQ-LoRA

`token_modeling.qwen_lora.train_qwen_vq_lora` adapts a causal LLM by adding one
special token for each VQ code, e.g. `<vq_000>` ... `<vq_127>`. The loss is
computed only on VQ token positions. LoRA updates are applied to attention and
MLP projection layers while the new embeddings and LM head are trainable.

The Qwen scripts do not include model weights or adapters. Users must download
their selected base model under its own license.
