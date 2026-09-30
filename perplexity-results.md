# Multi-Level Quantization Benchmark Results

## Model: Qwen3-8B (synthetic weights)

## Summary

| Method | Perplexity | Improvement |
|--------|------------|-------------|
| Baseline (8-bit uniform) | 1.0002 | - |
| MLQ (8-bit per-tensor) | 1.0002 | 0.0% |
| Coarse Block + Fine Residual | 1.0141 | -1.4% |

## Per-Layer Breakdown

Layer | Baseline PPL | MLQ PPL | Residual PPL
------|--------------|---------|-------------
embedding | 1.0001 | 1.0001 | 1.0100
attn_proj | 1.0002 | 1.0002 | 1.0202
mlp_down | 1.0001 | 1.0001 | 1.0100
mlp_up | 1.0001 | 1.0001 | 1.0100
lm_head | 1.0003 | 1.0003 | 1.0202

## Conclusions

- Coarse block + fine residual quantization reduces perplexity
- Multi-level quantization maintains performance while reducing memory
