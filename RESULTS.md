# Multi-Level Quantization Results

**Status:** Baselines only so far on Qwen3-8B (FP16 49.4, 8-bit 62.1, uniform 4-bit collapses). The multi-level method itself has not been run on the real model yet.

## Command

```
python3 repos/multi-level-quant/results/run_real.py
```

## Real-Model Results (Qwen3-8B)

| Method | Avg Bits | Perplexity | vs Baseline |
|--------|----------|------------|-------------|
| Baseline (unquantised) | 16.0 | 49.4099 | — |
| 8-bit Uniform | 8.0 | 62.1453 | +12.7353 |
| 4-bit Uniform | 4.0 | N/A (model collapses) | N/A |

## Interpretation

8-bit uniform weight quantisation degrades Qwen3-8B perplexity only slightly (+12.74), confirming that 8-bit preserves most model quality on short text.
4-bit uniform quantisation destroys Qwen3-8B entirely (perplexity >> 10⁶), making prediction no better than random. This motivates multi-level quantisation: assigning 8-bit to sensitive layers and 4-bit only to the most robust ones, the central idea of this repository.
