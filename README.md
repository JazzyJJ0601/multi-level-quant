# Multi-Level Quantization (MLQ)

Mixed 3/4-bit weight quantisation for LLMs: every decoder linear gets 3 or 4 bits, and a
calibrated sensitivity score decides which layers get the extra bit under a fixed memory budget.

Measured on **Qwen3-8B**: at 3.5 bits per weight, spending the 4-bit budget on the layers the
score picks gives **13.64 perplexity**, against **15.38** (mean of 3 seeds) when the same
budget goes to random layers, and 17.47 for plain 3-bit.

## Method

1. Run 8 × 512 tokens of WikiText-2 *train* through the model and record E[x²] for every input
   channel of every decoder linear (252 layers).
2. For each layer, compute the activation-weighted quantisation error, Σ (ΔW)² · E[x²]
   (a diagonal-Hessian proxy for output error), at 3 bits and at 4 bits.
3. Score = error removed by the 4th bit, per weight. Give 4 bits greedily to the highest-scoring
   layers until the budget (25% or 50% of weights at 4 bits) is spent; the rest get 3 bits.

Quantiser: asymmetric min-max round-to-nearest, one fp16 scale and zero-point per 128 weights
(fake-quantised: weights are quantised then dequantised to bf16 for evaluation).

## Results

Qwen3-8B, WikiText-2 *test*, 40 × 512 tokens. Packed size is computed (bits + 32/128 bits per
weight for scale and zero-point; embeddings, lm_head and norms stay 16-bit).

| Config | Avg bits | Packed size | Perplexity |
|---|---:|---:|---:|
| bf16 | 16 | 16.4 GB | 12.03 |
| Uniform 4-bit | 4.0 | 6.18 GB | 12.92 |
| **MLQ, 50% of weights at 4-bit** | 3.5 | 5.75 GB | **13.64** |
| Random layers, same budget (3 seeds) | 3.5 | 5.75 GB | 15.76 / 14.14 / 16.24 (mean 15.38) |
| **MLQ, 25% of weights at 4-bit** | 3.25 | 5.53 GB | **14.50** |
| Random layers, same budget (3 seeds) | 3.25 | 5.53 GB | 16.54 / 16.07 / 16.59 (mean 16.40) |
| Uniform 3-bit | 3.0 | 5.31 GB | 17.47 |

- The sensitivity score beats every random seed at both budgets.
- At 3.5 bits MLQ recovers 84% of the gap between uniform 3-bit and uniform 4-bit, for half
  the extra memory.
- The layers it protects first are the attention projections (q/k/v) of the last 8 blocks
  (29–35) and the final MLP down-projection.

Full numbers: [`results/real.json`](results/real.json). Details: [RESULTS.md](RESULTS.md).

## Honest limits

- Fake quantisation: perplexity is exact for the stated format, but there is no packed 3/4-bit
  kernel here, so no speed or measured-memory numbers.
- Plain RTN underneath. Combining the allocation with GPTQ/AWQ-style quantisers would lower all
  rows; whether the gap to random allocation stays is not tested.
- One model, one dataset, 20k evaluation tokens.
- An earlier version of this repo reported "4-bit uniform collapses" on Qwen3-8B. That came from a
  quantiser bug (one scale for a whole weight matrix) and is withdrawn: correct group-wise 4-bit
  costs +0.88 perplexity.

## Reproduce

Needs a CUDA GPU with ~20 GB and a local Qwen3-8B checkpoint (path in `results/qcommon.py`).

```bash
python results/run_real.py      # ~4 min on an RTX 3090 Ti, writes results/real.json
pytest -q tests                 # unit tests for the small NumPy prototype in src/
```

## License

MIT
