# Multi-Level Quantization: results

Model: Qwen3-8B (bf16). Quantiser: asymmetric RTN, group 128, applied to all 252 decoder linears.
Calibration: WikiText-2 train, 8 × 512 tokens. Evaluation: WikiText-2 test, first 40 × 512-token
windows. Command: `python results/run_real.py` (raw output in `results/real.json`).

| Config | Avg bits | 4-bit layers | Packed size | Perplexity |
|---|---:|---:|---:|---:|
| bf16 | 16 | – | 16.4 GB | 12.0346 |
| Uniform 4-bit | 4.0 | 252 | 6.18 GB | 12.9164 |
| Sensitivity, 50% budget | 3.5 | 122 | 5.75 GB | **13.6447** |
| Random, 50% budget, seed 0 | 3.5 | 122 | 5.75 GB | 15.7643 |
| Random, 50% budget, seed 1 | 3.5 | 128 | 5.75 GB | 14.1372 |
| Random, 50% budget, seed 2 | 3.5 | 133 | 5.75 GB | 16.2435 |
| Sensitivity, 25% budget | 3.25 | 76 | 5.53 GB | **14.5019** |
| Random, 25% budget, seed 0 | 3.25 | 66 | 5.53 GB | 16.5413 |
| Random, 25% budget, seed 1 | 3.25 | 67 | 5.53 GB | 16.0702 |
| Random, 25% budget, seed 2 | 3.25 | 65 | 5.53 GB | 16.5916 |
| Uniform 3-bit | 3.0 | 0 | 5.31 GB | 17.4711 |

**Reading it.** The sensitivity-chosen allocation beats all three random allocations at both
budgets. Random allocation varies a lot (14.14–16.24 at 3.5 bits), which itself shows that
*which* layers get the extra bit matters; the score finds a good set without searching.

**Most sensitive layers** (first 20 by score): q/k/v projections of blocks 29–35 and
`layers.35.mlp.down_proj`. The last blocks' attention is where 3-bit hurts most.

**Withdrawn.** The previous version of this file reported bf16 49.4 / 8-bit 62.1 / "4-bit collapses"
on 3 short prompts. That quantiser used one absmax scale per weight matrix, which is broken; all
of those numbers are replaced by the table above.
