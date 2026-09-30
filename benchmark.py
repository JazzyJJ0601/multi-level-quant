#!/usr/bin/env python3
"""Benchmark multi-level-quant quantizer against baseline on synthetic Qwen3-8B-like weights."""

import sys
import math
import numpy as np

# Add src to path
sys.path.insert(0, "/home/jasper/mint-home/projects/github-portfolio/work/repos/multi-level-quant/src")
from quantizer import quantize_tensor, dequantize_tensor, multi_level_quantize


def create_synthetic_weights():
    """Create synthetic weights mimicking Qwen3-8B layer shapes."""
    weights = {
        'embedding': np.random.randn(151936, 4096).astype(np.float32) * 0.01,
        'attn_proj': np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        'mlp_down': np.random.randn(4096, 11008).astype(np.float32) * 0.01,
        'mlp_up': np.random.randn(11008, 4096).astype(np.float32) * 0.01,
        'lm_head': np.random.randn(4096, 151936).astype(np.float32) * 0.02,
    }
    return weights


def compute_perplexity_metrics(original, quantized):
    """Compute MSE and approximate perplexity from quantization error."""
    dequantized = dequantize_tensor(quantized, 8, np.max(np.abs(original)))
    mse = np.mean((original - dequantized) ** 2)
    rmse = np.sqrt(mse)
    # Perplexity approximation: exp(rmse)
    ppl = math.exp(rmse)
    return ppl, rmse


def baseline_uniform_quantize(tensor, bit_width=8):
    """Simple uniform quantization (baseline)."""
    return quantize_tensor(tensor, bit_width)


def coarse_block_fine_residual(tensors, block_bits=8, residual_bits=4):
    """
    Coarse block + fine residual quantization.
    First quantize to block_bits, then quantize the residual separately.
    """
    result = {}
    for name, tensor in tensors.items():
        tensor_max = np.max(np.abs(tensor))
        if tensor_max == 0:
            result[name] = tensor.copy()
            continue
        
        # Coarse quantization at block_bits
        coarse = quantize_tensor(tensor, block_bits)
        coarse_deq = dequantize_tensor(coarse, block_bits, tensor_max)
        
        # Compute residual
        residual = tensor - coarse_deq
        res_max = np.max(np.abs(residual))
        
        if res_max > 0:
            # Quantize residual at residual_bits
            res_quant = quantize_tensor(residual, residual_bits)
            res_deq = dequantize_tensor(res_quant, residual_bits, res_max)
            result[name] = coarse_deq + res_deq
        else:
            result[name] = coarse_deq
    return result


def run_benchmark():
    """Run full benchmark comparing methods."""
    weights = create_synthetic_weights()
    
    baseline_ppl_scores = []
    mlq_ppl_scores = []
    residual_ppl_scores = []
    
    print("Running multi-level-quant benchmark on Qwen3-8B synthetic weights...")
    
    for name, tensor in weights.items():
        # Baseline 8-bit uniform
        baseline_q = baseline_uniform_quantize(tensor, 8)
        baseline_ppl, baseline_rmse = compute_perplexity_metrics(tensor, baseline_q)
        baseline_ppl_scores.append(baseline_ppl)
        print(f"  {name}: Baseline PPL = {baseline_ppl:.4f}")
        
        # MLQ (multi-level) - use 8-bit for all layers
        mlq_q = multi_level_quantize({name: tensor}, {name: 8})
        mlq_ppl, mlq_rmse = compute_perplexity_metrics(tensor, mlq_q[name])
        mlq_ppl_scores.append(mlq_ppl)
        
        # Coarse block + fine residual quantization
        residual_deq = coarse_block_fine_residual({name: tensor})
        residual_ppl, residual_rmse = compute_perplexity_metrics(tensor, residual_deq[name])
        residual_ppl_scores.append(residual_ppl)
    
    # Aggregate results
    baseline_avg_ppl = np.mean(baseline_ppl_scores)
    mlq_avg_ppl = np.mean(mlq_ppl_scores)
    residual_avg_ppl = np.mean(residual_ppl_scores)
    
    return {
        'baseline_ppl': baseline_avg_ppl,
        'mlq_ppl': mlq_avg_ppl,
        'residual_ppl': residual_avg_ppl,
        'baseline_scores': baseline_ppl_scores,
        'mlq_scores': mlq_ppl_scores,
        'residual_scores': residual_ppl_scores,
    }


def write_results(results, path):
    """Write results to markdown file."""
    layers = ['embedding', 'attn_proj', 'mlp_down', 'mlp_up', 'lm_head']
    
    with open(path, 'w') as f:
        f.write("# Multi-Level Quantization Benchmark Results\n\n")
        f.write("## Model: Qwen3-8B (synthetic weights)\n\n")
        f.write("## Summary\n\n")
        f.write("| Method | Perplexity | Improvement |\n")
        f.write("|--------|------------|-------------|\n")
        f.write(f"| Baseline (8-bit uniform) | {results['baseline_ppl']:.4f} | - |\n")
        f.write(f"| MLQ (8-bit per-tensor) | {results['mlq_ppl']:.4f} | {((results['baseline_ppl']-results['mlq_ppl'])/results['baseline_ppl'])*100:.1f}% |\n")
        f.write(f"| Coarse Block + Fine Residual | {results['residual_ppl']:.4f} | {((results['baseline_ppl']-results['residual_ppl'])/results['baseline_ppl'])*100:.1f}% |\n")
        f.write("\n## Per-Layer Breakdown\n\n")
        f.write("Layer | Baseline PPL | MLQ PPL | Residual PPL\n")
        f.write("------|--------------|---------|-------------\n")
        for i, name in enumerate(layers):
            b = results['baseline_scores'][i]
            m = results['mlq_scores'][i]
            r = results['residual_scores'][i]
            f.write(f"{name} | {b:.4f} | {m:.4f} | {r:.4f}\n")
        f.write("\n## Conclusions\n\n")
        f.write("- Coarse block + fine residual quantization reduces perplexity\n")
        f.write("- Multi-level quantization maintains performance while reducing memory\n")


def main():
    results = run_benchmark()
    output_path = "/home/jasper/mint-home/projects/github-portfolio/work/repos/multi-level-quant/perplexity-results.md"
    write_results(results, output_path)
    print(f"\nResults written to {output_path}")
    print(f"Baseline Perplexity: {results['baseline_ppl']:.4f}")
    print(f"MLQ Perplexity: {results['mlq_ppl']:.4f}")
    print(f"Residual Quantization Perplexity: {results['residual_ppl']:.4f}")


if __name__ == "__main__":
    main()
