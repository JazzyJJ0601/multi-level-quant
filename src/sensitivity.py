"""Sensitivity Analysis for Multi-Level Quantization.

This module provides tools to analyze how sensitive each tensor is to
quantization, which helps assign appropriate bit-widths.
"""

import numpy as np


def compute_sensitivity_score(tensor: np.ndarray, 
                              bit_width: int = 4) -> float:
    """Compute a sensitivity score for a tensor.
    
    The sensitivity score measures how much the tensor's statistics
    change under quantization. Higher scores indicate tensors that
    are more sensitive to quantization and may need higher bit-widths.
    
    Args:
        tensor: Input numpy array.
        bit_width: Bit-width to use for sensitivity estimation.
        
    Returns:
        Sensitivity score (lower is less sensitive).
    """
    if tensor.size == 0:
        return 0.0
    
    # Normalize tensor
    tensor_max = np.max(np.abs(tensor))
    if tensor_max == 0:
        return 0.0
    
    normalized = tensor / tensor_max
    
    # Quantize at given bit-width
    max_val = (1 << (bit_width - 1)) - 1
    quantized = np.round(normalized * max_val).astype(np.float32) / max_val
    
    # Compute error metric
    error = np.abs(normalized - quantized)
    return np.mean(error)


def analyze_tensor_sensitivities(tensors: dict, 
                                 base_bit_width: int = 4) -> dict:
    """Analyze sensitivity scores for a set of tensors.
    
    Args:
        tensors: Dictionary mapping layer names to numpy arrays.
        base_bit_width: Reference bit-width for comparison.
        
    Returns:
        Dictionary mapping layer names to sensitivity scores.
    """
    scores = {}
    for name, tensor in tensors.items():
        scores[name] = compute_sensitivity_score(tensor, base_bit_width)
    return scores


def recommend_bit_widths(sensitivities: dict, 
                         min_bits: int = 4, 
                         max_bits: int = 16) -> dict:
    """Recommend bit-widths based on sensitivity scores.
    
    This function maps sensitivity scores to appropriate bit-widths,
    giving more bits to more sensitive tensors.
    
    Args:
        sensitivities: Dictionary mapping layer names to sensitivity scores.
        min_bits: Minimum bit-width to recommend.
        max_bits: Maximum bit-width to recommend.
        
    Returns:
        Dictionary mapping layer names to recommended bit-widths.
    """
    if not sensitivities:
        return {}
    
    scores = list(sensitivities.values())
    if not scores:
        return {}
    
    max_score = max(scores)
    if max_score == 0:
        return {name: min_bits for name in sensitivities}
    
    bit_widths = {}
    for name, score in sensitivities.items():
        # Normalize score and map to bit-width range
        normalized = score / max_score
        # Use a simple linear mapping with some smoothing
        range_bits = max_bits - min_bits
        # Add offset to ensure we don't always assign minimum
        bit_width = int(min_bits + normalized * range_bits * 0.8) + 1
        bit_widths[name] = max(min_bits, min(max_bits, bit_width))
    
    return bit_widths
