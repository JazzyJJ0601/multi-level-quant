"""Multi-Level Quantization (MLQ) implementation.

This module provides a quantizer that assigns different bit-widths
per tensor based on sensitivity scores.
"""

import numpy as np


def clip_to_bit_width(tensor: np.ndarray, bit_width: int) -> np.ndarray:
    """Clip tensor values to the representable range of a given bit width."""
    # For signed integer representation
    max_val = (1 << (bit_width - 1)) - 1
    min_val = -(1 << (bit_width - 1))
    return np.clip(tensor, min_val, max_val)


def quantize_tensor(tensor: np.ndarray, bit_width: int) -> np.ndarray:
    """Quantize a tensor to a specific bit width.
    
    This performs symmetric uniform quantization by scaling the input
    to the integer range, rounding, and casting to integer.
    """
    # Avoid division by zero for constant tensors
    tensor_max = np.max(np.abs(tensor))
    if tensor_max == 0:
        return np.zeros_like(tensor, dtype=np.int32)
    
    # Normalize to [-1, 1]
    normalized = tensor / tensor_max
    
    # Clip to [-1, 1] for safety
    normalized = np.clip(normalized, -1.0, 1.0)
    
    # Scale to integer range
    max_val = (1 << (bit_width - 1)) - 1
    quantized = np.round(normalized * max_val).astype(np.int32)
    
    return quantized


def multi_level_quantize(tensors: dict, bit_widths: dict) -> dict:
    """Apply multi-level quantization to a dictionary of tensors.
    
    Args:
        tensors: Dictionary mapping layer names to numpy arrays.
        bit_widths: Dictionary mapping layer names to bit-widths.
        
    Returns:
        Dictionary of quantized tensors.
    """
    result = {}
    for name, tensor in tensors.items():
        if name in bit_widths:
            result[name] = quantize_tensor(tensor, bit_widths[name])
        else:
            # Default to 8-bit if no specific bit-width is assigned
            result[name] = quantize_tensor(tensor, 8)
    return result


def dequantize_tensor(quantized: np.ndarray, bit_width: int, 
                      original_max: float) -> np.ndarray:
    """Dequantize a tensor back to float.
    
    Args:
        quantized: The quantized integer tensor.
        bit_width: The bit-width used for quantization.
        original_max: The max absolute value of the original tensor.
        
    Returns:
        Dequantized float tensor.
    """
    max_val = (1 << (bit_width - 1)) - 1
    if max_val == 0:
        return np.zeros_like(quantized, dtype=np.float32)
    return (quantized.astype(np.float32) / max_val) * original_max
