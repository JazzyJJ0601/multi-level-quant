"""Tests for multi-level quantization."""

import numpy as np
from src.quantizer import quantize_tensor, multi_level_quantize, dequantize_tensor
from src.sensitivity import compute_sensitivity_score, recommend_bit_widths


def test_quantize_tensor_basic():
    """Test basic tensor quantization."""
    tensor = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)
    result = quantize_tensor(tensor, bit_width=8)
    assert result.shape == tensor.shape
    assert result.dtype == np.int32


def test_quantize_zeros():
    """Test quantization of zero tensor."""
    tensor = np.zeros((5, 5), dtype=np.float32)
    result = quantize_tensor(tensor, bit_width=4)
    assert np.all(result == 0)


def test_multi_level_quantize():
    """Test multi-level quantization with different bit-widths."""
    tensors = {
        'layer1': np.array([0.1, 0.5, 0.9], dtype=np.float32),
        'layer2': np.array([0.2, 0.4, 0.6], dtype=np.float32)
    }
    bit_widths = {'layer1': 4, 'layer2': 8}
    result = multi_level_quantize(tensors, bit_widths)
    assert 'layer1' in result
    assert 'layer2' in result


def test_recommend_bit_widths():
    """Test bit-width recommendation based on sensitivity."""
    sensitivities = {'layer1': 0.01, 'layer2': 0.5}
    result = recommend_bit_widths(sensitivities, min_bits=4, max_bits=16)
    assert 'layer1' in result
    assert 'layer2' in result
    assert result['layer2'] > result['layer1']


def test_quantize_dequantize_roundtrip():
    """Test that dequantization recovers original values approximately."""
    tensor = np.random.randn(10, 10).astype(np.float32)
    original_max = np.max(np.abs(tensor))
    quantized = quantize_tensor(tensor, bit_width=8)
    recovered = dequantize_tensor(quantized, bit_width=8, original_max=original_max)
    error = np.mean(np.abs(tensor - recovered))
    assert error < 0.1
