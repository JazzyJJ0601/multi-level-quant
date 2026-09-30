# Multi-Level Quantization (MLQ)

A lightweight library for applying multi-level quantization to neural network tensors. MLQ assigns different bit-widths to each tensor based on sensitivity analysis, enabling more efficient model compression while preserving accuracy.

## Features

- **Per-tensor quantization**: Assign different bit-widths to different layers
- **Sensitivity analysis**: Automatically identify which tensors are more sensitive to quantization
- **Symmetric uniform quantization**: Standard quantization method optimized for neural networks
- **Pure Python/Numpy**: Minimal dependencies, easy to integrate

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from src.quantizer import multi_level_quantize, dequantize_tensor
from src.sensitivity import analyze_tensor_sensitivities, recommend_bit_widths

# Define your tensors (e.g., model weights)
tensors = {
    'conv1': weights_conv1,
    'fc1': weights_fc1,
    'fc2': weights_fc2,
}

# Analyze sensitivity
sensitivities = analyze_tensor_sensitivities(tensors)

# Recommend bit-widths
bit_widths = recommend_bit_widths(sensitivities, min_bits=4, max_bits=16)

# Quantize
quantized_tensors = multi_level_quantize(tensors, bit_widths)
```

## API Reference

### quantizer.py
- `quantize_tensor(tensor, bit_width)`: Quantize a single tensor
- `multi_level_quantize(tensors, bit_widths)`: Quantize multiple tensors with different bit-widths
- `dequantize_tensor(quantized, bit_width, original_max)`: Convert back to float

### sensitivity.py
- `compute_sensitivity_score(tensor, bit_width)`: Calculate sensitivity for one tensor
- `analyze_tensor_sensitivities(tensors, base_bit_width)`: Analyze all tensors
- `recommend_bit_widths(sensitivities, min_bits, max_bits)`: Map sensitivities to bit-widths

## Testing

```bash
pytest
```

## License

MIT
