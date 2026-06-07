import torch

print("--- STEP 3: QUANTIZATION MATHEMATICAL VALIDATION ---")

# Define high-precision simulated network weights (FP32)
original_weights = torch.tensor([[1.5, -2.8, 3.9],
                                 [0.4, 4.7, -1.2]], dtype=torch.float32)

print("1. Original High-Precision Weights (Float32):")
print(original_weights)
print("Data type:", original_weights.dtype)
print("-" * 60)

# Calculate symmetric INT8 quantization scale factor
max_val = original_weights.abs().max()
scale = max_val / 127.0

# Quantize weights by scaling and rounding to signed 8-bit integer
quantized_weights = torch.round(original_weights / scale).to(torch.int8)

print("2. Quantized Tensor (Int8 - Deterministic Compression):")
print(quantized_weights)
print("Data type:", quantized_weights.dtype)
print("-" * 60)

# Dequantize back to FP32 for inference validation
restored_weights = quantized_weights.to(torch.float32) * scale

print("3. Dequantized Weights (Restored Float32 with minimal precision loss):")
print(restored_weights)
print("Data type:", restored_weights.dtype)
print("-" * 60)
