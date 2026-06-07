import torch
import triton
import triton.language as tl

# 1. NEW KERNEL IMPLEMENTATION (Dequantize + Bias Add + ReLU Fusion)
@triton.jit
def fused_quant_add_relu_kernel(
    x_ptr, bias_ptr, scale_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load Int8 data and convert to Float32 on the fly
    x_quant = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr) 

    # Mathematical Fusion
    dequantized_x = x_quant * scale
    fused_sum = dequantized_x + bias

    # Activation Fusion (ReLU)
    activated_output = tl.where(fused_sum > 0.0, fused_sum, 0.0)

    # Store result back to GPU memory
    tl.store(output_ptr + offsets, activated_output, mask=mask)


# 2. RUNNER / LAUNCHER FUNCTION
def launch_fused_kernel(x_quant, bias, scale):
    output = torch.empty_like(bias, dtype=torch.float32)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_quant_add_relu_kernel[grid](
        x_quant, bias, scale, output, n_elements, BLOCK_SIZE=1024
    )
    return output
