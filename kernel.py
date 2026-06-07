import torch
import triton
import triton.language as tl

# Define the fused kernel (already correct in image_5.png)
@triton.jit
def fused_quant_add_relu_kernel(
    x_ptr, bias_ptr, scale_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_quant = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr)

    dequantized_x = x_quant * scale
    fused_sum = dequantized_x + bias
    activated_output = tl.where(fused_sum > 0.0, fused_sum, 0.0)

    tl.store(output_ptr + offsets, activated_output, mask=mask)

def launch_fused_kernel(x_quant, bias, scale):
    output = torch.empty_like(bias, dtype=torch.float32)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_quant_add_relu_kernel[grid](x_quant, bias, scale, output, n_elements, BLOCK_SIZE=1024)
    return output

# --- CORRECTED BENCHMARKING CODE (image_6.png fix) ---

# The API changed. `arg_name` is no longer a top-level argument for Benchmark.
# It's now inferred or passed differently depending on the Triton version.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # The parameter varying on X-axis
        x_vals=[2**i for i in range(12, 22)],  # Values for 'size'
        x_log=True,  # Log scale for X-axis
        # arg_name='size', # This is what caused the TypeError
        # instead, we specify that 'size' is what will be passed.
        line_arg='provider',  # Argument to distinguish curves
        line_vals=['triton', 'pytorch'],
        line_names=['Custom Triton Fused Kernel', 'Standard PyTorch Baseline'],
        styles=[('blue', '-'), ('red', '--')],
        ylabel='GB/s',
        plot_name='Fused Quantized Kernel Inference Speed',
        args={},  # Base arguments
    )
)
def run_benchmark(size, provider):
    # Setup consistent input data for both implementations
    # The 'size' argument is passed from 'x_vals' by the test runner.
    scale = torch.tensor([0.02], device='cuda', dtype=torch.float32)
    x_quant = torch.randint(-127, 127, (size,), device='cuda', dtype=torch.int8)
    bias = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'pytorch':
        pytorch_baseline = lambda: torch.relu((x_quant.to(torch.float32) * scale) + bias)
        ms, min_ms, max_ms = triton.testing.do_bench(pytorch_baseline, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: launch_fused_kernel(x_quant, bias, scale), quantiles=quantiles)

    # Convert milliseconds per run to GB/s (bandwidth)
    # bytes_per_element * number_of_elements / milliseconds * scaling_factors
    # and we have to read x_quant, bias, scale and write output, roughly 3 reads, 1 write per element (or just input/output if idealized).
    # This GB/s math is just illustrative for the comparison.
    gbps = lambda ms: 3 * size * 4 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# Run and show plots
run_benchmark.run(show_plots=True, print_data=True)
