import torch
import triton
# Import your custom launcher from kernel.py
from kernel import launch_fused_kernel

# --- STEP 5: PERFORMANCE BENCHMARKING ---
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 22)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'pytorch'],
        line_names=['Custom Triton Fused Kernel', 'Standard PyTorch Baseline'],
        styles=[('blue', '-'), ('red', '--')],
        ylabel='GB/s',
        plot_name='Fused Quantized Kernel Inference Speed',
        args={},
    )
)
def run_benchmark(size, provider):
    scale = torch.tensor([0.02], device='cuda', dtype=torch.float32)
    x_quant = torch.randint(-127, 127, (size,), device='cuda', dtype=torch.int8)
    bias = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'pytorch':
        pytorch_baseline = lambda: torch.relu((x_quant.to(torch.float32) * scale) + bias)
        ms, min_ms, max_ms = triton.testing.do_bench(pytorch_baseline, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: launch_fused_kernel(x_quant, bias, scale), quantiles=quantiles)
        
    gbps = lambda ms: 3 * size * 4 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    print("--- RUNNING HARDWARE-AWARE INFRASTRUCTURE BENCHMARK ---")
    run_benchmark.run(show_plots=True, print_data=True)
