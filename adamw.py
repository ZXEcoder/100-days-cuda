import torch
import math
import triton
import triton.language as tl
import time
import matplotlib.pyplot as plt
import numpy as np

# Include the original AdamW kernel for comparison
@triton.jit
def adamw_kernel(
    param_ptr, grad_ptr, m_ptr, v_ptr,
    lr, beta1, beta2, eps, weight_decay,
    step, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    g = tl.load(grad_ptr + offsets, mask=mask)
    p = tl.load(param_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)
    
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    
    step_f = tl.full([1], step, dtype=tl.float32)
    
    correction_m = 1.0 - tl.exp(step_f * tl.log(beta1))
    correction_v = 1.0 - tl.exp(step_f * tl.log(beta2))  # Bug fixed: using beta2 instead of beta1
    m_hat = m / correction_m
    v_hat = v / correction_v
    
    update = m_hat / (tl.sqrt(v_hat) + eps) + weight_decay * p
    p_new = p - lr * update
    
    tl.store(param_ptr + offsets, p_new, mask=mask)
    tl.store(m_ptr + offsets, m, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)

def run_adamw_triton(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step):
    n = param.numel()
    grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']),)
    adamw_kernel[grid](
        param, grad, m, v,
        lr, beta1, beta2, eps, weight_decay,
        step, n,
        BLOCK_SIZE=512
    )

@triton.jit
def optimized_adamw_kernel(
    param_ptr, grad_ptr, m_ptr, v_ptr,
    lr, beta1, beta2, eps, weight_decay,
    step, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Use multiple program IDs for better grid utilization
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with proper masking
    p = tl.load(param_ptr + offsets, mask=mask)
    g = tl.load(grad_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)
    
    # Computing bias corrections - FIX: use beta2 for v's correction
    step_f = tl.full([1], step, dtype=tl.float32)
    correction_m = 1.0 - tl.exp(step_f * tl.log(beta1))
    correction_v = 1.0 - tl.exp(step_f * tl.log(beta2))  # Fixed: using beta2 instead of beta1
    
    # Update momentum and velocity
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g * g
    
    # Apply bias correction directly in the update - fuse operations
    m_hat = m_new / correction_m
    v_hat = v_new / correction_v
    
    # Fused weight decay and parameter update
    p_new = p - lr * (m_hat / (tl.sqrt(v_hat) + eps) + weight_decay * p)
    
    # Store results with masking
    tl.store(param_ptr + offsets, p_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

def run_optimized_adamw(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step):
    n = param.numel()
    # Auto-tune block size for different tensor sizes
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    optimized_adamw_kernel[grid](
        param, grad, m, v,
        lr, beta1, beta2, eps, weight_decay,
        step, n,
        BLOCK_SIZE=1024  # Increased block size for better throughput
    )

# Add performance-oriented kernel with auto-tuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def autotuned_adamw_kernel(
    param_ptr, grad_ptr, m_ptr, v_ptr,
    lr, beta1, beta2, eps, weight_decay,
    step, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Coalesced memory access pattern
    p = tl.load(param_ptr + offsets, mask=mask)
    g = tl.load(grad_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)
    
    # Precompute constants for reuse
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2
    
    # Direct power calculation for bias correction
    step_f = tl.full([1], step, dtype=tl.float32)
    correction_m = 1.0 - tl.exp(step_f * tl.log(beta1))
    correction_v = 1.0 - tl.exp(step_f * tl.log(beta2))
    
    # Update rules
    m_new = beta1 * m + one_minus_beta1 * g
    v_new = beta2 * v + one_minus_beta2 * g * g
    
    # Compute update - maximize arithmetic intensity
    denom = tl.sqrt(v_new / correction_v) + eps
    update = (m_new / correction_m) / denom + weight_decay * p
    p_new = p - lr * update
    
    # Store results
    tl.store(param_ptr + offsets, p_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

# FIXED: Removed the explicit n_elements parameter
def run_autotuned_adamw(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step):
    n = param.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    autotuned_adamw_kernel[grid](
        param, grad, m, v,
        lr, beta1, beta2, eps, weight_decay,
        step, n,  # n_elements is automatically passed through n here
        # Removed: n_elements=n  <- This was causing the error
    )

# Modified benchmark function to include all three implementations and calculate GB/s
def benchmark_all(sizes):
    triton_bw = []
    triton_optimized_bw = []
    triton_autotuned_bw = []
    pytorch_bw = []
    
    for N in sizes:
        # Setup tensors
        param = torch.randn(N, device=device, dtype=dtype)
        grad = torch.randn(N, device=device, dtype=dtype)
        m = torch.zeros(N, device=device, dtype=dtype)
        v = torch.zeros(N, device=device, dtype=dtype)
        
        # Calculate bytes processed per iteration
        # For AdamW: 4 reads (param, grad, m, v) and 3 writes (param, m, v)
        # Each element is dtype.itemsize bytes (e.g., 2 bytes for float16)
        bytes_per_iter = (4 + 3) * N * param.element_size()  # in bytes
        
        # Warmup runs
        for _ in range(10):
            run_adamw_triton(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, 1)
            run_optimized_adamw(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, 1)
            run_autotuned_adamw(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, 1)
        
        # Benchmark original Triton
        start = time.time()
        for i in range(1, 21):
            run_adamw_triton(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, i)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 20
        triton_bw.append(bytes_per_iter / triton_time / 1e9)  # Convert to GB/s
        
        # Benchmark optimized Triton
        start = time.time()
        for i in range(1, 21):
            run_optimized_adamw(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, i)
        torch.cuda.synchronize()
        triton_optimized_time = (time.time() - start) / 20
        triton_optimized_bw.append(bytes_per_iter / triton_optimized_time / 1e9)
        
        # Benchmark autotuned Triton
        start = time.time()
        for i in range(1, 21):
            run_autotuned_adamw(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, i)
        torch.cuda.synchronize()
        triton_autotuned_time = (time.time() - start) / 20
        triton_autotuned_bw.append(bytes_per_iter / triton_autotuned_time / 1e9)
        
        # Benchmark PyTorch
        param_torch = param.clone().detach().requires_grad_()
        optimizer = torch.optim.AdamW([param_torch], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)
        start = time.time()
        for i in range(20):
            param_torch.grad = grad.clone().detach()
            optimizer.step()
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 20
        pytorch_bw.append(bytes_per_iter / pytorch_time / 1e9)
        
        print(f"Size: {N}, Original: {triton_bw[-1]:.2f} GB/s, Optimized: {triton_optimized_bw[-1]:.2f} GB/s, " 
              f"Autotuned: {triton_autotuned_bw[-1]:.2f} GB/s, PyTorch: {pytorch_bw[-1]:.2f} GB/s")
    
    return triton_bw, triton_optimized_bw, triton_autotuned_bw, pytorch_bw

# Verify correctness of our optimized implementations
def verify_implementations():
    N = 10000
    param = torch.randn(N, device=device, dtype=dtype)
    grad = torch.randn(N, device=device, dtype=dtype)
    m = torch.zeros(N, device=device, dtype=dtype)
    v = torch.zeros(N, device=device, dtype=dtype)
    
    # PyTorch reference
    param_torch = param.clone().detach().requires_grad_()
    optimizer = torch.optim.AdamW([param_torch], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    param_torch.grad = grad.clone()
    optimizer.step()
    
    # Our implementations
    param1 = param.clone()
    param2 = param.clone()
    param3 = param.clone()
    m1, v1 = m.clone(), v.clone()
    m2, v2 = m.clone(), v.clone()
    m3, v3 = m.clone(), v.clone()
    
    run_adamw_triton(param1, grad, m1, v1, 1e-3, 0.9, 0.999, 1e-8, 1e-2, 1)
    run_optimized_adamw(param2, grad, m2, v2, 1e-3, 0.9, 0.999, 1e-8, 1e-2, 1)
    run_autotuned_adamw(param3, grad, m3, v3, 1e-3, 0.9, 0.999, 1e-8, 1e-2, 1)
    
    print(f"Original vs PyTorch diff: {(param1 - param_torch).abs().max().item()}")
    print(f"Optimized vs PyTorch diff: {(param2 - param_torch).abs().max().item()}")
    print(f"Autotuned vs PyTorch diff: {(param3 - param_torch).abs().max().item()}")

if __name__ == "__main__":
    N = 250000
    dtype = torch.float16
    device = 'cuda'
    
    # Verify implementations
    print("Verifying implementations...")
    verify_implementations()
    
    # Compare performance 
    print("\nComparing basic performance:")
    param = torch.randn(N, device=device, dtype=dtype)
    grad = torch.randn(N, device=device, dtype=dtype)
    m = torch.zeros(N, device=device, dtype=dtype)
    v = torch.zeros(N, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(10):
        run_optimized_adamw(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, 1)
    
    # Original Triton implementation
    start = time.time()
    for i in range(1, 101):
        run_adamw_triton(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, i)
    torch.cuda.synchronize()
    print("Original Triton AdamW Time: {:.6f} sec".format(time.time() - start))
    
    # Optimized Triton implementation
    start = time.time()
    for i in range(1, 101):
        run_optimized_adamw(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, i)
    torch.cuda.synchronize()
    print("Optimized Triton AdamW Time: {:.6f} sec".format(time.time() - start))
    
    # Autotuned Triton implementation
    start = time.time()
    for i in range(1, 101):
        run_autotuned_adamw(param.clone(), grad, m.clone(), v.clone(), 1e-3, 0.9, 0.999, 1e-8, 1e-2, i)
    torch.cuda.synchronize()
    print("Autotuned Triton AdamW Time: {:.6f} sec".format(time.time() - start))
    
    # PyTorch implementation
    param_torch = param.clone().detach().requires_grad_()
    optimizer = torch.optim.AdamW([param_torch], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)
    start = time.time()
    for i in range(100):
        param_torch.grad = grad.clone().detach()
        optimizer.step()
    torch.cuda.synchronize()
    print("PyTorch AdamW Time: {:.6f} sec".format(time.time() - start))
    
    # Run benchmark with different sizes
    print("\nRunning full benchmark...")
    sizes = [10000, 50000, 100000, 250000, 500000, 1000000]
    orig_bw, opt_bw, auto_bw, pytorch_bw = benchmark_all(sizes)
    
    # Plot bandwidth results with all implementations
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, orig_bw, '-o', label='Original Triton')
    plt.plot(sizes, opt_bw, '-s', label='Optimized Triton')
    plt.plot(sizes, auto_bw, '-^', label='Autotuned Triton')
    plt.plot(sizes, pytorch_bw, '-d', label='PyTorch')
    plt.xlabel('Tensor Size (elements)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('AdamW Performance: Triton vs PyTorch')
    plt.xscale('log')  # Use log scale for tensor sizes
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add speedup annotations for key points
    max_size_idx = -1  # Index of largest tensor size
    speedup = auto_bw[max_size_idx] / pytorch_bw[max_size_idx]
    plt.annotate(f"{speedup:.2f}x faster", 
                xy=(sizes[max_size_idx], auto_bw[max_size_idx]),
                xytext=(sizes[max_size_idx]*0.8, auto_bw[max_size_idx]*1.1),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    plt.savefig('adamw_bandwidth_comparison.png')
    
    # Also create a bar chart for the largest size for direct comparison
    plt.figure(figsize=(10, 6))
    implementations = ['Original Triton', 'Optimized Triton', 'Autotuned Triton', 'PyTorch']
    largest_size_bw = [orig_bw[-1], opt_bw[-1], auto_bw[-1], pytorch_bw[-1]]
    
    bars = plt.bar(implementations, largest_size_bw, color=['blue', 'green', 'red', 'purple'])
    plt.ylabel('Bandwidth (GB/s)')
    plt.title(f'AdamW Performance at {sizes[-1]//1000}K elements')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add bandwidth values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom')
                
    # Add speedup compared to PyTorch for each implementation
    for i, bw in enumerate(largest_size_bw[:-1]):  # Skip PyTorch
        speedup = bw / pytorch_bw[-1]
        plt.text(i, bw/2, f"{speedup:.2f}x", ha='center', color='white', fontweight='bold')
    
    plt.savefig('adamw_bandwidth_bars.png')
    plt.show()
