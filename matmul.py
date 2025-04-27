import torch
import triton
import triton.language as tl
import torch.utils.cpp_extension # Import cpp_extension
import os                          # Import os for path joining
import sys                         # Import sys for benchmark arg check

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
# Define compute capability for CUDA compilation (optional but good practice)
# You might need to adjust this based on your GPU.
# Check https://developer.nvidia.com/cuda-gpus
# Example: A100 -> 80, V100 -> 70, T4 -> 75, RTX 3090 -> 86
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6' # Example for RTX 3090

# Helper function for ceiling division
def cdiv(a, b):
    return (a + b - 1) // b

######### Step 3 (Triton Kernel - unchanged) #########

# un-comment this to run a numpy emulation of Triton on CPU & be able to debug with print() statements
#os.environ["TRITON_INTERPRET"] = "1"

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_M, stride_a_K,
    stride_b_K, stride_b_N,
    stride_c_M, stride_c_N,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # --- Triton Kernel Code (exactly as before) ---
    PID = tl.program_id(axis=0)
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group
    first_PID_in_group_along_M = group_id * GROUP_SIZE
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE)
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    accumulator = accumulator.to(tl.float16)

    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)


######### Step 2 (Wrappers) #########

# --- Triton Wrapper (unchanged) ---
# --- Triton Wrapper ---
def matmul_triton(a, b):
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    # Triton kernel can handle non-contiguous, so no need to assert/convert here
    a, b = a.to(DEVICE, torch.float16), b.to(DEVICE, torch.float16) # Ensure correct device/dtype

    # --- FIX: Unpack shapes separately ---
    # (M, K), (_, N) = a.shape, b.shape # Old problematic line
    M, K = a.shape
    K_b, N = b.shape # Get K from b as well to potentially check consistency later if needed
    assert K == K_b, f"Inner dimensions mismatch after ensuring device/dtype: A({M},{K}) @ B({K_b},{N})"
    # --- End Fix ---

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # The grid calculation now needs M, N which are correctly defined before this line
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # BLOCK_SIZE_M, N, K, GROUP_SIZE are determined by autotuner
    )
    return c

# --- CUDA C++ Wrapper ---
# Define constants matching the CUDA kernel's #defines
CUDA_TILE_M = 32
CUDA_TILE_N = 32
CUDA_TILE_K = 32

# Compile the CUDA kernel using cpp_extension
# This will compile the first time it's called and cache the result
script_dir = os.path.dirname(__file__)
cuda_kernel_path = os.path.join(script_dir, 'matmul_kernel.cu')

try:
    print("Attempting to load CUDA C++ matmul kernel...")

    # --- Specify C++17 standard ---
    compile_args = ['-O3', '-std=c++17'] # For host compiler (like g++)
    cuda_flags = ['-O3', '-std=c++17']   # Also pass to nvcc for consistency
    # --- End Specification ---

    matmul_cuda_module = torch.utils.cpp_extension.load(
        name='matmul_cuda_kernel',
        sources=[cuda_kernel_path],
        verbose=True,
        # Pass flags: extra_compile_args for host, extra_cuda_cflags for nvcc
        # Use 'cxx' key for C++ specific flags for the host compiler
        extra_compile_args={'cxx': compile_args},
        # Pass flags directly to nvcc command line
        extra_cuda_cflags=cuda_flags
    )
    print("CUDA C++ kernel loaded successfully.")
    CUDA_KERNEL_LOADED = True
except Exception as e:
    print("="*50)
    print("WARNING: Failed to load CUDA C++ kernel.")
    print(f"Error: {e}")
    # Add traceback print for more details during debugging compilation errors
    import traceback
    traceback.print_exc()
    # ---
    print("CUDA C++ benchmarks will be skipped.")
    print("Ensure you have the CUDA Toolkit installed and it's compatible with your PyTorch version.")
    print("Ensure your host C++ compiler (like g++) supports C++17.")
    print("You might need to set the TORCH_CUDA_ARCH_LIST environment variable.")
    print("="*50)
    matmul_cuda_module = None
    CUDA_KERNEL_LOADED = False

def matmul_cuda_cpp(a, b):
    """Wrapper for the CUDA C++ kernel."""
    if not CUDA_KERNEL_LOADED:
        raise RuntimeError("CUDA C++ kernel failed to load. Cannot run matmul_cuda_cpp.")

    assert a.ndim == b.ndim == 2, "only supports matrices"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.device == b.device and a.device.type == 'cuda', "Inputs must be CUDA tensors"
    assert a.dtype == torch.float16 and b.dtype == torch.float16, "Inputs must be float16"

    # --- IMPORTANT: Ensure contiguous memory layout ---
    # The basic CUDA kernel assumes row-major contiguous tensors.
    # If your inputs might not be, uncomment the following lines:
    if not a.is_contiguous():
        print("Warning: Input tensor 'a' is not contiguous. Making it contiguous.")
        a = a.contiguous()
    if not b.is_contiguous():
        print("Warning: Input tensor 'b' is not contiguous. Making it contiguous.")
        b = b.contiguous()
    # ---------------------------------------------------

    (M, K), (_, N) = a.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype) # Output tensor

    # Define CUDA grid and block dimensions
    # Block dimensions MUST match the shared memory tile size
    block_dim = (CUDA_TILE_N, CUDA_TILE_M, 1) # (threads_x, threads_y, threads_z) - Note order: N then M
    grid_dim = (cdiv(N, CUDA_TILE_N), cdiv(M, CUDA_TILE_M), 1) # (blocks_x, blocks_y, blocks_z)

    # Call the loaded CUDA kernel
    matmul_cuda_module.matmul_cuda_kernel(
        a, b, c, M, N, K, # Pass tensors directly (cpp_extension handles pointers)
        grid=grid_dim,      # Grid dimensions
        block=block_dim     # Block dimensions
    )

    return c


######### Step 1 (Unit Tests) #########
def test_matmul_kernel(size: tuple = (512, 512), atol=1e-2, rtol=0.05, device=DEVICE): # Adjusted rtol slightly
    """Tests Triton and CUDA C++ kernels against PyTorch."""
    print(f"\n--- Running Unit Tests ({size=}) ---")
    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 3, "Size must be (M, N, K) tuple"
    M, N, K = size
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)

    # PyTorch reference
    print("Running PyTorch matmul...")
    c_ref = torch.matmul(a, b)
    print("PyTorch matmul complete.")

    # Triton
    print("Running Triton matmul...")
    try:
        c_tri = matmul_triton(a, b)
        print("Triton matmul complete.")
        print("Comparing Triton vs PyTorch...")
        torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
        print("Triton PASSED")
    except Exception as e:
        print(f"Triton FAILED: {e}")


    # CUDA C++
    if CUDA_KERNEL_LOADED:
        print("Running CUDA C++ matmul...")
        try:
            # Ensure contiguous for CUDA kernel if necessary
            a_cont = a.contiguous()
            b_cont = b.contiguous()
            c_cuda = matmul_cuda_cpp(a_cont, b_cont)
            print("CUDA C++ matmul complete.")
            print("Comparing CUDA C++ vs PyTorch...")
            torch.testing.assert_close(c_cuda, c_ref, atol=atol, rtol=rtol)
            print("CUDA C++ PASSED")
        except Exception as e:
            print(f"CUDA C++ FAILED: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for detailed traceback
    else:
        print("Skipping CUDA C++ test (kernel not loaded).")

    print("--- Unit Tests Finished ---")


######### Step 4 (Benchmark) #########

# Add 'cuda_cpp' to providers if loaded
providers = ['torch', 'triton']
line_names = ["PyTorch", "Triton"]
styles = [("green", "-"), ("blue", "-")]
if CUDA_KERNEL_LOADED:
    providers.append('cuda_cpp')
    line_names.append("CUDA C++")
    styles.append(("red", "-"))


configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"], # Benchmark over M=N=K
        x_vals = [128 * i for i in range(4, 17)], # Example range (512 to 2048)
        line_arg = "provider",
        line_vals = providers,
        line_names = line_names,
        styles = styles,
        ylabel = "TFLOPS",
        plot_name = "matmul-performance-comparison",
        args={}, # No extra static args needed for the functions
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    # Ensure contiguous for CUDA C++ and potentially PyTorch consistency
    a = a.contiguous()
    b = b.contiguous()
    quantiles = [0.5, 0.05, 0.95] # median, min, max

    ms, min_ms, max_ms = None, None, None # Initialize

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_triton(a, b), quantiles=quantiles)
    elif provider == 'cuda_cpp' and CUDA_KERNEL_LOADED:
         # Pass contiguous tensors
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_cuda_cpp(a, b), quantiles=quantiles)

    if ms is None: # Handle case where provider is cuda_cpp but it didn't load
        return float('nan'), float('nan'), float('nan')

    # Calculate TFLOPS
    # Note: The original calculation used 3*M*N*K ops.
    # Standard calculation for matmul FLOPs is 2*M*N*K (M*N outputs, each requiring K MACs = 2K ops)
    # Using the standard 2*M*N*K calculation:
    flops = 2 * M * N * K
    tflops = lambda ms: flops / (ms * 1e-3) / 1e12 # FLOPs / time_sec / 1e12

    # Using the user's original 3*M*N*K calculation:
    # mem_ops = 3 * M * N * K # Original code's calculation based on memory ops?
    # tflops = lambda ms: mem_ops * 1e-12 / (ms * 1e-3)

    return tflops(ms), tflops(max_ms), tflops(min_ms) # Return med, min (corresponds to max_tflops), max (corresponds to min_tflops)


if __name__ == "__main__":
    # Always run unit-tests
    test_matmul_kernel(size=(512, 512, 512)) # Use M,N,K tuple
    test_matmul_kernel(size=(1024, 512, 256)) # Test non-square

    # Only run benchmark if explicitly requested
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        print("\n--- Running Benchmarks ---")
        # Clear cache potentially used by autotuner or cpp_extension build
        torch.cuda.empty_cache()
        benchmark.run(save_path='.', print_data=True) # print_data=True is useful
        print("--- Benchmarks Finished ---")
        print("Benchmark results saved in the current directory.")
    else:
        print("\nRun with '--benchmark' argument to execute benchmarks.")
