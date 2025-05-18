import torch
import triton
import triton.language as tl
import time
import matplotlib.pyplot as plt
import pandas as pd # For better table display

# Environment check for Triton
# print(f"PyTorch version: {torch.__version__}")
# print(f"Triton version: {triton.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"Current CUDA device: {torch.cuda.current_device()}")
#     print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 512}, num_warps=8),
        # Add more configs if D can be larger, or if D is often not a multiple of these
        # e.g., if D can be 1024, add a config for BLOCK_SIZE_D: 1024
    ],
    key=['D_FEATURE_DIM'], # D_FEATURE_DIM will be D from the input
)
@triton.jit
def swiglu_kernel_corrected(
    x_ptr,            # Pointer to input tensor x (B, 2*D)
    out_ptr,          # Pointer to output tensor out (B, D)
    B_BATCH_DIM,      # Batch dimension B
    D_FEATURE_DIM,    # Feature dimension D (of the output, half of x's feature dim)
    stride_x_batch,   # Stride of x for batch dimension
    stride_x_feature, # Stride of x for feature dimension
    stride_o_batch,   # Stride of out for batch dimension
    stride_o_feature, # Stride of out for feature dimension
    # D_FEATURE_DIM is already passed as a runtime arg,
    # but making it a tl.constexpr via autotuner key allows Triton to specialize.
    # BLOCK_SIZE_D is the tiling size for the D dimension, from autotuner.
    BLOCK_SIZE_D: tl.constexpr
):
    """
    Triton kernel for SwiGLU activation: out = (x1 * sigmoid(x1)) * x2
    where x is split into x1 and x2 along the feature dimension.
    x1 = x[..., :D_FEATURE_DIM]
    x2 = x[..., D_FEATURE_DIM:2*D_FEATURE_DIM]
    """
    # Get the program ID for the current batch
    pid_batch = tl.program_id(axis=0)  # Each program instance handles one batch entry

    # Iterate over the D_FEATURE_DIM in blocks of BLOCK_SIZE_D
    # This loop ensures the entire D_FEATURE_DIM is processed, even if D_FEATURE_DIM > BLOCK_SIZE_D.
    # The `range` here is evaluated during JIT compilation. Triton unrolls this loop
    # if D_FEATURE_DIM and BLOCK_SIZE_D are compile-time constants.
    for d_start_offset in range(0, D_FEATURE_DIM, BLOCK_SIZE_D):
        # Create a block of offsets for the current tile: [0, 1, ..., BLOCK_SIZE_D-1]
        d_offsets_in_tile = tl.arange(0, BLOCK_SIZE_D)
        
        # Actual column indices in the D_FEATURE_DIM for the current tile
        current_d_indices = d_start_offset + d_offsets_in_tile
        
        # Create a mask to guard against out-of-bounds access for the last tile,
        # if D_FEATURE_DIM is not perfectly divisible by BLOCK_SIZE_D.
        d_mask = current_d_indices < D_FEATURE_DIM
        
        # --- Calculate pointers for the input tensor x ---
        # x has shape (B_BATCH_DIM, 2 * D_FEATURE_DIM)
        # x1 corresponds to columns 0 to D_FEATURE_DIM-1
        # x2 corresponds to columns D_FEATURE_DIM to 2*D_FEATURE_DIM-1
        
        # Pointers for the x1 part of the input tensor for the current tile
        # x1 uses columns `current_d_indices` from the original x's feature dimension
        x1_tile_ptr = x_ptr + (pid_batch * stride_x_batch +
                               current_d_indices * stride_x_feature)
        
        # Pointers for the x2 part of the input tensor for the current tile
        # x2 uses columns `D_FEATURE_DIM + current_d_indices` from the original x's feature dimension
        x2_col_indices_in_x = D_FEATURE_DIM + current_d_indices
        x2_tile_ptr = x_ptr + (pid_batch * stride_x_batch +
                               x2_col_indices_in_x * stride_x_feature)
        
        # --- Calculate pointers for the output tensor out ---
        # out has shape (B_BATCH_DIM, D_FEATURE_DIM)
        # Output columns are `current_d_indices`
        o_tile_ptr = out_ptr + (pid_batch * stride_o_batch +
                                current_d_indices * stride_o_feature)
        
        # Load x1 and x2 for the current tile, applying the mask
        # `other=0.0` ensures that masked-out elements are loaded as 0.0, preventing NaNs
        # if they were involved in computation (though masked stores prevent writing them).
        x1_tile = tl.load(x1_tile_ptr, mask=d_mask, other=0.0)
        x2_tile = tl.load(x2_tile_ptr, mask=d_mask, other=0.0) # x2 also uses d_mask as it corresponds to the same output elements

        # --- Perform SwiGLU computation: (x1 * sigmoid(x1)) * x2 ---
        # Convert to float32 for intermediate calculations to maintain precision,
        # especially for sigmoid and subsequent multiplications.
        x1_f32 = x1_tile.to(tl.float32)
        x2_f32 = x2_tile.to(tl.float32)
        
        # Calculate SiLU(x1): x1 * sigmoid(x1)
        silu_x1_f32 = x1_f32 * tl.sigmoid(x1_f32)
        
        # Multiply by x2 to get the SwiGLU result
        swiglu_out_f32 = silu_x1_f32 * x2_f32
        
        # Convert the result back to the original data type of x1 (e.g., float16)
        output_tile_val = swiglu_out_f32.to(x1_tile.dtype)
        
        # Store the result for the current tile, applying the mask
        tl.store(o_tile_ptr, output_tile_val, mask=d_mask)


def swiglu_triton(x: torch.Tensor) -> torch.Tensor:
    B, two_D = x.shape
    assert two_D % 2 == 0, "Input feature dimension must be even for SwiGLU"
    D = two_D // 2
    
    # Create an empty output tensor with the correct shape and device
    out = torch.empty((B, D), device=x.device, dtype=x.dtype)

    # Grid for launching the kernel: one program instance per batch entry
    grid = (B,) # Launch B programs

    # Call the corrected Triton kernel
    swiglu_kernel_corrected[grid](
        x, out,             # Input and output tensors
        B, D,               # Batch and feature dimensions
        x.stride(0), x.stride(1),  # Strides for input tensor x
        out.stride(0), out.stride(1), # Strides for output tensor out
        # D_FEATURE_DIM=D is implicitly passed to autotuner via key matching.
        # BLOCK_SIZE_D is determined by autotuner.
    )
    return out


def calc_gbps(num_elements, dtype_size, time_seconds):
    """Calculates GB/s based on number of elements, dtype size, and time."""
    if time_seconds == 0:
        return float('inf')
    total_bytes = num_elements * dtype_size
    return total_bytes / time_seconds / 1e9


def benchmark():
    B_values = [256, 512, 1024, 2048]  # Batch sizes
    D_values = [256, 512, 1024, 2048, 4096]   # Feature dimensions (output D)

    results_data = []

    # Use float16 for typical GPU acceleration
    dtype = torch.float16
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    
    print(f"Benchmarking with dtype: {dtype}")
    print("-" * 80)
    print(f"{'Batch (B)':<10} | {'Dim (D)':<10} | {'Triton (ms)':<15} | {'PyTorch (ms)':<15} | {'Speedup (T/P)':<15} | {'Max Diff':<15} | {'Triton GB/s':<15} | {'PyTorch GB/s':<15}")
    print("-" * 80)

    for D in D_values:
        for B in B_values:
            # Ensure GPU has enough memory
            try:
                x = torch.randn((B, 2 * D), device='cuda', dtype=dtype)
            except RuntimeError as e:
                print(f"Skipping B={B}, D={D} due to OOM: {e}")
                continue

            # --- Triton Warmup & Benchmark ---
            # First run for JIT compilation and autotuning
            try:
                _ = swiglu_triton(x) 
                torch.cuda.synchronize()
                
                # Actual benchmark run
                t0 = time.time()
                y_triton = swiglu_triton(x)
                torch.cuda.synchronize()
                t1 = time.time()
                triton_time_ms = (t1 - t0) * 1000
            except Exception as e:
                print(f"Error during Triton execution for B={B}, D={D}: {e}")
                triton_time_ms = float('inf')
                y_triton = torch.zeros((B,D), device='cuda', dtype=dtype) # Placeholder for diff


            # --- PyTorch Reference Implementation ---
            x1_torch, x2_torch = x.chunk(2, dim=-1) # More robust way to split
            # x1_torch, x2_torch = x[:, :D], x[:, D:] # Original split
            
            # Warmup for PyTorch
            _ = x1_torch * torch.sigmoid(x1_torch) * x2_torch 
            torch.cuda.synchronize()
            
            # Actual benchmark run
            t2 = time.time()
            y_torch = x1_torch * torch.sigmoid(x1_torch) * x2_torch
            torch.cuda.synchronize()
            t3 = time.time()
            pytorch_time_ms = (t3 - t2) * 1000

            # Calculate difference
            try:
                max_diff = (y_triton - y_torch).abs().max().item()
            except Exception as e:
                max_diff = float('nan') # Error in calculation

            # Calculate GB/s
            # Input elements: B * 2 * D. Output elements: B * D.
            # Total elements read: 2 * B * D. Total elements written: B * D.
            # We'll use total elements accessed: 3 * B * D for GB/s.
            # Or, use input elements (B * 2 * D) as per original script for consistency.
            # Let's use total elements for a more comprehensive bandwidth measure.
            total_elements_moved = B * 3 * D 
            # elements_for_gbps = B * 2 * D # Original approach
            elements_for_gbps = total_elements_moved


            triton_gbps = calc_gbps(elements_for_gbps, dtype_size, triton_time_ms / 1000)
            pytorch_gbps = calc_gbps(elements_for_gbps, dtype_size, pytorch_time_ms / 1000)
            
            speedup = pytorch_time_ms / triton_time_ms if triton_time_ms > 0 else float('inf')

            print(f"{B:<10} | {D:<10} | {triton_time_ms:<15.3f} | {pytorch_time_ms:<15.3f} | {speedup:<15.2f}x | {max_diff:<15.6e} | {triton_gbps:<15.2f} | {pytorch_gbps:<15.2f}")

            results_data.append({
                "B": B, "D": D,
                "triton_time_ms": triton_time_ms,
                "pytorch_time_ms": pytorch_time_ms,
                "triton_gbps": triton_gbps,
                "pytorch_gbps": pytorch_gbps,
                "max_diff": max_diff,
                "speedup": speedup
            })

            del x, y_triton, y_torch, x1_torch, x2_torch
            torch.cuda.empty_cache()
    
    print("-" * 80)
    return pd.DataFrame(results_data)


def plot_results(df_results):
    if df_results.empty:
        print("No results to plot.")
        return

    # Ensure numeric types for plotting
    for col in ['triton_time_ms', 'pytorch_time_ms', 'triton_gbps', 'pytorch_gbps', 'speedup']:
        df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

    df_results = df_results.dropna(subset=['triton_time_ms', 'pytorch_time_ms', 'triton_gbps', 'pytorch_gbps'])
    if df_results.empty:
        print("Not enough valid data to plot after dropping NaNs.")
        return
        
    labels = [f"B={b},D={d}" for b, d in zip(df_results["B"], df_results["D"])]
    x_indices = range(len(labels))

    plt.figure(figsize=(18, 12)) # Adjusted figure size

    # Plot 1: Execution Time
    plt.subplot(2, 2, 1)
    plt.plot(x_indices, df_results["triton_time_ms"], label='Triton', marker='o', linestyle='-')
    plt.plot(x_indices, df_results["pytorch_time_ms"], label='PyTorch', marker='x', linestyle='--')
    plt.xticks(x_indices, labels, rotation=75, ha='right') # Improved rotation
    plt.ylabel("Time (ms)")
    plt.title("Execution Time (Lower is Better)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Memory Bandwidth (GB/s)
    plt.subplot(2, 2, 2)
    plt.plot(x_indices, df_results["triton_gbps"], label='Triton', marker='o', linestyle='-')
    plt.plot(x_indices, df_results["pytorch_gbps"], label='PyTorch', marker='x', linestyle='--')
    plt.xticks(x_indices, labels, rotation=75, ha='right') # Improved rotation
    plt.ylabel("Throughput (GB/s) (Higher is Better)")
    plt.title("Memory Bandwidth (Total Elements Moved)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 3: Speedup (PyTorch Time / Triton Time)
    plt.subplot(2, 2, 3)
    plt.bar(x_indices, df_results["speedup"], color='skyblue', label='Speedup (PyTorch/Triton)')
    plt.axhline(1.0, color='grey', linestyle='--', label='Baseline (1x)')
    plt.xticks(x_indices, labels, rotation=75, ha='right') # Improved rotation
    plt.ylabel("Speedup Factor")
    plt.title("Speedup of Triton over PyTorch (Higher is Better)")
    plt.legend()
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)


    # Plot 4: Max Difference
    plt.subplot(2, 2, 4)
    plt.plot(x_indices, df_results["max_diff"], label='Max Absolute Difference', marker='.', linestyle=':', color='red')
    plt.xticks(x_indices, labels, rotation=75, ha='right') # Improved rotation
    plt.ylabel("Max Absolute Difference")
    plt.title("Numerical Accuracy (Lower is Better)")
    plt.yscale('log') # Use log scale if differences are very small or vary widely
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout(pad=3.0) # Add padding
    
    # Save the plot to a file
    plot_filename = "swiglu_benchmark_results.png"
    try:
        plt.savefig(plot_filename)
        print(f"\nPlot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # To display the plot in environments that support it directly (e.g. Jupyter)
    # plt.show() # Comment this out if running in a script-only environment or if savefig is preferred

if __name__ == "__main__":
    # It's good practice to clear cache for autotuner if kernel definition changes significantly
    # However, Triton usually handles this. Forcing can be done by deleting the cache dir.
    # Example: `rm -rf ~/.triton/cache` (use with caution)
    
    # Check if CUDA is available before running
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting benchmark.")
    else:
        print(f"Running on: {torch.cuda.get_device_name(0)}")
        results_df = benchmark()
        if results_df is not None and not results_df.empty:
            print("\n--- Benchmark Results Summary ---")
            print(results_df.to_string())
            plot_results(results_df)
        else:
            print("Benchmarking did not produce results.")

