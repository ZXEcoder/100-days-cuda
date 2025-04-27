import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def naive_softmax(x):
    x_max = x.max(dim=1)[0] 
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    out = numerator / denominator[:, None]
    return out

@triton.jit 
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
): 
    row_start = tl.program_id(0) 
    row_step = tl.num_programs(0) 
    
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

@triton.jit
def _online_softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_ptr = input_ptr + pid * input_row_stride
    out_start_ptr = output_ptr + pid * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    running_max = tl.full((1,), float('-inf'), dtype=tl.float32)
    running_denom = tl.zeros((1,), dtype=tl.float32)

    for col_start in range(0, n_cols, BLOCK_SIZE):
        offs = col_start + col_offsets
        mask_block = offs < n_cols
        input_ptrs = row_start_ptr + offs

        x = tl.load(input_ptrs, mask=mask_block, other=float('-inf'))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_denom = running_denom * tl.exp(running_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        running_max = new_max

    for col_start in range(0, n_cols, BLOCK_SIZE):
        offs = col_start + col_offsets
        mask_block = offs < n_cols
        input_ptrs = row_start_ptr + offs
        output_ptrs = out_start_ptr + offs

        x = tl.load(input_ptrs, mask=mask_block, other=float('-inf'))
        y = tl.exp(x - running_max)
        y = y / running_denom
        tl.store(output_ptrs, y, mask=mask_block)

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2
    y = torch.empty_like(x)
    kernel = _softmax_kernel.warmup(x, y,
                                x.stride(0), y.stride(0),
                                n_rows, n_cols,
                                BLOCK_SIZE=BLOCK_SIZE,
                                num_stages=num_stages,
                                num_warps=num_warps,
                                grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
    grid = (num_programs, 1, 1)
    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
    )
    return y

def online_softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(min(n_cols, 4096))
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2
    y = torch.empty_like(x)
    kernel = _online_softmax_kernel.warmup(x, y,
                                x.stride(0), y.stride(0),
                                n_rows, n_cols,
                                BLOCK_SIZE=BLOCK_SIZE,
                                num_stages=num_stages,
                                num_warps=num_warps,
                                grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
    grid = (n_rows, 1, 1)
    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
    )
    return y

def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)
    z_tri = softmax(x)
    z_online = online_softmax(x)
    z_ref = torch.softmax(x, axis=1)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(z_online, z_ref, atol=atol, rtol=rtol)
    print("All tests PASSED")

def benchmark_all(M, N_values):
    torch_times = []
    triton_times = []
    online_times = []
    
    for N in N_values:
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        
        torch_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        triton_ms = triton.testing.do_bench(lambda: softmax(x))
        online_ms = triton.testing.do_bench(lambda: online_softmax(x))
        
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        
        torch_times.append(gbps(torch_ms))
        triton_times.append(gbps(triton_ms))
        online_times.append(gbps(online_ms))
        
        print(f"N={N}: PyTorch={gbps(torch_ms):.2f} GB/s, Triton={gbps(triton_ms):.2f} GB/s, Online={gbps(online_ms):.2f} GB/s")
    
    return N_values, torch_times, triton_times, online_times

def plot_results(N_values, torch_times, triton_times, online_times):
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, torch_times, 'g-', label='PyTorch')
    plt.plot(N_values, triton_times, 'b-', label='Triton')
    plt.plot(N_values, online_times, 'r-', label='Online Triton')
    plt.xlabel('N (columns)')
    plt.ylabel('GB/s')
    plt.title('Softmax Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('softmax_performance.png')
    plt.show()

if __name__ == "__main__":
    test_softmax_kernel(size=(1823, 781))
    
    M = 4096
    N_values = [128 * i for i in range(2, 50)]
    
    print("Running benchmarks...")
    N_values, torch_times, triton_times, online_times = benchmark_all(M, N_values)
    
    print("Plotting results...")
    plot_results(N_values, torch_times, triton_times, online_times)
