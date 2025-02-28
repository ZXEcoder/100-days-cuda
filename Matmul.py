#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void tiled_mat_mul_kernel(float* A, float* B, float* C, int N1, int N2, int N3) {
    // Compute row and column indices for C
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ float sh_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_b[TILE_WIDTH][TILE_WIDTH];
 
    float value = 0.0f;
    // Loop over all tiles
    for (int phase = 0; phase < ceil((float)N2 / TILE_WIDTH); phase++) {
        // Load a tile of A into shared memory
        if (row < N1 && phase * TILE_WIDTH + tx < N2)
            sh_a[ty][tx] = A[row * N2 + phase * TILE_WIDTH + tx];
        else
            sh_a[ty][tx] = 0.0f;
        
        // Load a tile of B into shared memory
        if (phase * TILE_WIDTH + ty < N2 && col < N3)
            sh_b[ty][tx] = B[(phase * TILE_WIDTH + ty) * N3 + col];
        else
            sh_b[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // Multiply the two tiles together
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_a[ty][k] * sh_b[k][tx];
        
        __syncthreads();
    }
    
    // Write the result to C if within bounds
    if (row < N1 && col < N3)
        C[row * N3 + col] = value;
}

int main() {
    // Matrix dimensions: A is N1 x N2, B is N2 x N3, and C is N1 x N3.
    int N1 = 64;
    int N2 = 64;
    int N3 = 64;
    size_t size_A = N1 * N2 * sizeof(float);
    size_t size_B = N2 * N3 * sizeof(float);
    size_t size_C = N1 * N3 * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // Initialize matrices with some example data.
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            h_A[i * N2 + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    for (int i = 0; i < N2; i++) {
        for (int j = 0; j < N3; j++) {
            h_B[i * N3 + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Allocate device memory.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy matrices A and B to device.
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions.
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N3 + TILE_WIDTH - 1) / TILE_WIDTH, (N1 + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the tiled matrix multiplication kernel.
    tiled_mat_mul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N1, N2, N3);
    cudaDeviceSynchronize();

    // Copy the result matrix C back to the host.
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print a small portion of the result matrix.
    printf("Result matrix C (first 5x5 block):\n");
    for (int i = 0; i < 5 && i < N1; i++) {
        for (int j = 0; j < 5 && j < N3; j++) {
            printf("%0.2f ", h_C[i * N3 + j]);
        }
        printf("\n");
    }

    // Free device memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory.
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
