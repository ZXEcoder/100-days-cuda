#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define FILTER_RADIUS 1
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)

// Constant memory for the filter (assume a 3x3 filter for FILTER_RADIUS=1)
__constant__ float F[FILTER_WIDTH][FILTER_WIDTH];

__global__ void convolution_cached_tiled_2D(const float *N, float *P, int width, int height) {
    // Compute global row and column for the output element
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Shared memory tile size: tile plus halo borders on all sides.
    const int shared_size = TILE_DIM + 2 * FILTER_RADIUS;
    __shared__ float N_s[shared_size][shared_size];

    // Compute indices in shared memory (offset by FILTER_RADIUS for the halo)
    int shared_x = threadIdx.x + FILTER_RADIUS;
    int shared_y = threadIdx.y + FILTER_RADIUS;

    // --- Load the central tile element ---
    if (row < height && col < width)
        N_s[shared_y][shared_x] = N[row * width + col];
    else
        N_s[shared_y][shared_x] = 0.0f;

    // --- Load halo regions ---
    // Left halo
    if (threadIdx.x < FILTER_RADIUS) {
        int halo_col = col - FILTER_RADIUS;
        if (halo_col >= 0 && row < height)
            N_s[shared_y][threadIdx.x] = N[row * width + halo_col];
        else
            N_s[shared_y][threadIdx.x] = 0.0f;
    }
    // Right halo
    if (threadIdx.x >= TILE_DIM - FILTER_RADIUS) {
        int halo_col = col + FILTER_RADIUS;
        if (halo_col < width && row < height)
            N_s[shared_y][shared_x + FILTER_RADIUS] = N[row * width + halo_col];
        else
            N_s[shared_y][shared_x + FILTER_RADIUS] = 0.0f;
    }
    // Top halo
    if (threadIdx.y < FILTER_RADIUS) {
        int halo_row = row - FILTER_RADIUS;
        if (halo_row >= 0 && col < width)
            N_s[threadIdx.y][shared_x] = N[halo_row * width + col];
        else
            N_s[threadIdx.y][shared_x] = 0.0f;
    }
    // Bottom halo
    if (threadIdx.y >= TILE_DIM - FILTER_RADIUS) {
        int halo_row = row + FILTER_RADIUS;
        if (halo_row < height && col < width)
            N_s[shared_y + FILTER_RADIUS][shared_x] = N[halo_row * width + col];
        else
            N_s[shared_y + FILTER_RADIUS][shared_x] = 0.0f;
    }
    // Top-left corner
    if (threadIdx.x < FILTER_RADIUS && threadIdx.y < FILTER_RADIUS) {
        int halo_col = col - FILTER_RADIUS;
        int halo_row = row - FILTER_RADIUS;
        if (halo_row >= 0 && halo_col >= 0)
            N_s[threadIdx.y][threadIdx.x] = N[halo_row * width + halo_col];
        else
            N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    // Top-right corner
    if (threadIdx.x >= TILE_DIM - FILTER_RADIUS && threadIdx.y < FILTER_RADIUS) {
        int halo_col = col + FILTER_RADIUS;
        int halo_row = row - FILTER_RADIUS;
        if (halo_row >= 0 && halo_col < width)
            N_s[threadIdx.y][shared_x + FILTER_RADIUS] = N[halo_row * width + halo_col];
        else
            N_s[threadIdx.y][shared_x + FILTER_RADIUS] = 0.0f;
    }
    // Bottom-left corner
    if (threadIdx.x < FILTER_RADIUS && threadIdx.y >= TILE_DIM - FILTER_RADIUS) {
        int halo_col = col - FILTER_RADIUS;
        int halo_row = row + FILTER_RADIUS;
        if (halo_row < height && halo_col >= 0)
            N_s[shared_y + FILTER_RADIUS][threadIdx.x] = N[halo_row * width + halo_col];
        else
            N_s[shared_y + FILTER_RADIUS][threadIdx.x] = 0.0f;
    }
    // Bottom-right corner
    if (threadIdx.x >= TILE_DIM - FILTER_RADIUS && threadIdx.y >= TILE_DIM - FILTER_RADIUS) {
        int halo_col = col + FILTER_RADIUS;
        int halo_row = row + FILTER_RADIUS;
        if (halo_row < height && halo_col < width)
            N_s[shared_y + FILTER_RADIUS][shared_x + FILTER_RADIUS] = N[halo_row * width + halo_col];
        else
            N_s[shared_y + FILTER_RADIUS][shared_x + FILTER_RADIUS] = 0.0f;
    }

    __syncthreads();

    // --- Convolution Computation ---
    if (row < height && col < width) {
        float Pvalue = 0.0f;
        // For a 3x3 filter, sum over FILTER_WIDTH x FILTER_WIDTH
        // Mathematical formulation:
        //   P(row, col) = Σ_{i=0}^{FILTER_WIDTH-1} Σ_{j=0}^{FILTER_WIDTH-1}
        //                 F[i][j] * N_s[shared_y - FILTER_RADIUS + i][shared_x - FILTER_RADIUS + j]
        for (int fRow = 0; fRow < FILTER_WIDTH; fRow++) {
            for (int fCol = 0; fCol < FILTER_WIDTH; fCol++) {
                Pvalue += F[fRow][fCol] * N_s[shared_y - FILTER_RADIUS + fRow][shared_x - FILTER_RADIUS + fCol];
            }
        }
        P[row * width + col] = Pvalue;
    }
}

//////////////////////////
// Main Function Example
//////////////////////////
int main() {
    // Image dimensions (for example, a 64x64 image)
    int width = 64;
    int height = 64;
    int image_size = width * height * sizeof(float);

    // Allocate host memory for the input and output images.
    float *h_input  = (float*)malloc(image_size);
    float *h_output = (float*)malloc(image_size);

    // Initialize the input image with some example values.
    // For clarity, we fill it with sequential numbers.
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = (float)(i * width + j);
        }
    }

    // Define a 3x3 filter, for example, an averaging filter.
    float h_filter[FILTER_WIDTH * FILTER_WIDTH] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };

    // Copy the filter into constant memory.
    cudaMemcpyToSymbol(F, h_filter, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

    // Allocate device memory.
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, image_size);
    cudaMalloc((void**)&d_output, image_size);

    // Copy the input image to device memory.
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions.
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    // Launch the convolution kernel.
    convolution_cached_tiled_2D<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy the result back to host memory.
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // Print a small portion of the output image.
    printf("Output (first 8x8 block):\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%6.2f ", h_output[i * width + j]);
        }
        printf("\n");
    }

    // Free allocated memory.
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
