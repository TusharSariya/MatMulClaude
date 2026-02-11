// Standard C headers for I/O and dynamic memory
#include <stdio.h>
#include <stdlib.h>
// CUDA runtime API: needed for cudaMalloc, cudaMemcpy, cudaEvent*, etc.
#include <cuda_runtime.h>

// Matrix dimension: we use N×N matrices (1024×1024 = 1M elements per matrix)
#define N 1024
// Block size: each thread block is BLOCK_SIZE×BLOCK_SIZE threads (16×16 = 256 threads)
#define BLOCK_SIZE 16

// Macro to wrap CUDA API calls and abort with a message if any call fails.
// "call" is the CUDA function (e.g. cudaMalloc). We store its return value in err,
// and if it's not cudaSuccess, we print file, line, and error string then exit.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Kernel that runs on the GPU. "__global__" means: callable from host, runs on device.
// Each thread computes one element of the output matrix C = A * B.
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    // Compute this thread's row index: which block (blockIdx.y) × block height (blockDim.y) + thread row (threadIdx.y)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Compute this thread's column index: block column × block width + thread column
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: only compute if (row, col) is inside the n×n matrix (important when n isn't a multiple of block size)
    if (row < n && col < n) {
        // Accumulator for the dot product of row of A and column of B
        float sum = 0.0f;
        // Dot product: sum over k of A[row,k] * B[k,col]
        for (int k = 0; k < n; k++) {
            // A is row-major: element (row, k) is at A[row * n + k]; B(k,col) at B[k * n + col]
            sum += A[row * n + k] * B[k * n + col];
        }
        // Write the result for this (row, col) into C
        C[row * n + col] = sum;
    }
}

// Tiled (shared-memory) kernel: each block loads a tile of A and B into shared memory,
// then all threads in the block cooperate to compute a tile of C. Much faster than naive due to reuse.
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    // Shared memory: visible only to threads in this block. sA and sB are tiles of A and B.
    // All threads in the block can read/write these; they live in fast on-chip memory.
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // Global row and column this thread is responsible for (same idea as naive kernel)
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tile "phases": we need (n / BLOCK_SIZE) tiles along the k dimension (rounded up)
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Column index in A for this tile: which column of A we load (threadIdx.x picks the column within the tile)
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        // Row index in B for this tile: which row of B we load (threadIdx.y picks the row within the tile)
        int bRow = t * BLOCK_SIZE + threadIdx.y;

        // Cooperatively load one tile of A into sA. Each thread loads one element; bounds check for non-multiple sizes
        sA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        // Cooperatively load one tile of B into sB
        sB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        // Barrier: wait until ALL threads in the block have finished writing to shared memory before any thread reads
        __syncthreads();

        // Compute partial dot product using the tile in shared memory (no global memory access here)
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        // Barrier before overwriting shared memory in the next iteration
        __syncthreads();
    }

    // Write the final sum to global memory only if (row, col) is in bounds
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main(void) {
    // Total bytes for one N×N float matrix (each float is 4 bytes)
    size_t bytes = N * N * sizeof(float);

    // Host (CPU) pointers: "h_" convention. Allocate with standard malloc for matrices A, B, and result C
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Seed the random number generator so results are reproducible
    srand(42);
    // Fill A and B with random floats in [0, 1]
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Device (GPU) pointers: "d_" convention. We'll allocate these with cudaMalloc
    float *d_A, *d_B, *d_C;
    // Allocate bytes of global memory on the GPU for each matrix; CUDA_CHECK aborts on failure
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    // Copy input data from host memory to device memory (host -> device)
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch configuration: how many threads per block and how many blocks
    // threads: 16×16 = 256 threads per block
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    // blocks: enough blocks so that the grid covers the whole N×N matrix (ceiling division)
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Events are used to measure GPU time (more accurate than CPU timers for kernels)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* --- Naive kernel --- */
    // Record the time when the GPU reaches this point (start)
    CUDA_CHECK(cudaEventRecord(start));
    // Launch kernel: <<<blocks, threads>>> means grid of blocks, each block has threads threads
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    // Record the time when the GPU reaches this point (stop)
    CUDA_CHECK(cudaEventRecord(stop));
    // Block the CPU until the stop event has been recorded (so we know the kernel finished)
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Elapsed time in milliseconds between start and stop
    float naive_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
    // Copy result matrix C from GPU back to CPU so we can print/verify
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("GPU naive  (%dx%d): %.3f ms\n", N, N, naive_ms);
    printf("C[0][0] = %f, C[%d][%d] = %f\n", h_C[0], N - 1, N - 1, h_C[(N - 1) * N + (N - 1)]);

    /* --- Tiled kernel --- */
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float tiled_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("GPU tiled  (%dx%d): %.3f ms\n", N, N, tiled_ms);
    printf("C[0][0] = %f, C[%d][%d] = %f\n", h_C[0], N - 1, N - 1, h_C[(N - 1) * N + (N - 1)]);

    // Free GPU events and device memory (good practice to match every cudaMalloc/cudaEventCreate)
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
