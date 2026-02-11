#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* ── Naive kernel (block size as runtime param, no shared mem) ───────────── */

__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

/* ── Tiled kernels — need compile-time block size for shared mem arrays ──── */

template <int BS>
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[BS][BS];
    __shared__ float sB[BS][BS];

    int row = blockIdx.y * BS + threadIdx.y;
    int col = blockIdx.x * BS + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + BS - 1) / BS; t++) {
        int aC = t * BS + threadIdx.x;
        int bR = t * BS + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < n && aC < n) ? A[row * n + aC] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bR < n && col < n) ? B[bR * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < BS; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

/* ── Dispatcher: launches the right template for a given block size ──────── */

typedef void (*launch_fn)(const float *, const float *, float *, int, dim3, dim3, float *);

template <int BS>
void launch_tiled(const float *d_A, const float *d_B, float *d_C, int n,
                  dim3 grid, dim3 block, float *out_ms) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* warm-up */
    matmul_tiled<BS><<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled<BS><<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(out_ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void launch_naive(const float *d_A, const float *d_B, float *d_C, int n,
                  dim3 grid, dim3 block, float *out_ms) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* warm-up */
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(out_ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    int mat_sizes[]   = {512, 1024, 2048, 4096};
    int block_sizes[] = {4, 8, 16, 32};
    int nmat  = sizeof(mat_sizes) / sizeof(mat_sizes[0]);
    int nblk  = sizeof(block_sizes) / sizeof(block_sizes[0]);

    cudaFree(0); /* init context */

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("┌──────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  GPU BLOCK SIZE SWEEP — %s                                          │\n", prop.name);
    printf("│  Max threads/block: %d   Shared mem/block: %zu KB                              │\n",
           prop.maxThreadsPerBlock, prop.sharedMemPerBlock / 1024);
    printf("│  Warp size: %d   SM count: %d                                                   │\n",
           prop.warpSize, prop.multiProcessorCount);
    printf("└──────────────────────────────────────────────────────────────────────────────────┘\n\n");

    /* ── For each matrix size ── */
    for (int m = 0; m < nmat; m++) {
        int n = mat_sizes[m];
        size_t bytes = (size_t)n * n * sizeof(float);
        double gflops_op = 2.0 * n * n * n / 1e9;

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        /* Fill with random data */
        float *h_tmp = (float *)malloc(bytes);
        srand(42);
        for (int i = 0; i < n * n; i++) h_tmp[i] = (float)rand() / RAND_MAX;
        CUDA_CHECK(cudaMemcpy(d_A, h_tmp, bytes, cudaMemcpyHostToDevice));
        for (int i = 0; i < n * n; i++) h_tmp[i] = (float)rand() / RAND_MAX;
        CUDA_CHECK(cudaMemcpy(d_B, h_tmp, bytes, cudaMemcpyHostToDevice));
        free(h_tmp);

        printf("━━━ N = %d ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", n);
        printf("%-12s │ %8s %8s │ %10s %8s %8s │ %10s %8s %8s │\n",
               "Block", "Threads", "Blocks", "Naive(ms)", "GFLOP/s", "Eff%",
               "Tiled(ms)", "GFLOP/s", "Eff%");
        printf("─────────────┼───────────────────┼──────────────────────────────┼──────────────────────────────┤\n");

        float best_naive = 1e9, best_tiled = 1e9;
        int best_naive_bs = 0, best_tiled_bs = 0;

        for (int b = 0; b < nblk; b++) {
            int bs = block_sizes[b];
            int threads_per_block = bs * bs;
            int grid_dim = (n + bs - 1) / bs;
            int total_blocks = grid_dim * grid_dim;

            dim3 block(bs, bs);
            dim3 grid(grid_dim, grid_dim);

            /* Skip if threads per block exceeds hardware limit */
            if (threads_per_block > prop.maxThreadsPerBlock) {
                printf("%2dx%-2d (%4d) │ %8d %8d │ %10s %8s %8s │ %10s %8s %8s │\n",
                       bs, bs, threads_per_block, threads_per_block, total_blocks,
                       "SKIP", "", "", "SKIP", "", "");
                continue;
            }

            float naive_ms, tiled_ms;

            /* Naive */
            launch_naive(d_A, d_B, d_C, n, grid, block, &naive_ms);
            double naive_gf = gflops_op / (naive_ms / 1e3);

            /* Tiled — dispatch to correct template */
            switch (bs) {
                case 4:  launch_tiled<4> (d_A, d_B, d_C, n, grid, block, &tiled_ms); break;
                case 8:  launch_tiled<8> (d_A, d_B, d_C, n, grid, block, &tiled_ms); break;
                case 16: launch_tiled<16>(d_A, d_B, d_C, n, grid, block, &tiled_ms); break;
                case 32: launch_tiled<32>(d_A, d_B, d_C, n, grid, block, &tiled_ms); break;
                default: tiled_ms = -1; break;
            }
            double tiled_gf = gflops_op / (tiled_ms / 1e3);

            /* Occupancy = threads_launched / (SM_count * max_threads_per_SM) */
            double peak_gflops = 1400.0; /* approximate for RTX 3060 Ti FP32 */
            double naive_eff = naive_gf / peak_gflops * 100.0;
            double tiled_eff = tiled_gf / peak_gflops * 100.0;

            printf("%2dx%-2d (%4d) │ %8d %8d │ %10.3f %7.0f  %6.1f%% │ %10.3f %7.0f  %6.1f%% │\n",
                   bs, bs, threads_per_block,
                   threads_per_block, total_blocks,
                   naive_ms, naive_gf, naive_eff,
                   tiled_ms, tiled_gf, tiled_eff);

            if (naive_ms < best_naive) { best_naive = naive_ms; best_naive_bs = bs; }
            if (tiled_ms < best_tiled) { best_tiled = tiled_ms; best_tiled_bs = bs; }
        }

        printf("  Best naive: %dx%d (%.3f ms)  │  Best tiled: %dx%d (%.3f ms)\n\n",
               best_naive_bs, best_naive_bs, best_naive,
               best_tiled_bs, best_tiled_bs, best_tiled);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}
