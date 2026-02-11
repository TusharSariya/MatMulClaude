#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* ── CPU ─────────────────────────────────────────────────────────────────── */

void matmul_cpu(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* ── GPU naive ───────────────────────────────────────────────────────────── */

__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

/* ── GPU tiled (shared memory) ───────────────────────────────────────────── */

__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        int bRow = t * BLOCK_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

/* ── Verification ────────────────────────────────────────────────────────── */

int verify(const float *ref, const float *test, int n, const char *label) {
    float max_err = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float err = fabsf(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    int pass = max_err < 1e-3f;
    printf("  %-12s max error vs CPU: %e  [%s]\n", label, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

/* ── Benchmark one size ──────────────────────────────────────────────────── */

void benchmark(int n) {
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A    = (float *)malloc(bytes);
    float *h_B    = (float *)malloc(bytes);
    float *h_Ccpu = (float *)malloc(bytes);
    float *h_Cgpu = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    printf("━━━ Matrix size: %d x %d ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", n, n);

    /* CPU */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matmul_cpu(h_A, h_B, h_Ccpu, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    /* GPU setup */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* GPU naive */
    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_Cgpu, d_C, bytes, cudaMemcpyDeviceToHost));

    /* GPU tiled */
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float tiled_ms;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));

    float *h_Ctiled = (float *)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_Ctiled, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Results */
    double gflops = 2.0 * n * n * n / 1e9;

    printf("  CPU          %10.3f ms  (%6.2f GFLOP/s)\n", cpu_ms, gflops / (cpu_ms / 1e3));
    printf("  GPU naive    %10.3f ms  (%6.2f GFLOP/s)  speedup: %.1fx\n",
           naive_ms, gflops / (naive_ms / 1e3), cpu_ms / naive_ms);
    printf("  GPU tiled    %10.3f ms  (%6.2f GFLOP/s)  speedup: %.1fx\n",
           tiled_ms, gflops / (tiled_ms / 1e3), cpu_ms / tiled_ms);

    verify(h_Ccpu, h_Cgpu, n, "naive");
    verify(h_Ccpu, h_Ctiled, n, "tiled");
    printf("\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_Ccpu);
    free(h_Cgpu);
    free(h_Ctiled);
}

int main(void) {
    int sizes[] = {256, 512, 1024, 2048};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    /* Warm up the GPU */
    cudaFree(0);

    for (int i = 0; i < nsizes; i++) {
        benchmark(sizes[i]);
    }

    return 0;
}
