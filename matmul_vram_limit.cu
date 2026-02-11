#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "    CUDA: %s\n", cudaGetErrorString(err));       \
            return err;                                                       \
        }                                                                     \
    } while (0)

__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int aC = t * BLOCK_SIZE + threadIdx.x;
        int bR = t * BLOCK_SIZE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < n && aC < n) ? A[row * n + aC] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bR < n && col < n) ? B[bR * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

const char *fmt_bytes(size_t b) {
    static char buf[32];
    if (b >= (size_t)1 << 30) snprintf(buf, sizeof(buf), "%.2f GB", b / (double)(1 << 30));
    else snprintf(buf, sizeof(buf), "%.0f MB", b / (double)(1 << 20));
    return buf;
}

cudaError_t try_size(int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    size_t total_need = 3 * bytes;
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("  %5d x %-5d  need %s (3 matrices), free %s ... ",
           n, n, fmt_bytes(total_need), fmt_bytes(free_mem));
    fflush(stdout);

    if (total_need + 32 * 1024 * 1024 > free_mem) {
        printf("SKIP (not enough VRAM)\n");
        return cudaErrorMemoryAllocation;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* Just zero-fill, we only care about whether it runs */
    cudaMemset(d_A, 0, bytes);
    cudaMemset(d_B, 0, bytes);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaError_t sync = cudaEventSynchronize(stop);
    if (sync != cudaSuccess) {
        printf("KERNEL FAIL: %s\n", cudaGetErrorString(sync));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return sync;
    }

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = 2.0 * n * n * n / 1e9 / (ms / 1e3);
    printf("OK  %.2f s  (%.0f GFLOP/s)\n", ms / 1000.0, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return cudaSuccess;
}

int main(void) {
    cudaFree(0);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("GPU VRAM limit finder — %s total, %s free\n\n", fmt_bytes(total_mem), fmt_bytes(free_mem));

    /* Find max N where 3*N*N*4 fits in VRAM */
    int max_n = (int)sqrt((double)free_mem / (3.0 * sizeof(float)));
    max_n = (max_n / 1024) * 1024; /* round down to nearest 1024 */

    /* Test from 16384 up to just past the limit */
    int sizes[] = {16384, 18432, 20480, 22528, 24576, 25600, 26112, 26624};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    int last_good = 0;
    for (int i = 0; i < nsizes; i++) {
        cudaError_t err = try_size(sizes[i]);
        if (err == cudaSuccess) {
            last_good = sizes[i];
        } else {
            break;
        }
    }

    if (last_good > 0) {
        size_t gb = 3UL * last_good * last_good * sizeof(float);
        printf("\n  >>> GPU breaking point: %d x %d (%s for 3 matrices) <<<\n",
               last_good, last_good, fmt_bytes(gb));
    }

    return 0;
}
