#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

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

const char *fmt(size_t b) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%.2f GB", b / (double)(1 << 30));
    return buf;
}

int try_n(int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    size_t need = 3 * bytes;
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("  %5d x %-5d  need %s, free %s ... ", n, n, fmt(need), fmt(free_mem));
    fflush(stdout);

    if (need + 32UL * 1024 * 1024 > free_mem) {
        printf("NO (insufficient VRAM)\n");
        return 0;
    }

    float *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, bytes) != cudaSuccess ||
        cudaMalloc(&d_B, bytes) != cudaSuccess ||
        cudaMalloc(&d_C, bytes) != cudaSuccess) {
        printf("NO (cudaMalloc failed)\n");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return 0;
    }

    cudaMemset(d_A, 0, bytes); cudaMemset(d_B, 0, bytes);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    if (cudaEventSynchronize(stop) != cudaSuccess) {
        printf("NO (kernel failed)\n");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return 0;
    }
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("YES  (%.2f s)\n", ms / 1000.0);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 1;
}

int main(void) {
    cudaFree(0);
    printf("Narrowing GPU VRAM limit (between 22528 and 24576)...\n\n");

    int sizes[] = {22528, 23040, 23552, 23808, 24064, 24320, 24576};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int last = 0;

    for (int i = 0; i < nsizes; i++) {
        if (try_n(sizes[i]))
            last = sizes[i];
        else
            break;
    }

    if (last) {
        size_t gb = 3UL * last * last * sizeof(float);
        printf("\n  >>> MAX GPU SIZE: %d x %d  (%s / %.2f GB VRAM) <<<\n",
               last, last, fmt(gb), 8.0);
    }
    return 0;
}
