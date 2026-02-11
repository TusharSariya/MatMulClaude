#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 16

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));     \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

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
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main(void) {
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    return 0;
}
