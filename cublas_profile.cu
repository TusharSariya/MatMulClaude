#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                           \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st = (call);                                           \
        if (st != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,\
                    (int)st);                                                 \
            exit(1);                                                           \
        }                                                                     \
    } while (0)

void fill_random_float(float *ptr, int total, unsigned seed) {
    srand(seed);
    for (int i = 0; i < total; i++) {
        ptr[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc, char **argv) {
    const char *mode = (argc > 1) ? argv[1] : "sgemm";
    int m = (argc > 2) ? atoi(argv[2]) : 1024;
    int n = (argc > 3) ? atoi(argv[3]) : 1024;
    int k = (argc > 4) ? atoi(argv[4]) : 1024;
    int iters = (argc > 5) ? atoi(argv[5]) : 20;

    size_t bytes_A = (size_t)m * k * sizeof(float);
    size_t bytes_B = (size_t)k * n * sizeof(float);
    size_t bytes_C = (size_t)m * n * sizeof(float);
    int total_A = m * k;
    int total_B = k * n;
    int total_C = m * n;

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    fill_random_float(h_A, total_A, 17);
    fill_random_float(h_B, total_B, 29);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (strcmp(mode, "tensorop") == 0) {
        half *h_Ah = (half *)malloc((size_t)total_A * sizeof(half));
        half *h_Bh = (half *)malloc((size_t)total_B * sizeof(half));
        for (int i = 0; i < total_A; i++) h_Ah[i] = __float2half(h_A[i]);
        for (int i = 0; i < total_B; i++) h_Bh[i] = __float2half(h_B[i]);

        half *d_Ah, *d_Bh, *d_Ch;
        CUDA_CHECK(cudaMalloc(&d_Ah, (size_t)total_A * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_Bh, (size_t)total_B * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_Ch, (size_t)total_C * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_Ah, h_Ah, (size_t)total_A * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Bh, h_Bh, (size_t)total_B * sizeof(half), cudaMemcpyHostToDevice));

        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        for (int i = 0; i < iters; i++) {
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      n, m, k, &alpha,
                                      d_Bh, CUDA_R_16F, n,
                                      d_Ah, CUDA_R_16F, k,
                                      &beta,
                                      d_Ch, CUDA_R_16F, n,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_Ah));
        CUDA_CHECK(cudaFree(d_Bh));
        CUDA_CHECK(cudaFree(d_Ch));
        free(h_Ah);
        free(h_Bh);
    } else {
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
        CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
        CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

        float alpha = 1.0f;
        float beta = 0.0f;
        for (int i = 0; i < iters; i++) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                     &alpha, d_B, n, d_A, k, &beta, d_C, n));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    free(h_A);
    free(h_B);
    return 0;
}
