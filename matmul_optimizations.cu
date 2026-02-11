#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define BLOCK 16
#define WARMUP_ITERS 3
#define MEASURE_ITERS 10

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

static inline double flops_gemm_mnk(int m, int n, int k) {
    return 2.0 * (double)m * (double)n * (double)k;
}

static inline double flops_gemm(int n) {
    return flops_gemm_mnk(n, n, n);
}

static inline double gflops_from_ms(double flops, float ms) {
    return flops / (ms * 1e6);
}

void fill_random(float *ptr, int n, unsigned seed) {
    srand(seed);
    int total = n * n;
    for (int i = 0; i < total; i++) {
        ptr[i] = (float)rand() / (float)RAND_MAX;
    }
}

void fill_random_total(float *ptr, int total, unsigned seed) {
    srand(seed);
    for (int i = 0; i < total; i++) {
        ptr[i] = (float)rand() / (float)RAND_MAX;
    }
}

void cpu_matmul_ref(const float *A, const float *B, float *C, int n) {
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

void cpu_matmul_ref_mnk(const float *A, const float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                sum += A[i * k + kk] * B[kk * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

float max_abs_err(const float *a, const float *b, int total) {
    float mx = 0.0f;
    for (int i = 0; i < total; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

__global__ void float_to_half_kernel(const float *in, half *out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void half_to_float_kernel(const half *in, float *out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = __half2float(in[idx]);
    }
}

__global__ void tiled_baseline(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[BLOCK][BLOCK];
    __shared__ float sB[BLOCK][BLOCK];

    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    float sum = 0.0f;

    int tiles = (n + BLOCK - 1) / BLOCK;
    for (int t = 0; t < tiles; t++) {
        int aCol = t * BLOCK + threadIdx.x;
        int bRow = t * BLOCK + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

// Launch bounds variant (same algorithm, different compiler resource target).
__global__ __launch_bounds__(256, 4)
void tiled_launch_bounds(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[BLOCK][BLOCK];
    __shared__ float sB[BLOCK][BLOCK];

    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    float sum = 0.0f;

    int tiles = (n + BLOCK - 1) / BLOCK;
    for (int t = 0; t < tiles; t++) {
        int aCol = t * BLOCK + threadIdx.x;
        int bRow = t * BLOCK + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

// Register tiling: each thread computes 1x2 outputs (two columns).
__global__ void tiled_reg_1x2(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[BLOCK][BLOCK];
    __shared__ float sB[BLOCK][BLOCK * 2];

    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col0 = blockIdx.x * (BLOCK * 2) + threadIdx.x;
    int col1 = col0 + BLOCK;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int tiles = (n + BLOCK - 1) / BLOCK;
    for (int t = 0; t < tiles; t++) {
        int aCol = t * BLOCK + threadIdx.x;
        int bRow = t * BLOCK + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < n && col0 < n) ? B[bRow * n + col0] : 0.0f;
        sB[threadIdx.y][threadIdx.x + BLOCK] = (bRow < n && col1 < n) ? B[bRow * n + col1] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK; k++) {
            float a = sA[threadIdx.y][k];
            sum0 += a * sB[k][threadIdx.x];
            sum1 += a * sB[k][threadIdx.x + BLOCK];
        }
        __syncthreads();
    }

    if (row < n && col0 < n) C[row * n + col0] = sum0;
    if (row < n && col1 < n) C[row * n + col1] = sum1;
}

// Vectorized global loads (float4) into shared memory.
__global__ void tiled_vec4_loads(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[BLOCK][BLOCK];
    __shared__ float sB[BLOCK][BLOCK];

    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    float sum = 0.0f;

    int tiles = n / BLOCK; // vector path assumes n is multiple of 16
    for (int t = 0; t < tiles; t++) {
        int aColBase = t * BLOCK;
        int bRow = t * BLOCK + threadIdx.y;

        if (threadIdx.x < 4) {
            int v = threadIdx.x;
            int aCol4 = aColBase + v * 4;
            int bCol4 = blockIdx.x * BLOCK + v * 4;

            const float4 *aPtr = (const float4 *)&A[row * n + aCol4];
            const float4 *bPtr = (const float4 *)&B[bRow * n + bCol4];
            float4 va = *aPtr;
            float4 vb = *bPtr;

            sA[threadIdx.y][v * 4 + 0] = va.x;
            sA[threadIdx.y][v * 4 + 1] = va.y;
            sA[threadIdx.y][v * 4 + 2] = va.z;
            sA[threadIdx.y][v * 4 + 3] = va.w;

            sB[threadIdx.y][v * 4 + 0] = vb.x;
            sB[threadIdx.y][v * 4 + 1] = vb.y;
            sB[threadIdx.y][v * 4 + 2] = vb.z;
            sB[threadIdx.y][v * 4 + 3] = vb.w;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) C[row * n + col] = sum;
}

// Software pipelined double buffer in shared memory (synchronous loads).
__global__ void tiled_double_buffer(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[2][BLOCK][BLOCK];
    __shared__ float sB[2][BLOCK][BLOCK];

    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    float sum = 0.0f;

    int tiles = (n + BLOCK - 1) / BLOCK;
    int buf = 0;

    int aCol0 = threadIdx.x;
    int bRow0 = threadIdx.y;
    sA[buf][threadIdx.y][threadIdx.x] = (row < n && aCol0 < n) ? A[row * n + aCol0] : 0.0f;
    sB[buf][threadIdx.y][threadIdx.x] = (bRow0 < n && col < n) ? B[bRow0 * n + col] : 0.0f;
    __syncthreads();

    for (int t = 0; t < tiles; t++) {
        int next = buf ^ 1;
        if (t + 1 < tiles) {
            int aCol = (t + 1) * BLOCK + threadIdx.x;
            int bRow = (t + 1) * BLOCK + threadIdx.y;
            sA[next][threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
            sB[next][threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        }

        #pragma unroll
        for (int k = 0; k < BLOCK; k++) {
            sum += sA[buf][threadIdx.y][k] * sB[buf][k][threadIdx.x];
        }
        __syncthreads();
        buf = next;
    }

    if (row < n && col < n) C[row * n + col] = sum;
}

typedef void (*KernelLaunchFn)(const float *, const float *, float *, int);

void launch_baseline(const float *A, const float *B, float *C, int n) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    tiled_baseline<<<grid, block>>>(A, B, C, n);
}

void launch_launchbounds(const float *A, const float *B, float *C, int n) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    tiled_launch_bounds<<<grid, block>>>(A, B, C, n);
}

void launch_reg1x2(const float *A, const float *B, float *C, int n) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK * 2 - 1) / (BLOCK * 2), (n + BLOCK - 1) / BLOCK);
    tiled_reg_1x2<<<grid, block>>>(A, B, C, n);
}

void launch_vec4(const float *A, const float *B, float *C, int n) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    tiled_vec4_loads<<<grid, block>>>(A, B, C, n);
}

void launch_double_buffer(const float *A, const float *B, float *C, int n) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    tiled_double_buffer<<<grid, block>>>(A, B, C, n);
}

float benchmark_cublas_sgemm_mnk(cublasHandle_t handle, const float *d_A, const float *d_B, float *d_C,
                                 int m, int n, int k);
float benchmark_cublas_tensorop_mnk(cublasHandle_t handle, const half *d_Ah, const half *d_Bh, half *d_Ch,
                                    int m, int n, int k);

float benchmark_kernel(KernelLaunchFn fn, const float *d_A, const float *d_B, float *d_C, int n) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        fn(d_A, d_B, d_C, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < MEASURE_ITERS; i++) {
        fn(d_A, d_B, d_C, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / MEASURE_ITERS;
}

float benchmark_cublas_sgemm(cublasHandle_t handle, const float *d_A, const float *d_B, float *d_C, int n) {
    return benchmark_cublas_sgemm_mnk(handle, d_A, d_B, d_C, n, n, n);
}

float benchmark_cublas_sgemm_mnk(cublasHandle_t handle, const float *d_A, const float *d_B, float *d_C,
                                 int m, int n, int k) {
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                 &alpha, d_B, n, d_A, k, &beta, d_C, n));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < MEASURE_ITERS; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                 &alpha, d_B, n, d_A, k, &beta, d_C, n));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / MEASURE_ITERS;
}

float benchmark_cublas_tensorop(cublasHandle_t handle, const half *d_Ah, const half *d_Bh, half *d_Ch, int n) {
    return benchmark_cublas_tensorop_mnk(handle, d_Ah, d_Bh, d_Ch, n, n, n);
}

float benchmark_cublas_tensorop_mnk(cublasHandle_t handle, const half *d_Ah, const half *d_Bh, half *d_Ch,
                                    int m, int n, int k) {
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k, &alpha,
                                  d_Bh, CUDA_R_16F, n,
                                  d_Ah, CUDA_R_16F, k,
                                  &beta,
                                  d_Ch, CUDA_R_16F, n,
                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < MEASURE_ITERS; i++) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k, &alpha,
                                  d_Bh, CUDA_R_16F, n,
                                  d_Ah, CUDA_R_16F, k,
                                  &beta,
                                  d_Ch, CUDA_R_16F, n,
                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / MEASURE_ITERS;
}

void run_method_1_and_2_and_4(int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    int total = n * n;
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);

    fill_random(h_A, n, 42);
    fill_random(h_B, n, 1234);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    printf("\n=== Method 1/2/4 Kernel Table (N=%d) ===\n", n);
    printf("%-24s %10s %12s\n", "Variant", "Time (ms)", "GFLOP/s");

    double flops = flops_gemm(n);

    float ms_base = benchmark_kernel(launch_baseline, d_A, d_B, d_C, n);
    printf("%-24s %10.3f %12.1f\n", "Tiled baseline", ms_base, gflops_from_ms(flops, ms_base));

    float ms_lb = benchmark_kernel(launch_launchbounds, d_A, d_B, d_C, n);
    printf("%-24s %10.3f %12.1f\n", "Tiled launch_bounds", ms_lb, gflops_from_ms(flops, ms_lb));

    float ms_db = benchmark_kernel(launch_double_buffer, d_A, d_B, d_C, n);
    printf("%-24s %10.3f %12.1f\n", "Tiled double-buffer", ms_db, gflops_from_ms(flops, ms_db));

    float ms_reg = benchmark_kernel(launch_reg1x2, d_A, d_B, d_C, n);
    printf("%-24s %10.3f %12.1f\n", "Register tile 1x2", ms_reg, gflops_from_ms(flops, ms_reg));

    if (n % BLOCK == 0) {
        float ms_vec = benchmark_kernel(launch_vec4, d_A, d_B, d_C, n);
        printf("%-24s %10.3f %12.1f\n", "Vec4 global loads", ms_vec, gflops_from_ms(flops, ms_vec));
    } else {
        printf("%-24s %10s %12s\n", "Vec4 global loads", "n/a", "n/a");
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float ms_cublas = benchmark_cublas_sgemm(handle, d_A, d_B, d_C, n);
    printf("%-24s %10.3f %12.1f\n", "cuBLAS SGEMM", ms_cublas, gflops_from_ms(flops, ms_cublas));

    half *d_Ah, *d_Bh, *d_Ch;
    CUDA_CHECK(cudaMalloc(&d_Ah, total * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_Bh, total * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_Ch, total * sizeof(half)));
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads>>>(d_A, d_Ah, total);
    float_to_half_kernel<<<blocks, threads>>>(d_B, d_Bh, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_tc = benchmark_cublas_tensorop(handle, d_Ah, d_Bh, d_Ch, n);
    printf("%-24s %10.3f %12.1f\n", "cuBLAS TensorOp FP16", ms_tc, gflops_from_ms(flops, ms_tc));

    // Quick accuracy check on SGEMM + TensorOp against CPU reference (single size only).
    printf("\nAccuracy spot-check (N=%d):\n", n);
    cpu_matmul_ref(h_A, h_B, h_ref, n);

    CUDA_CHECK(cudaMemcpy(h_out, d_C, bytes, cudaMemcpyDeviceToHost)); // from last SGEMM call
    printf("  SGEMM max abs error: %.6e\n", max_abs_err(h_ref, h_out, total));

    half_to_float_kernel<<<blocks, threads>>>(d_Ch, d_C, total);
    CUDA_CHECK(cudaMemcpy(h_out, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("  TensorOp max abs error: %.6e\n", max_abs_err(h_ref, h_out, total));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_Ah));
    CUDA_CHECK(cudaFree(d_Bh));
    CUDA_CHECK(cudaFree(d_Ch));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_ref);
    free(h_out);
}

void run_rectangular_cublas_validation(int m, int n, int k) {
    size_t bytes_A = (size_t)m * k * sizeof(float);
    size_t bytes_B = (size_t)k * n * sizeof(float);
    size_t bytes_C = (size_t)m * n * sizeof(float);
    int total_A = m * k;
    int total_B = k * n;
    int total_C = m * n;

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_ref = (float *)malloc(bytes_C);
    float *h_out = (float *)malloc(bytes_C);
    fill_random_total(h_A, total_A, 2026);
    fill_random_total(h_B, total_B, 2027);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    double flops = flops_gemm_mnk(m, n, k);
    float ms_sgemm = benchmark_cublas_sgemm_mnk(handle, d_A, d_B, d_C, m, n, k);

    half *d_Ah, *d_Bh, *d_Ch;
    CUDA_CHECK(cudaMalloc(&d_Ah, total_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_Bh, total_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_Ch, total_C * sizeof(half)));
    int threads = 256;
    int blocks_A = (total_A + threads - 1) / threads;
    int blocks_B = (total_B + threads - 1) / threads;
    float_to_half_kernel<<<blocks_A, threads>>>(d_A, d_Ah, total_A);
    float_to_half_kernel<<<blocks_B, threads>>>(d_B, d_Bh, total_B);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_tensor = benchmark_cublas_tensorop_mnk(handle, d_Ah, d_Bh, d_Ch, m, n, k);

    cpu_matmul_ref_mnk(h_A, h_B, h_ref, m, n, k);
    CUDA_CHECK(cudaMemcpy(h_out, d_C, bytes_C, cudaMemcpyDeviceToHost));
    float err_sgemm = max_abs_err(h_ref, h_out, total_C);

    int blocks_C = (total_C + threads - 1) / threads;
    half_to_float_kernel<<<blocks_C, threads>>>(d_Ch, d_C, total_C);
    CUDA_CHECK(cudaMemcpy(h_out, d_C, bytes_C, cudaMemcpyDeviceToHost));
    float err_tensor = max_abs_err(h_ref, h_out, total_C);

    printf("\n=== Rectangular cuBLAS Validation (M=%d, N=%d, K=%d) ===\n", m, n, k);
    printf("Row-major A(MxK) * B(KxN) -> C(MxN) via cuBLAS column-major mapping.\n");
    printf("%-24s %10s %12s %14s\n", "Variant", "Time (ms)", "GFLOP/s", "Max Abs Error");
    printf("%-24s %10.3f %12.1f %14.6e\n", "cuBLAS SGEMM", ms_sgemm,
           gflops_from_ms(flops, ms_sgemm), err_sgemm);
    printf("%-24s %10.3f %12.1f %14.6e\n", "cuBLAS TensorOp FP16", ms_tensor,
           gflops_from_ms(flops, ms_tensor), err_tensor);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_Ah));
    CUDA_CHECK(cudaFree(d_Bh));
    CUDA_CHECK(cudaFree(d_Ch));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_ref);
    free(h_out);
}

void run_method_3_overlap(int n, int batches) {
    size_t bytes = (size_t)n * n * sizeof(float);
    double flops_total = flops_gemm(n) * batches;

    float *hA_page = (float *)malloc(bytes);
    float *hB_page = (float *)malloc(bytes);
    float *hC_page = (float *)malloc(bytes);
    fill_random(hA_page, n, 7);
    fill_random(hB_page, n, 13);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

    // 1) Pageable synchronous pipeline.
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < batches; i++) {
        CUDA_CHECK(cudaMemcpy(dA, hA_page, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB_page, bytes, cudaMemcpyHostToDevice));
        tiled_baseline<<<grid, block>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaMemcpy(hC_page, dC, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_pageable = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pageable, start, stop));

    // 2) Pinned synchronous pipeline.
    float *hA_pin, *hB_pin, *hC_pin;
    CUDA_CHECK(cudaMallocHost(&hA_pin, bytes));
    CUDA_CHECK(cudaMallocHost(&hB_pin, bytes));
    CUDA_CHECK(cudaMallocHost(&hC_pin, bytes));
    memcpy(hA_pin, hA_page, bytes);
    memcpy(hB_pin, hB_page, bytes);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < batches; i++) {
        CUDA_CHECK(cudaMemcpy(dA, hA_pin, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB_pin, bytes, cudaMemcpyHostToDevice));
        tiled_baseline<<<grid, block>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaMemcpy(hC_pin, dC, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_pinned_sync = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pinned_sync, start, stop));

    // 3) Pinned + 2 streams double buffering.
    cudaStream_t s0, s1;
    CUDA_CHECK(cudaStreamCreate(&s0));
    CUDA_CHECK(cudaStreamCreate(&s1));
    float *dA2[2], *dB2[2], *dC2[2];
    CUDA_CHECK(cudaMalloc(&dA2[0], bytes));
    CUDA_CHECK(cudaMalloc(&dA2[1], bytes));
    CUDA_CHECK(cudaMalloc(&dB2[0], bytes));
    CUDA_CHECK(cudaMalloc(&dB2[1], bytes));
    CUDA_CHECK(cudaMalloc(&dC2[0], bytes));
    CUDA_CHECK(cudaMalloc(&dC2[1], bytes));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < batches; i++) {
        int buf = i & 1;
        cudaStream_t s = buf ? s1 : s0;
        CUDA_CHECK(cudaMemcpyAsync(dA2[buf], hA_pin, bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dB2[buf], hB_pin, bytes, cudaMemcpyHostToDevice, s));
        tiled_baseline<<<grid, block, 0, s>>>(dA2[buf], dB2[buf], dC2[buf], n);
        CUDA_CHECK(cudaMemcpyAsync(hC_pin, dC2[buf], bytes, cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s0));
    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_2stream = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_2stream, start, stop));

    // 4) CUDA graph captured from pinned sync pipeline.
    cudaStream_t gstream;
    CUDA_CHECK(cudaStreamCreate(&gstream));
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    CUDA_CHECK(cudaStreamBeginCapture(gstream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaMemcpyAsync(dA, hA_pin, bytes, cudaMemcpyHostToDevice, gstream));
    CUDA_CHECK(cudaMemcpyAsync(dB, hB_pin, bytes, cudaMemcpyHostToDevice, gstream));
    tiled_baseline<<<grid, block, 0, gstream>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaMemcpyAsync(hC_pin, dC, bytes, cudaMemcpyDeviceToHost, gstream));
    CUDA_CHECK(cudaStreamEndCapture(gstream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < batches; i++) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, gstream));
    }
    CUDA_CHECK(cudaStreamSynchronize(gstream));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_graph = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_graph, start, stop));

    printf("\n=== Method 3 End-to-End Overlap Table (N=%d, batches=%d) ===\n", n, batches);
    printf("%-30s %10s %14s\n", "Pipeline", "Time (ms)", "End-to-end GFLOP/s");
    printf("%-30s %10.3f %14.1f\n", "Pageable memcpy + sync", ms_pageable, gflops_from_ms(flops_total, ms_pageable));
    printf("%-30s %10.3f %14.1f\n", "Pinned memcpy + sync", ms_pinned_sync, gflops_from_ms(flops_total, ms_pinned_sync));
    printf("%-30s %10.3f %14.1f\n", "Pinned + 2 streams", ms_2stream, gflops_from_ms(flops_total, ms_2stream));
    printf("%-30s %10.3f %14.1f\n", "Pinned + CUDA graph", ms_graph, gflops_from_ms(flops_total, ms_graph));

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(gstream));

    CUDA_CHECK(cudaFree(dA2[0]));
    CUDA_CHECK(cudaFree(dA2[1]));
    CUDA_CHECK(cudaFree(dB2[0]));
    CUDA_CHECK(cudaFree(dB2[1]));
    CUDA_CHECK(cudaFree(dC2[0]));
    CUDA_CHECK(cudaFree(dC2[1]));
    CUDA_CHECK(cudaStreamDestroy(s0));
    CUDA_CHECK(cudaStreamDestroy(s1));

    CUDA_CHECK(cudaFreeHost(hA_pin));
    CUDA_CHECK(cudaFreeHost(hB_pin));
    CUDA_CHECK(cudaFreeHost(hC_pin));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA_page);
    free(hB_page);
    free(hC_page);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char **argv) {
    int square_n = 1024;
    int batches = 8;
    int rect_m = 768, rect_n = 1024, rect_k = 512;

    if (argc > 1) square_n = atoi(argv[1]);
    if (argc > 2) batches = atoi(argv[2]);
    if (argc > 5) {
        rect_m = atoi(argv[3]);
        rect_n = atoi(argv[4]);
        rect_k = atoi(argv[5]);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SMs: %d, compute capability: %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    // Warm up context.
    CUDA_CHECK(cudaFree(0));

    run_method_1_and_2_and_4(square_n);
    run_method_3_overlap(square_n, batches);
    run_rectangular_cublas_validation(rect_m, rect_n, rect_k);

    return 0;
}
