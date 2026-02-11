#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define CPU_TIMEOUT_SEC 120.0   /* skip CPU if previous run exceeded this */
#define GPU_TIMEOUT_MS  30000.0 /* flag GPU if it exceeds this */

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "    CUDA error: %s\n", cudaGetErrorString(err)); \
            return err;                                                       \
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

/* ── GPU tiled ───────────────────────────────────────────────────────────── */

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
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

double time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

const char *fmt_bytes(size_t b) {
    static char buf[32];
    if (b >= (size_t)1 << 30)
        snprintf(buf, sizeof(buf), "%.2f GB", b / (double)(1 << 30));
    else
        snprintf(buf, sizeof(buf), "%.0f MB", b / (double)(1 << 20));
    return buf;
}

const char *fmt_time(double ms) {
    static char buf[32];
    if (ms < 1.0)
        snprintf(buf, sizeof(buf), "%.3f ms", ms);
    else if (ms < 1000.0)
        snprintf(buf, sizeof(buf), "%.1f ms", ms);
    else
        snprintf(buf, sizeof(buf), "%.2f s", ms / 1000.0);
    return buf;
}

/* ── GPU stress ──────────────────────────────────────────────────────────── */

cudaError_t gpu_stress(int n, float *out_ms, double *out_gflops) {
    size_t bytes = (size_t)n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* Fill with random data on host, copy over */
    float *h_tmp = (float *)malloc(bytes);
    if (!h_tmp) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        fprintf(stderr, "    Host malloc failed for %s\n", fmt_bytes(bytes));
        return cudaErrorMemoryAllocation;
    }
    for (int i = 0; i < n * n; i++)
        h_tmp[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_A, h_tmp, bytes, cudaMemcpyHostToDevice));
    for (int i = 0; i < n * n; i++)
        h_tmp[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_B, h_tmp, bytes, cudaMemcpyHostToDevice));
    free(h_tmp);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Warm-up run */
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    /* Timed run */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(out_ms, start, stop));

    *out_gflops = 2.0 * n * n * n / 1e9 / (*out_ms / 1e3);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return cudaSuccess;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    int sizes[] = {256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
                   9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    srand(42);
    cudaFree(0); /* warm up GPU context */

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("┌──────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  MATRIX MULTIPLY STRESS TEST                                            │\n");
    printf("│  System RAM: ~%s available                                         │\n",
           fmt_bytes(23UL * 1024 * 1024 * 1024));
    printf("│  GPU VRAM:   %s total, %s free                              │\n",
           fmt_bytes(total_mem), fmt_bytes(free_mem));
    printf("│  CPU timeout: %.0fs  │  GPU OOM = graceful skip                         │\n",
           CPU_TIMEOUT_SEC);
    printf("└──────────────────────────────────────────────────────────────────────────┘\n\n");

    printf("%-8s │ %10s │ %-14s %8s │ %-14s %8s │ %s\n",
           "N", "Mem/matrix", "CPU time", "GFLOP/s", "GPU time", "GFLOP/s", "Speedup");
    printf("─────────┼────────────┼──────────────────────────┼─────────────────────────┼────────\n");

    int cpu_gave_up = 0;
    double last_cpu_ms = 0;
    int gpu_gave_up = 0;
    int gpu_max_n = 0;
    int cpu_max_n = 0;

    for (int s = 0; s < nsizes; s++) {
        int n = sizes[s];
        size_t bytes = (size_t)n * n * sizeof(float);
        double gflops_op = 2.0 * n * n * n / 1e9;

        printf("%-8d │ %10s │ ", n, fmt_bytes(bytes));
        fflush(stdout);

        /* ── CPU ── */
        double cpu_ms = -1, cpu_gf = 0;
        if (!cpu_gave_up) {
            /* Check if we can even allocate 3 matrices */
            float *a = (float *)malloc(bytes);
            float *b = (float *)malloc(bytes);
            float *c = (float *)malloc(bytes);
            if (!a || !b || !c) {
                printf("HOST OOM    %8s │ ", "");
                cpu_gave_up = 1;
            } else {
                for (int i = 0; i < n * n; i++) {
                    a[i] = (float)rand() / RAND_MAX;
                    b[i] = (float)rand() / RAND_MAX;
                }
                double t0 = time_sec();
                matmul_cpu(a, b, c, n);
                double t1 = time_sec();
                cpu_ms = (t1 - t0) * 1000.0;
                cpu_gf = gflops_op / (cpu_ms / 1e3);
                last_cpu_ms = cpu_ms;
                cpu_max_n = n;

                printf("%-14s %7.1f │ ", fmt_time(cpu_ms), cpu_gf);

                /* Auto-skip CPU if it's getting too slow */
                if (cpu_ms / 1000.0 > CPU_TIMEOUT_SEC) {
                    cpu_gave_up = 1;
                }
            }
            free(a); free(b); free(c);
        } else {
            printf("(skipped)    %8s │ ", "");
        }
        fflush(stdout);

        /* ── GPU ── */
        float gpu_ms_f = -1;
        double gpu_gf = 0;
        if (!gpu_gave_up) {
            size_t needed = 3 * bytes + (size_t)(64 * 1024 * 1024); /* 3 matrices + 64MB headroom */
            cudaMemGetInfo(&free_mem, &total_mem);
            if (needed > free_mem) {
                printf("VRAM OOM     %8s │ ", "");
                gpu_gave_up = 1;
            } else {
                cudaError_t err = gpu_stress(n, &gpu_ms_f, &gpu_gf);
                if (err != cudaSuccess) {
                    printf("CUDA ERR     %8s │ ", "");
                    gpu_gave_up = 1;
                    cudaDeviceReset();
                    cudaFree(0);
                } else {
                    gpu_max_n = n;
                    printf("%-14s %7.1f │ ", fmt_time(gpu_ms_f), gpu_gf);
                }
            }
        } else {
            printf("(skipped)    %8s │ ", "");
        }

        /* ── Speedup ── */
        if (cpu_ms > 0 && gpu_ms_f > 0) {
            printf("%.0fx", cpu_ms / gpu_ms_f);
        } else {
            printf("-");
        }
        printf("\n");
        fflush(stdout);

        if (cpu_gave_up && gpu_gave_up) {
            printf("\nBoth CPU and GPU have hit their limits. Stopping.\n");
            break;
        }
    }

    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  RESULTS SUMMARY                                                        │\n");
    printf("├──────────────────────────────────────────────────────────────────────────┤\n");
    if (cpu_max_n > 0)
        printf("│  CPU max practical size:  %5d x %-5d  (%.1fs at this size)           │\n",
               cpu_max_n, cpu_max_n, last_cpu_ms / 1000.0);
    printf("│  CPU bottleneck: O(N^3) time — doubles in size = 8x slower              │\n");
    if (gpu_max_n > 0) {
        size_t gb = 3UL * gpu_max_n * gpu_max_n * sizeof(float);
        printf("│  GPU max size:            %5d x %-5d  (3 matrices = %s)         │\n",
               gpu_max_n, gpu_max_n, fmt_bytes(gb));
    }
    printf("│  GPU bottleneck: VRAM — 3 x N x N x 4 bytes must fit in %s       │\n",
           fmt_bytes(total_mem));
    printf("└──────────────────────────────────────────────────────────────────────────┘\n");

    return 0;
}
