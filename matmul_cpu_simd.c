#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <pthread.h>
#include <unistd.h>
#include <omp.h>

/* ── Single-threaded scalar (baseline) ───────────────────────────────────── */

void matmul_scalar(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

/* ── AVX2+FMA SIMD (single-threaded) ────────────────────────────────────── */
/*
 * Transpose B so that column access becomes row access (consecutive in memory).
 * Then use AVX2 to process 8 floats at a time with FMA (fused multiply-add).
 */

void matmul_simd(const float *A, const float *B, float *C, int n) {
    /* Transpose B into Bt for coalesced access */
    float *Bt = (float *)aligned_alloc(32, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Bt[j * n + i] = B[i * n + j];

    for (int i = 0; i < n; i++) {
        const float *a_row = &A[i * n];
        for (int j = 0; j < n; j++) {
            const float *bt_row = &Bt[j * n];
            __m256 vsum = _mm256_setzero_ps();
            int k = 0;
            /* Process 8 floats per iteration with FMA */
            for (; k + 7 < n; k += 8) {
                __m256 va = _mm256_loadu_ps(&a_row[k]);
                __m256 vb = _mm256_loadu_ps(&bt_row[k]);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            /* Horizontal sum of the 8 floats in vsum */
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);
            /* Handle remaining elements */
            for (; k < n; k++)
                sum += a_row[k] * bt_row[k];
            C[i * n + j] = sum;
        }
    }
    free(Bt);
}

/* ── AVX2+FMA + OpenMP (multithreaded SIMD) ──────────────────────────────── */

void matmul_simd_omp(const float *A, const float *B, float *C, int n) {
    float *Bt = (float *)aligned_alloc(32, (size_t)n * n * sizeof(float));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Bt[j * n + i] = B[i * n + j];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const float *a_row = &A[i * n];
        for (int j = 0; j < n; j++) {
            const float *bt_row = &Bt[j * n];
            __m256 vsum = _mm256_setzero_ps();
            int k = 0;
            for (; k + 7 < n; k += 8) {
                __m256 va = _mm256_loadu_ps(&a_row[k]);
                __m256 vb = _mm256_loadu_ps(&bt_row[k]);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);
            for (; k < n; k++)
                sum += a_row[k] * bt_row[k];
            C[i * n + j] = sum;
        }
    }
    free(Bt);
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int verify(const float *ref, const float *test, int n, const char *label) {
    float max_err = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float err = fabsf(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    int pass = max_err < 1e-2f;
    if (!pass)
        printf("    %s FAILED: max error = %e\n", label, max_err);
    return pass;
}

/* ── Benchmark ───────────────────────────────────────────────────────────── */

void benchmark(int n, int skip_scalar) {
    size_t bytes = (size_t)n * n * sizeof(float);
    double gflops_op = 2.0 * n * n * n / 1e9;

    float *A   = (float *)aligned_alloc(32, bytes);
    float *B   = (float *)aligned_alloc(32, bytes);
    float *C_s = (float *)malloc(bytes);
    float *C_v = (float *)malloc(bytes);
    float *C_vo = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    double scalar_ms = -1, simd_ms, simd_omp_ms, t0, t1;

    /* Scalar */
    if (!skip_scalar) {
        t0 = now_sec();
        matmul_scalar(A, B, C_s, n);
        t1 = now_sec();
        scalar_ms = (t1 - t0) * 1000.0;
    }

    /* SIMD single-threaded */
    t0 = now_sec();
    matmul_simd(A, B, C_v, n);
    t1 = now_sec();
    simd_ms = (t1 - t0) * 1000.0;

    /* SIMD + OpenMP */
    t0 = now_sec();
    matmul_simd_omp(A, B, C_vo, n);
    t1 = now_sec();
    simd_omp_ms = (t1 - t0) * 1000.0;

    /* Verify */
    int ok = 1;
    if (!skip_scalar) {
        ok &= verify(C_s, C_v, n, "simd");
        ok &= verify(C_s, C_vo, n, "simd+omp");
    } else {
        ok &= verify(C_v, C_vo, n, "simd vs simd+omp");
    }

    /* Print */
    if (!skip_scalar) {
        printf("%-6d │ %10.1f ms %7.1f │ %10.1f ms %7.1f %6.1fx │ %10.1f ms %7.1f %6.1fx │ %s\n",
               n,
               scalar_ms, gflops_op / (scalar_ms / 1e3),
               simd_ms, gflops_op / (simd_ms / 1e3), scalar_ms / simd_ms,
               simd_omp_ms, gflops_op / (simd_omp_ms / 1e3), scalar_ms / simd_omp_ms,
               ok ? "PASS" : "FAIL");
    } else {
        printf("%-6d │ %10s   %7s │ %10.1f ms %7.1f %6s  │ %10.1f ms %7.1f %6s  │ %s\n",
               n, "(skip)", "",
               simd_ms, gflops_op / (simd_ms / 1e3), "-",
               simd_omp_ms, gflops_op / (simd_omp_ms / 1e3), "-",
               ok ? "PASS" : "FAIL");
    }

    free(A); free(B); free(C_s); free(C_v); free(C_vo);
}

int main(void) {
    int nthreads = omp_get_max_threads();

    printf("┌──────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  CPU SIMD: Scalar vs AVX2+FMA vs AVX2+FMA+OpenMP                                           │\n");
    printf("│  Threads: %d  │  AVX2 (256-bit = 8 floats/op)  │  FMA (fused multiply-add)                │\n", nthreads);
    printf("└──────────────────────────────────────────────────────────────────────────────────────────────┘\n\n");

    printf("%-6s │ %22s │ %28s │ %28s │ %s\n",
           "N", "Scalar", "AVX2+FMA (1 thread)", "AVX2+FMA+OpenMP", "Check");
    printf("%-6s │ %12s %8s │ %12s %8s %6s │ %12s %8s %6s │\n",
           "", "Time", "GFLOP/s", "Time", "GFLOP/s", "Spdup", "Time", "GFLOP/s", "Spdup");
    printf("───────┼────────────────────────┼──────────────────────────────┼──────────────────────────────┼──────\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int skip_scalar = 0;

    for (int s = 0; s < 5; s++) {
        benchmark(sizes[s], skip_scalar);
        fflush(stdout);
    }

    printf("\n");
    return 0;
}
