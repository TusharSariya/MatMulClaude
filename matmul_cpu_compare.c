#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <omp.h>

/* ── Single-threaded ─────────────────────────────────────────────────────── */

void matmul_single(const float *A, const float *B, float *C, int n) {
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

/* ── Pthreads ────────────────────────────────────────────────────────────── */

typedef struct {
    const float *A;
    const float *B;
    float *C;
    int n;
    int row_start;
    int row_end;
} thread_arg_t;

void *matmul_worker(void *arg) {
    thread_arg_t *t = (thread_arg_t *)arg;
    for (int i = t->row_start; i < t->row_end; i++) {
        for (int j = 0; j < t->n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < t->n; k++) {
                sum += t->A[i * t->n + k] * t->B[k * t->n + j];
            }
            t->C[i * t->n + j] = sum;
        }
    }
    return NULL;
}

void matmul_pthreads(const float *A, const float *B, float *C, int n, int nthreads) {
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    thread_arg_t *args = malloc(nthreads * sizeof(thread_arg_t));

    int rows_per = n / nthreads;
    int extra = n % nthreads;
    int row = 0;

    for (int t = 0; t < nthreads; t++) {
        args[t] = (thread_arg_t){A, B, C, n, row, row + rows_per + (t < extra ? 1 : 0)};
        row = args[t].row_end;
        pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
    }

    for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], NULL);

    free(threads);
    free(args);
}

/* ── OpenMP ──────────────────────────────────────────────────────────────── */

void matmul_openmp(const float *A, const float *B, float *C, int n) {
    #pragma omp parallel for collapse(2) schedule(static)
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
    int pass = max_err < 1e-3f;
    if (!pass)
        printf("    %s FAILED: max error = %e\n", label, max_err);
    return pass;
}

/* ── Benchmark ───────────────────────────────────────────────────────────── */

void benchmark(int n, int nthreads, int skip_single) {
    size_t bytes = (size_t)n * n * sizeof(float);
    double gflops_op = 2.0 * n * n * n / 1e9;

    float *A = malloc(bytes);
    float *B = malloc(bytes);
    float *C_single = malloc(bytes);
    float *C_pt = malloc(bytes);
    float *C_omp = malloc(bytes);

    if (!A || !B || !C_single || !C_pt || !C_omp) {
        printf("%-6d │  HOST OOM\n", n);
        free(A); free(B); free(C_single); free(C_pt); free(C_omp);
        return;
    }

    srand(42);
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    double single_ms = -1, pt_ms, omp_ms, t0, t1;

    /* Single-threaded */
    if (!skip_single) {
        t0 = now_sec();
        matmul_single(A, B, C_single, n);
        t1 = now_sec();
        single_ms = (t1 - t0) * 1000.0;
    }

    /* Pthreads */
    t0 = now_sec();
    matmul_pthreads(A, B, C_pt, n, nthreads);
    t1 = now_sec();
    pt_ms = (t1 - t0) * 1000.0;

    /* OpenMP */
    t0 = now_sec();
    matmul_openmp(A, B, C_omp, n);
    t1 = now_sec();
    omp_ms = (t1 - t0) * 1000.0;

    /* Verify */
    int ok = 1;
    if (!skip_single) {
        ok &= verify(C_single, C_pt, n, "pthreads");
        ok &= verify(C_single, C_omp, n, "openmp");
    } else {
        ok &= verify(C_pt, C_omp, n, "omp vs pt");
    }

    /* Print row */
    if (!skip_single) {
        printf("%-6d │ %10.1f ms  %7.1f │ %10.1f ms  %7.1f  %5.1fx │ %10.1f ms  %7.1f  %5.1fx │ %s\n",
               n,
               single_ms, gflops_op / (single_ms / 1e3),
               pt_ms, gflops_op / (pt_ms / 1e3), single_ms / pt_ms,
               omp_ms, gflops_op / (omp_ms / 1e3), single_ms / omp_ms,
               ok ? "PASS" : "FAIL");
    } else {
        printf("%-6d │ %10s   %7s │ %10.1f ms  %7.1f  %5s  │ %10.1f ms  %7.1f  %5s  │ %s\n",
               n,
               "(skip)", "",
               pt_ms, gflops_op / (pt_ms / 1e3), "-",
               omp_ms, gflops_op / (omp_ms / 1e3), "-",
               ok ? "PASS" : "FAIL");
    }

    free(A); free(B); free(C_single); free(C_pt); free(C_omp);
}

int main(void) {
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);

    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  CPU MATRIX MULTIPLY: Single-threaded vs Pthreads vs OpenMP                                    │\n");
    printf("│  Threads: %d                                                                                   │\n", nthreads);
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────────┘\n\n");

    printf("%-6s │ %22s │ %28s │ %28s │ %s\n",
           "N", "Single-threaded", "Pthreads", "OpenMP", "Check");
    printf("%-6s │ %12s  %8s │ %12s  %8s  %5s │ %12s  %8s  %5s │\n",
           "", "Time", "GFLOP/s", "Time", "GFLOP/s", "Spdup", "Time", "GFLOP/s", "Spdup");
    printf("───────┼────────────────────────┼──────────────────────────────┼──────────────────────────────┼──────\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    double single_timeout = 120.0;
    int skip_single = 0;

    for (int s = 0; s < nsizes; s++) {
        benchmark(sizes[s], nthreads, skip_single);
        fflush(stdout);

        /* Estimate if single-threaded would exceed timeout for next size */
        if (!skip_single && s < nsizes - 1) {
            /* O(N^3) scaling: next is ~8x slower if size doubles */
            /* Rough check: if this one took > timeout/8, skip next */
        }
    }

    printf("\n");
    return 0;
}
