#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define N 1024

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

void matmul_mt(const float *A, const float *B, float *C, int n, int nthreads) {
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    thread_arg_t *args = malloc(nthreads * sizeof(thread_arg_t));

    int rows_per = n / nthreads;
    int extra = n % nthreads;
    int row = 0;

    for (int t = 0; t < nthreads; t++) {
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].n = n;
        args[t].row_start = row;
        args[t].row_end = row + rows_per + (t < extra ? 1 : 0);
        row = args[t].row_end;
        pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
    }

    for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], NULL);

    free(threads);
    free(args);
}

int main(void) {
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);

    float *A = malloc(N * N * sizeof(float));
    float *B = malloc(N * N * sizeof(float));
    float *C = malloc(N * N * sizeof(float));

    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_mt(A, B, C, N, nthreads);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Pthreads matmul (%dx%d, %d threads): %.3f ms\n", N, N, nthreads, elapsed * 1000.0);
    printf("C[0][0] = %f, C[%d][%d] = %f\n", C[0], N - 1, N - 1, C[(N - 1) * N + (N - 1)]);

    free(A); free(B); free(C);
    return 0;
}
