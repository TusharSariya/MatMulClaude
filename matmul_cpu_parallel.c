#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 1024

void matmul_omp(const float *A, const float *B, float *C, int n) {
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

int main(void) {
    int nthreads = omp_get_max_threads();

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
    matmul_omp(A, B, C, N);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("OpenMP matmul (%dx%d, %d threads): %.3f ms\n", N, N, nthreads, elapsed * 1000.0);
    printf("C[0][0] = %f, C[%d][%d] = %f\n", C[0], N - 1, N - 1, C[(N - 1) * N + (N - 1)]);

    free(A); free(B); free(C);
    return 0;
}
