#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

static float A[N * N], B[N * N], C[N * N];

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

int main(void) {
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_cpu(A, B, C, N);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("CPU matmul (%dx%d): %.3f ms\n", N, N, elapsed * 1000.0);
    printf("C[0][0] = %f, C[%d][%d] = %f\n", C[0], N - 1, N - 1, C[(N - 1) * N + (N - 1)]);

    return 0;
}
