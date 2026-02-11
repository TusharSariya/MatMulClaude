# Matrix Multiplication: CPU vs GPU Benchmark Results

**Hardware:** Intel CPU (20 threads) · NVIDIA RTX 3060 Ti (8 GB VRAM) · 32 GB System RAM
**Software:** GCC -O2 · CUDA 12.4 · sm_86 · pthreads · OpenMP
**Date:** 2026-02-11

---

## Benchmark (CPU vs GPU Naive vs GPU Tiled)

| Size | Mem/Matrix | CPU Time | CPU GFLOP/s | GPU Naive | GPU Tiled | Tiled GFLOP/s | Speedup (Tiled) |
|-----:|-----------:|---------:|------------:|----------:|----------:|--------------:|----------------:|
| 256 | 0.25 MB | 7.3 ms | 4.6 | 0.336 ms | 0.038 ms | 879 | 190x |
| 512 | 1 MB | 60.1 ms | 4.5 | 0.610 ms | 0.218 ms | 1,233 | 276x |
| 1024 | 4 MB | 2,299 ms | 0.9 | 2.191 ms | 1.716 ms | 1,251 | 1,339x |
| 2048 | 16 MB | 19,157 ms | 0.9 | 18.664 ms | 14.066 ms | 1,221 | 1,362x |

All GPU results passed correctness verification against CPU (max error < 1e-3).

---

## Stress Test (Scaling to Limits)

| N | Mem/Matrix | CPU Time | CPU GFLOP/s | GPU Tiled | GPU GFLOP/s | Speedup |
|-----:|-----------:|---------:|------------:|----------:|------------:|--------:|
| 256 | 0.25 MB | 7.3 ms | 4.6 | 0.033 ms | 1,024 | 222x |
| 512 | 1 MB | 58.9 ms | 4.6 | 0.214 ms | 1,254 | 275x |
| 1024 | 4 MB | 2.42 s | 0.9 | 1.4 ms | 1,512 | 1,701x |
| 2048 | 16 MB | 19.45 s | 0.9 | 13.6 ms | 1,267 | 1,434x |
| 3072 | 36 MB | 109.74 s | 0.5 | 49.8 ms | 1,164 | 2,203x |
| 4096 | 64 MB | 361.99 s | 0.4 | 113.4 ms | 1,212 | 3,193x |
| 5120 | 100 MB | (skipped) | — | 215.6 ms | 1,245 | — |
| 6144 | 144 MB | (skipped) | — | 346.2 ms | 1,340 | — |
| 7168 | 196 MB | (skipped) | — | 563.9 ms | 1,306 | — |
| 8192 | 256 MB | (skipped) | — | 823.5 ms | 1,335 | — |
| 9216 | 324 MB | (skipped) | — | 1.61 s | 970 | — |
| 10240 | 400 MB | (skipped) | — | 2.01 s | 1,069 | — |
| 11264 | 484 MB | (skipped) | — | 2.15 s | 1,328 | — |
| 12288 | 576 MB | (skipped) | — | 2.80 s | 1,324 | — |
| 13312 | 676 MB | (skipped) | — | 3.59 s | 1,314 | — |
| 14336 | 784 MB | (skipped) | — | 4.49 s | 1,313 | — |
| 15360 | 900 MB | (skipped) | — | 5.54 s | 1,308 | — |
| 16384 | 1.00 GB | (skipped) | — | 6.72 s | 1,308 | — |

CPU was skipped after 4096x4096 (exceeded 2-minute timeout).

---

## CPU Implementations Comparison (20 threads)

Single-threaded vs pthreads (manual thread pool) vs OpenMP (`#pragma omp parallel for collapse(2)`).

| N | Single-threaded | GFLOP/s | Pthreads | GFLOP/s | Speedup | OpenMP | GFLOP/s | Speedup | Check |
|-----:|----------------:|--------:|---------:|--------:|--------:|-------:|--------:|--------:|:-----:|
| 256 | 17.4 ms | 1.9 | 1.4 ms | 23.8 | 12.3x | 2.6 ms | 12.8 | 6.6x | PASS |
| 512 | 58.6 ms | 4.6 | 10.6 ms | 25.4 | 5.5x | 10.6 ms | 25.3 | 5.5x | PASS |
| 1024 | 2,421 ms | 0.9 | 150.6 ms | 14.3 | 16.1x | 136.9 ms | 15.7 | 17.7x | PASS |
| 2048 | 18,770 ms | 0.9 | 2,488 ms | 6.9 | 7.5x | 2,458 ms | 7.0 | 7.6x | PASS |
| 4096 | 360,429 ms | 0.4 | 28,952 ms | 4.7 | 12.4x | 28,446 ms | 4.8 | 12.7x | PASS |

### CPU vs CPU vs GPU at 4096x4096

| Implementation | Time | GFLOP/s | vs Single-threaded |
|---|---:|---:|---:|
| CPU single-threaded | 360.4 s | 0.4 | 1x |
| CPU pthreads (20 threads) | 29.0 s | 4.7 | 12.4x |
| CPU OpenMP (20 threads) | 28.4 s | 4.8 | 12.7x |
| GPU tiled (RTX 3060 Ti) | 0.113 s | 1,212 | 3,193x |

---

## Breaking Points

### GPU

| Metric | Value |
|---|---|
| **Max matrix size** | **22,528 x 22,528** |
| VRAM used (3 matrices) | 5.67 GB of 8 GB |
| Next failed size | 23,040 x 23,040 (needed 5.93 GB, exceeded free VRAM) |
| Time at max size | ~17 s |
| Throughput at max size | ~1,320 GFLOP/s |
| **Bottleneck** | **VRAM** — 3 float32 matrices + CUDA context overhead must fit in 8 GB |

### CPU

| Metric | Value |
|---|---|
| **Last tested size** | **4,096 x 4,096** |
| Time at last size | 362 s (~6 minutes) |
| Throughput | 0.4 GFLOP/s at 4096, 4.6 GFLOP/s at 256 |
| Could allocate 22,528? | Yes (5.67 GB fits in 32 GB RAM) |
| Estimated time at 22,528 | ~60+ hours (extrapolated from O(N^3) scaling) |
| **Bottleneck** | **O(N^3) compute time** — not memory |

---

## Key Observations

1. **GPU is 1,000–3,200x faster** than single-threaded CPU, and the gap widens with matrix size.
2. **GPU throughput is stable** at ~1,300 GFLOP/s across all sizes (tiled shared-memory kernel, BLOCK_SIZE=16).
3. **CPU throughput degrades** from 4.6 GFLOP/s (256x256, fits in cache) to 0.4 GFLOP/s (4096x4096, cache-thrashing).
4. **Tiled kernel is ~30% faster** than the naive GPU kernel due to shared memory reducing global memory traffic.
5. **GPU wall is VRAM.** The RTX 3060 Ti can compute a 22,528x22,528 multiply in 17 seconds but cannot fit 23,040x23,040.
6. **CPU wall is time.** It has plenty of RAM (32 GB) but the O(N^3) algorithm makes large sizes impractical without multithreading or BLAS libraries.
7. **Multithreading helps but doesn't close the gap.** Pthreads and OpenMP deliver 5–17x speedup over single-threaded (on 20 cores), but the GPU is still **250x faster** than the best CPU result at 4096x4096.
8. **Pthreads vs OpenMP are nearly identical** in performance. OpenMP has a slight edge at larger sizes due to `collapse(2)` distributing work more evenly, while pthreads only splits by rows.
9. **CPU parallel scaling is sublinear.** 20 threads yield 7–17x speedup (not 20x) due to memory bandwidth saturation and cache contention at larger sizes.
