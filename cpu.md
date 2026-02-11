# CPU Matrix Multiplication: From Scalar to SIMD

A deep dive into each CPU implementation, what happens at the hardware level, and why each optimization matters. Written for the **i5-13600K** (Raptor Lake, 6 P-cores + 8 E-cores = 20 threads).

> **Note:** This document is AI-generated and intended for learning purposes.

---

## The Implementations

| # | Implementation | File | Key Idea |
|---|---|---|---|
| 1 | Scalar (single-threaded) | `matmul_cpu.c` | Baseline — 1 float per instruction |
| 2 | Pthreads (multithreaded) | `matmul_cpu_mt.c` | Split rows across threads manually |
| 3 | OpenMP (multithreaded) | `matmul_cpu_parallel.c` | One pragma to parallelize |
| 4 | AVX2+FMA (single-threaded) | `matmul_cpu_simd.c` | 8 floats per instruction |
| 5 | AVX2+FMA+OpenMP (both) | `matmul_cpu_simd.c` | SIMD × threads = max CPU performance |

### Results at 4096x4096

| Implementation | Time | GFLOP/s | vs Scalar |
|---|---:|---:|---:|
| Scalar | 323.0 s | 0.4 | 1x |
| Pthreads (20 threads) | 29.0 s | 4.7 | 11.1x |
| OpenMP (20 threads) | 28.4 s | 4.8 | 11.4x |
| AVX2+FMA (1 thread) | 10.2 s | 13.5 | 31.8x |
| AVX2+FMA+OpenMP (20 threads) | 1.26 s | 108.9 | 255.9x |
| GPU tiled (for reference) | 0.113 s | 1,212 | 2,858x |

---

## 1. Scalar: The Baseline

```c
void matmul_scalar(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}
```

### What the CPU does per inner iteration

Each `sum += A[i*n+k] * B[k*n+j]` becomes roughly:

1. **Load** `A[i*n+k]` into a 32-bit register (~4 cycles from L1)
2. **Load** `B[k*n+j]` into another register (~4 cycles from L1... if it's there)
3. **Multiply + add** with FMA: `vfmadd231ss` (~4 cycle latency)
4. Advance `k`, loop

One float per instruction. The CPU has 256-bit wide registers that could hold 8 floats, but scalar code only uses the bottom 32 bits.

```
YMM0 register (256 bits):
[  unused  |  unused  |  unused  |  unused  |  unused  |  unused  |  unused  |  sum  ]
  32 bits     32 bits    32 bits    32 bits    32 bits    32 bits    32 bits   32 bits

Only 12.5% of the register width is doing useful work.
```

### Why it's so slow: the B access pattern

Matrix B is stored row-major. The inner loop walks **down a column** of B:

```
B[0*n + j], B[1*n + j], B[2*n + j], ...

For n=1024, each element is 4096 bytes apart.
CPU cache lines are 64 bytes (16 floats).
```

**Access pattern for A (row — good):**
```
A: [a0][a1][a2][a3][a4][a5][a6][a7][a8]...  ← consecutive, 1 cache miss per 16 floats
    └──────── one cache line ──────────┘
```

**Access pattern for B (column — terrible):**
```
B: [b0]                                      ← cache line 0
         ... 4096 bytes gap ...
   [b1]                                      ← cache line 64
         ... 4096 bytes gap ...
   [b2]                                      ← cache line 128
         ...
Every single access misses L1, because stride (4096 bytes) >> cache line (64 bytes)
```

Your i5-13600K has a 2 MB L2 per P-core. A 1024×1024 float matrix is 4 MB — it doesn't even fit in L2. Column access means constant L2 misses falling through to the shared 24 MB L3, adding ~40-50 cycles per access instead of ~4 cycles for an L1 hit.

**This is the #1 reason scalar matmul is slow: not compute-bound, but memory-bound from cache misses on B.**

---

## 2. Pthreads: Manual Multithreading

```c
typedef struct {
    const float *A, *B;
    float *C;
    int n, row_start, row_end;
} thread_arg_t;

void *matmul_worker(void *arg) {
    thread_arg_t *t = (thread_arg_t *)arg;
    for (int i = t->row_start; i < t->row_end; i++)
        for (int j = 0; j < t->n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < t->n; k++)
                sum += t->A[i * t->n + k] * t->B[k * t->n + j];
            t->C[i * t->n + j] = sum;
        }
    return NULL;
}
```

Each thread gets a chunk of rows:
```
Thread 0:  rows 0-51     (52 rows for n=1024, 20 threads)
Thread 1:  rows 52-103
...
Thread 19: rows 988-1023

Each thread runs the exact same scalar inner loop.
The speedup comes from 20 cores working simultaneously.
```

### What happens under the hood

1. `pthread_create()` → Linux creates a kernel thread (a `task_struct`)
2. The scheduler assigns it to a CPU core (P-core or E-core)
3. Each thread runs independently — no synchronization needed until the end
4. `pthread_join()` → main thread waits for all workers to finish

### Why it's not 20x faster

With 20 threads you'd expect 20x speedup, but we only see ~11x. Why?

**Memory bandwidth saturation:** All 20 threads share the same DDR5 memory bus (~70 GB/s measured). Column access of B causes cache misses on every thread. 20 threads all hammering DRAM simultaneously saturate the bus.

**Heterogeneous cores:** The i5-13600K has 6 P-cores (fast, 5.1 GHz) and 8 E-cores (slower, 3.9 GHz, narrower execution). The OS treats them as 20 equal threads, but E-cores are ~3-5x slower per core for this workload. The total finishes at the speed of the slowest thread.

**Cache contention:** Multiple threads accessing overlapping regions of B evict each other's cache lines from the shared L3 (24 MB).

---

## 3. OpenMP: Same Thing, One Line

```c
void matmul_openmp(const float *A, const float *B, float *C, int n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}
```

One pragma replaces ~40 lines of pthread boilerplate. But what does it actually do?

### The Fork-Join Model

```
Main thread
    │
    ├── hits #pragma omp parallel ──→ FORK: spawns 19 worker threads (reused from pool)
    │                                       │
    │   Thread 0 ─── iterations 0 to 52,428       (of the flattened i×j space)
    │   Thread 1 ─── iterations 52,429 to 104,857
    │   ...
    │   Thread 19 ── iterations 996,147 to 1,048,575
    │                                       │
    │                                       ▼
    ├── implicit barrier ───────────→ JOIN: all threads sync, workers go back to sleep
    │
    ▼ continues
```

### What `collapse(2)` does

Without collapse, only the outer `i` loop is split across threads (1024 iterations / 20 threads = ~51 rows each).

With `collapse(2)`, the `i` and `j` loops are **flattened** into a single iteration space:

```
Without collapse:         With collapse(2):
  Thread 0: i=0..51         Thread 0: ij=0..52428
  Thread 1: i=52..103         → i=0..51, j=0..all of them
  ...                         → plus i=51, j=0..some
                              Thread 1: ij=52429..104857
                              ...

Total iterations: 1024      Total iterations: 1024 × 1024 = 1,048,576
Granularity: coarse (rows)  Granularity: fine (individual elements)
```

This matters when N is small relative to the thread count. For N=64 with 20 threads, without collapse each thread gets ~3 rows — poor load balance. With collapse, each thread gets ~200 (i,j) pairs.

### What `schedule(static)` does

Divides iterations into equal contiguous chunks at compile time. No runtime overhead — no atomic counters, no work queues. Each thread knows its range before the loop starts.

Alternative: `schedule(dynamic, chunk)` — threads grab work from a shared queue using atomics. Useful when iteration cost varies, but adds ~100-200 ns overhead per chunk. For matmul (uniform cost per iteration), static is optimal.

### Pthreads vs OpenMP: Why performance is identical

OpenMP is not a different threading technology. It's a **compiler directive** that generates pthread code:

```
#pragma omp parallel for    →    compiler generates:
                                   pthread_create() × 19
                                   index range calculation
                                   barrier at the end
                                   pthread_join() or futex wait
```

The inner loop machine code is **identical**. The CPU doesn't know what API created the thread. The only difference is programmer convenience.

| | Pthreads | OpenMP |
|---|---|---|
| Lines of code | ~40 | ~2 |
| Thread pool | Manual | Built-in |
| Work division | Manual index math | `schedule()` clause |
| Performance | Same | Same |

---

## 4. AVX2 + FMA: SIMD (The Big Jump)

This is where the real speedup comes from. SIMD (Single Instruction, Multiple Data) processes 8 floats simultaneously.

### AVX2 Registers

Your i5-13600K has 16 YMM registers, each 256 bits wide:

```
YMM0:  [ float7 | float6 | float5 | float4 | float3 | float2 | float1 | float0 ]
        ←─────────────────── 256 bits (32 bytes) ──────────────────────────────→

Each "lane" is 32 bits (one float). One instruction operates on all 8 lanes at once.
```

### The Two Key Intrinsics

**`_mm256_loadu_ps(ptr)`** — Load 8 consecutive floats:

```
Memory:  [1.0] [2.0] [3.0] [4.0] [5.0] [6.0] [7.0] [8.0]
          ptr   ptr+4 ptr+8 ...

                        │ _mm256_loadu_ps(ptr)
                        ▼

YMM reg: [ 8.0 | 7.0 | 6.0 | 5.0 | 4.0 | 3.0 | 2.0 | 1.0 ]

One instruction, one cycle throughput. Compiles to: VMOVUPS ymm, [mem]
```

**`_mm256_fmadd_ps(a, b, c)`** — Fused Multiply-Add on all 8 lanes:

```
a:    [ a7 | a6 | a5 | a4 | a3 | a2 | a1 | a0 ]
b:    [ b7 | b6 | b5 | b4 | b3 | b2 | b1 | b0 ]
c:    [ c7 | c6 | c5 | c4 | c3 | c2 | c1 | c0 ]

                   │ _mm256_fmadd_ps(a, b, c)
                   ▼  computes: a * b + c  for each lane

result: [ a7*b7+c7 | a6*b6+c6 | a5*b5+c5 | a4*b4+c4 | a3*b3+c3 | a2*b2+c2 | a1*b1+c1 | a0*b0+c0 ]

One instruction. 4-cycle latency. But the P-core has TWO FMA units, so throughput = 2 per cycle.
That's 8 floats × 2 operations (mul+add) × 2 per cycle = 32 FP ops per cycle per core.
```

### Why We Transpose B

The SIMD inner loop needs 8 **consecutive** floats from both A and B. A's row access is already consecutive. But B's column access is scattered:

```
WITHOUT TRANSPOSE — B column access:
  Need: B[0*n+j], B[1*n+j], B[2*n+j], ..., B[7*n+j]
  These are 4096 bytes apart (for n=1024). Can't use _mm256_loadu_ps.
  Would need _mm256_i32gather_ps (gather instruction): ~12-20 cycles, defeats the purpose.

WITH TRANSPOSE — Bt row access:
  Bt[j*n+0], Bt[j*n+1], ..., Bt[j*n+7]
  These are consecutive in memory. _mm256_loadu_ps loads all 8 in one instruction.
```

The transpose costs O(N²) — negligible compared to the O(N³) matmul. But it turns every inner loop iteration from a cache miss into a cache hit.

### Walking Through the SIMD Kernel

```c
void matmul_simd(const float *A, const float *B, float *C, int n) {
    // Step 1: Transpose B → O(N²), done once
    float *Bt = aligned_alloc(32, n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Bt[j * n + i] = B[i * n + j];
```

For each output element C[i][j]:

```c
    for (int i = 0; i < n; i++) {
        const float *a_row = &A[i * n];       // row i of A
        for (int j = 0; j < n; j++) {
            const float *bt_row = &Bt[j * n]; // row j of Bt (= column j of B)

            // 8 partial sums, one per SIMD lane
            __m256 vsum = _mm256_setzero_ps(); // [0,0,0,0,0,0,0,0]

            for (int k = 0; k + 7 < n; k += 8) {
                __m256 va = _mm256_loadu_ps(&a_row[k]);   // 8 floats from A
                __m256 vb = _mm256_loadu_ps(&bt_row[k]);  // 8 floats from Bt
                vsum = _mm256_fmadd_ps(va, vb, vsum);     // vsum += va * vb
            }
```

After the loop, `vsum` contains 8 partial sums that need to be added together:

```c
            // vsum = [s7, s6, s5, s4, s3, s2, s1, s0]
            // Need: s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7

            __m128 hi = _mm256_extractf128_ps(vsum, 1); // [s7, s6, s5, s4]
            __m128 lo = _mm256_castps256_ps128(vsum);    // [s3, s2, s1, s0]
            __m128 sum128 = _mm_add_ps(lo, hi);
            // sum128 = [s3+s7, s2+s6, s1+s5, s0+s4]

            sum128 = _mm_hadd_ps(sum128, sum128);
            // sum128 = [s2+s6+s3+s7, s0+s4+s1+s5, s2+s6+s3+s7, s0+s4+s1+s5]

            sum128 = _mm_hadd_ps(sum128, sum128);
            // sum128 = [total, total, total, total]

            float sum = _mm_cvtss_f32(sum128); // extract the scalar
```

**The horizontal sum is slow (~6-10 cycles) but only runs once per output element.** The FMA loop runs N/8 = 128 iterations for N=1024. So the horizontal sum is ~1% of the work.

### Why the Speedup Exceeds 8x

AVX2 processes 8 floats per instruction — you'd expect 8x speedup. But we measured **13-32x**. The extra comes from:

1. **Transposing B eliminates cache misses.** The scalar kernel spends most of its time waiting for B column data. The SIMD kernel with transposed B gets L1 hits on both A and Bt. This alone is worth 3-4x.

2. **FMA does two operations in one instruction.** `a * b + c` replaces separate multiply and add. Twice the work per instruction.

3. **Better instruction pipelining.** The compiler sees a tight loop with known trip count and can pipeline loads from future iterations while the current FMA executes.

So the actual breakdown is roughly: **8x (wider registers) × 1.5x (FMA) × 2-3x (cache friendliness) ≈ 24-36x** theoretical, with overhead reducing it to the observed 13-32x.

---

## 5. AVX2 + FMA + OpenMP: Maximum CPU Performance

```c
void matmul_simd_omp(const float *A, const float *B, float *C, int n) {
    float *Bt = aligned_alloc(32, n * n * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)          // transpose is also parallelized
        for (int j = 0; j < n; j++)
            Bt[j * n + i] = B[i * n + j];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {         // each thread gets ~51 rows
        // ... same SIMD inner loop as single-threaded version ...
    }
}
```

This combines both optimizations:
- **Each thread** runs the AVX2+FMA vectorized inner loop (8 floats/instruction)
- **20 threads** work on different rows of the output simultaneously

### Why it's multiplicative

SIMD and threading are **orthogonal** — they operate on different dimensions of parallelism:

```
                        Scalar          SIMD (AVX2+FMA)
                     ┌──────────┐     ┌──────────┐
1 thread             │  1 float │     │ 8 floats │  ← 8-32x faster per thread
                     │ per inst │     │ per inst │
                     └──────────┘     └──────────┘

                     ┌──────────┐     ┌──────────┐
20 threads           │ 20 × 1  │     │ 20 × 8   │  ← 20 threads × 8-32x each
                     │ = 20    │     │ = 160    │
                     └──────────┘     └──────────┘

Scalar → OpenMP:        ~11x   (thread-level parallelism)
Scalar → SIMD:          ~32x   (data-level parallelism)
Scalar → SIMD+OpenMP:   ~256x  (both combined)
```

### Why it's sublinear (256x, not 640x)

If SIMD gives 32x and 20 threads give 20x, shouldn't the combo give 640x?

**Memory bandwidth is the wall.** Your i5-13600K has ~70 GB/s DDR5 bandwidth (measured). With 20 threads all loading data through SIMD, the total demand far exceeds what the memory bus can deliver.

```
Single-threaded SIMD:
  Each FMA processes 8 floats (32 bytes) per ~0.5 cycles at 5.1 GHz
  Peak demand: ~326 GB/s per core  ← far exceeds 70 GB/s
  But most hits go to L1/L2 cache (thanks to transpose), so actual DRAM demand is manageable

20-threaded SIMD:
  20 threads × cache miss traffic = aggregate demand saturates DDR5
  Threads stall waiting for memory → diminishing returns
```

Additionally, **E-cores are slower for AVX2.** The 8 E-cores (Gracemont) have 128-bit native SIMD units. A 256-bit AVX2 instruction gets split into two 128-bit micro-ops. Combined with their lower clock (3.9 vs 5.1 GHz), each E-core delivers roughly **1/5th** the AVX2 throughput of a P-core.

```
Effective compute:
  6 P-cores × 2 hyperthreads × 32 FP ops/cycle × 5.1 GHz = 1,958 GFLOPS (but HT shares FMA units)
  Realistic P-core contribution: ~6 cores × 32 × 5.1 GHz = 979 GFLOPS
  8 E-cores × ~6.4 FP ops/cycle × 3.9 GHz = 200 GFLOPS
  Total theoretical: ~1,179 GFLOPS

  Measured: 109 GFLOPS at 4096×4096 → about 9% of theoretical peak
  The gap is entirely due to memory bandwidth limiting how fast data reaches the cores.
```

---

## 6. Why the GPU Still Wins

Even with every CPU optimization stacked, the GPU is still **11x faster**. Here's why:

### Raw numbers

| | i5-13600K (CPU) | RTX 3060 Ti (GPU) |
|---|---:|---:|
| FP32 compute cores | 6 P-cores + 8 E-cores | 4,864 CUDA cores |
| Peak FP32 TFLOPS | ~1.2 (realistic) | ~16.2 |
| Memory bandwidth | ~70 GB/s (DDR5) | 448 GB/s (GDDR6) |
| Memory latency | ~50-80 ns | ~300-500 ns |
| Design goal | Make 1 thread fast | Make 1000s of threads busy |

### The architectural difference

**CPU: latency-optimized.** A P-core devotes enormous silicon area to:
- Out-of-order execution (reorder buffer: 512 entries)
- Branch prediction (multiple predictors, speculative execution)
- Large caches (2 MB L2 per core, 24 MB shared L3)
- Deep pipeline (~20 stages)

All to make **one instruction stream** run as fast as possible. Great for general-purpose code, but wasteful for matmul where every iteration does the same thing.

**GPU: throughput-optimized.** Each CUDA core is tiny and simple:
- In-order execution
- No branch prediction
- Tiny per-core cache
- Relies on thousands of threads to hide latency

When one warp (32 threads) stalls on a memory load, the SM instantly switches to another warp — no cost. With 48 warps per SM, there's always something to do.

### Memory bandwidth is the ultimate differentiator

```
CPU memory bus:   70 GB/s (DDR5-5600, dual channel, 128-bit)
GPU memory bus:  448 GB/s (GDDR6, 256-bit, 14 Gbps)

GPU has 6.4x more memory bandwidth.
```

For a memory-bound workload like matmul (even with tiling), the side with more bandwidth wins. The GPU also has 128 KB of shared memory per SM — an explicitly-managed cache that lets the programmer guarantee data reuse. The CPU relies on hardware caching that can't be directly controlled.

### Where the CPU can actually win

- **Small matrices (N < 64):** GPU kernel launch overhead (~5-20 μs) exceeds the compute time
- **Data already in CPU RAM:** PCIe transfer to GPU (16 GB/s) can cost more than just computing on CPU
- **Irregular/branchy code:** GPUs handle divergent branches poorly (both paths of an if/else execute, with masking)
- **Low batch sizes in ML inference:** CPU is better when you can't fill the GPU with enough parallel work

---

## Summary: Optimization Stack

```
                     4096×4096 matmul
                     ────────────────

  Scalar CPU              0.4 GFLOP/s     ████
  + OpenMP (20 thr)       4.8 GFLOP/s     ██████████████████████████
  + AVX2+FMA (1 thr)     13.5 GFLOP/s     ██████████████████████████████████████████████████████████████████████████
  + AVX2+FMA+OMP        108.9 GFLOP/s     █████████████████████████████████...████████████████████████████████████████████████ (256x)
  GPU tiled            1,212   GFLOP/s     █████████████████████████████████████████████████████████████████████████████████████████████████████████...█████████████ (2,858x)

Each layer is multiplicative:
  SIMD:     8-32x  (wider registers + FMA + cache-friendly transpose)
  Threads:  8-12x  (20 cores, sublinear due to bandwidth)
  Combined: ~256x  (limited by DDR5 bandwidth at ~70 GB/s)
  GPU:      ~11x more on top  (448 GB/s bandwidth + 4,864 cores)
```

The key lesson: **a naive scalar loop uses less than 0.04% of your CPU's actual capability.** SIMD and multithreading aren't "nice to have" optimizations — they're the difference between a 5-minute computation and a 1-second computation.
