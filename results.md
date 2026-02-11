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

## CPU SIMD: AVX2 + FMA (i5-13600K)

AVX2 processes 8 floats per instruction (256-bit). FMA (fused multiply-add) computes `a * b + c` in a single instruction.
The SIMD kernel transposes B for coalesced access, then uses `_mm256_fmadd_ps` in the inner loop.

| N | Scalar | GFLOP/s | AVX2+FMA (1 thread) | GFLOP/s | Speedup | AVX2+FMA+OpenMP (20 threads) | GFLOP/s | Speedup | Check |
|-----:|-------:|--------:|--------------------:|--------:|--------:|-----------------------------:|--------:|--------:|:-----:|
| 256 | 19.5 ms | 1.7 | 1.1 ms | 29.2 | 16.9x | 0.5 ms | 72.7 | 42.2x | PASS |
| 512 | 99.0 ms | 2.7 | 7.9 ms | 34.1 | 12.6x | 1.6 ms | 166.4 | 61.4x | PASS |
| 1024 | 2,405 ms | 0.9 | 79.3 ms | 27.1 | 30.3x | 11.0 ms | 195.2 | 218.6x | PASS |
| 2048 | 20,525 ms | 0.8 | 721.0 ms | 23.8 | 28.5x | 94.8 ms | 181.3 | 216.6x | PASS |
| 4096 | 323,007 ms | 0.4 | 10,162 ms | 13.5 | 31.8x | 1,262 ms | 108.9 | 255.9x | PASS |

### Full Comparison at 4096x4096 (every implementation)

| Implementation | Time | GFLOP/s | vs Scalar | vs GPU |
|---|---:|---:|---:|---:|
| CPU scalar (1 thread) | 323.0 s | 0.4 | 1x | 2,858x slower |
| CPU OpenMP (20 threads) | 28.4 s | 4.8 | 11.4x | 251x slower |
| CPU AVX2+FMA (1 thread) | 10.2 s | 13.5 | 31.8x | 90x slower |
| CPU AVX2+FMA+OpenMP (20 threads) | 1.26 s | 108.9 | 255.9x | **11.2x slower** |
| GPU tiled 16x16 (RTX 3060 Ti) | 0.113 s | 1,212 | 2,858x | 1x |

SIMD+multithreading brings the CPU within **11x of the GPU** — compared to 3,193x without either optimization.

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

## GPU Block Size Sweep

RTX 3060 Ti: 38 SMs, warp size 32, max 1024 threads/block, 48 KB shared mem/block.
Tested block sizes 4x4 (16 threads) through 32x32 (1024 threads) on both naive and tiled kernels.

### N = 512

| Block | Threads/Block | Grid Blocks | Naive (ms) | Naive GFLOP/s | Naive Eff% | Tiled (ms) | Tiled GFLOP/s | Tiled Eff% |
|------:|--------------:|------------:|-----------:|--------------:|-----------:|-----------:|--------------:|-----------:|
| 4x4 | 16 | 16,384 | 0.669 | 401 | 28.7% | 0.885 | 303 | 21.7% |
| 8x8 | 64 | 4,096 | 0.340 | 790 | 56.4% | 0.276 | 971 | 69.4% |
| **16x16** | **256** | **1,024** | **0.274** | **978** | **69.9%** | **0.212** | **1,266** | **90.5%** |
| 32x32 | 1,024 | 256 | 0.279 | 961 | 68.6% | 0.223 | 1,202 | 85.9% |

### N = 1024

| Block | Threads/Block | Grid Blocks | Naive (ms) | Naive GFLOP/s | Naive Eff% | Tiled (ms) | Tiled GFLOP/s | Tiled Eff% |
|------:|--------------:|------------:|-----------:|--------------:|-----------:|-----------:|--------------:|-----------:|
| 4x4 | 16 | 65,536 | 5.625 | 382 | 27.3% | 10.636 | 202 | 14.4% |
| 8x8 | 64 | 16,384 | 2.647 | 811 | 57.9% | 2.107 | 1,019 | 72.8% |
| **16x16** | **256** | **4,096** | **2.146** | **1,001** | **71.5%** | **1.659** | **1,295** | **92.5%** |
| 32x32 | 1,024 | 1,024 | 2.234 | 961 | 68.7% | 1.808 | 1,188 | 84.9% |

### N = 2048

| Block | Threads/Block | Grid Blocks | Naive (ms) | Naive GFLOP/s | Naive Eff% | Tiled (ms) | Tiled GFLOP/s | Tiled Eff% |
|------:|--------------:|------------:|-----------:|--------------:|-----------:|-----------:|--------------:|-----------:|
| 4x4 | 16 | 262,144 | 45.396 | 378 | 27.0% | 87.116 | 197 | 14.1% |
| 8x8 | 64 | 65,536 | 22.440 | 766 | 54.7% | 18.063 | 951 | 67.9% |
| **16x16** | **256** | **16,384** | **16.197** | **1,061** | **75.8%** | **12.971** | **1,325** | **94.6%** |
| 32x32 | 1,024 | 4,096 | 17.100 | 1,005 | 71.8% | 15.136 | 1,135 | 81.1% |

### N = 4096

| Block | Threads/Block | Grid Blocks | Naive (ms) | Naive GFLOP/s | Naive Eff% | Tiled (ms) | Tiled GFLOP/s | Tiled Eff% |
|------:|--------------:|------------:|-----------:|--------------:|-----------:|-----------:|--------------:|-----------:|
| 4x4 | 16 | 1,048,576 | 460.746 | 298 | 21.3% | 776.708 | 177 | 12.6% |
| 8x8 | 64 | 262,144 | 190.939 | 720 | 51.4% | 161.697 | 850 | 60.7% |
| **16x16** | **256** | **65,536** | **140.122** | **981** | **70.1%** | **105.679** | **1,301** | **92.9%** |
| 32x32 | 1,024 | 16,384 | 142.051 | 968 | 69.1% | 118.244 | 1,162 | 83.0% |

### Block Size Sweep Summary

- **16x16 wins across all matrix sizes** for both naive and tiled kernels.
- **Tiled 16x16 reaches 90–95% efficiency** of peak FP32 throughput (~1,300 of ~1,400 GFLOP/s).
- **4x4 is terrible** — only 16 threads per block means most of each warp (32 threads) is idle, and the massive grid count creates scheduling overhead. The tiled kernel is actually *slower* than naive at 4x4 because the shared memory tile is so small that the sync overhead dominates.
- **32x32 is slightly worse than 16x16** despite full warps. At 1024 threads/block, register pressure limits occupancy (fewer blocks can run concurrently per SM).
- **8x8 is decent** — good enough for simple kernels but leaves performance on the table vs 16x16.
- **Naive kernel** peaks at ~1,000 GFLOP/s regardless of block size; it's memory-bound. **Tiled kernel** breaks through by reusing data in shared memory.

---

## Kernel Performance Analysis

Deep dive into what makes the tiled kernel faster and what limits both kernels, using compiler output (`nvcc --ptxas-options=-v`) and code-level analysis of our matmul kernels.

### Compiler Output (ptxas)

| Kernel | Registers/Thread | Spill Stores | Spill Loads | Shared Mem | Constant Mem |
|---|---:|---:|---:|---:|---:|
| Naive | 40 | 0 bytes | 0 bytes | 0 bytes | 380 bytes |
| Tiled 4x4 | 40 | 0 bytes | 0 bytes | 128 bytes | 380 bytes |
| Tiled 8x8 | 40 | 0 bytes | 0 bytes | 512 bytes | 380 bytes |
| Tiled 16x16 | 38 | 0 bytes | 0 bytes | 2,048 bytes | 380 bytes |
| Tiled 32x32 | 38 | 0 bytes | 0 bytes | 8,192 bytes | 380 bytes |

---

### 1. Memory Access Patterns (Coalesced Access)

When a warp of 32 threads reads global memory, the hardware combines their requests into cache-line-sized transactions (128 bytes). Consecutive threads reading consecutive addresses = 1 transaction. Scattered reads = up to 32 transactions.

**Naive kernel — loading A:**
```c
sum += A[row * n + k] * B[k * n + col];
//      ^^^^^^^^^^^^^^^
// row is constant within a warp (threads in a warp differ by col, not row)
// k is the loop variable, same for all threads
// So all 32 threads read A[row * n + k] — SAME address → broadcast, 1 transaction ✓
```

**Naive kernel — loading B:**
```c
sum += A[row * n + k] * B[k * n + col];
//                       ^^^^^^^^^^^^^
// k is same for all threads, col varies by threadIdx.x
// Thread 0 reads B[k*n + col], thread 1 reads B[k*n + col+1], ...
// Consecutive threads read consecutive addresses → COALESCED, 1 transaction ✓
```

**Naive kernel — the problem isn't coalescing, it's volume.** Each thread does N iterations, each reading one float from A and one from B. For a 1024×1024 matrix, that's 1024 global loads per thread, and there's no reuse — every value is fetched from DRAM (or L2 at best) every time.

**Tiled kernel — loading A into shared memory:**
```c
sA[threadIdx.y][threadIdx.x] = A[row * n + aCol];
// threadIdx.x varies across the warp → aCol varies consecutively
// Consecutive threads load consecutive addresses → COALESCED ✓
```

**Tiled kernel — loading B into shared memory:**
```c
sB[threadIdx.y][threadIdx.x] = B[bRow * n + col];
// col varies by threadIdx.x → consecutive → COALESCED ✓
```

**Tiled kernel — reading from shared memory (the inner loop):**
```c
sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
// No global memory access at all — reads from on-chip shared memory (~20 cycles vs ~400)
```

**Impact:** Both kernels have coalesced global access patterns. The difference is that the naive kernel reads from global memory N times per element, while the tiled kernel reads from global memory N/BLOCK_SIZE times per element and does the rest from shared memory. At BLOCK_SIZE=16, that's a **16x reduction in global memory traffic**.

| Kernel | Global loads per output element | Where inner loop reads from |
|---|---:|---|
| Naive | 2N (N from A + N from B) | Global memory (~400 cycles/access) |
| Tiled 16x16 | 2N/16 = N/8 | Shared memory (~20 cycles/access) |

---

### 2. Shared Memory Bank Conflicts

Shared memory has 32 banks, each 4 bytes wide. If multiple threads in a warp access different addresses in the same bank, the accesses are serialized.

**Loading phase — writing to shared memory:**
```c
sA[threadIdx.y][threadIdx.x] = ...;
// sA is float[16][16], row-major. threadIdx.x varies across the warp.
// sA[y][0] is at byte offset (y*16 + 0)*4, bank = (y*16 + 0) % 32
// sA[y][1] is at byte offset (y*16 + 1)*4, bank = (y*16 + 1) % 32
// Consecutive threadIdx.x → consecutive banks → NO CONFLICT ✓
```

**Compute phase — reading sA:**
```c
sA[threadIdx.y][k]
// All threads in the warp have the same threadIdx.y (same row within the block)
// and the same k (loop variable) → ALL read the SAME address → BROADCAST ✓
```

**Compute phase — reading sB:**
```c
sB[k][threadIdx.x]
// k is the same for all threads, threadIdx.x varies
// sB[k][0] → bank (k*16 + 0) % 32
// sB[k][1] → bank (k*16 + 1) % 32
// Consecutive threadIdx.x → consecutive banks → NO CONFLICT ✓
```

**Verdict: No bank conflicts in our 16×16 tiled kernel.** The access patterns are clean because the tile width (16) doesn't create stride-based collisions with the 32-bank layout. A 32×32 tile would also be conflict-free for the same reason (stride of 32 maps each column to a unique bank when accessing `sB[k][threadIdx.x]` since the full warp covers all 32 banks).

Note: A **4×4 tile is problematic** for a different reason — only 16 threads per block means the warp is half-empty, and the tiny tile (4 loads of 4 elements = 16 loads total) doesn't amortize the `__syncthreads()` cost. Two syncs per tile × 256 tiles (for N=1024) = 512 barriers, each costing ~20 cycles.

---

### 3. Instruction-Level Parallelism (ILP)

The GPU hides memory latency by switching between warps. When one warp is waiting on a memory load (~400 cycles), the warp scheduler issues instructions from another warp. But within a single warp, **instruction-level parallelism** — having multiple independent operations in flight — also matters.

**Naive kernel — limited ILP:**
```c
for (int k = 0; k < n; k++) {
    sum += A[row * n + k] * B[k * n + col];  // load A, load B, multiply, add to sum
}
// Each iteration depends on the previous (sum += ...) → serial dependency chain
// The loads of A and B CAN overlap (independent), but the multiply must wait for both
// The add must wait for the multiply → pipeline depth of ~4 instructions per iteration
// Only 2 loads can be in flight simultaneously (A and B for the current iteration)
```

**Tiled kernel — better ILP in the compute phase:**
```c
// Load phase: two independent loads can be pipelined
sA[threadIdx.y][threadIdx.x] = A[...];  // load 1
sB[threadIdx.y][threadIdx.x] = B[...];  // load 2 (independent, can overlap)
__syncthreads();

// Compute phase:
for (int k = 0; k < BLOCK_SIZE; k++) {
    sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
}
// Same dependency chain as naive (sum += ...), BUT:
// - Shared memory loads are ~20 cycles instead of ~400
// - The compiler can unroll this loop (BLOCK_SIZE=16 is a compile-time constant)
// - With unrolling, multiple iterations' loads can be issued ahead of time
// - The compiler sees 16 iterations and can pipeline reads from sA and sB
```

**Occupancy and latency hiding:**

The other dimension of ILP is **occupancy** — how many warps are available to hide latency.

| Config | Regs/Thread | Max Threads/SM | Warps/SM | Occupancy |
|---|---:|---:|---:|---:|
| Naive (40 regs, 256 threads/block) | 40 | 1,536* | 48 | 100% |
| Tiled 16x16 (38 regs, 256 threads/block) | 38 | 1,536* | 48 | 100% |
| Tiled 32x32 (38 regs, 1024 threads/block) | 38 | 1,024** | 32 | 66% |

\* 65,536 registers / 40 regs = 1,638 threads → capped at 1,536 (SM limit) → 6 blocks of 256

\** 1,024 threads/block → only 1 block fits before hitting the thread limit → 1,024 threads

At 100% occupancy, the SM has 48 warps to choose from. When a warp stalls on a global memory load (naive kernel), the scheduler can switch to any of the other 47 warps. This is why the naive kernel still achieves ~1,000 GFLOP/s despite no shared memory — it has enough warps to hide most of the latency.

The tiled kernel needs **less** latency hiding because shared memory loads are 20x faster, so even at 66% occupancy (32x32 config), it still performs well.

---

### 4. Register Spilling

When a kernel uses more registers than available, the compiler spills values to **local memory** — which is physically in GDDR6, accessed through the cache hierarchy. Spilling is devastating: a register access is ~1 cycle, a spill to L1 is ~30 cycles, and an L1 miss goes all the way to DRAM at ~400 cycles.

**Our kernels: zero spills across the board.**

```
Naive:       40 registers, 0 bytes spill stores, 0 bytes spill loads ✓
Tiled 4x4:  40 registers, 0 bytes spill stores, 0 bytes spill loads ✓
Tiled 8x8:  40 registers, 0 bytes spill stores, 0 bytes spill loads ✓
Tiled 16x16: 38 registers, 0 bytes spill stores, 0 bytes spill loads ✓
Tiled 32x32: 38 registers, 0 bytes spill stores, 0 bytes spill loads ✓
```

**Why no spills?** Our kernels are simple — the inner loop only needs a few variables: `sum` (accumulator), `row`, `col`, `k`, the loaded values from A/B, and a few address calculations. That fits comfortably in 38–40 registers, well under the 255 register-per-thread architectural limit.

**When spilling becomes a problem:** more complex kernels with many local variables, large unrolled loops, or deep function call stacks can exceed the register budget. The compiler flag `--ptxas-options=-v` (which produced the output above) is the first thing to check when a kernel is unexpectedly slow.

**The register-occupancy tradeoff:**

Even though our kernels don't spill, registers still limit performance indirectly:

```
Registers per SM:    65,536
Naive uses:          40 regs/thread

Max threads = 65,536 / 40 = 1,638 → capped at 1,536 by SM thread limit
If a kernel used 64 regs/thread: 65,536 / 64 = 1,024 threads → 66% occupancy
If a kernel used 128 regs/thread: 65,536 / 128 = 512 threads → 33% occupancy
```

More registers per thread → fewer threads per SM → fewer warps to hide latency → lower throughput. This is the fundamental occupancy tradeoff, and it's why the compiler sometimes uses fewer registers than optimal (to keep occupancy high) even at the cost of slightly more instructions.

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
10. **Both kernels have clean memory access patterns** — fully coalesced global loads and zero shared memory bank conflicts. The tiled kernel's advantage is purely from reducing global memory traffic by 16x (BLOCK_SIZE), not from fixing bad access patterns.
11. **Zero register spills across all kernel variants.** At 38–40 registers per thread, both kernels fit comfortably and achieve 100% occupancy at 16x16 block size (6 blocks × 256 threads = 1,536 threads per SM).
12. **The ~7% gap from theoretical peak** (1,300 vs ~1,400 GFLOP/s) is from warp scheduling overhead, instruction pipeline bubbles, and the serial dependency chain in `sum += a * b` limiting ILP within each warp.
13. **AVX2+FMA SIMD is a game-changer for CPU.** Single-threaded SIMD (8 floats/op + fused multiply-add) delivers 13.5–34 GFLOP/s — a 13–32x speedup over scalar, far exceeding the theoretical 8x from wider registers alone thanks to better cache behavior from transposing B.
14. **AVX2+FMA+OpenMP (20 threads) reaches 109–195 GFLOP/s** — a 256x improvement over scalar and brings the CPU within **11x of the GPU** at 4096x4096. The gap from 3,193x to 11x shows how much scalar code leaves on the table.
