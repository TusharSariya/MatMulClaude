# cuBLAS GEMM: Why It Is So Fast vs Custom Naive/Tiled Kernels

This document explains why `cuBLAS` matrix multiplication is usually much faster than hand-written baseline kernels, including your:
- naive kernel (`matmul_naive`)
- shared-memory tiled kernel (`matmul_tiled`)

Target GPU context: RTX 3060 Ti (Ampere, sm_86).

---

## 1) What cuBLAS Gives You

`cuBLAS` is NVIDIA's production BLAS library for GPUs.
For GEMM (`C = A * B + C`), it ships architecture-tuned kernels that are selected dynamically based on:
- matrix sizes and shapes
- data types
- transpose/layout options
- alignment and stride
- hardware generation

So you are not getting "one kernel"; you are getting a dispatch engine to a large family of tuned kernels.

---

## 2) Why cuBLAS Beats Typical Custom Kernels

### A) Multi-level tiling beyond simple block tiling

Most custom examples use one-level CTA/block tiling.
cuBLAS kernels typically use multiple levels:
- CTA tile (block-level)
- warp tile
- instruction tile / fragment tile

Each level is chosen to maximize reuse and minimize movement from slower memory tiers.

### B) Aggressive software pipelining

cuBLAS overlaps:
- global memory fetch
- shared memory staging
- register fragment consumption
- math execution

The result is fewer pipeline bubbles and better steady-state issue rate.

### C) Kernel specializations and autotuned heuristics

Different GEMM shapes need different kernels.
cuBLAS avoids one-size-fits-all by selecting variants optimized for:
- square matrices
- tall-skinny or short-wide matrices
- batched workloads
- different K-dimension depths

### D) Better instruction scheduling and register allocation

Vendor kernels are tuned around the exact SM scheduler/execution behavior.
Instruction order, unroll depth, and register lifetime are all tuned to maximize occupancy and utilization with minimal stalls.

### E) Tensor Core integration

When math mode and types permit, cuBLAS routes to Tensor Core kernels that are an entirely higher throughput class than CUDA-core scalar FMA kernels.

---

## 3) Comparing Against Your Naive Kernel

Your naive kernel is valuable as a correctness baseline, but performance-limited because:

1. Every output element repeatedly loads from global memory.
2. Minimal reuse from fast on-chip memories.
3. Inner loop has serial accumulation dependency.
4. High control/address overhead per useful FLOP.

cuBLAS avoids these bottlenecks with deep tiling and scheduling, so it spends far more cycles doing math rather than waiting/moving data.

---

## 4) Comparing Against Your Shared-Memory Tiled Kernel

Your tiled kernel fixes the biggest naive issue (global traffic) and is a good custom implementation.
Still, cuBLAS is usually much faster because:

1. **More levels of reuse**
   - not just shared-memory tile reuse, but warp/register fragment reuse.

2. **Higher math density**
   - especially with Tensor Core routes.

3. **Lower overhead per output**
   - stronger unrolling and microkernel design.

4. **Shape- and architecture-aware kernel choice**
   - your kernel is static; cuBLAS adapts.

5. **Battle-tested memory movement strategy**
   - carefully orchestrated staging/prefetch to keep units fed.

---

## 5) Understanding the Huge Gap in Your Measurements

From your optimization runs (`matmul_optimizations.cu`, `N=1024`), you observed:
- custom tiled kernel in ~1 TFLOP/s ballpark
- `cuBLAS SGEMM` in ~10 TFLOP/s ballpark
- `cuBLAS TensorOp FP16` in ~28 TFLOP/s ballpark

This is normal for three reasons:

1. **Kernel maturity gap**: simple educational kernel vs production microkernels.
2. **Hardware path gap**: CUDA-core FP32 path vs Tensor Core path.
3. **Dispatch/selection gap**: static custom kernel vs algorithm portfolio chosen per problem.

---

## 6) Why cuBLAS SGEMM (FP32) Is Already Much Faster

Even before Tensor Cores, cuBLAS FP32 GEMM can be dramatically ahead because:
- better tile shapes for SM occupancy
- deeper unroll and register blocking
- optimized shared-memory layout and access scheduling
- minimized instruction overhead around math loops

In short: it extracts much more useful work per cycle from the same CUDA cores.

---

## 7) Why cuBLAS TensorOp Is an Additional Leap

When using `cublasGemmEx` with TensorOp-friendly types and math mode:
- matrix multiply maps to Tensor Core instructions
- throughput per cycle jumps significantly
- remaining bottlenecks shift to data delivery and scheduling rather than raw FMA capacity

This is why TensorOp can be several times faster than cuBLAS FP32 SGEMM, which is itself much faster than basic custom kernels.

---

## 8) Numerical Precision: Performance vs Fidelity

cuBLAS speed modes involve precision tradeoffs:

1. **FP32 SGEMM**
   - highest fidelity among common fast paths
   - slower than reduced-precision TensorOp paths

2. **FP16/BF16/TF32 TensorOp paths**
   - much faster
   - different rounding/precision behavior
   - requires workload-specific error acceptance

Your measured max error difference (FP16 TensorOp > FP32 SGEMM) is expected.

---

## 9) Practical Workflow: How to Use cuBLAS as Performance Ceiling

1. Keep your custom kernel for learning and specialized behavior.
2. Benchmark every custom optimization against cuBLAS.
3. Decide target:
   - if you need absolute max throughput, use cuBLAS/cuBLASLt
   - if you need custom fusion/layout behavior, keep custom kernel and track gap
4. Report both kernel-only and end-to-end timing (copies + launch overhead).

This gives honest performance positioning.

---

## 10) Where cuBLAS Can Still Underperform

cuBLAS is usually best for GEMM, but there are cases where custom can win:
- very small matrices where launch overhead dominates
- highly fused operations where avoiding intermediate writes is critical
- unusual layouts or sparsity patterns not well covered by standard GEMM

For standard dense GEMM, cuBLAS is typically the correct baseline and often the winner.

---

## 11) Bottom Line

cuBLAS is so fast because it combines:
- architecture-specific kernel families
- deep multi-level tiling and pipelining
- aggressive register/shared-memory optimization
- dynamic kernel selection
- optional Tensor Core execution paths

Your naive and shared-memory kernels are excellent for understanding fundamentals, but cuBLAS represents industrial-strength GEMM engineering on top of specialized hardware.

---

## 12) Row-Major Argument Pattern Used In This Repo

Our host/device arrays are row-major (`A[MxK]`, `B[KxN]`, `C[MxN]`), while cuBLAS APIs are column-major.
The practical mapping is:

```c
// Row-major target: C = A * B
// Column-major cuBLAS call: C^T = B^T * A^T
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha, d_B, N,
                    d_A, K,
            &beta,  d_C, N);
```

TensorOp path follows the same dimension/leading-dimension mapping via `cublasGemmEx`.
See `cublas_cookbook.md` for the compact checklist and troubleshooting flow.

---

## 13) Latest Sweep Snapshot (Same Hardware, New Run)

Kernel-only (`matmul_optimizations.cu`, Method 1, custom tiled + cuBLAS):

| N | Tiled baseline | cuBLAS SGEMM | cuBLAS TensorOp FP16 |
|---:|---:|---:|---:|
| 512 | 1454.2 GFLOP/s | 7322.5 GFLOP/s | 21487.2 GFLOP/s |
| 1024 | 897.5 GFLOP/s | 6159.0 GFLOP/s | 28659.4 GFLOP/s |
| 2048 | 895.1 GFLOP/s | 6608.9 GFLOP/s | 21159.3 GFLOP/s |

Rectangular validation (`M=768, N=1024, K=512`):

| Variant | Throughput | Max Abs Error vs CPU |
|---|---:|---:|
| cuBLAS SGEMM | 8896.3 GFLOP/s | 2.441406e-04 |
| cuBLAS TensorOp FP16 | 26040.8 GFLOP/s | 6.962585e-02 |

Pattern remains consistent: SGEMM is the high-fidelity fast baseline, and TensorOp is much faster with larger numeric drift due to FP16 input quantization.

---

## 14) Profiling Note (Nsight Compute)

A profiling helper (`cublas_profile.cu`) was added so SGEMM and TensorOp can be profiled independently.
On this machine, `ncu` launch currently fails with:

- `Cuda driver is not compatible with Nsight Compute`

Action: align GPU driver and Nsight Compute versions, then run:

```bash
ncu --set speedOfLight --target-processes all --kernel-name-base demangled --launch-count 1 ./cublas_profile sgemm 1024 1024 1024 20
ncu --set speedOfLight --target-processes all --kernel-name-base demangled --launch-count 1 ./cublas_profile tensorop 1024 1024 1024 20
```

Once compatibility is fixed, compare tensor-pipe utilization and achieved FLOP metrics directly between the two paths.
