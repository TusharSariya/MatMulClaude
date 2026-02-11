# Tensor Cores: Why They Crush Naive and Basic Tiled Kernels

Deep dive into why NVIDIA Tensor Cores are dramatically faster than:
- your naive CUDA matmul (`matmul_naive`)
- your shared-memory tiled CUDA matmul (`matmul_tiled`)

Hardware context: RTX 3060 Ti (Ampere, sm_86).

---

## 1) The Core Difference in One Sentence

Your naive/tiled kernels run on standard FP32 CUDA cores, where each thread does scalar FMAs.
Tensor Core kernels execute matrix-multiply-accumulate (MMA) instructions that perform many FMAs per instruction on specialized hardware pipelines.

That changes the throughput class from "good CUDA core GEMM" to "Tensor Core GEMM".

---

## 2) What Tensor Cores Actually Compute

Conceptually, Tensor Cores do:

```
D = A * B + C
```

but on fixed-size matrix fragments per instruction (warp-level matrix math), not one scalar multiply-add at a time.

On Ampere, the Tensor Core path can use FP16/BF16/TF32 inputs and often FP32 accumulation.
This combination gives very high throughput while preserving much of FP32 accumulation quality.

---

## 3) Why Your Naive Kernel Is Fundamentally Slower

Your naive kernel:

```c
sum += A[row * n + k] * B[k * n + col];
```

### Bottlenecks

1. **No data reuse from shared memory**
   - Each thread repeatedly pulls from global memory.
   - Even coalesced accesses still incur heavy DRAM traffic.

2. **Low arithmetic intensity at thread level**
   - Work per fetched byte is limited.
   - Kernel tends memory-bound before compute units are saturated.

3. **Scalar accumulation chain**
   - `sum += ...` creates dependency each loop iteration.
   - Limits instruction-level parallelism inside a thread.

4. **Instruction overhead**
   - Many load/address/loop instructions relative to useful math.

Result: respectable but capped throughput for large GEMM.

---

## 4) Why Your Shared-Memory Tiled Kernel Is Better (But Still Not Tensor Core Class)

Your tiled kernel improves the two biggest naive problems:
- moves reuse to shared memory
- cuts global memory traffic by tile reuse

That is why it jumps from naive to around ~1.2-1.3 TFLOP/s in your earlier results.

But it still has ceilings:

1. **Still scalar FMA math model**
   - Each thread computes one (or a few) outputs with scalar-style loop.
   - You are not using Tensor Core MMA instructions.

2. **Synchronization cost**
   - `__syncthreads()` barriers per tile phase.
   - Overhead is amortized but still present.

3. **Shared-memory and register pipeline not as deep as vendor kernels**
   - cuBLAS/Tensor Core kernels are aggressively pipelined and software-scheduled.

4. **Instruction mix not fully optimized for GEMM**
   - Handwritten kernels are usually simpler than vendor microkernels.

So tiled is much better than naive, but still in CUDA-core GEMM territory.

---

## 5) Where Tensor Cores Get Their Massive Speedup

### A) Higher Effective Math Throughput Per Issued Instruction

Tensor Core MMA instructions do more fused multiply-accumulates per instruction than scalar/thread-level FMAs.
This massively increases math throughput density.

### B) Warp-Level Fragment Math

Instead of independent scalar threads, warps cooperate on matrix fragments.
This reduces per-output overhead and improves compute packing.

### C) Data Type Advantage

FP16/BF16/TF32 math paths are throughput-optimized in hardware.
When acceptable numerically, this unlocks much higher TFLOP/s than pure FP32 CUDA-core paths.

### D) Better Pipeline Feeding (in library implementations)

Tensor Core kernels in cuBLAS use deep pipelines:
- global -> shared staging
- asynchronous movement/prefetch patterns
- double-buffering
- carefully tuned register fragment reuse

This keeps Tensor Core units busy more consistently.

---

## 6) Interpreting Your Results

From your optimization benchmark (`matmul_optimizations.cu`):
- custom tiled FP32 kernel: ~0.8-1.1 TFLOP/s (varies run to run)
- cuBLAS SGEMM FP32: ~10 TFLOP/s class
- cuBLAS TensorOp FP16: ~28 TFLOP/s class

That pattern is expected:
1. custom basic kernels < vendor CUDA-core GEMM
2. vendor CUDA-core GEMM < vendor Tensor Core GEMM

The jump is not "just better coding" - it is mostly:
- specialized hardware path
- specialized kernel scheduling
- reduced precision path where acceptable

---

## 7) Accuracy Tradeoff: Why Tensor Core Error Was Higher

You measured larger max absolute error for FP16 TensorOp than FP32 SGEMM.

Reason:
- FP16 input has much smaller mantissa than FP32.
- Accumulation may still be FP32, but input quantization error enters every multiply.
- Large dot products can amplify rounding differences.

For many ML workloads, this is acceptable because:
- models are trained/inferred with mixed precision
- normalization/loss scaling techniques handle numeric range

For strict scientific FP32 fidelity, TensorOp FP16 may be unsuitable unless error bounds are validated.

---

## 8) Roofline View (Mental Model)

Think of three levels:

1. **Naive kernel**: mostly memory-bound, low reuse
2. **Shared-memory tiled**: improved arithmetic intensity, closer to compute roof
3. **Tensor Core/cuBLAS TensorOp**: much higher compute roof plus high reuse and better scheduling

So Tensor Core kernels win from both sides:
- they raise the roof
- they use a better ladder to reach it

---

## 9) If You Want to Push Your Own Kernel Closer

1. Add true asynchronous copy pipeline (`cp.async`) for Ampere.
2. Increase per-thread output tile (register tiling 2x2, 4x1, etc.).
3. Move toward warp-level matrix fragment logic.
4. Tune launch bounds and register pressure jointly.
5. Compare every step against cuBLAS to quantify gap closure.

Even then, matching cuBLAS Tensor Core kernels is very hard because they encode years of architecture-specific tuning.

---

## 10) Bottom Line

Tensor Cores are so fast versus your naive and shared-memory kernels because they combine:
- specialized matrix math hardware
- higher-throughput numeric formats
- warp-fragment execution model
- deeply optimized dataflow/pipelining in vendor kernels

Your tiled kernel is a strong CUDA-core baseline, but Tensor Core kernels operate in a different performance tier.
