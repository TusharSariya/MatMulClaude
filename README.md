# CUDA Matrix Multiplication

CPU vs GPU matrix multiplication benchmarks, exploring performance differences and hardware limits.

> **Note:** The majority of this code is AI-generated (Claude) and is intended for learning purposes only.

## Files

| File | Description |
|---|---|
| `matmul_cpu.c` | CPU-only matrix multiply (single-threaded, pure C) |
| `matmul_cpu_mt.c` | CPU matrix multiply with pthreads (manual thread pool) |
| `matmul_cpu_parallel.c` | CPU matrix multiply with OpenMP |
| `matmul_cpu_compare.c` | Benchmark comparing all 3 CPU implementations |
| `matmul_gpu.cu` | GPU matrix multiply with naive and tiled (shared memory) kernels |
| `matmul_bench.cu` | Combined benchmark — runs CPU and GPU side-by-side, verifies correctness |
| `matmul_stress.cu` | Stress test — scales from 256 to 16384, auto-skips CPU when too slow |
| `matmul_vram_limit.cu` | Finds the GPU's max matrix size before VRAM runs out |
| `matmul_vram_narrow.cu` | Narrows down the exact VRAM breaking point |
| `matmul_block_sweep.cu` | Tests block sizes 4x4 through 32x32 on naive + tiled kernels |
| `matmul_cpu_simd.c` | AVX2+FMA SIMD matmul, single-threaded and OpenMP |
| `matmul_profile.cu` | Minimal binary for GPU profiling with ncu |
| `matmul_optimizations.cu` | Advanced optimization study: cuBLAS/TensorOp, register tiling, double buffering, overlap, launch-bounds |
| `cublas_profile.cu` | Focused cuBLAS SGEMM/TensorOp profiling helper for Nsight tools |
| `tensoreCore.md` | Deep dive into Tensor Cores and why they outperform naive/tiled kernels |
| `cublas.md` | Deep dive into cuBLAS internals and why library GEMM outperforms custom baselines |
| `cublas_cookbook.md` | Practical cuBLAS argument patterns, row-major mapping, and troubleshooting checklist |
| `cpu.md` | Deep dive into scalar, pthreads, OpenMP, SIMD (AVX2+FMA) and why each is faster |
| `cudaMemoryHierarchy.md` | Guide to GPU memory hierarchy, coalescing, bank conflicts, ILP, spilling |
| `results.md` | Full benchmark results and analysis |

## Building

Requires GCC and the CUDA toolkit. Match the toolkit version to your driver — check with `nvidia-smi`.

```bash
# CPU (single-threaded)
gcc -O2 -o matmul_cpu matmul_cpu.c -lm

# CPU (pthreads)
gcc -O2 -o matmul_cpu_mt matmul_cpu_mt.c -lpthread -lm

# CPU (OpenMP)
gcc -O2 -fopenmp -o matmul_cpu_parallel matmul_cpu_parallel.c -lm

# CPU comparison (all 3)
gcc -O2 -fopenmp -o matmul_cpu_compare matmul_cpu_compare.c -lpthread -lm

# CPU SIMD (AVX2+FMA, requires Intel/AMD with AVX2 support)
gcc -O2 -mavx2 -mfma -fopenmp -o matmul_cpu_simd matmul_cpu_simd.c -lm -lpthread

# GPU (adjust cuda version and arch to match your setup)
/usr/local/cuda-12.4/bin/nvcc -O2 -arch=sm_86 -o matmul_gpu matmul_gpu.cu
/usr/local/cuda-12.4/bin/nvcc -O2 -arch=sm_86 -o matmul_bench matmul_bench.cu
/usr/local/cuda-12.4/bin/nvcc -O2 -arch=sm_86 -o matmul_stress matmul_stress.cu
/usr/local/cuda-12.4/bin/nvcc -O2 -arch=sm_86 -o matmul_block_sweep matmul_block_sweep.cu
/usr/local/cuda-12.4/bin/nvcc -O3 -arch=sm_86 -o matmul_optimizations matmul_optimizations.cu -lcublas
/usr/local/cuda-12.4/bin/nvcc -O3 -arch=sm_86 -o cublas_profile cublas_profile.cu -lcublas
```

Change `-arch=sm_86` to match your GPU (e.g. `sm_75` for Turing, `sm_89` for Ada Lovelace).

## Running

```bash
./matmul_cpu          # CPU single-threaded (1024x1024)
./matmul_cpu_mt       # CPU pthreads (1024x1024)
./matmul_cpu_parallel # CPU OpenMP (1024x1024)
./matmul_cpu_compare  # All 3 CPU implementations head-to-head (256 to 4096)
./matmul_gpu          # GPU only — naive + tiled kernels
./matmul_bench        # CPU vs GPU comparison (256 to 2048)
./matmul_stress       # Full stress test up to 16384
./matmul_vram_limit   # Find GPU VRAM ceiling
./matmul_block_sweep  # Block size sweep (4x4 to 32x32, naive + tiled)
./matmul_cpu_simd     # AVX2+FMA SIMD vs scalar (single-threaded and OpenMP)
./matmul_optimizations 1024 8            # Methods 1-4 + rectangular cuBLAS validation (defaults: M=768,N=1024,K=512)
./matmul_optimizations 1024 8 768 1024 512  # Explicit rectangular dimensions: M N K
./cublas_profile sgemm 1024 1024 1024 20     # SGEMM-only loop for profiler capture
./cublas_profile tensorop 1024 1024 1024 20  # TensorOp-only loop for profiler capture
```

## Results (RTX 3060 Ti, 8 GB)

See [results.md](results.md) for full data. Highlights:

- **16x16 block size is optimal** — 90–95% efficiency on tiled kernel across all matrix sizes
- GPU tiled kernel: **~1,300 GFLOP/s** (consistent across sizes)
- CPU single-threaded: **0.4–4.6 GFLOP/s**
- CPU pthreads/OpenMP (20 threads): **4.7–25 GFLOP/s** (5–17x over single-threaded)
- CPU AVX2+FMA+OpenMP (20 threads): **109–195 GFLOP/s** (up to 256x over scalar)
- GPU speedup vs best CPU (SIMD+OpenMP): **~11x** at 4096x4096
- GPU speedup vs scalar CPU: **~2,858x**
- GPU max size: **22,528 x 22,528** (limited by VRAM)
- CPU max practical size: **~4,096 x 4,096** (limited by O(N^3) time)
- New optimization study includes Tensor Core/cuBLAS baselines and transfer-overlap pipelines (see `results.md`)

## License

This project is for educational use.
