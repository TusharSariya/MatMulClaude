# CUDA Matrix Multiplication

CPU vs GPU matrix multiplication benchmarks, exploring performance differences and hardware limits.

> **Note:** The majority of this code is AI-generated (Claude) and is intended for learning purposes only.

## Files

| File | Description |
|---|---|
| `matmul_cpu.c` | CPU-only matrix multiply (single-threaded, pure C) |
| `matmul_gpu.cu` | GPU matrix multiply with naive and tiled (shared memory) kernels |
| `matmul_bench.cu` | Combined benchmark — runs CPU and GPU side-by-side, verifies correctness |
| `matmul_stress.cu` | Stress test — scales from 256 to 16384, auto-skips CPU when too slow |
| `matmul_vram_limit.cu` | Finds the GPU's max matrix size before VRAM runs out |
| `matmul_vram_narrow.cu` | Narrows down the exact VRAM breaking point |
| `results.md` | Full benchmark results and analysis |

## Building

Requires GCC and the CUDA toolkit. Match the toolkit version to your driver — check with `nvidia-smi`.

```bash
# CPU
gcc -O2 -o matmul_cpu matmul_cpu.c -lm

# GPU (adjust cuda version and arch to match your setup)
/usr/local/cuda-12.4/bin/nvcc -O2 -arch=sm_86 -o matmul_gpu matmul_gpu.cu
/usr/local/cuda-12.4/bin/nvcc -O2 -arch=sm_86 -o matmul_bench matmul_bench.cu
/usr/local/cuda-12.4/bin/nvcc -O2 -arch=sm_86 -o matmul_stress matmul_stress.cu
```

Change `-arch=sm_86` to match your GPU (e.g. `sm_75` for Turing, `sm_89` for Ada Lovelace).

## Running

```bash
./matmul_cpu          # CPU only (1024x1024)
./matmul_gpu          # GPU only — naive + tiled kernels
./matmul_bench        # CPU vs GPU comparison (256 to 2048)
./matmul_stress       # Full stress test up to 16384
./matmul_vram_limit   # Find GPU VRAM ceiling
```

## Results (RTX 3060 Ti, 8 GB)

See [results.md](results.md) for full data. Highlights:

- GPU tiled kernel: **~1,300 GFLOP/s** (consistent across sizes)
- CPU single-threaded: **0.4–4.6 GFLOP/s**
- GPU speedup: **1,000–3,200x** depending on matrix size
- GPU max size: **22,528 x 22,528** (limited by VRAM)
- CPU max practical size: **~4,096 x 4,096** (limited by O(N^3) time)

## License

This project is for educational use.
