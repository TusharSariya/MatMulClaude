# cuBLAS Cookbook (Row-Major C/C++)

Quick-reference patterns for this repo's GEMM usage.

---

## 1) Mental Model

cuBLAS APIs are column-major by default.  
If your host/device buffers are row-major (`A[MxK]`, `B[KxN]`, `C[MxN]`), map the operation by swapping operands in the cuBLAS call:

- Desired math (row-major): `C = A * B`
- cuBLAS call (column-major view): compute `C^T = B^T * A^T`

For `cublasSgemm`, that means:

```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,   // B viewed as col-major (N x K)
            d_A, K,   // A viewed as col-major (K x M)
            &beta,
            d_C, N);  // C viewed as col-major (N x M)
```

---

## 2) Leading Dimension Rules (`lda/ldb/ldc`)

Think "physical leading stride in the column-major view":

- `lda` = rows of op(A) in column-major memory view
- `ldb` = rows of op(B)
- `ldc` = rows of C

For repo row-major mapping above:

- `lda = N` for `d_B`
- `ldb = K` for `d_A`
- `ldc = N` for `d_C`

Wrong `ld*` usually gives either:
- clearly wrong output (bad indexing), or
- legal but slower access patterns (less common in this pattern).

---

## 3) Common GEMM Patterns

### FP32 reference path

```c
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
```

Use when fidelity is priority and as correctness baseline.

### Tensor Core path (FP16 inputs, FP32 compute)

```c
float alpha = 1.0f, beta = 0.0f;
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             N, M, K, &alpha,
             d_Bh, CUDA_R_16F, N,
             d_Ah, CUDA_R_16F, K,
             &beta,
             d_Ch, CUDA_R_16F, N,
             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

Use when throughput is priority and FP16 input quantization is acceptable.

---

## 4) Quick Validation Checklist

- Verify dimensions on paper first: `A[MxK]`, `B[KxN]`, `C[MxN]`.
- Start with `alpha=1`, `beta=0`.
- Compare against CPU reference (`max abs error`).
- Validate one square and one rectangular case.
- Re-run with fixed random seeds for reproducibility.

---

## 5) Troubleshooting (One Page)

- **Wrong numbers everywhere**: re-check `M/N/K` ordering and `lda/ldb/ldc`.
- **Looks transposed**: row-major to column-major mapping is wrong; swap A/B in cuBLAS call pattern above.
- **TensorOp error much larger**: expected for FP16 input quantization; compare against FP32 SGEMM and decide acceptable threshold.
- **Kernel-only fast but end-to-end slow**: optimize transfer path (pinned host memory, stream overlap, graph launch).
- **Profiler won't collect metrics**: ensure Nsight Compute version is driver-compatible.

---

## 6) Decision Rubric (Custom vs cuBLAS/cuBLASLt)

- Use **cuBLAS SGEMM** when you need robust FP32 baseline performance quickly.
- Use **cuBLAS TensorOp / GemmEx** when throughput dominates and reduced-precision input is acceptable.
- Use **cuBLASLt** when you need algorithm selection, richer layout control, or epilogue fusion.
- Keep **custom kernels** for learning, special fused ops, tiny-shape edge cases, or unusual data layouts.
- Always measure both **kernel-only** and **end-to-end** timing before deciding.
