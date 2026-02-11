# CUDA Memory Hierarchy

A reference guide to the GPU execution model and memory system, with specific numbers for the **RTX 3060 Ti** (Ampere GA104, sm_86, 38 SMs).

> **Note:** This document is AI-generated and intended for learning purposes.

---

## Execution Hierarchy

```
GPU (1 device)
└── SM (Streaming Multiprocessor) — 38 on RTX 3060 Ti
    ├── Block 0 (up to 16 blocks per SM)
    │   ├── Warp 0 — 32 threads, execute in lockstep
    │   ├── Warp 1 — 32 threads
    │   └── ...
    ├── Block 1
    │   └── ...
    └── ... (up to 48 warps total per SM = 1,536 threads)
```

| Level | RTX 3060 Ti | Notes |
|---|---|---|
| GPU | 1 device | 38 SMs, 4,864 FP32 cores |
| SM | 38 | Each has 128 FP32 cores, 4 warp schedulers |
| Blocks per SM | Up to 16 | Assigned to one SM, never migrates |
| Warps per SM | Up to 48 | 1,536 threads / 32 = 48 warps max |
| Threads per warp | 32 (fixed) | Execute same instruction in lockstep |
| Threads per SM | Up to 1,536 | Hard limit |

A **kernel launch** creates a grid of blocks. The hardware scheduler distributes blocks across SMs. Each SM runs multiple blocks concurrently, interleaving their warps to hide memory latency.

---

## Memory Types at a Glance

```
Speed                                          Size

Fastest ┌──────────────────┐ Smallest
   ~1 cy│   REGISTERS      │ 256 KB / SM (65,536 × 32-bit)
        │   per thread      │
        ├──────────────────┤
 ~20 cy │   SHARED MEMORY  │ up to 100 KB / SM
        │   per block       │
        ├──────────────────┤
 ~30 cy │   L1 CACHE       │ 28–128 KB / SM (shared pool with shared mem)
        │   per SM          │
        ├──────────────────┤
~200 cy │   L2 CACHE       │ 4 MB (device-wide)
        │   all SMs         │
        ├──────────────────┤
~400 cy │   GLOBAL MEMORY  │ 8 GB GDDR6
Slowest │   (VRAM)          │
        └──────────────────┘ Largest

Also: Constant Memory (64 KB, dedicated cache)
      Texture Memory (backed by global, spatial caching)
      Local Memory (per-thread spills, backed by global)
```

---

## Registers

| Property | Value |
|---|---|
| Scope | Private to each thread |
| Size per SM | 65,536 registers (256 KB) |
| Max per thread | 255 |
| Latency | ~1 cycle |
| Location | On-chip, inside the SM |

Registers are the fastest memory. Every variable in your kernel that fits in a register stays here. The total register file is shared across all threads on the SM, which creates a fundamental tradeoff:

```
More threads per SM  →  fewer registers per thread
Fewer registers      →  compiler may spill to local memory (slow)

Example:
  1,536 threads (max occupancy): 65,536 / 1,536 = 42 registers per thread
  1,024 threads:                 65,536 / 1,024 = 64 registers per thread
    512 threads:                 65,536 /   512 = 128 registers per thread
```

### Register Spilling

When a kernel needs more registers than available, the compiler "spills" values to **local memory** — which physically lives in GDDR6, accessed through the cache hierarchy:

```
Register access:     ~1 cycle
Spill hits L1:       ~30 cycles    (30x slower)
Spill hits L2:       ~200 cycles   (200x slower)
Spill misses to DRAM: ~400+ cycles  (400x slower)
```

You can check register usage with `nvcc --ptxas-options=-v` and constrain it with `__launch_bounds__` or `--maxrregcount`.

---

## Shared Memory

| Property | Value |
|---|---|
| Scope | All threads within the same **block** |
| Size per SM | Configurable, up to 100 KB (from a 128 KB shared pool with L1) |
| Default max per block | 48 KB |
| Latency | ~20–30 cycles |
| Bandwidth | ~128 bytes/clock/SM (~218 GB/s per SM at boost clock) |
| Location | On-chip SRAM, inside the SM |

Shared memory is the programmer's explicitly-managed cache. Threads in a block cooperate to load data from global memory into shared memory, then reuse it many times — this is exactly what the tiled matmul kernel does.

### Bank Conflicts

Shared memory is divided into **32 banks** (matching warp size). Each bank is 4 bytes wide and can serve one address per clock. Banks are interleaved:

```
Address 0–3   → Bank 0
Address 4–7   → Bank 1
...
Address 124–127 → Bank 31
Address 128–131 → Bank 0  (wraps around)
```

**No conflict** — all 32 threads hit different banks (1 transaction):
```c
// Thread i reads shared[i] → each thread hits a unique bank
float val = shared[threadIdx.x];
```

**Broadcast** — all threads read the same address (1 transaction):
```c
// All threads read shared[0] → hardware broadcasts
float val = shared[0];
```

**N-way conflict** — N threads hit the same bank at different addresses (N serial transactions):
```c
// Thread i reads shared[2*i] → only 16 banks used, 2-way conflict
// Thread i reads shared[32*i] → ALL hit bank 0, 32-way conflict (32x slower!)
```

**Fix with padding:**
```c
__shared__ float tile[32][33];  // 33 instead of 32
// Column access: stride of 33 × 4 bytes
// 33 mod 32 = 1, so successive rows hit successive banks → no conflict
```

---

## L1 Cache

| Property | Value |
|---|---|
| Scope | All threads on the same SM (hardware-managed) |
| Size per SM | 28–128 KB (shares 128 KB pool with shared memory) |
| Latency | ~30–34 cycles |
| Cache line | 128 bytes |
| Location | On-chip SRAM, inside the SM |

On Ampere, L1 and shared memory use the **same physical SRAM**. The split is configurable:

```
Shared Memory ↑  =  L1 Cache ↓
    0 KB shared  →  128 KB L1
   48 KB shared  →   80 KB L1
  100 KB shared  →   28 KB L1
```

L1 automatically caches global memory reads. You don't control what gets cached — the hardware manages eviction. Stores typically bypass L1 and go directly to L2.

---

## L2 Cache

| Property | Value |
|---|---|
| Scope | All SMs on the entire GPU (device-wide) |
| Size | 4 MB |
| Latency | ~200 cycles |
| Bandwidth | ~2–3 TB/s (internal crossbar) |
| Cache line | 128 bytes |
| Location | On-chip, between SMs and memory controllers |

L2 is the **coherence point** — all SMs see a consistent view of data here. On Ampere, you can use **L2 residency controls** (`cudaAccessPolicyWindow`) to pin hot data in L2.

### Cache Hierarchy Flow

When a thread reads from global memory:

```
Thread requests address
    │
    ▼
L1 Cache (per SM, ~30 cycles)
    │ miss
    ▼
L2 Cache (device-wide, ~200 cycles)
    │ miss
    ▼
GDDR6 DRAM (~400–800 cycles)
```

---

## Global Memory (VRAM)

| Property | Value |
|---|---|
| Scope | All threads, all blocks, all grids, and the host CPU |
| Size | 8 GB GDDR6 |
| Latency | 400–800 cycles (full miss) |
| Bandwidth | 448 GB/s peak (256-bit bus, 14 Gbps) |
| Location | Off-chip DRAM chips on the PCB |

Global memory is where your matrices live (`cudaMalloc`). It's the largest and slowest memory.

### Memory Coalescing

When a warp (32 threads) accesses global memory, the hardware **coalesces** individual requests into cache-line-sized transactions. This is critical for performance.

**Coalesced (ideal)** — consecutive threads read consecutive addresses:
```c
// 32 threads × 4 bytes = 128 bytes = 1 cache line = 1 transaction
float val = data[threadIdx.x + blockIdx.x * blockDim.x];
```

**Strided (wasteful)** — threads access every Nth element:
```c
// Stride 2: 32 threads touch 256 bytes → 2 cache lines, 50% wasted
float val = data[threadIdx.x * 2];

// Stride 32: 32 threads touch 4096 bytes → 32 cache lines, 97% wasted!
float val = data[threadIdx.x * 32];
```

**Random/scattered (worst)** — each thread reads an unrelated address:
```c
// Up to 32 separate 128-byte transactions for 128 bytes of useful data
// Effective bandwidth: 448 / 32 = ~14 GB/s instead of 448 GB/s
float val = data[random_index[threadIdx.x]];
```

**Rule of thumb:** structure your data so that thread 0 reads address N, thread 1 reads N+4, thread 2 reads N+8, etc. Prefer **Structure of Arrays** (SoA) over **Array of Structures** (AoS).

---

## Constant Memory

| Property | Value |
|---|---|
| Scope | All threads (read-only from device, written by host) |
| Size | 64 KB total |
| Cache | Dedicated ~8 KB per SM |
| Latency | ~4 cycles when all threads read the **same** address (broadcast) |
| Location | Physically in GDDR6, cached in dedicated on-chip constant cache |

```c
__constant__ float params[256];  // declared at file scope

// Host side:
cudaMemcpyToSymbol(params, host_data, sizeof(host_data));
```

**The catch:** constant memory is optimized for **uniform access** — all 32 threads in a warp reading the same address. If threads read different addresses, accesses are serialized (up to 32x slower). For divergent read-only data, use global memory with L1 caching instead.

---

## Texture Memory

| Property | Value |
|---|---|
| Scope | All threads (read-only from device) |
| Size | Backed by global memory (up to 8 GB) |
| Cache | Unified with L1 on Ampere (shared 128 KB pool) |
| Latency | ~300–500 cycles on miss |
| Location | Off-chip DRAM with on-chip cache |

Texture memory provides hardware-accelerated:
- 2D/3D spatial locality caching (optimized for nearby access patterns)
- Bilinear/trilinear interpolation
- Border/clamp/wrap addressing
- Automatic format conversion (e.g., 8-bit → float)

Mostly used in graphics and image processing. For general compute like matrix multiply, shared memory is the better tool.

---

## Local Memory

| Property | Value |
|---|---|
| Scope | Private to each thread |
| Size per thread | Up to 512 KB |
| Latency | Same as global memory (400–800 cycles on miss, ~30 on L1 hit) |
| Location | Physically in GDDR6 DRAM, cached through L1 and L2 |

Despite the name "local," this is **not fast**. It lives in DRAM. The hardware uses it for:
- Register spills
- Arrays with runtime-variable indices the compiler can't resolve
- Large structs that exceed register capacity

If `nvcc --ptxas-options=-v` shows high local memory usage ("lmem"), your kernel is likely register-spilling.

---

## Summary Table

| Memory | Location | Size (3060 Ti) | Latency | Scope | R/W |
|---|---|---|---:|---|---|
| Registers | On-chip (SM) | 256 KB/SM | ~1 cy | Thread | R/W |
| Shared Memory | On-chip (SM) | ≤100 KB/SM | ~20–30 cy | Block | R/W |
| L1 Cache | On-chip (SM) | 28–128 KB/SM | ~30 cy | SM (auto) | Transparent |
| L2 Cache | On-chip (global) | 4 MB | ~200 cy | Device (auto) | Transparent |
| Constant | Off-chip + cache | 64 KB + ~8 KB/SM | ~4 cy (broadcast) | Device | Read-only |
| Global (VRAM) | Off-chip GDDR6 | 8 GB | ~400–800 cy | Device + Host | R/W |
| Local | Off-chip GDDR6 | 512 KB/thread | ~400–800 cy | Thread | R/W |
| Texture | Off-chip + L1 | = Global | ~300–500 cy | Device | Read-only |

---

## Why This Matters for Matrix Multiply

In the tiled matmul kernel:

1. **Global memory** stores the full A, B, C matrices (allocated with `cudaMalloc`)
2. Each block loads a **tile** into **shared memory** (~20 cycle access instead of ~400)
3. Each thread accumulates a **sum in registers** (~1 cycle)
4. `__syncthreads()` ensures all threads finish loading before any thread reads
5. The tile is reused by all threads in the block — a 16×16 tile is loaded once but used 256 times

Without tiling, every multiply reads from global memory (400+ cycles). With tiling, most reads hit shared memory (20 cycles) — a **20x latency reduction** that directly translates to the ~30% throughput improvement seen in the benchmarks.

```
Naive kernel:   each thread does N global reads         → memory bound at ~1,000 GFLOP/s
Tiled kernel:   each thread does N/16 global reads      → hits ~1,300 GFLOP/s (93% efficiency)
                + N shared memory reads (20x faster)
```

The remaining 7% gap to theoretical peak is from warp scheduling overhead, bank conflicts, and instruction pipeline bubbles.

---

## Performance Factors

Beyond just choosing the right memory type, four factors determine whether your kernel actually runs fast.

---

### 1. Memory Coalescing (Global Memory Access Patterns)

Global memory is accessed in **cache-line-sized transactions** of 128 bytes. When a warp (32 threads) executes a load instruction, the hardware gathers all 32 addresses and groups them by which cache line they fall in. One transaction per cache line.

The best case: 32 threads each read a consecutive 4-byte float = 128 bytes = exactly 1 cache line = **1 transaction**. The worst case: 32 threads read 32 random addresses in 32 different cache lines = **32 transactions**, each pulling 128 bytes to deliver just 4 bytes of useful data.

```
COALESCED — consecutive threads read consecutive addresses:

Thread:     0     1     2     3    ...   31
Address:  [0x00] [0x04] [0x08] [0x0C]  [0x7C]
           └──────────────────────────────┘
                   1 cache line (128 bytes)
                   1 transaction
                   128 bytes transferred, 128 bytes used → 100% efficiency

Code:  float val = data[threadIdx.x];  // thread i reads element i
```

```
STRIDED — threads skip elements:

Thread:     0     1     2     3    ...   31
Address:  [0x00] [0x08] [0x10] [0x18]  [0xF8]
           └──────────────┘└──────────────┘
            cache line 0     cache line 1
            2 transactions
            256 bytes transferred, 128 bytes used → 50% efficiency

Code:  float val = data[threadIdx.x * 2];  // stride of 2
```

```
SCATTERED — threads read random locations:

Thread:     0       1       2       3     ...
Address:  [0x1A00] [0x40]  [0x3F80] [0x800] ...
           each in a different cache line
           up to 32 transactions
           4096 bytes transferred, 128 bytes used → 3% efficiency

Code:  float val = data[index[threadIdx.x]];  // random access
```

**Why it matters so much:** Global memory bandwidth is 448 GB/s on your 3060 Ti. With perfect coalescing you get all 448 GB/s. With stride-32 access, you get 448/32 = 14 GB/s — the memory bus is doing 32x the work for the same useful data.

**How to think about it:** Imagine 32 people (threads) at a library. Coalesced = they all request books from the same shelf, librarian grabs them all in one trip. Scattered = each person wants a book from a different floor, librarian makes 32 separate trips.

**Rules of thumb:**
- `data[threadIdx.x + offset]` — coalesced (consecutive threads, consecutive addresses)
- `data[threadIdx.x * stride]` — partially coalesced (wastes bandwidth proportional to stride)
- `data[row * width + threadIdx.x]` — coalesced (threadIdx.x in the innermost dimension)
- `data[threadIdx.x * width + col]` — NOT coalesced (threadIdx.x in the outer dimension = large stride)

**Structure of Arrays (SoA) vs Array of Structures (AoS):**

```c
// AoS — BAD for coalescing
struct Particle { float x, y, z, w; };
Particle particles[N];
// Thread i reads particles[i].x → stride of 16 bytes (sizeof(Particle))
// Threads access: [0], [16], [32], [48]... → 4 cache lines for 32 threads

// SoA — GOOD for coalescing
struct Particles { float x[N], y[N], z[N], w[N]; };
Particles p;
// Thread i reads p.x[i] → stride of 4 bytes (consecutive floats)
// Threads access: [0], [4], [8], [12]... → 1 cache line for 32 threads
```

---

### 2. Shared Memory Bank Conflicts

Shared memory is divided into **32 banks**, each 4 bytes wide, interleaved:

```
Byte address:   0-3     4-7     8-11   ...  124-127   128-131  132-135 ...
Bank:            0       1       2     ...    31         0        1    ...
```

Each bank can serve **one address per clock cycle**. When multiple threads in a warp access different addresses that map to the same bank, the accesses are serialized — this is a **bank conflict**.

**No conflict — each thread hits a different bank:**
```c
__shared__ float s[256];
float val = s[threadIdx.x];
// Thread 0 → bank 0, thread 1 → bank 1, ... thread 31 → bank 31
// All 32 banks active in parallel → 1 cycle
```

**Broadcast — all threads read the same address:**
```c
float val = s[0];
// All 32 threads read address 0 → bank 0
// Hardware broadcasts the value → 1 cycle (special case, NOT a conflict)
```

**2-way conflict — two threads per bank:**
```c
float val = s[threadIdx.x * 2];
// Thread 0 → s[0] → bank 0
// Thread 1 → s[2] → bank 2
// ...
// Thread 16 → s[32] → bank 0  ← same bank as thread 0!
// Only 16 unique banks used, 2 threads per bank → 2 serial accesses → 2 cycles
```

**32-way conflict — all threads in one bank (worst case):**
```c
float val = s[threadIdx.x * 32];
// Thread 0 → s[0]    → bank 0
// Thread 1 → s[32]   → bank 0
// Thread 2 → s[64]   → bank 0
// ALL hit bank 0 → 32 serial accesses → 32 cycles (32x slower!)
```

**Visualizing it:**

```
32 banks, 32 threads in a warp:

NO CONFLICT (stride 1):          2-WAY CONFLICT (stride 2):
Thread: 0  1  2  3 ... 31        Thread: 0  1  2  3 ... 15 16 17 ... 31
Bank:   0  1  2  3 ... 31        Bank:   0  2  4  6 ... 30  0  2 ...  30
        ↓  ↓  ↓  ↓     ↓                ↓↓    ↓↓        ↓↓
        1 access each             2 accesses per active bank

Throughput: 128 bytes/cycle       Throughput: 64 bytes/cycle
```

**The padding trick:**

2D shared memory arrays are the most common source of bank conflicts. Consider a column access pattern:

```c
__shared__ float tile[32][32];
// Accessing column: tile[threadIdx.x][col]  (fixed col, varying row)
// tile[0][col] → bank (0*32 + col) % 32 = col
// tile[1][col] → bank (1*32 + col) % 32 = col  ← SAME BANK!
// All 32 threads hit the same bank → 32-way conflict

// Fix: pad the inner dimension
__shared__ float tile[32][33];  // 33 instead of 32
// tile[0][col] → bank (0*33 + col) % 32
// tile[1][col] → bank (1*33 + col) % 32 = (col + 1) % 32  ← different bank!
// Each thread hits a unique bank → no conflict
```

**When to worry:** Bank conflicts matter most when shared memory is in the critical path (tight inner loops). A 2-way conflict halves shared memory throughput. If your kernel is compute-bound rather than memory-bound, small conflicts may not show up in wall-clock time.

---

### 3. Instruction-Level Parallelism (ILP)

A GPU hides latency primarily through **warp-level parallelism** — when one warp stalls waiting for data, the scheduler switches to another warp. But within a single warp, **instruction-level parallelism** also matters: can the hardware execute multiple instructions from the same warp simultaneously?

**The pipeline:** Each SM has 4 warp schedulers. Each scheduler picks one warp per cycle and issues one instruction from it. But the instruction takes multiple cycles to complete (the pipeline). If the next instruction depends on the result of the current one, the warp stalls until the result is ready.

**Low ILP — serial dependency chain:**
```c
// Every operation depends on the previous one
float a = x[i];          // cycle 1: issue load     (result ready at ~cycle 30 for L1)
float b = a * 2.0f;      // STALL until cycle 30... then issue multiply
float c = b + 1.0f;      // STALL until multiply finishes... then issue add
y[i] = c;                // STALL until add finishes... then issue store
```
The warp scheduler can't issue any of these instructions ahead of time because each depends on the previous result. The warp stalls at every step.

**High ILP — independent operations:**
```c
// Four independent loads — no dependencies between them
float a = x[i];          // cycle 1: issue load a
float b = y[i];          // cycle 1: issue load b (independent — can overlap!)
float c = z[i];          // cycle 2: issue load c
float d = w[i];          // cycle 2: issue load d
// All four loads are in flight simultaneously
// Results arrive around the same time
float result = a + b + c + d;  // only NOW do we have a dependency
```

**Why it matters for matmul:**

The inner loop of our kernels has a dependency chain:
```c
sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
// Decomposed:
// 1. Load sA[y][k]         ← shared memory read (~20 cycles)
// 2. Load sB[k][x]         ← shared memory read (~20 cycles, INDEPENDENT of #1)
// 3. Multiply #1 * #2      ← depends on BOTH loads
// 4. Add to sum             ← depends on multiply AND previous sum
```

Steps 1 and 2 are independent and can overlap. But step 4 depends on step 3 which depends on steps 1 and 2. And the next iteration's step 4 depends on this iteration's step 4 (`sum` is accumulating). This creates a serial chain that limits how fast one warp can execute.

**How the GPU compensates:**
- **Warp-level parallelism:** While warp 0 stalls on step 3 waiting for loads, the scheduler runs warp 1, 2, 3, etc. With 48 warps per SM, there's almost always a warp ready to go.
- **Loop unrolling:** When `BLOCK_SIZE` is a compile-time constant, the compiler can unroll the inner loop. With unrolling, loads from future iterations can be issued while the current iteration is still computing, effectively prefetching data.
- **Dual-issue:** On Ampere, the FP32/INT32 execution units can sometimes execute two instructions per cycle if they're independent and use different functional units.

**The tradeoff:** Higher ILP means the GPU needs fewer warps to stay busy, so you can tolerate lower occupancy. Lower ILP means you need more warps (higher occupancy) to cover the stalls. This is why 100% occupancy is more important for the naive kernel (long global memory stalls) than for the tiled kernel (short shared memory stalls).

---

### 4. Register Spilling

Every thread has access to a slice of the SM's register file. On your RTX 3060 Ti, each SM has **65,536 registers** (each 32 bits). The compiler assigns variables to registers automatically — they're the fastest storage (~1 cycle access).

**What spilling is:**

When a kernel needs more registers than available, the compiler **spills** — it saves some register values to **local memory** (which is physically in GDDR6 DRAM) and reloads them later. This is catastrophic for performance:

```
Access type:              Latency:        Slowdown vs register:
Register                    ~1 cycle       1x
Spill → L1 cache hit       ~30 cycles     30x
Spill → L2 cache hit       ~200 cycles    200x
Spill → DRAM (L2 miss)     ~400 cycles    400x
```

**When it happens:**

```c
// This kernel uses many local variables
__global__ void heavy_kernel(...) {
    float a, b, c, d, e, f, g, h;           // 8 registers
    float matrix[4][4];                       // 16 registers (if compiler keeps in regs)
    float temp1, temp2, temp3, temp4;         // 4 registers
    float accum[8];                           // 8 registers
    // ... lots of computation across all of these ...
    // If this exceeds the register budget → spill
}
```

The compiler reports spills with `nvcc --ptxas-options=-v`:
```
ptxas info: Used 128 registers, 56 bytes spill stores, 56 bytes spill loads
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 THIS IS BAD — values going to DRAM
```

Compare with a clean kernel (like our matmul):
```
ptxas info: Used 38 registers, 0 bytes spill stores, 0 bytes spill loads
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                NO SPILLS — everything stays in registers ✓
```

**Why it's not just about the 255-register limit:**

Even if your kernel uses far fewer than 255 registers per thread, the **total register file** creates a cap on how many threads can run:

```
SM register file:    65,536 registers

At 32 regs/thread:   65,536 / 32 = 2,048 → capped at 1,536 (SM limit) → 100% occupancy
At 40 regs/thread:   65,536 / 40 = 1,638 → capped at 1,536             → 100% occupancy
At 64 regs/thread:   65,536 / 64 = 1,024 threads                       → 66% occupancy
At 128 regs/thread:  65,536 / 128 = 512 threads                        → 33% occupancy
At 255 regs/thread:  65,536 / 255 = 257 threads (8 warps)              → 16% occupancy
```

The compiler balances two opposing goals:
1. **Use more registers** → fewer instructions (no spills, variables stay fast)
2. **Use fewer registers** → more threads → higher occupancy → better latency hiding

Sometimes the compiler intentionally spills a few values to keep occupancy up. You can override this with `__launch_bounds__`:

```c
// Tell compiler: this kernel will launch with max 256 threads/block,
// and we want at least 6 blocks per SM
__global__ void __launch_bounds__(256, 6) my_kernel(...) {
    // Compiler knows: 6 blocks × 256 threads = 1,536 threads
    // 65,536 / 1,536 = 42 registers per thread budget
    // It will try to fit within 42 registers, spilling if necessary
}
```

Or use `--maxrregcount=N` at compile time to set a hard cap across all kernels.

**How to diagnose:**
1. Compile with `nvcc --ptxas-options=-v` — look for "spill stores" and "spill loads"
2. Check register count — if it's high (>64), occupancy is likely limited
3. Look for large local arrays or deep call stacks in the kernel source
4. If spilling: simplify the kernel, split it into multiple passes, or accept lower occupancy
