# Combined Optimization Report

## Overview

This report consolidates four optimized implementations across CPU (AVX2/AVX-512 + OpenMP) and GPU (CUDA) for dense linear algebra and image convolution. It summarizes the **workloads**, **approaches**, and **cross-cutting techniques** to highlight how SIMD, blocking, thread-level parallelism, memory mapping, and GPU tiling drive performance.

---

## Workloads

* **W1:** Matrix–Vector multiply (`y = A·x`, 1000×1000, **AVX2, double**)
* **W2:** Matrix–Matrix multiply (`C = A·B`, **AVX-512, float**, blocked + OpenMP)
* **W3:** 3×3 image convolution (**AVX-512, float**, mapped I/O + thread pinning)
* **W4:** 3×3 image convolution (**CUDA, float**, shared-mem kernel caching)

---

## Approaches (by workload)

### W1 — Matrix–Vector Multiplication with AVX2

* **SIMD vectorization:** AVX2 intrinsics process 4 doubles per iteration.
* **Memory locality:** Row-major streaming of `A`; `x` reused from cache.
* **Loop structure:** Unrolled by vector width (4); vector reduction to scalar.
* **Measurement:** Runtime + derived **GFLOPS** from `2·N·N` FLOPs.

---

### W2 — Matrix–Matrix Multiplication with AVX-512 and Blocking

* **Mapped I/O:** `mmap` + `pread` for large matrices from disk.
* **Contiguous arena:** Single allocation for `A`, `B`, `C` to improve locality.
* **Cache blocking:** Tile size 32 to increase reuse and reduce misses.
* **Parallelism:** **OpenMP** over tiles/rows for multi-core scaling.
* **Wide SIMD:** AVX-512 (`loadu`, `fmadd`, `storeu`) for 16-wide float lanes.
* **Composite gain:** Blocking + SIMD + threading for high throughput.

---

### W3 — AVX-512–Accelerated 3×3 Convolution (CPU)

* **Mapped I/O:** Directly map input/output buffers to memory.
* **SIMD interior path:** 16-pixel stripes via AVX-512 (`loadu`, `set1`, `fmadd`).
* **Scalar edges:** Handle borders/right-edge remainder for correctness.
* **OpenMP:** Static scheduling across rows.
* **Thread pinning:** `sched_setaffinity` to reduce migration and stabilize caches.
* **Compiler/ISA:** `-O3`, unrolling, `-mavx512f` pragmas/targets enabled.

---

### W4 — CUDA-Accelerated 3×3 Convolution (GPU)

* **Host I/O:** `mmap` input, then explicit **H2D/D2H** transfers.
* **Shared-mem kernel caching:** Load 3×3 filter into `__shared__` once per block.
* **Tiled launch:** 2D grid of **32×32** threads; one output pixel per thread.
* **Coalesced access:** Row-major indexing for efficient global loads/stores.
* **Boundary checks:** Per-thread safe accumulation at edges.
* **Resource setup:** Device buffers for input/output/kernel; simple grid config.

---

## Cross-Cutting Optimization Themes

* **Data Parallelism:**

  * CPU: AVX2/AVX-512 vector lanes (4× FP64, 16× FP32).
  * GPU: Thousands of threads + fast shared memory.

* **Cache/Memory Efficiency:**

  * Blocking (W2) and interior-only SIMD (W3) improve reuse.
  * `mmap` reduces copies for large I/O (W2, W3).
  * CUDA shared memory caches small kernels (W4).

* **Thread-Level Parallelism:**

  * OpenMP (W2, W3) with static scheduling; thread pinning (W3).
  * CUDA grids/blocks expose massive parallelism (W4).

* **Numerical/ISA Choices:**

  * FP64 for mat-vec accuracy (W1); FP32 for high throughput (W2–W4).
  * FMA paths used where available (AVX-512 FMAs, CUDA FMAs).

* **Measurement & Validation:**

  * Report **GFLOPS**/pixel-throughput; time only the kernel region.
  * Validate against scalar/baseline results, especially at borders.

---

## Quick Comparison

| Aspect      | W1: AVX2 Mat-Vec    | W2: AVX-512 Mat-Mat        | W3: AVX-512 Conv                     | W4: CUDA Conv              |
| ----------- | ------------------- | -------------------------- | ------------------------------------ | -------------------------- |
| Parallelism | 4-wide FP64 SIMD    | 16-wide FP32 SIMD + OpenMP | 16-wide FP32 SIMD + OpenMP + pinning | 32×32 tiles, many blocks   |
| Data Reuse  | Vector `x` in cache | Tile/block reuse of A/B    | Interior stripes reuse via registers | Kernel in shared mem       |
| Memory/I-O  | In-mem arrays       | `mmap` + single arena      | `mmap` in/out                        | `mmap` host + H2D/D2H      |
| Boundaries  | N/A                 | N/A                        | Scalar edges/remainder               | Per-thread boundary checks |
| Best For    | Bandwidth-bound M·V | High-throughput GEMM       | CPU conv on large images             | GPU conv / massive images  |

---


## One-Line Summaries

* **W1:** Exploits AVX2 SIMD and cache-friendly access to speed up mat–vec; measures GFLOPS from kernel-only timing.
* **W2:** Combines mapped I/O, cache blocking, AVX-512, and OpenMP to approach GEMM roofline.
* **W3:** Uses AVX-512 stripes + OpenMP + thread pinning; scalar edges ensure correctness.
* **W4:** Tiles the image on GPU; caches 3×3 kernel in shared memory for high throughput and coalesced access.

