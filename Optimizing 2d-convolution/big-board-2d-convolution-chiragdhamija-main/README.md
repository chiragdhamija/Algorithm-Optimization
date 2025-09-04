### Approach: AVX-512–Accelerated 3×3 Convolution with Mapped I/O and Thread Pinning

* **Memory-mapped I/O:** Used `mmap` + `open`/`ftruncate` to map the input bitmap and the output buffer directly into memory, avoiding extra copies and enabling streaming access for large images.
* **SIMD vectorization (AVX-512):** For interior pixels, computed 16-pixel wide convolution stripes at once using `_mm512_loadu_ps`, `_mm512_set1_ps`, and `_mm512_fmadd_ps` to accumulate 9 taps (3×3 kernel) efficiently.
* **Scalar boundary handling:** Processed borders and right-edge remainder (where fewer than 16 columns remain) with scalar loops to maintain correctness.
* **OpenMP parallelization:** Parallelized over image rows with `#pragma omp parallel` and `#pragma omp for schedule(static)` to distribute work evenly across threads.
* **Thread affinity (pinning):** Pinned each OpenMP thread to distinct (even-numbered) CPU cores via `sched_setaffinity` to reduce migration, improve cache locality, and stabilize throughput.
* **Compute layout:** For each interior pixel block, iterated over the 3×3 neighborhood, broadcasting the kernel coefficient and multiplying with the loaded 16-wide input vector, accumulating into a vector register, then storing the 16 results to the output.
* **Compiler/ISA tuning:** Enabled high-level optimizations and target features with `#pragma GCC optimize("O3,unroll-loops")` and `#pragma GCC target("avx512f,...)` to ensure use of wide SIMD and aggressive loop unrolling.
