### Approach: Optimized Matrix–Matrix Multiplication with AVX-512 and Blocking

* Used **file-backed memory mapping (mmap + pread)** to efficiently load large input matrices from disk into memory.
* Allocated a **single contiguous memory region** to hold both input matrices and the result, improving locality and reducing allocation overhead.
* Applied **cache blocking (tiling)** with a block size of 32 to enhance cache reuse and minimize cache misses.
* Leveraged **OpenMP parallelization** to distribute block computations across multiple threads, scaling with available CPU cores.
* Employed **AVX-512 intrinsics** (`_mm512_loadu_ps`, `_mm512_fmadd_ps`, `_mm512_storeu_ps`) to process 16 single-precision elements in parallel, enabling wide SIMD acceleration.
* Combined blocking + SIMD + multi-threading to achieve high throughput while keeping memory usage efficient.
