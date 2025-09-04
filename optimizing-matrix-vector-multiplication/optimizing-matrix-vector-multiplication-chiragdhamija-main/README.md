### Approach: Matrix–Vector Multiplication with AVX2

* Implemented **SIMD vectorization** using AVX2 intrinsics to process 4 double-precision elements in parallel.
* Leveraged **contiguous row-major access** for the matrix and **cache reuse** of the vector to improve memory efficiency.
* Applied **loop unrolling by vector width (4)** to reduce loop overhead.
* Used a **reduction step** to accumulate partial sums from SIMD registers into the final result.
* Measured performance in **GFLOPS** based on FLOP count and execution time.
