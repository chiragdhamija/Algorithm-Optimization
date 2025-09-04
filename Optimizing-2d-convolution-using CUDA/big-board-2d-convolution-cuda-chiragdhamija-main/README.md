### Approach: CUDA-Accelerated 3×3 Convolution with Kernel Caching and Tiled Execution

* **Memory-mapped input:** Used `mmap` to map the bitmap file into host memory, avoiding extra copies before uploading to the GPU.
* **Kernel caching in shared memory:** Loaded the 3×3 filter into fast on-chip `__shared__` memory once per block to minimize repeated global-memory reads of coefficients.
* **Tiled parallelism (32×32 blocks):** Launched a 2D grid of 32×32 threads; each thread computes one output pixel, covering the image with tiles for high parallel occupancy.
* **Boundary-safe accumulation:** Per-thread 3×3 neighborhood accumulation with bounds checks to correctly handle image edges.
* **Coalesced global accesses (inputs/outputs):** Threads in a warp access adjacent elements (`row*numCols + col`), improving global memory throughput during input reads and output writes.
* **Explicit H2D/D2H transfers:** Copied image and kernel to device (`cudaMemcpy`), executed the kernel, then copied results back to host for file output.
* **Resource setup:** Allocated device buffers for input, output, and kernel; configured grid/block dimensions as `gridDim = ceil(W/32)×ceil(H/32)`, `blockDim = 32×32` to balance parallelism and simplicity.
