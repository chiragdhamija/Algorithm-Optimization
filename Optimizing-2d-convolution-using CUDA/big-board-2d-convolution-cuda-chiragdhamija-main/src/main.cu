#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace solution {

    __global__ void convolution2D(const float* __restrict__ input, float* __restrict__ output, int numRows, int numCols, const float* __restrict__ kernel) {
       
        __shared__ float tile[3][3];

       
        if (threadIdx.x < 3 && threadIdx.y < 3) {
            tile[threadIdx.y][threadIdx.x] = kernel[threadIdx.y * 3 + threadIdx.x];
        }
        __syncthreads();

      
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < numRows && col < numCols) {
            float pixel_value = 0.0f;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int r = row + i - 1;
                    int c = col + j - 1;
                    if (r >= 0 && r < numRows && c >= 0 && c < numCols) {
                        pixel_value += input[r * numCols + c] * tile[i][j];
                    }
                }
            }
            output[row * numCols + col] = pixel_value;
        }
    }

    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
        
        int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);
        float* img = reinterpret_cast<float*>(mmap(nullptr, sizeof(float) * num_rows * num_cols, PROT_READ, MAP_PRIVATE, bitmap_fd, 0));

      
        float *d_input, *d_output, *d_kernel;
        cudaMalloc(&d_input, sizeof(float) * num_rows * num_cols);
        cudaMalloc(&d_output, sizeof(float) * num_rows * num_cols);
        cudaMalloc(&d_kernel, sizeof(float) * 3 * 3);

  
        cudaMemcpy(d_input, img, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice);

      
        dim3 blockDim(32, 32);
        dim3 gridDim((num_cols + blockDim.x - 1) / blockDim.x, (num_rows + blockDim.y - 1) / blockDim.y);

        
        convolution2D<<<gridDim, blockDim>>>(d_input, d_output, num_rows, num_cols, d_kernel);
        
        
        float *output = new float[num_rows * num_cols];
        cudaMemcpy(output, d_output, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost);

       
        std::ofstream sol_fs(sol_path, std::ios::binary);
        sol_fs.write(reinterpret_cast<const char*>(output), sizeof(float) * num_rows * num_cols);
        sol_fs.close();

        
        return sol_path;
    }
};
