#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx512f,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <cstdint>
#include <filesystem>
#include <string>
#include <immintrin.h> 
#include <fcntl.h>     
#include <unistd.h>    
#include <sys/mman.h>  
#include <omp.h>       

namespace solution
{
    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols)
    {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
        int bitmap_fd = open(bitmap_path.c_str(), O_RDONLY);
        float *img = (float *)mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_READ, MAP_SHARED, bitmap_fd, 0);

        int sol_fd = open(sol_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        ftruncate(sol_fd, sizeof(float) * num_rows * num_cols);

        float *output = (float *)mmap(NULL, sizeof(float) * num_rows * num_cols, PROT_READ | PROT_WRITE, MAP_SHARED, sol_fd, 0);

        // Set number of threads
        omp_set_num_threads(48);

        #pragma omp parallel
        {
            // Get thread ID
            int thread_id = omp_get_thread_num();

            // Pin threads to even-numbered cores
            int core_id = thread_id * 2;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core_id, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

            #pragma omp for schedule(static)
            for (int y = 0; y < num_rows; ++y)
            {
                for (int x = 0; x < num_cols; ++x)
                {
                    if (y == 0 || y == num_rows - 1 || x == 0 || x == num_cols - 1 || (num_cols - x) <= 16)
                    {
                        float sum = 0.0f;
                        for (int ky = -1; ky <= 1; ++ky)
                        {
                            for (int kx = -1; kx <= 1; ++kx)
                            {
                                int ny = y + ky;
                                int nx = x + kx;
                                if (ny >= 0 && ny < num_rows && nx >= 0 && nx < num_cols)
                                {
                                    sum += img[ny * num_cols + nx] * kernel[ky + 1][kx + 1];
                                }
                            }
                        }
                        output[y * num_cols + x] = sum;
                    }
                    else
                    {
                        __m512 sum_vec = _mm512_setzero_ps(); 
                        for (int ky = -1; ky <= 1; ++ky)
                        {
                            for (int kx = -1; kx <= 1; ++kx)
                            {
                                __m512 img_vec = _mm512_loadu_ps(&img[(y + ky) * num_cols + (x + kx)]); 
                                __m512 kernel_vec = _mm512_set1_ps(kernel[ky + 1][kx + 1]);             
                                sum_vec = _mm512_fmadd_ps(img_vec, kernel_vec, sum_vec); 
                            }
                        }
                        x=(x+15)%num_cols;
                        _mm512_storeu_ps(&output[y * num_cols + x - 15], sum_vec);
                    }
                }
            }
        }

        return sol_path;
    }
};
