#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
// best solution till now
namespace solution {
    std::string compute(const std::string& m1_path, const std::string& m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);

        int m1_fd = open(m1_path.c_str(), O_RDONLY);
        int m2_fd = open(m2_path.c_str(), O_RDONLY);

        // Calculate total memory needed
        size_t total_mem_size = sizeof(float) * (n * k + k * m + n * m);

        // Map memory for all arrays
        auto memory = static_cast<float*>(mmap(NULL, total_mem_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));

        // Populate m1, m2, and result pointers
        auto m1 = memory;
        auto m2 = m1 + n * k;
        auto result = m2 + k * m;

        // Populate m1 and m2
        pread(m1_fd, m1, sizeof(float) * n * k, 0);
        pread(m2_fd, m2, sizeof(float) * k * m, 0);

        // Perform matrix multiplication
        constexpr int BLOCK_SIZE = 32; // Define your block size
        #pragma omp parallel for
        for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
            for (int bj = 0; bj < m; bj += BLOCK_SIZE) {
                for (int bk = 0; bk < k; bk += BLOCK_SIZE) {
                    for (int i = bi; i < std::min(n, bi + BLOCK_SIZE); ++i) {
                        for (int j = bj; j < std::min(m, bj + BLOCK_SIZE); j += 16) {
                            __m512 c = _mm512_loadu_ps(&result[i * m + j]);
                            for (int l = bk; l < std::min(k, bk + BLOCK_SIZE); ++l) {
                                float temp = m1[i * k + l];
                                __m512 a = _mm512_set1_ps(temp);
                                __m512 b = _mm512_loadu_ps(&m2[l * m + j]);
                                c = _mm512_fmadd_ps(a, b, c);
                            }
                            _mm512_storeu_ps(&result[i * m + j], c);
                        }
                    }
                }
            }
        }

        // Write result to file
        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * n * m);

        // Unmap memory
        // munmap(memory, total_mem_size);

        return sol_path;
    }
};