add_library(studentlib main.cpp)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add compiler flags for OpenMP
target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})

# Include directories
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Compiler options
target_compile_options(studentlib PRIVATE -Ofast -march=native -mavx512f) 

# Link against OpenMP library
target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)