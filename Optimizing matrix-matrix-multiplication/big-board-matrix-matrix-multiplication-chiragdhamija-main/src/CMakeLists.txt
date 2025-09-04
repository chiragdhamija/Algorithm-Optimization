add_library(studentlib main.cpp)

find_package(OpenMP REQUIRED)


target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})

target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_options(studentlib PRIVATE -Ofast -march=native  -fopenmp) 

target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)