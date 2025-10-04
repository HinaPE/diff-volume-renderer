cmake_minimum_required(VERSION 3.26)
if (WIN32)
    include(cmake/setup_vcpkg.cmake)
endif ()
project(dvren LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
endif ()

set(DVREN_CORE_SOURCES
        src/core/sampler.cu
        src/core/volume_forward.cu
        src/core/volume_backward.cu
        src/core/ctx_store.cu
        src/core/compositor.cu
        src/adapters/field_grid.cu
)

function(add_dvren_executable target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE include)
    target_compile_features(${target} PRIVATE cxx_std_23 cuda_std_17)
    include(${CMAKE_SOURCE_DIR}/cmake/setup_vdb.cmake)
    use_vdb(${target})
    include(${CMAKE_SOURCE_DIR}/cmake/setup_tcnn.cmake)
    use_tcnn(${target})
endfunction()
add_dvren_executable(dvren test/test_grad.cpp ${DVREN_CORE_SOURCES})

add_subdirectory(tutorials)
