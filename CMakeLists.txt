cmake_minimum_required(VERSION 3.26)
if (WIN32)
    include(cmake/setup_vcpkg.cmake)
endif ()
project(dvren LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

add_executable(dvren test/test_grad.cpp ${DVREN_CORE_SOURCES})
target_include_directories(dvren PRIVATE include)
include(cmake/setup_vdb.cmake)
use_vdb(dvren)
include(cmake/setup_tcnn.cmake)
use_tcnn(dvren)