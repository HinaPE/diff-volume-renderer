cmake_minimum_required(VERSION 3.26)
project(dvren_hp LANGUAGES CXX VERSION 0.1.0)

option(DVREN_BUILD_TESTS "build tests" ON)
option(DVREN_BUILD_CPU "build cpu backend" ON)
option(DVREN_BUILD_CUDA "build cuda backend" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (DVREN_BUILD_CUDA)
    enable_language(CUDA OPTIONAL)
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
        endif ()
    else ()
        message(STATUS "CUDA compiler not found, CUDA targets will be skipped.")
        set(DVREN_BUILD_CUDA OFF)
    endif ()
endif ()

set(DVREN_HP_CORE_SOURCES
    src/runtime/hp_runtime.cpp
    src/runtime/api_ray.cpp
    src/cpu/ray_cpu.cpp
)

if (DVREN_BUILD_CUDA)
    list(APPEND DVREN_HP_CORE_SOURCES
        src/cuda/ray_cuda.cu
    )
endif ()

add_library(dvren_hp_core STATIC
    ${DVREN_HP_CORE_SOURCES}
)
target_include_directories(dvren_hp_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(dvren_hp_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
)

if (DVREN_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(dvren_hp_core PUBLIC HP_WITH_CUDA=1)
    target_link_libraries(dvren_hp_core PUBLIC CUDA::cudart)
endif ()

add_executable(hp_runner
    src/runtime/runner.cpp
)
target_link_libraries(hp_runner PRIVATE dvren_hp_core)
