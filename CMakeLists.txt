cmake_minimum_required(VERSION 3.26)
if (WIN32)
    include(cmake/setup_vcpkg.cmake)
endif ()
project(dvren LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
endif ()

include(FetchContent)

set(DVREN_CORE_SOURCES
        src/core/sampler.cu
        src/core/volume_forward.cu
        src/core/volume_backward.cu
        src/core/ctx_store.cu
        src/core/compositor.cu
)

add_executable(dvren main.cpp ${DVREN_CORE_SOURCES})

# -------------------- OpenVDB --------------------
find_package(OpenVDB CONFIG REQUIRED)
target_link_libraries(dvren PRIVATE OpenVDB::openvdb)
# -------------------- OpenVDB --------------------


# -------------------- tiny-cuda-nn --------------------
FetchContent_Declare(tcnn
        GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
        GIT_TAG v2.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(tcnn)
target_link_libraries(dvren PRIVATE tiny-cuda-nn)
# -------------------- tiny-cuda-nn --------------------