cmake_minimum_required(VERSION 3.26)
if (WIN32)
    include(cmake/setup_vcpkg.cmake)
endif ()
project(dvren LANGUAGES CXX VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
endif ()

option(DVREN_ENABLE_CUDA "Enable CUDA components" ON)
option(DVREN_ENABLE_HOTPATH_TESTS "Build and run hotpath tests" OFF)
option(DVREN_BUILD_PROJECT_TESTS "Build dvren tests" ON)
option(DVREN_BUILD_APPS "Build dvren applications" ON)

if (DVREN_ENABLE_CUDA)
    enable_language(CUDA OPTIONAL)
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    else ()
        message(WARNING "CUDA compiler not found, disabling CUDA components")
        set(DVREN_ENABLE_CUDA OFF)
    endif ()
endif ()

set(DVREN_BUILD_CPU ON CACHE BOOL "hotpath cpu backend" FORCE)
set(DVREN_BUILD_CUDA ${DVREN_ENABLE_CUDA} CACHE BOOL "hotpath cuda backend" FORCE)
set(DVREN_BUILD_TESTS ${DVREN_ENABLE_HOTPATH_TESTS} CACHE BOOL "hotpath tests" FORCE)

add_subdirectory(hotpath)

add_library(dvren_core STATIC
    src/core/status.cpp
    src/core/context.cpp
    src/core/plan.cpp
    src/core/tensor_utils.cpp
)
target_include_directories(dvren_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(dvren_core PUBLIC dvren::hotpath)

add_library(dvren_fields STATIC
    src/fields/dense_grid.cpp
)
target_include_directories(dvren_fields PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(dvren_fields PUBLIC dvren_core)

add_library(dvren_renderer STATIC
    src/render/renderer.cpp
)
add_library(dvren::render ALIAS dvren_renderer)
target_include_directories(dvren_renderer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(dvren_renderer PUBLIC dvren_fields)

if (DVREN_BUILD_APPS)
    add_executable(dvren_render
        apps/dvren_render/main.cpp
    )
    target_link_libraries(dvren_render PRIVATE dvren::render)
endif ()

if (DVREN_BUILD_PROJECT_TESTS)
    enable_testing()
    add_executable(dvren_core_tests
        tests/core/test_core.cpp
    )
    target_link_libraries(dvren_core_tests PRIVATE dvren::render)
    add_test(NAME dvren_core_tests COMMAND dvren_core_tests)
endif ()
