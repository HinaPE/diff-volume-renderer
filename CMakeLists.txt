cmake_minimum_required(VERSION 3.26)
project(dvren LANGUAGES CXX CUDA VERSION 0.1.0)

option(DVREN_SHARED "shared lib" OFF)
option(DVREN_BUILD_EXAMPLES "examples" OFF)
option(DVREN_BUILD_TORCH "torch bindings" OFF)
option(DVREN_WITH_TCNN "tiny-cuda-nn" ON)
option(DVREN_WITH_VDB "OpenVDB" ON)
option(DVREN_USE_GRAPH "cuda graph" ON)
option(DVREN_FP16_FIELD "fp16 field forward" ON)
option(DVREN_RETAIN_RGB "retain rgb in ctx" OFF)
option(DVREN_USE_BITFIELD "bitfield skipping" OFF)
option(DVREN_DETERMINISTIC "deterministic" OFF)
option(DVREN_FAST_MATH "cuda fast math" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
endif()

include(FetchContent)

set(DVREN_CORE_SOURCES
        src/core/sampler.cu
        src/core/volume_forward.cu
        src/core/volume_backward.cu
        src/core/ctx_store.cu
        src/core/compositor.cu
)

add_executable(dvren main.cpp ${DVREN_CORE_SOURCES})

if(DVREN_WITH_TCNN)
    FetchContent_Declare(tcnn
            GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
            GIT_TAG v2.0
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(tcnn)
    target_link_libraries(dvren PRIVATE tiny-cuda-nn)
    target_compile_definitions(dvren PUBLIC DVREN_WITH_TCNN=1)
endif()

if(DVREN_WITH_VDB)
    FetchContent_Declare(zlib
            GIT_REPOSITORY https://github.com/madler/zlib.git
            GIT_TAG v1.3.1
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(zlib)

    set(TBB_VERSION v2022.2.0 CACHE STRING "oneTBB version")
    set(TBB_TEST OFF CACHE BOOL "" FORCE)
    set(TBB_STRICT OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(oneTBB URL https://github.com/uxlfoundation/oneTBB/archive/refs/tags/${TBB_VERSION}.tar.gz DOWNLOAD_EXTRACT_TIMESTAMP OFF)
    FetchContent_MakeAvailable(oneTBB)
    if (NOT TARGET TBB::tbb AND TARGET tbb)
        add_library(TBB::tbb ALIAS tbb)
    endif ()
    if (DEFINED oneTBB_BINARY_DIR)
        if (NOT DEFINED TBB_DIR OR NOT TBB_DIR)
            set(TBB_DIR "${oneTBB_BINARY_DIR}" CACHE PATH "Path to oneTBB build directory for find_package(TBB)" FORCE)
        endif ()
        list(FIND CMAKE_PREFIX_PATH "${oneTBB_BINARY_DIR}" _tbb_prefix_index)
        if (_tbb_prefix_index EQUAL -1)
            list(APPEND CMAKE_PREFIX_PATH "${oneTBB_BINARY_DIR}")
        endif ()
    endif ()
    if (DEFINED oneTBB_BINARY_DIR)
        set(_TBB_FAKE_CONFIG "${oneTBB_BINARY_DIR}/TBBConfig.cmake")
        if (NOT EXISTS "${_TBB_FAKE_CONFIG}")
            file(WRITE "${_TBB_FAKE_CONFIG}" "if(NOT TARGET TBB::tbb)\n" "  if(TARGET tbb)\n" "    add_library(TBB::tbb ALIAS tbb)\n" "  else()\n" "    message(FATAL_ERROR 'Expected tbb target not found when loading TBBConfig.cmake')\n" "  endif()\n" "endif()\n" "set(TBB_FOUND TRUE)\n" "set(TBB_VERSION ${TBB_VERSION})\n")
        endif ()
    endif ()

    # ---------- setup Boost here START ----------
    set(BOOST_PREBUILT_URL "https://archives.boost.io/release/1.89.0/binaries/boost_1_89_0-bin-msvc-all-32-64.7z" CACHE STRING "")
    set(BOOST_PREBUILT_ROOT "${CMAKE_BINARY_DIR}/_deps/boost-prebuilt" CACHE PATH "")
    set(SEVEN_ZIP_EXE "C:/Program Files/7-Zip/7z.exe" CACHE STRING "7z executable path")
    message(STATUS "Using 7z executable: ${SEVEN_ZIP_EXE}")

    file(MAKE_DIRECTORY "${BOOST_PREBUILT_ROOT}")
    set(_boost_7z "${BOOST_PREBUILT_ROOT}/boost_1_89_0.7z")

    if(NOT EXISTS "${BOOST_PREBUILT_ROOT}/boost_1_89_0")
        if(NOT EXISTS "${_boost_7z}")
            file(DOWNLOAD "${BOOST_PREBUILT_URL}" "${_boost_7z}" SHOW_PROGRESS)
        endif()
        execute_process(COMMAND ${SEVEN_ZIP_EXE} x -y "${_boost_7z}" -o${BOOST_PREBUILT_ROOT}
                RESULT_VARIABLE _7z_r)
        if(NOT _7z_r EQUAL 0)
            message(FATAL_ERROR "failed to extract boost with 7z, set SEVEN_ZIP_EXE to your 7z path")
        endif()
    endif()

    set(_BOOST_DIR "${BOOST_PREBUILT_ROOT}/boost_1_89_0")

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_ARCH "64")
    else()
        set(_ARCH "32")
    endif()

    if(MSVC_TOOLSET_VERSION)
        math(EXPR _MSVC_MAJ "${MSVC_TOOLSET_VERSION} / 10")
        math(EXPR _MSVC_MIN "${MSVC_TOOLSET_VERSION} % 10")
        set(_MSVC_TAG "${_MSVC_MAJ}.${_MSVC_MIN}")
    else()
        set(_MSVC_TAG "14.3")
    endif()

    set(BOOST_LIBRARYDIR "${_BOOST_DIR}/lib${_ARCH}-msvc-${_MSVC_TAG}")
    message(STATUS "Using Boost lib dir: ${BOOST_LIBRARYDIR}")

    set(Boost_NO_SYSTEM_PATHS ON)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)
    set(Boost_DIR "${BOOST_LIBRARYDIR}/cmake/Boost-1.89.0")
    find_package(Boost 1.89 REQUIRED COMPONENTS iostreams)
    # ---------- setup Boost here END ----------

    set(OPENVDB_BUILD_NANOVDB ON CACHE BOOL "")
    set(OPENVDB_BUILD_BINARIES OFF CACHE BOOL "")
    set(OPENVDB_BUILD_UNITTESTS OFF CACHE BOOL "")
    set(OPENVDB_BUILD_PYTHON_MODULE OFF CACHE BOOL "")
    set(OPENVDB_BUILD_DOCS OFF CACHE BOOL "")
    set(OPENVDB_BUILD_AX OFF CACHE BOOL "")
    set(USE_ZLIB ON CACHE BOOL "")
    set(USE_TBB ON CACHE BOOL "")
    set(USE_BLOSC OFF CACHE BOOL "")
    set(OPENVDB_USE_BLOSC OFF CACHE BOOL "")

    FetchContent_Declare(openvdb
            GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openvdb.git
            GIT_TAG v12.1.1
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(openvdb)

    target_link_libraries(dvren PUBLIC openvdb TBB::tbb ZLIB::ZLIB Boost::system Boost::iostreams)
    target_include_directories(dvren PUBLIC ${openvdb_SOURCE_DIR}/nanovdb)
    target_compile_definitions(dvren PUBLIC DVREN_HAS_OPENVDB=1 NANOVDB_USE_CUDA=1)
endif()
