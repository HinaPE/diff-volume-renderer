include(FetchContent)

FetchContent_Declare(tcnn
        GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
        GIT_TAG v2.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(tcnn)
target_link_libraries(dvren PRIVATE tiny-cuda-nn)

function(use_tcnn target)
    if (TARGET tiny-cuda-nn)
        target_link_libraries(${target} PRIVATE tiny-cuda-nn)
    else ()
        message(FATAL_ERROR "Cannot find tiny-cuda-nn target")
    endif ()
endfunction()