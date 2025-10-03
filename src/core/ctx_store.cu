#include <cuda_runtime.h>
#include "dvren.h"

using namespace dvren;

struct SavedCtx
{
    int width;
    int height;
    int n_steps;
    float* T;
    float* E;
    float* dt;
    float* xyz;
    float* rgb_cache;
    float* sigma_cache;
    FieldProvider field;
};

extern "C" void* dvren_alloc_ctx(size_t bytes)
{
    void* p = nullptr;
    cudaMalloc(&p, bytes);
    return p;
}

extern "C" void dvren_free_ctx(void* p)
{
    if (p) cudaFree(p);
}
