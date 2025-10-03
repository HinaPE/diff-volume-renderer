#include <cuda_runtime.h>
#include "dvren.h"

using namespace dvren;

struct CSR
{
    int* ray_offsets;
    float* t0;
    float* dt;
    int total;
};

extern "C" __global__ void k_build_csr(const float* tmin, const float* tmax, int n_rays, int n_steps, float dt, int* ray_offsets, float* t0, float* dt_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;
    float a = tmin[i];
    float b = tmax[i];
    float len = b - a;
    int steps = n_steps;
    if (len <= 0.f) steps = 0;
    ray_offsets[i] = steps;
    if (i == 0) dt_out[0] = dt;
    t0[i] = a;
    dt_out[i] = dt;
}

bool build_csr(const RaysSOA& rays, const SamplingSpec& spec, CSR& csr)
{
    int n_rays = rays.width * rays.height;
    cudaMalloc(&csr.ray_offsets, sizeof(int) * n_rays);
    cudaMalloc(&csr.t0, sizeof(float) * n_rays);
    cudaMalloc(&csr.dt, sizeof(float) * n_rays);
    int bs = 256;
    int gs = (n_rays + bs - 1) / bs;
    k_build_csr<<<gs, bs>>>(rays.tmin, rays.tmax, n_rays, spec.n_steps, spec.dt, csr.ray_offsets, csr.t0, csr.dt);
    cudaDeviceSynchronize();
    csr.total = n_rays * spec.n_steps;
    return true;
}

void free_csr(CSR& csr)
{
    if (csr.ray_offsets) cudaFree(csr.ray_offsets);
    if (csr.t0) cudaFree(csr.t0);
    if (csr.dt) cudaFree(csr.dt);
    csr = {};
}

extern "C" bool dvren_build_csr(const RaysSOA& rays, const SamplingSpec& spec, CSR& csr) { return build_csr(rays, spec, csr); }
extern "C" void dvren_free_csr(CSR& csr) { free_csr(csr); }
