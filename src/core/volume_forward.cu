#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include "dvren.h"

using namespace dvren;

struct DevBuf
{
    float* pix;
    float* T;
    float* E;
    float* xyz;
    float* sigma;
    float* rgb;
};

__device__ inline void ray_at(const float* o, const float* d, float t, float* p)
{
    p[0] = o[0] + t * d[0];
    p[1] = o[1] + t * d[1];
    p[2] = o[2] + t * d[2];
}

extern "C" __global__ void k_forward(const float* origins, const float* dirs, const float* tmin, int n_steps, float dt, int W, int H, float sigma_scale, float stop_thresh, float* out_rgb, FieldForwardFn field_f, void* field_u)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    const float* o = origins + idx * 3;
    const float* d = dirs + idx * 3;
    float t0 = tmin[idx];
    float T = 1.f;
    float3 C = make_float3(0.f, 0.f, 0.f);
    extern __shared__ float sh[];
    float* xyz = sh;
    float* sigma = xyz + n_steps * 3;
    float* rgb = sigma + n_steps;
    for (int i = 0; i < n_steps; ++i)
    {
        float t = t0 + dt * i;
        float* p = xyz + i * 3;
        p[0] = o[0] + t * d[0];
        p[1] = o[1] + t * d[1];
        p[2] = o[2] + t * d[2];
    }
    field_f(xyz, n_steps, sigma, rgb, field_u);
    for (int i = 0; i < n_steps; ++i)
    {
        float a = 1.f - expf(-sigma_scale * sigma[i] * dt);
        float w = T * a;
        C.x += w * rgb[i * 3 + 0];
        C.y += w * rgb[i * 3 + 1];
        C.z += w * rgb[i * 3 + 2];
        T *= 1.f - a;
        if (T < stop_thresh) break;
    }
    out_rgb[idx * 3 + 0] = C.x;
    out_rgb[idx * 3 + 1] = C.y;
    out_rgb[idx * 3 + 2] = C.z;
}

namespace
{
    struct HostCtx
    {
        int width;
        int height;
        int n_steps;
        float dt;
        float sigma_scale;
        float stop_thresh;
        FieldProvider field;
    };
}

bool dvren::volume_forward(const RaysSOA& rays, const SamplingSpec& spec, const FieldProvider& field, const RenderFlags& flags, ForwardOutputs& out)
{
    int W = rays.width;
    int H = rays.height;
    int n = W * H;
    float* d_img = nullptr;
    cudaMalloc(&d_img, sizeof(float) * n * 3);
    dim3 bs(16, 16, 1);
    dim3 gs((W + bs.x - 1) / bs.x, (H + bs.y - 1) / bs.y, 1);
    size_t sh = spec.n_steps * (3 + 1 + 3) * sizeof(float);
    HostCtx hctx;
    hctx.width = W;
    hctx.height = H;
    hctx.n_steps = spec.n_steps;
    hctx.dt = spec.dt;
    hctx.sigma_scale = spec.sigma_scale;
    hctx.stop_thresh = spec.stop_thresh;
    hctx.field = field;
    k_forward<<<gs, bs, sh>>>(rays.origins, rays.dirs, rays.tmin, spec.n_steps, spec.dt, W, H, spec.sigma_scale, spec.stop_thresh, d_img, field.fwd, field.user);
    cudaDeviceSynchronize();
    out.image.resize(n * 3);
    cudaMemcpy(out.image.data(), d_img, sizeof(float) * n * 3, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    std::vector<uint8_t> ctx(sizeof(HostCtx));
    std::memcpy(ctx.data(), &hctx, sizeof(HostCtx));
    void* d_ctx = nullptr;
    cudaMalloc(&d_ctx, ctx.size());
    cudaMemcpy(d_ctx, ctx.data(), ctx.size(), cudaMemcpyHostToDevice);
    out.saved_ctx = d_ctx;
    out.saved_ctx_bytes = ctx.size();
    return true;
}
