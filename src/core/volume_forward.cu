#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include "dvren.h"

using namespace dvren;

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
        FieldType type;
        void* device_ctx;
    };

    struct GridDenseDev
    {
        int nx, ny, nz;
        float bmin[3];
        float bmax[3];
        float* sigma;
        float* rgb;
    };

    __device__ inline void ray_at3(const float* o, const float* d, float t, float* p)
    {
        p[0] = o[0] + t * d[0];
        p[1] = o[1] + t * d[1];
        p[2] = o[2] + t * d[2];
    }

    __device__ inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
    __device__ inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

    __device__ inline void grid_trilinear(const GridDenseDev* g, const float* p, float& sigma, float rgb[3])
    {
        float ux = (p[0] - g->bmin[0]) / (g->bmax[0] - g->bmin[0]);
        float uy = (p[1] - g->bmin[1]) / (g->bmax[1] - g->bmin[1]);
        float uz = (p[2] - g->bmin[2]) / (g->bmax[2] - g->bmin[2]);
        ux = clampf(ux, 0.f, 1.f);
        uy = clampf(uy, 0.f, 1.f);
        uz = clampf(uz, 0.f, 1.f);
        float x = ux * (g->nx - 1), y = uy * (g->ny - 1), z = uz * (g->nz - 1);
        int x0 = (int)floorf(x), y0 = (int)floorf(y), z0 = (int)floorf(z);
        int x1 = x0 + 1 < g->nx ? x0 + 1 : x0, y1 = y0 + 1 < g->ny ? y0 + 1 : y0, z1 = z0 + 1 < g->nz ? z0 + 1 : z0;
        float tx = x - x0, ty = y - y0, tz = z - z0;
        auto idx = [&](int xi, int yi, int zi) { return (zi * g->ny + yi) * g->nx + xi; };
        int i000 = idx(x0, y0, z0), i100 = idx(x1, y0, z0), i010 = idx(x0, y1, z0), i110 = idx(x1, y1, z0);
        int i001 = idx(x0, y0, z1), i101 = idx(x1, y0, z1), i011 = idx(x0, y1, z1), i111 = idx(x1, y1, z1);
        float s000 = g->sigma[i000], s100 = g->sigma[i100], s010 = g->sigma[i010], s110 = g->sigma[i110];
        float s001 = g->sigma[i001], s101 = g->sigma[i101], s011 = g->sigma[i011], s111 = g->sigma[i111];
        float s00 = lerp(s000, s100, tx), s10 = lerp(s010, s110, tx), s01 = lerp(s001, s101, tx), s11 = lerp(s011, s111, tx);
        float s0 = lerp(s00, s10, ty), s1 = lerp(s01, s11, ty);
        sigma = lerp(s0, s1, tz);
        float r000 = g->rgb[i000 * 3 + 0], r100 = g->rgb[i100 * 3 + 0], r010 = g->rgb[i010 * 3 + 0], r110 = g->rgb[i110 * 3 + 0];
        float r001 = g->rgb[i001 * 3 + 0], r101 = g->rgb[i101 * 3 + 0], r011 = g->rgb[i011 * 3 + 0], r111 = g->rgb[i111 * 3 + 0];
        float r00 = lerp(r000, r100, tx), r10 = lerp(r010, r110, tx), r01 = lerp(r001, r101, tx), r11 = lerp(r011, r111, tx);
        float r0 = lerp(r00, r10, ty), r1 = lerp(r01, r11, ty);
        rgb[0] = lerp(r0, r1, tz);
        float g000 = g->rgb[i000 * 3 + 1], g100 = g->rgb[i100 * 3 + 1], g010 = g->rgb[i010 * 3 + 1], g110 = g->rgb[i110 * 3 + 1];
        float g001 = g->rgb[i001 * 3 + 1], g101 = g->rgb[i101 * 3 + 1], g011 = g->rgb[i011 * 3 + 1], g111 = g->rgb[i111 * 3 + 1];
        float g00 = lerp(g000, g100, tx), g10 = lerp(g010, g110, tx), g01 = lerp(g001, g101, tx), g11 = lerp(g011, g111, tx);
        float g0 = lerp(g00, g10, ty), g1 = lerp(g01, g11, ty);
        rgb[1] = lerp(g0, g1, tz);
        float b000 = g->rgb[i000 * 3 + 2], b100 = g->rgb[i100 * 3 + 2], b010 = g->rgb[i010 * 3 + 2], b110 = g->rgb[i110 * 3 + 2];
        float b001 = g->rgb[i001 * 3 + 2], b101 = g->rgb[i101 * 3 + 2], b011 = g->rgb[i011 * 3 + 2], b111 = g->rgb[i111 * 3 + 2];
        float b00 = lerp(b000, b100, tx), b10 = lerp(b010, b110, tx), b01 = lerp(b001, b101, tx), b11 = lerp(b011, b111, tx);
        float b0 = lerp(b00, b10, ty), b1 = lerp(b01, b11, ty);
        rgb[2] = lerp(b0, b1, tz);
    }

    __global__ void k_forward_grid_dense(const float* origins, const float* dirs, const float* tmin, int n_steps, float dt, int W, int H, float sigma_scale, float stop_thresh, float* out_rgb, const GridDenseDev* g)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= H) return;
        int idx = y * W + x;
        const float* o = origins + idx * 3;
        const float* d = dirs + idx * 3;
        float t0 = tmin[idx];
        float T = 1.f;
        float3 C = make_float3(0, 0, 0);
        for (int i = 0; i < n_steps; ++i)
        {
            float t = t0 + dt * i;
            float p[3];
            ray_at3(o, d, t, p);
            float sig;
            float col[3];
            grid_trilinear(g, p, sig, col);
            float a = 1.f - expf(-sigma_scale * sig * dt);
            float w = T * a;
            C.x += w * col[0];
            C.y += w * col[1];
            C.z += w * col[2];
            T *= 1.f - a;
            if (T < stop_thresh) break;
        }
        out_rgb[idx * 3 + 0] = C.x;
        out_rgb[idx * 3 + 1] = C.y;
        out_rgb[idx * 3 + 2] = C.z;
    }
}

bool dvren::volume_forward(const RaysSOA& rays, const SamplingSpec& spec, const FieldProvider& field, const RenderFlags& flags, ForwardOutputs& out)
{
    int W = rays.width, H = rays.height, n = W * H;
    float* d_img = nullptr;
    cudaMalloc(&d_img, sizeof(float) * n * 3);
    dim3 bs(16, 16, 1);
    dim3 gs((W + bs.x - 1) / bs.x, (H + bs.y - 1) / bs.y, 1);
    if (field.type == Field_Grid_Dense)
    {
        auto g = reinterpret_cast<const GridDenseDev*>(field.device_ctx);
        k_forward_grid_dense<<<gs,bs>>>(rays.origins, rays.dirs, rays.tmin, spec.n_steps, spec.dt, W, H, spec.sigma_scale, spec.stop_thresh, d_img, g);
    }
    else
    {
        cudaMemset(d_img, 0, sizeof(float) * n * 3);
    }
    cudaDeviceSynchronize();
    out.image.resize(n * 3);
    cudaMemcpy(out.image.data(), d_img, sizeof(float) * n * 3, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    HostCtx hctx{};
    hctx.width = W;
    hctx.height = H;
    hctx.n_steps = spec.n_steps;
    hctx.dt = spec.dt;
    hctx.sigma_scale = spec.sigma_scale;
    hctx.stop_thresh = spec.stop_thresh;
    hctx.type = field.type;
    hctx.device_ctx = field.device_ctx;
    void* d_ctx = nullptr;
    cudaMalloc(&d_ctx, sizeof(HostCtx));
    cudaMemcpy(d_ctx, &hctx, sizeof(HostCtx), cudaMemcpyHostToDevice);
    out.saved_ctx = d_ctx;
    out.saved_ctx_bytes = sizeof(HostCtx);
    return true;
}
