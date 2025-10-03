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
        const float* d_origins;
        const float* d_dirs;
        const float* d_tmin;
    };

    struct GridDenseDev
    {
        int nx, ny, nz;
        float bmin[3];
        float bmax[3];
        float* sigma;
        float* rgb;
        float* g_sigma;
        float* g_rgb;
    };

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
        float rc = lerp(r0, r1, tz);
        float g000 = g->rgb[i000 * 3 + 1], g100 = g->rgb[i100 * 3 + 1], g010 = g->rgb[i010 * 3 + 1], g110 = g->rgb[i110 * 3 + 1];
        float g001 = g->rgb[i001 * 3 + 1], g101 = g->rgb[i101 * 3 + 1], g011 = g->rgb[i011 * 3 + 1], g111 = g->rgb[i111 * 3 + 1];
        float g00 = lerp(g000, g100, tx), g10 = lerp(g010, g110, tx), g01 = lerp(g001, g101, tx), g11 = lerp(g011, g111, tx);
        float gc = lerp(lerp(g00, g10, ty), lerp(g01, g11, ty), tz);
        float b000 = g->rgb[i000 * 3 + 2], b100 = g->rgb[i100 * 3 + 2], b010 = g->rgb[i010 * 3 + 2], b110 = g->rgb[i110 * 3 + 2];
        float b001 = g->rgb[i001 * 3 + 2], b101 = g->rgb[i101 * 3 + 2], b011 = g->rgb[i011 * 3 + 2], b111 = g->rgb[i111 * 3 + 2];
        float b00 = lerp(b000, b100, tx), b10 = lerp(b010, b110, tx), b01 = lerp(b001, b101, tx), b11 = lerp(b011, b111, tx);
        float bc = lerp(lerp(b00, b10, ty), lerp(b01, b11, ty), tz);
        rgb[0] = rc;
        rgb[1] = gc;
        rgb[2] = bc;
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
        double T = 1.0;
        double Cr = 0.0, Cg = 0.0, Cb = 0.0;
        for (int i = 0; i < n_steps; ++i)
        {
            float t = t0 + dt * i;
            float p[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
            float sig;
            float col[3];
            grid_trilinear(g, p, sig, col);
            double a = 1.0 - exp(-(double)sigma_scale * (double)sig * (double)dt);
            double w = T * a;
            Cr += w * (double)col[0];
            Cg += w * (double)col[1];
            Cb += w * (double)col[2];
            T *= 1.0 - a;
            if (T < (double)stop_thresh) break;
        }
        out_rgb[idx * 3 + 0] = (float)Cr;
        out_rgb[idx * 3 + 1] = (float)Cg;
        out_rgb[idx * 3 + 2] = (float)Cb;
    }
}

bool dvren::volume_forward(const RaysSOA& rays, const SamplingSpec& spec, const FieldProvider& field, const RenderFlags& flags, ForwardOutputs& out)
{
    int W = rays.width, H = rays.height, n = W * H;
    float* d_img = nullptr;
    float *d_o = nullptr, *d_d = nullptr, *d_tmin = nullptr;
    cudaMalloc(&d_img, sizeof(float) * n * 3);
    cudaMalloc(&d_o, sizeof(float) * n * 3);
    cudaMalloc(&d_d, sizeof(float) * n * 3);
    cudaMalloc(&d_tmin, sizeof(float) * n);
    cudaMemcpy(d_o, rays.origins, sizeof(float) * n * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, rays.dirs, sizeof(float) * n * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmin, rays.tmin, sizeof(float) * n, cudaMemcpyHostToDevice);
    dim3 bs(16, 16, 1);
    dim3 gs((W + bs.x - 1) / bs.x, (H + bs.y - 1) / bs.y, 1);
    if (field.type == Field_Grid_Dense)
    {
        auto g = reinterpret_cast<const GridDenseDev*>(field.device_ctx);
        k_forward_grid_dense<<<gs,bs>>>(d_o, d_d, d_tmin, spec.n_steps, spec.dt, W, H, spec.sigma_scale, spec.stop_thresh, d_img, g);
    }
    else
    {
        cudaMemset(d_img, 0, sizeof(float) * n * 3);
    }
    cudaDeviceSynchronize();
    out.image.resize(n * 3);
    cudaMemcpy(out.image.data(), d_img, sizeof(float) * n * 3, cudaMemcpyDeviceToHost);
    HostCtx hctx{};
    hctx.width = W;
    hctx.height = H;
    hctx.n_steps = spec.n_steps;
    hctx.dt = spec.dt;
    hctx.sigma_scale = spec.sigma_scale;
    hctx.stop_thresh = spec.stop_thresh;
    hctx.type = field.type;
    hctx.device_ctx = field.device_ctx;
    hctx.d_origins = d_o;
    hctx.d_dirs = d_d;
    hctx.d_tmin = d_tmin;
    void* d_ctx = nullptr;
    cudaMalloc(&d_ctx, sizeof(HostCtx));
    cudaMemcpy(d_ctx, &hctx, sizeof(HostCtx), cudaMemcpyHostToDevice);
    out.saved_ctx = d_ctx;
    out.saved_ctx_bytes = sizeof(HostCtx);
    cudaFree(d_img);
    return true;
}
