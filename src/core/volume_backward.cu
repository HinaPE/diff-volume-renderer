#include <cuda_runtime.h>
#include <cmath>
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

    __device__ inline void grid_weights(const GridDenseDev* g, const float* p, int idxs[8], float w[8])
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
        float wx0 = 1.f - tx, wx1 = tx, wy0 = 1.f - ty, wy1 = ty, wz0 = 1.f - tz, wz1 = tz;
        idxs[0] = i000;
        idxs[1] = i100;
        idxs[2] = i010;
        idxs[3] = i110;
        idxs[4] = i001;
        idxs[5] = i101;
        idxs[6] = i011;
        idxs[7] = i111;
        w[0] = wx0 * wy0 * wz0;
        w[1] = wx1 * wy0 * wz0;
        w[2] = wx0 * wy1 * wz0;
        w[3] = wx1 * wy1 * wz0;
        w[4] = wx0 * wy0 * wz1;
        w[5] = wx1 * wy0 * wz1;
        w[6] = wx0 * wy1 * wz1;
        w[7] = wx1 * wy1 * wz1;
    }

    __device__ inline void sample_sigma_rgb(const GridDenseDev* g, const float* p, float& sig, float col[3])
    {
        int idxs[8];
        float w[8];
        grid_weights(g, p, idxs, w);
        float s = 0.f, r = 0.f, gg = 0.f, b = 0.f;
        for (int k = 0; k < 8; ++k)
        {
            int i = idxs[k];
            float wk = w[k];
            s += wk * g->sigma[i];
            r += wk * g->rgb[i * 3 + 0];
            gg += wk * g->rgb[i * 3 + 1];
            b += wk * g->rgb[i * 3 + 2];
        }
        sig = s;
        col[0] = r;
        col[1] = gg;
        col[2] = b;
    }

    __global__ void k_backward_grid_dense(
        const float* origins, const float* dirs, const float* tmin,
        int W, int H, int n_steps, float dt, float ksig, float stop_thresh,
        const float* dL_dimg, GridDenseDev* g)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= H) return;
        int pix = y * W + x;
        const float* o = origins + pix * 3;
        const float* d = dirs + pix * 3;
        float t0 = tmin[pix];
        float3 gC = make_float3(dL_dimg[pix * 3 + 0], dL_dimg[pix * 3 + 1], dL_dimg[pix * 3 + 2]);
        const int S = 1024;
        int steps = n_steps;
        if (steps > S) steps = S;
        float a_buf[S];
        float T_buf[S];
        int used = 0;
        float T = 1.f;
        for (int i = 0; i < steps; ++i)
        {
            float t = t0 + dt * i;
            float p[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
            float sig;
            float col[3];
            sample_sigma_rgb(g, p, sig, col);
            float a = 1.f - expf(-ksig * sig * dt);
            a_buf[i] = a;
            T_buf[i] = T;
            T *= 1.f - a;
            used = i + 1;
            if (T < stop_thresh) break;
        }
        float dT_next = 0.f;
        for (int i = used - 1; i >= 0; --i)
        {
            float T_i = T_buf[i];
            float a = a_buf[i];
            float t = t0 + dt * i;
            float p[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
            float sig;
            float col[3];
            sample_sigma_rgb(g, p, sig, col);
            float w_i = T_i * a;
            float3 dL_dc = make_float3(gC.x * w_i, gC.y * w_i, gC.z * w_i);
            float dL_dw = gC.x * col[0] + gC.y * col[1] + gC.z * col[2];
            float dL_dTi = dL_dw * a + dT_next * (1.f - a);
            float dL_dai = dL_dw * T_i + dT_next * (-T_i);
            float dai_dsigma = ksig * dt * (1.f - a);
            float dL_dsigma = dL_dai * dai_dsigma;
            int idxs[8];
            float ww[8];
            grid_weights(g, p, idxs, ww);
            for (int k = 0; k < 8; ++k)
            {
                int vi = idxs[k];
                float wk = ww[k];
                atomicAdd(&g->g_sigma[vi], dL_dsigma * wk);
                atomicAdd(&g->g_rgb[vi * 3 + 0], dL_dc.x * wk);
                atomicAdd(&g->g_rgb[vi * 3 + 1], dL_dc.y * wk);
                atomicAdd(&g->g_rgb[vi * 3 + 2], dL_dc.z * wk);
            }
            dT_next = dL_dTi;
        }
    }
}

bool dvren::volume_backward(void* saved_ctx, size_t saved_ctx_bytes, const float* dL_dimage, int width, int height, const FieldProvider& field)
{
    if (field.type != Field_Grid_Dense) return false;
    HostCtx h{};
    cudaMemcpy(&h, saved_ctx, sizeof(HostCtx), cudaMemcpyDeviceToHost);
    if (h.width != width || h.height != height) return false;
    auto g_dev = reinterpret_cast<GridDenseDev*>(field.device_ctx);
    GridDenseDev g_host{};
    cudaMemcpy(&g_host, g_dev, sizeof(GridDenseDev), cudaMemcpyDeviceToHost);
    if (!g_host.g_sigma || !g_host.g_rgb) return false;
    float* d_grad = nullptr;
    size_t n = static_cast<size_t>(width) * height * 3;
    cudaMalloc(&d_grad, sizeof(float) * n);
    cudaMemcpy(d_grad, dL_dimage, sizeof(float) * n, cudaMemcpyHostToDevice);
    dim3 bs(16, 16, 1);
    dim3 gs((width + bs.x - 1) / bs.x, (height + bs.y - 1) / bs.y, 1);
    k_backward_grid_dense<<<gs,bs>>>(
        h.d_origins, h.d_dirs, h.d_tmin,
        width, height, h.n_steps, h.dt, h.sigma_scale, h.stop_thresh,
        d_grad, g_dev);
    cudaDeviceSynchronize();
    cudaFree(d_grad);
    return true;
}
