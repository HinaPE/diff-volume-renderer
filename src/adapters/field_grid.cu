#include <cuda_runtime.h>
#include <cstring>
#include "dvren.h"

using namespace dvren;

namespace
{
    struct GridDenseDev
    {
        int nx, ny, nz;
        float bmin[3];
        float bmax[3];
        float* sigma;
        float* rgb;
    };

    __device__ inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

    __device__ inline void grid_trilinear(const GridDenseDev* g, const float* p, float& sigma, float rgb[3])
    {
        float ux = (p[0] - g->bmin[0]) / (g->bmax[0] - g->bmin[0]);
        float uy = (p[1] - g->bmin[1]) / (g->bmax[1] - g->bmin[1]);
        float uz = (p[2] - g->bmin[2]) / (g->bmax[2] - g->bmin[2]);
        ux = clampf(ux, 0.f, 1.f);
        uy = clampf(uy, 0.f, 1.f);
        uz = clampf(uz, 0.f, 1.f);
        float x = ux * (g->nx - 1);
        float y = uy * (g->ny - 1);
        float z = uz * (g->nz - 1);
        int x0 = (int)floorf(x), y0 = (int)floorf(y), z0 = (int)floorf(z);
        int x1 = x0 + 1 < g->nx ? x0 + 1 : x0;
        int y1 = y0 + 1 < g->ny ? y0 + 1 : y0;
        int z1 = z0 + 1 < g->nz ? z0 + 1 : z0;
        float tx = x - x0, ty = y - y0, tz = z - z0;
        auto idx = [&](int xi, int yi, int zi) { return (zi * g->ny + yi) * g->nx + xi; };
        int i000 = idx(x0, y0, z0), i100 = idx(x1, y0, z0), i010 = idx(x0, y1, z0), i110 = idx(x1, y1, z0);
        int i001 = idx(x0, y0, z1), i101 = idx(x1, y0, z1), i011 = idx(x0, y1, z1), i111 = idx(x1, y1, z1);
        float s000 = g->sigma[i000], s100 = g->sigma[i100], s010 = g->sigma[i010], s110 = g->sigma[i110];
        float s001 = g->sigma[i001], s101 = g->sigma[i101], s011 = g->sigma[i011], s111 = g->sigma[i111];
        float cr000 = g->rgb[i000 * 3 + 0], cg000 = g->rgb[i000 * 3 + 1], cb000 = g->rgb[i000 * 3 + 2];
        float cr100 = g->rgb[i100 * 3 + 0], cg100 = g->rgb[i100 * 3 + 1], cb100 = g->rgb[i100 * 3 + 2];
        float cr010 = g->rgb[i010 * 3 + 0], cg010 = g->rgb[i010 * 3 + 1], cb010 = g->rgb[i010 * 3 + 2];
        float cr110 = g->rgb[i110 * 3 + 0], cg110 = g->rgb[i110 * 3 + 1], cb110 = g->rgb[i110 * 3 + 2];
        float cr001 = g->rgb[i001 * 3 + 0], cg001 = g->rgb[i001 * 3 + 1], cb001 = g->rgb[i001 * 3 + 2];
        float cr101 = g->rgb[i101 * 3 + 0], cg101 = g->rgb[i101 * 3 + 1], cb101 = g->rgb[i101 * 3 + 2];
        float cr011 = g->rgb[i011 * 3 + 0], cg011 = g->rgb[i011 * 3 + 1], cb011 = g->rgb[i011 * 3 + 2];
        float cr111 = g->rgb[i111 * 3 + 0], cg111 = g->rgb[i111 * 3 + 1], cb111 = g->rgb[i111 * 3 + 2];
        auto lerp = [&](float a, float b, float t) { return a + (b - a) * t; };
        float s00 = lerp(s000, s100, tx), s10 = lerp(s010, s110, tx), s01 = lerp(s001, s101, tx), s11 = lerp(s011, s111, tx);
        float s0 = lerp(s00, s10, ty), s1 = lerp(s01, s11, ty);
        sigma = lerp(s0, s1, tz);
        float r00 = lerp(cr000, cr100, tx), r10 = lerp(cr010, cr110, tx), r01 = lerp(cr001, cr101, tx), r11 = lerp(cr011, cr111, tx);
        float r0 = lerp(r00, r10, ty), r1 = lerp(r01, r11, ty);
        rgb[0] = lerp(r0, r1, tz);
        float g00 = lerp(cg000, cg100, tx), g10 = lerp(cg010, cg110, tx), g01 = lerp(cg001, cg101, tx), g11 = lerp(cg011, cg111, tx);
        float g0 = lerp(g00, g10, ty), g1 = lerp(g01, g11, ty);
        rgb[1] = lerp(g0, g1, tz);
        float b00 = lerp(cb000, cb100, tx), b10 = lerp(cb010, cb110, tx), b01 = lerp(cb001, cb101, tx), b11 = lerp(cb011, cb111, tx);
        float b0 = lerp(b00, b10, ty), b1 = lerp(b01, b11, ty);
        rgb[2] = lerp(b0, b1, tz);
    }
}

bool dvren::field_grid_dense_create(const GridDenseDesc& desc, FieldProvider& out)
{
    GridDenseDev h{};
    h.nx = desc.nx;
    h.ny = desc.ny;
    h.nz = desc.nz;
    h.bmin[0] = desc.bbox_min[0];
    h.bmin[1] = desc.bbox_min[1];
    h.bmin[2] = desc.bbox_min[2];
    h.bmax[0] = desc.bbox_max[0];
    h.bmax[1] = desc.bbox_max[1];
    h.bmax[2] = desc.bbox_max[2];
    size_t vox = static_cast<size_t>(desc.nx) * desc.ny * desc.nz;
    cudaMalloc(&h.sigma, vox * sizeof(float));
    cudaMemcpy(h.sigma, desc.host_sigma, vox * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&h.rgb, vox * 3 * sizeof(float));
    cudaMemcpy(h.rgb, desc.host_rgb, vox * 3 * sizeof(float), cudaMemcpyHostToDevice);
    GridDenseDev* d = nullptr;
    cudaMalloc(&d, sizeof(GridDenseDev));
    cudaMemcpy(d, &h, sizeof(GridDenseDev), cudaMemcpyHostToDevice);
    out.type = Field_Grid_Dense;
    out.device_ctx = d;
    return true;
}

void dvren::field_grid_dense_destroy(FieldProvider& fp)
{
    if (fp.type != Field_Grid_Dense || fp.device_ctx == nullptr) return;
    GridDenseDev h{};
    cudaMemcpy(&h, fp.device_ctx, sizeof(GridDenseDev), cudaMemcpyDeviceToHost);
    if (h.sigma) cudaFree(h.sigma);
    if (h.rgb) cudaFree(h.rgb);
    cudaFree(fp.device_ctx);
    fp.device_ctx = nullptr;
}
