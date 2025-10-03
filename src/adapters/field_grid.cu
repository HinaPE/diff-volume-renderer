#include <cuda_runtime.h>
#include <vector>
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
        float* g_sigma;
        float* g_rgb;
    };
}

static size_t voxels_count(const GridDenseDev& g)
{
    return (size_t)g.nx * g.ny * g.nz;
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
    size_t vox = (size_t)desc.nx * desc.ny * desc.nz;
    cudaMalloc(&h.sigma, vox * sizeof(float));
    cudaMemcpy(h.sigma, desc.host_sigma, vox * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&h.rgb, vox * 3 * sizeof(float));
    cudaMemcpy(h.rgb, desc.host_rgb, vox * 3 * sizeof(float), cudaMemcpyHostToDevice);
    h.g_sigma = nullptr;
    h.g_rgb = nullptr;
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
    if (h.g_sigma) cudaFree(h.g_sigma);
    if (h.g_rgb) cudaFree(h.g_rgb);
    cudaFree(fp.device_ctx);
    fp.device_ctx = nullptr;
}

void dvren::field_grid_dense_zero_grad(FieldProvider& fp)
{
    if (fp.type != Field_Grid_Dense || fp.device_ctx == nullptr) return;
    GridDenseDev h{};
    cudaMemcpy(&h, fp.device_ctx, sizeof(GridDenseDev), cudaMemcpyDeviceToHost);
    size_t vox = voxels_count(h);
    if (!h.g_sigma) cudaMalloc(&h.g_sigma, vox * sizeof(float));
    if (!h.g_rgb) cudaMalloc(&h.g_rgb, vox * 3 * sizeof(float));
    cudaMemset(h.g_sigma, 0, vox * sizeof(float));
    cudaMemset(h.g_rgb, 0, vox * 3 * sizeof(float));
    cudaMemcpy(fp.device_ctx, &h, sizeof(GridDenseDev), cudaMemcpyHostToDevice);
}

bool dvren::field_grid_dense_download_grad(const FieldProvider& fp, std::vector<float>& sigma_g, std::vector<float>& rgb_g)
{
    if (fp.type != Field_Grid_Dense || fp.device_ctx == nullptr) return false;
    GridDenseDev h{};
    cudaMemcpy(&h, fp.device_ctx, sizeof(GridDenseDev), cudaMemcpyDeviceToHost);
    if (!h.g_sigma || !h.g_rgb) return false;
    size_t vox = voxels_count(h);
    sigma_g.resize(vox);
    rgb_g.resize(vox * 3);
    cudaMemcpy(sigma_g.data(), h.g_sigma, vox * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rgb_g.data(), h.g_rgb, vox * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    return true;
}
