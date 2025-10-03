#include <cstdio>
#include <vector>
#include <cmath>
#include "dvren.h"

using namespace dvren;

int main()
{
    int W = 320, H = 180;
    std::vector<float> origins(W * H * 3), dirs(W * H * 3), tmin(W * H), tmax(W * H);
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            int i = y * W + x;
            origins[i * 3 + 0] = 0.f;
            origins[i * 3 + 1] = 0.f;
            origins[i * 3 + 2] = -2.f;
            float u = (x + 0.5f) / W * 2.f - 1.f;
            float v = (y + 0.5f) / H * 2.f - 1.f;
            float dx = u, dy = -v, dz = 1.5f;
            float len = std::sqrt(dx * dx + dy * dy + dz * dz);
            dirs[i * 3 + 0] = dx / len;
            dirs[i * 3 + 1] = dy / len;
            dirs[i * 3 + 2] = dz / len;
            tmin[i] = 0.f;
            tmax[i] = 4.f;
        }
    }
    int nx = 64, ny = 64, nz = 64;
    size_t vox = (size_t)nx * ny * nz;
    std::vector<float> sigma(vox), rgb(vox * 3);
    for (int z = 0; z < nz; ++z)
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                size_t id = (size_t)z * ny * nx + y * nx + x;
                float fx = (x + 0.5f) / nx * 2.f - 1.f;
                float fy = (y + 0.5f) / ny * 2.f - 1.f;
                float fz = (z + 0.5f) / nz * 2.f - 1.f;
                float r = std::sqrt(fx * fx + fy * fy + fz * fz);
                sigma[id] = 10.f * std::exp(-r * r * 4.f);
                rgb[id * 3 + 0] = 0.9f;
                rgb[id * 3 + 1] = 0.4f + 0.3f * fx;
                rgb[id * 3 + 2] = 0.2f + 0.5f * (1.f - r);
            }
        }
    }
    GridDenseDesc gdesc{};
    gdesc.nx = nx;
    gdesc.ny = ny;
    gdesc.nz = nz;
    gdesc.bbox_min[0] = -1.f;
    gdesc.bbox_min[1] = -1.f;
    gdesc.bbox_min[2] = -1.f;
    gdesc.bbox_max[0] = 1.f;
    gdesc.bbox_max[1] = 1.f;
    gdesc.bbox_max[2] = 1.f;
    gdesc.host_sigma = sigma.data();
    gdesc.host_rgb = rgb.data();
    FieldProvider fp{};
    field_grid_dense_create(gdesc, fp);

    RaysSOA rays{W, H, origins.data(), dirs.data(), tmin.data(), tmax.data()};
    SamplingSpec spec{128, 0.03125f, 1.0f, 1e-3f};
    RenderFlags flags{0, 1};
    ForwardOutputs out{};
    volume_forward(rays, spec, fp, flags, out);
    printf("ok image %zu floats\n", out.image.size());

    field_grid_dense_destroy(fp);
    return 0;
}
