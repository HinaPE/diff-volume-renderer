#include <cstdio>
#include <vector>
#include <cmath>
#include "dvren.h"

using namespace dvren;

static void field_fwd(const float* xyz, int n, float* sigma, float* rgb, void* user)
{
    for (int i = 0; i < n; ++i)
    {
        float x = xyz[i * 3 + 0];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        float r = sqrtf(x * x + y * y + z * z);
        sigma[i] = 10.0f * expf(-r * r * 2.0f);
        rgb[i * 3 + 0] = 1.0f;
        rgb[i * 3 + 1] = 0.5f;
        rgb[i * 3 + 2] = 0.2f;
    }
}

static void field_bwd(const float* xyz, const float* d_sigma, const float* d_rgb, int n, void* user)
{
}

int main()
{
    int W = 320;
    int H = 180;
    std::vector<float> origins(W * H * 3);
    std::vector<float> dirs(W * H * 3);
    std::vector<float> tmin(W * H);
    std::vector<float> tmax(W * H);
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            int i = y * W + x;
            origins[i * 3 + 0] = 0.0f;
            origins[i * 3 + 1] = 0.0f;
            origins[i * 3 + 2] = -2.0f;
            float u = (x + 0.5f) / W * 2.0f - 1.0f;
            float v = (y + 0.5f) / H * 2.0f - 1.0f;
            dirs[i * 3 + 0] = u;
            dirs[i * 3 + 1] = -v;
            dirs[i * 3 + 2] = 1.5f;
            float len = sqrtf(dirs[i * 3 + 0] * dirs[i * 3 + 0] + dirs[i * 3 + 1] * dirs[i * 3 + 1] + dirs[i * 3 + 2] * dirs[i * 3 + 2]);
            dirs[i * 3 + 0] /= len;
            dirs[i * 3 + 1] /= len;
            dirs[i * 3 + 2] /= len;
            tmin[i] = 0.0f;
            tmax[i] = 4.0f;
        }
    }
    RaysSOA rays{W, H, origins.data(), dirs.data(), tmin.data(), tmax.data()};
    SamplingSpec spec;
    spec.n_steps = 128;
    spec.dt = 0.03125f;
    spec.sigma_scale = 1.0f;
    spec.stop_thresh = 1e-3f;
    FieldProvider field{field_fwd, field_bwd, nullptr};
    RenderFlags flags{0, 1};
    ForwardOutputs out;
    volume_forward(rays, spec, field, flags, out);
    printf("image_size=%zu bytes=%zu\n", out.image.size(), out.image.size() * sizeof(float));
    return 0;
}
