#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include "dvren.h"

using namespace dvren;

static void make_camera(int W, int H, std::vector<float>& o, std::vector<float>& d, std::vector<float>& t0, std::vector<float>& t1)
{
    o.resize(W * H * 3);
    d.resize(W * H * 3);
    t0.resize(W * H);
    t1.resize(W * H);
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            int i = y * W + x;
            o[i * 3 + 0] = 0.f;
            o[i * 3 + 1] = 0.f;
            o[i * 3 + 2] = -2.f;
            float u = (x + 0.5f) / W * 2.f - 1.f;
            float v = (y + 0.5f) / H * 2.f - 1.f;
            float dx = u, dy = -v, dz = 1.5f;
            float len = std::sqrt(dx * dx + dy * dy + dz * dz);
            d[i * 3 + 0] = dx / len;
            d[i * 3 + 1] = dy / len;
            d[i * 3 + 2] = dz / len;
            t0[i] = 0.f;
            t1[i] = 4.f;
        }
    }
}

int main()
{
    int W = 64, H = 64;
    std::vector<float> o, d, t0, t1;
    make_camera(W, H, o, d, t0, t1);
    int nx = 16, ny = 16, nz = 16;
    size_t vox = (size_t)nx * ny * nz;
    std::vector<float> sigma(vox, 0.0f), rgb(vox * 3, 0.0f);
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
                sigma[id] = 8.f * std::exp(-r * r * 3.f);
                rgb[id * 3 + 0] = 0.8f;
                rgb[id * 3 + 1] = 0.5f;
                rgb[id * 3 + 2] = 0.3f;
            }
        }
    }
    GridDenseDesc gd{};
    gd.nx = nx;
    gd.ny = ny;
    gd.nz = nz;
    gd.bbox_min[0] = -1.f;
    gd.bbox_min[1] = -1.f;
    gd.bbox_min[2] = -1.f;
    gd.bbox_max[0] = 1.f;
    gd.bbox_max[1] = 1.f;
    gd.bbox_max[2] = 1.f;
    gd.host_sigma = sigma.data();
    gd.host_rgb = rgb.data();
    FieldProvider fp{};
    field_grid_dense_create(gd, fp);

    RaysSOA rays{W, H, o.data(), d.data(), t0.data(), t1.data()};
    SamplingSpec spec{256, 0.0f, 1.0f, 0.0f};
    spec.dt = (t1[0] - t0[0]) / spec.n_steps;
    RenderFlags flags{0, 1};

    ForwardOutputs out{};
    field_grid_dense_zero_grad(fp);
    volume_forward(rays, spec, fp, flags, out);

    int px = W / 2, py = H / 2;
    std::vector<float> dL(W * H * 3, 0.f);
    dL[(py * W + px) * 3 + 0] = 1.f;

    volume_backward(out.saved_ctx, out.saved_ctx_bytes, dL.data(), W, H, fp);

    std::vector<float> gsig, grgb;
    field_grid_dense_download_grad(fp, gsig, grgb);

    int vx = nx / 2, vy = ny / 2, vz = nz / 2;
    size_t vid = (size_t)vz * ny * nx + vy * nx + vx;
    float g_analytic = gsig[vid];

    float eps = 1e-4f;
    float old = sigma[vid];
    sigma[vid] = old + eps;
    FieldProvider fp_pos{};
    gd.host_sigma = sigma.data();
    field_grid_dense_create(gd, fp_pos);
    ForwardOutputs out_pos{};
    volume_forward(rays, spec, fp_pos, flags, out_pos);
    float Lpos = out_pos.image[(py * W + px) * 3 + 0];
    field_grid_dense_destroy(fp_pos);
    sigma[vid] = old - eps;
    FieldProvider fp_neg{};
    gd.host_sigma = sigma.data();
    field_grid_dense_create(gd, fp_neg);
    ForwardOutputs out_neg{};
    volume_forward(rays, spec, fp_neg, flags, out_neg);
    float Lneg = out_neg.image[(py * W + px) * 3 + 0];
    field_grid_dense_destroy(fp_neg);
    sigma[vid] = old;

    float g_num = (Lpos - Lneg) / (2 * eps);
    float relerr = std::abs(g_analytic - g_num) / std::max(1e-6f, std::abs(g_num));
    printf("g_analytic=%g g_numeric=%g relerr=%g\n", g_analytic, g_num, relerr);

    field_grid_dense_destroy(fp);
    return 0;
}
