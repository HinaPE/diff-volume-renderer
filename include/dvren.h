#ifndef DIFFVOLUMERENDERER_DVREN_H
#define DIFFVOLUMERENDERER_DVREN_H
#include <cstdint>
#include <vector>

namespace dvren
{
    struct RaysSOA
    {
        int width;
        int height;
        const float* origins;
        const float* dirs;
        const float* tmin;
        const float* tmax;
    };

    struct SamplingSpec
    {
        int n_steps;
        float dt;
        float sigma_scale;
        float stop_thresh;
    };

    struct ForwardOutputs
    {
        std::vector<float> image;
        void* saved_ctx;
        size_t saved_ctx_bytes;
    };

    enum FieldType : uint32_t
    {
        Field_Grid_Dense = 1
    };

    struct FieldProvider
    {
        FieldType type;
        void* device_ctx;
    };

    struct RenderFlags
    {
        uint32_t retain_rgb;
        uint32_t use_fp16_field;
    };

    bool volume_forward(const RaysSOA& rays, const SamplingSpec& spec, const FieldProvider& field, const RenderFlags& flags, ForwardOutputs& out);
    bool volume_backward(void* saved_ctx, size_t saved_ctx_bytes, const float* dL_dimage, int width, int height, const FieldProvider& field);

    struct GridDenseDesc
    {
        int nx;
        int ny;
        int nz;
        float bbox_min[3];
        float bbox_max[3];
        const float* host_sigma;
        const float* host_rgb;
    };

    bool field_grid_dense_create(const GridDenseDesc& desc, FieldProvider& out);
    void field_grid_dense_destroy(FieldProvider& fp);
}
#endif //DIFFVOLUMERENDERER_DVREN_H
