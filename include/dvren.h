#ifndef DIFFVOLUMERENDERER_DVREN_H
#define DIFFVOLUMERENDERER_DVREN_H
#include <cstdint>
#include <vector>

namespace dvren {

    struct RaysSOA {
        int width;
        int height;
        const float* origins;
        const float* dirs;
        const float* tmin;
        const float* tmax;
    };

    struct SamplingSpec {
        int n_steps;
        float dt;
        float sigma_scale;
        float stop_thresh;
    };

    struct ForwardOutputs {
        std::vector<float> image;
        void* saved_ctx;
        size_t saved_ctx_bytes;
    };

    typedef void (*FieldForwardFn)(const float* xyz, int n, float* sigma, float* rgb, void* user);
    typedef void (*FieldBackwardFn)(const float* xyz, const float* d_sigma, const float* d_rgb, int n, void* user);

    struct FieldProvider {
        FieldForwardFn fwd;
        FieldBackwardFn bwd;
        void* user;
    };

    struct RenderFlags {
        uint32_t retain_rgb;
        uint32_t use_fp16_field;
    };

    bool volume_forward(const RaysSOA& rays, const SamplingSpec& spec, const FieldProvider& field, const RenderFlags& flags, ForwardOutputs& out);
    bool volume_backward(void* saved_ctx, size_t saved_ctx_bytes, const float* dL_dimage, int width, int height, const FieldProvider& field);
}
#endif //DIFFVOLUMERENDERER_DVREN_H