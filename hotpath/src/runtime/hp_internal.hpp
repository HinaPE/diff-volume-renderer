#pragma once

#include "hotpath/hp.h"

#include <cstddef>
#include <cstdint>

struct hp_ctx {
    hp_ctx_desc desc{};
    hp_version version{HP_VERSION_MAJOR, HP_VERSION_MINOR, HP_VERSION_PATCH};
};

struct hp_plan {
    hp_plan_desc desc{};
    const hp_ctx* ctx{};
};

enum class hp_field_kind : uint32_t {
    dense_sigma = 0u,
    dense_color = 1u,
    hash_mlp = 2u
};

struct hp_field {
    hp_field_kind kind{hp_field_kind::dense_sigma};
    hp_tensor source{};
    hp_interp_mode interp{HP_INTERP_LINEAR};
    hp_oob_policy oob{HP_OOB_ZERO};
    float world_min[3]{0.0f, 0.0f, 0.0f};
    float world_max[3]{1.0f, 1.0f, 1.0f};
};

namespace hp_internal {

hp_status ray_generate_cpu(const hp_plan* plan,
                           const hp_rays_t* override_or_null,
                           hp_rays_t* rays,
                           void* ws,
                           size_t ws_bytes);

#if defined(HP_WITH_CUDA)
hp_status ray_generate_cuda(const hp_plan* plan,
                            const hp_rays_t* override_or_null,
                            hp_rays_t* rays,
                            void* ws,
                            size_t ws_bytes);
#endif

hp_status samp_generate_cpu(const hp_plan* plan,
                            const hp_field* fs,
                            const hp_field* fc,
                            const hp_rays_t* rays,
                            hp_samp_t* samp,
                            void* ws,
                            size_t ws_bytes);

#if defined(HP_WITH_CUDA)
hp_status samp_generate_cuda(const hp_plan* plan,
                             const hp_field* fs,
                             const hp_field* fc,
                             const hp_rays_t* rays,
                             hp_samp_t* samp,
                             void* ws,
                             size_t ws_bytes);
#endif

hp_status int_generate_cpu(const hp_plan* plan,
                           const hp_samp_t* samp,
                           hp_intl_t* intl,
                           void* ws,
                           size_t ws_bytes);

#if defined(HP_WITH_CUDA)
hp_status int_generate_cuda(const hp_plan* plan,
                            const hp_samp_t* samp,
                            hp_intl_t* intl,
                            void* ws,
                            size_t ws_bytes);
#endif

hp_status img_generate_cpu(const hp_plan* plan,
                           const hp_intl_t* intl,
                           const hp_rays_t* rays,
                           hp_img_t* img,
                           void* ws,
                           size_t ws_bytes);

#if defined(HP_WITH_CUDA)
hp_status img_generate_cuda(const hp_plan* plan,
                            const hp_intl_t* intl,
                            const hp_rays_t* rays,
                            hp_img_t* img,
                            void* ws,
                            size_t ws_bytes);
#endif

hp_status samp_int_fused_cpu(const hp_plan* plan,
                             const hp_field* fs,
                             const hp_field* fc,
                             const hp_rays_t* rays,
                             hp_samp_t* samp,
                             hp_intl_t* intl,
                             void* ws,
                             size_t ws_bytes);

#if defined(HP_WITH_CUDA)
hp_status samp_int_fused_cuda(const hp_plan* plan,
                              const hp_field* fs,
                              const hp_field* fc,
                              const hp_rays_t* rays,
                              hp_samp_t* samp,
                              hp_intl_t* intl,
                              void* ws,
                              size_t ws_bytes);
#endif

float sample_grid_sigma_cpu(const hp_field* field, const float pos[3], hp_status* status);
void sample_grid_color_cpu(const hp_field* field, const float pos[3], float out_rgb[3], hp_status* status);

}  // namespace hp_internal
