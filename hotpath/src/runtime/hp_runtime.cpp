#include "hotpath/hp.h"
#include "hp_internal.hpp"

#include <climits>
#include <algorithm>
#include <iostream>
#include <new>

extern "C" {

HP_API hp_version hp_get_version(void) {
    return hp_version{HP_VERSION_MAJOR, HP_VERSION_MINOR, HP_VERSION_PATCH};
}

HP_API hp_status hp_ctx_create(const hp_ctx_desc* desc, hp_ctx** out_ctx) {
    if (out_ctx == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    auto* ctx = new (std::nothrow) hp_ctx();
    if (ctx == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }
    if (desc != nullptr) {
        ctx->desc = *desc;
    } else {
        ctx->desc = hp_ctx_desc{};
    }
    ctx->version = hp_get_version();
    *out_ctx = ctx;
    return HP_STATUS_SUCCESS;
}

HP_API void hp_ctx_release(hp_ctx* ctx) {
    delete ctx;
}

HP_API hp_status hp_ctx_get_desc(const hp_ctx* ctx, hp_ctx_desc* out_desc) {
    if (ctx == nullptr || out_desc == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    *out_desc = ctx->desc;
    return HP_STATUS_SUCCESS;
}

HP_API hp_status hp_plan_create(const hp_ctx* ctx, const hp_plan_desc* desc, hp_plan** out_plan) {
    if (ctx == nullptr || desc == nullptr || out_plan == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    auto* plan = new (std::nothrow) hp_plan();
    if (plan == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }
    plan->ctx = ctx;
    plan->desc = *desc;
    if (plan->desc.width == 0 || plan->desc.height == 0) {
        delete plan;
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (!(plan->desc.t_far > plan->desc.t_near)) {
        delete plan;
        return HP_STATUS_INVALID_ARGUMENT;
    }

    hp_camera_desc cam = plan->desc.camera;
    if (cam.model != HP_CAMERA_PINHOLE && cam.model != HP_CAMERA_ORTHOGRAPHIC) {
        cam.model = HP_CAMERA_PINHOLE;
    }
    const bool K_is_zero =
        cam.K[0] == 0.0f && cam.K[1] == 0.0f && cam.K[2] == 0.0f &&
        cam.K[3] == 0.0f && cam.K[4] == 0.0f && cam.K[5] == 0.0f &&
        cam.K[6] == 0.0f && cam.K[7] == 0.0f && cam.K[8] == 0.0f;
    if (K_is_zero) {
        cam.K[0] = 1.0f;
        cam.K[4] = 1.0f;
        cam.K[8] = 1.0f;
        cam.K[2] = static_cast<float>(plan->desc.width) * 0.5f;
        cam.K[5] = static_cast<float>(plan->desc.height) * 0.5f;
    }
    if (cam.K[0] == 0.0f) {
        cam.K[0] = 1.0f;
    }
    if (cam.K[4] == 0.0f) {
        cam.K[4] = 1.0f;
    }
    const bool c2w_is_zero =
        cam.c2w[0] == 0.0f && cam.c2w[1] == 0.0f && cam.c2w[2] == 0.0f && cam.c2w[3] == 0.0f &&
        cam.c2w[4] == 0.0f && cam.c2w[5] == 0.0f && cam.c2w[6] == 0.0f && cam.c2w[7] == 0.0f &&
        cam.c2w[8] == 0.0f && cam.c2w[9] == 0.0f && cam.c2w[10] == 0.0f && cam.c2w[11] == 0.0f;
    if (c2w_is_zero) {
        cam.c2w[0] = 1.0f;
        cam.c2w[5] = 1.0f;
        cam.c2w[10] = 1.0f;
    }
    if (cam.model == HP_CAMERA_ORTHOGRAPHIC && cam.ortho_scale <= 0.0f) {
        cam.ortho_scale = 1.0f;
    }
    plan->desc.camera = cam;

    hp_roi_desc roi = plan->desc.roi;
    if (roi.width == 0 || roi.height == 0) {
        roi.x = 0;
        roi.y = 0;
        roi.width = plan->desc.width;
        roi.height = plan->desc.height;
    }
    if (roi.x + roi.width > plan->desc.width || roi.y + roi.height > plan->desc.height) {
        delete plan;
        return HP_STATUS_INVALID_ARGUMENT;
    }
    plan->desc.roi = roi;
    const uint64_t roi_rays = static_cast<uint64_t>(roi.width) * static_cast<uint64_t>(roi.height);
    if (plan->desc.max_rays == 0U) {
        plan->desc.max_rays = static_cast<uint32_t>(std::min<uint64_t>(roi_rays, static_cast<uint64_t>(UINT32_MAX)));
    }
    if (roi_rays > plan->desc.max_rays) {
        delete plan;
        return HP_STATUS_INVALID_ARGUMENT;
    }

    hp_sampling_desc sampling = plan->desc.sampling;
    if (!(sampling.dt > 0.0f)) {
        const float span = plan->desc.t_far - plan->desc.t_near;
        const float default_dt = span > 0.0f ? span / 64.0f : 1.0f;
        sampling.dt = default_dt > 0.0f ? default_dt : 1.0f;
    }
    if (sampling.max_steps == 0U) {
        sampling.max_steps = 64U;
    }
    if (sampling.mode != HP_SAMPLING_FIXED && sampling.mode != HP_SAMPLING_STRATIFIED) {
        sampling.mode = HP_SAMPLING_FIXED;
    }
    plan->desc.sampling = sampling;

    if (plan->desc.max_samples == 0U) {
        const uint64_t suggested = static_cast<uint64_t>(plan->desc.max_rays) * static_cast<uint64_t>(sampling.max_steps);
        const uint64_t bounded = std::min<uint64_t>(suggested, static_cast<uint64_t>(UINT32_MAX));
        plan->desc.max_samples = bounded == 0 ? plan->desc.max_rays : static_cast<uint32_t>(bounded);
    }
    if (plan->desc.max_samples < plan->desc.max_rays) {
        delete plan;
        return HP_STATUS_INVALID_ARGUMENT;
    }

    *out_plan = plan;
    return HP_STATUS_SUCCESS;
}

HP_API void hp_plan_release(hp_plan* plan) {
    delete plan;
}

HP_API hp_status hp_plan_get_desc(const hp_plan* plan, hp_plan_desc* out_desc) {
    if (plan == nullptr || out_desc == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    *out_desc = plan->desc;
    return HP_STATUS_SUCCESS;
}

HP_API hp_status hp_samp(const hp_plan* plan,
                         const hp_field* fs,
                         const hp_field* fc,
                         const hp_rays_t* rays,
                         hp_samp_t* samp,
                         void* ws,
                         size_t ws_bytes) {
    if (plan == nullptr || rays == nullptr || samp == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const hp_memspace memspace = rays->origins.memspace;
    if (memspace == HP_MEMSPACE_DEVICE) {
#if defined(HP_WITH_CUDA)
        return hp_internal::samp_generate_cuda(plan, fs, fc, rays, samp, ws, ws_bytes);
#else
        (void)fs;
        (void)fc;
        (void)ws;
        (void)ws_bytes;
        return HP_STATUS_UNSUPPORTED;
#endif
    }

    return hp_internal::samp_generate_cpu(plan, fs, fc, rays, samp, ws, ws_bytes);
}

HP_API hp_status hp_int(const hp_plan*, const hp_samp_t*, hp_intl_t*, void*, size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
}

HP_API hp_status hp_img(const hp_plan*, const hp_intl_t*, const hp_rays_t*, hp_img_t*, void*, size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
}

HP_API hp_status hp_diff(const hp_plan*, const hp_tensor*, const hp_samp_t*, const hp_intl_t*, hp_grads_t*, void*, size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
}

HP_API hp_status hp_field_create_grid_sigma(const hp_ctx* ctx, const hp_tensor* grid, uint32_t interp, uint32_t oob, hp_field** out_field) {
    if (ctx == nullptr || grid == nullptr || out_field == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    auto* field = new (std::nothrow) hp_field();
    if (field == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }
    if (grid->dtype != HP_DTYPE_F32) {
        delete field;
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (grid->memspace != HP_MEMSPACE_HOST) {
        delete field;
        return HP_STATUS_UNSUPPORTED;
    }
    if (grid->rank < 3) {
        delete field;
        return HP_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < grid->rank; ++i) {
        if (grid->shape[i] <= 0) {
            delete field;
            return HP_STATUS_INVALID_ARGUMENT;
        }
    }
    field->kind = hp_field_kind::dense_sigma;
    field->source = *grid;
    field->interp = (interp == static_cast<uint32_t>(HP_INTERP_NEAREST)) ? HP_INTERP_NEAREST : HP_INTERP_LINEAR;
    field->oob = (oob == static_cast<uint32_t>(HP_OOB_CLAMP)) ? HP_OOB_CLAMP : HP_OOB_ZERO;
    field->world_min[0] = 0.0f;
    field->world_min[1] = 0.0f;
    field->world_min[2] = 0.0f;
    field->world_max[0] = 1.0f;
    field->world_max[1] = 1.0f;
    field->world_max[2] = 1.0f;
    (void)ctx;
    *out_field = field;
    return HP_STATUS_SUCCESS;
}

HP_API hp_status hp_field_create_grid_color(const hp_ctx* ctx, const hp_tensor* grid, uint32_t interp, uint32_t oob, hp_field** out_field) {
    if (ctx == nullptr || grid == nullptr || out_field == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    auto* field = new (std::nothrow) hp_field();
    if (field == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }
    if (grid->dtype != HP_DTYPE_F32) {
        delete field;
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (grid->memspace != HP_MEMSPACE_HOST) {
        delete field;
        return HP_STATUS_UNSUPPORTED;
    }
    if (grid->rank < 4 || grid->shape[grid->rank - 1] != 3) {
        delete field;
        return HP_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < grid->rank; ++i) {
        if (grid->shape[i] <= 0) {
            delete field;
            return HP_STATUS_INVALID_ARGUMENT;
        }
    }
    field->kind = hp_field_kind::dense_color;
    field->source = *grid;
    field->interp = (interp == static_cast<uint32_t>(HP_INTERP_NEAREST)) ? HP_INTERP_NEAREST : HP_INTERP_LINEAR;
    field->oob = (oob == static_cast<uint32_t>(HP_OOB_CLAMP)) ? HP_OOB_CLAMP : HP_OOB_ZERO;
    field->world_min[0] = 0.0f;
    field->world_min[1] = 0.0f;
    field->world_min[2] = 0.0f;
    field->world_max[0] = 1.0f;
    field->world_max[1] = 1.0f;
    field->world_max[2] = 1.0f;
    (void)ctx;
    *out_field = field;
    return HP_STATUS_SUCCESS;
}

HP_API hp_status hp_field_create_hash_mlp(const hp_ctx* ctx, const hp_tensor* params, hp_field** out_field) {
    if (ctx == nullptr || out_field == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    (void)params;
    (void)ctx;
    return HP_STATUS_NOT_IMPLEMENTED;
}

HP_API void hp_field_release(hp_field* field) {
    delete field;
}

}  // extern "C"
