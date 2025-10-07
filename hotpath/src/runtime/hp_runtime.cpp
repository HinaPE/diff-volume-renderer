#include "hotpath/hp.h"

#include <iostream>
#include <new>

extern "C" {

struct hp_ctx {
    hp_ctx_desc desc{};
    hp_version version{HP_VERSION_MAJOR, HP_VERSION_MINOR, HP_VERSION_PATCH};
};

struct hp_plan {
    hp_plan_desc desc{};
    const hp_ctx* ctx{};
};

enum class hp_field_kind : uint32_t {
    dense_sigma = 0,
    dense_color = 1,
    hash_mlp = 2
};

struct hp_field {
    hp_field_kind kind{hp_field_kind::dense_sigma};
    hp_tensor source{};
};

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

HP_API hp_status hp_ray(const hp_plan*, const hp_rays_t*, hp_rays_t*, void*, size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
}

HP_API hp_status hp_samp(const hp_plan*, const hp_field*, const hp_field*, const hp_rays_t*, hp_samp_t*, void*, size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
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

HP_API hp_status hp_field_create_grid_sigma(const hp_ctx* ctx, const hp_tensor* grid, uint32_t, uint32_t, hp_field** out_field) {
    if (ctx == nullptr || grid == nullptr || out_field == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    auto* field = new (std::nothrow) hp_field();
    if (field == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }
    field->kind = hp_field_kind::dense_sigma;
    field->source = *grid;
    (void)ctx;
    *out_field = field;
    return HP_STATUS_SUCCESS;
}

HP_API hp_status hp_field_create_grid_color(const hp_ctx* ctx, const hp_tensor* grid, uint32_t, uint32_t, hp_field** out_field) {
    if (ctx == nullptr || grid == nullptr || out_field == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    auto* field = new (std::nothrow) hp_field();
    if (field == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }
    field->kind = hp_field_kind::dense_color;
    field->source = *grid;
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

HP_API hp_status hp_runner_run(const hp_ctx*, const hp_runner_options*) {
    static constexpr const char* kEmptyScoreboard = "{\"cases\":[],\"summary\":{\"pass\":0,\"fail\":0}}\n";
    std::cout << kEmptyScoreboard;
    return HP_STATUS_SUCCESS;
}

}  // extern "C"
