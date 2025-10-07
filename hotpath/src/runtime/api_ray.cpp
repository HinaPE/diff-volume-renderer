#include "hp_internal.hpp"

#include <cstddef>

namespace {

bool validate_rays_handle(const hp_rays_t* rays) {
    if (rays == nullptr) {
        return false;
    }
    return true;
}

}  // namespace

extern "C" HP_API hp_status hp_ray(const hp_plan* plan,
                                   const hp_rays_t* override_or_null,
                                   hp_rays_t* rays,
                                   void* ws,
                                   size_t ws_bytes) {
    if (!validate_rays_handle(rays) || plan == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const hp_memspace memspace = rays->origins.memspace;
    if (memspace == HP_MEMSPACE_DEVICE) {
#if defined(HP_WITH_CUDA)
        return hp_internal::ray_generate_cuda(plan, override_or_null, rays, ws, ws_bytes);
#else
        (void)override_or_null;
        (void)ws;
        (void)ws_bytes;
        return HP_STATUS_UNSUPPORTED;
#endif
    }

    return hp_internal::ray_generate_cpu(plan, override_or_null, rays, ws, ws_bytes);
}

