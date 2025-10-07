#include "hp_internal.hpp"

#if defined(HP_WITH_CUDA)

#include <cuda_runtime.h>

namespace hp_internal {

hp_status diff_generate_cuda(const hp_plan*,
                             const hp_tensor*,
                             const hp_samp_t*,
                             const hp_intl_t*,
                             const hp_rays_t*,
                             hp_grads_t*,
                             void*,
                             size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
}

}  // namespace hp_internal

#endif  // HP_WITH_CUDA

