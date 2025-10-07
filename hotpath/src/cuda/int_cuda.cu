#include "hp_internal.hpp"

#if defined(HP_WITH_CUDA)

#include <cuda_runtime.h>

namespace hp_internal {

hp_status int_generate_cuda(const hp_plan*,
                            const hp_samp_t*,
                            hp_intl_t*,
                            void*,
                            size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
}

}  // namespace hp_internal

#endif  // HP_WITH_CUDA

