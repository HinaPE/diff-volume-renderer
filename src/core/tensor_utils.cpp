#include "dvren/core/tensor_utils.hpp"

#include <algorithm>

namespace dvren {

namespace {

void ComputeStrides(hp_tensor& tensor) {
    if (tensor.rank == 0) {
        return;
    }
    tensor.stride[tensor.rank - 1] = 1;
    for (int64_t index = static_cast<int64_t>(tensor.rank) - 2; index >= 0; --index) {
        const uint32_t current = static_cast<uint32_t>(index);
        const uint32_t next = static_cast<uint32_t>(index + 1);
        tensor.stride[current] = tensor.stride[next] * tensor.shape[next];
    }
}

}  // namespace

hp_tensor MakeHostTensor(void* data, hp_dtype dtype, const std::vector<int64_t>& shape) {
    hp_tensor tensor{};
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.memspace = HP_MEMSPACE_HOST;
    tensor.rank = static_cast<uint32_t>(shape.size());
    const uint32_t rank = tensor.rank;

    for (uint32_t i = 0; i < rank && i < 8U; ++i) {
        tensor.shape[i] = shape[i];
    }
    for (uint32_t i = rank; i < 8U; ++i) {
        tensor.shape[i] = 0;
        tensor.stride[i] = 0;
    }

    if (rank > 0U) {
        ComputeStrides(tensor);
    }
    return tensor;
}

hp_tensor MakeHostTensor(void* data, hp_dtype dtype, std::initializer_list<int64_t> shape) {
    std::vector<int64_t> shape_vec(shape);
    return MakeHostTensor(data, dtype, shape_vec);
}

}  // namespace dvren

