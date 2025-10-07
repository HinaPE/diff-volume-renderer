#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "hotpath/hp.h"

namespace dvren {

hp_tensor MakeHostTensor(void* data, hp_dtype dtype, const std::vector<int64_t>& shape);
hp_tensor MakeHostTensor(void* data, hp_dtype dtype, std::initializer_list<int64_t> shape);

template <typename T, size_t N>
hp_tensor MakeHostTensor(T* data, hp_dtype dtype, const std::array<int64_t, N>& shape) {
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    return MakeHostTensor(static_cast<void*>(data), dtype, shape_vec);
}

template <typename T>
hp_tensor MakeHostTensor(std::vector<T>& buffer, hp_dtype dtype, const std::vector<int64_t>& shape) {
    return MakeHostTensor(static_cast<void*>(buffer.data()), dtype, shape);
}

template <typename T>
hp_tensor MakeHostTensor(std::vector<T>& buffer, hp_dtype dtype, std::initializer_list<int64_t> shape) {
    return MakeHostTensor(static_cast<void*>(buffer.data()), dtype, shape);
}

}  // namespace dvren

