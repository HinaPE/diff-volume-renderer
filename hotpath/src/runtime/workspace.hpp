#pragma once

#include <cstddef>
#include <cstdint>

struct hp_workspace_allocator {
    std::byte* ptr{nullptr};
    size_t remaining{0};

    hp_workspace_allocator(void* base, size_t bytes) {
        ptr = static_cast<std::byte*>(base);
        remaining = bytes;
    }

    void* allocate(size_t bytes, size_t alignment) {
        if (bytes == 0) {
            return nullptr;
        }
        uintptr_t current = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t mask = static_cast<uintptr_t>(alignment - 1U);
        uintptr_t aligned = (current + mask) & ~mask;
        size_t padding = static_cast<size_t>(aligned - current);
        if (padding > remaining || bytes > (remaining - padding)) {
            return nullptr;
        }
        ptr += padding;
        remaining -= padding;
        void* out = ptr;
        ptr += bytes;
        remaining -= bytes;
        return out;
    }
};

