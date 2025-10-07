#pragma once

#include <string>

#include "dvren/core/status.hpp"
#include "hotpath/hp.h"

namespace dvren {

struct ContextOptions {
    uint32_t flags{0u};
    std::string preferred_device;
};

class Context {
public:
    Context() = default;
    ~Context();

    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    static Status Create(const ContextOptions& options, Context& out);

    [[nodiscard]] bool valid() const { return ctx_ != nullptr; }
    [[nodiscard]] const hp_ctx* handle() const { return ctx_; }
    [[nodiscard]] hp_ctx* handle() { return ctx_; }
    [[nodiscard]] const hp_ctx_desc& descriptor() const { return desc_; }

private:
    void Reset(hp_ctx* ctx, const hp_ctx_desc& desc);

    hp_ctx* ctx_{nullptr};
    hp_ctx_desc desc_{};
};

}  // namespace dvren

