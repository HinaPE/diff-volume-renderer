#include "dvren/core/context.hpp"

#include <utility>

namespace dvren {

Context::~Context() {
    if (ctx_ != nullptr) {
        hp_ctx_release(ctx_);
        ctx_ = nullptr;
    }
}

Context::Context(Context&& other) noexcept {
    ctx_ = other.ctx_;
    desc_ = other.desc_;
    other.ctx_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (ctx_ != nullptr) {
        hp_ctx_release(ctx_);
    }
    ctx_ = other.ctx_;
    desc_ = other.desc_;
    other.ctx_ = nullptr;
    return *this;
}

Status Context::Create(const ContextOptions& options, Context& out) {
    hp_ctx_desc desc{};
    desc.flags = options.flags;
    desc.preferred_device = options.preferred_device.empty() ? nullptr : options.preferred_device.c_str();
    desc.reserved = nullptr;

    hp_ctx* ctx_ptr = nullptr;
    const hp_status status = hp_ctx_create(&desc, &ctx_ptr);
    if (status != HP_STATUS_SUCCESS || ctx_ptr == nullptr) {
        return Status::FromHotpath(status, "hp_ctx_create failed");
    }

    hp_ctx_desc actual_desc{};
    const hp_status get_status = hp_ctx_get_desc(ctx_ptr, &actual_desc);
    if (get_status != HP_STATUS_SUCCESS) {
        hp_ctx_release(ctx_ptr);
        return Status::FromHotpath(get_status, "hp_ctx_get_desc failed");
    }

    out.Reset(ctx_ptr, actual_desc);
    return Status::Ok();
}

void Context::Reset(hp_ctx* ctx, const hp_ctx_desc& desc) {
    if (ctx_ != nullptr) {
        hp_ctx_release(ctx_);
    }
    ctx_ = ctx;
    desc_ = desc;
}

}  // namespace dvren

