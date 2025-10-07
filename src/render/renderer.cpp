#include "dvren/render/renderer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string_view>

#include "dvren/core/tensor_utils.hpp"

namespace dvren {

namespace {

double DurationMilliseconds(std::chrono::steady_clock::time_point begin,
                            std::chrono::steady_clock::time_point end) {
    using FpMs = std::chrono::duration<double, std::milli>;
    return std::chrono::duration_cast<FpMs>(end - begin).count();
}

template <typename T>
void ResizeBuffer(std::vector<T>& buffer, size_t count) {
    if (buffer.size() != count) {
        buffer.assign(count, T{});
    }
}

}  // namespace

Renderer::Renderer(const Context& ctx, const Plan& plan, RenderOptions options)
    : ctx_(&ctx), plan_(&plan), options_(options) {
    grad_camera_.assign(12, 0.0f);
    ConfigureGradientTensors(0);
}

Status Renderer::EnsureCapacity() {
    if (plan_ == nullptr || ctx_ == nullptr) {
        return Status(StatusCode::kInvalidArgument, "renderer is not bound to a context/plan");
    }

    const hp_plan_desc& desc = plan_->descriptor();
    const size_t roi_width = desc.roi.width > 0 ? desc.roi.width : desc.width;
    const size_t roi_height = desc.roi.height > 0 ? desc.roi.height : desc.height;
    const size_t ray_capacity = std::max<size_t>(static_cast<size_t>(desc.max_rays),
                                                 roi_width * roi_height);
    const size_t sample_capacity = std::max<size_t>(static_cast<size_t>(desc.max_samples), ray_capacity);
    const size_t pixel_capacity = std::max<size_t>(static_cast<size_t>(desc.width) * desc.height, 1U);

    if (ray_capacity == 0 || sample_capacity == 0 || pixel_capacity == 0) {
        return Status(StatusCode::kInvalidArgument, "plan capacities must be positive");
    }

    const bool ray_changed = ray_capacity != current_ray_capacity_;
    const bool sample_changed = sample_capacity != current_sample_capacity_;
    const bool pixel_changed = pixel_capacity != current_pixel_capacity_;

    if (ray_changed) {
        ResizeBuffer(ray_origins_, ray_capacity * 3ULL);
        ResizeBuffer(ray_dirs_, ray_capacity * 3ULL);
        ResizeBuffer(ray_t_near_, ray_capacity);
        ResizeBuffer(ray_t_far_, ray_capacity);
        ResizeBuffer(ray_pixels_, ray_capacity);
        ConfigureRayTensors(ray_capacity);
        ResizeBuffer(sample_offsets_, ray_capacity + 1ULL);
    }

    if (sample_changed) {
        ResizeBuffer(sample_positions_, sample_capacity * 3ULL);
        ResizeBuffer(sample_dt_, sample_capacity);
        ResizeBuffer(sample_sigma_, sample_capacity);
        ResizeBuffer(sample_color_, sample_capacity * 3ULL);
        ConfigureSampleTensors(sample_capacity, ray_capacity);

        ResizeBuffer(intl_radiance_, sample_capacity > 0 ? ray_capacity * 3ULL : 0ULL);
        ResizeBuffer(intl_trans_, ray_capacity);
        ResizeBuffer(intl_opacity_, ray_capacity);
        ResizeBuffer(intl_depth_, ray_capacity);
        ResizeBuffer(intl_aux_, sample_capacity * 4ULL);
        ConfigureIntegrationTensors(ray_capacity, sample_capacity);

        ResizeBuffer(grad_sigma_samples_, sample_capacity);
        ResizeBuffer(grad_color_samples_, sample_capacity * 3ULL);
        ConfigureGradientTensors(sample_capacity);
    }

    if (pixel_changed) {
        ResizeBuffer(img_image_, pixel_capacity * 3ULL);
        ResizeBuffer(img_trans_, pixel_capacity);
        ResizeBuffer(img_opacity_, pixel_capacity);
        ResizeBuffer(img_depth_, pixel_capacity);
        ResizeBuffer(img_hitmask_, pixel_capacity);
        ConfigureImageTensors(pixel_capacity);
    }

    const size_t required_ws = RequiredWorkspaceBytes(ray_capacity, sample_capacity);
    if (workspace_buffer_.size() != required_ws) {
        workspace_buffer_.assign(required_ws, std::byte{});
    }

    current_ray_capacity_ = ray_capacity;
    current_sample_capacity_ = sample_capacity;
    current_pixel_capacity_ = pixel_capacity;
    return Status::Ok();
}

void Renderer::ConfigureRayTensors(size_t ray_capacity) {
    const int64_t count = static_cast<int64_t>(ray_capacity);
    rays_.origins = MakeHostTensor(
        ray_origins_.empty() ? nullptr : static_cast<void*>(ray_origins_.data()),
        HP_DTYPE_F32,
        {count, 3});
    rays_.directions = MakeHostTensor(
        ray_dirs_.empty() ? nullptr : static_cast<void*>(ray_dirs_.data()),
        HP_DTYPE_F32,
        {count, 3});
    rays_.t_near = MakeHostTensor(
        ray_t_near_.empty() ? nullptr : static_cast<void*>(ray_t_near_.data()),
        HP_DTYPE_F32,
        {count});
    rays_.t_far = MakeHostTensor(
        ray_t_far_.empty() ? nullptr : static_cast<void*>(ray_t_far_.data()),
        HP_DTYPE_F32,
        {count});
    rays_.pixel_ids = MakeHostTensor(
        ray_pixels_.empty() ? nullptr : static_cast<void*>(ray_pixels_.data()),
        HP_DTYPE_U32,
        {count});
}

void Renderer::ConfigureSampleTensors(size_t sample_capacity, size_t ray_capacity) {
    const int64_t sample_count = static_cast<int64_t>(sample_capacity);
    samp_.positions = MakeHostTensor(
        sample_positions_.empty() ? nullptr : static_cast<void*>(sample_positions_.data()),
        HP_DTYPE_F32,
        {sample_count, 3});
    samp_.dt = MakeHostTensor(
        sample_dt_.empty() ? nullptr : static_cast<void*>(sample_dt_.data()),
        HP_DTYPE_F32,
        {sample_count});
    samp_.sigma = MakeHostTensor(
        sample_sigma_.empty() ? nullptr : static_cast<void*>(sample_sigma_.data()),
        HP_DTYPE_F32,
        {sample_count});
    samp_.color = MakeHostTensor(
        sample_color_.empty() ? nullptr : static_cast<void*>(sample_color_.data()),
        HP_DTYPE_F32,
        {sample_count, 3});
    samp_.ray_offset = MakeHostTensor(
        sample_offsets_.empty() ? nullptr : static_cast<void*>(sample_offsets_.data()),
        HP_DTYPE_U32,
        {static_cast<int64_t>(ray_capacity) + 1});
}

void Renderer::ConfigureIntegrationTensors(size_t ray_capacity, size_t sample_capacity) {
    const int64_t ray_count = static_cast<int64_t>(ray_capacity);
    const int64_t sample_count = static_cast<int64_t>(sample_capacity);

    intl_.radiance = MakeHostTensor(
        intl_radiance_.empty() ? nullptr : static_cast<void*>(intl_radiance_.data()),
        HP_DTYPE_F32,
        {ray_count, 3});
    intl_.transmittance = MakeHostTensor(
        intl_trans_.empty() ? nullptr : static_cast<void*>(intl_trans_.data()),
        HP_DTYPE_F32,
        {ray_count});
    intl_.opacity = MakeHostTensor(
        intl_opacity_.empty() ? nullptr : static_cast<void*>(intl_opacity_.data()),
        HP_DTYPE_F32,
        {ray_count});
    intl_.depth = MakeHostTensor(
        intl_depth_.empty() ? nullptr : static_cast<void*>(intl_depth_.data()),
        HP_DTYPE_F32,
        {ray_count});
    intl_.aux = MakeHostTensor(
        intl_aux_.empty() ? nullptr : static_cast<void*>(intl_aux_.data()),
        HP_DTYPE_F32,
        {sample_count, 4});
}

void Renderer::ConfigureImageTensors(size_t pixel_count) {
    const hp_plan_desc& desc = plan_->descriptor();
    const int64_t height = static_cast<int64_t>(desc.height);
    const int64_t width = static_cast<int64_t>(desc.width);

    img_.image = MakeHostTensor(
        img_image_.empty() ? nullptr : static_cast<void*>(img_image_.data()),
        HP_DTYPE_F32,
        {height, width, 3});
    img_.trans = MakeHostTensor(
        img_trans_.empty() ? nullptr : static_cast<void*>(img_trans_.data()),
        HP_DTYPE_F32,
        {height, width});
    img_.opacity = MakeHostTensor(
        img_opacity_.empty() ? nullptr : static_cast<void*>(img_opacity_.data()),
        HP_DTYPE_F32,
        {height, width});
    img_.depth = MakeHostTensor(
        img_depth_.empty() ? nullptr : static_cast<void*>(img_depth_.data()),
        HP_DTYPE_F32,
        {height, width});
    img_.hitmask = MakeHostTensor(
        img_hitmask_.empty() ? nullptr : static_cast<void*>(img_hitmask_.data()),
        HP_DTYPE_U32,
        {height, width});
}

void Renderer::ConfigureGradientTensors(size_t sample_capacity) {
    const int64_t sample_count = static_cast<int64_t>(sample_capacity);
    grads_.sigma = MakeHostTensor(
        grad_sigma_samples_.empty() ? nullptr : static_cast<void*>(grad_sigma_samples_.data()),
        HP_DTYPE_F32,
        {sample_count});
    grads_.color = MakeHostTensor(
        grad_color_samples_.empty() ? nullptr : static_cast<void*>(grad_color_samples_.data()),
        HP_DTYPE_F32,
        {sample_count, 3});
    grads_.camera = MakeHostTensor(
        grad_camera_.empty() ? nullptr : static_cast<void*>(grad_camera_.data()),
        HP_DTYPE_F32,
        {3, 4});
}

Status Renderer::Forward(const DenseGridField& field, ForwardResult& out) {
    if (!field.valid()) {
        return Status(StatusCode::kInvalidArgument, "field is invalid");
    }

    Status cap_status = EnsureCapacity();
    if (!cap_status.ok()) {
        return cap_status;
    }

    RenderStats stats{};
    auto total_begin = std::chrono::steady_clock::now();

#if !defined(HP_WITH_CUDA)
    if (options_.enable_graph) {
        stats.notes.emplace_back("graph_disabled_no_cuda");
    }
#else
    if (options_.enable_graph) {
        stats.notes.emplace_back("graph_not_implemented");
    }
#endif
    stats.notes.emplace_back(options_.use_fused_path ? "forward_mode=fused" : "forward_mode=staged");

    const bool use_fused = options_.use_fused_path && !workspace_buffer_.empty();

    auto ray_begin = std::chrono::steady_clock::now();
    hp_status hp = hp_ray(plan_->handle(), nullptr, &rays_, nullptr, 0);
    auto ray_end = std::chrono::steady_clock::now();
    if (hp != HP_STATUS_SUCCESS) {
        return Status::FromHotpath(hp, "hp_ray failed");
    }
    stats.ray_ms = DurationMilliseconds(ray_begin, ray_end);

    last_ray_count_ = (rays_.t_near.rank >= 1) ? static_cast<size_t>(rays_.t_near.shape[0]) : 0;
    if (last_ray_count_ == 0) {
        stats.total_ms = DurationMilliseconds(total_begin, std::chrono::steady_clock::now());
        out = ForwardResult{};
        out.stats = stats;
        return Status::Ok();
    }

    double integrate_ms = 0.0;
    auto samp_begin = std::chrono::steady_clock::now();
    if (use_fused) {
        hp = hp_samp_int_fused(plan_->handle(),
                               field.sigma_field(),
                               field.color_field(),
                               &rays_,
                               &samp_,
                               &intl_,
                               workspace_buffer_.data(),
                               workspace_buffer_.size());
        auto samp_end = std::chrono::steady_clock::now();
        stats.sample_ms = DurationMilliseconds(samp_begin, samp_end);
    } else {
        hp = hp_samp(plan_->handle(),
                     field.sigma_field(),
                     field.color_field(),
                     &rays_,
                     &samp_, workspace_buffer_.data(), workspace_buffer_.size());
        auto samp_end = std::chrono::steady_clock::now();
        stats.sample_ms = DurationMilliseconds(samp_begin, samp_end);
        if (hp == HP_STATUS_SUCCESS) {
            auto int_begin = std::chrono::steady_clock::now();
            hp = hp_int(plan_->handle(), &samp_, &intl_, workspace_buffer_.data(), workspace_buffer_.size());
            auto int_end = std::chrono::steady_clock::now();
            integrate_ms = DurationMilliseconds(int_begin, int_end);
        }
    }
    if (hp != HP_STATUS_SUCCESS) {
        return Status::FromHotpath(hp, use_fused ? "hp_samp_int_fused failed" : "hp_samp/hp_int failed");
    }
    stats.integrate_ms = integrate_ms;

    last_sample_count_ = (samp_.dt.rank >= 1) ? static_cast<size_t>(samp_.dt.shape[0]) : 0;
    if (use_fused) {
        CopyOutputsFromWorkspace(last_ray_count_, last_sample_count_);
    }

    auto img_begin = std::chrono::steady_clock::now();
    hp = hp_img(plan_->handle(), &intl_, &rays_, &img_, workspace_buffer_.data(), workspace_buffer_.size());
    auto img_end = std::chrono::steady_clock::now();
    if (hp != HP_STATUS_SUCCESS) {
        return Status::FromHotpath(hp, "hp_img failed");
    }
    stats.compose_ms = DurationMilliseconds(img_begin, img_end);

    const size_t pixel_count = static_cast<size_t>(plan_->descriptor().width) *
                               static_cast<size_t>(plan_->descriptor().height);
    const size_t image_elements = pixel_count * 3ULL;

    out.image.assign(img_image_.begin(), img_image_.begin() + image_elements);
    out.transmittance.assign(img_trans_.begin(), img_trans_.begin() + pixel_count);
    out.opacity.assign(img_opacity_.begin(), img_opacity_.begin() + pixel_count);
    out.depth.assign(img_depth_.begin(), img_depth_.begin() + pixel_count);
    out.hitmask.assign(img_hitmask_.begin(), img_hitmask_.begin() + pixel_count);
    out.ray_count = last_ray_count_;
    out.sample_count = last_sample_count_;

    auto total_end = std::chrono::steady_clock::now();
    stats.total_ms = DurationMilliseconds(total_begin, total_end);
    out.stats = std::move(stats);
    return Status::Ok();
}

Status Renderer::Backward(DenseGridField& field,
                          std::span<const float> dL_dI,
                          BackwardResult& out) {
    if (!field.valid()) {
        return Status(StatusCode::kInvalidArgument, "field is invalid");
    }
    if (last_ray_count_ == 0 || last_sample_count_ == 0) {
        return Status(StatusCode::kInvalidArgument, "forward pass not executed or produced zero samples");
    }

    const size_t expected = last_ray_count_ * 3ULL;
    if (dL_dI.size() != expected) {
        return Status(StatusCode::kInvalidArgument, "dL/dI size mismatch");
    }

    const size_t sample_count = last_sample_count_;
    std::fill(grad_sigma_samples_.begin(), grad_sigma_samples_.begin() + sample_count, 0.0f);
    std::fill(grad_color_samples_.begin(), grad_color_samples_.begin() + sample_count * 3ULL, 0.0f);
    std::fill(grad_camera_.begin(), grad_camera_.end(), 0.0f);

    hp_tensor grad_tensor = MakeHostTensor(
        const_cast<float*>(dL_dI.data()),
        HP_DTYPE_F32,
        {static_cast<int64_t>(last_ray_count_), 3});

    hp_status hp = hp_diff(plan_->handle(),
                           &grad_tensor,
                           &samp_,
                           &intl_,
                           &grads_, workspace_buffer_.data(), workspace_buffer_.size());
    if (hp != HP_STATUS_SUCCESS) {
        return Status::FromHotpath(hp, "hp_diff failed");
    }

    field.ZeroGradients();
    std::span<const float> sigma_span(grad_sigma_samples_.data(), sample_count);
    std::span<const float> color_span(grad_color_samples_.data(), sample_count * 3ULL);
    Status accumulate = field.AccumulateSampleGradients(samp_, sigma_span, color_span);
    if (!accumulate.ok()) {
        return accumulate;
    }

    out.sigma = field.sigma_gradients();
    out.color = field.color_gradients();
    std::copy(grad_camera_.begin(), grad_camera_.end(), out.camera.begin());
    out.sample_count = sample_count;
    return Status::Ok();
}

size_t Renderer::RequiredWorkspaceBytes(size_t ray_capacity, size_t sample_capacity) const {
    const size_t sample_bytes =
        sample_capacity * 3ULL * sizeof(float) +
        sample_capacity * sizeof(float) +
        sample_capacity * sizeof(float) +
        sample_capacity * 3ULL * sizeof(float) +
        (ray_capacity + 1ULL) * sizeof(uint32_t);

    const size_t intl_bytes =
        ray_capacity * 3ULL * sizeof(float) +
        ray_capacity * sizeof(float) +
        ray_capacity * sizeof(float) +
        ray_capacity * sizeof(float) +
        sample_capacity * 4ULL * sizeof(float);

    return sample_bytes + intl_bytes;
}

void Renderer::CopyOutputsFromWorkspace(size_t ray_count, size_t sample_count) {
    if (sample_count > 0) {
        std::memcpy(sample_positions_.data(), samp_.positions.data, sample_count * 3ULL * sizeof(float));
        std::memcpy(sample_dt_.data(), samp_.dt.data, sample_count * sizeof(float));
        std::memcpy(sample_sigma_.data(), samp_.sigma.data, sample_count * sizeof(float));
        std::memcpy(sample_color_.data(), samp_.color.data, sample_count * 3ULL * sizeof(float));
        std::memcpy(intl_aux_.data(), intl_.aux.data, sample_count * 4ULL * sizeof(float));
    }

    if (ray_count > 0) {
        std::memcpy(sample_offsets_.data(), samp_.ray_offset.data, (ray_count + 1ULL) * sizeof(uint32_t));
        std::memcpy(intl_radiance_.data(), intl_.radiance.data, ray_count * 3ULL * sizeof(float));
        std::memcpy(intl_trans_.data(), intl_.transmittance.data, ray_count * sizeof(float));
        std::memcpy(intl_opacity_.data(), intl_.opacity.data, ray_count * sizeof(float));
        std::memcpy(intl_depth_.data(), intl_.depth.data, ray_count * sizeof(float));
    }

    samp_.positions.data = sample_positions_.empty() ? nullptr : sample_positions_.data();
    samp_.dt.data = sample_dt_.empty() ? nullptr : sample_dt_.data();
    samp_.sigma.data = sample_sigma_.empty() ? nullptr : sample_sigma_.data();
    samp_.color.data = sample_color_.empty() ? nullptr : sample_color_.data();
    samp_.ray_offset.data = sample_offsets_.empty() ? nullptr : sample_offsets_.data();

    intl_.radiance.data = intl_radiance_.empty() ? nullptr : intl_radiance_.data();
    intl_.transmittance.data = intl_trans_.empty() ? nullptr : intl_trans_.data();
    intl_.opacity.data = intl_opacity_.empty() ? nullptr : intl_opacity_.data();
    intl_.depth.data = intl_depth_.empty() ? nullptr : intl_depth_.data();
    intl_.aux.data = intl_aux_.empty() ? nullptr : intl_aux_.data();
}

WorkspaceInfo Renderer::workspace_info() const {
    WorkspaceInfo info{};
    info.ray_buffer_bytes =
        ray_origins_.size() * sizeof(float) +
        ray_dirs_.size() * sizeof(float) +
        ray_t_near_.size() * sizeof(float) +
        ray_t_far_.size() * sizeof(float) +
        ray_pixels_.size() * sizeof(uint32_t);

    info.sample_buffer_bytes =
        sample_positions_.size() * sizeof(float) +
        sample_dt_.size() * sizeof(float) +
        sample_sigma_.size() * sizeof(float) +
        sample_color_.size() * sizeof(float) +
        sample_offsets_.size() * sizeof(uint32_t);

    info.integration_buffer_bytes =
        intl_radiance_.size() * sizeof(float) +
        intl_trans_.size() * sizeof(float) +
        intl_opacity_.size() * sizeof(float) +
        intl_depth_.size() * sizeof(float) +
        intl_aux_.size() * sizeof(float);

    info.image_buffer_bytes =
        img_image_.size() * sizeof(float) +
        img_trans_.size() * sizeof(float) +
        img_opacity_.size() * sizeof(float) +
        img_depth_.size() * sizeof(float) +
        img_hitmask_.size() * sizeof(uint32_t);

    info.gradient_buffer_bytes =
        grad_sigma_samples_.size() * sizeof(float) +
        grad_color_samples_.size() * sizeof(float) +
        grad_camera_.size() * sizeof(float);

    return info;
}

}  // namespace dvren




