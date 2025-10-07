#include "dvren/core/plan.hpp"

#include <algorithm>
#include <cstring>

namespace dvren {

namespace {

hp_camera_model ToHotpathCameraModel(CameraModel model) {
    switch (model) {
        case CameraModel::kOrthographic:
            return HP_CAMERA_ORTHOGRAPHIC;
        case CameraModel::kPinhole:
        default:
            return HP_CAMERA_PINHOLE;
    }
}

hp_sampling_mode ToHotpathSamplingMode(SamplingMode mode) {
    switch (mode) {
        case SamplingMode::kStratified:
            return HP_SAMPLING_STRATIFIED;
        case SamplingMode::kFixed:
        default:
            return HP_SAMPLING_FIXED;
    }
}

}  // namespace

Plan::~Plan() {
    if (plan_ != nullptr) {
        hp_plan_release(plan_);
        plan_ = nullptr;
    }
}

Plan::Plan(Plan&& other) noexcept {
    plan_ = other.plan_;
    desc_ = other.desc_;
    other.plan_ = nullptr;
}

Plan& Plan::operator=(Plan&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (plan_ != nullptr) {
        hp_plan_release(plan_);
    }
    plan_ = other.plan_;
    desc_ = other.desc_;
    other.plan_ = nullptr;
    return *this;
}

Status Plan::Create(const Context& ctx, const PlanDescriptor& descriptor, Plan& out) {
    if (!ctx.valid()) {
        return Status(StatusCode::kInvalidArgument, "context is invalid");
    }

    hp_plan_desc plan_desc{};
    plan_desc.width = descriptor.width;
    plan_desc.height = descriptor.height;
    plan_desc.t_near = descriptor.t_near;
    plan_desc.t_far = descriptor.t_far;
    plan_desc.max_rays = descriptor.max_rays;
    plan_desc.max_samples = descriptor.max_samples;
    plan_desc.seed = descriptor.seed;

    plan_desc.sampling.dt = descriptor.sampling.dt;
    plan_desc.sampling.max_steps = descriptor.sampling.max_steps;
    plan_desc.sampling.mode = ToHotpathSamplingMode(descriptor.sampling.mode);

    plan_desc.camera.model = ToHotpathCameraModel(descriptor.camera.model);
    plan_desc.camera.ortho_scale = descriptor.camera.ortho_scale;
    std::memcpy(plan_desc.camera.K, descriptor.camera.K.data(), descriptor.camera.K.size() * sizeof(float));
    std::memcpy(plan_desc.camera.c2w, descriptor.camera.c2w.data(), descriptor.camera.c2w.size() * sizeof(float));

    if (descriptor.roi.has_value()) {
        plan_desc.roi = hp_roi_desc{
            descriptor.roi->x,
            descriptor.roi->y,
            descriptor.roi->width,
            descriptor.roi->height
        };
    } else {
        plan_desc.roi = hp_roi_desc{0, 0, 0, 0};
    }

    hp_plan* plan_ptr = nullptr;
    const hp_status create_status = hp_plan_create(ctx.handle(), &plan_desc, &plan_ptr);
    if (create_status != HP_STATUS_SUCCESS || plan_ptr == nullptr) {
        return Status::FromHotpath(create_status, "hp_plan_create failed");
    }

    hp_plan_desc actual_desc{};
    const hp_status get_status = hp_plan_get_desc(plan_ptr, &actual_desc);
    if (get_status != HP_STATUS_SUCCESS) {
        hp_plan_release(plan_ptr);
        return Status::FromHotpath(get_status, "hp_plan_get_desc failed");
    }

    out.Reset(plan_ptr, actual_desc);
    return Status::Ok();
}

void Plan::Reset(hp_plan* plan, const hp_plan_desc& desc) {
    if (plan_ != nullptr) {
        hp_plan_release(plan_);
    }
    plan_ = plan;
    desc_ = desc;
}

}  // namespace dvren

