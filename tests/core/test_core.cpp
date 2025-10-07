#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/fields/dense_grid.hpp"
#include "dvren/render/renderer.hpp"

namespace {

int Fail(const std::string& message) {
    std::cerr << message << std::endl;
    return 1;
}

}  // namespace

int main() {
    dvren::Context ctx;
    dvren::Status status = dvren::Context::Create(dvren::ContextOptions{}, ctx);
    if (!status.ok()) {
        return Fail(std::string("Context::Create failed: ") + status.ToString());
    }

    dvren::PlanDescriptor plan_desc{};
    plan_desc.width = 4;
    plan_desc.height = 4;
    plan_desc.t_near = 0.0f;
    plan_desc.t_far = 1.0f;
    plan_desc.max_rays = plan_desc.width * plan_desc.height;
    plan_desc.max_samples = plan_desc.max_rays * plan_desc.sampling.max_steps;
    plan_desc.sampling.dt = 0.05f;
    plan_desc.sampling.max_steps = 32;
    plan_desc.sampling.mode = dvren::SamplingMode::kFixed;

    dvren::Plan plan;
    status = dvren::Plan::Create(ctx, plan_desc, plan);
    if (!status.ok()) {
        return Fail(std::string("Plan::Create failed: ") + status.ToString());
    }

    dvren::DenseGridConfig grid_config{};
    grid_config.resolution = {2, 2, 2};
    grid_config.bbox_min = {0.0f, 0.0f, 0.0f};
    grid_config.bbox_max = {1.0f, 1.0f, 1.0f};
    grid_config.sigma = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    grid_config.color = {
        1.0f, 0.5f, 0.2f,
        0.2f, 0.5f, 1.0f,
        0.4f, 0.9f, 0.1f,
        0.6f, 0.4f, 0.3f,
        0.3f, 0.7f, 0.8f,
        0.9f, 0.2f, 0.5f,
        0.7f, 0.1f, 0.6f,
        0.5f, 0.8f, 0.4f
    };

    dvren::DenseGridField field;
    status = dvren::DenseGridField::Create(ctx, grid_config, field);
    if (!status.ok()) {
        return Fail(std::string("DenseGridField::Create failed: ") + status.ToString());
    }

    dvren::Renderer renderer(ctx, plan, dvren::RenderOptions{});

    dvren::ForwardResult forward;
    status = renderer.Forward(field, forward);
    if (!status.ok()) {
        return Fail(std::string("Renderer::Forward failed: ") + status.ToString());
    }

    if (forward.image.empty() || forward.ray_count == 0 || forward.sample_count == 0) {
        return Fail("Forward result is empty");
    }

    const size_t grad_size = forward.ray_count * 3ULL;
    std::vector<float> dL_dI(grad_size, 1.0f);

    dvren::BackwardResult backward;
    status = renderer.Backward(field, dL_dI, backward);
    if (!status.ok()) {
        return Fail(std::string("Renderer::Backward failed: ") + status.ToString());
    }

    const size_t voxel_count = field.voxel_count();
    if (backward.sigma.size() != voxel_count) {
        return Fail("Sigma gradient size mismatch");
    }
    if (backward.color.size() != voxel_count * 3ULL) {
        return Fail("Color gradient size mismatch");
    }

    const float sigma_sum = std::accumulate(backward.sigma.begin(), backward.sigma.end(), 0.0f);
    if (!(sigma_sum > 0.0f)) {
        return Fail("Sigma gradients did not accumulate positive values");
    }

    return 0;
}


