#include <algorithm>
#include <cmath>
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

struct RunOutputs {
    dvren::ForwardResult forward;
    dvren::BackwardResult backward;
    dvren::WorkspaceInfo workspace;
};

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
    plan_desc.sampling.dt = 0.05f;
    plan_desc.sampling.max_steps = 32;
    plan_desc.sampling.mode = dvren::SamplingMode::kFixed;
    plan_desc.max_rays = plan_desc.width * plan_desc.height;
    plan_desc.max_samples = plan_desc.max_rays * plan_desc.sampling.max_steps;

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

    auto execute_run = [&](const dvren::RenderOptions& options, RunOutputs& outputs) -> dvren::Status {
        dvren::DenseGridField field;
        dvren::Status create_status = dvren::DenseGridField::Create(ctx, grid_config, field);
        if (!create_status.ok()) {
            return create_status;
        }

        dvren::Renderer renderer(ctx, plan, options);
        dvren::Status forward_status = renderer.Forward(field, outputs.forward);
        if (!forward_status.ok()) {
            return forward_status;
        }

        if (outputs.forward.image.empty() ||
            outputs.forward.ray_count == 0 ||
            outputs.forward.sample_count == 0) {
            return dvren::Status(dvren::StatusCode::kInternalError, "forward result invalid");
        }

        outputs.workspace = renderer.workspace_info();
        if (outputs.workspace.total_bytes() == 0) {
            return dvren::Status(dvren::StatusCode::kInternalError, "workspace info empty");
        }

        const size_t grad_size = outputs.forward.ray_count * 3ULL;
        std::vector<float> dL_dI(grad_size, 1.0f);

        dvren::Status backward_status = renderer.Backward(field, dL_dI, outputs.backward);
        if (!backward_status.ok()) {
            return backward_status;
        }

        return dvren::Status::Ok();
    };

    dvren::RenderOptions staged_opts{};
    staged_opts.use_fused_path = false;

    dvren::RenderOptions fused_opts{};
    fused_opts.use_fused_path = true;

    dvren::RenderOptions graph_opts{};
    graph_opts.use_fused_path = false;
    graph_opts.enable_graph = true;

    RunOutputs staged_outputs{};
    RunOutputs fused_outputs{};
    RunOutputs graph_outputs{};

    status = execute_run(staged_opts, staged_outputs);
    if (!status.ok()) {
        return Fail(std::string("Staged run failed: ") + status.ToString());
    }

    status = execute_run(fused_opts, fused_outputs);
    if (!status.ok()) {
        return Fail(std::string("Fused run failed: ") + status.ToString());
    }

    status = execute_run(graph_opts, graph_outputs);
    if (!status.ok()) {
        return Fail(std::string("Graph run failed: ") + status.ToString());
    }

    const auto compare_vectors = [](const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    float tolerance) -> bool {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::fabs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    };

    if (!compare_vectors(staged_outputs.forward.image, fused_outputs.forward.image, 1e-4f)) {
        return Fail("Forward images diverged between staged and fused paths");
    }
    if (!compare_vectors(staged_outputs.backward.sigma, fused_outputs.backward.sigma, 1e-4f)) {
        return Fail("Sigma gradients diverged between staged and fused paths");
    }
    if (!compare_vectors(staged_outputs.backward.color, fused_outputs.backward.color, 1e-4f)) {
        return Fail("Color gradients diverged between staged and fused paths");
    }

    const float sigma_sum = std::accumulate(staged_outputs.backward.sigma.begin(),
                                            staged_outputs.backward.sigma.end(),
                                            0.0f);
    if (!(sigma_sum > 0.0f)) {
        return Fail("Sigma gradients did not accumulate positive values");
    }

    if (!(graph_outputs.workspace.total_bytes() >= staged_outputs.workspace.total_bytes())) {
        return Fail("Graph workspace accounting invalid");
    }

    return 0;
}

