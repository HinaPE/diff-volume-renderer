#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/core/status.hpp"
#include "dvren/fields/dense_grid.hpp"

namespace dvren {

struct RenderOptions {
    bool use_fused_path{true};
    bool enable_graph{false};
    bool capture_stats{true};
};

struct WorkspaceInfo {
    size_t ray_buffer_bytes{0};
    size_t sample_buffer_bytes{0};
    size_t integration_buffer_bytes{0};
    size_t image_buffer_bytes{0};
    size_t gradient_buffer_bytes{0};
    size_t workspace_buffer_bytes{0};

    [[nodiscard]] size_t total_bytes() const {
        return ray_buffer_bytes +
               sample_buffer_bytes +
               integration_buffer_bytes +
               image_buffer_bytes +
               gradient_buffer_bytes +
               workspace_buffer_bytes;
    }
};

struct RenderStats {
    double total_ms{0.0};
    double ray_ms{0.0};
    double sample_ms{0.0};
    double integrate_ms{0.0};
    double compose_ms{0.0};
    std::vector<std::string> notes;
};

struct ForwardResult {
    std::vector<float> image;
    std::vector<float> transmittance;
    std::vector<float> opacity;
    std::vector<float> depth;
    std::vector<uint32_t> hitmask;
    size_t ray_count{0};
    size_t sample_count{0};
    RenderStats stats;
};

struct BackwardResult {
    std::vector<float> sigma;
    std::vector<float> color;
    std::array<float, 12> camera{};
    size_t sample_count{0};
};

class Renderer {
public:
    Renderer(const Context& ctx, const Plan& plan, RenderOptions options = {});
    ~Renderer();

    Status Forward(const DenseGridField& field, ForwardResult& out);
    Status Backward(DenseGridField& field,
                    std::span<const float> dL_dI,
                    BackwardResult& out);

    [[nodiscard]] WorkspaceInfo workspace_info() const;
    [[nodiscard]] const RenderOptions& options() const { return options_; }

private:
    Status EnsureCapacity();
    void ConfigureRayTensors(size_t ray_capacity);
    void ConfigureSampleTensors(size_t sample_capacity, size_t ray_capacity);
    void ConfigureIntegrationTensors(size_t ray_capacity, size_t sample_capacity);
    void ConfigureImageTensors(size_t pixel_count);
    void ConfigureGradientTensors(size_t sample_capacity);
    size_t RequiredWorkspaceBytes(size_t ray_capacity, size_t sample_capacity) const;
    void CopyOutputsFromWorkspace(size_t ray_count, size_t sample_count);
#if defined(HP_WITH_CUDA)
    Status CaptureGraphForward(const DenseGridField& field);
    Status CaptureGraphBackward(const DenseGridField& field, const hp_tensor& grad_tensor);
    void ReleaseGraph();
#endif

    const Context* ctx_{nullptr};
    const Plan* plan_{nullptr};
    RenderOptions options_{};

    std::vector<float> ray_origins_;
    std::vector<float> ray_dirs_;
    std::vector<float> ray_t_near_;
    std::vector<float> ray_t_far_;
    std::vector<uint32_t> ray_pixels_;

    std::vector<float> sample_positions_;
    std::vector<float> sample_dt_;
    std::vector<float> sample_sigma_;
    std::vector<float> sample_color_;
    std::vector<uint32_t> sample_offsets_;

    std::vector<float> intl_radiance_;
    std::vector<float> intl_trans_;
    std::vector<float> intl_opacity_;
    std::vector<float> intl_depth_;
    std::vector<float> intl_aux_;

    std::vector<float> img_image_;
    std::vector<float> img_trans_;
    std::vector<float> img_opacity_;
    std::vector<float> img_depth_;
    std::vector<uint32_t> img_hitmask_;

    std::vector<float> grad_sigma_samples_;
    std::vector<float> grad_color_samples_;
    std::vector<float> grad_camera_;

    std::vector<std::byte> workspace_buffer_;

    hp_rays_t rays_{};
    hp_samp_t samp_{};
    hp_intl_t intl_{};
    hp_img_t img_{};
    hp_grads_t grads_{};

    size_t current_ray_capacity_{0};
    size_t current_sample_capacity_{0};
    size_t current_pixel_capacity_{0};

    size_t last_ray_count_{0};
    size_t last_sample_count_{0};

#if defined(HP_WITH_CUDA)
    void* graph_handle_{nullptr};
    bool graph_forward_captured_{false};
    bool graph_backward_captured_{false};
    std::string graph_last_error_;
#endif
};

}  // namespace dvren

