#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <span>

#include "dvren/core/context.hpp"
#include "dvren/core/status.hpp"

namespace dvren {

struct DenseGridConfig {
    std::array<int32_t, 3> resolution{1, 1, 1};
    std::vector<float> sigma;
    std::vector<float> color;
    std::array<float, 3> bbox_min{0.0f, 0.0f, 0.0f};
    std::array<float, 3> bbox_max{1.0f, 1.0f, 1.0f};
    hp_interp_mode interp{HP_INTERP_LINEAR};
    hp_oob_policy oob{HP_OOB_ZERO};
};

class DenseGridField {
public:
    DenseGridField() = default;
    ~DenseGridField();

    DenseGridField(DenseGridField&& other) noexcept;
    DenseGridField& operator=(DenseGridField&& other) noexcept;

    DenseGridField(const DenseGridField&) = delete;
    DenseGridField& operator=(const DenseGridField&) = delete;

    static Status Create(const Context& ctx, const DenseGridConfig& config, DenseGridField& out);

    [[nodiscard]] bool valid() const { return sigma_field_ != nullptr && color_field_ != nullptr; }
    [[nodiscard]] const hp_field* sigma_field() const { return sigma_field_; }
    [[nodiscard]] hp_field* sigma_field() { return sigma_field_; }
    [[nodiscard]] const hp_field* color_field() const { return color_field_; }
    [[nodiscard]] hp_field* color_field() { return color_field_; }
    [[nodiscard]] std::array<int32_t, 3> resolution() const { return resolution_; }
    [[nodiscard]] std::array<float, 3> bbox_min() const { return bbox_min_; }
    [[nodiscard]] std::array<float, 3> bbox_max() const { return bbox_max_; }
    [[nodiscard]] hp_interp_mode interpolation() const { return interp_; }
    [[nodiscard]] hp_oob_policy oob_policy() const { return oob_; }

    void ZeroGradients();
    Status AccumulateSampleGradients(const hp_samp_t& samples,
                                     std::span<const float> grad_sigma,
                                     std::span<const float> grad_color);

    [[nodiscard]] const std::vector<float>& sigma_gradients() const { return sigma_grad_; }
    [[nodiscard]] const std::vector<float>& color_gradients() const { return color_grad_; }
    [[nodiscard]] size_t voxel_count() const { return static_cast<size_t>(resolution_[0]) * static_cast<size_t>(resolution_[1]) * static_cast<size_t>(resolution_[2]); }

private:
    void Reset(hp_field* sigma,
               hp_field* color,
               std::array<int32_t, 3> resolution,
               const DenseGridConfig& config);
    void Release();

    hp_field* sigma_field_{nullptr};
    hp_field* color_field_{nullptr};
    std::array<int32_t, 3> resolution_{1, 1, 1};
    std::array<float, 3> bbox_min_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> bbox_max_{1.0f, 1.0f, 1.0f};
    hp_interp_mode interp_{HP_INTERP_LINEAR};
    hp_oob_policy oob_{HP_OOB_ZERO};

    std::vector<float> sigma_data_;
    std::vector<float> color_data_;
    std::vector<float> sigma_grad_;
    std::vector<float> color_grad_;
};

}  // namespace dvren

