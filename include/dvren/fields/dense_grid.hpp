#pragma once

#include <array>
#include <cstdint>
#include <vector>

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

private:
    void Reset(hp_field* sigma, hp_field* color, std::array<int32_t, 3> resolution);
    void Release();

    hp_field* sigma_field_{nullptr};
    hp_field* color_field_{nullptr};
    std::array<int32_t, 3> resolution_{1, 1, 1};
};

}  // namespace dvren

