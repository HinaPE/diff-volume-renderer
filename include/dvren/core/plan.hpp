#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "dvren/core/context.hpp"
#include "dvren/core/status.hpp"
#include "hotpath/hp.h"

namespace dvren {

enum class CameraModel {
    kPinhole = 0,
    kOrthographic = 1
};

struct CameraDesc {
    CameraModel model{CameraModel::kPinhole};
    std::array<float, 9> K{1.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f,
                           0.0f, 0.0f, 1.0f};
    std::array<float, 12> c2w{1.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 0.0f};
    float ortho_scale{1.0f};
};

struct Roi {
    uint32_t x{0};
    uint32_t y{0};
    uint32_t width{0};
    uint32_t height{0};
};

enum class SamplingMode {
    kFixed = 0,
    kStratified = 1
};

struct SamplingConfig {
    float dt{0.0f};
    uint32_t max_steps{0};
    SamplingMode mode{SamplingMode::kFixed};
};

struct PlanDescriptor {
    uint32_t width{0};
    uint32_t height{0};
    float t_near{0.0f};
    float t_far{1.0f};
    SamplingConfig sampling{};
    std::optional<Roi> roi{};
    uint32_t max_rays{0};
    uint32_t max_samples{0};
    uint64_t seed{0};
    CameraDesc camera{};
};

class Plan {
public:
    Plan() = default;
    ~Plan();

    Plan(Plan&& other) noexcept;
    Plan& operator=(Plan&& other) noexcept;

    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    static Status Create(const Context& ctx, const PlanDescriptor& descriptor, Plan& out);

    [[nodiscard]] bool valid() const { return plan_ != nullptr; }
    [[nodiscard]] const hp_plan* handle() const { return plan_; }
    [[nodiscard]] hp_plan* handle() { return plan_; }
    [[nodiscard]] const hp_plan_desc& descriptor() const { return desc_; }

private:
    void Reset(hp_plan* plan, const hp_plan_desc& desc);

    hp_plan* plan_{nullptr};
    hp_plan_desc desc_{};
};

}  // namespace dvren

