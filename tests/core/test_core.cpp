#include <iostream>
#include <vector>

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/fields/dense_grid.hpp"
#include "hotpath/hp.h"

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
    plan_desc.width = 2;
    plan_desc.height = 2;
    plan_desc.t_near = 0.0f;
    plan_desc.t_far = 1.0f;
    plan_desc.sampling.dt = 0.1f;
    plan_desc.sampling.max_steps = 8;
    plan_desc.sampling.mode = dvren::SamplingMode::kFixed;

    dvren::Plan plan;
    status = dvren::Plan::Create(ctx, plan_desc, plan);
    if (!status.ok()) {
        return Fail(std::string("Plan::Create failed: ") + status.ToString());
    }

    dvren::DenseGridConfig grid_config{};
    grid_config.resolution = {2, 2, 2};
    grid_config.sigma.assign(8, 0.5f);
    grid_config.color.assign(8 * 3, 0.25f);

    dvren::DenseGridField field;
    status = dvren::DenseGridField::Create(ctx, grid_config, field);
    if (!status.ok()) {
        return Fail(std::string("DenseGridField::Create failed: ") + status.ToString());
    }

    const hp_plan_desc& desc = plan.descriptor();
    const size_t ray_capacity = static_cast<size_t>(desc.max_rays);
    const size_t sample_capacity = static_cast<size_t>(desc.max_samples);
    if (ray_capacity == 0 || sample_capacity == 0) {
        return Fail("plan capacities are zero");
    }

    std::vector<float> ray_origins(ray_capacity * 3ULL);
    std::vector<float> ray_dirs(ray_capacity * 3ULL);
    std::vector<float> ray_t_near(ray_capacity);
    std::vector<float> ray_t_far(ray_capacity);
    std::vector<uint32_t> ray_pixel_ids(ray_capacity);

    hp_rays_t rays{};
    rays.origins.data = ray_origins.data();
    rays.directions.data = ray_dirs.data();
    rays.t_near.data = ray_t_near.data();
    rays.t_far.data = ray_t_far.data();
    rays.pixel_ids.data = ray_pixel_ids.data();

    std::vector<float> sample_positions(sample_capacity * 3ULL);
    std::vector<float> sample_dt(sample_capacity);
    std::vector<float> sample_sigma(sample_capacity);
    std::vector<float> sample_color(sample_capacity * 3ULL);
    std::vector<uint32_t> sample_offsets(ray_capacity + 1ULL);

    hp_samp_t samp{};
    samp.positions.data = sample_positions.data();
    samp.dt.data = sample_dt.data();
    samp.sigma.data = sample_sigma.data();
    samp.color.data = sample_color.data();
    samp.ray_offset.data = sample_offsets.data();

    std::vector<float> intl_radiance(ray_capacity * 3ULL);
    std::vector<float> intl_trans(ray_capacity);
    std::vector<float> intl_opacity(ray_capacity);
    std::vector<float> intl_depth(ray_capacity);
    std::vector<float> intl_aux(sample_capacity * 4ULL);

    hp_intl_t intl{};
    intl.radiance.data = intl_radiance.data();
    intl.transmittance.data = intl_trans.data();
    intl.opacity.data = intl_opacity.data();
    intl.depth.data = intl_depth.data();
    intl.aux.data = intl_aux.data();

    std::vector<float> img_image(static_cast<size_t>(desc.width) * desc.height * 3ULL);
    std::vector<float> img_trans(static_cast<size_t>(desc.width) * desc.height);
    std::vector<float> img_opacity(static_cast<size_t>(desc.width) * desc.height);
    std::vector<float> img_depth(static_cast<size_t>(desc.width) * desc.height);
    std::vector<uint32_t> img_hitmask(static_cast<size_t>(desc.width) * desc.height);

    hp_img_t img{};
    img.image.data = img_image.data();
    img.trans.data = img_trans.data();
    img.opacity.data = img_opacity.data();
    img.depth.data = img_depth.data();
    img.hitmask.data = img_hitmask.data();

    if (hp_ray(plan.handle(), nullptr, &rays, nullptr, 0) != HP_STATUS_SUCCESS) {
        return Fail("hp_ray failed");
    }
    if (hp_samp(plan.handle(), field.sigma_field(), field.color_field(), &rays, &samp, nullptr, 0) != HP_STATUS_SUCCESS) {
        return Fail("hp_samp failed");
    }
    if (hp_int(plan.handle(), &samp, &intl, nullptr, 0) != HP_STATUS_SUCCESS) {
        return Fail("hp_int failed");
    }
    if (hp_img(plan.handle(), &intl, &rays, &img, nullptr, 0) != HP_STATUS_SUCCESS) {
        return Fail("hp_img failed");
    }

    if (img_image.empty()) {
        return Fail("image buffer is empty");
    }
    return 0;
}

