#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#if defined(HP_WITH_CUDA)
#include <cuda_runtime.h>
#endif

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/fields/dense_grid.hpp"
#include "dvren/render/renderer.hpp"
#include "hotpath/hp.h"
#include "smoke_test_utils.hpp"

namespace {

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

#if defined(HP_WITH_CUDA)
struct DeviceAllocation {
    void* ptr{nullptr};

    DeviceAllocation() = default;
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    DeviceAllocation(DeviceAllocation&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    ~DeviceAllocation() {
        release();
    }

    cudaError_t allocate(size_t bytes) {
        release();
        if (bytes == 0) {
            ptr = nullptr;
            return cudaSuccess;
        }
        return cudaMalloc(&ptr, bytes);
    }

    void release() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
};

bool CheckCuda(cudaError_t err, const char* label) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << label << "): " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

size_t ComputeWorkspaceBytes(const hp_plan_desc& desc) {
    const size_t roi_width = desc.roi.width > 0 ? desc.roi.width : desc.width;
    const size_t roi_height = desc.roi.height > 0 ? desc.roi.height : desc.height;
    const size_t ray_capacity = std::max<size_t>(desc.max_rays, roi_width * roi_height);
    const size_t sample_capacity = static_cast<size_t>(desc.max_samples);

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

bool ValidateCudaForwardParity(const dvren::Plan& plan,
                               const dvren::DenseGridField& field,
                               const dvren::ForwardResult& cpu_forward) {
    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err == cudaErrorNoDevice || device_err == cudaErrorInsufficientDriver) {
        std::cout << "Skipping CUDA parity check: " << cudaGetErrorString(device_err) << std::endl;
        cudaGetLastError();
        return true;
    }
    if (device_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(device_err) << std::endl;
        return false;
    }
    if (device_count == 0) {
        std::cout << "Skipping CUDA parity check: no CUDA devices reported." << std::endl;
        return true;
    }

    const hp_plan_desc& desc = plan.descriptor();
    const size_t roi_width = desc.roi.width > 0 ? desc.roi.width : desc.width;
    const size_t roi_height = desc.roi.height > 0 ? desc.roi.height : desc.height;
    const size_t ray_count = roi_width * roi_height;
    const size_t pixel_count = static_cast<size_t>(desc.width) * static_cast<size_t>(desc.height);
    const size_t workspace_bytes = ComputeWorkspaceBytes(desc);
    if (workspace_bytes == 0) {
        std::cerr << "CUDA workspace requirement computed as zero." << std::endl;
        return false;
    }

    DeviceAllocation workspace;
    if (!CheckCuda(workspace.allocate(workspace_bytes), "cudaMalloc workspace")) {
        return false;
    }

    const size_t vec3_bytes = ray_count * 3ULL * sizeof(float);
    const size_t scalar_bytes = ray_count * sizeof(float);
    const size_t ids_bytes = ray_count * sizeof(uint32_t);

    DeviceAllocation origins;
    DeviceAllocation directions;
    DeviceAllocation t_near;
    DeviceAllocation t_far;
    DeviceAllocation pixel_ids;

    if (!CheckCuda(origins.allocate(vec3_bytes), "cudaMalloc ray origins") ||
        !CheckCuda(directions.allocate(vec3_bytes), "cudaMalloc ray directions") ||
        !CheckCuda(t_near.allocate(scalar_bytes), "cudaMalloc ray t_near") ||
        !CheckCuda(t_far.allocate(scalar_bytes), "cudaMalloc ray t_far") ||
        !CheckCuda(pixel_ids.allocate(ids_bytes), "cudaMalloc ray pixel ids")) {
        return false;
    }

    hp_rays_t rays{};
    rays.origins.data = origins.ptr;
    rays.origins.memspace = HP_MEMSPACE_DEVICE;
    rays.directions.data = directions.ptr;
    rays.directions.memspace = HP_MEMSPACE_DEVICE;
    rays.t_near.data = t_near.ptr;
    rays.t_near.memspace = HP_MEMSPACE_DEVICE;
    rays.t_far.data = t_far.ptr;
    rays.t_far.memspace = HP_MEMSPACE_DEVICE;
    rays.pixel_ids.data = pixel_ids.ptr;
    rays.pixel_ids.memspace = HP_MEMSPACE_DEVICE;

    hp_status hp = hp_ray(plan.handle(), nullptr, &rays, nullptr, 0);
    if (hp != HP_STATUS_SUCCESS) {
        std::cerr << dvren::Status::FromHotpath(hp, "hp_ray (CUDA)").ToString() << std::endl;
        return false;
    }
    if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after hp_ray")) {
        return false;
    }

    hp_samp_t samp{};
    hp_intl_t intl{};
    hp = hp_samp_int_fused(plan.handle(),
                           field.sigma_field(),
                           field.color_field(),
                           &rays,
                           &samp,
                           &intl,
                           workspace.ptr,
                           workspace_bytes);
    if (hp != HP_STATUS_SUCCESS) {
        std::cerr << dvren::Status::FromHotpath(hp, "hp_samp_int_fused (CUDA)").ToString() << std::endl;
        return false;
    }
    if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after hp_samp_int_fused")) {
        return false;
    }

    const size_t gpu_ray_count = (rays.t_near.rank >= 1) ? static_cast<size_t>(rays.t_near.shape[0]) : 0;
    const size_t gpu_sample_count = (samp.dt.rank >= 1) ? static_cast<size_t>(samp.dt.shape[0]) : 0;
    if (gpu_ray_count != cpu_forward.ray_count) {
        std::cerr << "CUDA ray count mismatch: got " << gpu_ray_count
                  << " expected " << cpu_forward.ray_count << std::endl;
        return false;
    }
    if (gpu_sample_count != cpu_forward.sample_count) {
        std::cerr << "CUDA sample count mismatch: got " << gpu_sample_count
                  << " expected " << cpu_forward.sample_count << std::endl;
        return false;
    }

    hp_img_t img{};
    hp = hp_img(plan.handle(), &intl, &rays, &img, workspace.ptr, workspace_bytes);
    if (hp != HP_STATUS_SUCCESS) {
        std::cerr << dvren::Status::FromHotpath(hp, "hp_img (CUDA)").ToString() << std::endl;
        return false;
    }
    if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after hp_img")) {
        return false;
    }

    std::vector<float> cuda_image(pixel_count * 3ULL, 0.0f);
    std::vector<float> cuda_trans(pixel_count, 0.0f);
    std::vector<float> cuda_opacity(pixel_count, 0.0f);
    std::vector<float> cuda_depth(pixel_count, 0.0f);
    std::vector<uint32_t> cuda_hitmask(pixel_count, 0U);

    if (!CheckCuda(cudaMemcpy(cuda_image.data(), img.image.data, cuda_image.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy image") ||
        !CheckCuda(cudaMemcpy(cuda_trans.data(), img.trans.data, cuda_trans.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy transmittance") ||
        !CheckCuda(cudaMemcpy(cuda_opacity.data(), img.opacity.data, cuda_opacity.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy opacity") ||
        !CheckCuda(cudaMemcpy(cuda_depth.data(), img.depth.data, cuda_depth.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy depth") ||
        !CheckCuda(cudaMemcpy(cuda_hitmask.data(), img.hitmask.data, cuda_hitmask.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy hitmask")) {
        return false;
    }

    const float image_diff = MaxAbsDiff(cuda_image, cpu_forward.image);
    if (image_diff > 2e-3f) {
        std::cerr << "CUDA image mismatch (max abs diff = " << image_diff << ")" << std::endl;
        return false;
    }
    const float trans_diff = MaxAbsDiff(cuda_trans, cpu_forward.transmittance);
    if (trans_diff > 2e-3f) {
        std::cerr << "CUDA transmittance mismatch (max abs diff = " << trans_diff << ")" << std::endl;
        return false;
    }
    const float opacity_diff = MaxAbsDiff(cuda_opacity, cpu_forward.opacity);
    if (opacity_diff > 2e-3f) {
        std::cerr << "CUDA opacity mismatch (max abs diff = " << opacity_diff << ")" << std::endl;
        return false;
    }
    const float depth_diff = MaxAbsDiff(cuda_depth, cpu_forward.depth);
    if (depth_diff > 1e-2f) {
        std::cerr << "CUDA depth mismatch (max abs diff = " << depth_diff << ")" << std::endl;
        return false;
    }

    for (size_t i = 0; i < cuda_hitmask.size(); ++i) {
        if (cuda_hitmask[i] != cpu_forward.hitmask[i]) {
            std::cerr << "CUDA hitmask mismatch at pixel index " << i << std::endl;
            return false;
        }
    }

    return true;
}
#endif

}  // namespace

int main() {
    dvren::Context ctx;
    dvren::Status status = dvren::Context::Create(dvren::ContextOptions{}, ctx);
    if (!status.ok()) {
        std::cerr << "Context::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::PlanDescriptor plan_desc{};
    plan_desc.width = 32;
    plan_desc.height = 32;
    plan_desc.t_near = 0.0f;
    plan_desc.t_far = 2.0f;
    plan_desc.sampling.dt = 0.02f;
    plan_desc.sampling.max_steps = 160;
    plan_desc.sampling.mode = dvren::SamplingMode::kFixed;
    plan_desc.max_rays = plan_desc.width * plan_desc.height;
    plan_desc.max_samples = plan_desc.max_rays * plan_desc.sampling.max_steps;
    plan_desc.seed = 1337;

    plan_desc.camera.model = dvren::CameraModel::kPinhole;
    plan_desc.camera.K = {48.0f, 0.0f, 16.0f,
                          0.0f, 48.0f, 16.0f,
                          0.0f, 0.0f, 1.0f};
    plan_desc.camera.c2w = {1.0f, 0.0f, 0.0f, 0.5f,
                            0.0f, 1.0f, 0.0f, 0.5f,
                            0.0f, 0.0f, 1.0f, -1.2f};

    dvren::Plan plan;
    status = dvren::Plan::Create(ctx, plan_desc, plan);
    if (!status.ok()) {
        std::cerr << "Plan::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::DenseGridConfig grid_config{};
    grid_config.resolution = {48, 48, 64};
    grid_config.bbox_min = {0.0f, 0.0f, 0.0f};
    grid_config.bbox_max = {1.0f, 1.0f, 1.0f};
    grid_config.interp = HP_INTERP_LINEAR;
    grid_config.oob = HP_OOB_ZERO;
    smoke_test::PopulateSmokeGrid(grid_config);

    dvren::DenseGridField field;
    status = dvren::DenseGridField::Create(ctx, grid_config, field);
    if (!status.ok()) {
        std::cerr << "DenseGridField::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::Renderer renderer(ctx, plan, dvren::RenderOptions{});
    dvren::ForwardResult forward{};
    status = renderer.Forward(field, forward);
    if (!status.ok()) {
        std::cerr << "Renderer::Forward failed: " << status.ToString() << std::endl;
        return 1;
    }

    const hp_plan_desc& actual_desc = plan.descriptor();

    if (forward.image.size() != static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height) * 3ULL) {
        std::cerr << "Forward image size mismatch" << std::endl;
        return 1;
    }
    if (forward.transmittance.size() != static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height)) {
        std::cerr << "Forward transmittance size mismatch" << std::endl;
        return 1;
    }
    if (forward.opacity.size() != forward.transmittance.size() ||
        forward.depth.size() != forward.transmittance.size() ||
        forward.hitmask.size() != forward.transmittance.size()) {
        std::cerr << "Forward auxiliary buffer size mismatch" << std::endl;
        return 1;
    }

    const size_t expected_ray_count = static_cast<size_t>(actual_desc.roi.width) * static_cast<size_t>(actual_desc.roi.height);
    if (forward.ray_count != expected_ray_count) {
        std::cerr << "Unexpected ray count: " << forward.ray_count << " expected " << expected_ray_count << std::endl;
        return 1;
    }
    if (forward.sample_count == 0) {
        std::cerr << "Forward produced zero samples" << std::endl;
        return 1;
    }

    const size_t pixel_count = static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height);
    std::vector<float> ref_image(pixel_count * 3ULL, 0.0f);
    std::vector<float> ref_trans(pixel_count, 0.0f);
    std::vector<float> ref_opacity(pixel_count, 0.0f);
    std::vector<float> ref_depth(pixel_count, actual_desc.t_far);

    for (uint32_t py = 0; py < actual_desc.height; ++py) {
        for (uint32_t px = 0; px < actual_desc.width; ++px) {
            const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(actual_desc.width) + px;
            const size_t base = idx * 3ULL;

            smoke_test::PixelEvaluation eval = smoke_test::IntegratePixel(actual_desc, grid_config, px, py);
            ref_image[base + 0] = eval.radiance[0];
            ref_image[base + 1] = eval.radiance[1];
            ref_image[base + 2] = eval.radiance[2];
            ref_trans[idx] = eval.transmittance;
            ref_opacity[idx] = eval.opacity;
            ref_depth[idx] = eval.depth;
        }
    }

    const float image_diff = MaxAbsDiff(forward.image, ref_image);
    const float trans_diff = MaxAbsDiff(forward.transmittance, ref_trans);
    const float opacity_diff = MaxAbsDiff(forward.opacity, ref_opacity);
    const float depth_diff = MaxAbsDiff(forward.depth, ref_depth);

    if (image_diff > 2e-3f) {
        std::cerr << "Image mismatch, max abs diff = " << image_diff << std::endl;
        return 1;
    }
    if (trans_diff > 2e-3f) {
        std::cerr << "Transmittance mismatch, max abs diff = " << trans_diff << std::endl;
        return 1;
    }
    if (opacity_diff > 2e-3f) {
        std::cerr << "Opacity mismatch, max abs diff = " << opacity_diff << std::endl;
        return 1;
    }
    if (depth_diff > 1e-2f) {
        std::cerr << "Depth mismatch, max abs diff = " << depth_diff << std::endl;
        return 1;
    }

    const auto image_extrema = std::minmax_element(forward.image.begin(), forward.image.end());
    if (!(image_extrema.second != forward.image.end() &&
          *image_extrema.second > *image_extrema.first + 5e-3f)) {
        std::cerr << "Forward image lacks intensity variation" << std::endl;
        return 1;
    }

    const auto trans_extrema = std::minmax_element(forward.transmittance.begin(), forward.transmittance.end());
    if (!(trans_extrema.second != forward.transmittance.end() &&
          *trans_extrema.second > *trans_extrema.first + 5e-3f)) {
        std::cerr << "Transmittance lacks variation" << std::endl;
        return 1;
    }

    if (std::all_of(forward.hitmask.begin(), forward.hitmask.end(), [](uint32_t v) { return v == 0U; })) {
        std::cerr << "Hitmask contains no hits" << std::endl;
        return 1;
    }

    for (float value : forward.transmittance) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            std::cerr << "Transmittance out of range" << std::endl;
            return 1;
        }
    }
    for (float value : forward.opacity) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            std::cerr << "Opacity out of range" << std::endl;
            return 1;
        }
    }

#if defined(HP_WITH_CUDA)
    if (!ValidateCudaForwardParity(plan, field, forward)) {
        return 1;
    }
#endif

    return 0;
}
