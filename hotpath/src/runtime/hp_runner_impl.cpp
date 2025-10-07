#include "hotpath/hp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(HP_WITH_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

enum class TestStatus {
    pass,
    fail,
    skip
};

struct CaseResult {
    std::string name;
    TestStatus status;
    std::string message;
};

std::string status_to_string(TestStatus status) {
    switch (status) {
        case TestStatus::pass:
            return "pass";
        case TestStatus::fail:
            return "fail";
        case TestStatus::skip:
            return "skip";
    }
    return "unknown";
}

const char* status_to_cstr(hp_status status) {
    switch (status) {
        case HP_STATUS_SUCCESS:
            return "success";
        case HP_STATUS_INVALID_ARGUMENT:
            return "invalid_argument";
        case HP_STATUS_OUT_OF_MEMORY:
            return "out_of_memory";
        case HP_STATUS_NOT_IMPLEMENTED:
            return "not_implemented";
        case HP_STATUS_UNSUPPORTED:
            return "unsupported";
        case HP_STATUS_INTERNAL_ERROR:
            return "internal_error";
        default:
            return "unknown";
    }
}

std::string escape_json(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (unsigned char c : input) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}
std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return std::string();
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::vector<std::string> default_cases() {
    return {
        "ray_cpu_basic",
        "ray_cpu_roi",
        "ray_cpu_override",
        "ray_cuda_basic",
        "samp_cpu_basic",
        "samp_cpu_oob_zero",
        "samp_cpu_oob_clamp",
        "samp_cpu_stratified_determinism"
    };
}

std::vector<std::string> load_manifest_cases(const hp_runner_options* options) {
    std::filesystem::path manifest_path = "tests/manifest.yaml";
    if (options != nullptr && options->manifest_path != nullptr) {
        manifest_path = options->manifest_path;
    }

    std::ifstream file(manifest_path);
    if (!file.is_open()) {
        return default_cases();
    }

    std::vector<std::string> cases;
    bool in_cases = false;
    std::string line;
    while (std::getline(file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        if (!in_cases) {
            if (trimmed.rfind("cases", 0) == 0) {
                in_cases = true;
            }
            continue;
        }
        if (trimmed[0] == '-') {
            const auto name_pos = trimmed.find_first_not_of("- \t");
            if (name_pos == std::string::npos) {
                continue;
            }
            std::string name = trimmed.substr(name_pos);
            const auto comment_pos = name.find('#');
            if (comment_pos != std::string::npos) {
                name = name.substr(0, comment_pos);
            }
            name = trim(name);
            if (!name.empty()) {
                cases.push_back(name);
            }
        }
    }

    if (cases.empty()) {
        return default_cases();
    }
    return cases;
}

std::string build_scoreboard(const std::vector<CaseResult>& cases) {
    std::ostringstream oss;
    oss << "{\"cases\":[";
    bool first = true;
    size_t pass_count = 0;
    size_t fail_count = 0;
    size_t skip_count = 0;
    for (const auto& c : cases) {
        if (!first) {
            oss << ',';
        }
        first = false;
        oss << "{\"name\":\"" << escape_json(c.name) << "\",\"status\":\"" << status_to_string(c.status) << "\"";
        if (!c.message.empty()) {
            oss << ",\"message\":\"" << escape_json(c.message) << "\"";
        }
        oss << "}";
        if (c.status == TestStatus::pass) {
            ++pass_count;
        } else if (c.status == TestStatus::fail) {
            ++fail_count;
        } else if (c.status == TestStatus::skip) {
            ++skip_count;
        }
    }
    oss << "],\"summary\":{\"pass\":" << pass_count << ",\"fail\":" << fail_count;
    if (skip_count > 0) {
        oss << ",\"skip\":" << skip_count;
    }
    oss << "}}\n";
    return oss.str();
}

hp_plan_desc make_basic_plan_desc(uint32_t width, uint32_t height, float t_near, float t_far) {
    hp_plan_desc desc{};
    desc.width = width;
    desc.height = height;
    desc.t_near = t_near;
    desc.t_far = t_far;
    desc.max_rays = width * height;
    desc.max_samples = desc.max_rays * 16U;
    desc.seed = 0;
    desc.camera.model = HP_CAMERA_PINHOLE;
    desc.camera.K[0] = 1.0f;
    desc.camera.K[4] = 1.0f;
    desc.camera.K[8] = 1.0f;
    desc.camera.K[2] = static_cast<float>(width) * 0.5f;
    desc.camera.K[5] = static_cast<float>(height) * 0.5f;
    desc.camera.c2w[0] = 1.0f;
    desc.camera.c2w[5] = 1.0f;
    desc.camera.c2w[10] = 1.0f;
    desc.roi.x = 0;
    desc.roi.y = 0;
    desc.roi.width = width;
    desc.roi.height = height;
    desc.sampling.dt = 0.25f;
    desc.sampling.max_steps = 16;
    desc.sampling.mode = HP_SAMPLING_FIXED;
    return desc;
}

std::vector<float> copy_vec3_buffer(const void* ptr, size_t count) {
    std::vector<float> out(count * 3U, 0.0f);
    if (ptr != nullptr && count > 0) {
        std::memcpy(out.data(), ptr, out.size() * sizeof(float));
    }
    return out;
}

std::vector<float> copy_scalar_f_buffer(const void* ptr, size_t count) {
    std::vector<float> out(count, 0.0f);
    if (ptr != nullptr && count > 0) {
        std::memcpy(out.data(), ptr, out.size() * sizeof(float));
    }
    return out;
}

std::vector<uint32_t> copy_scalar_u32_buffer(const void* ptr, size_t count) {
    std::vector<uint32_t> out(count, 0U);
    if (ptr != nullptr && count > 0) {
        std::memcpy(out.data(), ptr, out.size() * sizeof(uint32_t));
    }
    return out;
}

bool directions_normalized(const std::vector<float>& dirs, float tolerance, std::string& message) {
    for (size_t i = 0; i + 2 < dirs.size(); i += 3) {
        const float dx = dirs[i + 0];
        const float dy = dirs[i + 1];
        const float dz = dirs[i + 2];
        const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (std::fabs(len - 1.0f) > tolerance) {
            std::ostringstream oss;
            oss << "direction length out of tolerance (" << len << ")";
            message = oss.str();
            return false;
        }
    }
    return true;
}

bool check_monotone_t(const std::vector<float>& positions,
                      const std::vector<float>& origins,
                      const std::vector<float>& directions,
                      float tolerance,
                      std::string& message) {
    if (positions.empty()) {
        return true;
    }
    const float ox = origins[0];
    const float oy = origins[1];
    const float oz = origins[2];
    const float dx = directions[0];
    const float dy = directions[1];
    const float dz = directions[2];

    float prev_t = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < positions.size(); i += 3) {
        const float px = positions[i + 0] - ox;
        const float py = positions[i + 1] - oy;
        const float pz = positions[i + 2] - oz;
        const float proj = px * dx + py * dy + pz * dz;
        if (proj + tolerance < prev_t) {
            std::ostringstream oss;
            oss << "non-monotone t: " << proj << " after " << prev_t;
            message = oss.str();
            return false;
        }
        prev_t = proj;
    }
    return true;
}

hp_tensor make_tensor(float* data, hp_dtype dtype, uint32_t rank, const std::array<int64_t, 4>& shape) {
    hp_tensor t{};
    t.data = data;
    t.dtype = dtype;
    t.memspace = HP_MEMSPACE_HOST;
    t.rank = rank;
    for (uint32_t i = 0; i < rank; ++i) {
        t.shape[i] = shape[i];
    }
    if (rank >= 1) {
        t.stride[rank - 1] = 1;
        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
            t.stride[i] = t.stride[i + 1] * t.shape[i + 1];
        }
    }
    return t;
}

CaseResult test_ray_cpu_basic(hp_ctx* ctx, const hp_runner_options*) {
    CaseResult result{"ray_cpu_basic", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(4, 4, 0.1f, 2.0f);
    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    const size_t ray_count = static_cast<size_t>(plan_desc.width) * static_cast<size_t>(plan_desc.height);
    std::array<std::byte, 4096> workspace_a{};
    std::array<std::byte, 4096> workspace_b{};
    hp_rays_t rays{};

    status = hp_ray(plan, nullptr, &rays, workspace_a.data(), workspace_a.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    const std::vector<float> origins = copy_vec3_buffer(rays.origins.data, ray_count);
    const std::vector<float> directions = copy_vec3_buffer(rays.directions.data, ray_count);
    const std::vector<float> t_near = copy_scalar_f_buffer(rays.t_near.data, ray_count);
    const std::vector<float> t_far = copy_scalar_f_buffer(rays.t_far.data, ray_count);
    const std::vector<uint32_t> pixel_ids = copy_scalar_u32_buffer(rays.pixel_ids.data, ray_count);

    std::string dir_message;
    if (!directions_normalized(directions, 1e-4f, dir_message)) {
        result.status = TestStatus::fail;
        result.message = dir_message;
        hp_plan_release(plan);
        return result;
    }

    for (size_t idx = 0; idx < ray_count; ++idx) {
        const size_t base = idx * 3U;
        const float ox = origins[base + 0];
        const float oy = origins[base + 1];
        const float oz = origins[base + 2];
        if (std::fabs(ox) > 1e-6f || std::fabs(oy) > 1e-6f || std::fabs(oz) > 1e-6f) {
            result.status = TestStatus::fail;
            result.message = "origin not at camera position";
            hp_plan_release(plan);
            return result;
        }
        if (std::fabs(t_near[idx] - plan_desc.t_near) > 1e-6f ||
            std::fabs(t_far[idx] - plan_desc.t_far) > 1e-6f) {
            result.status = TestStatus::fail;
            result.message = "incorrect t bounds";
            hp_plan_release(plan);
            return result;
        }
        const uint32_t expected_px = static_cast<uint32_t>(idx % plan_desc.width);
        const uint32_t expected_py = static_cast<uint32_t>(idx / plan_desc.width);
        const uint32_t expected_id = expected_py * plan_desc.width + expected_px;
        if (pixel_ids[idx] != expected_id) {
            result.status = TestStatus::fail;
            result.message = "pixel id mismatch";
            hp_plan_release(plan);
            return result;
        }
    }

    hp_rays_t rays_second{};
    status = hp_ray(plan, nullptr, &rays_second, workspace_b.data(), workspace_b.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray second pass failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    const size_t vec3_bytes = ray_count * 3U * sizeof(float);
    const size_t scalar_bytes = ray_count * sizeof(float);
    const size_t ids_bytes = ray_count * sizeof(uint32_t);
    if (std::memcmp(origins.data(), rays_second.origins.data, vec3_bytes) != 0 ||
        std::memcmp(directions.data(), rays_second.directions.data, vec3_bytes) != 0 ||
        std::memcmp(t_near.data(), rays_second.t_near.data, scalar_bytes) != 0 ||
        std::memcmp(t_far.data(), rays_second.t_far.data, scalar_bytes) != 0 ||
        std::memcmp(pixel_ids.data(), rays_second.pixel_ids.data, ids_bytes) != 0) {
        result.status = TestStatus::fail;
        result.message = "outputs differ between deterministic runs";
        hp_plan_release(plan);
        return result;
    }

    hp_plan_release(plan);
    return result;
}

CaseResult test_ray_cpu_roi(hp_ctx* ctx, const hp_runner_options*) {
    CaseResult result{"ray_cpu_roi", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(8, 6, 0.5f, 3.0f);
    plan_desc.roi.x = 2;
    plan_desc.roi.y = 1;
    plan_desc.roi.width = 3;
    plan_desc.roi.height = 2;
    plan_desc.max_rays = plan_desc.roi.width * plan_desc.roi.height;

    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    const size_t ray_count = static_cast<size_t>(plan_desc.roi.width) * static_cast<size_t>(plan_desc.roi.height);
    std::array<std::byte, 4096> workspace{};
    hp_rays_t rays{};
    status = hp_ray(plan, nullptr, &rays, workspace.data(), workspace.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    const std::vector<uint32_t> pixel_ids = copy_scalar_u32_buffer(rays.pixel_ids.data, ray_count);
    for (uint32_t local_y = 0; local_y < plan_desc.roi.height; ++local_y) {
        for (uint32_t local_x = 0; local_x < plan_desc.roi.width; ++local_x) {
            const size_t idx = static_cast<size_t>(local_y) * plan_desc.roi.width + local_x;
            const uint32_t py = plan_desc.roi.y + local_y;
            const uint32_t px = plan_desc.roi.x + local_x;
            const uint32_t expected_id = py * plan_desc.width + px;
            if (pixel_ids[idx] != expected_id) {
                result.status = TestStatus::fail;
                result.message = "ROI pixel id mismatch";
                hp_plan_release(plan);
                return result;
            }
        }
    }

    hp_plan_release(plan);
    return result;
}

CaseResult test_ray_cpu_override(hp_ctx* ctx, const hp_runner_options*) {
    CaseResult result{"ray_cpu_override", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(2, 1, 0.2f, 4.0f);
    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    constexpr size_t ray_count = 2;
    std::vector<float> override_origins{0.0f, 0.0f, 1.0f,
                                        1.0f, 0.0f, 1.0f};
    std::vector<float> override_dirs{0.0f, 0.0f, 1.0f,
                                     0.0f, 1.0f, 0.0f};
    std::vector<float> override_near{0.25f, 0.5f};
    std::vector<float> override_far{5.0f, 6.0f};
    std::vector<uint32_t> override_ids{3U, 7U};

    hp_rays_t override_rays{};
    override_rays.origins = make_tensor(override_origins.data(), HP_DTYPE_F32, 2, {static_cast<int64_t>(ray_count), 3, 0, 0});
    override_rays.directions = make_tensor(override_dirs.data(), HP_DTYPE_F32, 2, {static_cast<int64_t>(ray_count), 3, 0, 0});
    override_rays.t_near = make_tensor(override_near.data(), HP_DTYPE_F32, 1, {static_cast<int64_t>(ray_count), 0, 0, 0});
    override_rays.t_far = make_tensor(override_far.data(), HP_DTYPE_F32, 1, {static_cast<int64_t>(ray_count), 0, 0, 0});
    override_rays.pixel_ids = make_tensor(reinterpret_cast<float*>(override_ids.data()), HP_DTYPE_U32, 1, {static_cast<int64_t>(ray_count), 0, 0, 0});
    override_rays.pixel_ids.memspace = HP_MEMSPACE_HOST;

    std::array<std::byte, 1024> workspace{};
    hp_rays_t output{};
    hp_status ray_status = hp_ray(plan, &override_rays, &output, workspace.data(), workspace.size());
    if (ray_status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray override failed: ") + status_to_cstr(ray_status);
        hp_plan_release(plan);
        return result;
    }

    const size_t vec3_bytes = ray_count * 3U * sizeof(float);
    const size_t scalar_bytes = ray_count * sizeof(float);
    const size_t ids_bytes = ray_count * sizeof(uint32_t);
    if (std::memcmp(output.origins.data, override_origins.data(), vec3_bytes) != 0 ||
        std::memcmp(output.directions.data, override_dirs.data(), vec3_bytes) != 0 ||
        std::memcmp(output.t_near.data, override_near.data(), scalar_bytes) != 0 ||
        std::memcmp(output.t_far.data, override_far.data(), scalar_bytes) != 0 ||
        std::memcmp(output.pixel_ids.data, override_ids.data(), ids_bytes) != 0) {
        result.status = TestStatus::fail;
        result.message = "override results do not match inputs";
        hp_plan_release(plan);
        return result;
    }

    hp_plan_release(plan);
    return result;
}

#if defined(HP_WITH_CUDA)
CaseResult test_ray_cuda_basic(hp_ctx* ctx, const hp_runner_options*) {
    CaseResult result{"ray_cuda_basic", TestStatus::pass, ""};
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        result.status = TestStatus::skip;
        result.message = "no CUDA device available";
        return result;
    }

    hp_plan_desc plan_desc = make_basic_plan_desc(4, 4, 0.1f, 2.0f);
    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    const size_t ray_count = static_cast<size_t>(plan_desc.width) * static_cast<size_t>(plan_desc.height);
    const size_t vec3_bytes = ray_count * 3U * sizeof(float);
    const size_t scalar_bytes = ray_count * sizeof(float);
    const size_t ids_bytes = ray_count * sizeof(uint32_t);

    float* d_origins = nullptr;
    float* d_dirs = nullptr;
    float* d_tnear = nullptr;
    float* d_tfar = nullptr;
    uint32_t* d_ids = nullptr;

    if (cudaMalloc(reinterpret_cast<void**>(&d_origins), vec3_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_dirs), vec3_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_tnear), scalar_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_tfar), scalar_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_ids), ids_bytes) != cudaSuccess) {
        result.status = TestStatus::skip;
        result.message = "cudaMalloc failed";
        cudaFree(d_origins);
        cudaFree(d_dirs);
        cudaFree(d_tnear);
        cudaFree(d_tfar);
        cudaFree(d_ids);
        hp_plan_release(plan);
        return result;
    }

    hp_rays_t rays{};
    rays.origins.data = d_origins;
    rays.origins.memspace = HP_MEMSPACE_DEVICE;
    rays.directions.data = d_dirs;
    rays.directions.memspace = HP_MEMSPACE_DEVICE;
    rays.t_near.data = d_tnear;
    rays.t_near.memspace = HP_MEMSPACE_DEVICE;
    rays.t_far.data = d_tfar;
    rays.t_far.memspace = HP_MEMSPACE_DEVICE;
    rays.pixel_ids.data = d_ids;
    rays.pixel_ids.memspace = HP_MEMSPACE_DEVICE;

    status = hp_ray(plan, nullptr, &rays, nullptr, 0);
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray CUDA failed: ") + status_to_cstr(status);
        cudaFree(d_origins);
        cudaFree(d_dirs);
        cudaFree(d_tnear);
        cudaFree(d_tfar);
        cudaFree(d_ids);
        hp_plan_release(plan);
        return result;
    }

    std::vector<float> h_origins(ray_count * 3U, 0.0f);
    std::vector<float> h_dirs(ray_count * 3U, 0.0f);
    std::vector<float> h_tnear(ray_count, 0.0f);
    std::vector<float> h_tfar(ray_count, 0.0f);
    std::vector<uint32_t> h_ids(ray_count, 0U);

    cudaMemcpy(h_origins.data(), d_origins, vec3_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dirs.data(), d_dirs, vec3_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tnear.data(), d_tnear, scalar_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tfar.data(), d_tfar, scalar_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids.data(), d_ids, ids_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_origins);
    cudaFree(d_dirs);
    cudaFree(d_tnear);
    cudaFree(d_tfar);
    cudaFree(d_ids);

    std::string dir_message;
    if (!directions_normalized(h_dirs, 1e-3f, dir_message)) {
        result.status = TestStatus::fail;
        result.message = dir_message;
        hp_plan_release(plan);
        return result;
    }

    for (size_t idx = 0; idx < ray_count; ++idx) {
        const size_t base = idx * 3U;
        if (std::fabs(h_origins[base + 0]) > 1e-4f || std::fabs(h_origins[base + 1]) > 1e-4f) {
            result.status = TestStatus::fail;
            result.message = "CUDA origin mismatch";
            hp_plan_release(plan);
            return result;
        }
        if (std::fabs(h_tnear[idx] - plan_desc.t_near) > 1e-4f ||
            std::fabs(h_tfar[idx] - plan_desc.t_far) > 1e-4f) {
            result.status = TestStatus::fail;
            result.message = "CUDA t bounds mismatch";
            hp_plan_release(plan);
            return result;
        }
        const uint32_t expected_px = static_cast<uint32_t>(idx % plan_desc.width);
        const uint32_t expected_py = static_cast<uint32_t>(idx / plan_desc.width);
        const uint32_t expected_id = expected_py * plan_desc.width + expected_px;
        if (h_ids[idx] != expected_id) {
            result.status = TestStatus::fail;
            result.message = "CUDA pixel id mismatch";
            hp_plan_release(plan);
            return result;
        }
    }

    hp_plan_release(plan);
    return result;
}
#endif  // HP_WITH_CUDA

CaseResult test_samp_cpu_basic(hp_ctx* ctx, const hp_runner_options*) {
    CaseResult result{"samp_cpu_basic", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(1, 1, 0.0f, 1.0f);
    plan_desc.sampling.dt = 0.25f;
    plan_desc.sampling.max_steps = 8;
    plan_desc.max_rays = 1;
    plan_desc.max_samples = 16;

    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    std::array<std::byte, 4096> ray_ws{};
    hp_rays_t rays{};
    status = hp_ray(plan, nullptr, &rays, ray_ws.data(), ray_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    std::vector<float> sigma_grid(8, 1.0f);
    std::vector<float> color_grid{
        0.0f, 0.0f, 0.0f,
        0.1f, 0.0f, 0.0f,
        0.0f, 0.1f, 0.0f,
        0.1f, 0.1f, 0.0f,
        0.0f, 0.0f, 0.5f,
        0.1f, 0.0f, 0.5f,
        0.0f, 0.1f, 0.5f,
        0.1f, 0.1f, 0.5f
    };

    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, 3, {2, 2, 2, 0});
    hp_tensor color_tensor = make_tensor(color_grid.data(), HP_DTYPE_F32, 4, {2, 2, 2, 3});

    hp_field* fs = nullptr;
    hp_field* fc = nullptr;
    status = hp_field_create_grid_sigma(ctx, &sigma_tensor, HP_INTERP_LINEAR, HP_OOB_ZERO, &fs);
    if (status != HP_STATUS_SUCCESS || fs == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_field_create_grid_sigma failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }
    status = hp_field_create_grid_color(ctx, &color_tensor, HP_INTERP_LINEAR, HP_OOB_ZERO, &fc);
    if (status != HP_STATUS_SUCCESS || fc == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_field_create_grid_color failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 8192> samp_ws{};
    hp_samp_t samp{};
    status = hp_samp(plan, fs, fc, &rays, &samp, samp_ws.data(), samp_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_samp failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const size_t ray_count = 1;
    const std::vector<float> positions = copy_vec3_buffer(samp.positions.data, static_cast<size_t>(samp.positions.shape[0]));
    const std::vector<float> dts = copy_scalar_f_buffer(samp.dt.data, static_cast<size_t>(samp.dt.shape[0]));
    const std::vector<uint32_t> offsets = copy_scalar_u32_buffer(samp.ray_offset.data, ray_count + 1);
    const std::vector<float> sigma = copy_scalar_f_buffer(samp.sigma.data, static_cast<size_t>(samp.sigma.shape[0]));

    if (offsets.size() != 2 || offsets[0] != 0 || offsets[1] != sigma.size()) {
        result.status = TestStatus::fail;
        result.message = "ray offsets malformed";
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const std::vector<float> ray_origins = copy_vec3_buffer(rays.origins.data, ray_count);
    const std::vector<float> ray_dirs = copy_vec3_buffer(rays.directions.data, ray_count);
    std::string message;
    if (!check_monotone_t(positions, ray_origins, ray_dirs, 1e-4f, message)) {
        result.status = TestStatus::fail;
        result.message = message;
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    for (float dt : dts) {
        if (!(dt > 0.0f)) {
            result.status = TestStatus::fail;
            result.message = "non-positive dt";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
    }
    for (float s : sigma) {
        if (std::fabs(s - 1.0f) > 1e-4f) {
            result.status = TestStatus::fail;
            result.message = "sigma values incorrect";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
    }

    hp_field_release(fs);
    hp_field_release(fc);
    hp_plan_release(plan);
    return result;
}

CaseResult test_samp_cpu_oob_zero(hp_ctx* ctx, const hp_runner_options*) {
    CaseResult result{"samp_cpu_oob_zero", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(1, 1, 0.0f, 2.0f);
    plan_desc.sampling.dt = 0.5f;
    plan_desc.sampling.max_steps = 8;
    plan_desc.max_rays = 1;
    plan_desc.max_samples = 16;

    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    std::array<std::byte, 4096> ray_ws{};
    hp_rays_t rays{};
    status = hp_ray(plan, nullptr, &rays, ray_ws.data(), ray_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    std::vector<float> sigma_grid(8, 1.0f);
    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, 3, {2, 2, 2, 0});

    hp_field* fs = nullptr;
    status = hp_field_create_grid_sigma(ctx, &sigma_tensor, HP_INTERP_LINEAR, HP_OOB_ZERO, &fs);
    if (status != HP_STATUS_SUCCESS || fs == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_field_create_grid_sigma failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 8192> samp_ws{};
    hp_samp_t samp{};
    status = hp_samp(plan, fs, nullptr, &rays, &samp, samp_ws.data(), samp_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_samp failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    const std::vector<float> sigma = copy_scalar_f_buffer(samp.sigma.data, static_cast<size_t>(samp.sigma.shape[0]));
    const std::vector<float> positions = copy_vec3_buffer(samp.positions.data, static_cast<size_t>(samp.positions.shape[0]));
    const std::vector<float> ray_origins = copy_vec3_buffer(rays.origins.data, 1);
    const std::vector<float> ray_dirs = copy_vec3_buffer(rays.directions.data, 1);

    for (size_t i = 0; i < sigma.size(); ++i) {
        const size_t base = i * 3U;
        const float px = positions[base + 0] - ray_origins[0];
        const float py = positions[base + 1] - ray_origins[1];
        const float pz = positions[base + 2] - ray_origins[2];
        const float t = px * ray_dirs[0] + py * ray_dirs[1] + pz * ray_dirs[2];
        if (t > 1.0f + 1e-4f && std::fabs(sigma[i]) > 1e-6f) {
            result.status = TestStatus::fail;
            result.message = "OOB zero policy violated";
            hp_field_release(fs);
            hp_plan_release(plan);
            return result;
        }
    }

    hp_field_release(fs);
    hp_plan_release(plan);
    return result;
}

CaseResult test_samp_cpu_oob_clamp(hp_ctx* ctx, const hp_runner_options*) {
    CaseResult result{"samp_cpu_oob_clamp", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(1, 1, 0.0f, 2.0f);
    plan_desc.sampling.dt = 0.5f;
    plan_desc.sampling.max_steps = 8;
    plan_desc.max_rays = 1;
    plan_desc.max_samples = 16;

    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    std::array<std::byte, 4096> ray_ws{};
    hp_rays_t rays{};
    status = hp_ray(plan, nullptr, &rays, ray_ws.data(), ray_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    std::vector<float> sigma_grid(8, 0.0f);
    sigma_grid[4] = 2.0f;
    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, 3, {2, 2, 2, 0});

    hp_field* fs = nullptr;
    status = hp_field_create_grid_sigma(ctx, &sigma_tensor, HP_INTERP_NEAREST, HP_OOB_CLAMP, &fs);
    if (status != HP_STATUS_SUCCESS || fs == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_field_create_grid_sigma failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 8192> samp_ws{};
    hp_samp_t samp{};
    status = hp_samp(plan, fs, nullptr, &rays, &samp, samp_ws.data(), samp_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_samp failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    const std::vector<float> sigma = copy_scalar_f_buffer(samp.sigma.data, static_cast<size_t>(samp.sigma.shape[0]));
    bool found_clamped = false;
    for (float s : sigma) {
        if (std::fabs(s - 2.0f) < 1e-3f) {
            found_clamped = true;
        }
    }
    if (!found_clamped) {
        result.status = TestStatus::fail;
        result.message = "clamp policy not applied";
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    hp_field_release(fs);
    hp_plan_release(plan);
    return result;
}

CaseResult test_samp_cpu_stratified(hp_ctx* ctx, const hp_runner_options*) {
    CaseResult result{"samp_cpu_stratified_determinism", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(1, 1, 0.0f, 1.0f);
    plan_desc.sampling.dt = 0.2f;
    plan_desc.sampling.max_steps = 16;
    plan_desc.sampling.mode = HP_SAMPLING_STRATIFIED;
    plan_desc.max_rays = 1;
    plan_desc.max_samples = 32;
    plan_desc.seed = 12345ULL;

    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    std::array<std::byte, 4096> ray_ws{};
    hp_rays_t rays{};
    status = hp_ray(plan, nullptr, &rays, ray_ws.data(), ray_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    std::vector<float> sigma_grid(8, 1.0f);
    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, 3, {2, 2, 2, 0});
    hp_field* fs = nullptr;
    status = hp_field_create_grid_sigma(ctx, &sigma_tensor, HP_INTERP_LINEAR, HP_OOB_ZERO, &fs);
    if (status != HP_STATUS_SUCCESS || fs == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_field_create_grid_sigma failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 8192> samp_ws_a{};
    std::array<std::byte, 8192> samp_ws_b{};
    hp_samp_t samp_a{};
    hp_samp_t samp_b{};

    status = hp_samp(plan, fs, nullptr, &rays, &samp_a, samp_ws_a.data(), samp_ws_a.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_samp first pass failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }
    status = hp_samp(plan, fs, nullptr, &rays, &samp_b, samp_ws_b.data(), samp_ws_b.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_samp second pass failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    const size_t sample_count = static_cast<size_t>(samp_a.dt.shape[0]);
    if (sample_count == 0 || sample_count != static_cast<size_t>(samp_b.dt.shape[0])) {
        result.status = TestStatus::fail;
        result.message = "unexpected sample counts";
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    const size_t dt_bytes = sample_count * sizeof(float);
    if (std::memcmp(samp_a.dt.data, samp_b.dt.data, dt_bytes) != 0) {
        result.status = TestStatus::fail;
        result.message = "stratified dt buffers are not deterministic";
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    const std::vector<float> positions = copy_vec3_buffer(samp_a.positions.data, sample_count);
    const std::vector<float> ray_origins = copy_vec3_buffer(rays.origins.data, 1);
    const std::vector<float> ray_dirs = copy_vec3_buffer(rays.directions.data, 1);

    bool found_non_midpoint = false;
    for (size_t i = 0; i < sample_count; ++i) {
        const size_t base = i * 3U;
        const float px = positions[base + 0] - ray_origins[0];
        const float py = positions[base + 1] - ray_origins[1];
        const float pz = positions[base + 2] - ray_origins[2];
        const float t = px * ray_dirs[0] + py * ray_dirs[1] + pz * ray_dirs[2];
        const float relative = (t - plan_desc.t_near) / plan_desc.sampling.dt;
        const float frac = relative - std::floor(relative);
        if (std::fabs(frac - 0.5f) > 1e-2f) {
            found_non_midpoint = true;
            break;
        }
    }
    if (!found_non_midpoint) {
        result.status = TestStatus::fail;
        result.message = "stratified sampling produced midpoints only";
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    hp_field_release(fs);
    hp_plan_release(plan);
    return result;
}

using TestFn = CaseResult(*)(hp_ctx*, const hp_runner_options*);

std::unordered_map<std::string, TestFn> build_registry() {
    return {
        {"ray_cpu_basic", test_ray_cpu_basic},
        {"ray_cpu_roi", test_ray_cpu_roi},
        {"ray_cpu_override", test_ray_cpu_override},
#if defined(HP_WITH_CUDA)
        {"ray_cuda_basic", test_ray_cuda_basic},
#endif
        {"samp_cpu_basic", test_samp_cpu_basic},
        {"samp_cpu_oob_zero", test_samp_cpu_oob_zero},
        {"samp_cpu_oob_clamp", test_samp_cpu_oob_clamp},
        {"samp_cpu_stratified_determinism", test_samp_cpu_stratified}
    };
}

}  // namespace

extern "C" HP_API hp_status hp_runner_run(const hp_ctx* ctx, const hp_runner_options* options) {
    if (ctx == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    std::vector<std::string> case_names = load_manifest_cases(options);
    const auto registry = build_registry();
    std::vector<CaseResult> results;
    results.reserve(case_names.size());

    for (const auto& name : case_names) {
        auto it = registry.find(name);
        if (it == registry.end()) {
            results.push_back(CaseResult{name, TestStatus::fail, "unknown test"});
            continue;
        }
        results.push_back(it->second(const_cast<hp_ctx*>(ctx), options));
    }

    const std::string scoreboard = build_scoreboard(results);
    std::cout << scoreboard;

    bool has_failures = false;
    for (const auto& r : results) {
        if (r.status == TestStatus::fail) {
            has_failures = true;
            break;
        }
    }
    return has_failures ? HP_STATUS_INTERNAL_ERROR : HP_STATUS_SUCCESS;
}
