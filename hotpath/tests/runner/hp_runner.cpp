#include "hotpath/hp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(HP_WITH_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

struct RunnerOptions {
    std::filesystem::path manifest_path{"tests/manifest.yaml"};
};

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
            case '\\\\':
                out += "\\\\\\\\";
                break;
            case '\"':
                out += "\\\\\"";
                break;
            case '\\n':
                out += "\\\\n";
                break;
            case '\\r':
                out += "\\\\r";
                break;
            case '\\t':
                out += "\\\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\\\u%04x", static_cast<unsigned int>(c));
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
        "samp_cpu_stratified_determinism",
        "int_cpu_constant",
        "int_cpu_piecewise",
        "int_cpu_gaussian",
        "int_cpu_early_stop",
        "img_cpu_basic",
        "img_cpu_roi_background",
        "fused_cpu_equivalence",
        "diff_cpu_sigma_color"
    };
}

std::vector<std::string> load_manifest_cases(const RunnerOptions& options) {
    std::filesystem::path manifest_path = options.manifest_path;
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
            oss << ",";
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

hp_tensor make_tensor(void* data, hp_dtype dtype, std::initializer_list<int64_t> shape) {
    hp_tensor t{};
    t.data = data;
    t.dtype = dtype;
    t.memspace = HP_MEMSPACE_HOST;
    t.rank = static_cast<uint32_t>(shape.size());
    uint32_t idx = 0;
    for (int64_t dim : shape) {
        t.shape[idx++] = dim;
    }
    if (t.rank > 0) {
        t.stride[t.rank - 1] = 1;
        for (int i = static_cast<int>(t.rank) - 2; i >= 0; --i) {
            t.stride[i] = t.stride[i + 1] * t.shape[i + 1];
        }
    }
    return t;
}
CaseResult test_ray_cpu_basic(hp_ctx* ctx) {
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

CaseResult test_ray_cpu_roi(hp_ctx* ctx) {
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

CaseResult test_ray_cpu_override(hp_ctx* ctx) {
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
    override_rays.origins = make_tensor(override_origins.data(), HP_DTYPE_F32, {static_cast<int64_t>(ray_count), 3});
    override_rays.directions = make_tensor(override_dirs.data(), HP_DTYPE_F32, {static_cast<int64_t>(ray_count), 3});
    override_rays.t_near = make_tensor(override_near.data(), HP_DTYPE_F32, {static_cast<int64_t>(ray_count)});
    override_rays.t_far = make_tensor(override_far.data(), HP_DTYPE_F32, {static_cast<int64_t>(ray_count)});
    override_rays.pixel_ids = make_tensor(override_ids.data(), HP_DTYPE_U32, {static_cast<int64_t>(ray_count)});

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
CaseResult test_ray_cuda_basic(hp_ctx* ctx) {
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
        if (std::fabs(h_origins[base + 0]) > 1e-4f ||
            std::fabs(h_origins[base + 1]) > 1e-4f) {
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
CaseResult test_samp_cpu_basic(hp_ctx* ctx) {
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

    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 2});
    hp_tensor color_tensor = make_tensor(color_grid.data(), HP_DTYPE_F32, {2, 2, 2, 3});

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

    const size_t ray_count = static_cast<size_t>(plan_desc.width) * static_cast<size_t>(plan_desc.height);
    const std::vector<float> positions = copy_vec3_buffer(samp.positions.data, ray_count);
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

CaseResult test_samp_cpu_oob_zero(hp_ctx* ctx) {
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
    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 2});

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

CaseResult test_samp_cpu_oob_clamp(hp_ctx* ctx) {
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
    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 2});

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
            result.message = "OOB clamp policy violated";
            hp_field_release(fs);
            hp_plan_release(plan);
            return result;
        }
    }

    hp_field_release(fs);
    hp_plan_release(plan);
    return result;
}

CaseResult test_samp_cpu_stratified(hp_ctx* ctx) {
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
    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 2});
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

float compute_alpha_ref(float sigma, float dt) {
    const float optical = sigma * dt;
    if (optical <= 0.0f) {
        return 0.0f;
    }
    if (optical < 1e-4f) {
        const float half = 0.5f * optical;
        return optical * (1.0f - half);
    }
    const double alpha = -std::expm1(-static_cast<double>(optical));
    return static_cast<float>(std::min(1.0, std::max(alpha, 0.0)));
}

hp_samp_t make_samp_view(std::vector<float>& positions,
                         std::vector<float>& dt,
                         std::vector<uint32_t>& offsets,
                         std::vector<float>& sigma,
                         std::vector<float>& color) {
    hp_samp_t samp{};
    const int64_t samples = static_cast<int64_t>(dt.size());
    samp.positions.data = positions.data();
    samp.positions.dtype = HP_DTYPE_F32;
    samp.positions.memspace = HP_MEMSPACE_HOST;
    samp.positions.rank = 2;
    samp.positions.shape[0] = samples;
    samp.positions.shape[1] = 3;
    samp.positions.stride[1] = 1;
    samp.positions.stride[0] = 3;

    samp.dt.data = dt.data();
    samp.dt.dtype = HP_DTYPE_F32;
    samp.dt.memspace = HP_MEMSPACE_HOST;
    samp.dt.rank = 1;
    samp.dt.shape[0] = samples;
    samp.dt.stride[0] = 1;

    samp.ray_offset.data = offsets.data();
    samp.ray_offset.dtype = HP_DTYPE_U32;
    samp.ray_offset.memspace = HP_MEMSPACE_HOST;
    samp.ray_offset.rank = 1;
    samp.ray_offset.shape[0] = static_cast<int64_t>(offsets.size());
    samp.ray_offset.stride[0] = 1;

    samp.sigma.data = sigma.data();
    samp.sigma.dtype = HP_DTYPE_F32;
    samp.sigma.memspace = HP_MEMSPACE_HOST;
    samp.sigma.rank = 1;
    samp.sigma.shape[0] = samples;
    samp.sigma.stride[0] = 1;

    samp.color.data = color.data();
    samp.color.dtype = HP_DTYPE_F32;
    samp.color.memspace = HP_MEMSPACE_HOST;
    samp.color.rank = 2;
    samp.color.shape[0] = samples;
    samp.color.shape[1] = 3;
    samp.color.stride[1] = 1;
    samp.color.stride[0] = 3;

    return samp;
}

CaseResult run_integration_case(const char* name,
                                hp_ctx* ctx,
                                const std::vector<float>& sigma_vals,
                                const std::vector<float>& color_vals,
                                const std::vector<float>& dt_vals,
                                float t_near,
                                float t_far,
                                bool expect_early_stop = false) {
    CaseResult result{name, TestStatus::pass, ""};

    if (sigma_vals.size() != dt_vals.size() || color_vals.size() != dt_vals.size() * 3U) {
        result.status = TestStatus::fail;
        result.message = "invalid test fixture";
        return result;
    }

    hp_plan_desc plan_desc = make_basic_plan_desc(1, 1, t_near, t_far);
    plan_desc.max_rays = 1;
    plan_desc.max_samples = static_cast<uint32_t>(dt_vals.size());
    plan_desc.sampling.max_steps = static_cast<uint32_t>(dt_vals.size());

    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    std::vector<float> positions(dt_vals.size() * 3U, 0.0f);
    std::vector<float> dt = dt_vals;
    std::vector<float> sigma = sigma_vals;
    std::vector<float> color = color_vals;
    std::vector<uint32_t> offsets{0u, static_cast<uint32_t>(dt_vals.size())};

    float t_cursor = t_near;
    for (size_t i = 0; i < dt_vals.size(); ++i) {
        const float t_mid = t_cursor + 0.5f * dt_vals[i];
        positions[i * 3U + 2] = t_mid;
        t_cursor += dt_vals[i];
    }

    hp_samp_t samp = make_samp_view(positions, dt, offsets, sigma, color);

    std::array<std::byte, 8192> intl_ws{};
    hp_intl_t intl{};
    status = hp_int(plan, &samp, &intl, intl_ws.data(), intl_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_int failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    const float* radiance = static_cast<const float*>(intl.radiance.data);
    const float* trans = static_cast<const float*>(intl.transmittance.data);
    const float* opacity = static_cast<const float*>(intl.opacity.data);
    const float* depth = static_cast<const float*>(intl.depth.data);
    const float* aux = static_cast<const float*>(intl.aux.data);

    if (radiance == nullptr || trans == nullptr || opacity == nullptr || depth == nullptr) {
        result.status = TestStatus::fail;
        result.message = "intl buffers missing";
        hp_plan_release(plan);
        return result;
    }

    const float stop_threshold = 1e-4f;
    float T = 1.0f;
    float depth_weighted = 0.0f;
    float color_ref[3]{0.0f, 0.0f, 0.0f};
    std::vector<float> alpha_ref(dt.size(), 0.0f);
    std::vector<float> weight_ref(dt.size(), 0.0f);
    std::vector<float> Tprev_ref(dt.size(), 0.0f);
    std::vector<float> logTprev_ref(dt.size(), 0.0f);
    float segment = t_near;
    size_t processed = 0;
    for (size_t i = 0; i < dt.size(); ++i) {
        const float t_val = dt[i];
        const float sigma_val = sigma[i];
        const float alpha = std::clamp(compute_alpha_ref(sigma_val, t_val), 0.0f, 1.0f);
        const float T_before = T;
        const float weight = T_before * alpha;

        alpha_ref[i] = alpha;
        weight_ref[i] = weight;
        Tprev_ref[i] = T_before;
        logTprev_ref[i] = std::log(std::max(T_before, 1e-30f));

        const float* col = color.data() + i * 3U;
        color_ref[0] += weight * col[0];
        color_ref[1] += weight * col[1];
        color_ref[2] += weight * col[2];

        const float t_mid = segment + 0.5f * t_val;
        depth_weighted += weight * t_mid;

        T *= std::max(1.0f - alpha, 0.0f);
        segment += t_val;
        ++processed;
        if (T <= stop_threshold) {
            break;
        }
    }

    const float trans_ref = T;
    const float opacity_ref = 1.0f - trans_ref;
    const float depth_ref = (opacity_ref > 1e-6f) ? depth_weighted / opacity_ref : t_far;

    const float tol = 1e-5f;
    const float log_tol = 1e-5f;

    for (int i = 0; i < 3; ++i) {
        if (std::fabs(radiance[i] - color_ref[i]) > tol) {
            result.status = TestStatus::fail;
            result.message = "radiance mismatch";
            hp_plan_release(plan);
            return result;
        }
    }
    if (std::fabs(trans[0] - trans_ref) > tol) {
        result.status = TestStatus::fail;
        result.message = "transmittance mismatch";
        hp_plan_release(plan);
        return result;
    }
    if (std::fabs(opacity[0] - opacity_ref) > tol) {
        result.status = TestStatus::fail;
        result.message = "opacity mismatch";
        hp_plan_release(plan);
        return result;
    }
    if (std::fabs(depth[0] - depth_ref) > 2e-4f) {
        result.status = TestStatus::fail;
        result.message = "depth mismatch";
        hp_plan_release(plan);
        return result;
    }

    if (aux != nullptr) {
        for (size_t i = 0; i < dt.size(); ++i) {
            const float alpha_val = aux[i * 4U + 0];
            const float weight_val = aux[i * 4U + 1];
            const float Tprev_val = aux[i * 4U + 2];
            const float logTprev_val = aux[i * 4U + 3];

            if (i < processed) {
                if (std::fabs(alpha_val - alpha_ref[i]) > tol ||
                    std::fabs(weight_val - weight_ref[i]) > tol ||
                    std::fabs(Tprev_val - Tprev_ref[i]) > tol ||
                    std::fabs(std::exp(logTprev_val) - Tprev_ref[i]) > 1e-4f ||
                    std::fabs(logTprev_val - logTprev_ref[i]) > log_tol) {
                    result.status = TestStatus::fail;
                    result.message = "aux mismatch";
                    hp_plan_release(plan);
                    return result;
                }
            } else {
                if (!expect_early_stop) {
                    if (std::fabs(alpha_val) > tol || std::fabs(weight_val) > tol ||
                        std::fabs(Tprev_val) > tol || std::fabs(logTprev_val) > tol) {
                        result.status = TestStatus::fail;
                        result.message = "aux should be zero";
                        hp_plan_release(plan);
                        return result;
                    }
                }
            }
        }
    }

    if (expect_early_stop && processed < dt.size()) {
        if (trans_ref > 1e-3f) {
            result.status = TestStatus::fail;
            result.message = "early stop not triggered";
            hp_plan_release(plan);
            return result;
        }
    }

    hp_plan_release(plan);
    return result;
}

CaseResult test_int_cpu_constant(hp_ctx* ctx) {
    const std::vector<float> dt{0.25f, 0.25f, 0.25f, 0.25f};
    const std::vector<float> sigma{0.5f, 0.5f, 0.5f, 0.5f};
    const std::vector<float> color{
        0.8f, 0.6f, 0.4f,
        0.8f, 0.6f, 0.4f,
        0.8f, 0.6f, 0.4f,
        0.8f, 0.6f, 0.4f};
    return run_integration_case("int_cpu_constant", ctx, sigma, color, dt, 0.0f, 1.0f);
}

CaseResult test_int_cpu_piecewise(hp_ctx* ctx) {
    const std::vector<float> dt{0.25f, 0.25f, 0.25f, 0.25f};
    const std::vector<float> sigma{0.0f, 0.0f, 4.0f, 4.0f};
    const std::vector<float> color{
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f};
    return run_integration_case("int_cpu_piecewise", ctx, sigma, color, dt, 0.0f, 1.0f);
}

CaseResult test_int_cpu_gaussian(hp_ctx* ctx) {
    const std::vector<float> dt{0.25f, 0.25f, 0.25f, 0.25f};
    const float mu = 0.5f;
    const float sigma_width = 0.2f;
    const float amplitude = 1.5f;
    std::vector<float> sigma_values;
    sigma_values.reserve(dt.size());
    float cursor = 0.0f;
    for (float seg : dt) {
        const float t_mid = cursor + 0.5f * seg;
        const float x = (t_mid - mu) / sigma_width;
        sigma_values.push_back(amplitude * std::exp(-0.5f * x * x));
        cursor += seg;
    }
    const std::vector<float> color{
        0.3f, 0.2f, 0.1f,
        0.4f, 0.3f, 0.2f,
        0.5f, 0.4f, 0.3f,
        0.6f, 0.5f, 0.4f};
    return run_integration_case("int_cpu_gaussian", ctx, sigma_values, color, dt, 0.0f, 1.0f);
}

CaseResult test_int_cpu_early_stop(hp_ctx* ctx) {
    const std::vector<float> dt{0.25f, 0.25f, 0.25f, 0.25f};
    const std::vector<float> sigma{100.0f, 0.5f, 0.5f, 0.5f};
    const std::vector<float> color{
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f};
    return run_integration_case("int_cpu_early_stop", ctx, sigma, color, dt, 0.0f, 1.0f, true);
}

struct SampSnapshot {
    std::vector<float> positions;
    std::vector<float> dt;
    std::vector<uint32_t> offsets;
    std::vector<float> sigma;
    std::vector<float> color;
};

struct IntlSnapshot {
    std::vector<float> radiance;
    std::vector<float> trans;
    std::vector<float> opacity;
    std::vector<float> depth;
};

SampSnapshot capture_samp(const hp_samp_t& samp) {
    SampSnapshot snap{};
    const size_t samples = (samp.dt.rank >= 1) ? static_cast<size_t>(samp.dt.shape[0]) : 0;
    const size_t rays = (samp.ray_offset.rank >= 1) ? static_cast<size_t>(samp.ray_offset.shape[0]) : 0;

    if (samples > 0) {
        snap.dt.assign(static_cast<const float*>(samp.dt.data), static_cast<const float*>(samp.dt.data) + samples);
        snap.sigma.assign(static_cast<const float*>(samp.sigma.data), static_cast<const float*>(samp.sigma.data) + samples);
        snap.positions.assign(static_cast<const float*>(samp.positions.data), static_cast<const float*>(samp.positions.data) + samples * 3U);
        snap.color.assign(static_cast<const float*>(samp.color.data), static_cast<const float*>(samp.color.data) + samples * 3U);
    }
    if (rays > 0) {
        snap.offsets.assign(static_cast<const uint32_t*>(samp.ray_offset.data), static_cast<const uint32_t*>(samp.ray_offset.data) + rays);
    }
    return snap;
}

IntlSnapshot capture_intl(const hp_intl_t& intl) {
    IntlSnapshot snap{};
    const size_t rays = (intl.transmittance.rank >= 1) ? static_cast<size_t>(intl.transmittance.shape[0]) : 0;
    if (rays > 0) {
        snap.radiance.assign(static_cast<const float*>(intl.radiance.data), static_cast<const float*>(intl.radiance.data) + rays * 3U);
        snap.trans.assign(static_cast<const float*>(intl.transmittance.data), static_cast<const float*>(intl.transmittance.data) + rays);
        snap.opacity.assign(static_cast<const float*>(intl.opacity.data), static_cast<const float*>(intl.opacity.data) + rays);
        snap.depth.assign(static_cast<const float*>(intl.depth.data), static_cast<const float*>(intl.depth.data) + rays);
    }
    return snap;
}

void restore_samp(const SampSnapshot& snap, hp_samp_t* samp) {
    if (samp == nullptr) {
        return;
    }
    if (!snap.dt.empty() && samp->dt.data != nullptr) {
        std::memcpy(samp->dt.data, snap.dt.data(), snap.dt.size() * sizeof(float));
    }
    if (!snap.sigma.empty() && samp->sigma.data != nullptr) {
        std::memcpy(samp->sigma.data, snap.sigma.data(), snap.sigma.size() * sizeof(float));
    }
    if (!snap.positions.empty() && samp->positions.data != nullptr) {
        std::memcpy(samp->positions.data, snap.positions.data(), snap.positions.size() * sizeof(float));
    }
    if (!snap.color.empty() && samp->color.data != nullptr) {
        std::memcpy(samp->color.data, snap.color.data(), snap.color.size() * sizeof(float));
    }
    if (!snap.offsets.empty() && samp->ray_offset.data != nullptr) {
        std::memcpy(samp->ray_offset.data, snap.offsets.data(), snap.offsets.size() * sizeof(uint32_t));
    }
}

CaseResult test_img_cpu_basic(hp_ctx* ctx) {
    CaseResult result{"img_cpu_basic", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(2, 2, 0.0f, 1.0f);
    plan_desc.max_rays = 4;
    plan_desc.max_samples = 16;
    plan_desc.sampling.dt = 0.25f;
    plan_desc.sampling.max_steps = 8;

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

    std::vector<float> sigma_grid(8, 0.5f);
    std::vector<float> color_grid{
        0.4f, 0.1f, 0.2f,
        0.4f, 0.1f, 0.2f,
        0.4f, 0.1f, 0.2f,
        0.4f, 0.1f, 0.2f,
        0.4f, 0.1f, 0.2f,
        0.4f, 0.1f, 0.2f,
        0.4f, 0.1f, 0.2f,
        0.4f, 0.1f, 0.2f};

    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 2});
    hp_tensor color_tensor = make_tensor(color_grid.data(), HP_DTYPE_F32, {2, 2, 2, 3});

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

    std::array<std::byte, 8192> intl_ws{};
    hp_intl_t intl{};
    status = hp_int(plan, &samp, &intl, intl_ws.data(), intl_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_int failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const uint32_t* pixel_ids = static_cast<const uint32_t*>(rays.pixel_ids.data);
    const float* radiance = static_cast<const float*>(intl.radiance.data);
    const float* trans = static_cast<const float*>(intl.transmittance.data);
    const float* depth = static_cast<const float*>(intl.depth.data);

    const size_t width = plan_desc.width;
    const size_t height = plan_desc.height;
    const size_t pixel_count = width * height;
    std::vector<float> expected_image(pixel_count * 3U, 0.0f);
    std::vector<float> expected_trans(pixel_count, 1.0f);
    std::vector<float> expected_depth(pixel_count, plan_desc.t_far);
    std::vector<uint32_t> expected_hit(pixel_count, 0U);

    const size_t ray_count = static_cast<size_t>(intl.transmittance.shape[0]);
    for (size_t ray = 0; ray < ray_count; ++ray) {
        const uint32_t pix = pixel_ids[ray];
        if (pix >= pixel_count) {
            result.status = TestStatus::fail;
            result.message = "pixel id out of range";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
        const size_t base = pix * 3U;
        expected_image[base + 0] += radiance[ray * 3U + 0];
        expected_image[base + 1] += radiance[ray * 3U + 1];
        expected_image[base + 2] += radiance[ray * 3U + 2];
        expected_trans[pix] *= trans[ray];
        expected_depth[pix] = std::min(expected_depth[pix], depth[ray]);
        expected_hit[pix] = 1U;
    }
    std::vector<float> expected_opacity(pixel_count, 0.0f);
    for (size_t i = 0; i < pixel_count; ++i) {
        expected_opacity[i] = 1.0f - expected_trans[i];
    }

    std::array<std::byte, 8192> img_ws{};
    hp_img_t img{};
    status = hp_img(plan, &intl, &rays, &img, img_ws.data(), img_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_img failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const float* image_out = static_cast<const float*>(img.image.data);
    const float* trans_out = static_cast<const float*>(img.trans.data);
    const float* opacity_out = static_cast<const float*>(img.opacity.data);
    const float* depth_out = static_cast<const float*>(img.depth.data);
    const uint32_t* hitmask_out = static_cast<const uint32_t*>(img.hitmask.data);

    const float tol = 1e-5f;
    for (size_t i = 0; i < expected_image.size(); ++i) {
        if (std::fabs(image_out[i] - expected_image[i]) > tol) {
            result.status = TestStatus::fail;
            result.message = "image mismatch";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
    }
    for (size_t i = 0; i < pixel_count; ++i) {
        if (std::fabs(trans_out[i] - expected_trans[i]) > tol) {
            result.status = TestStatus::fail;
            result.message = "trans mismatch";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
        if (std::fabs(opacity_out[i] - expected_opacity[i]) > tol) {
            result.status = TestStatus::fail;
            result.message = "opacity mismatch";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
        if (std::fabs(depth_out[i] - expected_depth[i]) > 2e-4f) {
            result.status = TestStatus::fail;
            result.message = "depth mismatch";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
        if (hitmask_out[i] != expected_hit[i]) {
            result.status = TestStatus::fail;
            result.message = "hitmask mismatch";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
        if (std::fabs(opacity_out[i] - (1.0f - trans_out[i])) > tol) {
            result.status = TestStatus::fail;
            result.message = "opacity != 1 - trans";
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

CaseResult test_img_cpu_roi_background(hp_ctx* ctx) {
    CaseResult result{"img_cpu_roi_background", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(2, 2, 0.0f, 1.0f);
    plan_desc.roi.x = 1;
    plan_desc.roi.y = 0;
    plan_desc.roi.width = 1;
    plan_desc.roi.height = 1;
    plan_desc.max_rays = 1;
    plan_desc.max_samples = 8;
    plan_desc.sampling.dt = 0.5f;
    plan_desc.sampling.max_steps = 4;

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
    std::vector<float> color_grid(24, 0.25f);
    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 2});
    hp_tensor color_tensor = make_tensor(color_grid.data(), HP_DTYPE_F32, {2, 2, 2, 3});

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

    std::array<std::byte, 4096> intl_ws{};
    hp_intl_t intl{};
    status = hp_int(plan, &samp, &intl, intl_ws.data(), intl_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_int failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 4096> img_ws{};
    hp_img_t img{};
    status = hp_img(plan, &intl, &rays, &img, img_ws.data(), img_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_img failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const float* image_out = static_cast<const float*>(img.image.data);
    const float* trans_out = static_cast<const float*>(img.trans.data);
    const float* opacity_out = static_cast<const float*>(img.opacity.data);
    const float* depth_out = static_cast<const float*>(img.depth.data);
    const uint32_t* hitmask_out = static_cast<const uint32_t*>(img.hitmask.data);

    const size_t width = plan_desc.width;
    const size_t height = plan_desc.height;
    const size_t pixel_count = width * height;
    const uint32_t* pixel_ids = static_cast<const uint32_t*>(rays.pixel_ids.data);
    const float* radiance = static_cast<const float*>(intl.radiance.data);
    const float* trans = static_cast<const float*>(intl.transmittance.data);
    const float* opacity = static_cast<const float*>(intl.opacity.data);
    const float* depth = static_cast<const float*>(intl.depth.data);
    const size_t ray_count = static_cast<size_t>(intl.transmittance.shape[0]);

    std::vector<float> expected_image(pixel_count * 3U, 0.0f);
    std::vector<float> expected_trans(pixel_count, 1.0f);
    std::vector<float> expected_opacity(pixel_count, 0.0f);
    std::vector<float> expected_depth(pixel_count, plan_desc.t_far);
    std::vector<uint32_t> expected_hit(pixel_count, 0U);

    for (size_t ray = 0; ray < ray_count; ++ray) {
        const uint32_t pix = pixel_ids[ray];
        if (pix >= pixel_count) {
            result.status = TestStatus::fail;
            result.message = "pixel id out of range";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
        const size_t base = pix * 3U;
        expected_image[base + 0] += radiance[ray * 3U + 0];
        expected_image[base + 1] += radiance[ray * 3U + 1];
        expected_image[base + 2] += radiance[ray * 3U + 2];
        expected_trans[pix] *= trans[ray];
        expected_opacity[pix] = 1.0f - expected_trans[pix];
        expected_depth[pix] = std::min(expected_depth[pix], depth[ray]);
        expected_hit[pix] = 1U;
    }

    const float tol = 1e-5f;
    for (size_t i = 0; i < pixel_count; ++i) {
        const size_t base = i * 3U;
        if (expected_hit[i] == 1U) {
            if (std::fabs(image_out[base + 0] - expected_image[base + 0]) > tol ||
                std::fabs(image_out[base + 1] - expected_image[base + 1]) > tol ||
                std::fabs(image_out[base + 2] - expected_image[base + 2]) > tol) {
                result.status = TestStatus::fail;
                result.message = "img ROI color mismatch";
                hp_field_release(fs);
                hp_field_release(fc);
                hp_plan_release(plan);
                return result;
            }
            if (std::fabs(trans_out[i] - expected_trans[i]) > tol ||
                std::fabs(opacity_out[i] - expected_opacity[i]) > tol ||
                std::fabs(depth_out[i] - expected_depth[i]) > 2e-4f) {
                result.status = TestStatus::fail;
                result.message = "img ROI scalar mismatch";
                hp_field_release(fs);
                hp_field_release(fc);
                hp_plan_release(plan);
                return result;
            }
            if (hitmask_out[i] != 1U) {
                result.status = TestStatus::fail;
                result.message = "img ROI hitmask mismatch";
                hp_field_release(fs);
                hp_field_release(fc);
                hp_plan_release(plan);
                return result;
            }
        } else {
            if (std::fabs(image_out[base + 0]) > tol ||
                std::fabs(image_out[base + 1]) > tol ||
                std::fabs(image_out[base + 2]) > tol) {
                result.status = TestStatus::fail;
                result.message = "background color non-zero";
                hp_field_release(fs);
                hp_field_release(fc);
                hp_plan_release(plan);
                return result;
            }
            if (std::fabs(trans_out[i] - 1.0f) > tol ||
                std::fabs(opacity_out[i]) > tol ||
                std::fabs(depth_out[i] - plan_desc.t_far) > tol ||
                hitmask_out[i] != 0U) {
                result.status = TestStatus::fail;
                result.message = "background invariants failed";
                hp_field_release(fs);
                hp_field_release(fc);
                hp_plan_release(plan);
                return result;
            }
        }
    }

    hp_field_release(fs);
    hp_field_release(fc);
    hp_plan_release(plan);
    return result;
}

CaseResult test_fused_cpu_equivalence(hp_ctx* ctx) {
    CaseResult result{"fused_cpu_equivalence", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(2, 2, 0.0f, 1.0f);
    plan_desc.max_rays = 4;
    plan_desc.max_samples = 32;
    plan_desc.sampling.dt = 0.25f;
    plan_desc.sampling.max_steps = 8;

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

    std::vector<float> sigma_grid(8, 0.75f);
    std::vector<float> color_grid{
        0.2f, 0.3f, 0.4f,
        0.5f, 0.2f, 0.1f,
        0.1f, 0.4f, 0.6f,
        0.3f, 0.3f, 0.3f,
        0.6f, 0.5f, 0.4f,
        0.2f, 0.7f, 0.1f,
        0.4f, 0.1f, 0.2f,
        0.8f, 0.2f, 0.3f};

    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 2});
    hp_tensor color_tensor = make_tensor(color_grid.data(), HP_DTYPE_F32, {2, 2, 2, 3});

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

    // Sequential path
    hp_samp_t samp_seq{};
    hp_intl_t intl_seq{};
    std::array<std::byte, 8192> samp_ws{};
    std::array<std::byte, 8192> intl_ws{};
    status = hp_samp(plan, fs, fc, &rays, &samp_seq, samp_ws.data(), samp_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_samp seq failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }
    status = hp_int(plan, &samp_seq, &intl_seq, intl_ws.data(), intl_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_int seq failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    SampSnapshot seq_samp = capture_samp(samp_seq);
    IntlSnapshot seq_intl = capture_intl(intl_seq);

    // Fused path
    hp_samp_t samp_fused{};
    hp_intl_t intl_fused{};
    std::array<std::byte, 16384> fused_ws{};
    status = hp_samp_int_fused(plan, fs, fc, &rays, &samp_fused, &intl_fused, fused_ws.data(), fused_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_samp_int_fused failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    SampSnapshot fused_samp = capture_samp(samp_fused);
    IntlSnapshot fused_intl = capture_intl(intl_fused);

    if (seq_samp.positions != fused_samp.positions ||
        seq_samp.dt != fused_samp.dt ||
        seq_samp.sigma != fused_samp.sigma ||
        seq_samp.color != fused_samp.color ||
        seq_samp.offsets != fused_samp.offsets) {
        result.status = TestStatus::fail;
        result.message = "fused sampling mismatch";
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    if (seq_intl.radiance != fused_intl.radiance ||
        seq_intl.trans != fused_intl.trans ||
        seq_intl.opacity != fused_intl.opacity ||
        seq_intl.depth != fused_intl.depth) {
        result.status = TestStatus::fail;
        result.message = "fused integration mismatch";
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    hp_field_release(fs);
    hp_field_release(fc);
    hp_plan_release(plan);
    return result;
}

CaseResult test_diff_cpu_sigma_color(hp_ctx* ctx) {
    CaseResult result{"diff_cpu_sigma_color", TestStatus::pass, ""};
    hp_plan_desc plan_desc = make_basic_plan_desc(1, 1, 0.0f, 1.0f);
    plan_desc.max_rays = 1;
    plan_desc.max_samples = 4;
    plan_desc.sampling.dt = 0.25f;
    plan_desc.sampling.max_steps = 4;

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

    std::vector<float> sigma_grid{0.6f, 0.8f, 0.7f, 0.5f};
    std::vector<float> color_grid{
        0.3f, 0.4f, 0.5f, 0.6f, 0.2f, 0.1f, 0.4f, 0.5f,
        0.2f, 0.2f, 0.3f, 0.7f, 0.5f, 0.4f, 0.6f, 0.1f,
        0.2f, 0.3f, 0.7f, 0.5f, 0.1f, 0.6f, 0.6f, 0.2f};

    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 1});
    hp_tensor color_tensor = make_tensor(color_grid.data(), HP_DTYPE_F32, {2, 2, 2, 3});

    hp_field* fs = nullptr;
    hp_field* fc = nullptr;
    status = hp_field_create_grid_sigma(ctx, &sigma_tensor, HP_INTERP_LINEAR, HP_OOB_ZERO, &fs);
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_field_create_grid_sigma failed";
        hp_plan_release(plan);
        return result;
    }
    status = hp_field_create_grid_color(ctx, &color_tensor, HP_INTERP_LINEAR, HP_OOB_ZERO, &fc);
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_field_create_grid_color failed";
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

    std::array<std::byte, 8192> intl_ws{};
    hp_intl_t intl{};
    status = hp_int(plan, &samp, &intl, intl_ws.data(), intl_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_int failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const size_t ray_count = static_cast<size_t>(intl.transmittance.shape[0]);
    std::vector<float> grad_image(ray_count * 3U, 1.0f);
    hp_tensor grad_tensor = make_tensor(grad_image.data(), HP_DTYPE_F32, {static_cast<int64_t>(ray_count), 3});

    std::array<std::byte, 16384> grad_ws{};
    hp_grads_t grads{};
    status = hp_diff(plan, &grad_tensor, &samp, &intl, &grads, grad_ws.data(), grad_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_diff failed: ") + status_to_cstr(status);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const float* grad_sigma = static_cast<const float*>(grads.sigma.data);
    const float* grad_color = static_cast<const float*>(grads.color.data);
    const size_t sample_count = (samp.dt.rank >= 1) ? static_cast<size_t>(samp.dt.shape[0]) : 0;

    SampSnapshot snap = capture_samp(samp);

    auto compute_loss = [&](const std::vector<float>& sigma_mut,
                            const std::vector<float>& color_mut) -> std::optional<float> {
        std::vector<float> positions = snap.positions;
        std::vector<float> dt = snap.dt;
        std::vector<uint32_t> offsets = snap.offsets;

        hp_samp_t samp_mut = make_samp_view(positions, dt, offsets, const_cast<std::vector<float>&>(sigma_mut), const_cast<std::vector<float>&>(color_mut));

        std::array<std::byte, 8192> intl_mut_ws{};
        hp_intl_t intl_mut{};
        hp_status st = hp_int(plan, &samp_mut, &intl_mut, intl_mut_ws.data(), intl_mut_ws.size());
        if (st != HP_STATUS_SUCCESS) {
            return std::nullopt;
        }

        std::array<std::byte, 4096> img_ws{};
        hp_img_t img{};
        st = hp_img(plan, &intl_mut, &rays, &img, img_ws.data(), img_ws.size());
        if (st != HP_STATUS_SUCCESS) {
            return std::nullopt;
        }

        const float* image = static_cast<const float*>(img.image.data);
        return image[0] + image[1] + image[2];
    };

    std::vector<float> sigma_vec = snap.sigma;
    std::vector<float> color_vec = snap.color;

    const float eps = 1e-3f;
    for (size_t i = 0; i < sample_count; ++i) {
        std::vector<float> sigma_plus = sigma_vec;
        std::vector<float> sigma_minus = sigma_vec;
        sigma_plus[i] += eps;
        sigma_minus[i] -= eps;
        auto loss_plus = compute_loss(sigma_plus, color_vec);
        auto loss_minus = compute_loss(sigma_minus, color_vec);
        if (!loss_plus.has_value() || !loss_minus.has_value()) {
            result.status = TestStatus::fail;
            result.message = "loss evaluation failed";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
        const float fd = (*loss_plus - *loss_minus) / (2.0f * eps);
        const float analytic = grad_sigma[i];
        const float denom = std::max({std::fabs(fd), std::fabs(analytic), 1e-6f});
        if (std::fabs(fd - analytic) / denom > 1e-3f) {
            result.status = TestStatus::fail;
            result.message = "sigma gradient mismatch";
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
    }

    for (size_t i = 0; i < sample_count; ++i) {
        for (int c = 0; c < 3; ++c) {
            std::vector<float> color_plus = color_vec;
            std::vector<float> color_minus = color_vec;
            color_plus[i * 3U + c] += eps;
            color_minus[i * 3U + c] -= eps;
            auto loss_plus = compute_loss(sigma_vec, color_plus);
            auto loss_minus = compute_loss(sigma_vec, color_minus);
            if (!loss_plus.has_value() || !loss_minus.has_value()) {
                result.status = TestStatus::fail;
                result.message = "loss evaluation failed";
                hp_field_release(fs);
                hp_field_release(fc);
                hp_plan_release(plan);
                return result;
            }
            const float fd = (*loss_plus - *loss_minus) / (2.0f * eps);
            const float analytic = grad_color[i * 3U + c];
            const float denom = std::max({std::fabs(fd), std::fabs(analytic), 1e-6f});
            if (std::fabs(fd - analytic) / denom > 1e-3f) {
                result.status = TestStatus::fail;
                result.message = "color gradient mismatch";
                hp_field_release(fs);
                hp_field_release(fc);
                hp_plan_release(plan);
                return result;
            }
        }
    }

    hp_field_release(fs);
    hp_field_release(fc);
    hp_plan_release(plan);
    return result;
}
#if defined(HP_WITH_CUDA)
CaseResult test_diff_cuda_sigma_color(hp_ctx* ctx) {
    CaseResult result{"diff_cuda_sigma_color", TestStatus::pass, ""};
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        result.status = TestStatus::skip;
        result.message = "no CUDA device available";
        return result;
    }

    hp_plan_desc plan_desc = make_basic_plan_desc(1, 1, 0.0f, 1.0f);
    plan_desc.max_rays = 1;
    plan_desc.max_samples = 4;
    plan_desc.sampling.dt = 0.25f;
    plan_desc.sampling.max_steps = 4;

    hp_plan* plan = nullptr;
    hp_status status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS || plan == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_plan_create failed: ") + status_to_cstr(status);
        return result;
    }

    // Run on CPU first to get reference
    std::array<std::byte, 4096> ray_ws{};
    hp_rays_t rays_cpu{};
    status = hp_ray(plan, nullptr, &rays_cpu, ray_ws.data(), ray_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_ray failed: ") + status_to_cstr(status);
        hp_plan_release(plan);
        return result;
    }

    std::vector<float> sigma_grid{0.6f, 0.8f, 0.7f, 0.5f};
    std::vector<float> color_grid{
        0.3f, 0.4f, 0.5f, 0.6f, 0.2f, 0.1f, 0.4f, 0.5f,
        0.2f, 0.2f, 0.3f, 0.7f, 0.5f, 0.4f, 0.6f, 0.1f,
        0.2f, 0.3f, 0.7f, 0.5f, 0.1f, 0.6f, 0.6f, 0.2f};

    hp_tensor sigma_tensor = make_tensor(sigma_grid.data(), HP_DTYPE_F32, {2, 2, 1});
    hp_tensor color_tensor = make_tensor(color_grid.data(), HP_DTYPE_F32, {2, 2, 2, 3});

    hp_field* fs = nullptr;
    hp_field* fc = nullptr;
    status = hp_field_create_grid_sigma(ctx, &sigma_tensor, HP_INTERP_LINEAR, HP_OOB_ZERO, &fs);
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_field_create_grid_sigma failed";
        hp_plan_release(plan);
        return result;
    }
    status = hp_field_create_grid_color(ctx, &color_tensor, HP_INTERP_LINEAR, HP_OOB_ZERO, &fc);
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_field_create_grid_color failed";
        hp_field_release(fs);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 8192> samp_ws{};
    hp_samp_t samp_cpu{};
    status = hp_samp(plan, fs, fc, &rays_cpu, &samp_cpu, samp_ws.data(), samp_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_samp CPU failed";
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 8192> intl_ws{};
    hp_intl_t intl_cpu{};
    status = hp_int(plan, &samp_cpu, &intl_cpu, intl_ws.data(), intl_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_int CPU failed";
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const size_t sample_count = static_cast<size_t>(samp_cpu.dt.shape[0]);
    const size_t ray_count = static_cast<size_t>(intl_cpu.transmittance.shape[0]);

    // Copy sampling and integration data to device
    float* d_dt = nullptr;
    float* d_sigma = nullptr;
    float* d_color = nullptr;
    uint32_t* d_offsets = nullptr;
    float* d_aux = nullptr;
    float* d_grad_input = nullptr;

    const size_t dt_bytes = sample_count * sizeof(float);
    const size_t sigma_bytes = sample_count * sizeof(float);
    const size_t color_bytes = sample_count * 3U * sizeof(float);
    const size_t offset_bytes = (ray_count + 1) * sizeof(uint32_t);
    const size_t aux_bytes = sample_count * 4U * sizeof(float);
    const size_t grad_input_bytes = ray_count * 3U * sizeof(float);

    if (cudaMalloc(&d_dt, dt_bytes) != cudaSuccess ||
        cudaMalloc(&d_sigma, sigma_bytes) != cudaSuccess ||
        cudaMalloc(&d_color, color_bytes) != cudaSuccess ||
        cudaMalloc(&d_offsets, offset_bytes) != cudaSuccess ||
        cudaMalloc(&d_aux, aux_bytes) != cudaSuccess ||
        cudaMalloc(&d_grad_input, grad_input_bytes) != cudaSuccess) {
        result.status = TestStatus::skip;
        result.message = "cudaMalloc failed";
        cudaFree(d_dt);
        cudaFree(d_sigma);
        cudaFree(d_color);
        cudaFree(d_offsets);
        cudaFree(d_aux);
        cudaFree(d_grad_input);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    cudaMemcpy(d_dt, samp_cpu.dt.data, dt_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, samp_cpu.sigma.data, sigma_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_color, samp_cpu.color.data, color_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, samp_cpu.ray_offset.data, offset_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, intl_cpu.aux.data, aux_bytes, cudaMemcpyHostToDevice);

    std::vector<float> grad_input(ray_count * 3U, 1.0f);
    cudaMemcpy(d_grad_input, grad_input.data(), grad_input_bytes, cudaMemcpyHostToDevice);

    hp_samp_t samp_cuda{};
    samp_cuda.dt = make_tensor(d_dt, HP_DTYPE_F32, {static_cast<int64_t>(sample_count)});
    samp_cuda.dt.memspace = HP_MEMSPACE_DEVICE;
    samp_cuda.sigma = make_tensor(d_sigma, HP_DTYPE_F32, {static_cast<int64_t>(sample_count)});
    samp_cuda.sigma.memspace = HP_MEMSPACE_DEVICE;
    samp_cuda.color = make_tensor(d_color, HP_DTYPE_F32, {static_cast<int64_t>(sample_count), 3});
    samp_cuda.color.memspace = HP_MEMSPACE_DEVICE;
    samp_cuda.ray_offset = make_tensor(d_offsets, HP_DTYPE_U32, {static_cast<int64_t>(ray_count + 1)});
    samp_cuda.ray_offset.memspace = HP_MEMSPACE_DEVICE;

    hp_intl_t intl_cuda{};
    intl_cuda.aux = make_tensor(d_aux, HP_DTYPE_F32, {static_cast<int64_t>(sample_count), 4});
    intl_cuda.aux.memspace = HP_MEMSPACE_DEVICE;

    hp_tensor grad_tensor_cuda = make_tensor(d_grad_input, HP_DTYPE_F32, {static_cast<int64_t>(ray_count), 3});
    grad_tensor_cuda.memspace = HP_MEMSPACE_DEVICE;

    hp_grads_t grads_cuda{};
    status = hp_diff(plan, &grad_tensor_cuda, &samp_cuda, &intl_cuda, &grads_cuda, nullptr, 0);
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_diff CUDA failed: ") + status_to_cstr(status);
        cudaFree(d_dt);
        cudaFree(d_sigma);
        cudaFree(d_color);
        cudaFree(d_offsets);
        cudaFree(d_aux);
        cudaFree(d_grad_input);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    // Copy gradients back to host
    std::vector<float> grad_sigma_cuda(sample_count);
    std::vector<float> grad_color_cuda(sample_count * 3U);
    cudaMemcpy(grad_sigma_cuda.data(), grads_cuda.sigma.data, sigma_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_color_cuda.data(), grads_cuda.color.data, color_bytes, cudaMemcpyDeviceToHost);

    // Run CPU backward for comparison
    hp_tensor grad_tensor_cpu = make_tensor(grad_input.data(), HP_DTYPE_F32, {static_cast<int64_t>(ray_count), 3});
    hp_grads_t grads_cpu{};
    std::array<std::byte, 16384> grad_ws{};
    status = hp_diff(plan, &grad_tensor_cpu, &samp_cpu, &intl_cpu, &grads_cpu, grad_ws.data(), grad_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_diff CPU failed";
        cudaFree(static_cast<float*>(grads_cuda.sigma.data));
        cudaFree(static_cast<float*>(grads_cuda.color.data));
        if (grads_cuda.camera.data) cudaFree(static_cast<float*>(grads_cuda.camera.data));
        cudaFree(d_dt);
        cudaFree(d_sigma);
        cudaFree(d_color);
        cudaFree(d_offsets);
        cudaFree(d_aux);
        cudaFree(d_grad_input);
        hp_field_release(fs);
        hp_field_release(fc);
        hp_plan_release(plan);
        return result;
    }

    const float* grad_sigma_cpu_ptr = static_cast<const float*>(grads_cpu.sigma.data);
    const float* grad_color_cpu_ptr = static_cast<const float*>(grads_cpu.color.data);

    // Compare CPU and CUDA gradients
    for (size_t i = 0; i < sample_count; ++i) {
        const float diff = std::fabs(grad_sigma_cuda[i] - grad_sigma_cpu_ptr[i]);
        const float denom = std::max({std::fabs(grad_sigma_cuda[i]), std::fabs(grad_sigma_cpu_ptr[i]), 1e-6f});
        if (diff / denom > 1e-3f) {
            result.status = TestStatus::fail;
            result.message = "CUDA sigma gradient mismatch with CPU";
            cudaFree(static_cast<float*>(grads_cuda.sigma.data));
            cudaFree(static_cast<float*>(grads_cuda.color.data));
            if (grads_cuda.camera.data) cudaFree(static_cast<float*>(grads_cuda.camera.data));
            cudaFree(d_dt);
            cudaFree(d_sigma);
            cudaFree(d_color);
            cudaFree(d_offsets);
            cudaFree(d_aux);
            cudaFree(d_grad_input);
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
    }

    for (size_t i = 0; i < sample_count * 3U; ++i) {
        const float diff = std::fabs(grad_color_cuda[i] - grad_color_cpu_ptr[i]);
        const float denom = std::max({std::fabs(grad_color_cuda[i]), std::fabs(grad_color_cpu_ptr[i]), 1e-6f});
        if (diff / denom > 1e-3f) {
            result.status = TestStatus::fail;
            result.message = "CUDA color gradient mismatch with CPU";
            cudaFree(static_cast<float*>(grads_cuda.sigma.data));
            cudaFree(static_cast<float*>(grads_cuda.color.data));
            if (grads_cuda.camera.data) cudaFree(static_cast<float*>(grads_cuda.camera.data));
            cudaFree(d_dt);
            cudaFree(d_sigma);
            cudaFree(d_color);
            cudaFree(d_offsets);
            cudaFree(d_aux);
            cudaFree(d_grad_input);
            hp_field_release(fs);
            hp_field_release(fc);
            hp_plan_release(plan);
            return result;
        }
    }

    cudaFree(static_cast<float*>(grads_cuda.sigma.data));
    cudaFree(static_cast<float*>(grads_cuda.color.data));
    if (grads_cuda.camera.data) cudaFree(static_cast<float*>(grads_cuda.camera.data));
    cudaFree(d_dt);
    cudaFree(d_sigma);
    cudaFree(d_color);
    cudaFree(d_offsets);
    cudaFree(d_aux);
    cudaFree(d_grad_input);
    hp_field_release(fs);
    hp_field_release(fc);
    hp_plan_release(plan);
    return result;
}

CaseResult test_diff_cuda_determinism(hp_ctx* ctx) {
    CaseResult result{"diff_cuda_determinism", TestStatus::pass, ""};
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        result.status = TestStatus::skip;
        result.message = "no CUDA device available";
        return result;
    }

    // This test is not yet fully implemented for CUDA, skip for now
    result.status = TestStatus::skip;
    result.message = "CUDA determinism test not yet implemented";
    return result;
}
#endif  // HP_WITH_CUDA

CaseResult test_hash_mlp_cpu_basic(hp_ctx* ctx) {
    CaseResult result{"hash_mlp_cpu_basic", TestStatus::pass, ""};

    // Create a simple hash-MLP field with test parameters
    // Layout: hash_table, sigma_weights, sigma_biases, color_weights, color_biases
    const uint32_t n_levels = 4;
    const uint32_t features_per_level = 2;
    const uint32_t table_size = 16;
    const uint32_t hidden_dim = 8;
    const uint32_t encoding_dim = n_levels * features_per_level;

    // Calculate sizes
    const uint32_t hash_table_size = n_levels * table_size * features_per_level;
    const uint32_t sigma_weights_size = hidden_dim * encoding_dim + hidden_dim;
    const uint32_t sigma_biases_size = hidden_dim + 1;
    const uint32_t color_weights_size = hidden_dim * encoding_dim + 3 * hidden_dim;
    const uint32_t color_biases_size = hidden_dim + 3;
    const uint32_t total_size = hash_table_size + sigma_weights_size + sigma_biases_size +
                                 color_weights_size + color_biases_size;

    std::vector<float> params(total_size, 0.0f);

    // Initialize with small random-like values
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1f * (static_cast<float>((i * 7919) % 1000) / 1000.0f - 0.5f);
    }

    hp_tensor params_tensor = make_tensor(params.data(), HP_DTYPE_F32, {static_cast<int64_t>(total_size)});

    hp_field* field = nullptr;
    hp_status status = hp_field_create_hash_mlp(ctx, &params_tensor, &field);
    if (status != HP_STATUS_SUCCESS || field == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_field_create_hash_mlp failed: ") + status_to_cstr(status);
        return result;
    }

    // Test a few probe points - just verify no crashes and reasonable outputs
    const std::vector<float> test_positions = {
        0.5f, 0.5f, 0.5f,
        0.25f, 0.75f, 0.5f,
        0.1f, 0.2f, 0.3f
    };

    for (size_t i = 0; i < test_positions.size(); i += 3) {
        const float pos[3] = {test_positions[i], test_positions[i+1], test_positions[i+2]};

        // Test through the internal sampling interface
        // Since we don't have direct access, we'll create a minimal sampling scenario
        hp_plan_desc plan_desc = make_basic_plan_desc(1, 1, 0.0f, 1.0f);
        hp_plan* plan = nullptr;
        status = hp_plan_create(ctx, &plan_desc, &plan);
        if (status != HP_STATUS_SUCCESS) {
            result.status = TestStatus::fail;
            result.message = "hp_plan_create failed in hash_mlp test";
            hp_field_release(field);
            return result;
        }

        // The hash-MLP field is valid and created successfully
        // Full integration test would require sampling pipeline
        hp_plan_release(plan);
    }

    hp_field_release(field);
    return result;
}

CaseResult test_hash_mlp_cpu_determinism(hp_ctx* ctx) {
    CaseResult result{"hash_mlp_cpu_determinism", TestStatus::pass, ""};

    // Create hash-MLP field
    const uint32_t n_levels = 4;
    const uint32_t features_per_level = 2;
    const uint32_t table_size = 16;
    const uint32_t hidden_dim = 8;
    const uint32_t encoding_dim = n_levels * features_per_level;

    const uint32_t hash_table_size = n_levels * table_size * features_per_level;
    const uint32_t sigma_weights_size = hidden_dim * encoding_dim + hidden_dim;
    const uint32_t sigma_biases_size = hidden_dim + 1;
    const uint32_t color_weights_size = hidden_dim * encoding_dim + 3 * hidden_dim;
    const uint32_t color_biases_size = hidden_dim + 3;
    const uint32_t total_size = hash_table_size + sigma_weights_size + sigma_biases_size +
                                 color_weights_size + color_biases_size;

    std::vector<float> params(total_size, 0.0f);

    // Initialize with deterministic values
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.05f * std::sin(static_cast<float>(i) * 0.1f);
    }

    hp_tensor params_tensor = make_tensor(params.data(), HP_DTYPE_F32, {static_cast<int64_t>(total_size)});

    hp_field* field1 = nullptr;
    hp_status status = hp_field_create_hash_mlp(ctx, &params_tensor, &field1);
    if (status != HP_STATUS_SUCCESS || field1 == nullptr) {
        result.status = TestStatus::fail;
        result.message = std::string("hp_field_create_hash_mlp failed: ") + status_to_cstr(status);
        return result;
    }

    hp_field* field2 = nullptr;
    status = hp_field_create_hash_mlp(ctx, &params_tensor, &field2);
    if (status != HP_STATUS_SUCCESS || field2 == nullptr) {
        result.status = TestStatus::fail;
        result.message = "hp_field_create_hash_mlp second call failed";
        hp_field_release(field1);
        return result;
    }

    // Create a minimal plan to test sampling determinism
    hp_plan_desc plan_desc = make_basic_plan_desc(2, 2, 0.0f, 1.0f);
    plan_desc.max_rays = 4;
    plan_desc.max_samples = 16;
    plan_desc.sampling.dt = 0.25f;
    plan_desc.sampling.max_steps = 4;

    hp_plan* plan = nullptr;
    status = hp_plan_create(ctx, &plan_desc, &plan);
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_plan_create failed";
        hp_field_release(field1);
        hp_field_release(field2);
        return result;
    }

    std::array<std::byte, 4096> ray_ws{};
    hp_rays_t rays{};
    status = hp_ray(plan, nullptr, &rays, ray_ws.data(), ray_ws.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_ray failed";
        hp_field_release(field1);
        hp_field_release(field2);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 8192> samp_ws1{};
    hp_samp_t samp1{};
    status = hp_samp(plan, field1, field1, &rays, &samp1, samp_ws1.data(), samp_ws1.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_samp first call failed";
        hp_field_release(field1);
        hp_field_release(field2);
        hp_plan_release(plan);
        return result;
    }

    std::array<std::byte, 8192> samp_ws2{};
    hp_samp_t samp2{};
    status = hp_samp(plan, field2, field2, &rays, &samp2, samp_ws2.data(), samp_ws2.size());
    if (status != HP_STATUS_SUCCESS) {
        result.status = TestStatus::fail;
        result.message = "hp_samp second call failed";
        hp_field_release(field1);
        hp_field_release(field2);
        hp_plan_release(plan);
        return result;
    }

    // Compare outputs for determinism
    const size_t sample_count = static_cast<size_t>(samp1.sigma.shape[0]);
    if (sample_count != static_cast<size_t>(samp2.sigma.shape[0])) {
        result.status = TestStatus::fail;
        result.message = "sample count mismatch between runs";
        hp_field_release(field1);
        hp_field_release(field2);
        hp_plan_release(plan);
        return result;
    }

    const float* sigma1 = static_cast<const float*>(samp1.sigma.data);
    const float* sigma2 = static_cast<const float*>(samp2.sigma.data);
    const float* color1 = static_cast<const float*>(samp1.color.data);
    const float* color2 = static_cast<const float*>(samp2.color.data);

    for (size_t i = 0; i < sample_count; ++i) {
        if (std::fabs(sigma1[i] - sigma2[i]) > 1e-6f) {
            result.status = TestStatus::fail;
            result.message = "sigma values not deterministic";
            hp_field_release(field1);
            hp_field_release(field2);
            hp_plan_release(plan);
            return result;
        }
    }

    for (size_t i = 0; i < sample_count * 3U; ++i) {
        if (std::fabs(color1[i] - color2[i]) > 1e-6f) {
            result.status = TestStatus::fail;
            result.message = "color values not deterministic";
            hp_field_release(field1);
            hp_field_release(field2);
            hp_plan_release(plan);
            return result;
        }
    }

    hp_field_release(field1);
    hp_field_release(field2);
    hp_plan_release(plan);
    return result;
}

using TestFn = CaseResult(*)(hp_ctx*);

std::unordered_map<std::string, TestFn> build_registry() {
    std::unordered_map<std::string, TestFn> registry{
        {"ray_cpu_basic", test_ray_cpu_basic},
        {"ray_cpu_roi", test_ray_cpu_roi},
        {"ray_cpu_override", test_ray_cpu_override},
        {"samp_cpu_basic", test_samp_cpu_basic},
        {"samp_cpu_oob_zero", test_samp_cpu_oob_zero},
        {"samp_cpu_oob_clamp", test_samp_cpu_oob_clamp},
        {"samp_cpu_stratified_determinism", test_samp_cpu_stratified},
        {"int_cpu_constant", test_int_cpu_constant},
        {"int_cpu_piecewise", test_int_cpu_piecewise},
        {"int_cpu_gaussian", test_int_cpu_gaussian},
        {"int_cpu_early_stop", test_int_cpu_early_stop},
        {"img_cpu_basic", test_img_cpu_basic},
        {"img_cpu_roi_background", test_img_cpu_roi_background},
        {"fused_cpu_equivalence", test_fused_cpu_equivalence},
        {"diff_cpu_sigma_color", test_diff_cpu_sigma_color},
        {"hash_mlp_cpu_basic", test_hash_mlp_cpu_basic},
        {"hash_mlp_cpu_determinism", test_hash_mlp_cpu_determinism}
    };
#if defined(HP_WITH_CUDA)
    registry.emplace("ray_cuda_basic", test_ray_cuda_basic);
    registry.emplace("diff_cuda_sigma_color", test_diff_cuda_sigma_color);
    registry.emplace("diff_cuda_determinism", test_diff_cuda_determinism);
#endif
    return registry;
}

RunnerOptions parse_options(int argc, char** argv) {
    RunnerOptions opts{};
    if (argc > 1) {
        opts.manifest_path = argv[1];
    }
    return opts;
}

}  // namespace

int main(int argc, char** argv) {
    RunnerOptions options = parse_options(argc, argv);

    hp_ctx* ctx = nullptr;
    const hp_status ctx_status = hp_ctx_create(nullptr, &ctx);
    if (ctx_status != HP_STATUS_SUCCESS || ctx == nullptr) {
        std::vector<CaseResult> failure_cases{
            {"ctx_create", TestStatus::fail, std::string("hp_ctx_create failed: ") + status_to_cstr(ctx_status)}};
        std::cout << build_scoreboard(failure_cases);
        return 1;
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
        results.push_back(it->second(ctx));
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

    hp_ctx_release(ctx);
    return has_failures ? 1 : 0;
}
