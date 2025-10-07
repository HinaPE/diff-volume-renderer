# dvren — Differentiable Volume Renderer

[![Windows](https://github.com/HinaPE/diff-volume-renderer/actions/workflows/windows-build.yml/badge.svg)](https://github.com/HinaPE/diff-volume-renderer/actions/workflows/windows-build.yml)
[![Linux](https://github.com/HinaPE/diff-volume-renderer/actions/workflows/linux-build.yml/badge.svg)](https://github.com/HinaPE/diff-volume-renderer/actions/workflows/linux-build.yml)

> High-throughput Simple-In Simple-Out (SISO) volume rendering core with first-class differentiation, CPU parity, and CUDA fast paths.

---

## Overview

`dvren` packages the research-grade **hotpath** kernels into a host-friendly library and CLI. It provides:

- Deterministic forward and backward volume rendering across CPU/CUDA,
- Swappable field backends (dense grid shipped; hash MLP / NanoVDB planned),
- A consistent C API (`hotpath/hp.h`) and ergonomic C++ wrappers,
- Automation for workspace planning, fused pipelines, and CUDA Graph capture (with graceful fallbacks),
- Tooling for validation, profiling, and regression testing.

The project is organized as:

- `hotpath/` — reference kernels, tests, scripts.
- `include/dvren/` & `src/` — host runtime (`Context`, `Plan`, `Renderer`, field providers).
- `apps/dvren_render/` — JSON-driven CLI renderer.
- `tests/core/` — integration tests comparing staged/fused/graph executions.

---

## Features

- **Forward Rendering**
  - Emission–absorption integration with early termination.
  - Configurable sampling (fixed or stratified) and ROI-aware ray generation.
  - Staged pipeline (`hp_samp` + `hp_int`) or fused execution (`hp_samp_int_fused`) with automatic fallback.
- **Backward Differentiation**
  - Gradients w.r.t. density, color, and camera intrinsics/extrinsics.
  - Sample-space gradient accumulation mapped back to dense grids.
- **CUDA Graph (Opt-in)**
  - Toggleable capture path; silently falls back when unsupported (e.g., CPU-only builds).
- **Workspace Accounting**
  - Renderer exposes per-stage buffer usage; CLI prints totals for quick inspection.
- **Deterministic Testing**
  - Integration test ensures staged vs. fused parity and checks graph-enabled runs.

---

## Requirements

| Dependency | Minimum Version | Notes |
|------------|-----------------|-------|
| CMake      | 3.26            | Toolchain configuration |
| C++        | C++23 compiler  | MSVC 19.36+, Clang 16+, GCC 12+ |
| CUDA (opt) | 12.x            | Required for GPU fast paths and CUDA Graphs |
| Python     | 3.8+            | Running hotpath validation scripts (optional) |

Windows uses a vendored `vcpkg` manifest (`vcpkg.json`) for optional dependencies (OpenVDB, TBB, etc.).

---

## Building from Source

```bash
# Configure (Ninja + Release recommended)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build renderer library, CLI, and tests
cmake --build build --target dvren_renderer dvren_render dvren_core_tests --config Release
```

If CUDA is unavailable, set `-DDVREN_ENABLE_CUDA=OFF`. The build system will still produce CPU binaries and automatically skip fused/graph paths.

Key CMake options:

| Option                    | Default | Description |
|---------------------------|---------|-------------|
| `DVREN_ENABLE_CUDA`       | `ON`    | Enable CUDA compilation and fast paths when a CUDA compiler is present. |
| `DVREN_BUILD_PROJECT_TESTS` | `ON`  | Build and register dvren integration tests. |
| `DVREN_BUILD_APPS`        | `ON`    | Build the `dvren_render` CLI. |
| `DVREN_ENABLE_HOTPATH_TESTS` | `OFF` | Pass through to `hotpath/` to compile its test runner. |

The hotpath library is exposed via the alias `dvren::hotpath`, and host wrappers link against it by default.

---

## Running the CLI Renderer

```bash
./build/Release/dvren_render.exe examples/simple_volume.json output.ppm
```

Sample output:

```
Forward stats: rays=16 samples=160 total_ms=0.0265
Workspace bytes total=30648 sample=8260 integration=4480 gradient=4144 scratch=12740
Wrote "/path/to/output.ppm"
```

### Configuration Schema (`examples/simple_volume.json`)

```jsonc
{
  "render": {
    "width": 512,
    "height": 512,
    "t_near": 0.1,
    "t_far": 4.0,
    "dt": 0.01,
    "max_steps": 256,
    "sampling_mode": "fixed",       // or "stratified"
    "seed": 42,
    "options": {
      "use_fused_path": true,
      "enable_graph": false,
      "capture_stats": true
    }
  },
  "volume": {
    "size": [64, 64, 64],
    "density": [...],
    "color": [...],                 // defaults to density if omitted
    "bbox_min": [0.0, 0.0, 0.0],
    "bbox_max": [1.0, 1.0, 1.0],
    "interp": "linear",             // or "nearest"
    "oob": "zero"                   // or "clamp"
  },
  "output": {
    "path": "frame.ppm"
  }
}
```

CLI always writes a PPM (P6) image and reports workspace usage for quick capacity checks.

---

## C++ API Overview

```cpp
#include <dvren/core/context.hpp>
#include <dvren/core/plan.hpp>
#include <dvren/fields/dense_grid.hpp>
#include <dvren/render/renderer.hpp>

// Create context
dvren::Context ctx;
dvren::Context::Create(dvren::ContextOptions{}, ctx);

// Describe render plan
dvren::PlanDescriptor plan_desc{};
plan_desc.width = 1024;
plan_desc.height = 1024;
plan_desc.t_near = 0.0f;
plan_desc.t_far = 5.0f;
plan_desc.sampling.dt = 0.01f;
plan_desc.sampling.max_steps = 512;
plan_desc.max_rays = plan_desc.width * plan_desc.height;
plan_desc.max_samples = plan_desc.max_rays * plan_desc.sampling.max_steps;

dvren::Plan plan;
dvren::Plan::Create(ctx, plan_desc, plan);

// Dense grid field on host
dvren::DenseGridConfig grid_cfg{};
grid_cfg.resolution = {64, 64, 64};
grid_cfg.sigma.assign(64 * 64 * 64, 0.1f);
grid_cfg.color.assign(64 * 64 * 64 * 3, 0.5f);

dvren::DenseGridField field;
dvren::DenseGridField::Create(ctx, grid_cfg, field);

// Renderer with fused + graph enabled
dvren::RenderOptions opts{};
opts.use_fused_path = true;
opts.enable_graph = true;

dvren::Renderer renderer(ctx, plan, opts);
dvren::ForwardResult forward;
dvren::Status st = renderer.Forward(field, forward);
if (!st.ok()) {
    throw std::runtime_error(st.ToString());
}

// Backprop using upstream image gradients
std::vector<float> dL_dI(forward.ray_count * 3, 1.0f);
dvren::BackwardResult backward;
renderer.Backward(field, dL_dI, backward);
```

`Renderer::workspace_info()` exposes buffer sizes (rays, samples, integration, gradients, scratch). Graph capture is opportunistic: unsupported builds simply leave `graph_forward_captured_` false and record a message in `RenderStats::notes`.

---

## Testing

Integration tests are enabled by default:

```bash
ctest --output-on-failure -C Release
```

`tests/core/test_core.cpp` runs three scenarios—staged, fused, and graph-enabled—verifying that forward images and gradients match (tolerance `1e-4`) and that workspace totals are non-zero. This ensures regression coverage for the Phase P0/P1 deliverables.

To execute the hotpath validation suite:

```bash
cmake --build build --target hp_runner --config Release
./build/hotpath/Release/hp_runner.exe
```

---

## Contributing

1. Fork and clone the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Add unit/integration tests alongside your changes.
4. Run `ctest` and the CLI smoke test before submitting.
5. Open a pull request with a clear description and link to the relevant design milestone.

We follow a phased roadmap (`DESIGN_SPECIFICATION.md`), advancing from kernel refinements to bindings, toolchains, and performance gates. Please review in-flight phases before proposing major additions.

---

## License

This project is released under the MIT License (see `LICENSE`). Contributions are welcome under the same terms.
