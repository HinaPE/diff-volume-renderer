# DVREN: Full-System Design Specification

**Status**: Draft 2025-10-07  
**Owner**: DVREN Core Team  
**Audience**: Renderer, ML, and Systems engineers contributing to `diff-volume-renderer`

---

## 0. Abstract

The DVREN project delivers a production-ready differentiable volume renderer that exposes a stable host-facing API, supports dense and neural fields, integrates with optimization pipelines, and ships with robust testing and tooling. The `hotpath/` subproject already implements the research-grade, performance-critical SISO pipeline (ray → sample → integrate → compose → backward) with CPU parity and CUDA fast paths. This document defines the surrounding system required to build a fully featured renderer around `hotpath/`, covering host runtime orchestration, asset and field management, differentiable training loops, bindings, tools, and a staged roadmap.

---

## 1. Scope and Non-Goals

### In Scope
- Wrap `hotpath` as an internal engine while presenting a cohesive C++ API (`libdvren`) and Python bindings.
- Manage scenes, camera orchestration, field assets (dense grids, hash-MLPs, future sparse structures), and differentiable training jobs.
- Provide end-to-end rendering and backward pipelines with batching, caching, mixed precision, and gradient accumulation utilities.
- Deliver tooling: configuration system, profiler, deterministic replay, sample fixtures, and CI integration.
- Ship runnable applications (CLI demos, regression harness) and documentation.

### Out of Scope (for this cycle)
- Full-featured GUI viewer; recommend minimal CLI inspection tools.
- Advanced light transport (multiple scattering, participating media beyond emission–absorption).
- Distributed multi-node training; single-process multi-GPU is a stretch goal.
- Asset authoring tools (we ingest but do not edit volumes).

---

## 2. System Overview

```
            +----------------------------+
            | Applications & Bindings    |
            |  - CLI tools               |
            |  - Python API (PyTorch)    |
            +-------------+--------------+
                          |
            +-------------v--------------+
            | Host Runtime (libdvren)    |
            |  - Context/Plan builder    |
            |  - Scene & camera system   |
            |  - Batch scheduler         |
            |  - Field registry          |
            |  - Differentiation utils   |
            +-------------+--------------+
                          |
            +-------------v--------------+
            | Hotpath Engine             |
            |  - hp_ray / hp_samp ...    |
            |  - Fused + Graph paths     |
            |  - CPU reference, CUDA     |
            +-------------+--------------+
                          |
            +-------------v--------------+
            | Device Runtime             |
            |  - CUDA streams, allocs    |
            |  - Memory pools            |
            |  - Graph capture/exe       |
            +----------------------------+
```

Data flows from host-level scene description through `libdvren` orchestration into `hotpath` APIs. Outputs (images, auxiliary buffers, gradients) are wrapped back into host tensors and made available to downstream consumers (optimizers, learning frameworks).

---

## 3. Architectural Layers

| Layer | Description | Key Deliverables |
|-------|-------------|------------------|
| **hotpath** | Existing research-grade kernels (SISO forward/backward, fused pipeline, CUDA Graph). | `hp_*` API, tests, perf gates. |
| **core** | Host runtime bridging `libdvren` calls to `hotpath`, managing contexts, resource lifetimes, error propagation, and logging. | `core/context.hpp`, `core/plan.hpp`, `core/error.hpp`. |
| **render** | High-level forward/backward orchestration (batched rendering, checkpointing, event tracing). | `render/renderer.hpp`, `render/batch_executor.cpp`. |
| **fields** | Field providers (dense grid, hash-MLP, occupancy-aware, future sparse). Implements conversions, gradient uploads/downloads. | `fields/dense.hpp`, `fields/hash_mlp.hpp`, `fields/occupancy.hpp`. |
| **scene** | Cameras, transforms, animation tracks, ray packing utilities, ROI management. | `scene/camera.hpp`, `scene/ray_packer.cpp`. |
| **diff** | Optimizer integration (Adam/SGD wrappers), gradient accumulation, mixed precision scaler, autodiff glue for PyTorch/JAX. | `diff/optimizer.hpp`, `bindings/python/autograd.cpp`. |
| **io** | Asset ingestion (OpenVDB, NumPy grids, config loading), dataset samplers. | `io/vdb_loader.cpp`, `io/config_loader.cpp`. |
| **tools** | Profilers, deterministic replay, CLI wrappers, benchmarking harness. | `tools/profile_cli.cpp`, `tools/replay.cpp`. |
| **tests** | Contract, unit, integration, perf, and smoke tests layered atop hotpath coverage. | `tests/unit/`, `tests/integration/`, `tests/perf/`. |
| **docs** | User guides, API references, tutorials. | `docs/architecture.md`, `docs/tutorials/`. |

---

## 4. Core Concepts and Data Model

### 4.1 Context & Plan
- `dvren::Context` wraps `hp_ctx`; attaches logging, allocator policies, device selection hints, and capability flags (CUDA presence, graph support, mixed precision).
- `dvren::Plan` composes render settings: image resolution, ROI, sampling config, integrator parameters, gradient toggles. Plans are immutable and cache compiled substructures (CUDA graph, workspace sizing).

### 4.2 Tensor Abstraction
- `dvren::TensorView` (host) and `dvren::DeviceTensor` (device) expose strided data referencing either user buffers or internally managed pools.
- Align shapes with `hp_tensor` but add small type-safe wrapper and convenience constructors.

### 4.3 Fields
- Dense grid: world-space bounding boxes, optional mip pyramid, gradient buffers (host or device) with zeroing and download APIs.
- Hash-MLP: metadata describing encoding (levels, features per level, network widths) and parameter tensor.
- Future (roadmap): NanoVDB, Neural SDF.

### 4.4 Scene Graph Lite
- Minimal DAG storing transforms, volume instances, camera rigs.
- Ray packing converts scenes to `hp_rays_t` via ROI/culling.
- Supports time sampling for motion blur (later milestone).

### 4.5 Differentiation
- `dvren::BackwardContext` stores handle to `hotpath` saved buffers, gradient accumulation mode (write, atomic, reduce).
- Mixed precision scaler with dynamic loss scaling to avoid NaNs.
- Optional reparameterization (e.g., density = softplus(raw_density)).

---

## 5. Forward Pipeline

1. **Scene Preparation**: Cameras and volume fields registered with `dvren::Scene`; static validation ensures alignment between density and emission fields.
2. **Plan Resolution**: Per-frame/per-batch `Plan` determined from render settings (ROI, sampling mode, seed) and device capabilities.
3. **Ray Generation**: 
   - Use `scene::RayPacker` to produce `hp_rays_t`.
   - Prefer ROI-specific layouts; fall back to `hp_ray` if user requests canonical camera sampling.
4. **Sampling & Integration**:
   - Default path uses `hp_samp_int_fused` for CUDA; CPU uses separate `hp_samp` + `hp_int`.
   - Workspace sizing computed via `Plan::workspace_sizes()`.
5. **Image Composition**: `hp_img` writes outputs to `dvren::FrameBuffer`, storing RGB, alpha, transmittance, depth, and hit mask.
6. **Auxiliary Capture**:
   - `ForwardResult` retains sample offsets, transmittance curves, and timing stats for diagnostics.
7. **Graph Execution**:
   - For steady-state loops, capture `hp_graph_*` once per plan; subsequent execution bypasses kernel launch overhead.

---

## 6. Backward Pipeline

1. **Gradient Preparation**: Accept `dL/dImage` from users (host or device). Validate dtype/layout.
2. **Invoke `hp_diff`** with previously captured sampling/integration buffers.
3. **Field Gradient Accumulation**: Route gradients into field providers; optionally combine with host-side loss regularizers before applying optimizer.
4. **Camera Gradient Handling**: Convert `hp_grads_t.camera` into extrinsic/intrinsic derivatives; integrate with scene parameter update modules.
5. **Optimizer Step**: Provide extensible optimizer registry (Adam, L-BFGS, SGD), optionally using fused CUDA primitives for speed.

Backward must remain deterministic: the host runtime enforces ordered reduction and seeded sampling; any stochastic components must log seeds for replay.

---

## 7. Error Handling, Logging, and Diagnostics

- Central `dvren::Status` type mirrors `hp_status` but annotates with module, message, and optional remediation hints.
- Structured logging (JSONL) toggled via `Context` flags; integrates with profiling pipelines.
- Diagnostic captures: per-ray dumps, gradient histograms, occupancy stats, CPU reference comparisons.

---

## 8. Build, Packaging, and Dependencies

- **CMake**: root project builds `hotpath` (optionally as subproject) and `libdvren`. Provide targets:
  - `dvren::hotpath` (existing)
  - `dvren::core`
  - `dvren::fields`
  - `dvren::bindings_python`
  - `dvren::tools`
- **Third-Party**:
  - Mandatory: CUDA 12.x, C++23 (host).
  - Optional: OpenVDB/NanoVDB, tiny-cuda-nn, PyTorch (autoload), Eigen (host math), fmt/spdlog (logging).
- Provide `find_package` modules or vendored wrappers for optional dependencies.
- Artifacts: static and shared builds, Python wheel (`pip install dvren`), CLI binaries.

---

## 9. Testing and CI Strategy

| Scope | Description | Source |
|-------|-------------|--------|
| **Contract** | Leverage `hotpath/tests` manifest; run CPU+CUDA OJ-style tests. | Existing. |
| **Unit** | Host runtime units (plan builder, tensor utils, field registry). | New `tests/unit/`. |
| **Integration** | End-to-end render + backward on fixtures (dense grid slab, hash-MLP sphere). | `tests/integration/`. |
| **Performance** | Continuously monitor rays/s, samples/s, backward grads/s using `tools/profile_cli`. Thresholds stored in `tests/perf/thresholds.yaml`. | Existing + new. |
| **Determinism** | Replay runs with identical seeds, positive and negative tests. | New harness. |
| **Bindings** | Python autograd gradient parity vs. numerical finite differences. | `bindings/python/tests`. |

CI matrix covers Windows + Linux (CUDA-enabled runners). Build artifacts uploaded for reproducibility; thresholds locked via scripts mirroring `hotpath/scripts`.

---

## 10. Documentation Plan

- `docs/quickstart.md`: From install to rendering first frame.
- `docs/api_cpp.md` and `docs/api_python.md`: Reference for high-level APIs.
- `docs/pipelines/nerf_training.md`: Example training loop using gradients.
- Auto-generated Doxygen pages for C++; Sphinx for Python.
- Keep `hotpath/DESIGN_SPECIFICATION.md` referenced from top-level docs, clarifying division of responsibilities.

---

## 11. Security and Reliability Considerations

- Sandboxed execution for tests reading untrusted assets.
- Bounds checking on host tensors before passing into `hotpath`.
- Optional validation mode runs CPU and CUDA paths and compares, gated for debugging builds.
- Memory pooling to avoid fragmentation; guard pages in debug builds.
- Telemetry opt-in only; no implicit network activity.

---

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Divergence between host abstractions and hotpath API changes | Build breakage, regressions | Wrap hotpath calls in thin adapters; add ABI compatibility tests. |
| High memory footprint for dense grids | OOM on consumer GPUs | Implement tiled loading, FP16 paths, occupancy compression early in roadmap. |
| Python binding maintenance overhead | Slower releases | Auto-generate bindings with pybind11 macros; add ABI smoke tests. |
| Determinism regressions under heavy batching | Test flakes | Record seeds, add deterministic replay tests, incorporate `hp_graph` counters. |
| Stretch goals (NanoVDB, multi-GPU) delay core features | Scope creep | Enforce milestone gates; only start stretch work post P1 acceptance. |

---

## 13. Roadmap

### Phase P0 — Infrastructure Bring-up (Weeks 1-2)
1. Integrate `hotpath` CMake into root build; expose `dvren::hotpath` target.
2. Implement `core/context`, `core/status`, `core/plan` wrappers with unit tests.
3. Create dense grid field provider bridging host tensors to `hp_field_create_*`.
4. Add minimal CLI `dvren_render` that loads JSON config, renders via CPU path, dumps image.
5. Establish CI pipeline reusing `hotpath/scripts/validate_gates.py`.

### Phase P1 — Differentiable Forward/Backward (Weeks 3-5)
1. Implement renderer orchestrator (forward, fused CUDA path, graph capture toggles).
2. Implement backward wrapper returning gradients to fields and camera structs.
3. Add integration tests: dense grid slab forward/backward, match CPU vs CUDA outputs.
4. Expose C++ API headers under `include/dvren/`.
5. Introduce telemetry/logging, workspace inspection tools.

### Phase P2 — Python & Optimization Loop (Weeks 6-8)
1. pybind11 bindings for Context, Plan, Fields, Renderer.
2. Autograd integration for PyTorch (custom `Function`) and minimal JAX prototype.
3. Implement optimizers (Adam/SGD) and gradient accumulation utilities.
4. Provide sample notebook + CLI training demo (fit synthetic Gaussian).
5. Add binding tests ensuring gradient parity.

### Phase P3 — Performance & Tooling (Weeks 9-11)
1. Integrate occupancy grid skipping and gradient-aware early termination toggles.
2. Memory pool & mixed precision support.
3. Profiling CLI with JSON reports; deterministic replay harness.
4. Performance thresholds for fused kernels vs. baseline.
5. Document best practices, update quickstart.

### Phase P4 — Extended Fields and Assets (Weeks 12-14)
1. NanoVDB ingestion (optional dependency), bridging to `hp_field`.
2. Hash-MLP parameter loader; connect to tiny-cuda-nn (if available).
3. Scene animation hooks (camera paths, per-frame parameters).
4. Additional unit/perf tests for new field types.

### Phase P5 — Release Hardening (Weeks 15-16)
1. Finalize API review; tag v0.2.0.
2. Freeze performance thresholds; archive golden fixtures.
3. Package Python wheels, CLI binaries; update docs and changelog.
4. Postmortem + backlog triage for next cycle (multi-GPU, viewer).

---

## 14. Acceptance Criteria

- Ability to render and backpropagate through a dense grid scene using both CPU and CUDA paths with identical outputs within tolerance.
- Python users can script optimization loops using provided bindings and receive gradients.
- CI runs hotpath contract tests plus new integration/unit suites on Windows and Linux.
- Documentation covers installation, API usage, and at least one worked differentiable example.
- Performance targets meet or exceed hotpath baselines; graphs demonstrate ≤5% variance under steady-state.

---

## 15. References

- `hotpath/DESIGN_SPECIFICATION.md` — authoritative reference for SISO pipeline and kernel contracts.
- `hotpath/include/hotpath/hp.h` — C API signatures for engine integration.
- `hotpath/tests/manifest.yaml` — canonical functional and performance test coverage.
- DVREN README (root) — legacy API snapshot; new design supersedes sections as milestones land.

---

This specification aligns the broader DVREN project around the proven hotpath implementation while charting a clear path to a fully featured differentiable renderer suited for research and production workloads.

