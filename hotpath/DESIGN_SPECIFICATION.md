# DVREN-HOTPATH: Design Specification (for AI Agents)

## 0. Abstract

This document specifies a research-grade, highly performant **differentiable volume renderer** with a **SISO (Simple-In Simple-Out)** modular architecture. The renderer targets two simultaneous goals: (i) serve as a drop-in, high-throughput backend for Instant-NGP–style pipelines; (ii) remain extensible to a general-purpose differentiable volumetric renderer. The system is **data-driven**, **decoupled**, and **C++23/CUDA**-centric with CPU parity for reference and testing. Determinism, numerical robustness, and composable interfaces are first-class constraints.

## 1. Scope and Non-Goals

* In scope: emission–absorption volume rendering, forward and backward (gradients w.r.t. densities, colors, and camera), dense-grid and hash-MLP fields, fused kernels, CUDA Graph capture, OJ-style tests (functional and performance).
* Non-goals: full scene graph, path tracing effects beyond emission–absorption, production UI/UX, file I/O beyond small test fixtures.

## 2. Core Objectives

* **P0 Mandatory**: exact forward renderer (rays, sampling, field evaluation, integration, image composition), exact backward pass, deterministic counters, minimal field backends (dense grid, hash-MLP stub), fused sampling+integration, CUDA Graph steady-state, strict OJ tests and CI gates.
* **Performance**: end-to-end throughput competitive with state-of-the-art Instant-NGP paths; memory footprint minimized via SoA layouts and fused paths; stable microsecond-level kernel variance under steady-state.
* **Generalization**: clean interfaces to swap fields, samplers, and integrators; config-driven, no hard-coded scene logic.

## 3. Design Principles

* **SISO narrow waist**: every module is a pure function from typed tensors to typed tensors; no global mutable state beyond explicit counters.
* **Data-driven**: all behavior is configured by compact structs and manifest-like inputs; no compile-time specialization required for correctness.
* **CPU reference, CUDA fastpath**: CPU provides ground-truth and determinism; CUDA provides throughput. Interfaces are identical.
* **Determinism**: fixed RNG seeds, ordered or compensated reductions, stable sort keys, and reproducible memory iteration order.
* **Numerical robustness**: log-transmittance path for high optical depth, `expm1` for alpha, compensated reductions where needed.
* **Additive evolution**: API evolves via capability flags and `next` pointers; no breaking parameter reordering.

## 4. System Overview (Level-1)

Black-box I/O:

* **Inputs**: camera parameters, volume field(s) (density and color), renderer config, optional rays override, optional loss gradient dL/dI.
* **Outputs**: rendered images (RGB, transmittance, opacity, depth, hit mask), intermediate integration buffers for backward, gradients (dSigma, dColor, dCam), counters, performance metrics.

## 5. Module Decomposition (Level-2, SISO)

1. **Ray Generation (`f_ray`)**
   In: camera, ROI, AABB, seed
   Out: rays O, D, T0, T1, pixel ids P

2. **Sampling + Field Query (`f_samp`)**
   In: rays, field_sigma, field_color, sampling policy
   Out: per-sample positions X, dt, RayId, SegId, Off; Sigma, Color

3. **Integration (`f_int`)**
   In: per-sample Sigma, Color, dt, per-ray segment offsets
   Out: per-ray C, Tfinal, optional Depth; persisted Alpha/Tau, Tprev/LogTprev, Valid, HitMask

4. **Image Composition (`f_img`)**
   In: per-ray integration outputs + pixel ids P
   Out: Image, Trans, Opacity, Depth, Hit, Count

5. **Fused Sampling+Integration (`f_fused`)**
   In: camera/rays, fields, sampling policy
   Out: union of `f_samp` minimal buffers + `f_int` outputs; equivalent numerics within tolerance

6. **Backward (`f_diff`)**
   In: dL/dI, `f_samp` buffers, `f_int` buffers, rays
   Out: gradients dSigma, dColor, dCam; optional dTF; counters

7. **Fields (`field_*`)**
   Dense grid trilinear and hash-MLP. Uniform world-to-grid mapping, OOB policy, interpolation mode.

## 6. Minimal API Surface (signatures only)

ASCII-only, no comments, function names unchanged. Example:

```
hp_status hp_ray (const hp_plan* plan, const hp_rays_t* override_or_null, hp_rays_t* rays, void* ws, size_t ws_bytes);
hp_status hp_samp(const hp_plan* plan, const hp_field* fs, const hp_field* fc, const hp_rays_t* rays, hp_samp_t* samp, void* ws, size_t ws_bytes);
hp_status hp_int (const hp_plan* plan, const hp_samp_t* samp, hp_intl_t* intl, void* ws, size_t ws_bytes);
hp_status hp_img (const hp_plan* plan, const hp_intl_t* intl, const hp_rays_t* rays, hp_img_t* img, void* ws, size_t ws_bytes);
hp_status hp_diff(const hp_plan* plan, const hp_tensor* dL_dI, const hp_samp_t* samp, const hp_intl_t* intl, hp_grads_t* grads, void* ws, size_t ws_bytes);

hp_status hp_field_create_grid_sigma(const hp_ctx* ctx, const hp_tensor* grid, uint32_t interp, uint32_t oob, hp_field** out);
hp_status hp_field_create_grid_color(const hp_ctx* ctx, const hp_tensor* grid, uint32_t interp, uint32_t oob, hp_field** out);
hp_status hp_field_create_hash_mlp  (const hp_ctx* ctx, const hp_tensor* params, hp_field** out);
```

## 7. Data Layout and Types

* **SoA** arrays for rays, samples, and images, 128B alignment where possible.
* **Tensors** are unowned views (ptr + shape + stride + dtype + memspace).
* **DTypes**: F16/BF16 compute allowed, F32 accumulation; images and gradients in F32.
* **Offsets**: per-ray prefix `Off` with `Off[0]=0`, `Off[N]=M`, monotone.

## 8. Numerical Contracts

* Emission–absorption:

    * `alpha_k = 1 - exp(-sigma_k * dt_k)`
    * `T_{k+1} = T_k * (1 - alpha_k)`, `T_0 = 1`
    * `C = sum_k T_k * alpha_k * c_k`
* Stability: log-T path for high optical depth; `expm1` for alpha; no NaN/Inf permitted; clamp or zero OOB policy.
* Determinism: fixed seeds; ordered reductions via shuffles and shared-memory trees; canonical sort keys for pixel bucketing.

## 9. Performance Principles

* Fused sampling+integration to remove a global-memory roundtrip.
* Occupancy skipping via bitfields or hierarchical masks (optional in P0 if dense fields are used).
* CUDA Graph capture to amortize launch overhead under steady-state loops.
* Persistent-CTA or grid-stride per-ray loops tuned per device.

## 10. Error Handling and Versioning

* Status codes: OK, ERROR, NOT_SUPPORTED. No exceptions across the C ABI.
* Capability discovery via `hp_header.caps` and `hp_plan` creation.
* Backward-compatible evolution: only additive parameters via `next` pointer chains or optional tensors (empty = disabled).

## 11. Build and Isolation

* `hotpath/` is a standalone subproject: builds CPU reference and CUDA fastpath, plus OJ runner when activated.
* Thin adapters map `hp_*` to the main project API without touching hotpath kernels.

---

# OJ-Style Testing Specification (for AI Agents)

## A. Testing Philosophy

* **OJ mindset**: a manifest describes cases; the runner executes and judges.
* **Separation**: algorithm code never parses YAML; the runner never reimplements math.
* **Deterministic adjudication**: thresholds and seeds are explicit and versioned.

## B. Test Inputs

* `manifest.yaml`: declarative cases for functional tests and numerical goldens.
* `thresholds.yaml`: scalar gates for correctness, gradient, performance, stability, determinism.
* `perf_scenarios.yaml`: reproducible microbenchmarks (sizes, warmup, iters, flags).

## C. Assertion Taxonomy

* Geometry and contract: `norm_eq`, `gt`, `monotone_t`, `off_well_formed`, `oob_policy`.
* Numeric equivalence: `max_abs`, `rel_err`, fused-vs-nonfused deltas.
* Gradient checks: sparse finite differences per-parameter type with relative error gates.
* Determinism: `memcmp_same` or `max_abs` with tiny tolerance.
* Stability: `no_nan_inf`, early-stop rate ranges.
* Performance: throughput or speedup gates (`mrays/s`, `samples/s`, `pixels/s`, `grads/s`, `fused_speedup`).

## D. Artifacts and Reporting

* `scoreboard.json`: per-case pass/fail and metrics.
* `GOLDENS/artifacts/*`: `.png`, `.npy`, `.json` arrays and plots.
* Counters: rays, samples, early_stop, saturated_alpha exposed for sanity checks.

## E. CI Gates

* All functional assertions pass.
* All gradient checks pass with configured tolerances.
* Performance meets or exceeds targets on reference hardware.
* Stability: zero NaN/Inf; OOM returns error code, not crash.
* Determinism: pass for fixed seeds.

---

# Implementation Roadmap (Agent Playbook)

## Milestone HP0: Bootstrap

**Deliverables**

* Minimal C ABI headers, context/plan, empty runner skeleton, empty manifests.
  **Acceptance**
* Build succeeds; runner prints empty scoreboard.

## Milestone HP1: Ray Generation (CPU, CUDA)

**Code**

* `ray_cpu.cpp`, `ray_cuda.cu`, plan dispatch.
  **Functional Tests**
* D normalized, `T1 > T0`, ROI/AABB clipping exact, determinism memcmp.
  **Perf Tests**
* `mrays/s` baseline on `H=W=1024`.
  **Artifacts**
* Scoreboard, counters json.
  **Exit**
* All green; thresholds frozen.

## Milestone HP2: Sampling + Dense Grid Field (CPU ref, CUDA placeholder OK)

**Code**

* `samp_cpu.cpp`, `grid_dense_cpu.cpp`, CUDA stubs ready; plan dispatch.
  **Functional Tests**
* Monotone t, `Off` well-formed, OOB zero/clamp consistent, stratified KS p>=0.05.
  **Perf Tests**
* `samples/s` baseline on `H=W=256..512`, `dt`, `max_steps` fixed.
  **Exit**
* All green; thresholds frozen.

## Milestone HP3: Integration (CPU ref, CUDA)

**Code**

* `int_cpu.cpp`, `int_cuda.cu`, persisted buffers for backward.
  **Functional Tests**
* Analytic goldens: constant, piecewise slabs, 3D Gaussian; logT vs T within small tolerance; early-stop reasonable range.
  **Perf Tests**
* `integrated_samples/s`.
  **Exit**
* All green; thresholds frozen.

## Milestone HP4: Image Composition (CPU ref, CUDA)

**Code**

* `img_cpu.cpp`, `img_cuda.cu`, stable bucketed composition and atomic fallback.
  **Functional Tests**
* `Opacity = 1 - Trans`, empty pixels = background, bucket counts exact; path equivalence sorted vs atomic.
  **Perf Tests**
* `pixels/s` baseline on `H=W=1024`.
  **Exit**
* All green.

## Milestone HP5: Fused Sampling+Integration (CUDA)

**Code**

* `fused/samp_int_fused.cu` with minimal persisted buffers.
  **Functional Tests**
* Equivalence to non-fused for C and Tfinal within thresholds.
  **Perf Tests**
* `fused_speedup >= 1.2`, memory peak reduced.
  **Exit**
* All green.

## Milestone HP6: Backward (CUDA; CPU FD referee)

**Code**

* `diff_cuda.cu`, `diff_cpu.cpp` sparse finite-difference probes.
  **Functional Tests**
* rel_err(dSigma)<=1e-3, rel_err(dColor)<=1e-3, rel_err(dCam)<=2e-3, determinism memcmp; fused backward equals non-fused.
  **Perf Tests**
* `grads/s` on fixed fixture.
  **Exit**
* All green.

## Milestone HP7: Field Backend Hash-MLP (CUDA; CPU stub)

**Code**

* `hash_mlp_cuda.cu` deterministic hash and matvec order; CPU probe stub.
  **Functional Tests**
* Agreement at probe points; gradients match FD.
  **Perf Tests**
* Higher `samples/s` vs dense grid under occupancy fixtures.
  **Exit**
* All green.

## Milestone HP8: Steady-State CUDA Graph

**Code**

* `graph_cuda.cpp` capture f_ray -> f_fused -> f_img -> f_diff loop.
  **Perf Tests**
* Latency reduction vs non-graph; throughput not worse; timing variance low.
  **Exit**
* All green.

## Milestone HP9: Hardening and CI

**Code/Infra**

* Profiling scripts, CI workflow, artifact archival, thresholds locking.
  **Gates**
* Contract, Grad, Perf, Stability, Determinism, Artifacts all pass on clean hardware profile.

---

# Per-Module Contracts (summary)

* **f_ray**: `||D||=1`, `T1>T0`, ROI/AABB respected, deterministic output for seed.
* **f_samp**: per-ray monotone `t`, `dt>0`, `Off[0]=0`, `Off[N]=M` monotone, OOB policy exact, stratified sampling statistics pass.
* **f_int**: analytic fixtures within tolerance; logT path equals T path within tiny epsilon in benign regimes; no NaN/Inf; early-stop sensible.
* **f_img**: invariants hold; sorted vs atomic equivalence.
* **f_fused**: numerical equivalence vs non-fused; speedup and memory reduction.
* **f_diff**: gradient rel errors within gates; deterministic; fused vs non-fused equality.
* **field_*:** trilinear dense grid exact vs CPU; hash-MLP deterministic and FD-aligned.

---

# Configuration and Fixtures (agent guidance)

* **Camera**: pinhole and ortho models; intrinsics `K[9]`, `C2W[12]`, near/far; ROI.
* **Sampling**: `dt`, `max_steps`, mode (fixed, stratified), interpolation enum, OOB enum.
* **Fields**: dense grid shapes `(nz,ny,nx)` with world AABB; hash-MLP params blob; seed fixed.
* **Fixtures**: constant volumes, piecewise slabs, Gaussian blobs, occupancy shells; seeds recorded.

---

# Performance Targets (initial)

* RayGen: `mrays/s` target configurable; record device and driver.
* Sampling: `samples/s` >= baseline (device-dependent).
* Integration: `integrated_samples/s` >= baseline.
* Image composition: `pixels/s` >= baseline.
* Backward: `grads/s` >= baseline.
* Fused speedup: `>= 1.2` initial.
  Targets are stored in `thresholds.yaml` and must be version-locked per environment.

---

# Execution Order for Agents

1. Create `hotpath/` scaffold (HP0) and confirm runner or standalone tests build and execute.
2. Implement and greenlight HP1..HP3; freeze thresholds after each.
3. Add HP4 and validate invariants; freeze.
4. Implement HP5 fused and prove equivalence and speedup.
5. Implement HP6 backward with FD referee; lock gradient tolerances.
6. Add HP7 hash-MLP backend; verify determinism and FD.
7. Capture HP8 CUDA Graph; report latency and variance reductions.
8. Finalize HP9 CI gates and profiling; archive artifacts and hardware hashes.

---

# Failure Handling and Debugging (agent checklist)

* If a functional test fails, dump per-ray or per-sample tensors for the smallest failing fixture and diff against CPU reference.
* If gradients fail FD gates, ensure logT vs T consistency, `expm1` usage, and reduction ordering.
* If determinism fails, inspect any atomics, shared-memory race conditions, or non-stable sort keys.
* If perf regresses, verify fused path is active, occupancy and launch parameters are appropriate, and CUDA Graph is enabled for steady-state loops.

---

# Deliverables Summary

* Source tree: `hotpath/` with `include/`, `src/` (cpu, cuda, fields, runtime), `tests/` (contract, perf), `TESTS/` manifests, `GOLDENS/` artifacts.
* Minimal C ABI for all P0 modules.
* OJ-style manifests and thresholds with JSON scoreboard artifacts.
* CI pipeline and profiling scripts.

This specification and roadmap enable an AI agent to plan, implement, and validate the hotpath renderer in clearly staged increments while preserving determinism, numerical integrity, and performance.
