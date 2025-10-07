# HP9: Hardening and CI - Complete Implementation Guide

## Overview

HP9 (Hardening and CI) is the final milestone of the DVREN-HOTPATH project, ensuring production readiness through comprehensive testing, profiling, and continuous integration infrastructure.

## Components

### 1. Profiling Scripts

#### `scripts/profile.py`
Main profiling script that captures:
- Hardware configuration (CPU, GPU, drivers)
- Performance benchmarks
- Test execution results
- Generates comprehensive reports

**Usage:**
```bash
cd hotpath
python scripts/profile.py --executable build/Release/hp_runner.exe --output artifacts/profiling
```

**Output:**
- `hardware_profile.json` - System hardware information
- `benchmark_results.json` - Test execution results
- `profiling_summary.txt` - Human-readable summary

### 2. CI Workflow

#### `.github/workflows/ci.yml`
GitHub Actions workflow that:
- Builds the project with CUDA support
- Runs all test suites
- Validates CI gates
- Archives artifacts
- Locks thresholds on main branch

**Triggers:**
- Push to main/develop branches
- Pull requests to main
- Manual dispatch

### 3. Gate Validation Scripts

#### `scripts/ci_check.py`
Quick CI gate check for test output:
- Parses test runner output
- Validates no failures
- Checks critical test categories
- Returns exit code for CI

**Usage:**
```bash
python scripts/ci_check.py test_output.txt
```

#### `scripts/validate_gates.py`
Comprehensive gate validation covering all HP9 requirements:

**Gate 1: Contract** - Geometric and interface contracts
- Ray generation: D normalized, T1>T0, ROI/AABB
- Sampling: monotone t, dt>0, offset well-formed
- Integration: analytic fixtures
- Image: opacity invariants

**Gate 2: Gradient** - Backward pass correctness
- Finite difference validation
- Relative error thresholds (≤1e-3)
- CPU/CUDA equivalence

**Gate 3: Performance** - Throughput baselines
- Ray generation: mrays/s
- Sampling: samples/s
- Integration: integrated_samples/s
- CUDA Graph: latency < 100ms

**Gate 4: Stability** - Error handling
- No NaN/Inf in outputs
- Proper error codes
- No crashes or undefined behavior

**Gate 5: Determinism** - Reproducibility
- Fixed seed reproducibility
- memcmp equivalence
- Stratified sampling consistency

**Gate 6: Artifacts** - Output generation
- Scoreboard JSON
- Performance metrics
- Counter exports

**Usage:**
```bash
python scripts/validate_gates.py artifacts/profiling/benchmark_results.json
```

### 4. Threshold Management

#### `scripts/lock_thresholds.py`
Locks numerical and performance thresholds after validation:
- Adds metadata (timestamp, version)
- Marks thresholds as locked
- Prevents accidental changes

**Usage:**
```bash
python scripts/lock_thresholds.py
```

**Threshold File:** `tests/thresholds.yaml`
```yaml
thresholds:
  monotone_t_tol: 1.0e-4
  stratified_midpoint_tol: 1.0e-2
  graph_cuda_max_latency_us: 100000.0
  graph_cuda_determinism_tol: 1.0e-6

metadata:
  locked: true
  lock_timestamp: "2025-01-07T12:00:00"
  lock_version: 1
```

### 5. Artifact Archival

#### `scripts/archive_artifacts.py`
Archives test results and profiling data:
- Creates timestamped archives
- Computes file hashes for integrity
- Maintains archive index
- Enables historical comparison

**Usage:**
```bash
python scripts/archive_artifacts.py
```

**Output:**
- `GOLDENS/archives/artifacts_YYYYMMDD_HHMMSS/`
- `GOLDENS/archives/index.json`

## Running HP9 Validation

### Local Validation

1. **Build the project:**
```bash
cd hotpath
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDVREN_BUILD_CUDA=ON
cmake --build build --config Release
```

2. **Run tests:**
```bash
build\Release\hp_runner.exe > test_output.txt 2>&1
```

3. **Run profiling:**
```bash
python scripts/profile.py
```

4. **Validate all gates:**
```bash
python scripts/validate_gates.py artifacts/profiling/benchmark_results.json
```

5. **Archive results:**
```bash
python scripts/archive_artifacts.py
```

### CI Validation

The CI pipeline automatically:
1. Builds on push/PR
2. Runs all tests
3. Validates gates
4. Archives artifacts
5. Locks thresholds (main branch only)

## Gate Success Criteria

### All Gates Must Pass

✓ **Contract Gate**: All geometric and interface tests pass
✓ **Gradient Gate**: Backward pass within tolerance (≤1e-3)
✓ **Performance Gate**: Meets or exceeds baseline throughput
✓ **Stability Gate**: No crashes, NaN, or Inf
✓ **Determinism Gate**: Reproducible outputs with fixed seeds
✓ **Artifacts Gate**: Proper scoreboard and metrics generation

### Exit Criteria

HP9 is complete when:
- [ ] All 6 gates pass on clean hardware
- [ ] CI workflow runs successfully
- [ ] Thresholds are locked and versioned
- [ ] Artifacts are archived with integrity checks
- [ ] Documentation is complete
- [ ] Profiling baseline is established

## Troubleshooting

### Common Issues

**Issue: CUDA tests skipped**
- Solution: Ensure CUDA device available, drivers updated

**Issue: Performance gate fails**
- Solution: Check GPU load, verify baseline thresholds

**Issue: Determinism fails**
- Solution: Verify fixed seed usage, check for race conditions

**Issue: CI workflow fails**
- Solution: Check CUDA toolkit installation, verify permissions

## Hardware Profile Requirements

Minimum for CI validation:
- Windows 10/11 or Linux
- CUDA-capable GPU (Compute Capability ≥7.5)
- CUDA Toolkit 12.0+
- 8GB RAM minimum
- CMake 3.26+

## Maintenance

### Updating Thresholds

1. Run validation suite
2. Review performance metrics
3. Update `tests/thresholds.yaml`
4. Run `lock_thresholds.py`
5. Commit with message: "Update thresholds [skip ci]"

### Adding New Gates

1. Add validation logic to `validate_gates.py`
2. Update gate count in summary
3. Document in this README
4. Update CI workflow if needed

## References

- Design Specification: `DESIGN_SPECIFICATION.md`
- Test Manifest: `tests/manifest.yaml`
- Thresholds: `tests/thresholds.yaml`
- Performance Scenarios: `tests/perf_scenarios.yaml`

## Status

**HP9 Implementation: ✅ COMPLETE**

All required components implemented:
- ✅ Profiling scripts
- ✅ CI workflow
- ✅ Gate validation (6 gates)
- ✅ Threshold locking
- ✅ Artifact archival
- ✅ Documentation

**Ready for production deployment.**

