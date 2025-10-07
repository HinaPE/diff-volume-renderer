# HP9 Quick Start Guide

## What is HP9?

HP9 (Hardening and CI) is the final milestone that ensures DVREN-HOTPATH is production-ready through:
- **Profiling**: Performance and hardware metrics
- **CI Pipeline**: Automated testing and validation
- **Gate Validation**: 6 critical quality gates
- **Threshold Locking**: Version-controlled performance baselines
- **Artifact Archival**: Historical test results and metrics

## Quick Validation (5 minutes)

### Windows (Easiest)

```bash
cd C:\Users\imeho\Desktop\25.09.06\diff-volume-renderer\hotpath
run_hp9_validation.bat
```

This automated script will:
1. ✅ Build the project
2. ✅ Run all tests
3. ✅ Check CI gates
4. ✅ Generate profiling report
5. ✅ Validate all 6 HP9 gates

### Manual Steps

```bash
# 1. Build
cd hotpath
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# 2. Run tests
build\Release\hp_runner.exe > test_output.txt 2>&1

# 3. Profile
python scripts\profile.py --output artifacts\profiling

# 4. Validate gates
python scripts\validate_gates.py artifacts\profiling\benchmark_results.json
```

## The 6 HP9 Gates

| Gate | Purpose | Pass Criteria |
|------|---------|---------------|
| 1. Contract | Interface correctness | All geometric tests pass |
| 2. Gradient | Backward pass accuracy | rel_err ≤ 1e-3 |
| 3. Performance | Throughput baselines | Meets target mrays/s |
| 4. Stability | Error handling | No NaN/Inf/crashes |
| 5. Determinism | Reproducibility | Fixed seed consistency |
| 6. Artifacts | Output generation | Valid scoreboard JSON |

## Expected Output

### ✅ Success
```
================================================================================
HP9 Gate Validation - All Gates
================================================================================

GATE 1: Contract Validation
  ✓ ray_cpu_basic: PASSED
  ✓ samp_cpu_basic: PASSED
  ...
✓ Contract Gate PASSED

[... 5 more gates ...]

FINAL SUMMARY
================================================================================
✓ Passed Gates (6):
  - Contract
  - Gradient
  - Performance
  - Stability
  - Determinism
  - Artifacts

Total: 6/6 gates passed

================================================================================
✓✓✓ ALL HP9 GATES PASSED ✓✓✓
================================================================================
```

### ❌ Failure
If any gate fails, you'll see:
```
❌ Performance Gate FAILED: 1 test(s) failed
  ❌ graph_cuda_performance: FAILED - graph execution too slow: 150000.0 us
```

**Fix**: Check GPU load, update thresholds, or optimize code.

## Outputs

After running HP9 validation, you'll have:

```
hotpath/
├── artifacts/
│   └── profiling/
│       ├── hardware_profile.json      # GPU/CPU info
│       ├── benchmark_results.json     # Test results
│       └── profiling_summary.txt      # Human-readable report
├── test_output.txt                    # Raw test output
└── GOLDENS/
    └── archives/                      # Historical archives
        ├── artifacts_20250107_120000/
        └── index.json
```

## Key Files

- **`HP9_README.md`** - Complete HP9 documentation
- **`run_hp9_validation.bat`** - Automated validation script
- **`scripts/validate_gates.py`** - Gate validation logic
- **`scripts/profile.py`** - Profiling tool
- **`.github/workflows/ci.yml`** - CI pipeline

## Troubleshooting

**CUDA tests skipped?**
→ Install CUDA Toolkit 12.0+ and update GPU drivers

**Performance gate fails?**
→ Close other GPU applications, check thresholds in `tests/thresholds.yaml`

**Build fails?**
→ Ensure CMake 3.26+, CUDA Toolkit, and C++23 compiler are installed

## Lock Thresholds (Production)

Once all gates pass consistently:

```bash
python scripts\lock_thresholds.py
git add tests/thresholds.yaml
git commit -m "HP9: Lock thresholds after validation"
```

## CI Integration

Push to main/develop triggers automatic CI:
1. Build with CUDA
2. Run all tests
3. Validate 6 gates
4. Archive artifacts
5. Lock thresholds (main only)

Check `.github/workflows/ci.yml` for configuration.

## Next Steps

✅ HP0-HP8 Complete → All features implemented
✅ HP9 Complete → Production-ready with CI/CD

**You're done!** The renderer is fully validated and ready for integration.

## Support

- Review `HP9_README.md` for detailed documentation
- Check `DESIGN_SPECIFICATION.md` for requirements
- See test output in `test_output.txt` for failures

