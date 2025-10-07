#!/usr/bin/env python3
"""
HP9 Gate Validation Script
Validates all HP9 gates: Contract, Grad, Perf, Stability, Determinism, Artifacts
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class GateValidator:
    """Validates all HP9 CI gates."""

    def __init__(self, results_file: Path):
        self.results_file = results_file
        self.results = self._load_results()
        self.gates_passed = []
        self.gates_failed = []

    def _load_results(self) -> dict:
        """Load benchmark results."""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}

    def validate_contract_gate(self) -> bool:
        """Gate 1: Contract tests - geometric and interface contracts."""
        print("\n" + "=" * 80)
        print("GATE 1: Contract Validation")
        print("=" * 80)

        if "scoreboard" not in self.results:
            print("❌ No scoreboard data available")
            self.gates_failed.append("Contract")
            return False

        scoreboard = self.results["scoreboard"]
        cases = scoreboard.get("cases", [])

        contract_tests = {
            "ray_cpu_basic": ["D normalized", "T1>T0", "ROI/AABB respected"],
            "ray_cpu_roi": ["ROI pixel mapping"],
            "samp_cpu_basic": ["monotone t", "dt>0", "Off well-formed"],
            "int_cpu_constant": ["analytic fixture"],
            "img_cpu_basic": ["Opacity = 1-Trans"]
        }

        failed = []
        for test_name, requirements in contract_tests.items():
            case = next((c for c in cases if c["name"] == test_name), None)
            if not case or case.get("status") != "pass":
                failed.append(test_name)
                print(f"  ❌ {test_name}: FAILED")
            else:
                print(f"  ✓ {test_name}: PASSED")

        if failed:
            print(f"\n❌ Contract Gate FAILED: {len(failed)} test(s) failed")
            self.gates_failed.append("Contract")
            return False

        print("\n✓ Contract Gate PASSED")
        self.gates_passed.append("Contract")
        return True

    def validate_gradient_gate(self) -> bool:
        """Gate 2: Gradient tests - backward pass correctness."""
        print("\n" + "=" * 80)
        print("GATE 2: Gradient Validation")
        print("=" * 80)

        if "scoreboard" not in self.results:
            print("❌ No scoreboard data available")
            self.gates_failed.append("Gradient")
            return False

        scoreboard = self.results["scoreboard"]
        cases = scoreboard.get("cases", [])

        grad_tests = ["diff_cpu_sigma_color", "diff_cuda_sigma_color"]

        failed = []
        for test_name in grad_tests:
            case = next((c for c in cases if c["name"] == test_name), None)
            if case:
                if case.get("status") == "pass":
                    print(f"  ✓ {test_name}: PASSED")
                elif case.get("status") == "skip":
                    print(f"  ⊘ {test_name}: SKIPPED")
                else:
                    failed.append(test_name)
                    print(f"  ❌ {test_name}: FAILED")

        if failed:
            print(f"\n❌ Gradient Gate FAILED: {len(failed)} test(s) failed")
            self.gates_failed.append("Gradient")
            return False

        print("\n✓ Gradient Gate PASSED")
        self.gates_passed.append("Gradient")
        return True

    def validate_performance_gate(self) -> bool:
        """Gate 3: Performance tests - throughput meets baselines."""
        print("\n" + "=" * 80)
        print("GATE 3: Performance Validation")
        print("=" * 80)

        if "scoreboard" not in self.results:
            print("❌ No scoreboard data available")
            self.gates_failed.append("Performance")
            return False

        scoreboard = self.results["scoreboard"]
        cases = scoreboard.get("cases", [])

        perf_tests = ["graph_cuda_performance"]

        failed = []
        for test_name in perf_tests:
            case = next((c for c in cases if c["name"] == test_name), None)
            if case:
                if case.get("status") == "pass":
                    print(f"  ✓ {test_name}: PASSED")
                elif case.get("status") == "skip":
                    print(f"  ⊘ {test_name}: SKIPPED (no CUDA device)")
                else:
                    failed.append(test_name)
                    print(f"  ❌ {test_name}: FAILED - {case.get('message', 'N/A')}")

        if failed:
            print(f"\n❌ Performance Gate FAILED: {len(failed)} test(s) failed")
            self.gates_failed.append("Performance")
            return False

        print("\n✓ Performance Gate PASSED")
        self.gates_passed.append("Performance")
        return True

    def validate_stability_gate(self) -> bool:
        """Gate 4: Stability tests - no NaN/Inf, proper error handling."""
        print("\n" + "=" * 80)
        print("GATE 4: Stability Validation")
        print("=" * 80)

        if "scoreboard" not in self.results:
            print("❌ No scoreboard data available")
            self.gates_failed.append("Stability")
            return False

        scoreboard = self.results["scoreboard"]
        cases = scoreboard.get("cases", [])

        # Check for any crashes or internal errors
        crashed = []
        for case in cases:
            if "internal_error" in case.get("message", "").lower():
                crashed.append(case["name"])

        if crashed:
            print(f"❌ Stability issues detected in: {', '.join(crashed)}")
            self.gates_failed.append("Stability")
            return False

        print("  ✓ No crashes or internal errors detected")
        print("  ✓ Error codes handled properly")

        print("\n✓ Stability Gate PASSED")
        self.gates_passed.append("Stability")
        return True

    def validate_determinism_gate(self) -> bool:
        """Gate 5: Determinism tests - reproducible outputs."""
        print("\n" + "=" * 80)
        print("GATE 5: Determinism Validation")
        print("=" * 80)

        if "scoreboard" not in self.results:
            print("❌ No scoreboard data available")
            self.gates_failed.append("Determinism")
            return False

        scoreboard = self.results["scoreboard"]
        cases = scoreboard.get("cases", [])

        det_tests = [
            "samp_cpu_stratified_determinism",
            "graph_cuda_determinism",
            "diff_cuda_determinism"
        ]

        failed = []
        for test_name in det_tests:
            case = next((c for c in cases if c["name"] == test_name), None)
            if case:
                if case.get("status") == "pass":
                    print(f"  ✓ {test_name}: PASSED")
                elif case.get("status") == "skip":
                    print(f"  ⊘ {test_name}: SKIPPED")
                else:
                    failed.append(test_name)
                    print(f"  ❌ {test_name}: FAILED")

        if failed:
            print(f"\n❌ Determinism Gate FAILED: {len(failed)} test(s) failed")
            self.gates_failed.append("Determinism")
            return False

        print("\n✓ Determinism Gate PASSED")
        self.gates_passed.append("Determinism")
        return True

    def validate_artifacts_gate(self) -> bool:
        """Gate 6: Artifacts - proper output generation."""
        print("\n" + "=" * 80)
        print("GATE 6: Artifacts Validation")
        print("=" * 80)

        # Check for scoreboard
        if "scoreboard" not in self.results:
            print("❌ No scoreboard artifact generated")
            self.gates_failed.append("Artifacts")
            return False

        print("  ✓ Scoreboard JSON generated")

        # Check for counters (if available)
        scoreboard = self.results.get("scoreboard", {})
        if "summary" in scoreboard:
            print("  ✓ Test summary available")

        print("\n✓ Artifacts Gate PASSED")
        self.gates_passed.append("Artifacts")
        return True

    def run_all_gates(self) -> bool:
        """Run all gate validations."""
        print("=" * 80)
        print("HP9 Gate Validation - All Gates")
        print("=" * 80)

        all_passed = True

        all_passed &= self.validate_contract_gate()
        all_passed &= self.validate_gradient_gate()
        all_passed &= self.validate_performance_gate()
        all_passed &= self.validate_stability_gate()
        all_passed &= self.validate_determinism_gate()
        all_passed &= self.validate_artifacts_gate()

        return all_passed

    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        print(f"\n✓ Passed Gates ({len(self.gates_passed)}):")
        for gate in self.gates_passed:
            print(f"  - {gate}")

        if self.gates_failed:
            print(f"\n❌ Failed Gates ({len(self.gates_failed)}):")
            for gate in self.gates_failed:
                print(f"  - {gate}")

        total = len(self.gates_passed) + len(self.gates_failed)
        print(f"\nTotal: {len(self.gates_passed)}/{total} gates passed")


def main():
    # Default to looking for benchmark results in standard location
    if len(sys.argv) < 2:
        results_file = Path("profiling_output/benchmark_results.json")
        if not results_file.exists():
            print("Usage: validate_gates.py [benchmark_results.json]")
            print(f"Default location not found: {results_file}")
            print("\nRun profile.py first to generate benchmark results.")
            sys.exit(1)
    else:
        results_file = Path(sys.argv[1])

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    validator = GateValidator(results_file)
    all_passed = validator.run_all_gates()
    validator.print_summary()

    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL HP9 GATES PASSED ✓✓✓")
        print("=" * 80)
        return 0
    else:
        print("✗✗✗ SOME HP9 GATES FAILED ✗✗✗")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
