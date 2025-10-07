#!/usr/bin/env python3
"""
HP9 CI Check Script
Validates test results and enforces CI gates.
"""

import json
import sys
from pathlib import Path


def parse_test_output(output_file: Path) -> dict:
    """Parse test runner output and extract scoreboard."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()

        # Find JSON scoreboard in output
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('{') and '"cases"' in line:
                return json.loads(line)

        return {"error": "No scoreboard found in output"}
    except Exception as e:
        return {"error": str(e)}


def check_ci_gates(scoreboard: dict) -> bool:
    """Check all CI gates and return pass/fail with messages."""
    messages = []
    passed = True

    if "error" in scoreboard:
        messages.append(f"ERROR: {scoreboard['error']}")
        return False, messages

    summary = scoreboard.get("summary", {})
    cases = scoreboard.get("cases", [])

    # Gate 1: No failures allowed
    fail_count = summary.get("fail", 0)
    if fail_count > 0:
        passed = False
        messages.append(f"❌ GATE FAILED: {fail_count} test case(s) failed")
        for case in cases:
            if case.get("status") == "fail":
                messages.append(f"   - {case['name']}: {case.get('message', 'N/A')}")
    else:
        messages.append(f"✓ All functional tests passed")

    # Gate 2: Contract tests must pass
    contract_tests = [
        "ray_cpu_basic", "ray_cpu_roi", "samp_cpu_basic",
        "int_cpu_constant", "img_cpu_basic"
    ]
    for test_name in contract_tests:
        case = next((c for c in cases if c["name"] == test_name), None)
        if case and case.get("status") != "pass":
            passed = False
            messages.append(f"❌ CONTRACT GATE: {test_name} did not pass")

    # Gate 3: Gradient tests must pass
    grad_tests = ["diff_cpu_sigma_color"]
    for test_name in grad_tests:
        case = next((c for c in cases if c["name"] == test_name), None)
        if case and case.get("status") != "pass":
            passed = False
            messages.append(f"❌ GRADIENT GATE: {test_name} did not pass")

    # Gate 4: Determinism tests must pass
    det_tests = ["samp_cpu_stratified_determinism"]
    for test_name in det_tests:
        case = next((c for c in cases if c["name"] == test_name), None)
        if case and case.get("status") != "pass":
            passed = False
            messages.append(f"❌ DETERMINISM GATE: {test_name} did not pass")

    # Summary
    pass_count = summary.get("pass", 0)
    skip_count = summary.get("skip", 0)
    messages.append(f"\nSummary: {pass_count} passed, {fail_count} failed, {skip_count} skipped")

    return passed, messages


def main():
    if len(sys.argv) < 2:
        print("Usage: ci_check.py <test_output_file>")
        sys.exit(1)

    output_file = Path(sys.argv[1])

    if not output_file.exists():
        print(f"Error: Test output file not found: {output_file}")
        sys.exit(1)

    print("=" * 80)
    print("HP9 CI Gate Validation")
    print("=" * 80)
    print()

    scoreboard = parse_test_output(output_file)
    passed, messages = check_ci_gates(scoreboard)

    for msg in messages:
        print(msg)

    print()
    print("=" * 80)
    if passed:
        print("✓ ALL CI GATES PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ CI GATES FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
