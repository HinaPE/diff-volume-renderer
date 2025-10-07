#!/usr/bin/env python3
"""
HP9 Profiling Script
Captures performance metrics, hardware info, and generates profiling reports.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import argparse


def get_hardware_profile() -> Dict[str, Any]:
    """Capture hardware and driver information."""
    profile = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": sys.platform,
    }

    # Try to get CUDA device info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            devices = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    devices.append({
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_total": parts[2]
                    })
            profile["cuda_devices"] = devices
    except Exception as e:
        profile["cuda_error"] = str(e)

    return profile


def run_benchmark(executable: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run benchmark and collect metrics."""
    try:
        result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=executable.parent.parent
        )

        benchmark_result = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        # Try to parse scoreboard JSON
        if result.stdout:
            try:
                # Look for JSON output
                for line in result.stdout.split('\n'):
                    if line.strip().startswith('{'):
                        scoreboard = json.loads(line)
                        benchmark_result["scoreboard"] = scoreboard
                        break
            except json.JSONDecodeError:
                pass

        return benchmark_result
    except subprocess.TimeoutExpired:
        return {"error": "benchmark timeout"}
    except Exception as e:
        return {"error": str(e)}


def generate_profiling_report(
    hardware: Dict[str, Any],
    benchmark: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate comprehensive profiling report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save hardware profile
    hw_file = output_dir / "hardware_profile.json"
    with open(hw_file, 'w') as f:
        json.dump(hardware, f, indent=2)
    print(f"Hardware profile saved to: {hw_file}")

    # Save benchmark results
    bench_file = output_dir / "benchmark_results.json"
    with open(bench_file, 'w') as f:
        json.dump(benchmark, f, indent=2)
    print(f"Benchmark results saved to: {bench_file}")

    # Generate summary report
    summary_file = output_dir / "profiling_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DVREN-HOTPATH HP9 Profiling Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Hardware Profile:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Timestamp: {hardware.get('timestamp', 'N/A')}\n")
        f.write(f"Platform: {hardware.get('platform', 'N/A')}\n")

        if "cuda_devices" in hardware:
            f.write("\nCUDA Devices:\n")
            for i, dev in enumerate(hardware["cuda_devices"]):
                f.write(f"  Device {i}: {dev['name']}\n")
                f.write(f"    Driver: {dev['driver_version']}\n")
                f.write(f"    Memory: {dev['memory_total']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Benchmark Results:\n")
        f.write("-" * 80 + "\n")

        if "scoreboard" in benchmark:
            scoreboard = benchmark["scoreboard"]
            summary = scoreboard.get("summary", {})
            f.write(f"Total Cases: {sum(summary.values())}\n")
            f.write(f"Passed: {summary.get('pass', 0)}\n")
            f.write(f"Failed: {summary.get('fail', 0)}\n")
            f.write(f"Skipped: {summary.get('skip', 0)}\n")

            if summary.get('fail', 0) > 0:
                f.write("\nFailed Cases:\n")
                for case in scoreboard.get("cases", []):
                    if case.get("status") == "fail":
                        f.write(f"  - {case['name']}: {case.get('message', 'N/A')}\n")
        else:
            f.write("No scoreboard data available\n")
            if "error" in benchmark:
                f.write(f"Error: {benchmark['error']}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="HP9 Profiling Script")
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build/Release/hp_runner.exe"),
        help="Path to test runner executable"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling_output"),
        help="Output directory for profiling data"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DVREN-HOTPATH HP9 Profiling")
    print("=" * 80)
    print()

    # Capture hardware profile
    print("Capturing hardware profile...")
    hardware = get_hardware_profile()
    print(f"Platform: {hardware['platform']}")
    if "cuda_devices" in hardware:
        print(f"CUDA Devices: {len(hardware['cuda_devices'])}")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    if not args.executable.exists():
        print(f"Error: Executable not found: {args.executable}")
        sys.exit(1)

    benchmark = run_benchmark(args.executable, {})

    if benchmark.get("returncode") == 0:
        print("✓ Benchmarks completed successfully")
    else:
        print("✗ Benchmarks failed or encountered errors")
    print()

    # Generate report
    print("Generating profiling report...")
    generate_profiling_report(hardware, benchmark, args.output)
    print()

    print("=" * 80)
    print("Profiling complete!")
    print("=" * 80)

    return 0 if benchmark.get("returncode") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

