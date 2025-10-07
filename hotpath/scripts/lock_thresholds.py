#!/usr/bin/env python3
"""
HP9 Threshold Locking Script
Locks performance and numerical thresholds after validation.
"""

import yaml
import json
import sys
from pathlib import Path
from datetime import datetime


def load_thresholds(threshold_file: Path) -> dict:
    """Load current thresholds from YAML."""
    try:
        with open(threshold_file, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading thresholds: {e}")
        return {}


def lock_thresholds(threshold_file: Path) -> bool:
    """Lock thresholds with version and timestamp."""
    thresholds = load_thresholds(threshold_file)

    if not thresholds:
        print("Warning: No thresholds to lock")
        return False

    # Add metadata
    if 'metadata' not in thresholds:
        thresholds['metadata'] = {}

    thresholds['metadata']['locked'] = True
    thresholds['metadata']['lock_timestamp'] = datetime.now().isoformat()
    thresholds['metadata']['lock_version'] = 1

    # Write back
    try:
        with open(threshold_file, 'w') as f:
            yaml.dump(thresholds, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Thresholds locked: {threshold_file}")
        return True
    except Exception as e:
        print(f"Error writing thresholds: {e}")
        return False


def main():
    threshold_file = Path("tests/thresholds.yaml")

    if not threshold_file.exists():
        print(f"Error: Threshold file not found: {threshold_file}")
        sys.exit(1)

    print("=" * 80)
    print("HP9 Threshold Locking")
    print("=" * 80)
    print()

    success = lock_thresholds(threshold_file)

    print()
    print("=" * 80)
    if success:
        print("✓ Thresholds locked successfully")
        return 0
    else:
        print("✗ Failed to lock thresholds")
        return 1


if __name__ == "__main__":
    sys.exit(main())

