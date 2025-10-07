#!/usr/bin/env python3
"""
HP9 Artifact Archival Script
Archives test results, profiling data, and performance metrics.
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
import hashlib


def compute_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return "error"


def archive_artifacts(source_dir: Path, archive_dir: Path) -> dict:
    """Archive all artifacts with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"artifacts_{timestamp}"
    archive_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": timestamp,
        "artifacts": []
    }

    # Archive profiling output
    profiling_dir = source_dir / "profiling"
    if profiling_dir.exists():
        dest = archive_path / "profiling"
        shutil.copytree(profiling_dir, dest, dirs_exist_ok=True)

        for file in dest.rglob("*"):
            if file.is_file():
                manifest["artifacts"].append({
                    "path": str(file.relative_to(archive_path)),
                    "size": file.stat().st_size,
                    "hash": compute_hash(file)
                })

    # Save manifest
    manifest_file = archive_path / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Artifacts archived to: {archive_path}")
    print(f"  Total artifacts: {len(manifest['artifacts'])}")

    return manifest


def create_archive_index(archive_dir: Path):
    """Create index of all archived runs."""
    index = {
        "archives": [],
        "latest": None
    }

    for archive_path in sorted(archive_dir.glob("artifacts_*")):
        manifest_file = archive_path / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)

            index["archives"].append({
                "path": archive_path.name,
                "timestamp": manifest.get("timestamp"),
                "artifact_count": len(manifest.get("artifacts", []))
            })

    if index["archives"]:
        index["latest"] = index["archives"][-1]["path"]

    index_file = archive_dir / "index.json"
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"✓ Archive index updated: {index_file}")


def main():
    source_dir = Path("artifacts")
    archive_dir = Path("GOLDENS/archives")

    print("=" * 80)
    print("HP9 Artifact Archival")
    print("=" * 80)
    print()

    if not source_dir.exists():
        print(f"Warning: Source directory not found: {source_dir}")
        print("Creating directory...")
        source_dir.mkdir(parents=True, exist_ok=True)

    archive_dir.mkdir(parents=True, exist_ok=True)

    manifest = archive_artifacts(source_dir, archive_dir)
    create_archive_index(archive_dir)

    print()
    print("=" * 80)
    print("✓ Archival complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

