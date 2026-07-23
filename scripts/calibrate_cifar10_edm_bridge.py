#!/usr/bin/env python3
"""Run the locked CIFAR-10 EDM strength-and-window calibration split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_DIR = ROOT / "notebooks" / "toy_data" / "optional"
sys.path.insert(0, str(OPTIONAL_DIR))

from edm_cifar10_bridge import calibrate_strengths, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=OPTIONAL_DIR / "cifar10_bridge_manifest.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".validation" / "cifar10-edm-calibration.json"
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    result = calibrate_strengths(
        manifest, manifest["calibration"]["candidate_strengths"], verify_hashes=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["selected_strength"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
