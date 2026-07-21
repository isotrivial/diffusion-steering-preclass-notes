#!/usr/bin/env python3
"""Execute the maintained toy course in order and write a machine-readable report."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = ROOT / "notebooks" / "toy_data"
DEFAULT_OUTPUT_DIR = COURSE_DIR / "executed"
NOTEBOOKS = [
    "00_diffusion_and_flow_matching_foundations.ipynb",
    "01_train_unconditional_flow_matching.ipynb",
    "02_denoisers_and_deterministic_samplers.ipynb",
    "03_gaussian_denoisers_and_noise_alignment.ipynb",
    "04_training_free_gradient_guidance.ipynb",
    "05_activation_steering_and_capstone.ipynb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell timeout in seconds")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise SystemExit(
            "Install notebooks/toy_data/requirements-course.txt before running the course"
        ) from exc

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "repository": str(ROOT),
        "course_directory": str(COURSE_DIR),
        "started_at_unix": time.time(),
        "notebooks": [],
    }

    exit_code = 0
    for name in NOTEBOOKS:
        source_path = COURSE_DIR / name
        executed_path = output_dir / name
        started = time.perf_counter()
        record = {"name": name, "source": str(source_path), "executed": str(executed_path)}
        print(f"[run] {name}", flush=True)

        try:
            with source_path.open("r", encoding="utf-8") as handle:
                notebook = nbformat.read(handle, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(COURSE_DIR)}},
                allow_errors=False,
            )
            client.execute()
            with executed_path.open("w", encoding="utf-8") as handle:
                nbformat.write(notebook, handle)
            record["status"] = "passed"
        except Exception as exc:  # The traceback is part of the validation artifact.
            exit_code = 1
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            try:
                with executed_path.open("w", encoding="utf-8") as handle:
                    nbformat.write(notebook, handle)
            except Exception:
                pass
        record["seconds"] = time.perf_counter() - started
        report["notebooks"].append(record)
        print(f"[{record['status']}] {name} ({record['seconds']:.1f}s)", flush=True)

        report_path = output_dir / "execution-report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if exit_code and not args.keep_going:
            break

    report["finished_at_unix"] = time.time()
    report["status"] = "passed" if exit_code == 0 else "failed"
    report_path = output_dir / "execution-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

