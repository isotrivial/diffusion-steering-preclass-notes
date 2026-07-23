#!/usr/bin/env python3
"""Execute the CIFAR-10 EDM image experiment and preserve its output."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from run_toy_notes import inline_matplotlib


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_DIR = ROOT / "notebooks" / "toy_data" / "optional"
SOURCE = OPTIONAL_DIR / "cifar10_steering_with_unconditional_edm.ipynb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=OPTIONAL_DIR / "executed", help="Execution artifact directory"
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--kernel-name", default=os.environ.get("CIFAR10_EDM_KERNEL_NAME", "python3")
    )
    args = parser.parse_args()

    import nbformat
    from nbclient import NotebookClient

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executed = output_dir / SOURCE.name
    record = {
        "source": str(SOURCE),
        "executed": str(executed),
        "started_at_unix": time.time(),
    }
    started = time.perf_counter()
    notebook = nbformat.read(SOURCE, as_version=4)
    try:
        with inline_matplotlib(notebook):
            NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name=args.kernel_name,
                resources={"metadata": {"path": str(OPTIONAL_DIR)}},
                allow_errors=False,
            ).execute()
        record["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        record.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        exit_code = 1
    record["seconds"] = time.perf_counter() - started
    record["finished_at_unix"] = time.time()
    nbformat.write(notebook, executed)
    (output_dir / "execution-report.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
