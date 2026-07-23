#!/usr/bin/env python3
"""Execute the maintained pre-class notebooks and write a validation report."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = ROOT / "notebooks" / "toy_data"
DEFAULT_OUTPUT_DIR = NOTES_DIR / "executed"
NOTEBOOKS = [
    "00_diffusion_and_flow_matching_foundations.ipynb",
    "01_train_unconditional_flow_matching.ipynb",
    "02_denoisers_and_deterministic_samplers.ipynb",
    "03_gaussian_denoisers_and_noise_alignment.ipynb",
    "04_training_free_gradient_guidance.ipynb",
    "05_activation_steering_and_method_comparison.ipynb",
]

INLINE_MATPLOTLIB_BOOTSTRAP = """
from IPython import get_ipython as _get_ipython
_ipython = _get_ipython()
if _ipython is not None:
    _ipython.run_line_magic("matplotlib", "inline")
del _ipython, _get_ipython
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell timeout in seconds")
    parser.add_argument(
        "--kernel-name", default=os.environ.get("TOY_NOTES_KERNEL_NAME", "python3")
    )
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


@contextmanager
def inline_matplotlib(notebook):
    """Enable inline plots without adding execution plumbing to the saved notebook."""
    setup_cell = next(
        (cell for cell in notebook.cells if cell.get("cell_type") == "code"),
        None,
    )
    if setup_cell is None:
        yield
        return

    original_source = setup_cell.source
    setup_cell.source = f"{INLINE_MATPLOTLIB_BOOTSTRAP}\n\n{original_source}"
    try:
        yield
    finally:
        setup_cell.source = original_source


def main() -> int:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise SystemExit(
            "Install notebooks/toy_data/requirements-notes.txt before running the notebooks"
        ) from exc

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "repository": str(ROOT),
        "notes_directory": str(NOTES_DIR),
        "started_at_unix": time.time(),
        "notebooks": [],
    }

    exit_code = 0
    for name in NOTEBOOKS:
        source_path = NOTES_DIR / name
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
                kernel_name=args.kernel_name,
                resources={"metadata": {"path": str(NOTES_DIR)}},
                allow_errors=False,
            )
            with inline_matplotlib(notebook):
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
