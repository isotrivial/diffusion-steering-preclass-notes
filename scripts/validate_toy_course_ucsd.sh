#!/usr/bin/env bash
set -euo pipefail

VALIDATION_DIR="${TOY_COURSE_VALIDATION_DIR:-.validation/toy-course}"
VENV_DIR="${TOY_COURSE_VENV_DIR:-.venv-toy-course}"

mkdir -p "${VALIDATION_DIR}"
python3 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --disable-pip-version-check --quiet \
  -r notebooks/toy_data/requirements-course.txt

python - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the validation allocation")

summary = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "visible_gpu_count": torch.cuda.device_count(),
    "gpu": torch.cuda.get_device_name(0),
}
print(json.dumps(summary, indent=2))
PY

python -m pytest -q notebooks/toy_data/tests/test_toy_course.py \
  2>&1 | tee "${VALIDATION_DIR}/unit-tests.log"

python scripts/run_toy_course_notebooks.py \
  --output-dir "${VALIDATION_DIR}/executed" \
  2>&1 | tee "${VALIDATION_DIR}/notebooks.log"

python - "${VALIDATION_DIR}/executed/execution-report.json" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text())
records = report.get("notebooks", [])
if report.get("status") != "passed" or len(records) != 6:
    raise SystemExit(f"notebook validation failed: {report_path}")
if any(record.get("status") != "passed" for record in records):
    raise SystemExit(f"at least one notebook failed: {report_path}")

print("validated notebooks:")
for record in records:
    executed = json.loads(Path(record["executed"]).read_text())
    image_count = sum(
        "image/png" in output.get("data", {})
        for cell in executed.get("cells", [])
        for output in cell.get("outputs", [])
    )
    if image_count == 0:
        raise SystemExit(f"no inline plots were captured for {record['name']}")
    print(f"- {record['name']}: {record['seconds']:.1f}s, {image_count} inline plots")
PY

touch "${VALIDATION_DIR}/VALIDATION_PASSED"
