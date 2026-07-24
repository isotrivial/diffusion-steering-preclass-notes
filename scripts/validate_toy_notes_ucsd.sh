#!/usr/bin/env bash
set -euo pipefail

VALIDATION_DIR="${TOY_NOTES_VALIDATION_DIR:-.validation/toy-notes}"
VENV_DIR="${TOY_NOTES_VENV_DIR:-.venv-toy-notes}"
BASE_PYTHON="${TOY_NOTES_BASE_PYTHON:-python3}"

mkdir -p "${VALIDATION_DIR}"
rm -f "${VALIDATION_DIR}/VALIDATION_PASSED"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

export TOY_NOTES_TRAIN_STEPS=2500
export TOY_NOTES_EIGHT_STEPS=3000
export TOY_NOTES_CIRCLE_STEPS=3000
export MNIST_DATA_DIR="${MNIST_DATA_DIR:-/data/qingsong/sos/data/MNIST/raw}"

python -m pip install --disable-pip-version-check --quiet \
  -r notebooks/toy_data/requirements-notes.txt
python -m ipykernel install --prefix "${VENV_DIR}" --name toy-notes-dct \
  --display-name "Toy notes validation" >/dev/null
export JUPYTER_PATH="${VENV_DIR}/share/jupyter${JUPYTER_PATH:+:${JUPYTER_PATH}}"

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

python -m pytest -q \
  notebooks/toy_data/tests/test_toy_notes.py \
  notebooks/toy_data/tests/test_notebook_pedagogy.py \
  2>&1 | tee "${VALIDATION_DIR}/unit-tests.log"

python scripts/run_toy_notes.py \
  --kernel-name toy-notes-dct \
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
minimum_images = {
    "00_diffusion_and_flow_matching_foundations.ipynb": 4,
    "01_train_unconditional_flow_matching.ipynb": 9,
    "02_denoisers_and_deterministic_samplers.ipynb": 4,
    "03_gaussian_denoisers_and_noise_alignment.ipynb": 5,
    "05_activation_steering_and_method_comparison.ipynb": 4,
}
for record in records:
    executed = json.loads(Path(record["executed"]).read_text())
    image_count = sum(
        "image/png" in output.get("data", {})
        for cell in executed.get("cells", [])
        for output in cell.get("outputs", [])
    )
    minimum = minimum_images.get(record["name"], 1)
    if image_count < minimum:
        raise SystemExit(
            f"expected at least {minimum} inline plots for {record['name']}, found {image_count}"
        )
    print(f"- {record['name']}: {record['seconds']:.1f}s, {image_count} inline plots")
PY

touch "${VALIDATION_DIR}/VALIDATION_PASSED"
