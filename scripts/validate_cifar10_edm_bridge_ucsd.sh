#!/usr/bin/env bash
set -euo pipefail

VALIDATION_DIR="${CIFAR10_EDM_VALIDATION_DIR:-.validation/cifar10-edm}"
VENV_DIR="${CIFAR10_EDM_VENV_DIR:-.venv-edm-bridge}"
BASE_PYTHON="${CIFAR10_EDM_BASE_PYTHON:-/data/qingsong/miniforge3/envs/dctdiff/bin/python}"
EXTERNAL_DIR="${CIFAR10_EDM_EXTERNAL_DIR:-.external}"
EDM_DIR="${EXTERNAL_DIR}/edm"
EVALUATOR_DIR="${EXTERNAL_DIR}/pytorch-cifar-models"
EDM_COMMIT="008a4e5316c8e3bfe61a62f874bddba254295afb"
EVALUATOR_COMMIT="786c16252c0fc58ee9adac063f8337cc4a7a497a"

mkdir -p "${VALIDATION_DIR}" "${EXTERNAL_DIR}"
rm -f "${VALIDATION_DIR}/VALIDATION_PASSED"

if [[ ! -d "${EDM_DIR}/.git" ]]; then
  git clone --quiet https://github.com/NVlabs/edm.git "${EDM_DIR}"
  git -C "${EDM_DIR}" checkout --quiet "${EDM_COMMIT}"
fi
if [[ "$(git -C "${EDM_DIR}" rev-parse HEAD)" != "${EDM_COMMIT}" ]]; then
  echo "EDM source checkout is not pinned to ${EDM_COMMIT}" >&2
  exit 1
fi

if [[ ! -d "${EVALUATOR_DIR}/.git" ]]; then
  git clone --quiet https://github.com/chenyaofo/pytorch-cifar-models.git "${EVALUATOR_DIR}"
  git -C "${EVALUATOR_DIR}" checkout --quiet "${EVALUATOR_COMMIT}"
fi
if [[ "$(git -C "${EVALUATOR_DIR}" rev-parse HEAD)" != "${EVALUATOR_COMMIT}" ]]; then
  echo "evaluator source checkout is not pinned to ${EVALUATOR_COMMIT}" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --disable-pip-version-check --quiet \
  -r notebooks/toy_data/requirements-notes.txt
python -m ipykernel install --prefix "${VENV_DIR}" --name cifar10-edm-bridge \
  --display-name "CIFAR-10 EDM bridge" >/dev/null
export JUPYTER_PATH="${VENV_DIR}/share/jupyter${JUPYTER_PATH:+:${JUPYTER_PATH}}"
export CIFAR10_EDM_RESULT_PATH="$(pwd)/${VALIDATION_DIR}/results.json"

python scripts/build_cifar10_bridge.py
python -m pytest -q notebooks/toy_data/tests/test_edm_cifar10_bridge.py \
  2>&1 | tee "${VALIDATION_DIR}/unit-tests.log"
python scripts/run_cifar10_bridge.py --kernel-name cifar10-edm-bridge \
  --output-dir "${VALIDATION_DIR}/executed" \
  2>&1 | tee "${VALIDATION_DIR}/notebook.log"

python - "${VALIDATION_DIR}/executed/cifar10_steering_with_unconditional_edm.ipynb" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
notebook = json.loads(path.read_text())
image_count = sum(
    "image/png" in output.get("data", {})
    for cell in notebook.get("cells", [])
    for output in cell.get("outputs", [])
)
text = []
for cell in notebook.get("cells", []):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            value = output.get("text", "")
            text.append("".join(value) if isinstance(value, list) else value)
        value = output.get("data", {}).get("text/plain", "")
        text.append("".join(value) if isinstance(value, list) else value)
joined = "\n".join(text)
if image_count < 3:
    raise SystemExit(f"expected at least three embedded figures, found {image_count}")
if "CIFAR-10 EDM protocol verdict: PASS" not in joined:
    raise SystemExit("the CIFAR-10 experiment did not pass its locked protocol checks")
print(f"validated CIFAR-10 experiment with {image_count} embedded figures")
PY

python - "${VALIDATION_DIR}/results.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if result["zero_endpoint_max_abs"] >= 1e-6:
    raise SystemExit("zero-strength endpoint tolerance failed")
if result["deterministic_repeat_max_abs"] >= 1e-6:
    raise SystemExit("deterministic repeat tolerance failed")
if result["release_status"] != "PASS" or not all(result["checks"].values()):
    raise SystemExit("result summary did not pass every protocol check")
print("validated JSON result summary")
PY

touch "${VALIDATION_DIR}/VALIDATION_PASSED"
