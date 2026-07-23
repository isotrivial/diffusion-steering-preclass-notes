#!/usr/bin/env bash
set -euo pipefail

VALIDATION_DIR="${ALL_NOTES_VALIDATION_DIR:-.validation/all-notes}"
mkdir -p "${VALIDATION_DIR}"
rm -f "${VALIDATION_DIR}/VALIDATION_PASSED"

export TOY_NOTES_VALIDATION_DIR="${VALIDATION_DIR}/toy"
export CIFAR10_EDM_VALIDATION_DIR="${VALIDATION_DIR}/cifar10"

bash scripts/validate_toy_notes_ucsd.sh
bash scripts/validate_cifar10_edm_bridge_ucsd.sh

touch "${VALIDATION_DIR}/VALIDATION_PASSED"
