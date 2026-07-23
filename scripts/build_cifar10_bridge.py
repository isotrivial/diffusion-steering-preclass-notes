#!/usr/bin/env python3
"""Build the unconditional-EDM CIFAR-10 steering notebook."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "toy_data" / "optional"
NOTEBOOK = OUT / "cifar10_steering_with_unconditional_edm.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells = [
    markdown(
        r"""
# Image experiment: steering an unconditional CIFAR-10 EDM model

The 2D notes make flow, denoising, and steering visible, but a small MLP is not
evidence that the same idea matters in an image generator. Here we test a
class-minus-full PCA denoiser correction inside NVIDIA's **unconditional**
CIFAR-10 EDM checkpoint. The network never receives a class label.

Paired baseline, zero-strength, wrong-class, diversity, and frozen-evaluator
controls ask whether any class change is attributable to the correction rather
than sampling luck or a generic image perturbation. Re-execution requires the
external assets locked in `cifar10_bridge_manifest.json` and a CUDA GPU; the
published executed copy contains the figures and measured results.
"""
    ),
    code(
        r"""
from pathlib import Path
import json
import os
import sys

OPTIONAL_DIR = Path.cwd()
if not (OPTIONAL_DIR / "edm_cifar10_bridge.py").exists():
    OPTIONAL_DIR = Path("notebooks/toy_data/optional")
sys.path.insert(0, str(OPTIONAL_DIR.resolve()))

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from edm_cifar10_bridge import *

MANIFEST_PATH = OPTIONAL_DIR / "cifar10_bridge_manifest.json"
manifest = load_manifest(MANIFEST_PATH)
CALIBRATION_PATH = OPTIONAL_DIR / "cifar10_edm_calibration.json"
calibration_result = json.loads(CALIBRATION_PATH.read_text())
print("CUDA:", torch.cuda.is_available())
print("checkpoint:", manifest["checkpoint"]["path"])
print("checkpoint SHA-256:", manifest["checkpoint"]["sha256"])
print("unconditional:", manifest["generator"]["unconditional"])
"""
    ),
    markdown(
        r"""
## 1. Denoised estimates and deterministic sampling

At noise level $\sigma$, EDM's network returns a clean-image estimate
$D_\theta(x,\sigma)$. The deterministic probability-flow sampler follows the
direction

$$d(x,\sigma)=\frac{x-D_\theta(x,\sigma)}{\sigma}.$$

The notebook uses the official 18-step Heun setting with stochastic churn set
to zero. Once the initial noise is fixed, the full trajectory is fixed. Reusing
the same noise for every row therefore isolates the intervention rather than
sampling luck.

The generator checkpoint, source revision, sampler, PCA files, evaluator, and
seed ranges are fixed before evaluation by the manifest.
"""
    ),
    code(
        r"""
summary_rows = [
    ("EDM source commit", manifest["generator"]["source_commit"]),
    ("model", manifest["generator"]["architecture"]),
    ("sampler", f"{manifest['sampler']['method']}, {manifest['sampler']['steps']} steps / {manifest['sampler']['network_evaluations']} evaluations"),
    ("target", CIFAR10_CLASSES[manifest["steering"]["target_class"]]),
    ("wrong-class control", CIFAR10_CLASSES[manifest["steering"]["wrong_class"]]),
    ("correction strength", manifest["steering"]["strength"]),
    ("guided transitions", f"{manifest['steering']['start_step']} through {manifest['steering']['end_step']}"),
    ("evaluation samples", manifest["evaluation"]["sample_count"]),
    ("sampling seeds", f"{manifest['evaluation']['sampling_seed']} through {manifest['evaluation']['sampling_seed'] + manifest['evaluation']['sample_count'] - 1}"),
]
pd.DataFrame(summary_rows, columns=["locked item", "value"])
"""
    ),
    markdown(
        r"""
## 2. Why noise alignment can steer

Fit one Gaussian/PCA model to all CIFAR-10 training images and another to one
class. Each defines a Wiener denoiser: a coarse estimate of the clean image
that could have produced the current noisy state under that distribution. Their
difference asks how the estimate changes when we replace the broad data
distribution with the target-class distribution.

During the first eight Heun transitions, whose starting noise levels run from
$\sigma=80$ through about $5.32$, we replace the network's clean estimate by

$$D_{guided}=D_\theta+s\left(D_{class}-D_{full}\right).$$

The correction compares two fitted image distributions. There is no selected
image, centroid attraction, or point-attraction controller. The EDM network is
frozen. At high noise the scene is still malleable, so a coarse class-dependent
change can affect the later deterministic trajectory; after the window closes,
the original nonlinear EDM denoiser supplies the remaining detail. `s=0` still
executes the complete correction path and must reproduce the baseline endpoint.

### Calibration before the revised evaluation

The earlier setting made only a small change in the image grid. We therefore
swept strength and window length on development seeds `10000-10063`. A first
revision chosen right at the diversity boundary missed the locked evaluation
floor by `0.002`; we rejected it rather than lowering the threshold afterward.
The final rule requires a 70% calibration feature-covariance margin and no near
duplicates, then chooses the largest mean cat probability. Class-histogram
entropy is reported but is not constrained: successful class steering should
change the proportion of predicted labels. The final evaluation below uses a
second new range, `30000-30255`, not used by either earlier evaluation.
"""
    ),
    code(
        r"""
selected_strength = calibration_result["selected_strength"]
selected_end_step = calibration_result["selected_end_step"]
selected_calibration = next(
    row for row in calibration_result["candidates"]
    if row["strength"] == selected_strength and row["end_step"] == selected_end_step
)
pd.DataFrame(
    [
        {
            "setting": "unconditional",
            "cat rate": calibration_result["baseline"]["target_rate"],
            "mean cat probability": calibration_result["baseline"]["target_probability"],
            "feature trace ratio": 1.0,
            "near duplicates": calibration_result["baseline"]["near_duplicate_rate"],
        },
        {
            "setting": f"strength {selected_strength:g}, through step {selected_end_step}",
            "cat rate": selected_calibration["target_rate"],
            "mean cat probability": selected_calibration["target_probability"],
            "feature trace ratio": selected_calibration["feature_trace_ratio"],
            "near duplicates": selected_calibration["near_duplicate_rate"],
        },
    ]
).set_index("setting").round(3)
"""
    ),
    code(
        r"""
result = run_bridge(manifest, verify_hashes=True, verify_evaluator=True)
result_path = os.environ.get("CIFAR10_EDM_RESULT_PATH")
if result_path:
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    write_json_summary(result, result_path)
print("device:", result["device"])
print("measured evaluator test accuracy:", f"{result['evaluator_test_accuracy']:.2%}")
print("total runtime:", f"{result['total_runtime_seconds']:.1f}s")
print("peak GPU memory:", f"{result['peak_gpu_memory_gb']:.2f} GiB")
if result_path:
    print("result summary:", result_path)
"""
    ),
    markdown(
        r"""
## 3. Paired images

Each column below begins from exactly the same Gaussian noise. The twelve seeds
are fixed, evenly spaced across all 256 evaluation seeds, and are **not** chosen
by their classifier outcomes. Compare down a column, not across unrelated
samples. Labels and $P(\mathrm{cat})$ are frozen-evaluator diagnostics, not
ground truth. The exactly redundant zero-strength images are omitted from this
large panel and checked numerically below. The ship row asks whether a generic
PCA perturbation also increases the cat score.
"""
    ),
    code(
        r"""
plot_method_grid(result, count=12)
plt.show()
"""
    ),
    markdown(
        r"""
## 4. Probe the denoiser along time

These panels show the network's current clean-image estimate for one trajectory
that changes from a non-cat baseline prediction to a cat prediction under the
target correction. It is selected transparently as the first such seed only to
make the mechanism visible; the fixed grid and aggregate metrics, not this
example, evaluate the method. Early estimates are uncertain and coarse; later
estimates become recognizable. After the declared window, the same
unconditional network continues both trajectories.
"""
    ),
    code(
        r"""
trajectory_index = first_target_conversion_index(result)
print("illustrative trajectory seed:", result["seeds"][trajectory_index])
plot_denoised_trajectory(result, sample_index=trajectory_index)
plt.show()
"""
    ),
    markdown(
        r"""
## 5. Evaluate control and side effects

A frozen ResNet-56 reports predicted labels, target probabilities, and
penultimate-layer features. The metrics deliberately include more than target
rate:

- distance from generated features to real target-class features;
- generated feature-covariance trace relative to baseline;
- class-histogram entropy and near-duplicate rate;
- paired target-rate uncertainty, runtime, and memory.

The evaluator is diagnostic evidence, not a proof that every steered image is a
valid member of the requested class.
"""
    ),
    code(
        r"""
metric_rows = []
for method in ["baseline", "zero", "target", "wrong"]:
    values = result["metrics"][method]
    metric_rows.append({
        "method": method,
        "target rate": values["target_rate"],
        "target probability": values["target_probability"],
        "target feature distance": values["target_feature_mean_distance"],
        "feature trace": values["feature_covariance_trace"],
        "class entropy": values["class_histogram_entropy"],
        "near duplicates": values["near_duplicate_rate"],
        "seconds": values["runtime_seconds"],
    })
metrics_table = pd.DataFrame(metric_rows).set_index("method")
metrics_table.round(4)
"""
    ),
    code(
        r"""
plot_metric_comparison(result)
plt.show()
"""
    ),
    markdown(
        r"""
## 6. Locked evaluation checks

After calibration, the revised thresholds and new evaluation seeds were locked.
The protocol passes only if every check below passes. In particular, the cat
rate must reach 40%, improve by at least 25 percentage points, have a positive
paired bootstrap lower bound, and preserve at least 60% of baseline feature
covariance. The wrong-class correction must not explain the cat gain. Passing
these checks supports a measurable but still partial steering result; it does
not mean every image is visibly a cat.
"""
    ),
    code(
        r"""
checks_table = pd.DataFrame(
    [{"check": name.replace("_", " "), "passed": passed} for name, passed in result["checks"].items()]
).set_index("check")
display(checks_table)
print("target-rate gain:", f"{result['target_rate_gain']:+.3f}")
print("paired gain 95% interval:", [round(value, 3) for value in result["paired_target_rate_gain_95ci"]])
print("wrong-class target-rate gain:", f"{result['wrong_class_target_rate_gain']:+.3f}")
print("zero-strength endpoint max |difference|:", f"{result['zero_endpoint_max_abs']:.3e}")
print("deterministic repeat max |difference|:", f"{result['deterministic_repeat_max_abs']:.3e}")
print("CIFAR-10 EDM protocol verdict:", result["release_status"])
interpretation = (
    "partial class steering, not reliable class-conditional generation"
    if result["release_status"] == "PASS"
    else "negative result under the locked protocol"
)
print("scientific interpretation:", interpretation)
"""
    ),
    markdown(
        r"""
## What this notebook establishes

If the checks pass, the narrow conclusion is that an early, distribution-level
PCA denoiser correction provides measurable but incomplete class steering for
this fixed unconditional EDM model under paired controls. Inspect the images as
well as the classifier summaries: some paired trajectories convert clearly,
some remain ambiguous, and some do not become cats. It does **not** establish
that PCA guidance is optimal, that the classifier captures human judgment, or
that the same settings transfer to other datasets.

This bridge demonstrates noise alignment. Activation steering of a specific
EDM hidden block would require a separate hook definition, held-out activation
probe, matched shuffled direction, and its own evaluation.

**Sources**

- Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364).
- Li, Dai, and Qu, [*Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure*](https://arxiv.org/abs/2410.24060).
- [Official NVIDIA EDM implementation](https://github.com/NVlabs/edm).
- [CIFAR pretrained evaluator source](https://github.com/chenyaofo/pytorch-cifar-models).
"""
    ),
]


notebook = {
    "cells": [dict(cell, id=f"cell-{index:03d}") for index, cell in enumerate(cells)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(NOTEBOOK)
