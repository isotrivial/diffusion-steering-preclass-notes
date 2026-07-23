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
# 06 - Noise alignment in an unconditional CIFAR-10 EDM

The toy notebooks made each steering mechanism visible in two dimensions. This
final note asks whether one of them, high-noise Gaussian/PCA noise alignment,
still has a measurable effect inside NVIDIA's pretrained unconditional
CIFAR-10 EDM model. We add the target-class minus full-data PCA denoiser
correction only during the earliest sampling steps and compare each guided
image with the baseline from the same initial noise, alongside zero-strength
and wrong-class controls. The goal is deliberately modest: look for consistent
partial steering toward cats without mistaking classifier movement, a generic
perturbation, or loss of diversity for reliable class-conditional generation.
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

All rows use the same pretrained checkpoint, 18-step deterministic Heun
schedule, initial-noise seeds, and frozen evaluator. Reusing this setup makes
the comparison paired: the baseline, zero-strength, target-class, and
wrong-class rows differ only in the correction applied during the early
high-noise window.
"""
    ),
    code(
        r"""
summary_rows = [
    ("model", manifest["generator"]["architecture"]),
    ("sampler", f"{manifest['sampler']['method']}, {manifest['sampler']['steps']} steps / {manifest['sampler']['network_evaluations']} evaluations"),
    ("target", CIFAR10_CLASSES[manifest["steering"]["target_class"]]),
    ("wrong-class control", CIFAR10_CLASSES[manifest["steering"]["wrong_class"]]),
    ("correction strength", manifest["steering"]["strength"]),
    ("guided transitions", f"{manifest['steering']['start_step']} through {manifest['steering']['end_step']}"),
    ("evaluation samples", manifest["evaluation"]["sample_count"]),
]
pd.DataFrame(summary_rows, columns=["setting", "value"])
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

### Choosing the steering setting before the final comparison

The correction strength and end step were chosen on development seeds that do
not appear in the final 256-image evaluation. Candidate settings had to retain
at least 70% of the baseline feature-covariance trace and avoid near-duplicate
outputs; among the settings that met those conditions, the selected one had the
highest mean cat probability. This separation lets the final grid and metrics
function as new evidence rather than as examples used to tune the method.
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
"""
    ),
    markdown(
        r"""
## 3. Read the paired image grid

Each column begins from the same Gaussian noise, so compare down a column rather
than across unrelated samples. The twelve displayed seeds are evenly spaced
across the 256 final evaluation seeds and were not chosen because their
classifier outcomes looked favorable. The zero-strength row is omitted from
the large panel because it reproduces the baseline. Evaluator labels and
$P(\mathrm{cat})$ are annotations for comparing rows, not ground-truth
descriptions of the images.

Across the columns, the target correction produces a mixed but structured
effect: some baseline images become clearly more cat-like, some shift only in
coarse shape or texture, and others remain ambiguous or non-cat. The ship
correction is a specificity control. It asks whether any class-based PCA
perturbation would produce the same increase in cat scores; in the reported
result, it does not reproduce the target-class pattern. The grid therefore
supports partial, class-specific movement, not uniform conversion.
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
## 4. How an early correction can affect a late image

The trajectory figure follows the first evaluation seed whose baseline endpoint
is not predicted as a cat but whose target-guided endpoint is. It is
intentionally illustrative rather than representative: one converted seed can
show where two paths separate, but only the fixed image grid and aggregate
metrics can show how often the effect occurs.

At the largest noise levels, both clean-image estimates are uncertain and
dominated by broad structure. While noise alignment is active, the
target-guided and baseline estimates begin to diverge. After the early window
closes, both trajectories again use the same frozen unconditional EDM denoiser,
but they now arrive at later steps from different states, so the difference can
persist and sharpen. The figure shows how a brief early correction can
influence a final image without implying that every seed follows the same
pattern.
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
## 5. Read target preference, target fit, and diversity together

The evaluator provides three complementary kinds of evidence. Predicted cat
rate and mean cat probability measure target preference. Distance to real cat
features asks whether the generated batch moves toward the target distribution
rather than merely crossing one classifier boundary. Feature-covariance trace,
class-histogram entropy, and near-duplicate rate describe how much variation
remains, while runtime and memory show the computational cost of the correction.

On the 256 final seeds, the frozen evaluator's predicted cat rate rises from
6.6% for the baseline to 41.8% with target guidance, a gain of 35.2 percentage
points. The guided batch retains 70.9% of the baseline feature variance. The
zero-strength run reproduces the baseline, showing that merely executing the
correction path does not alter the endpoint, and the wrong-class correction
does not account for the cat increase.

These measurements support a real, class-specific shift, but they do not imply
reliable generation. A 41.8% cat rate still means that most guided outputs
receive another predicted label, and classifier features cannot settle whether
every changed image looks convincing to a person. The metric panel should
therefore be read alongside the paired images, not as a substitute for them.
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
## 6. How strong is the evidence?

Before the final seeds were examined, we defined a meaningful shift as at least
a 40% predicted cat rate, at least a 25-percentage-point improvement over the
paired baseline, a positive lower bound from the paired bootstrap, and
retention of at least 60% of baseline feature covariance. We also required the
wrong-class correction not to reproduce the target gain. These reference
levels make the intended trade-off explicit: the method must change class
preference by more than a small fluctuation without collapsing the batch.

The reported target-guided result clears the rate, gain, and retained-variation
reference levels, while the paired uncertainty and wrong-class comparison
support the same interpretation. These controls reduce several simple
explanations, including sampling luck, a no-op implementation, and a generic
PCA perturbation, but they do not turn the result into a guarantee. The
evidence concerns one checkpoint, one target class, one correction schedule,
and one frozen evaluator.
"""
    ),
    code(
        r"""
evidence_summary = pd.DataFrame({
    "measurement": [
        "target-rate gain",
        "paired gain 95% interval",
        "target/baseline feature-trace ratio",
        "wrong-class target-rate gain",
        "zero-strength endpoint max difference",
        "deterministic repeat max difference",
    ],
    "value": [
        f"{result['target_rate_gain']:+.3f}",
        str([round(value, 3) for value in result["paired_target_rate_gain_95ci"]]),
        f"{result['metrics']['target']['feature_covariance_trace'] / result['metrics']['baseline']['feature_covariance_trace']:.3f}",
        f"{result['wrong_class_target_rate_gain']:+.3f}",
        f"{result['zero_endpoint_max_abs']:.3e}",
        f"{result['deterministic_repeat_max_abs']:.3e}",
    ],
})
evidence_summary.set_index("measurement")
"""
    ),
    markdown(
        r"""
## 7. What the image example adds

Taken together, the images and metrics support a narrow conclusion: an early
class-minus-full PCA denoiser correction moves this frozen unconditional EDM
model toward cats, raising the frozen evaluator's predicted cat rate from 6.6%
to 41.8% while retaining 70.9% of baseline feature variance. That is measurable
partial steering, not reliable class-conditional generation. Some paired
trajectories convert clearly, some remain ambiguous, and most guided outputs
are still not classified as cats.

The result does not show that PCA guidance is optimal, that evaluator labels
are equivalent to human judgment, or that the same strength and window
transfer to other classes, datasets, or generators. It is the image-scale
continuation of Notebook `03`'s noise-alignment mechanism. Activation steering
in an EDM hidden block would be a different experiment, requiring a specified
intervention layer, a direction learned and tested on separate activations,
matched shuffled controls, and its own image-level evidence.

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
