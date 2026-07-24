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
# 06 - A class-minus-full correction in an unconditional CIFAR-10 EDM

The 2D notes let us draw every state and steering vector. This final note asks
whether the same distribution-level idea can produce a visible change in a
pretrained image generator. We use NVIDIA's unconditional CIFAR-10 EDM and
compare images generated from exactly the same initial noise.

The intervention remains narrow: during the first few, very noisy sampling
steps, add the difference between a target-class PCA denoiser and a full-data
PCA denoiser to EDM's clean-image estimate. The paper calls this kind of
class-minus-full correction **noise alignment**. Here we lead with the literal
operation so it cannot be confused with reusing the same random seed.

**EDM time convention:** sampling begins at large noise $\sigma$ and proceeds
toward $\sigma=0$. This is the reverse numerical direction from the toy notes'
label $t=0\rightarrow1$, although both samplers move from noise to data.
"""
    ),
    code(
        r"""
from pathlib import Path
import os
import sys

OPTIONAL_DIR = Path.cwd()
if not (OPTIONAL_DIR / "edm_cifar10_bridge.py").exists():
    OPTIONAL_DIR = Path("notebooks/toy_data/optional")
sys.path.insert(0, str(OPTIONAL_DIR.resolve()))

import matplotlib.pyplot as plt
import pandas as pd

from edm_cifar10_bridge import *

MANIFEST_PATH = OPTIONAL_DIR / "cifar10_bridge_manifest.json"
manifest = load_manifest(MANIFEST_PATH)
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(2, 1, figsize=(10, 3.2))
for ax in axes:
    ax.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#444444"})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

axes[0].text(0.1, 0.72, "t=0: noise", ha="center")
axes[0].text(0.9, 0.72, "t=1: data", ha="center")
axes[0].text(0.5, 0.18, "toy I-CFM: t increases", ha="center", color="#0072B2")
axes[1].text(0.1, 0.72, "large sigma: noise", ha="center")
axes[1].text(0.9, 0.72, "sigma near 0: data", ha="center")
axes[1].text(0.5, 0.18, "EDM sampling: sigma decreases", ha="center", color="#D55E00")
fig.suptitle("Two coordinates, the same noise-to-data direction")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
## 1. Where can the correction enter the sampler?

At noise level $\sigma$, EDM predicts a clean-image estimate
$D_\theta(x,\sigma)$. Its deterministic probability-flow sampler integrates

$$\frac{dx}{d\sigma}=\frac{x-D_\theta(x,\sigma)}{\sigma}$$

while $\sigma$ decreases. Each numerical step therefore has negative
$\Delta\sigma$, so the corresponding state update moves toward the denoised
estimate rather than away from it.

Changing the clean estimate changes the next deterministic update. We use the
official 18-step Heun schedule with stochastic churn disabled, so one initial
noise tensor determines one complete trajectory.

The only intervention written in this notebook is

$$D_{guided}=D_\theta+s\left(D_{class}-D_{full}\right).$$

`D_class` and `D_full` are posterior means from two fitted PCA distributions.
Neither is a selected image or destination point. The EDM checkpoint remains
frozen.
"""
    ),
    code(
        r"""
def combine_denoisers(network_estimate, class_estimate, full_estimate, strength):
    return network_estimate + strength * (class_estimate - full_estimate)
"""
    ),
    markdown(
        r"""
## 2. A same-seed image experiment

Every method below uses the same checkpoint, solver, schedule, evaluator, and
256 initial noise tensors. Four rows have distinct roles:

- **baseline:** the original unconditional EDM sampler;
- **zero:** the correction code runs with `s=0` and should match baseline;
- **target:** use the cat-minus-full PCA denoiser difference;
- **wrong class:** use ship-minus-full as a class-specificity comparison.

The correction is active only during the first eight Heun transitions, from
$\sigma=80$ to approximately $\sigma=5.32$. The original EDM denoiser handles
all later detail.

**Before you run:** Which pair should be numerically identical? If both cat and
ship corrections raised the cat rate by the same amount, what simpler
explanation would remain?

The selected strength, calibration seeds, asset hashes, and evaluator checks
are recorded in `CIFAR10_BRIDGE.md`; they are intentionally kept out of this
reader-facing argument. The setting was selected using disjoint development
seeds; the paired images and aggregate measurements below use 256 different
final-evaluation seeds.
"""
    ),
    code(
        r"""
result = run_bridge(
    manifest,
    verify_hashes=True,
    verify_evaluator=True,
    denoiser_combiner=combine_denoisers,
)
result_path = os.environ.get("CIFAR10_EDM_RESULT_PATH")
if result_path:
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    write_json_summary(result, result_path)
"""
    ),
    markdown(
        r"""
## 3. Compare down each column

Each column starts from one shared noise seed. Compare the baseline, cat, and
ship rows vertically; comparisons across columns mix different initial noise.
The twelve columns are evenly spaced through the evaluation batch rather than
chosen for favorable outcomes. The zero row is omitted from the image panel
because it reproduces the baseline.

**Before you view the grid:** Expect a steering method to succeed unevenly.
Which visual evidence would distinguish partial steering from reliable
class-conditional generation?
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
Some columns become more cat-like, some change mainly in coarse shape or
texture, and others remain ambiguous. The ship comparison does not reproduce
the same pattern in this run. It tests that one alternative class correction,
not every possible nonspecific perturbation. The appropriate reading is a
class-specific but incomplete shift, not uniform conversion.

## 4. Why can an early edit survive later steps?

The next figure follows one illustrative seed whose endpoint changes from a
non-cat evaluator label to cat. It is not presented as a typical seed. Its job
is to show *when* two paired deterministic trajectories separate.

**Before you view the trajectory:** Once the correction window closes, both
runs again use the same EDM denoiser. Must the two paths reunite, or can an
earlier state difference persist?
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
The paths need not reunite. After the early correction changes one state, every
later deterministic update is evaluated at a different input. The shared
nonlinear denoiser can preserve and sharpen that difference even though no
additional correction is applied.

## 5. Read three measurements together

No single number settles the result. We therefore compare:

1. **target prediction rate**, which measures movement toward the requested
   evaluator label;
2. **distance to real target features**, which asks whether the batch moves
   toward real cat examples rather than only crossing a decision boundary;
3. **retained feature variation**, which warns when steering collapses the
   generated batch.

**Before you view the metrics:** A useful intervention should improve the first
two quantities without driving retained variation close to zero.
"""
    ),
    code(
        r"""
baseline_trace = result["metrics"]["baseline"]["feature_covariance_trace"]
metric_rows = []
for method in ["baseline", "zero", "target", "wrong"]:
    values = result["metrics"][method]
    metric_rows.append({
        "method": method,
        "target rate": values["target_rate"],
        "target feature distance": values["target_feature_mean_distance"],
        "retained feature variation": values["feature_covariance_trace"] / baseline_trace,
    })
pd.DataFrame(metric_rows).set_index("method").round(4)
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
On these 256 seeds, the frozen evaluator's cat rate rises from 6.6% to 41.8%,
while the target-guided batch retains 70.9% of the baseline feature variation.
Zero strength reproduces the baseline, and the ship correction does not explain
the cat increase in this comparison. The images and measurements agree on the
same conclusion: the early class-minus-full correction produces measurable
partial steering.

Most target-guided outputs are still not classified as cats. Evaluator features
also do not replace human inspection. The experiment therefore does not show a
reliable class-conditional generator, an optimal correction, or automatic
transfer to other classes and checkpoints.

The conceptual bridge from the toy notes is now complete: a denoiser can expose
a distribution-level intervention, and deterministic sampling makes its effect
traceable from one initial noise tensor to one changed endpoint.

**Think before class:** If the correction stayed active until very small
$\sigma$, would you expect more coarse class change, more fine-detail
distortion, or both? The current experiment does not vary that schedule on the
final-evaluation seeds.

**Sources**

- Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364).
- Li, Dai, and Qu, [*Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure*](https://arxiv.org/abs/2410.24060).
- [Official NVIDIA EDM implementation](https://github.com/NVlabs/edm).
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
