#!/usr/bin/env python3
"""Build the maintained toy-data pre-class notebooks from readable cell sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "toy_data"


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


def notebook(cells: list[dict]) -> dict:
    normalized_cells = []
    for index, cell in enumerate(cells):
        normalized = dict(cell)
        normalized["id"] = f"cell-{index:03d}"
        normalized_cells.append(normalized)
    return {
        "cells": normalized_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = r'''
import sys
from pathlib import Path

NOTES_DIR = Path.cwd()
if not (NOTES_DIR / "toy_notes.py").exists():
    NOTES_DIR = Path("notebooks/toy_data")
sys.path.insert(0, str(NOTES_DIR.resolve()))

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from toy_notes import *

SEED = 2026
set_seed(SEED)
device = choose_device()
print(environment_summary(device))
'''


NOTEBOOKS = {
    "00_diffusion_and_flow_matching_foundations.ipynb": [
        markdown(r'''
# 00 - From noise to data: diffusion and flow-matching foundations

How can a model turn unstructured noise into varied, data-like samples? We
start in two dimensions because every distribution, arrow, and trajectory can
be drawn. That visible setting will make the later image and steering examples
less mysterious. Python and vectors are enough background; no stochastic
calculus is assumed.

The toy dataset has three labeled clusters, but the generative model in later
notebooks will be trained **without labels**. Labels are retained only to test
post-hoc steering.
'''),
        code(SETUP),
        markdown(r'''
## 1. What is a generative model?

A generative model learns a **sampling procedure**. It is given a finite set of
training examples and learns how to create new samples with similar
distribution-level structure. It does not simply return a stored training
example.

For the flow model in these notes:

- an easy source distribution supplies fresh Gaussian noise;
- a neural network learns a time-dependent velocity;
- an ODE sampler follows that velocity from noise to a generated endpoint.

The neural network alone is therefore not the complete generator. The source
distribution, trained velocity model, and sampler work together. During
generation, each fresh noise seed starts a new sample.
'''),
        code(r'''
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

def process_box(ax, xy, width, height, text, color):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color, edgecolor="#333333", linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)

def process_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.4, color="#333333"))

fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.6))
for ax in axes:
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

axes[0].set_title("Training: use examples to learn a reusable velocity rule", loc="left", fontsize=12)
process_box(axes[0], (0.02, 0.56), 0.20, 0.25, "data examples\n$x_{data}$", "#D9EAD3")
process_box(axes[0], (0.02, 0.13), 0.20, 0.25, "noise samples\n$x_{noise}$", "#E7E6E6")
process_box(axes[0], (0.36, 0.34), 0.25, 0.28, "build a training pair\n$(x_t, x_{data}-x_{noise})$", "#FCE5CD")
process_box(axes[0], (0.75, 0.34), 0.21, 0.28, "fit velocity model\n$v_\\theta(t,x_t)$", "#CFE2F3")
process_arrow(axes[0], (0.22, 0.68), (0.36, 0.51))
process_arrow(axes[0], (0.22, 0.25), (0.36, 0.45))
process_arrow(axes[0], (0.61, 0.48), (0.75, 0.48))
axes[0].text(0.02, 0.02, "The network input contains $(t,x_t)$, not class labels.", fontsize=9, color="#555555")

axes[1].set_title("Generation: start from fresh noise; no clean target is supplied", loc="left", fontsize=12)
process_box(axes[1], (0.02, 0.34), 0.20, 0.28, "fresh noise seed\n$x_{noise}$", "#E7E6E6")
process_box(axes[1], (0.38, 0.34), 0.25, 0.28, "frozen $v_\\theta$\n+ ODE sampler", "#CFE2F3")
process_box(axes[1], (0.78, 0.34), 0.18, 0.28, "new generated\nsample", "#EAD1DC")
process_arrow(axes[1], (0.22, 0.48), (0.38, 0.48))
process_arrow(axes[1], (0.63, 0.48), (0.78, 0.48))
axes[1].text(0.02, 0.12, "Different seeds provide diversity.", fontsize=9, color="#555555")
axes[1].text(0.66, 0.12, "Same seed + same settings = same endpoint.", fontsize=9, color="#555555")

fig.suptitle("How a flow model learns and generates", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
'''),
        markdown(r'''
### Unconditional, conditional, and post-hoc control

- An **unconditional** generator models all training examples without receiving
  a requested class. That is the model trained here.
- A **conditional** generator receives extra information, such as a class label
  or text prompt, during training and generation.
- **Post-hoc steering** keeps a trained generator frozen and modifies the
  sampling computation. Notebooks `03`--`05` study this third setting.

Random noise is useful because it is easy to sample and supplies a different
starting state for each output. A deterministic sampler can still generate a
diverse collection: it maps different initial noise seeds to different
endpoints.
'''),
        markdown(r'''
## 2. Why begin with a dataset we can see?

A probability distribution describes which regions are likely and how often
different kinds of samples occur. Images live in thousands or millions of
dimensions, where that structure is difficult to draw. Here every sample has
only two coordinates, so a large scatter plot makes cluster locations,
frequencies, shapes, and spread directly visible. The three colors play the
role of semantic classes.
'''),
        code(r'''
clean, labels = sample_labeled_mixture(2400, device=device)
fig, ax = plt.subplots(figsize=(6, 5))
plot_labeled_points(clean, labels, ax=ax, title="Clean toy data")
ax.legend(frameon=False)
plt.show()
'''),
        markdown(r'''
## 3. Forward noising

First sample base noise, then use the linear path

$$x_{noise}=s\epsilon,\qquad \epsilon\sim\mathcal N(0,I),$$

$$x_t=(1-t)x_{noise}+t x_{data},\qquad s=1.8.$$

- At `t=1`, the point is clean data.
- At `t=0`, the point is Gaussian noise.
- Lower `t` therefore means higher noise in this notebook series.

A diffusion model learns to undo a noising process. Flow matching learns the
velocity of a path between the same endpoints.

### Diffusion and flow matching in one table

| Family | Typical learned quantity | Generation step |
|---|---|---|
| Diffusion model | noise, score, or a denoised estimate | repeatedly remove noise with a stochastic or deterministic sampler |
| Flow-matching model | velocity along a chosen probability path | integrate an ODE from the source distribution to data |

Both are time-dependent generative models. Here `t` indexes position along the
noise-to-data path; it is not elapsed computer time. We use flow matching
because the two-dimensional velocity arrows and complete trajectories are easy
to draw, then connect the same ideas to diffusion denoisers and steering.
'''),
        code(r'''
set_seed(SEED)
clean_small, labels_small = sample_labeled_mixture(1500, device=device)
fixed_noise = sample_base(clean_small.shape[0], device=device)

times = [1.0, 0.70, 0.35, 0.08]
fig, axes = plt.subplots(1, len(times), figsize=(15, 3.8))
for ax, t_value in zip(axes, times):
    t = torch.full((clean_small.shape[0],), t_value, device=device)
    noisy = forward_noising(clean_small, t, fixed_noise)
    plot_labeled_points(noisy, labels_small, ax=ax, title=f"t={t_value:.2f}", alpha=0.42)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
plt.tight_layout()
plt.show()
'''),
        markdown(r'''
## 4. What flow matching predicts

Choose a noise point `x_noise`, a data point `x_data`, and a time `t`. Along the
straight conditional path, the target velocity is simply

$$u_t = x_{data} - x_{noise}.$$

Training repeatedly asks a neural network to predict this arrow from only
`(x_t, t)`. Because many endpoint pairs can pass near the same `x_t`, the
network learns an average velocity field, not a memorized arrow for each pair.
The pair-specific target is available while constructing training examples but
is not available during generation.
'''),
        code(r'''
set_seed(SEED + 1)
x_data, pair_labels = sample_labeled_mixture(24, device=device)
x_noise = sample_base(24, device=device)
t_value = 0.42
x_t = forward_noising(x_data, torch.full((24,), t_value, device=device), x_noise)
u_t = x_data - x_noise

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x_noise[:, 0].cpu(), x_noise[:, 1].cpu(), s=18, color="#777777", label="noise endpoint")
ax.scatter(x_data[:, 0].cpu(), x_data[:, 1].cpu(), s=28, color="#0072B2", label="data endpoint")
ax.scatter(x_t[:, 0].cpu(), x_t[:, 1].cpu(), s=24, color="#E69F00", label="training state")
ax.quiver(
    x_t[:, 0].cpu(), x_t[:, 1].cpu(),
    u_t[:, 0].cpu(), u_t[:, 1].cpu(),
    angles="xy", scale_units="xy", scale=3.2, width=0.004, color="#222222", alpha=0.75,
)
ax.set_title("One flow-matching minibatch")
ax.set_aspect("equal")
ax.legend(frameon=False)
plt.show()
'''),
        markdown(r'''
## 5. Why keep the model and sampler separate?

Later we will change the numerical solver and add steering terms. Those changes
are easier to reason about if we know which part was learned and which part was
chosen at sampling time.

| Object | Job | Learned? |
|---|---|---|
| Training dataset | Supplies examples of the distribution to imitate | No |
| Base distribution | Supplies fresh, easy-to-sample Gaussian noise | No |
| Noising path | Defines intermediate training states | Chosen by us |
| Neural model | Predicts the local velocity `v_theta(t,x_t)` | Yes |
| Denoiser view | Converts velocity into a clean-data estimate | Derived from the model |
| Sampler | Numerically follows model predictions | Chosen by us |

In these notes, **generator** refers to the complete sampling procedure: base
distribution + trained neural model + sampler. Replacing Euler with RK4 changes
the sampler, not the learned model.

Notebook `01` trains the velocity model. Notebook `02` converts its velocity
prediction into a denoised estimate and compares deterministic samplers.

**Check your understanding**

1. What is visible at high noise: exact class details or only coarse structure?
2. Why is `x_data - x_noise` available during training but unavailable during generation?
3. Which component changes when Euler is replaced by RK4: the trained model or the sampler?
'''),
    ],
    "01_train_unconditional_flow_matching.ipynb": [
        markdown(r'''
# 01 - Train an unconditional flow-matching model

The equations in notebook `00` become useful only if a learned field actually
moves an entire source distribution into a target distribution. We therefore
train a small MLP ourselves, inspect what it gets right and wrong, and then test
the same recipe on three visibly different geometries. The model never receives
a class label.
'''),
        code(SETUP),
        markdown(r'''
## 1. Why train the model ourselves?

A pretrained checkpoint would hide the connection between sampled endpoint
pairs and the velocity loss. This small example lets us see the complete
training rule before using larger pretrained image models.

For each minibatch:

1. sample independent `x_noise` and `x_data`;
2. sample `t` uniformly from `[0,1]`;
3. form `x_t = t*x_data + (1-t)*x_noise`;
4. regress `v_theta(t,x_t)` onto `x_data - x_noise` with mean-squared error.

This is I-CFM. It is not optimal-transport CFM because we do not solve an
optimal coupling between noise and data samples.
'''),
        code(r'''
# The helper implements the four lines above and caches the checkpoint so every
# later notebook can also run independently.
model, losses, trained_now = load_or_train_model(device=device)
print("trained in this run:", trained_now)
print("checkpoint:", CHECKPOINT_PATH.relative_to(NOTES_DIR))
print("parameters:", sum(p.numel() for p in model.parameters()))
'''),
        code(r'''
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(losses, color="#0072B2", alpha=0.35, lw=0.8)
if len(losses) >= 100:
    window = 100
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(np.arange(window - 1, len(losses)), smooth, color="#D55E00", lw=2, label="100-step mean")
    ax.legend(frameon=False)
ax.set(xlabel="optimization step", ylabel="MSE", title="Flow-matching training loss")
ax.grid(alpha=0.2)
plt.show()
'''),
        markdown(r'''
## 2. How does a fitted field become samples?

Once an initial noise batch is fixed, the ODE sampler is deterministic:

$$\frac{dx}{dt}=v_\theta(t,x), \qquad x(0)=x_{noise}.$$

We use Heun's method here. Notebook `02` compares it with Euler and RK4.
'''),
        code(r'''
set_seed(SEED + 10)
x0_eval = sample_base(3000, device=device)
trajectory, sample_seconds = timed_sample(base_velocity(model), x0_eval, steps=80, method="heun")
generated = trajectory[-1].to(device)
target, target_labels = sample_labeled_mixture(3000, device=device)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
axes[0].scatter(x0_eval[:, 0].cpu(), x0_eval[:, 1].cpu(), s=6, alpha=0.25, color="#777777")
axes[0].set_title("Initial Gaussian noise")
plot_labeled_points(target, target_labels, ax=axes[1], title="Target data", alpha=0.35)
axes[2].scatter(generated[:, 0].cpu(), generated[:, 1].cpu(), s=7, alpha=0.35, color="#0072B2")
axes[2].set_title("Generated endpoints")
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.2, 4.5)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
print(f"sampling time: {sample_seconds:.3f} s")
'''),
        code(r'''
stats = estimate_gaussian_stats(target, target_labels)
predicted = predict_classes(generated, stats)
generated_balance = torch.bincount(predicted, minlength=len(CLASS_NAMES)).float()
generated_balance /= generated_balance.sum()

diagnostics = pd.DataFrame({
    "diagnostic": [
        "Wasserstein distance to full target",
        "generated class balance max error",
        "smallest generated mode fraction",
        "all samples finite",
    ],
    "value": [
        wasserstein_distance(generated, target),
        float(torch.max(torch.abs(generated_balance - 1 / len(CLASS_NAMES))).cpu()),
        float(generated_balance.min().cpu()),
        bool(torch.isfinite(generated).all()),
    ],
})
display(diagnostics)
print("generated class balance:", generated_balance.cpu().numpy().round(3))

# These are broad smoke checks, not a proof of distributional equality.
generation_checks = pd.DataFrame({
    "broad sanity check": ["finite endpoints", "all three modes represented", "no severe mode imbalance"],
    "passed": [
        bool(torch.isfinite(generated).all()),
        bool((generated_balance > 0.15).all()),
        bool(torch.max(torch.abs(generated_balance - 1 / len(CLASS_NAMES))) < 0.10),
    ],
})
display(generation_checks)
print("broad visual/metric sanity verdict:", "PASS" if generation_checks["passed"].all() else "CHECK OUTPUT")
'''),
        markdown(r'''
### Why use Wasserstein distance in addition to a plot?

A class rate or a mean checks only one property. Wasserstein distance asks for
the cheapest way to match generated points to target points, where moving a
point farther costs more. The code uses equal-size deterministic subsets and
reports the mean matched distance. It is a finite-sample estimate, not the
unknown population distance.

The lines below make the definition literal. Lower is better. We still inspect
mode coverage and spread because one scalar cannot diagnose every failure.
'''),
        code(r'''
from scipy.optimize import linear_sum_assignment

visual_count = 120
visual_indexes = torch.linspace(0, generated.shape[0] - 1, visual_count).round().long().to(device)
target_indexes = torch.linspace(0, target.shape[0] - 1, visual_count).round().long().to(device)
visual_generated = generated[visual_indexes]
visual_target = target[target_indexes]
matching_costs = torch.cdist(visual_generated, visual_target).cpu().numpy()
generated_rows, target_columns = linear_sum_assignment(matching_costs)
matched_distances = matching_costs[generated_rows, target_columns]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].scatter(
    visual_target[:, 0].cpu(), visual_target[:, 1].cpu(),
    s=16, alpha=0.45, color="#D55E00", label="target",
)
axes[0].scatter(
    visual_generated[:, 0].cpu(), visual_generated[:, 1].cpu(),
    s=16, alpha=0.45, color="#0072B2", label="generated",
)
axes[0].set(title="1. Two empirical distributions", aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.2, 4.5))
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].legend(frameon=False, loc="upper left")

for row, column in zip(generated_rows, target_columns):
    endpoints = torch.stack([visual_generated[row], visual_target[column]]).cpu()
    axes[1].plot(endpoints[:, 0], endpoints[:, 1], color="#999999", alpha=0.22, lw=0.7)
axes[1].scatter(visual_target[:, 0].cpu(), visual_target[:, 1].cpu(), s=13, alpha=0.55, color="#D55E00")
axes[1].scatter(visual_generated[:, 0].cpu(), visual_generated[:, 1].cpu(), s=13, alpha=0.55, color="#0072B2")
axes[1].set(title="2. Minimum-cost one-to-one matching", aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.2, 4.5))
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].hist(matched_distances, bins=18, color="#0072B2", alpha=0.82)
axes[2].axvline(matched_distances.mean(), color="#D55E00", lw=2, label="mean matched distance")
axes[2].set(title="3. Distances in the optimal matching", xlabel="matched Euclidean distance", ylabel="number of pairs")
axes[2].legend(frameon=False)
axes[2].grid(alpha=0.18, axis="y")

fig.suptitle(f"Empirical Wasserstein distance = {matched_distances.mean():.3f}", y=1.02)
plt.tight_layout()
plt.show()
'''),
        code(r'''
# A vector field is easier to understand when it is drawn.
grid_1d = torch.linspace(-4, 4, 19, device=device)
gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, t_value in zip(axes, [0.10, 0.50, 0.90]):
    t = torch.full((grid.shape[0],), t_value, device=device)
    with torch.no_grad():
        vectors = model(t, grid)
    ax.quiver(
        grid[:, 0].cpu(), grid[:, 1].cpu(), vectors[:, 0].cpu(), vectors[:, 1].cpu(),
        angles="xy", scale_units="xy", scale=18, width=0.003, color="#333333",
    )
    ax.set_title(f"Learned velocity at t={t_value:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-4.3, 4.3)
plt.tight_layout()
plt.show()
'''),
        markdown(r'''
## 3. Does the method learn a curved distribution?

Three Gaussian clusters could leave the impression that flow matching only
learns a few destination centers. A noisy circle is a stricter visual test: it
has continuously many valid endpoints and a thin curved support. We train a
separate unconditional model and ask whether it forms the ring without leaving
large angular gaps.
'''),
        code(r'''
main_checkpoint_digest_before = file_sha256(CHECKPOINT_PATH)
circle_train_started = time.perf_counter()
circle_model, circle_losses, circle_trained_now = load_or_train_circle_model(device=device)
circle_train_seconds = time.perf_counter() - circle_train_started
assert file_sha256(CHECKPOINT_PATH) == main_checkpoint_digest_before

circle_payload = torch.load(CIRCLE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
set_seed(SEED + 71)
circle_x0 = sample_base(4096, device=device)
circle_target, _ = sample_noisy_circle(4096, device=device)
circle_sample_started = time.perf_counter()
circle_path = integrate_ode(base_velocity(circle_model), circle_x0, steps=80, method="heun")
circle_sample_seconds = time.perf_counter() - circle_sample_started
circle_generated = circle_path[-1].to(device)

target_angles = torch.remainder(torch.atan2(circle_target[:, 1], circle_target[:, 0]), 2 * np.pi)
generated_angles = torch.remainder(torch.atan2(circle_generated[:, 1], circle_generated[:, 0]), 2 * np.pi)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
axes[0].scatter(circle_x0[:, 0].cpu(), circle_x0[:, 1].cpu(), s=5, alpha=0.20, color="#777777")
axes[0].set_title("Gaussian source")
axes[1].scatter(circle_target[:, 0].cpu(), circle_target[:, 1].cpu(), c=target_angles.cpu(), cmap="twilight", s=5, alpha=0.35)
axes[1].set_title("Noisy-circle target")
axes[2].scatter(circle_generated[:, 0].cpu(), circle_generated[:, 1].cpu(), c=generated_angles.cpu(), cmap="twilight", s=5, alpha=0.35)
axes[2].set_title("Generated endpoints")
axes[3].plot(circle_losses, color="#0072B2", alpha=0.45, lw=0.8)
if len(circle_losses) >= 100:
    circle_smooth = np.convolve(circle_losses, np.ones(100) / 100, mode="valid")
    axes[3].plot(np.arange(99, len(circle_losses)), circle_smooth, color="#D55E00", lw=2)
axes[3].set(title="Circle training loss", xlabel="optimization step", ylabel="MSE")
axes[3].grid(alpha=0.2)
for ax in axes[:3]:
    ax.set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
print(f"circle load/train seconds: {circle_train_seconds:.2f}")
print(f"circle sampling seconds: {circle_sample_seconds:.2f}")
'''),
        code(r'''
snapshot_steps = [0, 20, 40, 60, 80]
circle_plot_indexes = torch.linspace(0, circle_path.shape[1] - 1, 1800).round().long()
circle_final_colors = generated_angles[circle_plot_indexes.to(device)].cpu()

fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(16, 3.4))
for ax, step_index in zip(axes, snapshot_steps):
    points = circle_path[step_index][circle_plot_indexes]
    ax.scatter(points[:, 0], points[:, 1], c=circle_final_colors, cmap="twilight", s=5, alpha=0.38)
    if step_index == 80:
        reference = circle_target[:900].cpu()
        ax.scatter(reference[:, 0], reference[:, 1], s=7, facecolors="none", edgecolors="#444444", alpha=0.15, linewidths=0.4)
    ax.set_title(f"t={step_index / 80:.2f}")
    ax.set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("A deterministic flow forms a continuous ring; color follows final angle", y=1.02)
plt.tight_layout()
plt.show()
'''),
        code(r'''
circle_target_radii = torch.linalg.norm(circle_target, dim=1)
circle_generated_radii = torch.linalg.norm(circle_generated, dim=1)
circle_radial_mae = float(torch.mean(torch.abs(circle_generated_radii - CIRCLE_RADIUS)).cpu())

angular_bins = 16
target_bins = torch.floor(angular_bins * target_angles / (2 * np.pi)).long().clamp_max(angular_bins - 1)
generated_bins = torch.floor(angular_bins * generated_angles / (2 * np.pi)).long().clamp_max(angular_bins - 1)
target_occupancy = torch.bincount(target_bins, minlength=angular_bins).float()
target_occupancy /= target_occupancy.sum()
generated_occupancy = torch.bincount(generated_bins, minlength=angular_bins).float()
generated_occupancy /= generated_occupancy.sum()
circle_angular_tv = float((0.5 * torch.abs(generated_occupancy - target_occupancy).sum()).cpu())

circle_source_wasserstein = wasserstein_distance(circle_x0, circle_target)
circle_generated_wasserstein = wasserstein_distance(circle_generated, circle_target)
circle_wasserstein_ratio = circle_generated_wasserstein / circle_source_wasserstein

circle_diagnostics = pd.DataFrame({
    "diagnostic": [
        "target mean radius", "generated mean radius", "generated radial MAE",
        "angular occupancy total variation", "source-to-target Wasserstein distance",
        "generated-to-target Wasserstein distance", "generated/source distance ratio",
    ],
    "value": [
        float(circle_target_radii.mean().cpu()), float(circle_generated_radii.mean().cpu()),
        circle_radial_mae, circle_angular_tv, circle_source_wasserstein,
        circle_generated_wasserstein, circle_wasserstein_ratio,
    ],
})
display(circle_diagnostics.round(4))

circle_checks = pd.DataFrame({
    "check": [
        "all endpoints finite", "radial MAE at most 0.35",
        "angular occupancy TV at most 0.15", "Wasserstein distance less than 55% of source",
        "checkpoint metadata matches requested geometry",
    ],
    "passed": [
        bool(torch.isfinite(circle_generated).all()),
        circle_radial_mae <= 0.35,
        circle_angular_tv <= 0.15,
        circle_wasserstein_ratio <= 0.55,
        circle_payload["training_steps"] == int(os.environ.get("TOY_NOTES_CIRCLE_STEPS", "3000"))
        and circle_payload["radius"] == CIRCLE_RADIUS,
    ],
})
display(circle_checks)
print("circle geometry verdict:", "PASS" if circle_checks["passed"].all() else "CHECK OUTPUT")
'''),
        markdown(r'''
## 4. Why also inspect eight separated modes?

The three-class model is kept for the steering notebooks because its labels are
easy to discuss, while the circle tests continuous curved geometry. Eight narrow
components expose a different failure mode: a model can omit destinations or
assign them the wrong amount of probability. This is still learned I-CFM; no
destination center is used by the sampler.

We color each particle by the component nearest to its **final** endpoint, then
trace the same particle backward through stored time slices. The colors are for
reading the map after sampling; the model never receives them.
'''),
        code(r'''
import os
import time

main_checkpoint_digest_before = file_sha256(CHECKPOINT_PATH)
ring_train_started = time.perf_counter()
ring_model, ring_losses, ring_trained_now = load_or_train_eight_gaussian_model(device=device)
ring_train_seconds = time.perf_counter() - ring_train_started
assert file_sha256(CHECKPOINT_PATH) == main_checkpoint_digest_before

ring_payload = torch.load(EIGHT_GAUSSIAN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
ring_config = pd.DataFrame({
    "setting": ["training steps", "training seed", "base std", "component std", "hidden width", "cache used", "load/train seconds"],
    "value": [
        ring_payload["training_steps"], ring_payload["training_seed"], ring_payload["base_std"],
        ring_payload["component_std"], ring_payload["hidden_dim"], not ring_trained_now, ring_train_seconds,
    ],
})
display(ring_config)
print("eight-Gaussian checkpoint SHA-256:", file_sha256(EIGHT_GAUSSIAN_CHECKPOINT_PATH))

set_seed(SEED + 81)
ring_x0 = sample_base(4096, device=device)
ring_target_labels = torch.arange(8, device=device).repeat_interleave(512)
ring_target, ring_target_labels = sample_eight_gaussians(
    ring_target_labels.numel(), device=device, labels=ring_target_labels
)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
axes[0].scatter(ring_x0[:, 0].cpu(), ring_x0[:, 1].cpu(), s=5, alpha=0.22, color="#777777")
axes[0].set_title("Fixed Gaussian source")
for component in range(8):
    points = ring_target[ring_target_labels == component].cpu()
    axes[1].scatter(points[:, 0], points[:, 1], s=6, alpha=0.30, color=EIGHT_GAUSSIAN_COLORS[component])
axes[1].set_title("Balanced eight-Gaussian target")
for ax in axes:
    ax.set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

ring_sample_started = time.perf_counter()
ring_path = integrate_ode(base_velocity(ring_model), ring_x0, steps=80, method="heun")
ring_sample_seconds = time.perf_counter() - ring_sample_started
ring_generated = ring_path[-1].to(device)
ring_centers = EIGHT_GAUSSIAN_CENTERS.to(device)
ring_assignments = torch.cdist(ring_generated, ring_centers).argmin(dim=1)
print(f"eight-mode sampling seconds: {ring_sample_seconds:.3f}")
'''),
        code(r'''
snapshot_steps = [0, 20, 40, 60, 80]
plot_indexes = torch.linspace(0, ring_path.shape[1] - 1, 1600).long()
plot_assignments = ring_assignments[plot_indexes.to(device)].cpu()
fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(16, 3.4))
for ax, step_index in zip(axes, snapshot_steps):
    for component in range(8):
        mask = plot_assignments == component
        points = ring_path[step_index][plot_indexes][mask]
        ax.scatter(
            points[:, 0], points[:, 1], s=5, alpha=0.33,
            color=EIGHT_GAUSSIAN_COLORS[component],
        )
        if step_index == 80:
            reference = ring_target[ring_target_labels == component][:90].detach().cpu()
            ax.scatter(
                reference[:, 0], reference[:, 1], s=8, facecolors="none", alpha=0.28,
                edgecolors=EIGHT_GAUSSIAN_COLORS[component], linewidths=0.5,
            )
    ax.set_title(f"t={step_index / 80:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("One deterministic learned flow map; color is assigned from the final endpoint", y=1.02)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6.2, 6.0))
for component in range(8):
    reference = ring_target[ring_target_labels == component][:120].detach().cpu()
    ax.scatter(reference[:, 0], reference[:, 1], s=7, alpha=0.10, color=EIGHT_GAUSSIAN_COLORS[component])
indexes = torch.linspace(0, ring_path.shape[1] - 1, 64).long()
for index in indexes:
    index = int(index)
    component = int(ring_assignments[index].cpu())
    ax.plot(
        ring_path[:, index, 0], ring_path[:, index, 1],
        color=EIGHT_GAUSSIAN_COLORS[component], alpha=0.50, lw=0.9,
    )
    ax.scatter(ring_path[0, index, 0], ring_path[0, index, 1], s=17, facecolors="none", edgecolors=EIGHT_GAUSSIAN_COLORS[component], linewidths=0.7)
    ax.scatter(ring_path[-1, index, 0], ring_path[-1, index, 1], s=12, color=EIGHT_GAUSSIAN_COLORS[component])
ax.set(title="64 complete trajectories through the learned eight-mode flow", aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
ax.set_xticks([])
ax.set_yticks([])
plt.show()
'''),
        code(r'''
generated_counts = torch.bincount(ring_assignments, minlength=8).float()
generated_fractions = generated_counts / generated_counts.sum()
target_assignments = torch.cdist(ring_target, ring_centers).argmin(dim=1)
target_counts = torch.bincount(target_assignments, minlength=8).float()
target_fractions = target_counts / target_counts.sum()

uniform_fractions = torch.full_like(generated_fractions, 1 / 8)
occupancy_tv = float((0.5 * torch.abs(generated_fractions - uniform_fractions).sum()).cpu())
source_wasserstein = wasserstein_distance(ring_x0, ring_target)
ring_wasserstein = wasserstein_distance(ring_generated, ring_target)
ring_wasserstein_ratio = ring_wasserstein / source_wasserstein
covered_modes = int((generated_fractions >= 0.05).sum().cpu())

mode_rows = []
for component in range(8):
    generated_mode = ring_generated[ring_assignments == component]
    target_mode = ring_target[ring_target_labels == component]
    target_mean = target_mode.mean(dim=0)
    target_rms = torch.sqrt((target_mode - target_mean).square().sum(dim=1).mean())
    centroid_error = torch.linalg.norm(generated_mode.mean(dim=0) - target_mean) / target_rms.clamp_min(1e-8)
    generated_trace = torch.var(generated_mode, dim=0, unbiased=False).sum()
    target_trace = torch.var(target_mode, dim=0, unbiased=False).sum()
    mode_rows.append({
        "mode": component,
        "generated share": float(generated_fractions[component].cpu()),
        "target share": float(target_fractions[component].cpu()),
        "normalized centroid error": float(centroid_error.cpu()),
        "within-mode trace ratio": float((generated_trace / target_trace).cpu()),
    })
mode_table = pd.DataFrame(mode_rows)
display(mode_table.round(3))

ring_diagnostics = pd.DataFrame({
    "diagnostic": [
        "modes with at least 5% generated mass",
        "occupancy total variation from uniform",
        "source-to-target Wasserstein distance",
        "generated-to-target Wasserstein distance",
        "generated/source Wasserstein distance ratio",
        "mean normalized centroid error",
        "median within-mode trace ratio",
        "all endpoints finite",
    ],
    "value": [
        covered_modes,
        occupancy_tv,
        source_wasserstein,
        ring_wasserstein,
        ring_wasserstein_ratio,
        mode_table["normalized centroid error"].mean(),
        mode_table["within-mode trace ratio"].median(),
        bool(torch.isfinite(ring_generated).all()),
    ],
})
display(ring_diagnostics.round(4))
print("generated occupancy:", generated_fractions.cpu().numpy().round(3))
print("target occupancy:   ", target_fractions.cpu().numpy().round(3))

ring_checks = pd.DataFrame({
    "release check": [
        "all samples finite",
        "all eight mode shares at least 0.05",
        "occupancy total variation at most 0.15",
        "generated Wasserstein distance at most half source Wasserstein distance",
        "mean normalized centroid error at most 1.5",
        "median trace ratio in [0.5, 2.5]",
        "every trace ratio at least 0.25",
        "checkpoint metadata matches requested geometry",
    ],
    "passed": [
        bool(torch.isfinite(ring_generated).all()),
        covered_modes == 8,
        occupancy_tv <= 0.15,
        ring_wasserstein_ratio <= 0.50,
        mode_table["normalized centroid error"].mean() <= 1.5,
        0.5 <= mode_table["within-mode trace ratio"].median() <= 2.5,
        mode_table["within-mode trace ratio"].min() >= 0.25,
        ring_payload["training_steps"] == int(os.environ.get("TOY_NOTES_EIGHT_STEPS", "3000"))
        and ring_payload["component_std"] == EIGHT_GAUSSIAN_STD,
    ],
})
display(ring_checks)
print("eight-mode map release verdict:", "PASS" if ring_checks["passed"].all() else "CHECK OUTPUT")
'''),
        markdown(r'''
The snapshot figure shows **where the learned ODE sends regions of the initial
Gaussian**, while the diagnostics check mode coverage and occupancy. Passing
these broad checks does not prove that every local density is exact; the
Wasserstein distance and within-component spread still matter.
'''),
        markdown(r'''
## What these examples establish

The loss curve alone is not enough. The three-cluster sample checks familiar
multimodal structure, the circle checks continuous curved geometry, and the
eight-Gaussian map makes missing modes visible. Together they give us a trained
unconditional model and a reason to inspect sampling and steering with more
than one metric.

The three cached models live in `artifacts/toy_flow_model.pt`,
`artifacts/circle_flow_model.pt`, and `artifacts/eight_gaussian_flow_model.pt`.

**Questions**

1. Why can a low minibatch MSE coexist with visibly poor samples?
2. What information about class identity did the unconditional model receive?
3. At which time does the vector field mostly choose coarse destination modes?
'''),
    ],
    "02_denoisers_and_deterministic_samplers.ipynb": [
        markdown(r'''
# 02 - Denoisers and deterministic samplers

Our network predicts velocity, while diffusion papers and the CIFAR experiment
often describe a denoised or clean-image estimate. Steering will be much easier
to understand once we can translate between those two languages. We then ask a
separate practical question: how accurately must a numerical solver follow the
learned field?
'''),
        code(SETUP),
        code(r'''
model, losses, trained_now = load_or_train_model(device=device)
print("trained in this run:", trained_now)
'''),
        markdown(r'''
## 1. Why can a velocity prediction be read as a denoiser?

Start with one exact training pair. Along the linear conditional path,

$$x_t=x_{noise}+t(x_{data}-x_{noise}).$$

Its pair-specific velocity is $u=x_{data}-x_{noise}$. Rearranging the path gives

$$x_{data}=x_t+(1-t)u.$$

This first identity is exact because the two endpoints are known while we build
a training example. During generation the network sees only $(x_t,t)$, not
those endpoints. Mean-squared-error training therefore learns the conditional
average

$$v^*(x_t,t)=\mathbb E[u\mid x_t,t].$$

Substituting that average into the exact identity gives

$$D^*(x_t,t)=x_t+(1-t)v^*(x_t,t)
=\mathbb E[x_{data}\mid x_t,t].$$

For the trained network we use the same conversion:

$$D_\theta(x_t,t)=x_t+(1-t)v_\theta(x_t,t).$$

The result is the best mean-squared-error clean estimate under the training
pair distribution. It is an average over plausible clean endpoints, not a
guarantee that the model has identified one hidden original. When several modes
are plausible, the average can sit between them; in images, such averaging can
look blurry.

Away from $t=1$, the conversion can also be inverted as
$v_\theta=(D_\theta-x_t)/(1-t)$. Near $t=1$ that division is poorly conditioned,
so code should not use it exactly at the endpoint. This dependence on the
chosen linear path is important: a different path has different conversion
factors.
'''),
        code(r'''
set_seed(SEED + 19)
visual_clean, visual_labels = sample_labeled_mixture(180, device=device)
visual_noise = sample_base(180, device=device)
visual_t_value = 0.35
visual_t = torch.full((180,), visual_t_value, device=device)
visual_xt = forward_noising(visual_clean, visual_t, visual_noise)
pair_velocity = visual_clean - visual_noise
with torch.no_grad():
    learned_velocity = model(visual_t, visual_xt)
    clean_estimate = denoised_from_velocity(visual_xt, visual_t, learned_velocity)

subset = torch.arange(0, 180, 6, device=device)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax in axes:
    ax.scatter(visual_xt[:, 0].cpu(), visual_xt[:, 1].cpu(), s=8, alpha=0.16, color="#777777")
    ax.set(aspect="equal", xlim=(-5, 5), ylim=(-5, 5))
    ax.set_xticks([])
    ax.set_yticks([])

axes[0].quiver(
    visual_xt[subset, 0].cpu(), visual_xt[subset, 1].cpu(),
    pair_velocity[subset, 0].cpu(), pair_velocity[subset, 1].cpu(),
    angles="xy", scale_units="xy", scale=3.0, width=0.004, color="#D55E00",
)
axes[0].set_title("Training target: endpoint pair is known")

axes[1].quiver(
    visual_xt[subset, 0].cpu(), visual_xt[subset, 1].cpu(),
    learned_velocity[subset, 0].cpu(), learned_velocity[subset, 1].cpu(),
    angles="xy", scale_units="xy", scale=3.0, width=0.004, color="#0072B2",
)
axes[1].set_title("Learned velocity: only (t, x_t) is known")

for index in subset:
    estimate_line = torch.stack([visual_xt[index], clean_estimate[index]]).cpu()
    axes[2].plot(estimate_line[:, 0], estimate_line[:, 1], color="#999999", alpha=0.45, lw=0.8)
axes[2].scatter(clean_estimate[:, 0].cpu(), clean_estimate[:, 1].cpu(), s=10, alpha=0.35, color="#0072B2", label="D_theta")
axes[2].scatter(visual_clean[:, 0].cpu(), visual_clean[:, 1].cpu(), s=14, alpha=0.25, facecolors="none", edgecolors="#D55E00", label="paired clean endpoint")
axes[2].set_title("D_theta = current state + remaining velocity")
axes[2].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle(f"The velocity-to-denoiser conversion at t={visual_t_value:.2f}", y=1.02)
plt.tight_layout()
plt.show()
'''),
        code(r'''
set_seed(SEED + 20)
clean, labels = sample_labeled_mixture(450, device=device)
noise = sample_base(clean.shape[0], device=device)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, t_value in zip(axes, [0.15, 0.45, 0.75]):
    t = torch.full((clean.shape[0],), t_value, device=device)
    xt = forward_noising(clean, t, noise)
    with torch.no_grad():
        velocity = model(t, xt)
        denoised = denoised_from_velocity(xt, t, velocity)

    subset = torch.arange(0, clean.shape[0], 15, device=device)
    ax.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), s=7, alpha=0.18, color="#777777", label="current state")
    ax.scatter(denoised[:, 0].cpu(), denoised[:, 1].cpu(), s=8, alpha=0.35, color="#0072B2", label="denoised estimate")
    arrows = denoised[subset] - xt[subset]
    ax.quiver(
        xt[subset, 0].cpu(), xt[subset, 1].cpu(), arrows[:, 0].cpu(), arrows[:, 1].cpu(),
        angles="xy", scale_units="xy", scale=1, width=0.004, color="#D55E00", alpha=0.8,
    )
    ax.set_title(f"Denoiser view at t={t_value:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
axes[0].legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.show()
'''),
        markdown(r'''
## 2. Solver comparison

The model defines the arrows; the solver decides how to follow them. A fair
solver comparison changes only the solver and step count. It keeps the model,
initial noise tensor, and evaluation reference fixed.

Deterministic is conditional on the initial state: repeating the same
`x_noise` repeats the trajectory, while different initial noise points can
still produce a diverse batch of endpoints.

We use a 256-step RK4 trajectory as a numerical reference. This reference does
not remove model error; it only makes solver error small.
'''),
        code(r'''
set_seed(SEED + 21)
x0 = sample_base(1200, device=device)
target, _ = sample_labeled_mixture(1200, device=device)

reference_path, reference_seconds = timed_sample(base_velocity(model), x0, steps=256, method="rk4")
reference_endpoint = reference_path[-1]

rows = []
saved_paths = {}
for method in ["euler", "heun", "rk4"]:
    for steps in [8, 16, 32, 64]:
        path, seconds = timed_sample(base_velocity(model), x0, steps=steps, method=method)
        paired_rmse = torch.sqrt(torch.mean((path[-1] - reference_endpoint) ** 2)).item()
        rows.append({
            "solver": method,
            "steps": steps,
            "model evaluations": steps * {"euler": 1, "heun": 2, "rk4": 4}[method],
            "paired endpoint RMSE": paired_rmse,
            "Wasserstein distance to target": wasserstein_distance(path[-1], target.cpu()),
            "seconds": seconds,
        })
        if (method, steps) in {("euler", 8), ("heun", 16), ("rk4", 16)}:
            saved_paths[f"{method}, {steps} steps"] = path

solver_table = pd.DataFrame(rows)
display(solver_table.round(4))
print(f"reference RK4 runtime: {reference_seconds:.3f} s")
'''),
        code(r'''
plot_trajectory_comparison(saved_paths, n_lines=22)
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
for method, group in solver_table.groupby("solver"):
    ax.loglog(group["model evaluations"], group["paired endpoint RMSE"], marker="o", label=method)
ax.set(xlabel="model evaluations", ylabel="paired endpoint RMSE", title="Numerical accuracy at fixed initial noise")
ax.grid(alpha=0.25, which="both")
ax.legend(frameon=False)
plt.show()
'''),
        markdown(r'''
## 3. Two meanings that must remain separate

**Paired initial noise** is an experimental control. With a deterministic
sampler, the same initial tensor gives sample-by-sample correspondence between
two methods. Endpoint differences can then be attributed to the changed method.

**Noise alignment in NA-RFM** is a steering method. It uses the difference
between a target-class PCA denoiser and a full-data PCA denoiser during the
high-noise window. Notebook `03` implements that mechanism in 2D.

Using the same seed does not implement NA-RFM noise alignment. Conversely, a
noise-alignment experiment should still use paired initial noise for fair
evaluation.

**Questions**

1. Why does RK4 usually need fewer steps but more model evaluations per step?
2. Why is endpoint RMSE meaningful only because every solver starts from the exact same tensor?
3. Does a more accurate ODE solver guarantee a better trained vector field?
'''),
    ],
    "03_gaussian_denoisers_and_noise_alignment.ipynb": [
        markdown(r'''
# 03 - Class steering with Gaussian/PCA denoisers

At high noise, a generator cannot yet know fine sample details, but means and
major directions of variation can still provide useful structure. This
notebook first tests that idea on MNIST pixels, then uses the difference between
a target-class and full-data denoiser as a coarse post-hoc steering signal:

$$\Delta D = D_{target\ class} - D_{full\ data}.$$

There is no hand-written anchor or attraction toward a coordinate. The final
2D experiment is the visible counterpart of the high-noise noise-alignment
stage in NA-RFM.
'''),
        code(SETUP),
        code(r'''
model, losses, trained_now = load_or_train_model(device=device)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
checkpoint_digest_before = model_state_digest(model)

set_seed(SEED + 301)
steering_data, steering_labels = sample_labeled_mixture(30000, device=device)
steering_stats = estimate_gaussian_stats(steering_data, steering_labels)
set_seed(SEED + 302)
evaluation_data, evaluation_labels = sample_labeled_mixture(30000, device=device)
evaluation_stats = estimate_gaussian_stats(evaluation_data, evaluation_labels)
TARGET_CLASS = 2
set_seed(SEED + 303)
target_reference = sample_class(1800, TARGET_CLASS, device=device)
print("target class:", CLASS_NAMES[TARGET_CLASS])
print("frozen checkpoint digest:", checkpoint_digest_before[:16])
'''),
        markdown(r'''
## 1. Why a Gaussian denoiser makes sense at high noise

Write the base point as $x_{noise}=s\epsilon$ with
$\epsilon\sim\mathcal{N}(0,I)$ and `s=1.8`, then divide the path by `t`:

$$\frac{x_t}{t}=x_{data}+s\frac{1-t}{t}\epsilon.$$

The right side is clean data plus Gaussian noise with effective noise level
`sigma_eff=s(1-t)/t`. For a Gaussian data model, the posterior mean is

$$D(y,\sigma)=\mu+C(C+\sigma^2I)^{-1}(y-\mu).$$

If $C=U\Lambda U^\top$, each principal-component coordinate is multiplied by
$\lambda_i/(\lambda_i+\sigma^2)$. Large-variance directions survive; uncertain
small-variance directions shrink toward the mean. This is why PCA is a useful
way to compute the same rule for images.

The factor `s` matters because our base distribution is not unit variance.
At high noise the denoiser trusts coarse mean/covariance statistics; at low noise
it stays close to the observed point.

The full-data Gaussian is deliberately crude because the complete dataset is a
mixture. That limitation is useful: subtracting the full denoiser from the
target-class denoiser isolates a coarse class correction.

[Li, Dai, and Qu (NeurIPS 2024)](https://arxiv.org/abs/2410.24060) report that
diffusion denoisers become increasingly linear in a generalization regime and
that their best linear approximations are close to empirical Gaussian
denoisers. Their study includes FFHQ and checks on CIFAR-10, AFHQ, and
LSUN-Churches. The MNIST calculation below is our teaching demonstration, not
an experiment from that paper. We do not present it as ImageNet evidence; the
later CIFAR-10 notebook supplies the image-generator experiment in this series.
'''),
        markdown(r'''
## 2. Does a covariance model retain useful image structure?

MNIST makes all 784 pixel coordinates visible without needing a large network.
We fit low-rank Gaussian models on the training split only: one to all digits
and one to digit 3. The leading covariance directions are `eigendigits`.
On held-out test images, we add known Gaussian noise and compare the noisy input
with both posterior-mean denoisers.
'''),
        code(r'''
MNIST_TARGET = 3
MNIST_RANK = 64
MNIST_SIGMA = 0.45

mnist_train_images, mnist_train_labels = load_mnist_split(train=True, device=device)
mnist_test_images, mnist_test_labels = load_mnist_split(train=False, device=device)

full_fit_indexes = torch.linspace(
    0, mnist_train_images.shape[0] - 1, 12000, device=device
).round().long()
class_fit_pool = torch.where(mnist_train_labels == MNIST_TARGET)[0]
class_fit_indexes = class_fit_pool[: min(6000, class_fit_pool.numel())]

set_seed(SEED + 330)
mnist_full_gaussian = fit_low_rank_gaussian(
    mnist_train_images[full_fit_indexes], rank=MNIST_RANK
)
set_seed(SEED + 331)
mnist_class_gaussian = fit_low_rank_gaussian(
    mnist_train_images[class_fit_indexes], rank=MNIST_RANK
)

print("MNIST training images used for full Gaussian:", full_fit_indexes.numel())
print("MNIST digit-3 training images used:", class_fit_indexes.numel())
print("PCA rank:", MNIST_RANK)
'''),
        code(r'''
fig, axes = plt.subplots(2, 6, figsize=(10, 3.8))
axes[0, 0].imshow(mnist_full_gaussian.mean.reshape(28, 28).cpu(), cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title("all-digit mean")
axes[1, 0].imshow(mnist_class_gaussian.mean.reshape(28, 28).cpu(), cmap="gray", vmin=0, vmax=1)
axes[1, 0].set_title("digit-3 mean")
for column in range(1, 6):
    for row, stats, label in [
        (0, mnist_full_gaussian, "all"),
        (1, mnist_class_gaussian, "digit 3"),
    ]:
        component = stats.components[:, column - 1].reshape(28, 28).cpu()
        scale = float(component.abs().max().clamp_min(1e-8))
        axes[row, column].imshow(component, cmap="RdBu_r", vmin=-scale, vmax=scale)
        axes[row, column].set_title(f"{label} PC {column}")
for ax in axes.flat:
    ax.axis("off")
fig.suptitle("Means and leading covariance directions learned from MNIST training images", y=1.02)
plt.tight_layout()
plt.show()
'''),
        code(r'''
held_out_pool = torch.where(mnist_test_labels == MNIST_TARGET)[0]
held_out_indexes = held_out_pool[:256]
mnist_clean = mnist_test_images[held_out_indexes]
set_seed(SEED + 332)
mnist_noisy = mnist_clean + MNIST_SIGMA * torch.randn_like(mnist_clean)
mnist_full_denoised = low_rank_gaussian_denoise(
    mnist_noisy, MNIST_SIGMA, mnist_full_gaussian
)
mnist_class_denoised = low_rank_gaussian_denoise(
    mnist_noisy, MNIST_SIGMA, mnist_class_gaussian
)

def image_mse(images):
    return float(torch.mean((images - mnist_clean).square()).cpu())

mnist_mse = {
    "noisy input": image_mse(mnist_noisy),
    "all-digit Gaussian": image_mse(mnist_full_denoised),
    "digit-3 Gaussian": image_mse(mnist_class_denoised),
}
mnist_metrics = pd.DataFrame({
    "method": list(mnist_mse),
    "held-out MSE": list(mnist_mse.values()),
    "PSNR (dB)": [10 * np.log10(1.0 / value) for value in mnist_mse.values()],
})
display(mnist_metrics.round(4))

display_count = 10
fig, axes = plt.subplots(4, display_count, figsize=(13, 5.4))
rows = [
    ("clean test image", mnist_clean),
    (f"noisy, sigma={MNIST_SIGMA}", mnist_noisy),
    ("all-digit Gaussian", mnist_full_denoised),
    ("digit-3 Gaussian", mnist_class_denoised),
]
for row, (label, images) in enumerate(rows):
    for column in range(display_count):
        axes[row, column].imshow(images[column].reshape(28, 28).detach().cpu().clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        axes[row, column].axis("off")
    axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=34)
fig.suptitle("Gaussian/PCA posterior means on held-out digit-3 images", y=1.01)
plt.tight_layout()
plt.show()

mnist_checks = pd.DataFrame({
    "check": ["all-digit Gaussian beats noisy input", "digit-3 Gaussian beats noisy input"],
    "passed": [
        mnist_mse["all-digit Gaussian"] < mnist_mse["noisy input"],
        mnist_mse["digit-3 Gaussian"] < mnist_mse["noisy input"],
    ],
})
display(mnist_checks)
print("MNIST Gaussian denoising verdict:", "PASS" if mnist_checks["passed"].all() else "CHECK OUTPUT")
'''),
        markdown(r'''
## 3. How does denoising become a steering signal?

The image example shows what one Gaussian posterior mean preserves and loses.
For steering, we compare two such estimates at the same noisy state. Their
difference asks for class-level structure that the target model favors more
than the full-data model. The gate below limits that correction to the
high-noise part of the trajectory.
'''),
        code(r'''
grid_1d = torch.linspace(-4.5, 4.5, 19, device=device)
gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
t_value = 0.18
t = torch.full((grid.shape[0],), t_value, device=device)
delta = noise_alignment_delta(grid, t, TARGET_CLASS, steering_stats)

gate_times = torch.linspace(0, 1, 300, device=device)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
axes[0].plot(gate_times.cpu(), window_gate(gate_times, 0.04, 0.35).cpu(), color="#0072B2", lw=2)
axes[0].set(xlabel="t: noise to data", ylabel="gate g(t)", title="Correction is active only at high noise")
axes[0].grid(alpha=0.2)
axes[1].quiver(
    grid[:, 0].cpu(), grid[:, 1].cpu(), delta[:, 0].cpu(), delta[:, 1].cpu(),
    angles="xy", scale_units="xy", scale=1.7, width=0.004, color=CLASS_COLORS[TARGET_CLASS], alpha=0.8,
)
plot_labeled_points(
    steering_data[:2500], steering_labels[:2500], ax=axes[1],
    title=f"D_target - D_full at t={t_value} (clean-estimate coordinates)", alpha=0.18,
)
axes[1].set_xlim(-4.8, 4.8)
axes[1].set_ylim(-4.8, 4.8)
plt.tight_layout()
plt.show()
'''),
        markdown(r'''
## 4. Does the correction improve paired samples?

All methods below use the same initial noise, Heun solver, step count, and target
reference. The only change is the noise-alignment strength. The correction is
active for `t in [0.04, 0.35]`, the high-noise part of our noise-to-data path.
'''),
        code(r'''
set_seed(SEED + 30)
x0 = sample_base(1800, device=device)

methods = {
    "baseline": base_velocity(model),
    "noise alignment 0.0": make_noise_aligned_velocity(
        model, steering_stats, target_class=TARGET_CLASS, strength=0.0, start=0.04, end=0.35
    ),
}
for strength in [0.4, 0.8, 1.2, 1.6]:
    methods[f"noise alignment {strength:.1f}"] = make_noise_aligned_velocity(
        model, steering_stats, target_class=TARGET_CLASS, strength=strength, start=0.04, end=0.35
    )

rows = []
paths = {}
for name, velocity in methods.items():
    path, seconds = timed_sample(velocity, x0, steps=64, method="heun")
    metrics = evaluate_samples(
        path[-1].to(device), target_reference, evaluation_stats, target_class=TARGET_CLASS
    )
    strength = np.nan if name == "baseline" else float(name.rsplit(" ", 1)[-1])
    rows.append({"method": name, "strength": strength, **metrics, "seconds": seconds})
    paths[name] = path

noise_alignment_table = pd.DataFrame(rows)
display(noise_alignment_table.round(4))

zero_strength_error = float(torch.max(torch.abs(paths["baseline"][-1] - paths["noise alignment 0.0"][-1])))
print(f"paired endpoint max difference, baseline vs zero strength: {zero_strength_error:.2e}")
assert zero_strength_error < 1e-6
assert model_state_digest(model) == checkpoint_digest_before
print("checkpoint digest unchanged after every rollout: True")
'''),
        code(r'''
baseline_endpoints = paths["baseline"][-1].to(device)
aligned_endpoints = paths["noise alignment 1.6"][-1].to(device)
baseline_classes = predict_classes(baseline_endpoints, evaluation_stats)
aligned_classes = predict_classes(aligned_endpoints, evaluation_stats)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
for ax, endpoints, predicted, title in [
    (axes[0], baseline_endpoints, baseline_classes, "Baseline endpoints"),
    (axes[1], aligned_endpoints, aligned_classes, "Noise alignment, strength 1.6"),
]:
    ax.scatter(target_reference[:, 0].cpu(), target_reference[:, 1].cpu(), s=8, alpha=0.12, color="#555555", label="target examples")
    for class_id, class_name in enumerate(CLASS_NAMES):
        subset = endpoints[predicted == class_id].cpu()
        ax.scatter(subset[:, 0], subset[:, 1], s=7, alpha=0.38, color=CLASS_COLORS[class_id], label=class_name)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

axes[2].scatter(target_reference[:, 0].cpu(), target_reference[:, 1].cpu(), s=8, alpha=0.10, color="#555555")
paired_indexes = torch.linspace(0, baseline_endpoints.shape[0] - 1, 70).long().to(device)
starts = baseline_endpoints[paired_indexes]
changes = aligned_endpoints[paired_indexes] - starts
axes[2].quiver(
    starts[:, 0].cpu(), starts[:, 1].cpu(), changes[:, 0].cpu(), changes[:, 1].cpu(),
    angles="xy", scale_units="xy", scale=1, width=0.004, color="#0072B2", alpha=0.55,
)
axes[2].set_title("Paired baseline-to-steered endpoint changes")
for ax in axes:
    ax.set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.2, 4.5))
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

sweep = noise_alignment_table.dropna(subset=["strength"]).sort_values("strength")
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
metric_specs = [
    ("target_rate", "target-class rate (higher)", None),
    ("target_wasserstein", "Wasserstein distance to target (lower)", None),
    ("mean_error", "target mean error (lower)", None),
    ("diversity_ratio", "diversity ratio", 1.0),
]
for ax, (column, label, ideal) in zip(axes, metric_specs):
    ax.plot(sweep["strength"], sweep[column], marker="o", color="#0072B2")
    if ideal is not None:
        ax.axhline(ideal, color="#777777", ls="--", label="target value")
        ax.legend(frameon=False, fontsize=8)
    ax.set(xlabel="noise-alignment strength", ylabel=label)
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
'''),
        code(r'''
baseline_row = noise_alignment_table.set_index("method").loc["baseline"]
aligned_row = noise_alignment_table.set_index("method").loc["noise alignment 1.6"]

control_gain = aligned_row["target_rate"] - baseline_row["target_rate"]
wasserstein_reduction = baseline_row["target_wasserstein"] - aligned_row["target_wasserstein"]
mean_error_reduction = baseline_row["mean_error"] - aligned_row["mean_error"]

alignment_evidence = pd.DataFrame({
    "question": [
        "zero strength reproduces baseline",
        "target-class rate improves by at least 0.15",
        "Wasserstein distance to target decreases by at least 0.20",
        "mean error decreases by at least 0.20",
        "resembles full class-conditional sampling",
    ],
    "observed": [
        zero_strength_error,
        control_gain,
        wasserstein_reduction,
        mean_error_reduction,
        aligned_row["target_rate"],
    ],
    "supported": [
        zero_strength_error < 1e-6,
        control_gain >= 0.15,
        wasserstein_reduction >= 0.20,
        mean_error_reduction >= 0.20,
        aligned_row["target_rate"] >= 0.90 and 0.5 <= aligned_row["diversity_ratio"] <= 2.0,
    ],
})
display(alignment_evidence.round(4))

partial_success = bool(alignment_evidence.loc[1:3, "supported"].all())
full_conditional = bool(alignment_evidence.loc[4, "supported"])
print("partial coarse steering supported:", partial_success)
print("full class-conditional generation supported:", full_conditional)
if partial_success:
    print("WHAT WORKED: target rate increased while target Wasserstein distance and mean error both decreased.")
if not full_conditional:
    print("WHAT DID NOT HAPPEN: off-target mass and excessive spread remain; this is not full class conditioning.")
print("WHAT THE METHOD IS NOT: no point was selected and no point-attraction controller was used.")
'''),
        markdown(r'''
## Interpretation guardrails

- Higher target-class rate measures control, not sample quality.
- Lower Wasserstein distance measures agreement with target examples.
- A diversity ratio far below one indicates collapse; far above one indicates
  excessive spread.
- The model remains unconditional and unchanged. Only small class/full summary
  statistics are prepared offline.
- In high-dimensional image experiments, PCA gives a low-rank covariance model.
  In 2D we keep both principal directions, so the computation is a full Gaussian
  denoiser.
- The intended result is **partial coarse steering**: target rate, Wasserstein distance, and mean
  error should improve together. A high remaining diversity ratio or many
  off-target endpoints rules out a claim of full class-conditional generation.

**Questions**

1. Why should this coarse correction be most useful at high noise?
2. Which strength best balances target rate and target Wasserstein distance in your run?
3. What information is lost when a multimodal full dataset is approximated by one Gaussian?
'''),
    ],
    "04_training_free_gradient_guidance.ipynb": [
        markdown(r'''
# 04 - Post-hoc gradient guidance

The Gaussian correction in notebook `03` is cheap but limited to information
captured by means and covariances. If a differentiable objective describes the
desired class or constraint more directly, its gradient offers a more flexible
local steering direction. This notebook tests that alternative without
retraining the generator.

Here **post-hoc** means that the flow model is frozen. It does not mean the
sampler is free: every active solver evaluation requires a backward pass.
'''),
        code(SETUP),
        code(r'''
model, losses, trained_now = load_or_train_model(device=device)
reference_data, reference_labels = sample_labeled_mixture(30000, device=device)
stats = estimate_gaussian_stats(reference_data, reference_labels)
TARGET_CLASS = 2
'''),
        markdown(r'''
## 1. Why does an objective gradient provide a direction?

The three class Gaussians define `p(class | x)` in clean data space. During
sampling we first compute the model's denoised estimate `D_theta(x_t,t)`, then
differentiate `log p(target | D_theta)` with respect to the current state.

This is not a vector toward a preset coordinate. It is the gradient of a
class-level objective and changes with the state and time.
'''),
        code(r'''
grid_1d = torch.linspace(-4.5, 4.5, 80, device=device)
gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).requires_grad_(True)
posterior = class_log_probabilities(grid, stats).softmax(dim=1)[:, TARGET_CLASS]
gradient = torch.autograd.grad(torch.log(posterior.clamp_min(1e-8)).sum(), grid)[0]

probability_image = posterior.reshape(gx.shape).detach().cpu()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
contour = ax.contourf(gx.cpu(), gy.cpu(), probability_image, levels=20, cmap="viridis")
subset = torch.arange(0, grid.shape[0], 160, device=device)
direction = gradient[subset] / gradient[subset].norm(dim=1, keepdim=True).clamp_min(1e-6)
ax.quiver(
    grid[subset, 0].detach().cpu(), grid[subset, 1].detach().cpu(),
    direction[:, 0].detach().cpu(), direction[:, 1].detach().cpu(),
    color="white", alpha=0.75, scale=18,
)
fig.colorbar(contour, ax=ax, label="target-class posterior")
ax.set_title("Class objective and its gradient")
ax.set_aspect("equal")
plt.show()
'''),
        markdown(r'''
## 2. Compare forward-only and backward-pass guidance

All methods again share the exact same initial noise. Gradient guidance is
active over the middle/late window, where the denoised estimate is informative.
The Gaussian/PCA method is active only at high noise.
'''),
        code(r'''
set_seed(SEED + 40)
x0 = sample_base(1200, device=device)
target_reference = sample_class(1200, TARGET_CLASS, device=device)

methods = {
    "baseline": base_velocity(model),
    "PCA/Gaussian, forward only": make_noise_aligned_velocity(
        model, stats, target_class=TARGET_CLASS, strength=1.6
    ),
    "gradient 1.5": make_gradient_guided_velocity(
        model, stats, target_class=TARGET_CLASS, strength=1.5, start=0.15, end=0.92
    ),
    "gradient 3.0": make_gradient_guided_velocity(
        model, stats, target_class=TARGET_CLASS, strength=3.0, start=0.15, end=0.92
    ),
}

rows = []
paths = {}
for name, velocity in methods.items():
    path, seconds = timed_sample(velocity, x0, steps=48, method="heun")
    metrics = evaluate_samples(path[-1].to(device), target_reference, stats, target_class=TARGET_CLASS)
    rows.append({"method": name, **metrics, "seconds": seconds})
    paths[name] = path

gradient_table = pd.DataFrame(rows)
display(gradient_table.round(4))
'''),
        code(r'''
plot_trajectory_comparison(
    {
        "baseline": paths["baseline"],
        "PCA/Gaussian": paths["PCA/Gaussian, forward only"],
        "gradient": paths["gradient 3.0"],
    },
    target_reference=target_reference[:700],
)
plt.show()

relative = gradient_table.copy()
relative["runtime / baseline"] = relative["seconds"] / relative.loc[0, "seconds"]
display(relative[["method", "target_rate", "target_wasserstein", "diversity_ratio", "runtime / baseline"]].round(3))
'''),
        markdown(r'''
## Cost model

With Heun and 48 steps, the baseline makes 96 forward evaluations. Gradient
guidance can add a backward pass to most of those evaluations. The exact runtime
depends on hardware, but the computational distinction is structural:

| Method | Offline target data | Inference gradients | Main online work |
|---|---:|---:|---|
| Baseline | no | no | model forward passes |
| Gaussian/PCA | yes | no | forwards + small matrix-vector operations |
| Gradient guidance | yes or external classifier | yes | forwards + backward passes |

**Questions**

1. Why can a gradient become unreliable at very high noise?
2. Does the largest target rate also give the best target Wasserstein distance and diversity?
3. Which comparison isolates the price of inference-time backpropagation?
'''),
    ],
    "05_activation_steering_and_method_comparison.ipynb": [
        markdown(r'''
# 05 - Hidden-feature steering and method comparison

Gradient guidance is flexible, but its backward passes can be expensive. A
different possibility is that the frozen generator already stores useful class
information in an internal representation. We first test whether that
information is genuinely readable, then ask whether editing it changes samples.

We collect hidden features from labeled examples after forward noising, learn a
target-vs-rest direction offline, and add that direction during sampling. No
gradient is computed at inference.
'''),
        code(SETUP),
        code(r'''
model, losses, trained_now = load_or_train_model(device=device)
reference_data, reference_labels = sample_labeled_mixture(30000, device=device)
stats = estimate_gaussian_stats(reference_data, reference_labels)
TARGET_CLASS = 2
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
checkpoint_digest_before = model_state_digest(model)
print("frozen checkpoint digest:", checkpoint_digest_before[:16])
'''),
        markdown(r'''
## 1. Why intervene inside the network?

The velocity MLP first converts the current state `(t, x_t)` into a hidden
feature vector $h_t$. The intervention changes that feature before the final
layers compute velocity:

$$h'_t = h_t + \alpha\,g(t)\,\operatorname{RMS}(h_t)\,d_{0.90}.$$

Here $d_{0.90}$ is a unit target-vs-rest direction learned from a labeled
activation training split and checked on separate held-out activations,
$\alpha$ controls strength, and $g(t)$ turns the edit on only in a chosen time
window. The RMS factor puts the unit direction on the scale of the current
layer.

This does **not** pull the sample toward a point. It changes an internal feature,
which may change the model's velocity; the ODE sampler then accumulates those
velocity changes. Model weights stay frozen.
'''),
        code(r'''
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

def activation_box(ax, xy, width, text, color):
    patch = FancyBboxPatch(
        xy, width, 0.18,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=color, edgecolor="#333333", linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + 0.09, text, ha="center", va="center", fontsize=9.5)

def activation_arrow(ax, start, end, color="#333333"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.35, color=color))

fig, ax = plt.subplots(figsize=(12, 4.4))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")

ax.text(0.01, 0.80, "Ordinary forward pass", fontsize=11, weight="bold")
activation_box(ax, (0.03, 0.56), 0.14, "state\n$(t,x_t)$", "#E7E6E6")
activation_box(ax, (0.25, 0.56), 0.16, "hidden feature\n$h_t$", "#CFE2F3")
activation_box(ax, (0.51, 0.56), 0.18, "later MLP\nlayers", "#D9EAD3")
activation_box(ax, (0.79, 0.56), 0.17, "velocity\n$v_\\theta(t,x_t)$", "#EAD1DC")
activation_arrow(ax, (0.17, 0.65), (0.25, 0.65))
activation_arrow(ax, (0.41, 0.65), (0.51, 0.65))
activation_arrow(ax, (0.69, 0.65), (0.79, 0.65))

ax.text(0.01, 0.34, "Steered forward pass", fontsize=11, weight="bold")
activation_box(ax, (0.03, 0.10), 0.14, "same state\n$(t,x_t)$", "#E7E6E6")
activation_box(ax, (0.25, 0.10), 0.16, "same hidden\n$h_t$", "#CFE2F3")
activation_box(ax, (0.48, 0.10), 0.24, "edited feature\n$h'_t=h_t+\\alpha g(t)\\,\\mathrm{RMS}(h_t)d_{0.90}$", "#FCE5CD")
activation_box(ax, (0.79, 0.10), 0.17, "changed\nvelocity", "#EAD1DC")
activation_arrow(ax, (0.17, 0.19), (0.25, 0.19))
activation_arrow(ax, (0.41, 0.19), (0.48, 0.19))
activation_arrow(ax, (0.72, 0.19), (0.79, 0.19))
ax.text(0.60, 0.41, "learned direction $d_{0.90}$", ha="center", fontsize=9.5, color="#D55E00")
activation_arrow(ax, (0.60, 0.39), (0.60, 0.29), color="#D55E00")

fig.suptitle("Activation steering edits a hidden feature, not the sample position", fontsize=14, y=0.98)
plt.tight_layout()
plt.show()
'''),
        markdown(r'''
## 2. When is class information visible in the hidden layer?

For each time, we draw separate balanced training and test datasets, corrupt
them with separate noise draws, and record one hidden layer. A ridge probe is
fit only on the training activations and scored only on the untouched test
activations. Twenty shuffled-training-label probes provide an empirical null.
Their mean is usually near the one-in-three chance rate, but an individual
shuffle can score higher by accidentally assigning output names to separable
clusters. We therefore compare the true probe with the null's 95th percentile,
not with the single largest shuffle. No probe is used to train the generator.
A successful probe shows that class can be read from `h`; it does not show that
editing `h` will control generation.
'''),
        code(r'''
def wilson_interval(successes, total, z=1.96):
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return center - radius, center + radius

probe_rows = []
directions = {}
probes = {}
feature_sets = {}
for time_index, t_value in enumerate([0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.90]):
    set_seed(SEED + 500 + 2 * time_index)
    train_features, train_labels = collect_forward_activations(
        model, t_value=t_value, n_per_class=600
    )
    set_seed(SEED + 501 + 2 * time_index)
    test_features, test_labels = collect_forward_activations(
        model, t_value=t_value, n_per_class=400
    )

    probe = fit_linear_probe(train_features, train_labels)
    train_accuracy = (probe.predict(train_features) == train_labels).float().mean().item()
    test_predictions = probe.predict(test_features)
    test_successes = int((test_predictions == test_labels).sum().item())
    test_accuracy = test_successes / test_labels.numel()
    interval_low, interval_high = wilson_interval(test_successes, test_labels.numel())

    shuffled_accuracies = []
    for null_index in range(20):
        set_seed(SEED + 2000 + 100 * time_index + null_index)
        shuffled_labels = train_labels[torch.randperm(train_labels.numel(), device=device)]
        shuffled_probe = fit_linear_probe(train_features, shuffled_labels)
        shuffled_accuracy = (
            shuffled_probe.predict(test_features) == test_labels
        ).float().mean().item()
        shuffled_accuracies.append(shuffled_accuracy)

    direction = discriminant_direction(train_features, train_labels, TARGET_CLASS)
    directions[t_value] = direction
    probes[t_value] = probe
    feature_sets[t_value] = (train_features, train_labels, test_features, test_labels)
    probe_rows.append({
        "t": t_value,
        "effective noise std": BASE_STD * (1-t_value) / t_value,
        "training accuracy": train_accuracy,
        "held-out accuracy": test_accuracy,
        "95% interval low": interval_low,
        "95% interval high": interval_high,
        "shuffled mean": float(np.mean(shuffled_accuracies)),
        "shuffled std": float(np.std(shuffled_accuracies)),
        "shuffled 95th percentile": float(np.quantile(shuffled_accuracies, 0.95)),
        "shuffled max": float(np.max(shuffled_accuracies)),
    })

probe_table = pd.DataFrame(probe_rows)
display(probe_table.round(3))

fig, ax = plt.subplots(figsize=(7, 3.8))
lower_error = np.maximum(
    0.0, probe_table["held-out accuracy"] - probe_table["95% interval low"]
)
upper_error = np.maximum(
    0.0, probe_table["95% interval high"] - probe_table["held-out accuracy"]
)
ax.errorbar(
    probe_table["t"], probe_table["held-out accuracy"],
    yerr=np.vstack([lower_error, upper_error]), marker="o", capsize=3,
    color="#0072B2", label="held-out true-label probe",
)
ax.plot(probe_table["t"], probe_table["shuffled mean"], color="#D55E00", label="20 shuffled-label probes")
ax.fill_between(
    probe_table["t"],
    probe_table["shuffled mean"] - probe_table["shuffled std"],
    probe_table["shuffled mean"] + probe_table["shuffled std"],
    color="#D55E00", alpha=0.18,
)
ax.axhline(1 / len(CLASS_NAMES), color="#777777", ls="--", label="chance")
ax.set(xlabel="t (noise -> data)", ylabel="held-out accuracy", ylim=(0.25, 1.02), title="Class information across time")
ax.annotate("near chance", (0.10, probe_table.loc[0, "held-out accuracy"]), xytext=(8, 14), textcoords="offset points")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
plt.show()
'''),
        code(r'''
# Fit display PCA only on training activations, then transform held-out activations.
REFERENCE_T = 0.90
train_features, train_labels, test_features, test_labels = feature_sets[REFERENCE_T]
pca_display = fit_pca_projection(train_features)
projected_test = pca_display.transform(test_features)

fig, ax = plt.subplots(figsize=(6, 5))
plot_labeled_points(
    projected_test, test_labels, ax=ax,
    title=f"Held-out activations in train-fit PCA coordinates at t={REFERENCE_T}", alpha=0.35,
)
ax.legend(frameon=False)
plt.show()

reference_predictions = probes[REFERENCE_T].predict(test_features)
confusion = torch.bincount(
    test_labels * len(CLASS_NAMES) + reference_predictions,
    minlength=len(CLASS_NAMES) ** 2,
).reshape(len(CLASS_NAMES), len(CLASS_NAMES)).float()
confusion_fraction = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
image = ax.imshow(confusion_fraction.cpu(), vmin=0, vmax=1, cmap="Blues")
for row in range(len(CLASS_NAMES)):
    for column in range(len(CLASS_NAMES)):
        value = float(confusion_fraction[row, column].cpu())
        ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black")
ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
ax.set(xlabel="predicted class", ylabel="true class", title=f"Held-out probe confusion at t={REFERENCE_T}")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

reference_direction = directions[REFERENCE_T]
cosine_table = pd.DataFrame([
    {"t": t_value, f"cosine with t={REFERENCE_T:.2f} direction": float(torch.dot(direction, reference_direction).cpu())}
    for t_value, direction in directions.items()
])
display(cosine_table.round(3))

cosine_column = f"cosine with t={REFERENCE_T:.2f} direction"
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.plot(cosine_table["t"], cosine_table[cosine_column], marker="o", lw=2, color="#0072B2")
ax.axhline(0, color="#777777", lw=1, ls="--")
ax.axvspan(0.40, 0.90, color="#E69F00", alpha=0.16, label="activation-only window")
ax.scatter([REFERENCE_T], [1.0], s=90, color="#D55E00", zorder=3, label="$d_{0.90}$ reference")
ax.set(
    xlabel="t (noise to data)", ylabel="cosine similarity",
    title="A late activation direction is not the local direction at every time",
    xlim=(0.05, 0.95), ylim=(-0.35, 1.08),
)
ax.grid(alpha=0.2)
ax.legend(frameon=False, loc="lower right")
plt.show()
'''),
        markdown(r'''
### Reading fixed-direction transfer

We estimate $d_{0.90}$ at late time and reuse it as a fixed intervention vector
over the calibrated window $[0.40, 0.90]$. Because time-specific directions are
only weakly aligned with $d_{0.90}$ early in that window -- for example, cosine
similarity is 0.177 at $t=0.40$ -- we do not interpret the intervention as
following the local class direction at every time. The result shows that a
late-time decodable direction can be causally useful as part of an
endpoint-tuned control schedule; it does not show temporal invariance of the
representation, RFM/AGOP, or full NA-RFM.
'''),
        code(r'''
def direction_statistics(direction):
    scores = test_features @ direction
    positive = scores[test_labels == TARGET_CLASS]
    negative = scores[test_labels != TARGET_CLASS]
    gap = positive.mean() - negative.mean()
    pooled_std = torch.sqrt(0.5 * (positive.var(unbiased=False) + negative.var(unbiased=False))).clamp_min(1e-8)
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=device, dtype=torch.float32)
    n_positive = positive.numel()
    n_negative = negative.numel()
    auc = (
        ranks[test_labels == TARGET_CLASS].sum() - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)
    return float(gap.cpu()), float((gap / pooled_std).cpu()), float(auc.cpu())

shuffled_directions = []
null_direction_rows = []
for null_index in range(50):
    set_seed(SEED + 4000 + null_index)
    shuffled_train_labels = train_labels[torch.randperm(train_labels.numel(), device=device)]
    direction = discriminant_direction(train_features, shuffled_train_labels, TARGET_CLASS)
    shuffled_directions.append(direction)
    raw_gap, normalized_gap, auc = direction_statistics(direction)
    null_direction_rows.append({
        "shuffle": null_index,
        "held-out raw gap": raw_gap,
        "held-out normalized gap": normalized_gap,
        "held-out AUC": auc,
    })

shuffled_direction = shuffled_directions[0]
true_raw_gap, true_normalized_gap, true_auc = direction_statistics(reference_direction)
null_direction_table = pd.DataFrame(null_direction_rows)
direction_summary = pd.DataFrame({
    "direction": ["true labels", "50 shuffled-label mean", "largest shuffled-label gap"],
    "normalized held-out gap": [
        true_normalized_gap,
        null_direction_table["held-out normalized gap"].mean(),
        null_direction_table["held-out normalized gap"].max(),
    ],
    "held-out AUC": [
        true_auc,
        null_direction_table["held-out AUC"].mean(),
        null_direction_table.loc[null_direction_table["held-out normalized gap"].idxmax(), "held-out AUC"],
    ],
})
display(direction_summary.round(3))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(null_direction_table["held-out normalized gap"], bins=12, color="#999999", alpha=0.75, label="50 shuffled-label directions")
ax.axvline(true_normalized_gap, color="#D55E00", lw=2.5, label="true-label direction")
ax.set(xlabel="normalized target-vs-rest gap on held-out activations", ylabel="count", title="Direction null distribution")
ax.legend(frameon=False)
plt.show()

null_99 = float(null_direction_table["held-out normalized gap"].quantile(0.99))
reference_row = probe_table.set_index("t").loc[REFERENCE_T]
probe_evidence = pd.DataFrame({
    "check": [
        "held-out probe accuracy is above 0.80",
        "held-out accuracy beats the shuffled-label 95th percentile by 0.20",
        "true direction exceeds 99th percentile of shuffled directions",
        "true-direction held-out AUC is above 0.80",
    ],
    "observed": [
        reference_row["held-out accuracy"],
        reference_row["held-out accuracy"] - reference_row["shuffled 95th percentile"],
        true_normalized_gap - null_99,
        true_auc,
    ],
    "supported": [
        reference_row["held-out accuracy"] >= 0.80,
        reference_row["held-out accuracy"] - reference_row["shuffled 95th percentile"] >= 0.20,
        true_normalized_gap > null_99,
        true_auc >= 0.80,
    ],
})
display(probe_evidence.round(3))
label_direction_separated = bool(probe_evidence["supported"].all())
print("held-out class-decodability supported:", label_direction_separated)
print("PCA is display-only; the held-out probe and null tests are the decoding evidence.")
'''),
        markdown(r'''
## 3. Activation intervention

The target direction is covariance-aware: it separates target examples from
the remaining classes in hidden space. During an active time window, the model
adds a scaled unit direction to the selected hidden layer. The scale is relative
to the current feature RMS, so it remains comparable across time.

This is a genuine hidden-feature intervention, not sample-space attraction. It
is still simpler than NA-RFM: the paper uses RFM directions in image-model
activation tensors, while this notebook uses a linear discriminant direction in
one MLP layer.

We collect the direction at `t = 0.90`, where the forward activations are close
to clean data and strongly class-decodable. Collection time and intervention
window are different choices: a small paired calibration on separate noise
seeds selected the settings below by target Wasserstein distance. The activation-only and
two-stage methods are calibrated separately because early noise alignment
changes the trajectories seen by the later activation edit.
'''),
        code(r'''
set_seed(SEED + 50)
x0 = sample_base(1400, device=device)
target_reference = sample_class(1400, TARGET_CLASS, device=device)

ACTIVATION_STRENGTH = 7.0
ACTIVATION_WINDOW = (0.40, 0.90)
COMBINED_ACTIVATION_STRENGTH = 5.0
COMBINED_ACTIVATION_WINDOW = (0.35, 0.85)

methods = {
    "baseline": base_velocity(model),
    "activation 0.0": make_activation_steered_velocity(
        model, reference_direction, strength=0.0,
        start=ACTIVATION_WINDOW[0], end=ACTIVATION_WINDOW[1],
    ),
    "shuffled activation": make_activation_steered_velocity(
        model, shuffled_direction, strength=ACTIVATION_STRENGTH,
        start=ACTIVATION_WINDOW[0], end=ACTIVATION_WINDOW[1],
    ),
    "activation steering": make_activation_steered_velocity(
        model, reference_direction, strength=ACTIVATION_STRENGTH,
        start=ACTIVATION_WINDOW[0], end=ACTIVATION_WINDOW[1],
    ),
    "noise alignment": make_noise_aligned_velocity(
        model, stats, target_class=TARGET_CLASS, strength=1.6
    ),
    "gradient guidance": make_gradient_guided_velocity(
        model, stats, target_class=TARGET_CLASS, strength=3.0, start=0.15, end=0.92
    ),
    "noise + activation": make_combined_velocity(
        model, stats, reference_direction,
        target_class=TARGET_CLASS, noise_strength=1.6,
        activation_strength=COMBINED_ACTIVATION_STRENGTH,
        activation_window=COMBINED_ACTIVATION_WINDOW,
    ),
}

rows = []
paths = {}
for name, velocity in methods.items():
    path, seconds = timed_sample(velocity, x0, steps=56, method="heun")
    metrics = evaluate_samples(path[-1].to(device), target_reference, stats, target_class=TARGET_CLASS)
    rows.append({"method": name, **metrics, "seconds": seconds})
    paths[name] = path

comparison_table = pd.DataFrame(rows)
comparison_table["runtime / baseline"] = comparison_table["seconds"] / comparison_table.loc[0, "seconds"]
zero_activation_error = float(torch.max(torch.abs(paths["baseline"][-1] - paths["activation 0.0"][-1])))
assert zero_activation_error < 1e-6
assert model_state_digest(model) == checkpoint_digest_before

null_control_table = comparison_table[
    comparison_table["method"].isin(["baseline", "activation 0.0", "shuffled activation", "activation steering"])
]
method_table = comparison_table[
    comparison_table["method"].isin(["baseline", "noise alignment", "gradient guidance", "activation steering", "noise + activation"])
]
print(f"paired endpoint max difference, baseline vs activation 0.0: {zero_activation_error:.2e}")
print("checkpoint digest unchanged after every rollout: True")
display(null_control_table.round(4))
display(method_table.round(4))
'''),
        code(r'''
plot_trajectory_comparison(
    {
        "baseline": paths["baseline"],
        "shuffled direction": paths["shuffled activation"],
        "activation steering": paths["activation steering"],
        "noise + activation": paths["noise + activation"],
    },
    target_reference=target_reference[:700],
    n_lines=18,
)
plt.show()

fig, ax = plt.subplots(figsize=(7.5, 5))
marker_by_method = {
    "baseline": "o", "shuffled activation": "s",
    "activation steering": "D", "noise alignment": "^", "gradient guidance": "v",
    "noise + activation": "P",
}
label_offsets = {
    "baseline": (-52, -12),
    "shuffled activation": (5, 8),
    "activation steering": (5, 4),
    "noise alignment": (5, 4),
    "gradient guidance": (5, 4),
    "noise + activation": (5, -2),
}
tradeoff_table = comparison_table[comparison_table["method"] != "activation 0.0"]
for _, row in tradeoff_table.iterrows():
    ax.scatter(row["target_wasserstein"], row["target_rate"], s=80, marker=marker_by_method[row["method"]])
    ax.annotate(
        row["method"], (row["target_wasserstein"], row["target_rate"]),
        xytext=label_offsets[row["method"]], textcoords="offset points", fontsize=9,
    )
ax.set(xlabel="Wasserstein distance to target class (lower is better)", ylabel="target-class rate (higher is better)", title="Control-quality trade-off")
ax.grid(alpha=0.2)
plt.show()
'''),
        code(r'''
method_rows = comparison_table.set_index("method")
activation_gain = method_rows.loc["activation steering", "target_rate"] - method_rows.loc["baseline", "target_rate"]
shuffled_gain = method_rows.loc["shuffled activation", "target_rate"] - method_rows.loc["baseline", "target_rate"]
label_specific_margin = activation_gain - shuffled_gain
activation_wasserstein_reduction = method_rows.loc["baseline", "target_wasserstein"] - method_rows.loc["activation steering", "target_wasserstein"]
best_control_method = comparison_table.loc[comparison_table["target_rate"].idxmax(), "method"]
best_target_match_method = comparison_table.loc[comparison_table["target_wasserstein"].idxmin(), "method"]

intervention_evidence = pd.DataFrame({
    "check": [
        "activation-only target-rate gain at least 0.10",
        "activation-only Wasserstein distance decreases",
        "true direction beats shuffled-direction gain by at least 0.05",
        "combined method has strongest target control",
    ],
    "observed": [activation_gain, activation_wasserstein_reduction, label_specific_margin, best_control_method],
    "supported": [
        activation_gain >= 0.10,
        activation_wasserstein_reduction > 0,
        label_specific_margin >= 0.05 and label_direction_separated,
        best_control_method == "noise + activation",
    ],
})
display(intervention_evidence.round(4))
print("strongest target control:", best_control_method)
print("lowest Wasserstein distance to target examples:", best_target_match_method)
print("activation-only paired improvement supported:", bool(intervention_evidence.loc[:1, "supported"].all()))
label_specific_steering = bool(intervention_evidence.loc[2, "supported"])
print("label-specific activation steering supported:", label_specific_steering)
if not label_specific_steering:
    print("claim downgrade: the generator is sensitive to activation perturbations, but label specificity is not established.")
'''),
        markdown(r'''
## 4. Further comparisons

To explore sensitivity, vary one factor at a time while keeping the paired
protocol fixed:

1. **Collection time:** learn directions at `t = 0.40, 0.55, 0.70, 0.85, 0.90`.
2. **Intervention window:** early, middle, or late with fixed direction/strength.
3. **Activation strength:** at least three values with fixed direction/window.
4. **Method:** baseline, noise alignment, gradient, activation, and combined.
5. **Robustness:** repeat the decisive comparison over three initial-noise seeds.

The useful quantities to compare are target-class rate, target Wasserstein distance, diversity
ratio, runtime, and paired trajectories. A method is not "best" without stating
the quality constraint used for comparison.

**Claim boundaries**

- Supported: class is linearly decodable from this hidden layer on held-out
  examples, especially at later `t`.
- Conditional: the intervention is label-specific only when the true direction
  beats the matched shuffled-label direction in the paired sampling test.
- Not supported: class lives in one unique direction, the feature is causally
  necessary, `d_0.90` equals the local target direction at every intervention
  time, hidden representations are temporally invariant, or this linear
  discriminant is the RFM component of full NA-RFM.

## Paper bridge

| NA-RFM component | Toy implementation | Important limitation |
|---|---|---|
| High-noise noise alignment | class Gaussian denoiser minus full Gaussian denoiser | 2D full covariance, not image PCA |
| Forward activation collection | labeled points corrupted to a reference `t` | one small MLP layer |
| Target direction | covariance-aware linear discriminant | not Recursive Feature Machine |
| Online activation edit | fixed feature direction over a time window | no U-Net feature map or amplification pass |
| Gradient-free inference | yes for noise/activation methods | gradient baseline still backpropagates |

**Questions for discussion**

1. Does probe accuracy predict steering success at the same collection time?
2. Over which times is the learned direction stable enough to reuse?
3. When does combining coarse high-noise and later activation control help?
4. Which conclusion from 2D would require a new experiment before claiming it for images?
'''),
    ],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = OUT / name
        path.write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
