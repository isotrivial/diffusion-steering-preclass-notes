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

This notebook builds the visual vocabulary used by the remaining notes. The
audience is expected to know Python and vectors, but not stochastic calculus.

**Learning goals**

1. Recognize a forward noising path and a reverse generative path.
2. Read a time-dependent vector field as arrows that move points.
3. Explain the flow-matching training pair `(x_t, x_data - x_noise)`.
4. Distinguish a model, a denoiser, and a numerical sampler.

The toy dataset has three labeled clusters, but the generative model in later
notebooks will be trained **without labels**. Labels are retained only to test
post-hoc steering.
'''),
        code(SETUP),
        markdown(r'''
## 1. A dataset we can see

Images live in thousands or millions of dimensions. Here every sample has only
two coordinates, so a scatter plot shows the complete data distribution. The
three colors play the role of semantic classes.
'''),
        code(r'''
clean, labels = sample_labeled_mixture(2400, device=device)
fig, ax = plt.subplots(figsize=(6, 5))
plot_labeled_points(clean, labels, ax=ax, title="Clean toy data")
ax.legend(frameon=False)
plt.show()
'''),
        markdown(r'''
## 2. Forward noising

We use the linear path

$$x_t = t\,x_{data} + (1-t)\,\epsilon.$$

Here `epsilon` is Gaussian with standard deviation `s=1.8` in each coordinate.

- At `t=1`, the point is clean data.
- At `t=0`, the point is Gaussian noise.
- Lower `t` therefore means higher noise in this notebook series.

A diffusion model learns to undo a noising process. Flow matching learns the
velocity of a path between the same endpoints.
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
## 3. What flow matching predicts

Choose a noise point `x_noise`, a data point `x_data`, and a time `t`. Along the
straight conditional path, the target velocity is simply

$$u_t = x_{data} - x_{noise}.$$

Training repeatedly asks a neural network to predict this arrow from only
`(x_t, t)`. Because many endpoint pairs can pass near the same `x_t`, the
network learns an average velocity field, not a memorized arrow for each pair.
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
## 4. Four objects that must not be confused

| Object | Job | Example in this series |
|---|---|---|
| Noising path | Defines intermediate training states | `x_t = t*x_data + (1-t)*noise` |
| Neural model | Predicts a local quantity | velocity `v_theta(t, x_t)` |
| Denoiser | Estimates clean data from a noisy state | `D_theta(x_t,t)` |
| Sampler | Numerically follows model predictions | Euler, Heun, or RK4 |

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

We now train a small MLP to transport Gaussian noise into the three-cluster data
distribution. The model never receives a class label.

**Learning goals**

1. Implement independent conditional flow matching (I-CFM).
2. Sample the learned ODE with a deterministic solver.
3. Diagnose distribution quality with plots and metrics, not loss alone.
4. Save one checkpoint used by the later steering notebooks.
'''),
        code(SETUP),
        markdown(r'''
## 1. The complete training objective

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
print("checkpoint:", CHECKPOINT_PATH)
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
## 2. Generate samples by solving an ODE

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
    "diagnostic": ["sliced Wasserstein to full target", "generated class balance max error", "all samples finite"],
    "value": [
        sliced_wasserstein(generated, target),
        float(torch.max(torch.abs(generated_balance - 1 / len(CLASS_NAMES))).cpu()),
        bool(torch.isfinite(generated).all()),
    ],
})
display(diagnostics)
print("generated class balance:", generated_balance.cpu().numpy().round(3))
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
## Checkpoint

Before continuing, verify that:

- the loss is finite and decreases;
- generated points cover all three modes;
- the endpoint plot resembles the target distribution;
- `artifacts/toy_flow_model.pt` exists.

**Questions**

1. Why can a low minibatch MSE coexist with visibly poor samples?
2. What information about class identity did the unconditional model receive?
3. At which time does the vector field mostly choose coarse destination modes?
'''),
    ],
    "02_denoisers_and_deterministic_samplers.ipynb": [
        markdown(r'''
# 02 - Denoisers and deterministic samplers

This notebook connects three views of the same linear flow: velocity prediction,
clean-data prediction, and numerical ODE integration.

**Learning goals**

1. Convert a velocity prediction into a denoised estimate.
2. Visualize what the model believes the final clean sample will be over time.
3. Compare Euler, Heun, and RK4 under a fixed model and fixed initial noise.
4. Explain why paired initial noise is essential for deterministic comparisons.
'''),
        code(SETUP),
        code(r'''
model, losses, trained_now = load_or_train_model(device=device)
print("trained in this run:", trained_now)
'''),
        markdown(r'''
## 1. Velocity and denoiser are two parameterizations

Along the linear conditional path,

$$x_t=x_{noise}+t(x_{data}-x_{noise}).$$

If a model predicts the velocity `v_theta`, its clean-data estimate is

$$D_\theta(x_t,t)=x_t+(1-t)v_\theta(x_t,t).$$

This identity is specific to our linear path. It lets us inspect a denoiser even
though the network was trained to output velocity.
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
            "SWD to target": sliced_wasserstein(path[-1], target.cpu()),
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

We now add the first post-hoc class control. There is no hand-written attraction
toward a coordinate. The guidance signal is the difference
between two denoisers estimated from labeled examples:

$$\Delta D = D_{target\ class} - D_{full\ data}.$$

This is the 2D counterpart of the high-noise noise-alignment stage in NA-RFM.

**Learning goals**

1. Understand a Gaussian denoiser as a noise-dependent shrinkage rule.
2. Visualize class-conditional versus unconditional denoising.
3. Apply their difference only in the high-noise sampling window.
4. Evaluate class control, target fidelity, diversity, and cost separately.
'''),
        code(SETUP),
        code(r'''
model, losses, trained_now = load_or_train_model(device=device)
reference_data, reference_labels = sample_labeled_mixture(30000, device=device)
stats = estimate_gaussian_stats(reference_data, reference_labels)
TARGET_CLASS = 2
print("target class:", CLASS_NAMES[TARGET_CLASS])
'''),
        markdown(r'''
## 1. Why a Gaussian denoiser makes sense at high noise

Write the base point as $x_{noise}=s\epsilon$ with
$\epsilon\sim\mathcal{N}(0,I)$ and `s=1.8`, then divide the path by `t`:

$$\frac{x_t}{t}=x_{data}+s\frac{1-t}{t}\epsilon.$$

The right side is clean data plus Gaussian noise with effective noise level
`sigma_eff=s(1-t)/t`. For a Gaussian data model, the posterior mean has a closed
form. The factor `s` matters because our base distribution is not unit variance.
At high noise the denoiser trusts coarse mean/covariance statistics; at low noise
it stays close to the observed point.

The full-data Gaussian is deliberately crude because the complete dataset is a
mixture. That limitation is useful: subtracting the full denoiser from the
target-class denoiser isolates a coarse class correction.
'''),
        code(r'''
grid_1d = torch.linspace(-4.5, 4.5, 19, device=device)
gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
t_value = 0.18
t = torch.full((grid.shape[0],), t_value, device=device)
delta = noise_alignment_delta(grid, t, TARGET_CLASS, stats)

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.quiver(
    grid[:, 0].cpu(), grid[:, 1].cpu(), delta[:, 0].cpu(), delta[:, 1].cpu(),
    angles="xy", scale_units="xy", scale=1.7, width=0.004, color=CLASS_COLORS[TARGET_CLASS], alpha=0.8,
)
plot_labeled_points(reference_data[:2500], reference_labels[:2500], ax=ax, title=f"High-noise PCA/Gaussian correction for '{CLASS_NAMES[TARGET_CLASS]}'", alpha=0.18)
ax.set_xlim(-4.8, 4.8)
ax.set_ylim(-4.8, 4.8)
plt.show()
'''),
        markdown(r'''
## 2. Paired class-steering experiment

All methods below use the same initial noise, Heun solver, step count, and target
reference. The only change is the noise-alignment strength. The correction is
active for `t in [0.04, 0.35]`, the high-noise part of our noise-to-data path.
'''),
        code(r'''
set_seed(SEED + 30)
x0 = sample_base(1800, device=device)
target_reference = sample_class(1800, TARGET_CLASS, device=device)

methods = {"baseline": base_velocity(model)}
for strength in [0.4, 0.8, 1.2, 1.6]:
    methods[f"noise alignment {strength:.1f}"] = make_noise_aligned_velocity(
        model, stats, target_class=TARGET_CLASS, strength=strength, start=0.04, end=0.35
    )

rows = []
paths = {}
for name, velocity in methods.items():
    path, seconds = timed_sample(velocity, x0, steps=64, method="heun")
    metrics = evaluate_samples(path[-1].to(device), target_reference, stats, target_class=TARGET_CLASS)
    rows.append({"method": name, **metrics, "seconds": seconds})
    paths[name] = path

noise_alignment_table = pd.DataFrame(rows)
display(noise_alignment_table.round(4))
'''),
        code(r'''
plot_trajectory_comparison(
    {"baseline": paths["baseline"], "noise alignment 1.2": paths["noise alignment 1.2"]},
    target_class=TARGET_CLASS,
)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(noise_alignment_table["method"], noise_alignment_table["target_rate"], marker="o", color="#0072B2")
axes[0].set_ylabel("target-class rate")
axes[1].plot(noise_alignment_table["method"], noise_alignment_table["target_swd"], marker="o", color="#D55E00")
axes[1].set_ylabel("SWD to target class")
for ax in axes:
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
'''),
        markdown(r'''
## Interpretation guardrails

- Higher target-class rate measures control, not sample quality.
- Lower SWD measures agreement with target examples.
- A diversity ratio far below one indicates collapse; far above one indicates
  excessive spread.
- The model remains unconditional and unchanged. Only small class/full summary
  statistics are prepared offline.
- In high-dimensional image experiments, PCA gives a low-rank covariance model.
  In 2D we keep both principal directions, so the computation is a full Gaussian
  denoiser.

**Questions**

1. Why should this coarse correction be most useful at high noise?
2. Which strength best balances target rate and target SWD in your run?
3. What information is lost when a multimodal full dataset is approximated by one Gaussian?
'''),
    ],
    "04_training_free_gradient_guidance.ipynb": [
        markdown(r'''
# 04 - Post-hoc gradient guidance

This notebook implements a standard alternative to retraining a conditional
generator: differentiate a target-class objective during sampling.

Here **post-hoc** means that the flow model is frozen. It does not mean the
sampler is free: every active solver evaluation requires a backward pass.

**Learning goals**

1. Read a classifier log probability as an energy landscape.
2. Use its gradient as the local direction of fastest class-probability increase.
3. Compare gradient guidance with forward-only Gaussian/PCA guidance.
4. Report control, fidelity, diversity, and runtime together.
'''),
        code(SETUP),
        code(r'''
model, losses, trained_now = load_or_train_model(device=device)
reference_data, reference_labels = sample_labeled_mixture(30000, device=device)
stats = estimate_gaussian_stats(reference_data, reference_labels)
TARGET_CLASS = 2
'''),
        markdown(r'''
## 1. A clean-data class energy

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
    target_class=TARGET_CLASS,
)
plt.show()

relative = gradient_table.copy()
relative["runtime / baseline"] = relative["seconds"] / relative.loc[0, "seconds"]
display(relative[["method", "target_rate", "target_swd", "diversity_ratio", "runtime / baseline"]].round(3))
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
2. Does the largest target rate also give the best target SWD and diversity?
3. Which comparison isolates the price of inference-time backpropagation?
'''),
    ],
    "05_activation_steering_and_method_comparison.ipynb": [
        markdown(r'''
# 05 - Hidden-feature steering and method comparison

This note moves the intervention from 2D sample space into a hidden
layer of the unconditional velocity network.

We collect hidden features from labeled examples after forward noising, learn a
target-vs-rest direction offline, and add that direction during sampling. No
gradient is computed at inference.

**Learning goals**

1. Measure when hidden activations become class-informative.
2. Learn and visualize a class direction in feature space.
3. Test whether one direction transfers across nearby timesteps.
4. Compare baseline, noise alignment, gradient guidance, activation steering,
   and a two-stage combination under paired conditions.
'''),
        code(SETUP),
        code(r'''
model, losses, trained_now = load_or_train_model(device=device)
reference_data, reference_labels = sample_labeled_mixture(30000, device=device)
stats = estimate_gaussian_stats(reference_data, reference_labels)
TARGET_CLASS = 2
'''),
        markdown(r'''
## 1. When is class information visible in the hidden layer?

For each time, we corrupt labeled data with the forward process, record one
hidden layer, and fit a small ridge linear probe. High probe accuracy means the
activation contains linearly readable class information. The probe is a
measurement tool; it is not used by the generative model during training.
'''),
        code(r'''
probe_rows = []
directions = {}
for t_value in [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]:
    features, labels = collect_forward_activations(model, t_value=t_value, n_per_class=700)
    permutation = torch.randperm(features.shape[0], device=device)
    split = int(0.7 * features.shape[0])
    train_idx, test_idx = permutation[:split], permutation[split:]
    probe = fit_linear_probe(features[train_idx], labels[train_idx])
    accuracy = (probe.predict(features[test_idx]) == labels[test_idx]).float().mean().item()
    direction = discriminant_direction(features[train_idx], labels[train_idx], TARGET_CLASS)
    directions[t_value] = direction
    probe_rows.append({
        "t": t_value,
        "effective noise std": BASE_STD * (1-t_value) / t_value,
        "linear probe accuracy": accuracy,
    })

probe_table = pd.DataFrame(probe_rows)
display(probe_table.round(3))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(probe_table["t"], probe_table["linear probe accuracy"], marker="o", color="#0072B2")
ax.axhline(1 / len(CLASS_NAMES), color="#777777", ls="--", label="chance")
ax.set(xlabel="t (noise -> data)", ylabel="held-out accuracy", ylim=(0.25, 1.02), title="Class information across time")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
plt.show()
'''),
        code(r'''
# Visualize the hidden features at one reference time using PCA only for display.
REFERENCE_T = 0.70
features, labels = collect_forward_activations(model, t_value=REFERENCE_T, n_per_class=700)
projected = pca_project(features)

fig, ax = plt.subplots(figsize=(6, 5))
plot_labeled_points(projected, labels, ax=ax, title=f"Hidden activations projected to 2D at t={REFERENCE_T}", alpha=0.35)
ax.legend(frameon=False)
plt.show()

reference_direction = directions[REFERENCE_T]
cosines = []
for t_value, direction in directions.items():
    cosines.append({"t": t_value, "cosine with reference direction": float(torch.dot(direction, reference_direction).cpu())})
cosine_table = pd.DataFrame(cosines)
display(cosine_table.round(3))

# Null control: destroy the feature-label relationship before fitting a direction.
shuffled_labels = labels[torch.randperm(labels.shape[0], device=device)]
shuffled_direction = discriminant_direction(features, shuffled_labels, TARGET_CLASS)

def true_label_score_gap(direction):
    scores = features @ direction
    return float((scores[labels == TARGET_CLASS].mean() - scores[labels != TARGET_CLASS].mean()).cpu())

null_control = pd.DataFrame({
    "direction": ["true labels", "shuffled labels"],
    "target-vs-rest score gap": [
        true_label_score_gap(reference_direction),
        true_label_score_gap(shuffled_direction),
    ],
})
display(null_control.round(3))
assert null_control.loc[0, "target-vs-rest score gap"] > 0
'''),
        markdown(r'''
## 2. Activation intervention

The target direction is covariance-aware: it separates target examples from
the remaining classes in hidden space. During an active time window, the model
adds a scaled unit direction to the selected hidden layer. The scale is relative
to the current feature RMS, so it remains comparable across time.

This is a genuine hidden-feature intervention, not sample-space attraction. It
is still simpler than NA-RFM: the paper uses RFM directions in image-model
activation tensors, while this notebook uses a linear discriminant direction in
one MLP layer.
'''),
        code(r'''
set_seed(SEED + 50)
x0 = sample_base(1400, device=device)
target_reference = sample_class(1400, TARGET_CLASS, device=device)

methods = {
    "baseline": base_velocity(model),
    "noise alignment": make_noise_aligned_velocity(
        model, stats, target_class=TARGET_CLASS, strength=1.6
    ),
    "gradient guidance": make_gradient_guided_velocity(
        model, stats, target_class=TARGET_CLASS, strength=3.0, start=0.15, end=0.92
    ),
    "activation steering": make_activation_steered_velocity(
        model, reference_direction, strength=4.0, start=0.38, end=0.90
    ),
    "noise + activation": make_combined_velocity(
        model, stats, reference_direction,
        target_class=TARGET_CLASS, noise_strength=1.6, activation_strength=4.0,
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
display(comparison_table.round(4))
'''),
        code(r'''
plot_trajectory_comparison(
    {
        "baseline": paths["baseline"],
        "noise alignment": paths["noise alignment"],
        "activation steering": paths["activation steering"],
        "noise + activation": paths["noise + activation"],
    },
    target_class=TARGET_CLASS,
    n_lines=18,
)
plt.show()

fig, ax = plt.subplots(figsize=(7.5, 5))
for _, row in comparison_table.iterrows():
    ax.scatter(row["target_swd"], row["target_rate"], s=80)
    ax.annotate(row["method"], (row["target_swd"], row["target_rate"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
ax.set(xlabel="SWD to target class (lower is better)", ylabel="target-class rate (higher is better)", title="Control-quality trade-off")
ax.grid(alpha=0.2)
plt.show()
'''),
        markdown(r'''
## 3. Further comparisons

To explore sensitivity, vary one factor at a time while keeping the paired
protocol fixed:

1. **Collection time:** learn directions at `t = 0.40, 0.55, 0.70, 0.85`.
2. **Intervention window:** early, middle, or late with fixed direction/strength.
3. **Activation strength:** at least three values with fixed direction/window.
4. **Method:** baseline, noise alignment, gradient, activation, and combined.
5. **Robustness:** repeat the decisive comparison over three initial-noise seeds.

The useful quantities to compare are target-class rate, target SWD, diversity
ratio, runtime, and paired trajectories. A method is not "best" without stating
the quality constraint used for comparison.

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
