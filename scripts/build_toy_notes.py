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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from toy_notes import *
from notebook_views import *

SEED = 2026
set_seed(SEED)
device = choose_device()
print("device:", device)
'''


NOTEBOOKS = {
    "00_diffusion_and_flow_matching_foundations.ipynb": [
        markdown(r'''
# 00 - From noise to data: diffusion and flow-matching foundations

Every note in this series follows one question: how can a generator turn fresh
noise into a new sample, and where can we intervene after training? We begin in
two dimensions, where the data distribution, Gaussian source, learned velocity
field, and numerical sampler can all be drawn. The model is unconditional:
labels are withheld from its training input. Later steering signals will come
from fitted class distributions, differentiable objectives, or hidden-feature
directions, never from attraction to a chosen target point.

Python, vectors, and basic probability are enough background; no stochastic
calculus is assumed.
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
plot_generator_pipeline()
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
    t = torch.full((clean_small.shape[0], 1), t_value, device=device)
    noisy = (1 - t) * fixed_noise + t * clean_small
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
x_data, _ = sample_labeled_mixture(24, device=device)
x_noise = sample_base(24, device=device)
t_value = 0.42
t = torch.full((24, 1), t_value, device=device)
x_t = (1 - t) * x_noise + t * x_data
target_velocity = x_data - x_noise

plot_flow_matching_batch(x_noise, x_data, x_t, target_velocity)
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

**Check your understanding**

1. What is visible at high noise: exact class details or only coarse structure?
2. Why is `x_data - x_noise` available during training but unavailable during generation?
3. Which component changes when Euler is replaced by RK4: the trained model or the sampler?

The key separation is now in place: the data define what must be learned, the
base distribution supplies diverse starting points, the network predicts a
local velocity, and the sampler turns those predictions into complete
trajectories. None of this yet shows that the learned field works. Notebook
`01` makes the construction concrete by training the velocity model and
watching the same learning rule succeed, and sometimes fail, across clustered,
curved, and many-mode distributions.
'''),
    ],
    "01_train_unconditional_flow_matching.ipynb": [
        markdown(r'''
# 01 - Train an unconditional flow-matching model

Notebook `00` separated the source distribution, learned field, and sampler. We
now train the learned part: a small unconditional MLP that predicts velocity
along independently paired noise-to-data paths. The goal is not merely to lower
a loss curve, but to see whether local velocity predictions assemble into a
convincing global transport. We begin with three clusters, then reuse the same
training idea on a noisy circle and eight separated modes so that curved
support, missing modes, uneven occupancy, and incorrect spread become visible
rather than hidden inside one score.
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
def train_visible_flow(model, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    losses = []
    model.train()

    for step in range(steps):
        x_data, _ = sample_labeled_mixture(1024, device=device)
        x_noise = sample_base(1024, device=device)
        t = torch.rand(1024, device=device)
        x_t = (1 - t[:, None]) * x_noise + t[:, None] * x_data
        target_velocity = x_data - x_noise

        loss = (model(t, x_t) - target_velocity).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    return losses

model, losses, trained_now = load_or_train_model(device=device, trainer=train_visible_flow)
print("trained now:", trained_now, "| parameters:", sum(p.numel() for p in model.parameters()))
'''),
        code(r'''
plot_loss_curve(losses, title="Flow-matching training loss")
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

plot_flow_endpoints(x0_eval, target, target_labels, generated)
print(f"sampling time: {sample_seconds:.3f} s")
'''),
        code(r'''
diagnostics = pd.DataFrame(flow_endpoint_diagnostics(generated, target, target_labels))
display(diagnostics.round(4))
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
plot_wasserstein_matching(generated, target)
'''),
        code(r'''
plot_velocity_field(model, device)
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
circle = run_flow_case(
    load_or_train_circle_model, sample_noisy_circle,
    device=device, seed=SEED + 71,
)
'''),
        code(r'''
plot_circle_flow(circle)
'''),
        code(r'''
display(pd.DataFrame(circle_flow_metrics(circle)).round(4))
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
balanced_labels = torch.arange(8, device=device).repeat_interleave(512)
eight_modes = run_flow_case(
    load_or_train_eight_gaussian_model,
    sample_eight_gaussians,
    device=device,
    seed=SEED + 81,
    sampler_kwargs={"labels": balanced_labels},
)
'''),
        code(r'''
plot_eight_gaussian_flow(eight_modes)
'''),
        code(r'''
eight_summary, per_mode = eight_gaussian_metrics(eight_modes)
display(pd.DataFrame(per_mode).round(3))
display(pd.DataFrame(eight_summary).round(4))
'''),
        markdown(r'''
The snapshot figure shows **where the learned ODE sends regions of the initial
Gaussian**, while the diagnostics describe mode coverage, occupancy, and
within-mode spread. No single plot or scalar captures every mismatch between
the generated and target distributions.
'''),
        markdown(r'''
## From a trained field to a denoised estimate

**Questions**

1. Why can a low minibatch MSE coexist with visibly poor samples?
2. What information about class identity did the unconditional model receive?
3. At which time does the vector field mostly choose coarse destination modes?

These examples show why training loss is only the beginning of the story. The
circle and eight-mode models are diagnostic detours: they reveal
curved-support and occupancy failures, while the remaining toy notes return to
the three-cluster model for class-level steering. With one trained field in
hand, Notebook `02` asks how to read its prediction as a denoised estimate and
how faithfully different deterministic solvers follow the same field.
'''),
    ],
    "02_denoisers_and_deterministic_samplers.ipynb": [
        markdown(r'''
# 02 - Denoisers and deterministic samplers

Notebook `01` gave us a trained velocity field, but the steering methods that
follow are easier to describe as changes to a clean-data estimate. For the
linear I-CFM path, these are two views of the same learned prediction: velocity
describes how the current state should move, while
$D_\theta(x_t,t)=x_t+(1-t)v_\theta(x_t,t)$ estimates the clean endpoint
compatible with that state. After deriving this bridge, we reuse the same
initial noise to compare Euler, Heun, and RK4, separating errors in numerical
integration from errors in the learned field.
'''),
        code(SETUP),
        code(r'''
model, losses, _ = load_or_train_model(device=device)
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
    clean_estimate = visual_xt + (1 - visual_t[:, None]) * learned_velocity

plot_velocity_to_denoiser(
    visual_xt, pair_velocity, learned_velocity, clean_estimate, visual_clean,
    t_value=visual_t_value,
)
'''),
        code(r'''
set_seed(SEED + 20)
clean, _ = sample_labeled_mixture(450, device=device)
noise = sample_base(clean.shape[0], device=device)
plot_denoiser_times(model, clean, noise)
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
def euler_step(velocity, t, x, dt):
    return x + dt * velocity(t, x)

def heun_step(velocity, t, x, dt):
    slope_now = velocity(t, x)
    predicted = x + dt * slope_now
    slope_next = velocity(t + dt, predicted)
    return x + 0.5 * dt * (slope_now + slope_next)

def integrate_with_step(velocity, x0, steps, step_fn):
    x = x0.detach().clone()
    path = [x.cpu()]
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full((len(x),), index * dt, device=x.device)
        x = step_fn(velocity, t, x, dt).detach()
        path.append(x.cpu())
    return torch.stack(path)
'''),
        code(r'''
set_seed(SEED + 21)
x0 = sample_base(1200, device=device)
target, _ = sample_labeled_mixture(1200, device=device)

velocity = base_velocity(model)
reference_path = integrate_ode(velocity, x0, steps=256, method="rk4")
paths = {}
for name, step_fn in {"euler": euler_step, "heun": heun_step}.items():
    for steps in [8, 16, 32, 64]:
        paths[(name, steps)] = integrate_with_step(velocity, x0, steps, step_fn)
for steps in [8, 16, 32, 64]:
    paths[("rk4", steps)] = integrate_ode(velocity, x0, steps=steps, method="rk4")

rows, saved_paths = solver_metrics_from_paths(paths, reference_path[-1], target)
solver_table = pd.DataFrame(rows)
display(solver_table.round(4))
'''),
        code(r'''
plot_trajectory_comparison(saved_paths, n_lines=22)
plt.show()
plot_solver_accuracy(solver_table)
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

A deterministic sampler gives us more than reproducibility: when two methods
start from the same initial noise, their trajectories are paired sample by
sample, so the resulting differences can be attributed to the changed sampling
rule rather than to luck. That pairing is an evaluation tool, not the steering
method called noise alignment. Notebook `03` now uses the denoiser view itself
to construct a class-level correction from two fitted distributions, target
class minus full data, while the trajectory is still highly noisy.
'''),
    ],
    "03_gaussian_denoisers_and_noise_alignment.ipynb": [
        markdown(r'''
# 03 - Class steering with Gaussian/PCA denoisers

Notebook `02` showed that a velocity model can also be read as a denoiser. We
now ask whether a simpler denoiser, built only from a distribution's mean and
principal directions, can supply useful class information while the sample is
still dominated by noise. We first test Gaussian/PCA posterior means on
held-out MNIST images, then compare a target-class denoiser with a full-data
denoiser at the same 2D state. Their difference,

$$\Delta D = D_{\text{target}} - D_{\text{full}},$$

is a coarse distribution-level correction. It does not pull the sample toward
a chosen image, centroid, or coordinate.
'''),
        code(SETUP),
        code(r'''
model, losses, _ = load_or_train_model(device=device)
model.eval()
model.requires_grad_(False)

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

[Li, Dai, and Qu (NeurIPS 2024)](https://arxiv.org/abs/2410.24060) provide
broader evidence that learned diffusion denoisers can approach empirical
Gaussian denoisers in a generalization regime. The MNIST calculation below is
our teaching example; Notebook `06` supplies the pretrained image-generator
experiment for this series.
'''),
        code(r'''
def gaussian_posterior_mean(y, sigma, mean, covariance):
    """E[x_clean | y] when x_clean is Gaussian and y = x_clean + noise."""
    sigma = torch.as_tensor(sigma, device=y.device).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(y.shape[0])
    identity = torch.eye(y.shape[1], device=y.device).expand(y.shape[0], -1, -1)
    covariance = covariance.expand(y.shape[0], -1, -1)
    system = covariance + sigma[:, None, None].square() * identity
    solved = torch.linalg.solve(system, (y - mean).unsqueeze(-1))
    return mean + (covariance @ solved).squeeze(-1)

def pca_posterior_mean(y, sigma, stats):
    centered = y - stats.mean
    coordinates = centered @ stats.components
    sigma_squared = torch.as_tensor(sigma, device=y.device).reshape(-1, 1).square()
    variances = stats.variances[None, :]
    shrinkage = variances / (variances + sigma_squared)
    return stats.mean + (coordinates * shrinkage) @ stats.components.T
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
plot_pca_components(mnist_full_gaussian, mnist_class_gaussian)
'''),
        code(r'''
held_out_pool = torch.where(mnist_test_labels == MNIST_TARGET)[0]
held_out_indexes = held_out_pool[:256]
mnist_clean = mnist_test_images[held_out_indexes]
set_seed(SEED + 332)
mnist_noisy = mnist_clean + MNIST_SIGMA * torch.randn_like(mnist_clean)
mnist_full_denoised = pca_posterior_mean(mnist_noisy, MNIST_SIGMA, mnist_full_gaussian)
mnist_class_denoised = pca_posterior_mean(mnist_noisy, MNIST_SIGMA, mnist_class_gaussian)

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

plot_image_rows([
    ("clean test image", mnist_clean),
    (f"noisy, sigma={MNIST_SIGMA}", mnist_noisy),
    ("all-digit Gaussian", mnist_full_denoised),
    ("digit-3 Gaussian", mnist_class_denoised),
], title="Gaussian/PCA posterior means on held-out digit-3 images")
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
t = torch.full((grid.shape[0], 1), t_value, device=device)
y = grid / t
sigma = BASE_STD * (1 - t_value) / t_value

D_target = gaussian_posterior_mean(
    y, sigma, steering_stats.means[TARGET_CLASS], steering_stats.covariances[TARGET_CLASS]
)
D_full = gaussian_posterior_mean(
    y, sigma, steering_stats.full_mean, steering_stats.full_covariance
)
denoiser_delta = D_target - D_full

plot_noise_alignment_field(
    grid, denoiser_delta, steering_data, steering_labels,
    target_class=TARGET_CLASS, t_value=t_value, start=0.04, end=0.35,
)
'''),
        markdown(r'''
## 4. Does the correction improve paired samples?

All methods below use the same initial noise, Heun solver, step count, and target
reference. The only change is the noise-alignment strength. The correction is
active for `t in [0.04, 0.35]`, the high-noise part of our noise-to-data path.
'''),
        code(r'''
def noise_aligned_velocity(strength):
    def velocity(t, x):
        with torch.no_grad():
            base = model(t, x)
            t_column = t[:, None].clamp_min(0.04)
            y = x / t_column
            sigma = BASE_STD * (1 - t) / t.clamp_min(0.04)
            target_estimate = gaussian_posterior_mean(
                y, sigma, steering_stats.means[TARGET_CLASS],
                steering_stats.covariances[TARGET_CLASS],
            )
            full_estimate = gaussian_posterior_mean(
                y, sigma, steering_stats.full_mean, steering_stats.full_covariance,
            )
            gate = window_gate(t, 0.04, 0.35)[:, None]
            remaining = (1 - t_column).clamp_min(0.04)
            return base + strength * gate * (target_estimate - full_estimate) / remaining
    return velocity
'''),
        code(r'''
set_seed(SEED + 30)
x0 = sample_base(1800, device=device)
strengths = [0.0, 0.4, 0.8, 1.2, 1.6]
methods = {"baseline": base_velocity(model)}
methods.update({f"noise alignment {s:.1f}": noise_aligned_velocity(s) for s in strengths})

rows, paths = compare_steering_methods(
    methods, x0, target_reference, evaluation_stats,
    target_class=TARGET_CLASS, steps=64,
)
noise_alignment_table = pd.DataFrame(rows)
noise_alignment_table["strength"] = [
    np.nan if name == "baseline" else float(name.rsplit(" ", 1)[-1])
    for name in noise_alignment_table["method"]
]
display(noise_alignment_table.round(4))
'''),
        code(r'''
plot_noise_alignment_comparison(
    paths, noise_alignment_table, target_reference, evaluation_stats, strength=1.6,
)
'''),
        code(r'''
selected = noise_alignment_table.set_index("method").loc[
    ["baseline", "noise alignment 1.6"],
    ["target_rate", "target_wasserstein", "mean_error", "diversity_ratio"],
]
display(selected.round(4))
'''),
        markdown(r'''
## Reading the paired result

Read the endpoints and metrics together. A higher target-class rate indicates
stronger control, a lower Wasserstein distance indicates better agreement with
target examples, and the diversity ratio shows whether that gain came from
collapse or excessive spread. The model remains unconditional and unchanged;
only the target-class and full-data summary statistics are prepared offline.
In 2D both principal directions are retained, while an image implementation
uses a low-rank PCA covariance model.

**Questions**

1. Why should this coarse correction be most useful at high noise?
2. Which strength best balances target rate and target Wasserstein distance in your run?
3. What information is lost when a multimodal full dataset is approximated by one Gaussian?

In this toy setting, the class-minus-full Gaussian correction produces partial
coarse steering by a frozen unconditional model; it does not recover a full
class-conditional distribution. Because means and covariances can express only
limited structure, Notebook `04` replaces those fixed summaries with the
gradient of a differentiable class objective.
'''),
    ],
    "04_training_free_gradient_guidance.ipynb": [
        markdown(r'''
# 04 - Post-hoc objective-gradient guidance

Gaussian/PCA noise alignment is efficient because its correction can be
computed with forward passes and small linear operations, but it can use only
the structure captured by fitted means and covariances. A differentiable class
objective offers a more adaptive alternative: at each active sampling state,
its gradient points toward the local change that would raise the target score
of the model's denoised estimate. The generator remains frozen, and the
direction is not aimed at a preset point, but obtaining it requires
backpropagation through the denoiser during sampling.
'''),
        code(SETUP),
        code(r'''
model, losses, _ = load_or_train_model(device=device)
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
def gradient_guided_velocity(strength):
    def velocity(t, x):
        gate = window_gate(t, 0.15, 0.92)[:, None]
        if gate.max() == 0:
            return model(t, x).detach()

        x_for_gradient = x.detach().requires_grad_(True)
        base = model(t, x_for_gradient)
        clean_estimate = x_for_gradient + (1 - t[:, None]) * base
        log_prob = class_log_probabilities(clean_estimate, stats).log_softmax(dim=1)
        objective = log_prob[:, TARGET_CLASS].sum()
        gradient = torch.autograd.grad(objective, x_for_gradient)[0]
        direction = gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return base.detach() + strength * gate * direction.detach()

    return velocity
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
    "gradient 1.5": gradient_guided_velocity(1.5),
    "gradient 3.0": gradient_guided_velocity(3.0),
}

rows, paths = compare_steering_methods(
    methods, x0, target_reference, stats, target_class=TARGET_CLASS, steps=48,
)
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

Objective-gradient guidance makes a different trade-off from noise alignment:
its direction adapts to the current state, but each active solver evaluation
requires backpropagation, and the signal is least trustworthy before the
denoised estimate carries useful class information. The next question is
whether a frozen model can supply a useful class direction internally, without
an inference-time backward pass. Notebook `05` therefore probes a hidden layer,
learns a target-versus-rest direction offline, and tests whether editing that
representation changes the generated trajectories.
'''),
    ],
    "05_activation_steering_and_method_comparison.ipynb": [
        markdown(r'''
# 05 - Hidden-feature steering and method comparison

Notebook `04` obtained a steering direction by differentiating an external
objective at every active step. Here we ask whether a useful direction is
already present inside the frozen velocity network. We record one hidden layer
on labeled, noised examples, test on separate examples whether class can be
decoded from that representation, and learn a target-versus-rest direction
offline. During sampling we add that direction to the hidden feature rather
than to the sample position, so the downstream layers decide how the velocity
changes and no inference-time gradient is needed.
'''),
        code(SETUP),
        code(r'''
model, losses, _ = load_or_train_model(device=device)
reference_data, reference_labels = sample_labeled_mixture(30000, device=device)
stats = estimate_gaussian_stats(reference_data, reference_labels)
TARGET_CLASS = 2
model.eval()
model.requires_grad_(False)
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
plot_activation_pipeline()
'''),
        markdown(r'''
## 2. When is class information visible in the hidden layer?

At each time, we record activations from separate balanced training and test
sets with independent noise. A linear probe is fitted only on the training
activations and scored on the untouched test activations. A shuffled-label
probe provides a simple comparison without the true class relationship.
The comparison asks whether class is readable from the hidden state; it does
not yet show that editing the state will control generation.
'''),
        code(r'''
def collect_activations(model, *, t_value, n_per_class):
    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(n_per_class)
    x_data, labels = sample_labeled_mixture(
        len(labels), device=device, labels=labels,
    )
    x_noise = sample_base(len(labels), device=device)
    t = torch.full((len(labels),), t_value, device=device)
    x_t = (1 - t[:, None]) * x_noise + t[:, None] * x_data
    with torch.no_grad():
        features = model.features(t, x_t)
    return features, labels
'''),
        code(r'''
REFERENCE_T = 0.90
set_seed(SEED + 500)
train_features, train_labels = collect_activations(
    model, t_value=REFERENCE_T, n_per_class=600,
)
set_seed(SEED + 501)
test_features, test_labels = collect_activations(
    model, t_value=REFERENCE_T, n_per_class=400,
)

probe = fit_linear_probe(train_features, train_labels)
probe_accuracy = (probe.predict(test_features) == test_labels).float().mean()

positive = train_features[train_labels == TARGET_CLASS]
negative = train_features[train_labels != TARGET_CLASS]
mean_difference = positive.mean(0) - negative.mean(0)
centered = train_features - train_features.mean(0)
covariance = centered.T @ centered / (len(centered) - 1)
ridge = 0.1 * torch.eye(covariance.shape[0], device=device)
reference_direction = torch.linalg.solve(covariance + ridge, mean_difference)
reference_direction /= reference_direction.norm()

shuffled_labels = train_labels[torch.randperm(len(train_labels), device=device)]
shuffled_probe = fit_linear_probe(train_features, shuffled_labels)
shuffled_accuracy = (shuffled_probe.predict(test_features) == test_labels).float().mean()
shuffled_direction = discriminant_direction(train_features, shuffled_labels, TARGET_CLASS)

display(pd.Series({
    "held-out probe accuracy": float(probe_accuracy),
    "shuffled-label accuracy": float(shuffled_accuracy),
}, name=f"t={REFERENCE_T}").round(3))
'''),
        code(r'''
pca_display = fit_pca_projection(train_features)
projected_test = pca_display.transform(test_features)

fig, ax = plt.subplots(figsize=(6, 5))
plot_labeled_points(
    projected_test, test_labels, ax=ax,
    title=f"Held-out activations at t={REFERENCE_T}", alpha=0.35,
)
ax.legend(frameon=False)
plt.show()
'''),
        markdown(r'''
### Reading fixed-direction transfer

We estimate $d_{0.90}$ at late time and reuse it as a fixed intervention vector
over the window $[0.40, 0.90]$. The cosine plot shows that time-specific
directions are only weakly aligned with $d_{0.90}$ near the beginning of that
window. The intervention therefore does not follow the local class direction
at every time. It asks a narrower question: can one late-time direction,
repeated over part of the trajectory, produce a useful endpoint change?
'''),
        code(r'''
probe_rows, directions = activation_probe_sweep(
    model,
    target_class=TARGET_CLASS,
    times=[0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.90],
    seed=SEED + 700,
    collect=collect_activations,
)
probe_table = pd.DataFrame(probe_rows)
display(probe_table.round(3))
plot_activation_probe_sweep(
    probe_table, directions, reference_direction, reference_t=REFERENCE_T,
)
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

We collect the direction at `t = 0.90`, where the held-out probe is strongest,
then reuse it over a wider intervention window. Collection time and
intervention time are different choices: the cosine plot above shows that the
late direction is not the local class direction at every step. The paired
comparison below asks whether this fixed edit is nevertheless useful.
'''),
        code(r'''
def activation_velocity(direction, strength, start, end):
    direction = direction / direction.norm()

    def velocity(t, x):
        gate = window_gate(t, start, end)[:, None]
        with torch.no_grad():
            hidden = model.features(t, x)
            feature_scale = hidden.square().mean(dim=1, keepdim=True).sqrt()
            edited = hidden + strength * gate * feature_scale * direction[None, :]
            return model.velocity_from_features(edited)

    return velocity
'''),
        code(r'''
set_seed(SEED + 50)
x0 = sample_base(1400, device=device)
target_reference = sample_class(1400, TARGET_CLASS, device=device)

ACTIVATION_STRENGTH = 7.0
ACTIVATION_WINDOW = (0.40, 0.90)
COMBINED_ACTIVATION_STRENGTH = 5.0
COMBINED_ACTIVATION_WINDOW = (0.35, 0.85)
'''),
        code(r'''
methods = {
    "baseline": base_velocity(model),
    "activation 0.0": activation_velocity(reference_direction, 0.0, *ACTIVATION_WINDOW),
    "shuffled activation": activation_velocity(
        shuffled_direction, ACTIVATION_STRENGTH, *ACTIVATION_WINDOW,
    ),
    "activation steering": activation_velocity(
        reference_direction, ACTIVATION_STRENGTH, *ACTIVATION_WINDOW,
    ),
    "noise alignment": make_noise_aligned_velocity(
        model, stats, target_class=TARGET_CLASS, strength=1.6
    ),
    "gradient guidance": make_gradient_guided_velocity(
        model, stats, target_class=TARGET_CLASS, strength=3.0, start=0.15, end=0.92
    ),
    "noise + activation": add_noise_alignment(
        activation_velocity(
            reference_direction,
            COMBINED_ACTIVATION_STRENGTH,
            *COMBINED_ACTIVATION_WINDOW,
        ),
        stats,
        target_class=TARGET_CLASS,
        strength=1.6,
    ),
}
'''),
        code(r'''
rows, paths = compare_steering_methods(
    methods, x0, target_reference, stats, target_class=TARGET_CLASS, steps=56,
)
comparison_table = pd.DataFrame(rows)
comparison_table["runtime / baseline"] = comparison_table["seconds"] / comparison_table.loc[0, "seconds"]
display(comparison_table.round(4))
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
plot_method_tradeoff(comparison_table)
'''),
        code(r'''
method_rows = comparison_table.set_index("method")
activation_effects = pd.Series({
    "activation target-rate gain": method_rows.loc["activation steering", "target_rate"] - method_rows.loc["baseline", "target_rate"],
    "shuffled-direction gain": method_rows.loc["shuffled activation", "target_rate"] - method_rows.loc["baseline", "target_rate"],
    "activation Wasserstein reduction": method_rows.loc["baseline", "target_wasserstein"] - method_rows.loc["activation steering", "target_wasserstein"],
})
display(activation_effects.round(4))
'''),
        markdown(r'''
## 4. Reading the method comparison

The methods intervene at different places. Noise alignment changes a denoised
estimate early, objective-gradient guidance differentiates a class score, and
activation steering edits a hidden feature. Target-class rate describes
control, Wasserstein distance describes agreement with target examples,
diversity describes spread, and runtime exposes the price of online
backpropagation. No method is best without naming the trade-off that matters.

**Questions for discussion**

1. Does probe accuracy predict steering success at the same collection time?
2. Over which times is the learned direction stable enough to reuse?
3. When does combining coarse high-noise and later activation control help?
4. Which conclusion from 2D would require a new experiment before claiming it for images?

The probe and intervention answer different questions: held-out decoding shows
that class information is readable from the hidden layer, while paired
sampling tests whether one fixed late-time direction is useful over the chosen
intervention window. Neither result implies a unique class axis or a
time-invariant representation. Across the toy series, we have now compared
three distinct intervention sites. Notebook `06` carries only noise alignment
to a pretrained image model and asks whether paired images and batch-level
features still reveal a measurable, partial shift.
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
