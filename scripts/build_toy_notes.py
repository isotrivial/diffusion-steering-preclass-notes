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
# 00 - How does noise become a new sample?

The series follows one generator from training to post-hoc steering. We begin
with the pieces that are easiest to confuse: the data distribution, random
starting noise, the neural prediction, and the numerical sampler. In two
dimensions all four can be drawn before we meet the same ideas in images.

The model is unconditional: class labels are never given to its velocity
network. Later notes use labels only to construct and evaluate controls after
training.

**Time convention for Notes 00--05:** `t=0` is noise and `t=1` is data.
'''),
        code(SETUP),
        markdown(r'''
## 1. What is a generative model?

A generative model learns a **sampling procedure** from examples. A useful
generator produces new samples with the structure of the training distribution
rather than returning a stored example.

For the flow model in these notes:

- an easy source distribution supplies fresh Gaussian noise;
- a neural network learns a time-dependent velocity;
- an ODE sampler follows that velocity from noise to a generated endpoint.

The neural network alone is not the complete generator. The source
distribution, trained model, and sampler work together.

**Before you run:** In the diagram, which object is learned from data? Which
object supplies the variation between generated samples?
'''),
        code(r'''
plot_generator_pipeline()
'''),
        markdown(r'''
The velocity network is learned. Fresh noise supplies different starting
states, while the sampler turns each starting state into one endpoint.

### Three ways to request control

- An **unconditional** generator models all training examples without receiving
  a requested class. That is the model trained here.
- A **conditional** generator receives extra information, such as a class label
  or text prompt, during training and generation.
- **Post-hoc steering** keeps a trained generator frozen and modifies the
  sampling computation. Notebooks `03`--`05` study this third setting.

Post-hoc steering is the setting used later: the unconditional model remains
frozen while the sampling computation is changed. A deterministic sampler can
still generate a diverse collection because different initial noise points
follow different deterministic trajectories.
'''),
        markdown(r'''
## 2. What information does noise hide?

A probability distribution describes which regions are likely and how often
different kinds of samples occur. Images live in thousands or millions of
dimensions, where that structure is difficult to draw. Here every sample has
only two coordinates, so a large scatter plot makes cluster locations,
frequencies, shapes, and spread directly visible. The three colors play the
role of mixture-component labels. Later notes treat those labels as toy classes
for steering and evaluation. The velocity model will not receive them.
'''),
        code(r'''
clean, labels = sample_labeled_mixture(2400, device=device)
fig, ax = plt.subplots(figsize=(6, 5))
plot_labeled_points(clean, labels, ax=ax, title="Clean toy data")
ax.legend(frameon=False)
plt.show()
'''),
        markdown(r'''
## 3. One path, read in two directions

First sample base noise, then use the linear path

$$x_{noise}=s\epsilon,\qquad \epsilon\sim\mathcal N(0,I),$$

$$x_t=(1-t)x_{noise}+t x_{data},\qquad s=1.8.$$

- At `t=1`, the point is clean data.
- At `t=0`, the point is Gaussian noise.
- Lower `t` therefore means higher noise in this notebook series.

A diffusion model learns to undo a noising process. Flow matching learns the
velocity along a path between the same endpoints.

### Diffusion and flow matching in one table

| Family | Typical learned quantity | Generation step |
|---|---|---|
| Diffusion model | noise, score, or a denoised estimate | repeatedly remove noise with a stochastic or deterministic sampler |
| Flow-matching model | velocity along a chosen probability path | integrate an ODE from the source distribution to data |

Both are time-dependent generative models. Here `t` indexes a position along
the noise-to-data path; it is not elapsed computer time.

**Before you run:** As `t` increases from noise to data, which structure should
become visible first: the exact identity of every point, or the coarse cluster
layout? The same clean points and noise points are reused in every panel.
'''),
        code(r'''
set_seed(SEED)
clean_small, labels_small = sample_labeled_mixture(1500, device=device)
fixed_noise = sample_base(clean_small.shape[0], device=device)

times = [0.08, 0.35, 0.70, 1.0]
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
The coarse three-cluster layout appears before individual points reach their
clean endpoints. This early-to-late change in available information will matter
when we choose a steering window.

**Change one thing:** Replace `0.35` in `times` with another value between zero
and one. Rerun only the preceding cell and locate when the clusters first become
easy to distinguish.

## 4. What flow matching predicts

Choose a noise point `x_noise`, a data point `x_data`, and a time `t`. Along the
straight conditional path, the target velocity is simply

$$u_t = x_{data} - x_{noise}.$$

Training repeatedly asks a neural network to predict this arrow from only
`(x_t, t)`. Because many endpoint pairs can pass near the same `x_t`, the
network learns an average velocity field, not a memorized arrow for each pair.
The pair-specific target is available while constructing training examples but
is not available during generation.

**Before you run:** The arrows below come from individual endpoint pairs. If
two different pairs pass through nearly the same state, can the network recover
both pair-specific arrows from `(x_t,t)` alone?
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
No. The network sees the state and time, not the hidden endpoint pair. Under
mean-squared-error training it learns the conditional average of compatible
pair velocities. Notebook `01` tests whether those local averages assemble into
a useful global flow.

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
only the sampler; the source distribution and learned velocity field remain the
same.

Notebook `01` now supplies the missing piece: train the velocity field and ask
whether accurate local predictions actually produce the whole data
distribution.
'''),
    ],
    "01_train_unconditional_flow_matching.ipynb": [
        markdown(r'''
# 01 - Can local velocity predictions generate the whole distribution?

Notebook `00` defined the training arrows. We now fit a small unconditional MLP
and follow its learned field from fresh noise to generated samples. The central
test is visual: a falling training loss is useful, but does it guarantee that
the generated batch covers every mode with the right shape and frequency?

Three clusters form the main experiment used throughout the steering notes. A
circle and eight separated Gaussians appear later as short transfer checks for
curved support and missing modes.

**Time convention:** `t=0` is noise and `t=1` is data.
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

**Before you run:** Predict which claim a low minibatch loss supports directly:
accurate sampled training arrows, correct generated mode frequencies, or both.
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

model, losses, _ = load_or_train_model(device=device, trainer=train_visible_flow)
print("trainable parameters:", sum(p.numel() for p in model.parameters()))
'''),
        code(r'''
plot_loss_curve(losses, title="Flow-matching training loss")
'''),
        markdown(r'''
The loss measures sampled velocity targets. It does not directly count modes
or compare generated endpoints with data, so generation must be checked by
actually integrating the field.

## 2. How does a fitted field become samples?

Once an initial noise batch is fixed, the ODE sampler is deterministic:

$$\frac{dx}{dt}=v_\theta(t,x), \qquad x(0)=x_{noise}.$$

We use Heun's method here. Notebook `02` compares it with Euler and RK4.

**Before you run:** At early times, should particles already be separated by
their eventual destination, or can the partition emerge gradually? Colors in
the snapshot figure will be assigned from final endpoints, after generation.
'''),
        code(r'''
set_seed(SEED + 10)
x0_eval = sample_base(3000, device=device)
trajectory, sample_seconds = timed_sample(base_velocity(model), x0_eval, steps=80, method="heun")
generated = trajectory[-1].to(device)
target, target_labels = sample_labeled_mixture(3000, device=device)

plot_flow_endpoints(x0_eval, target, target_labels, generated)
plot_flow_snapshots(trajectory, generated, target, target_labels)
print(f"sampling time: {sample_seconds:.3f} s")
'''),
        code(r'''
diagnostics = pd.DataFrame(flow_endpoint_diagnostics(generated, target, target_labels))
display(diagnostics.round(4))
'''),
        markdown(r'''
### Why use an empirical Wasserstein matching distance in addition to a plot?

The time snapshots show one learned map, while the endpoint panel reveals
whether its final distribution resembles the data. A class rate or mean checks
only one property. The empirical Wasserstein matching below finds a minimum-cost
one-to-one matching between equal-size sample subsets and reports their mean
distance. It is a finite-sample diagnostic, not the unknown population
population Wasserstein distance.

The lines below make the definition literal. Lower is better. We still inspect
mode coverage and spread because one scalar cannot diagnose every failure.
'''),
        code(r'''
_ = plot_wasserstein_matching(generated, target)
'''),
        code(r'''
plot_velocity_field(model, device)
'''),
        markdown(r'''
## 3. Transfer check: does the same rule learn a curve?

Three Gaussian clusters could leave the impression that flow matching only
learns a few destination centers. A noisy circle is a stricter visual test: it
has continuously many valid endpoints and a thin curved support. We train a
separate unconditional model and ask whether it forms the ring without leaving
large angular gaps.

**Before you run:** A low radial error is not enough to reproduce a circle.
What additional failure would appear as a large empty arc?
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
circle_metrics = pd.DataFrame(circle_flow_metrics(circle))
display(circle_metrics[circle_metrics["diagnostic"].isin([
    "generated radial MAE",
    "angular occupancy total variation",
    "generated-to-target empirical matching distance",
])].round(4))
'''),
        markdown(r'''
## 4. Transfer check: can one flow cover eight modes?

The three-class model is kept for the steering notebooks because its labels are
easy to discuss, while the circle tests continuous curved geometry. Eight narrow
components expose a different failure mode: a model can omit destinations or
assign them the wrong amount of probability. This is still learned I-CFM; no
destination center is used by the sampler.

We color each particle by the component nearest to its **final** endpoint, then
trace the same particle backward through stored time slices. The colors are for
reading the map after sampling; the model never receives them.

**Before you run:** Could a model produce convincing points near seven centers
and still have a moderate average distance to the full target? Look for both
coverage and occupancy in the result.
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
display(pd.DataFrame(eight_summary).round(4))
'''),
        markdown(r'''
The snapshot figure shows **where the learned ODE sends regions of the initial
Gaussian**, while the diagnostics describe mode coverage, occupancy, and
within-mode spread. No single plot or scalar captures every mismatch between
the generated and target distributions.
'''),
        markdown(r'''
## What carries forward?

The circle and eight-mode cases are transfer checks, not new storylines. They
show why we inspect trajectories, coverage, occupancy, and spread rather than
trusting one training loss or one distance.

The remaining toy notes return to the same three-cluster checkpoint. Notebook
`02` holds that model and its initial noise fixed, then asks how much of a
generated endpoint is determined by the learned field and how much by the
numerical solver.

**Change one thing:** In the main sampling cell, replace `steps=80` with
`steps=12` and rerun only the endpoint and snapshot cells. The model is
unchanged; any new distortion comes from following the same field more
coarsely.
'''),
    ],
    "02_denoisers_and_deterministic_samplers.ipynb": [
        markdown(r'''
# 02 - Same field, different deterministic samplers

Notebook `01` trained one velocity field. We now freeze that field and ask two
questions. First, how closely do Euler, Heun, and RK4 follow it? Second, how can
the same velocity prediction be read as an estimate of clean data?

Keeping the model and initial noise fixed makes both questions concrete. A
solver changes the numerical path. The denoiser conversion changes how we read
the model's prediction, not what the model learned.

**Time convention:** `t=0` is noise and `t=1` is data.
'''),
        code(SETUP),
        code(r'''
model, losses, _ = load_or_train_model(device=device)
'''),
        markdown(r'''
## 1. What does deterministic mean here?

The model defines the arrows; the solver decides how to follow them. A fair
solver comparison changes only the solver and step count. It keeps the model,
initial noise tensor, and evaluation reference fixed.

Deterministic is conditional on the initial state: repeating the same
`x_noise` repeats the trajectory, while different initial noise points can
still produce a diverse batch of endpoints.

We use a 256-step RK4 trajectory as a numerical reference. This does not remove
model error; it only makes the numerical integration error small.

**Before you run:** Euler uses one model evaluation per step, Heun uses two, and
RK4 uses four. At a fixed number of steps, which should follow a curved field
most accurately? At a fixed number of model evaluations, is the answer
necessarily the same?
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

STEP_COUNTS = [8, 16, 32, 64]
'''),
        code(r'''
set_seed(SEED + 21)
x0 = sample_base(1200, device=device)
target, _ = sample_labeled_mixture(1200, device=device)

velocity = base_velocity(model)
reference_path = integrate_ode(velocity, x0, steps=256, method="rk4")
paths = {}
for name, step_fn in {"euler": euler_step, "heun": heun_step}.items():
    for steps in STEP_COUNTS:
        paths[(name, steps)] = integrate_with_step(velocity, x0, steps, step_fn)
for steps in STEP_COUNTS:
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
The paired endpoint error is meaningful because every solver starts from the
same tensor. A more accurate solver follows the learned field more faithfully;
it cannot repair a poorly learned field.

**Change one thing:** Replace `STEP_COUNTS` with `[4, 8, 16]`. Rerun the solver
cells and find the smallest computation budget at which the generated modes
remain recognizable.

## 2. How can velocity be read as a clean estimate?

For one known training pair,

$$x_t=x_{noise}+t(x_{data}-x_{noise}),\qquad
u=x_{data}-x_{noise}.$$

Rearranging gives the exact identity

$$x_{data}=x_t+(1-t)u.$$

During generation the endpoint pair is unknown, so the network substitutes its
learned velocity for `u`:

$$D_\theta(x_t,t)=x_t+(1-t)v_\theta(x_t,t).$$

**Before you run:** At `t=0.35`, should this estimate recover each hidden clean
endpoint exactly, or average over several endpoints compatible with the same
noisy state?
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
        markdown(r'''
The exact pair velocity recovers its paired endpoint because both endpoints were
used to construct it. The network sees only `(x_t,t)`. With mean-squared-error
training, the population prediction is a conditional average,

$$v^*(x_t,t)=\mathbb E[u\mid x_t,t],$$

so the converted estimate is

$$D^*(x_t,t)=\mathbb E[x_{data}\mid x_t,t].$$

It is a posterior mean, not a reconstruction of one secretly known original.
The full derivation and the endpoint caveat are in `BACKGROUND.md`.
'''),
        code(r'''
set_seed(SEED + 20)
clean, _ = sample_labeled_mixture(450, device=device)
noise = sample_base(clean.shape[0], device=device)
plot_denoiser_times(model, clean, noise)
'''),
        markdown(r'''
## 3. Keep two uses of noise separate

A **same-seed paired comparison** is an evaluation design: two deterministic
methods start from the exact same noise tensor, so endpoint differences can be
attributed to the changed sampling rule.

Notebook `03` introduces a steering signal formed by subtracting a full-data
denoiser from a target-class denoiser. We will call it the
**class-minus-full denoiser correction** and show the subtraction explicitly.
It is unrelated to reusing a seed.

Notebook `03` begins with a visible question: can a denoiser built only from a
mean and principal directions recover useful structure from noisy MNIST
images?
'''),
    ],
    "03_gaussian_denoisers_and_noise_alignment.ipynb": [
        markdown(r'''
# 03 - From a Gaussian denoiser to a class-minus-full correction

Notebook `02` showed that a velocity model can also be read as a denoiser. We
now ask whether a much simpler denoiser, built only from a distribution's mean
and principal directions, can recover useful structure. We first test that
question on held-out MNIST images. We then compare a target-class denoiser with
a full-data denoiser at the same 2D state. Their difference,

$$\Delta D = D_{\text{target}} - D_{\text{full}},$$

is a coarse distribution-level correction, not a vector toward a selected
sample.

**Time convention:** `t=0` is noise and `t=1` is data.
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
## 1. A denoiser built from means and principal directions

A Gaussian model records a mean and covariance. In PCA coordinates, the
covariance tells us which directions vary strongly across clean examples. A
Gaussian posterior mean preserves evidence along those high-variance
directions and shrinks uncertain low-variance directions toward the mean.

The next two functions implement the same rule once with a full covariance
matrix and once in a low-rank PCA basis. The image result comes first; the
formula that explains it follows immediately afterward.
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
## 2. Can this simple model recover a noisy digit?

MNIST makes all 784 pixel coordinates visible without needing a large network.
We fit low-rank Gaussian models on the training split only: one to all digits
and one to digit 3. The leading covariance directions are `eigendigits`.
On held-out test images, we first check that the full-data model reduces error
across all ten digits. We then compare both posterior-mean denoisers on digit 3.

**Before you run:** The all-digit model has more training images, while the
digit-3 model has a more specific prior. Which should better recover a held-out
3 at this noise level? Which is more likely to erase unusual handwriting?
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
balanced_indexes = torch.cat([
    torch.where(mnist_test_labels == digit)[0][:64]
    for digit in range(10)
])
balanced_clean = mnist_test_images[balanced_indexes]
set_seed(SEED + 332)
balanced_noisy = balanced_clean + MNIST_SIGMA * torch.randn_like(balanced_clean)
balanced_denoised = pca_posterior_mean(
    balanced_noisy, MNIST_SIGMA, mnist_full_gaussian,
)
display(pd.Series({
    "noisy MSE, all digits": float(torch.mean((balanced_noisy - balanced_clean).square())),
    "full-data PCA MSE, all digits": float(torch.mean((balanced_denoised - balanced_clean).square())),
}).round(4))
'''),
        code(r'''
held_out_pool = torch.where(mnist_test_labels == MNIST_TARGET)[0]
held_out_indexes = held_out_pool[:256]
mnist_clean = mnist_test_images[held_out_indexes]
set_seed(SEED + 333)
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
The class-specific model usually removes more noise from typical 3s, but its
stronger prior can also suppress atypical details. The images make the trade-off
visible before we write it algebraically.

## 3. What rule produced those images?

Write the base point as $x_{noise}=s\epsilon$, with
$\epsilon\sim\mathcal{N}(0,I)$ and `s=1.8`. For `t>0`, divide the linear path by
`t`:

$$y=\frac{x_t}{t}=x_{data}+\sigma_{eff}\epsilon,\qquad
\sigma_{eff}=s\frac{1-t}{t}.$$

The flow state has become an ordinary additive-noise observation. Under a
Gaussian data model with mean $\mu$ and covariance $C$, its posterior mean is

$$D(y,\sigma)=\mu+C(C+\sigma^2I)^{-1}(y-\mu).$$

In a principal direction with variance $\lambda_i$, the observed coordinate is
multiplied by $\lambda_i/(\lambda_i+\sigma^2)$. High-variance structure is
retained more strongly; uncertain directions shrink more. The factor `s` is
required because the toy base distribution is not unit variance.
'''),
        code(r'''
plot_pca_shrinkage(mnist_class_gaussian)
'''),
        markdown(r'''
The curves show the rule without a matrix inverse: as noise increases, every
observation coordinate receives less weight, but high-variance principal
directions retain influence longer.

[Li, Dai, and Qu (NeurIPS 2024)](https://arxiv.org/abs/2410.24060) provide
broader evidence connecting diffusion denoisers with empirical Gaussian
denoisers in a generalization regime. Our MNIST panel is a small teaching test,
not a reproduction of that paper.

**Change one thing:** Set `MNIST_SIGMA` to `0.25` or `0.70` and rerun the MNIST
cells. Compare how much the observation and the class prior control the output.

## 4. How does denoising become a steering signal?

The image example shows what one Gaussian posterior mean preserves and loses.
For steering, we compare two such estimates at the same noisy state. Their
difference asks for structure favored by the target-class distribution relative
to the broad distribution. The gate below limits that correction to the
high-noise part of the trajectory. This literal subtraction is the mechanism
called **noise alignment** in the paper.

**Before you run:** Should the arrows point toward one common coordinate, or
change with the local state because two posterior means are being compared?
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

plot_class_minus_full_field(
    grid, denoiser_delta, steering_data, steering_labels,
    target_class=TARGET_CLASS, t_value=t_value, start=0.04, end=0.35,
)
'''),
        markdown(r'''
## 5. Does the correction improve paired samples?

All methods below use the same initial noise, Heun solver, step count, and target
reference. The only change is the class-minus-full correction strength. The
correction is active for `t in [0.04, 0.35]`, the high-noise part of our
noise-to-data path.

**Before you run:** Increasing strength should move more samples toward the
target class. Must target fit and diversity improve monotonically as well?
'''),
        code(r'''
def class_minus_full_velocity(strength):
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
CORRECTION_STRENGTH = 1.6
strengths = [0.0, 0.8, CORRECTION_STRENGTH]
methods = {"baseline": base_velocity(model)}
methods.update({
    f"class-minus-full {s:.1f}": class_minus_full_velocity(s)
    for s in strengths
})

rows, paths = compare_steering_methods(
    methods, x0, target_reference, evaluation_stats,
    target_class=TARGET_CLASS, steps=64,
)
correction_table = pd.DataFrame(rows)
correction_table["strength"] = [
    np.nan if name == "baseline" else float(name.rsplit(" ", 1)[-1])
    for name in correction_table["method"]
]
'''),
        code(r'''
plot_class_minus_full_comparison(
    paths, correction_table, target_reference, evaluation_stats,
    strength=CORRECTION_STRENGTH,
)
'''),
        code(r'''
selected = correction_table.set_index("method").loc[
    ["baseline", f"class-minus-full {CORRECTION_STRENGTH:.1f}"],
    ["target_rate", "target_wasserstein", "mean_error", "diversity_ratio"],
]
display(selected.rename(columns={
    "target_rate": "target-class rate",
    "target_wasserstein": "empirical matching distance",
    "mean_error": "target mean error",
    "diversity_ratio": "diversity ratio",
}).round(4))
'''),
        markdown(r'''
## Reading the paired result

Read the endpoints and metrics together. A higher target-class rate indicates
stronger control, a lower empirical matching distance indicates better
agreement with target examples, and the diversity ratio shows whether that
gain came from collapse or excessive spread. The diversity ratio is the total
coordinate variance of the generated batch divided by that of target-class
examples: `1` matches their overall scale, values below `1` are too narrow, and
large values remain too broad. The model remains unconditional and unchanged;
only the target-class and full-data statistics are fitted offline.

The useful signal is coarse because one mean and covariance cannot represent
every detail of a multimodal distribution. In this toy setting it produces
partial steering, not a full class-conditional generator. Notebook `04` asks
what changes when a state-dependent objective gradient replaces those fixed
summary statistics.
'''),
    ],
    "04_training_free_gradient_guidance.ipynb": [
        markdown(r'''
# 04 - Objective-gradient guidance during sampling

The class-minus-full correction uses fixed means and covariances. A
differentiable class objective can instead produce a direction that changes
with the current state. The generator remains frozen, but every active solver
evaluation now requires a backward pass through its denoised estimate.

This notebook asks two questions that a final target rate alone cannot answer:
when is the gradient informative, and what changes when its strength is
doubled?

**Time convention:** `t=0` is noise and `t=1` is data.
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
class-level objective and changes with the state and time. The implementation
normalizes each gradient before adding it, so the direction comes from the
objective while the chosen guidance strength sets the update size. A higher
class score alone does not guarantee a realistic endpoint.

**Before you run:** On the contour plot, should a gradient arrow point along a
level curve or across it? What happens to the gradient near a local maximum?
'''),
        code(r'''
grid_1d = torch.linspace(-4.5, 4.5, 80, device=device)
gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).requires_grad_(True)
log_posterior = class_log_probabilities(grid, stats).log_softmax(dim=1)[:, TARGET_CLASS]
posterior = log_posterior.exp()
gradient = torch.autograd.grad(log_posterior.sum(), grid)[0]

probability_image = posterior.reshape(gx.shape).detach().cpu()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
contour = ax.contourf(gx.cpu(), gy.cpu(), probability_image, levels=20, cmap="viridis")
arrow_points = grid.reshape(*gx.shape, 2)[::8, ::8].reshape(-1, 2)
arrow_gradient = gradient.reshape(*gx.shape, 2)[::8, ::8].reshape(-1, 2)
direction = arrow_gradient / arrow_gradient.norm(dim=1, keepdim=True).clamp_min(1e-6)
ax.quiver(
    arrow_points[:, 0].detach().cpu(), arrow_points[:, 1].detach().cpu(),
    direction[:, 0].detach().cpu(), direction[:, 1].detach().cpu(),
    color="white", alpha=0.75, scale=18,
)
fig.colorbar(contour, ax=ax, label="target-class posterior")
ax.set_title("Class objective and its gradient")
ax.set_aspect("equal")
plt.show()
'''),
        markdown(r'''
## 2. When should the gradient be used?

At very high noise, the denoised estimate carries little class information. At
later times it is more informative, but an aggressive edit can damage a nearly
formed sample. We compare an early window, a broad middle/late window at
strength `3.0`, and the same broad window at strength `6.0`. Every method starts
from the exact same noise tensor.

**Before you run:** Rank the three gradient settings by expected target-class
rate, matching distance, spread, and runtime. Does doubling the strength improve
all of the displayed endpoint measures, or does one begin to worsen?
'''),
        code(r'''
def gradient_guided_velocity(strength, start, end):
    def velocity(t, x):
        gate = window_gate(t, start, end)[:, None]
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
    "class-minus-full": make_class_minus_full_velocity(
        model, stats, target_class=TARGET_CLASS, strength=1.6
    ),
    "gradient, early": gradient_guided_velocity(3.0, 0.02, 0.25),
    "gradient, middle/late": gradient_guided_velocity(3.0, 0.15, 0.92),
    "gradient, strength 6.0": gradient_guided_velocity(6.0, 0.15, 0.92),
}

rows, paths = compare_steering_methods(
    methods, x0, target_reference, stats, target_class=TARGET_CLASS, steps=48,
)
gradient_table = pd.DataFrame(rows)
'''),
        code(r'''
plot_trajectory_comparison(
    {
        "baseline": paths["baseline"],
        "early gradient": paths["gradient, early"],
        "middle/late gradient": paths["gradient, middle/late"],
        "strength 6.0": paths["gradient, strength 6.0"],
    },
    target_reference=target_reference[:700],
)
plt.show()

relative = gradient_table.copy()
relative["runtime / baseline"] = relative["seconds"] / relative.loc[0, "seconds"]
display(relative[[
    "method", "target_rate", "target_wasserstein", "diversity_ratio",
    "runtime / baseline",
]].rename(columns={
    "target_rate": "target-class rate",
    "target_wasserstein": "empirical matching distance",
    "diversity_ratio": "diversity ratio",
}).round(3))
'''),
        markdown(r'''
## Cost model

With Heun and 48 steps, the baseline makes 96 forward evaluations. Gradient
guidance can add a backward pass to most of those evaluations. The exact runtime
depends on hardware, but the computational distinction is structural:

| Method | Offline target data | Inference gradients | Main online work |
|---|---:|---:|---|
| Baseline | no | no | model forward passes |
| Class-minus-full PCA | yes | no | forwards + small matrix-vector operations |
| Gradient guidance | yes or external classifier | yes | forwards + backward passes |

The strength-`6.0` setting is not a demonstrated failure in this sweep. It
improves the displayed target rate and matching distance, and its diversity
ratio moves closer to the target value of `1`, although the samples remain
broader than the target distribution. Both broad-window gradient settings are
substantially slower than the baseline because active guidance requires
backward passes. Stronger guidance can eventually damage quality or diversity,
but these rows do not establish that failure.

**Change one thing:** In the methods cell, change only the end of the
middle/late window from `0.92` to `0.60`. Rerun the comparison and ask whether
stopping before samples are nearly formed changes the trade-off.

Objective-gradient guidance adapts to each state but pays for online
backpropagation. Notebook `05` asks whether class information already readable
inside the frozen velocity network can support a forward-only intervention.
'''),
    ],
    "05_activation_steering_and_method_comparison.ipynb": [
        markdown(r'''
# 05 - Does a readable hidden feature provide control?

Notebook `04` obtained a steering direction by differentiating an external
objective at every active step. Here we first ask whether class is linearly
decodable from a hidden layer. We then ask a separate causal question: does one
direction fitted from those features change same-seed generated endpoints more
than zero and shuffled-direction controls? A strong readout does not guarantee
strong control.

We fit and test the probe on separate noised examples, learn one target-vs-rest
direction at `t=0.90`, and then intervene inside the network without an
inference-time backward pass.

**Time convention:** `t=0` is noise and `t=1` is data.
'''),
        code(SETUP),
        code(r'''
model, losses, _ = load_or_train_model(device=device)
reference_data, reference_labels = sample_labeled_mixture(30000, device=device)
stats = estimate_gaussian_stats(reference_data, reference_labels)
TARGET_CLASS = 2
model.eval()
_ = model.requires_grad_(False)
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

**Before you run:** The diagram changes a hidden feature but leaves the current
sample state untouched. Which part of the network converts that edit into a
different velocity?
'''),
        code(r'''
plot_activation_pipeline()
'''),
        markdown(r'''
## 2. When is class linearly decodable from the hidden layer?

At each time, we record activations from separate balanced training and test
sets with independent noise. A linear probe is fitted only on the training
activations and scored on the untouched test activations. A shuffled-label
probe provides a simple comparison without the true class relationship.
The comparison asks whether class is readable from the hidden state; it does
not yet show that editing the state will control generation.

**Before you run:** At `t=0.90`, should held-out accuracy remain above the
shuffled-label comparison? A strong result would show readable information, but
would it already demonstrate causal control?
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
        markdown(r'''
### Reading fixed-direction transfer

We estimate $d_{0.90}$ at late time and reuse it as a fixed intervention vector
over the window $[0.40, 0.90]$. The cosine plot shows that time-specific
directions are only weakly aligned with $d_{0.90}$ near the beginning of that
window. The intervention therefore does not follow the local class direction
at every time. It asks a narrower question: can one late-time direction,
repeated over part of the trajectory, produce a useful endpoint change?

**Before you run:** Predict where the held-out probe will approach chance and
where a direction learned at `t=0.90` will lose alignment with the local
target-vs-rest direction.
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

We collect the direction at `t=0.90`, where the held-out probe is strong, then
reuse it over a wider intervention window. Collection time and intervention
time are different choices. The cosine plot already warned that one late
direction is not the local class direction at every step.

**Before you run:** Which control is more informative here: zero strength or a
same-size direction fitted after shuffling labels? What would it mean if both
changed the target rate as much as the true direction?
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
    "class-minus-full": make_class_minus_full_velocity(
        model, stats, target_class=TARGET_CLASS, strength=1.6
    ),
    "gradient guidance": make_gradient_guided_velocity(
        model, stats, target_class=TARGET_CLASS, strength=3.0, start=0.15, end=0.92
    ),
}
'''),
        code(r'''
rows, paths = compare_steering_methods(
    methods, x0, target_reference, stats, target_class=TARGET_CLASS, steps=56,
)
comparison_table = pd.DataFrame(rows)
comparison_table["runtime / baseline"] = comparison_table["seconds"] / comparison_table.loc[0, "seconds"]
control_names = [
    "baseline", "activation 0.0", "shuffled activation", "activation steering",
]
control_table = comparison_table.set_index("method").loc[control_names].reset_index()
display(control_table[[
    "method", "target_rate", "target_wasserstein", "diversity_ratio",
]].rename(columns={
    "target_rate": "target-class rate",
    "target_wasserstein": "empirical matching distance",
    "diversity_ratio": "diversity ratio",
}).round(4))
'''),
        code(r'''
plot_trajectory_comparison(
    {
        "baseline": paths["baseline"],
        "shuffled direction": paths["shuffled activation"],
        "activation steering": paths["activation steering"],
    },
    target_reference=target_reference[:700],
    n_lines=18,
)
plt.show()
method_rows = comparison_table.set_index("method")
activation_effects = pd.Series({
    "activation target-rate gain": method_rows.loc["activation steering", "target_rate"] - method_rows.loc["baseline", "target_rate"],
    "shuffled-direction gain": method_rows.loc["shuffled activation", "target_rate"] - method_rows.loc["baseline", "target_rate"],
    "activation matching-distance reduction": method_rows.loc["baseline", "target_wasserstein"] - method_rows.loc["activation steering", "target_wasserstein"],
})
display(activation_effects.round(4))
'''),
        markdown(r'''
The zero-strength row checks the complete activation-edit code path without an
intervention. The shuffled direction checks whether an arbitrary fitted vector
of the same size produces the same endpoint effect. Only the true-versus-control
comparison addresses the causal usefulness of this fitted direction. Here the
true direction raises target occupancy more than the shuffled control, but it
also stretches many endpoints beyond the target cloud. That is a causal but
imperfect intervention, not evidence of a clean class axis.

## 4. Compare three intervention sites

We now keep one tested setting for each mechanism. These are selected operating
points with different windows, signal sources, and online costs; the comparison
does not establish one generally best method.
'''),
        code(r'''
method_names = [
    "baseline", "class-minus-full", "gradient guidance", "activation steering",
]
method_table = comparison_table.set_index("method").loc[method_names].reset_index()
display(method_table[[
    "method", "target_rate", "target_wasserstein", "diversity_ratio",
    "runtime / baseline",
]].rename(columns={
    "target_rate": "target-class rate",
    "target_wasserstein": "empirical matching distance",
    "diversity_ratio": "diversity ratio",
}).round(4))
plot_method_tradeoff(method_table)
'''),
        markdown(r'''
## 5. Reading the method comparison

The methods intervene at different places. The class-minus-full correction
changes a denoised estimate early, objective-gradient guidance differentiates a
class score, and activation steering edits a hidden feature. Target-class rate
describes control, empirical matching distance describes agreement with target
examples, diversity describes spread, and runtime exposes the price of online
backpropagation. No method is best without naming the trade-off that matters.

The probe and intervention answer different questions: held-out decoding shows
that class information is readable from the hidden layer, while paired
sampling tests whether one fixed late-time direction is useful over the chosen
intervention window. Neither result implies a unique class axis or a
time-invariant representation. Activation steering can be weaker than the
probe accuracy suggests because the direction is fixed while the representation
rotates across time, and because decodability does not guarantee that the
downstream velocity is sensitive in the same direction.

**Change one thing:** Replace `REFERENCE_T = 0.90` with `0.70`, rerun the probe
and intervention sections, and compare both direction alignment and endpoint
control. Do not compare target rate alone.

Across the toy series, we have now compared three intervention sites. Notebook
`06` carries only the class-minus-full denoiser correction to a pretrained
image model and asks whether same-seed image pairs reveal a measurable but
partial shift.
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
