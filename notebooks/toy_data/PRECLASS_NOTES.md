# Reading Guide: From Noise to Post-Hoc Steering

## Why these notes begin with generation

Steering is easiest to misunderstand when the generator itself is still a
black box. A diffusion or flow model does not retrieve a stored example, and a
deterministic sampler does not produce the same output every time. Generation
starts from a fresh random draw, follows a learned local rule, and accumulates
many small updates into a new sample.

The seven notes build that picture before adding control. They use small
examples because every state, vector field, trajectory, and failure can be
drawn. The last note then asks which part of the argument survives in NVIDIA's
pretrained unconditional CIFAR-10 EDM.

The recurring question is:

> How can we bias a frozen unconditional generator toward a class without
> pulling every trajectory toward one chosen example or coordinate?

The answer develops in stages: learn an unconditional flow, understand the
sampler, reinterpret velocity as a denoised estimate, and then compare three
post-hoc interventions. Labels never enter the toy generator during training.
They are used later to fit or evaluate steering signals.

## How to read the notebooks

Each note follows the same short rhythm:

1. A visible phenomenon raises one concrete question.
2. A **Before you run** prompt asks for a prediction.
3. One paired experiment changes a single mechanism while keeping the initial
   noise and other settings fixed.
4. The figure is interpreted immediately, including what it does not show.
5. A **Change one thing** prompt exposes a bounded parameter worth exploring.

The executed notebooks are the reading copies. The source notebooks remain
output-free so that rerunning them produces a clean experiment rather than a
mixture of old and new results.

## The argument across the series

### 00: How does noise become a new sample?

The first note separates four objects that are often conflated: a data
distribution, a noising path, a learned velocity field, and a numerical
sampler. The figures show a distribution changing across time rather than a
single clean example being reconstructed from its own corrupted copy. This
leaves a practical question for the next note: can a neural network learn the
required local velocities from random training pairs?

### 01: Can local velocity predictions generate the whole distribution?

A small MLP is trained by flow matching on three unlabeled Gaussian clusters.
The notebook connects the training target to sampled trajectories and then
checks the generated point cloud, not just the loss curve. A noisy circle asks
whether the model can form thin curved support. Eight separated Gaussians make
mode coverage, occupancy imbalance, and the flow map easy to inspect. These
transfer examples show why one scalar loss is not enough to understand a
generator.

### 02: Same field, different deterministic samplers

Euler, Heun, and RK4 integrate the same learned ODE from exactly the same noise
tensor. Their paired endpoints reveal numerical error without confusing it
with different random draws. Only after that comparison does the notebook
derive

```text
D_theta(x_t,t) = x_t + (1-t) v_theta(x_t,t),
```

the clean estimate associated with the linear flow path. This conversion
matters because the steering methods are easier to state as edits to a
denoiser, even though the toy network predicts velocity.

### 03: From a Gaussian denoiser to a class-minus-full correction

Before using covariance for control, the note tests a Gaussian/PCA posterior
mean on held-out noisy MNIST digits. The eigendigits show what the model keeps,
and reconstruction error shows where the approximation helps. The same idea is
then made fully visible in 2D: subtract the full-data Gaussian denoiser from a
target-class Gaussian denoiser and add that difference during an early,
high-noise interval. The steering paper calls this **noise alignment**. In the
notes, **class-minus-full correction** is used first because it states exactly
what is computed.

This is a distributional correction, not attraction to a point. Means and
covariances can provide coarse class bias, but they cannot represent all of a
non-Gaussian class distribution.

### 04: Objective-gradient guidance during sampling

A differentiable class objective provides a state-dependent direction rather
than one Gaussian approximation. The notebook draws the gradient field and
compares early, middle-to-late, and doubled-strength interventions from paired
noise. In this bounded sweep, the doubled strength improves every displayed
endpoint measure; the demonstrated disadvantage of gradient guidance is greater
inference cost. The added flexibility has a computational cost: the sampler
must differentiate through the clean estimate whenever guidance is active.

### 05: Does a readable hidden feature provide control?

The final toy note probes an internal MLP layer across time. A held-out linear
probe measures when class information becomes readable, while direction
alignment shows whether one fixed edit remains meaningful as the representation
changes. The intervention then tests causality with zero-strength and shuffled-
direction controls. This distinction is central: a feature can encode class
information without being a strong control direction for the downstream
velocity.

The closing comparison places the three mechanisms side by side: a cheap
class-minus-full correction, flexible online gradient guidance, and a
forward-only hidden-feature edit whose effect depends on layer and time.

### 06: A class-minus-full correction in an unconditional CIFAR-10 EDM

The image note uses NVIDIA's pretrained unconditional CIFAR-10 EDM. It receives
no class label. The sampler adds a full-image PCA-basis cat-minus-full denoiser
correction only at high noise and compares baseline, zero-strength, target, and
wrong-class runs from identical seeds. Paired images and one trajectory make
the intervention visible; target prediction, target-feature distance, and
retained feature variation quantify the tradeoff.

The result is deliberately narrow. It demonstrates a measurable class-specific
shift while preserving substantial variation, not reliable class-conditional
generation. Full asset, calibration, and evaluator details live in
[the experiment record](optional/CIFAR10_BRIDGE.md) rather than interrupting the
reader-facing notebook.

## Minimal background

Readers should be comfortable with Python arrays and plots, basic PyTorch, and
vectors, means, and variances. The notes introduce ODE integration, Gaussian
conditioning, PCA, classifiers, and hidden representations when they first
become useful. No stochastic calculus is assumed.

The complete generator in the toy notes is

```text
base Gaussian + learned velocity model + deterministic ODE sampler.
```

Training uses data to fit the velocity model. Generation starts from fresh
noise and has no paired clean answer. Once a noise tensor, checkpoint, solver,
and settings are fixed, the ODE rollout is deterministic. Reusing that exact
noise tensor across methods is therefore an experimental control; it is not a
steering method.

## Terms used consistently

| Term | Meaning here |
|---|---|
| unconditional generator | A generator that receives no requested class or prompt |
| base distribution | The easy Gaussian distribution sampled at the start |
| velocity field | A local prediction of how the current state should change with time |
| deterministic sampler | A numerical ODE solver whose trajectory is fixed by its initial state and settings |
| same-seed comparison | Reuse the exact initial noise tensor to isolate the effect of a method |
| denoised estimate | The clean-data estimate obtained from the linear-path velocity parameterization |
| class-minus-full correction | Target-class Gaussian/PCA denoiser minus the full-data Gaussian/PCA denoiser |
| noise alignment | The paper's name for the class-minus-full distributional correction |
| objective-gradient guidance | An inference-time gradient of a differentiable target objective |
| activation steering | A fitted direction added to an internal feature layer |

The training path is independent conditional flow matching with a linear
source-data coupling. No optimal-transport coupling is solved. Post-hoc fitting
is described explicitly rather than being hidden under the phrase "training
free."

## Reading quantitative comparisons

Paired method comparisons hold the checkpoint, initial noise tensor, target,
solver, step count, sample count, and evaluator fixed. No single number is
treated as a certificate of sample quality. The notes use an empirical
Wasserstein matching distance together with visible samples, mode coverage,
mean error, and diversity. The diversity ratio divides generated total variance
by target-class total variance, so `1` matches the target scale, values below
`1` are too narrow, and large values remain too broad. For the circle, radial
error and angular occupancy separate ring thickness from missing arcs. For
eight Gaussians, component coverage and occupancy reveal missing or
overrepresented modes.

The reported empirical Wasserstein matching distance is the mean distance under
a minimum-cost one-to-one matching between equal-size deterministic subsets.
It is useful for these small 2D comparisons, but it remains a finite-sample
summary and should be read with the figures.

## References

- Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747).
- Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364).
- Li, Dai, and Qu, [*Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure*](https://arxiv.org/abs/2410.24060).
- Song, Meng, and Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502).
- Wang, Belkin, and Wang, [*General and Efficient Steering of Diffusion Models*](https://arxiv.org/abs/2602.11395).
- Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239).
