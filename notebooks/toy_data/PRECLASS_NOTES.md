# Pre-Class Notes: Flow Matching and Post-Hoc Steering

## Purpose

These six notebooks and the accompanying CIFAR-10 experiment are preparatory
notes for undergraduates who may have limited background in differential
equations, diffusion models, or representation learning. They build visual
intuition before class discussion.

No prior generative-model taxonomy is assumed. Notebook `00` introduces the
minimum vocabulary visually before using path equations.

The root `README.md` links to output-bearing snapshots for readers who do not
want to run training first. The notebooks in this directory remain clean,
output-free source files.

The notebooks use one unconditional flow-matching MLP. Class labels never enter
its training input. Labels appear later only for evaluation and for constructing
post-hoc steering signals.

Notebook `01` also trains separate unlabeled circle and eight-Gaussian models
to make continuous geometry and multimodal flow maps easy to see. Those
illustration models are not used by the steering notebooks.

## Useful Background

Readers should be comfortable with:

- Python functions, arrays, plotting, and dictionaries;
- PyTorch tensors, modules, optimizers, and automatic differentiation;
- vectors, matrix multiplication, means, variances, and basic probability;
- the idea that an ODE specifies a local rate of change.

Gaussian distributions, covariance, PCA, numerical integration, classifiers,
and hidden representations are introduced visually as they are needed.

### Generative-model vocabulary used throughout

The training dataset supplies examples of an unknown distribution. A base
Gaussian supplies easy-to-sample random starting points. The neural model learns
a local prediction, and a sampler turns those predictions into a complete
trajectory. In this series, the full generator is therefore

```text
base distribution + trained velocity model + ODE sampler.
```

Training uses data examples to fit the velocity model. Generation starts from
fresh noise and does not receive a paired clean answer. Different noise seeds
provide diversity even when the ODE sampler is deterministic.

## Suggested Reading Order

### 00: Diffusion and flow-matching foundations

We first need a concrete picture of what turns noise into a sample. This note
separates the source distribution, learned velocity field, and numerical
sampler, then draws forward noising and source-to-data paths.

### 01: Unconditional flow-matching training

Training one small model exposes the connection between the flow-matching loss
and generated samples. A three-component mixture checks familiar modes, a noisy
circle tests curved continuous support, and eight Gaussians reveal missing-mode
and occupancy failures.

### 02: Denoisers and deterministic samplers

The next methods are written in denoiser language even though our network
predicts velocity. This note derives the conditional-expectation meaning of
`D_theta(x_t,t) = x_t + (1-t)v_theta(x_t,t)`, explains the endpoint caveat,
and compares Euler, Heun, and RK4 from exactly the same initial noise tensor.

### 03: Gaussian/PCA noise alignment

Before using a Gaussian correction for steering, we test whether low-rank
covariance structure can denoise held-out MNIST pixels. We then visualize the
target-class minus full-data denoiser in 2D and apply it only during a
high-noise window. A zero-strength control separates partial coarse steering
from full class-conditional generation.

### 04: Post-hoc gradient guidance

Covariance guidance is efficient but restrictive. A differentiable class
objective is more flexible, so this note visualizes its gradient and compares
its behavior and backward-pass cost with the forward-only Gaussian/PCA
correction.

### 05: Hidden-feature steering and method comparison

The final toy note asks whether a forward-only edit can act inside the frozen
network. It first measures whether class is readable from held-out hidden
features, then tests a target-vs-rest direction against shuffled controls and
compares activation steering with noise alignment and gradient guidance.

## Terminology

| Phrase | Meaning in these notes |
|---|---|
| generative model | A learned procedure for producing new samples that resemble a data distribution |
| generative sampling | Transform fresh base noise into a new data-like sample without a paired known clean answer |
| unconditional generator | A generator that receives no requested class or prompt |
| base distribution | The easy Gaussian source used to start generation |
| I-CFM | Independent source/data coupling with a linear conditional path |
| deterministic sampler | The trajectory is fixed once model, solver, settings, and initial noise are fixed |
| paired initial noise | Reusing the exact initial tensor to isolate method effects |
| denoised estimate | Clean-data estimate derived from the linear-path velocity parameterization |
| noise alignment | Target-class Gaussian/PCA denoiser minus full-data Gaussian/PCA denoiser at high noise |
| post-hoc gradient guidance | Frozen generator plus an inference-time objective gradient |
| activation steering | A learned direction added to an internal feature layer |
| gradient-free inference | No backward pass during sampling; offline statistics or direction fitting may still be required |

The path in these notebooks is I-CFM, not OT-CFM: no optimal transport coupling
is solved. Post-hoc fitting should not be called "no training" without stating
what was fit. Paired initial noise is an evaluation control, not the
noise-alignment steering method.

## Reading the Comparisons

When two methods are compared, the checkpoint, initial noise tensor, target
reference, solver, step count, sample count, and evaluation statistics are held
fixed. The tables show several quantities because target-class rate alone does
not establish that a steering method preserves fidelity or diversity:

- target-class rate;
- Wasserstein distance to target examples;
- mean error relative to the target distribution;
- diversity ratio;
- wall-clock time.

The circle adds radial and angular-occupancy checks. The eight-Gaussian flow map
adds mode coverage, component occupancy error, and within-component spread.
These are broad sanity checks, not a claim that the learned and target
densities are exactly equal.

The reported Wasserstein distance is a finite-sample estimate: equal-size
subsets are paired by minimum total Euclidean matching cost, and the mean
matched distance is reported. Lower is better. The visual construction appears
in notebook `01`; it should be read with coverage and diversity rather than as
a complete quality certificate.

## Questions for Class

1. Which components together form the complete generator in these notes?
2. Why can a deterministic sampler produce diverse outputs from different noise seeds?
3. Why is a denoised estimate useful when the trained network predicts velocity?
4. Why does deterministic sampling make paired initial noise informative?
5. Why is Gaussian/PCA guidance concentrated at high noise?
6. Why can gradient guidance be more expensive than forward-only guidance?
7. At what times do hidden features become class-informative?
8. Which conclusions from the 2D examples require new evidence before applying
   them to image diffusion models?

## CIFAR-10 image experiment

The toy examples make mechanisms visible; CIFAR-10 checks whether the same
distribution-level correction measurably changes a real pretrained nonlinear
generator. The experiment uses the official unconditional NVIDIA EDM
checkpoint with deterministic sampling and a class-minus-full PCA denoiser
correction. Its manifest locks checkpoint hashes, PCA provenance, evaluator,
calibration/evaluation seed splits, and paired zero- and wrong-class controls.
Read the [executed CIFAR-10 notebook](executed/cifar10_steering_with_unconditional_edm.ipynb)
and [experiment contract](optional/CIFAR10_BRIDGE.md).

## References

- Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747).
- Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364).
- Li, Dai, and Qu, [*Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure*](https://arxiv.org/abs/2410.24060).
- Song, Meng, and Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502).
- Wang, Belkin, and Wang, [*General and Efficient Steering of Diffusion Models*](https://arxiv.org/abs/2602.11395).
- Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239).
