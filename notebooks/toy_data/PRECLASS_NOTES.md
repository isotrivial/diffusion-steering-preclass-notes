# Pre-Class Notes: Flow Matching and Post-Hoc Steering

## Purpose

These six notebooks are preparatory notes for undergraduates who may have
limited background in differential equations, diffusion models, or
representation learning. They are intended to establish shared vocabulary and
visual intuition before class discussion.

The notebooks use one unconditional flow-matching MLP. Class labels never enter
its training input. Labels appear later only for evaluation and for constructing
post-hoc steering signals.

Notebook `01` also trains a separate unlabeled eight-Gaussian model solely to
make the learned flow map easy to see. That illustration model is not used by
the steering notebooks.

## Useful Background

Readers should be comfortable with:

- Python functions, arrays, plotting, and dictionaries;
- PyTorch tensors, modules, optimizers, and automatic differentiation;
- vectors, matrix multiplication, means, variances, and basic probability;
- the idea that an ODE specifies a local rate of change.

Gaussian distributions, covariance, PCA, numerical integration, classifiers,
and hidden representations are introduced visually as they are needed.

## Suggested Reading Order

### 00: Diffusion and flow-matching foundations

Visualize clean classes at several noise levels, inspect straight
source-to-data paths, and distinguish four objects: noising path, neural model,
denoiser, and sampler.

### 01: Unconditional flow-matching training

Train I-CFM on an unlabeled three-component mixture, integrate the learned ODE,
and inspect generated samples and the time-dependent vector field. A second
unconditional model on eight Gaussian modes shows the learned deterministic
flow map at several times and checks mode coverage and occupancy.

### 02: Denoisers and deterministic samplers

Derive `D_theta(x_t,t) = x_t + (1-t)v_theta(x_t,t)` and compare Euler, Heun,
and RK4 from exactly the same initial noise tensor.

### 03: Gaussian/PCA noise alignment

Fit target-class and full-data Gaussian statistics, visualize the difference
between their denoisers, and apply that correction only during a high-noise
window. A zero-strength control and explicit verdict separate partial coarse
steering from full class-conditional generation.

### 04: Post-hoc gradient guidance

Visualize a class-posterior objective and differentiate it through the model's
clean estimate during sampling. Compare its behavior and cost with the
forward-only Gaussian/PCA correction.

### 05: Hidden-feature steering and method comparison

Measure class readability across time, visualize hidden features, learn a
covariance-aware target-vs-rest direction, and compare activation steering with
noise alignment and gradient guidance. Probes use held-out examples, a
shuffled-label control, a fresh-batch confusion matrix, and an independent null
direction.

## Terminology

| Phrase | Meaning in these notes |
|---|---|
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
- sliced Wasserstein distance to target examples;
- mean error relative to the target distribution;
- diversity ratio;
- wall-clock time.

The eight-Gaussian flow map additionally reports mode coverage, component
occupancy error, within-component spread, and SWD. These are broad sanity checks,
not a claim that the learned and target densities are exactly equal.

## Questions for Class

1. Why is a denoised estimate useful when the trained network predicts velocity?
2. Why does deterministic sampling make paired initial noise informative?
3. Why is Gaussian/PCA guidance concentrated at high noise?
4. Why can gradient guidance be more expensive than forward-only guidance?
5. At what times do hidden features become class-informative?
6. Which conclusions from the 2D examples require new evidence before applying
   them to image diffusion models?

## Optional image bridge

CIFAR-10 is not a seventh core note. The optional executable bridge uses the
official unconditional NVIDIA EDM checkpoint with deterministic sampling and a
class-minus-full PCA denoiser correction. Its manifest locks checkpoint hashes,
PCA provenance, evaluator, calibration/evaluation seed splits, and paired zero
and wrong-class controls. See `optional/CIFAR10_BRIDGE.md`.

## References

- Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747).
- Song, Meng, and Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502).
- Wang, Belkin, and Wang, [*General and Efficient Steering of Diffusion Models*](https://arxiv.org/abs/2602.11395).
- Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239).
