# Reading Guide: Flow Matching and Post-Hoc Steering

## One question across the series

These seven notes follow one question: how can a frozen unconditional generator
be steered without pulling a sample toward a chosen point? The answer requires
several ideas in order. We first separate the source distribution, learned
model, and sampler; then train and inspect a flow; then translate velocity into
a denoised estimate; and only then compare distributional, gradient-based, and
hidden-feature interventions.

The intended reader may have limited background in differential equations,
diffusion models, or representation learning. No generative-model taxonomy or
stochastic calculus is assumed. Each mathematical object is introduced in
words, equations, and figures before it is used for steering.

The toy steering notes use one unconditional flow-matching MLP. Class labels
never enter its training input. Labels appear later only to fit post-hoc
signals and to compare their effects.

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

## Reading path

### 00: Diffusion and flow-matching foundations

We first need a concrete picture of what turns noise into a sample. This note
separates the source distribution, learned velocity field, and numerical
sampler, then leaves the central unresolved question: can the velocity field
actually be learned?

### 01: Unconditional flow-matching training

Training one small model exposes the connection between the flow-matching loss
and generated samples. A three-component mixture checks familiar modes, a noisy
circle tests curved continuous support, and eight Gaussians reveal missing-mode
and occupancy failures. These visible failures motivate a closer look at what
the model predicts and how the sampler follows it.

### 02: Denoisers and deterministic samplers

The next methods are written in denoiser language even though our network
predicts velocity. This note derives the conditional-expectation meaning of
`D_theta(x_t,t) = x_t + (1-t)v_theta(x_t,t)`, explains the endpoint caveat,
and compares Euler, Heun, and RK4 from exactly the same initial noise tensor.
That paired deterministic setup becomes the common comparison tool for every
steering method that follows.

### 03: Gaussian/PCA noise alignment

Before using a Gaussian correction for steering, we test whether low-rank
covariance structure can denoise held-out MNIST pixels. We then visualize the
target-class minus full-data denoiser in 2D and apply it only during a
high-noise window. A zero-strength control separates partial coarse steering
from full class-conditional generation. Its limitation is equally important:
means and covariances describe only coarse structure.

### 04: Post-hoc objective-gradient guidance

Covariance guidance is efficient but restrictive. A differentiable class
objective is more flexible, so this note visualizes its gradient and compares
its behavior and backward-pass cost with the forward-only Gaussian/PCA
correction. This raises the next question: can useful class information be
accessed without an online backward pass?

### 05: Hidden-feature steering and method comparison

The final toy note asks whether a forward-only edit can act inside the frozen
network. It first measures whether class is readable from held-out hidden
features, then tests a target-vs-rest direction against shuffled controls and
compares activation steering with noise alignment and objective-gradient
guidance.

### 06: Noise alignment in an unconditional CIFAR-10 EDM

The final note carries the high-noise class-minus-full PCA correction to
NVIDIA's pretrained unconditional CIFAR-10 EDM. Paired images show how an early
change can alter a late sample, while target preference, target-feature
distance, and retained variation show why the result is meaningful but partial.

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

## Questions to carry into class

1. Which components together form the complete generator in these notes?
2. Why can a deterministic sampler produce diverse outputs from different noise seeds?
3. Why is a denoised estimate useful when the trained network predicts velocity?
4. Why does deterministic sampling make paired initial noise informative?
5. Why is Gaussian/PCA guidance concentrated at high noise?
6. Why can gradient guidance be more expensive than forward-only guidance?
7. At what times do hidden features become class-informative?
8. Which conclusions from the 2D examples require new evidence before applying
   them to image diffusion models?

The [executed notebooks](executed/) contain all figures and tables for reading.
The output-free sources live one directory above. Reproduction details for
Notebook `06` are kept separately in
[the CIFAR-10 experiment record](optional/CIFAR10_BRIDGE.md), so the main note
can stay focused on the mechanism and its interpretation.

## References

- Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747).
- Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364).
- Li, Dai, and Qu, [*Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure*](https://arxiv.org/abs/2410.24060).
- Song, Meng, and Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502).
- Wang, Belkin, and Wang, [*General and Efficient Steering of Diffusion Models*](https://arxiv.org/abs/2602.11395).
- Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239).
