# Background Notes

This file supplies the minimum derivations needed by the notebooks. The main
pre-class notes stay visual; these details explain why the formulas are valid.

## 1. What a generative model learns

We observe a finite training dataset sampled from an unknown data distribution.
A generative model learns a procedure whose new outputs resemble that
distribution. It is not a lookup table. During ordinary generation, fresh base
noise is transformed into a new output; no paired clean answer is supplied.

For a deterministic flow model, the complete sampling procedure is

```text
fresh base noise -> trained velocity model + ODE sampler -> generated sample.
```

The base distribution is deliberately simple, usually Gaussian. Different
initial noise seeds provide different starting points and therefore different
outputs. "Deterministic sampler" means that one fixed seed follows one fixed
trajectory when the model and numerical settings are unchanged; it does not
mean all seeds produce the same sample.

An unconditional model receives no requested class or prompt. A conditional
model receives such information as an input. The model in these notes is
unconditional: labels are absent from its training input and are used later
only to construct post-hoc controls and evaluate their effects.

Diffusion and flow matching both learn time-dependent structure between noise
and data. Diffusion models commonly predict noise, a score, or a denoised
estimate and may use stochastic or deterministic sampling. Flow matching learns
a velocity field and integrates an ODE. In these notes, the velocity network is
the learned neural component; the source distribution and ODE solver complete
the generator.

## 2. Linear I-CFM path

Sample independent endpoints, using base standard deviation `s = 1.8` in these
notebooks:

```text
x_noise = s * epsilon,   epsilon ~ N(0,I)
x_data  ~ empirical data distribution
```

Define

```text
x_t = (1-t) x_noise + t x_data
```

Differentiating with respect to `t` gives the conditional target velocity

```text
u_t = x_data - x_noise.
```

The neural model minimizes mean-squared error between `v_theta(t,x_t)` and this
target. The target for one sampled pair is constant, while the learned population
field depends on both location and time because many sampled paths overlap.

## 3. Velocity-to-denoiser conversion

First consider one training pair, for which both endpoints are known. Write

```text
u = x_data - x_noise.
```

The path identity can then be rearranged exactly:

```text
x_data = x_t + (1-t) u.
```

At generation time the network sees only `(x_t,t)`, not `x_noise`,
`x_data`, or the pair-specific `u`. Under mean-squared-error training, the
population-optimal prediction is the conditional average

```text
v*(x_t,t) = E[u | x_t,t].
```

Because `x_t` and `t` are fixed inside this conditional expectation,

```text
x_t + (1-t) v*(x_t,t)
    = E[x_t + (1-t)u | x_t,t]
    = E[x_data | x_t,t].
```

This motivates the trained model's denoiser view:

```text
D_theta(x_t,t) = x_t + (1-t) v_theta(t,x_t).
```

The estimate is a conditional average over plausible clean endpoints. It need
not equal one pair's endpoint; where several outcomes are plausible it may lie
between modes. In image models, posterior averaging is one reason a
mean-squared-error estimate can look smooth or blurry.

For `t < 1`, the same information can be written as

```text
v_theta(t,x_t) = (D_theta(x_t,t) - x_t) / (1-t).
```

This inverse conversion becomes ill-conditioned near `t=1` and is undefined
at the endpoint. Both formulas are specific to this linear path. They are a
parameterization bridge, not evidence that the MLP was trained with a separate
diffusion-denoising objective.

## 4. Deterministic ODE sampling

Generation solves

```text
dx/dt = v_theta(t,x),   x(0) = x_noise.
```

Euler, Heun, and RK4 are numerical approximations to the same ODE. Once the
initial tensor, model, solver, step count, and floating-point environment are
fixed, the rollout contains no further random draw.

This is why paired initial noise is a strong experimental control: two methods
start from the same individual samples, not merely from the same distribution.

## 5. Additive-noise coordinates

For `t > 0`, divide the linear path by `t` and substitute
`x_noise = s * epsilon`:

```text
y = x_t / t = x_data + sigma_eff * epsilon
sigma_eff = s (1-t) / t.
```

The flow state can therefore be viewed in additive standard-Gaussian-noise
coordinates. Small `t` corresponds to large `sigma_eff` and high noise. Keeping
the factor `s` is necessary because the base distribution used here is not unit
variance.

## 6. Gaussian/PCA denoiser

Assume clean data are Gaussian with mean `mu` and covariance `C`, and observe

```text
y = x_data + sigma * epsilon,   epsilon ~ N(0,I).
```

The posterior mean is

```text
D(y,sigma) = mu + C (C + sigma^2 I)^(-1) (y-mu).
```

Along directions with high data variance, the denoiser trusts the observation
more. Along low-variance directions, it shrinks more strongly toward the mean.
In high dimensions, the eigendecomposition of `C` gives the usual PCA form.

If `C = U diag(lambda_i) U^T`, then each centered principal-component
coordinate is multiplied by

```text
lambda_i / (lambda_i + sigma^2).
```

This makes the behavior concrete: directions with large dataset variance are
retained more strongly, while low-variance directions shrink toward the mean.
Notebook `03` visualizes these directions as MNIST eigendigits and verifies
on held-out noisy images that both an all-digit and a digit-specific Gaussian
posterior mean reduce pixel MSE.

[Li, Dai, and Qu (NeurIPS 2024)](https://arxiv.org/abs/2410.24060) report that
diffusion denoisers become increasingly linear in a generalization regime and
that their best linear approximations are close to empirical Gaussian
denoisers. Their experiments include FFHQ and checks on CIFAR-10, AFHQ, and
LSUN-Churches. That paper does not provide the ImageNet experiment for these
notes. Our larger image evidence is instead the manifest-locked CIFAR-10
experiment using NVIDIA's official unconditional EDM checkpoint.

The notebook noise-alignment signal is

```text
Delta D = D_target_class - D_full_data.
```

It is computed from labeled examples offline and applied only at high noise.
This is distinct from reusing the same initial noise tensor for evaluation.

## 7. Gradient guidance

Suppose `s_c(x)` is a differentiable score for target class `c`. The gradient

```text
grad_x log s_c(x)
```

is the local direction that most rapidly increases the target log score. In the
notebooks, the score is evaluated on `D_theta(x_t,t)`, so automatic differentiation
passes through both the clean estimate and velocity model at every active solver
evaluation.

This is post-hoc because the generator is frozen, but it is not cheap or
gradient-free at inference.

## 8. Why hidden-feature steering can work

The MLP computes

```text
(x,t) -> hidden feature h -> velocity v.
```

If target and background examples separate in `h`, a target-vs-rest direction
can change the downstream velocity in a class-relevant way. Repeated local
velocity changes accumulate over an ODE trajectory.

The notebook intervention is

```text
h_steered = h + alpha * g(t) * RMS(h) * d_reference,
```

where `d_reference` is a unit direction learned from activations, `alpha` is a
strength, and `g(t)` is a time-window gate. This edits a hidden feature, not the
sample position. The downstream layers decide how that edit changes velocity.
A successful linear probe shows that class information is readable from `h`;
it does not by itself show that moving along the probe-related direction will
control generation.

Three questions keep readability and control separate:

1. a linear probe can decode class from the feature;
2. direction alignment is measured rather than assumed across time;
3. injection improves control without unacceptable fidelity or diversity loss.

## 9. Reading the circle and eight-Gaussian flow maps

The circle asks whether the model can form thin, continuous curved support
rather than merely choosing among a few centers. Radial error checks ring
thickness and location; angular occupancy checks for gaps; Wasserstein distance
compares the generated and target point clouds.

The eight-Gaussian example stores a deterministic trajectory for every sampled
base point. Only after sampling, each endpoint is assigned to its nearest data
component for coloring and occupancy diagnostics. That assignment is an
evaluation label. It is not provided to the model and does not exert a force
toward a component center.

Mode coverage asks whether every data component receives nontrivial generated
mass. Occupancy error compares generated and target component frequencies. The
mean nearest-component distance checks within-component spread. No one metric
establishes distributional equality.

For Wasserstein distance, the implementation chooses equal-size deterministic
subsets and solves a minimum-cost one-to-one matching using Euclidean distance.
It reports the mean length of the matched pairs. This is a finite-sample
empirical estimate: lower is better, but separate coverage and spread checks
can still reveal failures hidden by one scalar summary.

## 10. From the toy mechanisms to the image experiment

The toy model is not a U-Net, image diffusion model, or implementation of the
full NA-RFM pipeline. It does implement the following mechanisms in a setting
where every state and vector is visible:

- a linear flow-matching path and deterministic ODE sampler;
- high-noise target/full Gaussian denoiser correction;
- post-hoc class-energy gradients;
- offline feature-direction learning and online hidden-feature injection;
- paired ablations over timing, strength, quality, diversity, and runtime.

Notebook `03` already moves the Gaussian denoiser to MNIST pixels, and the
Notebook `06` image experiment uses a real pretrained EDM denoiser with
high-dimensional PCA statistics. Activation steering of an image U-Net still
requires a specified layer, activation tensors, held-out probes, matched
controls, stronger quality metrics, and broader validation.

| Idea | Toy realization | What changes at image scale |
|---|---|---|
| High-noise noise alignment | target-class Gaussian denoiser minus full-data Gaussian denoiser | Notebook `06` uses low-rank PCA statistics over CIFAR-10 images |
| Objective-gradient guidance | differentiate a class score through the 2D denoised estimate | a larger model would require an image-level objective and online backpropagation |
| Activation collection | record one MLP hidden layer on labeled, noised points | an image experiment must choose a particular network block and tensor representation |
| Activation direction | covariance-aware target-versus-rest direction | full NA-RFM uses a richer feature-learning construction |
| Online activation edit | add one fixed direction over a time window | image tensors require a defined broadcast, normalization, and matched control |

The table is a map between ideas, not an equivalence claim. Notebook `06`
extends only noise alignment to images; the hidden-feature experiment remains a
separate next step.
