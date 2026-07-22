# Background Notes

This file supplies the minimum derivations needed by the notebooks. The main
pre-class notes stay visual; these details explain why the formulas are valid.

## 1. What a generative model learns

We observe a finite training dataset sampled from an unknown data distribution.
A generative model learns a procedure whose new outputs resemble that
distribution. It is not a lookup table and it does not need to reconstruct one
particular training example during generation.

Generative sampling and reconstruction answer different questions. Sampling
draws fresh base noise and produces a new data-like output with no paired clean
answer. Reconstruction starts from an observation tied to one particular item
and estimates that same item. A generative model may still compute a denoised
estimate internally during sampling; that does not turn sampling into retrieval
or reconstruction.

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

Rearrange the path:

```text
x_data = x_t + (1-t) (x_data - x_noise).
```

Replacing the exact pair velocity with the learned velocity gives

```text
D_theta(x_t,t) = x_t + (1-t) v_theta(t,x_t).
```

This is a clean-data estimate for the linear parameterization. It does not mean
the MLP was trained with a separate diffusion-denoising loss.

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

The notebooks check three empirical requirements instead of assuming them:

1. a linear probe can decode class from the feature;
2. direction alignment is measured rather than assumed across time;
3. injection improves control without unacceptable fidelity or diversity loss.

## 9. Reading the eight-Gaussian flow map

The eight-Gaussian example stores a deterministic trajectory for every sampled
base point. Only after sampling, each endpoint is assigned to its nearest data
component for coloring and occupancy diagnostics. That assignment is an
evaluation label. It is not provided to the model and does not exert a force
toward a component center.

Mode coverage asks whether every data component receives nontrivial generated
mass. Occupancy error compares generated and target component frequencies. The
mean nearest-component distance checks within-component spread, while sliced
Wasserstein distance compares the full generated and target point sets. No one
metric establishes distributional equality.

For each two-dimensional unit direction `d`, SWD projects both clouds onto a
line, sorts the projected values, and compares corresponding sorted positions.
With `K` directions and `n` points, the implementation uses

```text
SWD = sqrt(mean over k,i of (sort(x @ d_k)[i] - sort(y @ d_k)[i])^2).
```

Repeating across directions makes the comparison sensitive to more than the
mean of the cloud. Lower is better. A small SWD is useful evidence of broad
distributional agreement, but separate mode-coverage and spread checks can
still reveal failures hidden by one scalar summary.

## 10. Scope boundary

The toy model is not a U-Net, image diffusion model, or implementation of the
full NA-RFM pipeline. It does implement the following mechanisms in a setting
where every state and vector is visible:

- a linear flow-matching path and deterministic ODE sampler;
- high-noise target/full Gaussian denoiser correction;
- post-hoc class-energy gradients;
- offline feature-direction learning and online hidden-feature injection;
- paired ablations over timing, strength, quality, diversity, and runtime.

Moving to images requires a real denoiser parameterization, high-dimensional PCA
or sample-space computation, layer selection, activation tensors, stronger
quality metrics, and substantially broader validation.
