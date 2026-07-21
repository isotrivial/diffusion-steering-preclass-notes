# Background Notes

This note supplies the minimum derivations needed by the notebooks. The main
course path stays visual; these details are available when students ask why a
formula is valid.

## 1. Linear I-CFM path

Sample independent endpoints, using base standard deviation `s = 1.8` in this
course:

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

## 2. Velocity-to-denoiser conversion

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

## 3. Deterministic ODE sampling

Generation solves

```text
dx/dt = v_theta(t,x),   x(0) = x_noise.
```

Euler, Heun, and RK4 are numerical approximations to the same ODE. Once the
initial tensor, model, solver, step count, and floating-point environment are
fixed, the rollout contains no further random draw.

This is why paired initial noise is a strong experimental control: two methods
start from the same individual samples, not merely from the same distribution.

## 4. Additive-noise coordinates

For `t > 0`, divide the linear path by `t` and substitute
`x_noise = s * epsilon`:

```text
y = x_t / t = x_data + sigma_eff * epsilon
sigma_eff = s (1-t) / t.
```

The flow state can therefore be viewed in additive standard-Gaussian-noise
coordinates. Small `t` corresponds to large `sigma_eff` and high noise. Keeping
the factor `s` is necessary because the course base distribution is not unit
variance.

## 5. Gaussian/PCA denoiser

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

The course noise-alignment signal is

```text
Delta D = D_target_class - D_full_data.
```

It is computed from labeled examples offline and applied only at high noise.
This is distinct from reusing the same initial noise tensor for evaluation.

## 6. Gradient guidance

Suppose `s_c(x)` is a differentiable score for target class `c`. The gradient

```text
grad_x log s_c(x)
```

is the local direction that most rapidly increases the target log score. In the
course, the score is evaluated on `D_theta(x_t,t)`, so automatic differentiation
passes through both the clean estimate and velocity model at every active solver
evaluation.

This is post-hoc because the generator is frozen, but it is not cheap or
gradient-free at inference.

## 7. Why hidden-feature steering can work

The MLP computes

```text
(x,t) -> hidden feature h -> velocity v.
```

If target and background examples separate in `h`, a target-vs-rest direction
can change the downstream velocity in a class-relevant way. Repeated local
velocity changes accumulate over an ODE trajectory.

The notebooks check three empirical requirements instead of assuming them:

1. a linear probe can decode class from the feature;
2. learned directions remain aligned across nearby times;
3. injection improves control without unacceptable fidelity or diversity loss.

## 8. Scope boundary

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
