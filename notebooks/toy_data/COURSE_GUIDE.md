# Course Guide: Flow Matching and Post-Hoc Steering

## Audience and scope

This module is designed for undergraduates who have used Python and PyTorch but
may have limited background in differential equations, diffusion models, or
representation learning. Every core mathematical object is first shown in 2D.

The module uses one unconditional flow-matching MLP. Class labels never enter
its training input. Labels are used later for evaluation and for constructing
post-hoc steering signals.

Recommended format: six 90-minute meetings plus a capstone week. Notebooks `00`
through `04` can also be assigned as guided labs; notebook `05` is the assessed
capstone.

## Prerequisites

Required:

- Python functions, arrays, plotting, and dictionaries;
- PyTorch tensors, modules, optimizers, and automatic differentiation;
- vectors, matrix multiplication, means, variances, and basic probability;
- the idea that an ODE specifies a local rate of change.

Helpful but introduced as needed:

- Gaussian distributions and covariance;
- eigenvectors/PCA;
- numerical integration;
- classifiers and hidden representations.

## Course map

### Notebook 00: foundations

Students visualize clean classes at several noise levels, inspect straight
source-to-data paths, and distinguish four objects: noising path, neural model,
denoiser, and sampler.

Evidence: one annotated path figure and written answers to three concept checks.

### Notebook 01: unconditional flow matching

Students train I-CFM on an unlabeled three-component mixture, integrate the
learned ODE, inspect time slices of the vector field, and verify distributional
coverage.

Evidence: checkpoint, loss curve, generated sample plot, vector-field plot, and
finite diagnostics.

### Notebook 02: denoisers and deterministic samplers

Students derive `D_theta(x_t,t) = x_t + (1-t)v_theta(x_t,t)`, inspect denoised
estimates, and compare Euler, Heun, and RK4 against a high-resolution numerical
reference using exactly the same initial noise tensor.

Evidence: solver table, convergence plot, and a clear explanation of paired
initial noise.

### Notebook 03: Gaussian/PCA noise alignment

Students fit target-class and full-data Gaussian statistics, visualize the
difference between their denoisers, and apply that correction only during a
high-noise window.

Evidence: correction-field plot, paired strength sweep, and a control-quality
trade-off analysis.

### Notebook 04: post-hoc gradient guidance

Students visualize a class-posterior energy landscape and backpropagate through
the model's clean estimate during sampling. They compare this method with the
forward-only Gaussian/PCA correction.

Evidence: energy/gradient plot, paired method table, trajectories, and runtime
ratio.

### Notebook 05: activation steering and capstone

Students measure class probe accuracy across time, visualize hidden features,
learn a covariance-aware target-vs-rest direction, and inject it into one hidden
layer. The final comparison includes baseline, Gaussian/PCA, gradient,
activation, and combined guidance.

Evidence: probe curve, feature projection, direction-transfer table, capstone
metrics, trajectories, and a control-quality plot.

## Terminology contract

Use these phrases consistently:

| Phrase | Meaning in this module |
|---|---|
| I-CFM | Independent source/data coupling with a linear conditional path |
| deterministic sampler | The trajectory is fixed once model, solver, settings, and initial noise are fixed |
| paired initial noise | Reusing the exact initial tensor to isolate method effects |
| denoised estimate | Clean-data estimate derived from the linear-path velocity parameterization |
| noise alignment | Target-class PCA/Gaussian denoiser minus full-data PCA/Gaussian denoiser at high noise |
| post-hoc gradient guidance | Frozen generator plus an inference-time objective gradient |
| activation steering | A learned direction added to an internal feature layer |
| gradient-free inference | No backward pass during sampling; offline statistics or direction fitting may still be required |

Do not call the course path OT-CFM: no optimal transport coupling is solved.
Do not describe post-hoc fitting as "no training" without explaining what was
fit. Do not equate paired initial noise with the noise-alignment method.

## Experimental contract

Every comparison must keep the following fixed unless it is the tested factor:

- trained checkpoint;
- initial noise tensor, not merely an integer seed;
- target-reference tensor;
- solver and number of steps;
- sample count;
- evaluation classifier/statistics;
- hardware and timing protocol.

Every method table must report at least:

- target-class rate;
- sliced Wasserstein distance to target examples;
- mean error relative to the target distribution;
- diversity ratio;
- wall-clock time.

Target-class rate alone is never sufficient evidence of a good steering method.

## Capstone assignment

Preregister one target class, one checkpoint, one solver, one sample count, and
the ablation grid before examining results.

Required comparisons:

1. baseline;
2. Gaussian/PCA noise alignment;
3. post-hoc gradient guidance;
4. activation steering;
5. high-noise alignment followed by activation steering.

Required ablations:

- at least three activation collection times;
- early, middle, and late intervention windows;
- at least three strengths;
- three matched initial-noise seeds for the decisive comparison;
- one zero-strength control;
- one shuffled-label activation-direction control.

Required report structure:

1. question and hypothesis;
2. model and fixed evaluation protocol;
3. main control-quality result;
4. runtime comparison;
5. one failure mode;
6. toy-to-image limitation;
7. next experiment on a real denoiser architecture.

## Grading

Core notebooks `00`-`04`: 10 points each.

| Category | Points | Standard |
|---|---:|---|
| Correct execution | 4 | Required cells run, shapes are valid, and checks are finite |
| Evidence | 3 | Required plots and tables directly support the answers |
| Interpretation | 2 | Claims distinguish control, fidelity, diversity, and cost |
| Reproducibility | 1 | Seed, device, checkpoint, solver, and sample count are recorded |

Capstone notebook `05`: 20 points.

| Category | Points | Standard |
|---|---:|---|
| Correct experiment | 6 | All required methods and controls run |
| Fair comparison | 4 | Initial tensors, references, solver, and seeds are matched |
| Evidence | 4 | Metrics and trajectory plots show both result and failure case |
| Interpretation | 4 | Conclusion is supported and scope is stated accurately |
| Reproducibility package | 2 | Executed notebook, table, and configuration record are present |

## Readings

Core:

- Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747).
- Song, Meng, and Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502).
- Wang, Belkin, and Wang, [*General and Efficient Steering of Diffusion Models*](https://arxiv.org/abs/2602.11395).

Optional background:

- Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239).
- Dhariwal and Nichol, [*Diffusion Models Beat GANs on Image Synthesis*](https://arxiv.org/abs/2105.05233).
- Radhakrishnan et al., [*Mechanism for Feature Learning in Neural Networks and Backpropagation-Free Machine Learning Models*](https://arxiv.org/abs/2211.05109).
