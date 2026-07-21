# Course Guide: Flow Matching and Paper-Inspired Steering on Toy Data

## Module summary

This module uses 2D generative flows to teach three connected skills:

1. reasoning about flow-matching objectives and ODE trajectories;
2. designing and evaluating inference-time interventions;
3. translating a toy experiment into a defensible higher-dimensional research
   plan.

The four notebooks form one sequence. Notebook `03` is the assessed lab.

## Scope and paper alignment

The module is inspired by arXiv:2602.11395, but it is not a reproduction of
NA-RFM.

| Component | arXiv:2602.11395 | Notebook `03` |
|---|---|---|
| Generative model | Diffusion U-Net or related denoiser | 2D flow-matching velocity MLP |
| High-noise guidance | PCA-based noise alignment | Omitted |
| Steering signal | Target-vs-rest RFM direction learned offline | Hand-designed state-space attraction field |
| Intervention site | Internal activation block | Output velocity field |
| Timing | Intermediate/late noise window | User-defined ODE-time window |
| Main ablations | Collection noise, block, window, direction learning | Window, strength, and schedule profile |
| Quality evidence | Target accuracy, FID/KID, diversity, cost | Anchor control, SWD/Chamfer, diversity, path length, cost |

The notebook therefore teaches the **experimental logic** of steering:
paired controls, timing, intervention strength, quality trade-offs, and careful
claims. It must not describe its state-space field as a learned RFM activation
direction.

> **Terminology:** In the steering paper, RFM means *Recursive Feature Machine*.
> It is unrelated to the acronym FM used for Flow Matching.

## Learning outcomes

By the end of the module, students should be able to:

1. Write the OT-constant interpolation  
   `x_t = (1 - t) x_0 + t x_1` and its conditional target velocity
   `u_t = x_1 - x_0`.
2. Explain how a learned velocity field defines an ODE that transports a base
   distribution to a target distribution.
3. Diagnose baseline quality using endpoint geometry as well as training loss.
4. Implement and visualize a time-gated steering field.
5. Construct a paired ablation in which all methods use the same initial samples
   and target-reference samples.
6. Separate a control metric from a target-fidelity or diversity metric.
7. Explain which parts of NA-RFM are represented by analogy and which are
   absent.
8. Propose a higher-dimensional activation-steering experiment with suitable
   blocks, noise levels, controls, and metrics.

## Prerequisites

Required:

- Python functions, loops, dictionaries, and plotting;
- PyTorch tensors, modules, optimizers, and `torch.no_grad`;
- basic vector and matrix operations;
- introductory probability and ordinary differential equations.

Helpful but not required:

- diffusion-model terminology;
- continuous normalizing flows;
- optimal transport;
- PCA, classifiers, and representation learning.

### Readiness check

Students are ready if they can answer all three questions:

1. Why must the scalar ODE time be broadcast across a batch?
2. Why can a low training loss coexist with a poor generated endpoint
   distribution?
3. Why does comparing two methods with different initial noise weaken a causal
   claim about steering?

## Readings

### Flow-matching background

- Lipman et al., *Flow Matching for Generative Modeling*,
  arXiv:2210.02747.
- Albergo et al., *Stochastic Interpolants*,
  arXiv:2209.15571.
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data
  with Rectified Flow*, arXiv:2209.03003.

### Primary steering reading

- Wang, Belkin, and Wang,
  *General and Efficient Steering of Diffusion Models*,
  arXiv:2602.11395.

For the lab, focus on:

- the method overview and empirical observations;
- noise alignment versus activation steering;
- activation collection and online intervention;
- guidance-window ablations;
- block-selection ablations;
- RFM versus difference-of-means direction learning;
- sample-space computation for high-dimensional activations.

## Four-notebook flow

| Notebook | Main question | Student checkpoint |
|---|---|---|
| `00` | What vector field would transport one distribution to another? | Derive `x_t` and `u_t`; identify the learned quantity. |
| `01` | Can the model recover different support geometries? | Compare loss curves with endpoint plots for circle, disk, and Swiss roll. |
| `02` | How does a time-localized control field change trajectories? | Predict and test one early and one late intervention. |
| `03` | Which steering settings improve control without destroying fidelity? | Produce a paired ablation, figure, and paper-alignment analysis. |

## Lab protocol

For each notebook, students should run and document:

- explicit goals before coding;
- reproducibility metadata;
- one sanity-check plot that verifies the notebook contract.

## Notebook `03` experimental protocol

1. **State a hypothesis before running the grid.**
   Example: “Late steering will preserve target geometry better than early
   steering at equal steering dose.”

2. **Pass a baseline quality gate.**
   Do not interpret steering results until the endpoint cloud visibly covers
   the intended target support and all metrics are finite.

3. **Use common random numbers.**
   For each target, generate one `x0_eval` tensor and reuse it for the baseline
   and every steering configuration.

4. **Use one fixed target-reference bank.**
   Measure generated-to-target and target-to-generated distances against sampled
   target points. Do not substitute the control anchor for the target
   distribution.

5. **Separate ablation questions.**
   - Timing: early versus middle versus late, with strength and profile fixed.
   - Strength: low versus medium versus high, with timing and profile fixed.
   - Profile: constant versus triangular versus cosine at matched integrated
     steering dose.

6. **Repeat only the decisive comparison.**
   Rerun the baseline and the selected configuration with at least three seeds.
   Report mean and standard deviation. The entire grid need not be repeated.

7. **Treat runtime cautiously.**
   Synchronize CUDA before timing and label single-run timings as descriptive,
   not as a benchmark.

## Lab questions

1. The center anchor is off the support of the discrete circle and ring. Can an
   intervention improve anchor control while making the generated distribution
   worse?

2. In this notebook, `t = 0` is noise and `t = 1` is data. Which time window
   is the closest toy analogue of the paper's intermediate/late low-noise
   activation-steering window?

3. Do early, middle, and late windows differ when duration and steering dose are
   held constant?

4. Constant, triangular, and cosine gates have different areas when they share the
   same peak. Does an apparent profile advantage remain after matching integrated
   dose?

5. How do intervention failures differ across:
   - discrete support: 12 circle points;
   - filled support: uniform disk;
   - thin curved support: Swiss roll;
   - annular support: ring?

6. Find one configuration that improves anchor control but harms diversity or
   target SWD. Why would reporting only the control metric be misleading?

7. Which result is consistent with the paper's timing observations, and why is
   it still not a replication of the paper?

8. What additional data and model access would be required to replace the toy
   attraction field with a learned activation direction?

## Required deliverables

### A. Reproducibility record

Report:

- seed or seeds;
- CPU/GPU device;
- training steps and batch size;
- evaluation sample count;
- ODE solver and number of time points;
- exact configuration labels.

### B. Ablation table

Include the baseline and at least:

- three timing conditions;
- three steering strengths;
- three dose-matched profiles.

Report:

- mean and 90th-percentile distance to the control anchor;
- sliced Wasserstein distance or symmetric Chamfer distance to target samples;
- diversity ratio;
- mean trajectory length;
- approximate runtime.

Do not name a single “best” configuration without stating the quality constraint
used to select it.

### C. Figure

Submit one fixed-seed comparison containing:

- target-reference points;
- baseline endpoints or trajectories;
- selected steered endpoints or trajectories;
- a caption that states the configuration and the claim supported by the
  figure.

### D. Analysis

Write 400–600 words that:

1. state the original hypothesis;
2. cite at least two numerical results;
3. explain one control-versus-fidelity trade-off;
4. compare two support geometries;
5. identify one limitation of the experiment;
6. distinguish the toy intervention from NA-RFM.

## Grader rubric — 10 points

- **2 points — Reproducibility:** complete configuration record and successful,
  finite runs.
- **2 points — Experimental design:** shared initial samples, valid baseline,
  and controlled ablations.
- **2 points — Metrics and figure:** control and fidelity are both measured and
  visualized correctly.
- **3 points — Interpretation:** numerical evidence supports the claims; paper
  alignment and limitations are accurate.
- **1 point — Communication:** table, caption, and prose are clear and
  self-contained.

A submission that ranks configurations only by distance to the center anchor
cannot receive full metrics or interpretation credit.

## Optional advanced extension assignment

### From 2D state steering to high-dimensional activation steering

**Goal:** replace the toy output-space control field with a target-specific
direction applied to an internal activation of a higher-dimensional diffusion
model.

A larger unconditional denoiser on Fashion-MNIST or CIFAR-10 can be used.

### Required work

1. Define one target concept using positive and background examples.
2. Select one internal block and collect forward-process activations at three
   noise levels: high, intermediate, and low.
3. Train a simple linear probe and report whether the concept is decodable at each
   noise level.
4. Construct:
   - a difference-of-means direction; and
   - an RFM or other covariance-aware discriminative direction when feasible.
5. Inject each fixed direction into the selected block over early, middle, and
   late sampling windows using identical initial noise seeds.
6. Evaluate:
   - target success or classifier accuracy;
   - FID, KID, SWD, or another domain-appropriate quality metric;
   - diversity or recall;
   - inference cost relative to unconditional sampling.
7. Test at least one second block or one second activation-collection noise
   level.
8. Explain whether the learned direction transfers across timesteps.

### Extension deliverables

- one diagram identifying the edited model block and activation tensor;
- one table covering direction type, collection noise, and intervention window;
- one fixed-seed baseline-versus-steered sample grid;
- a 500-word discussion of direction transfer, quality trade-offs, and
  computational cost.

**Stretch:** add the paper's high-noise PCA noise-alignment stage and compare
activation-only steering against the combined two-stage method.

## References

- Lipman et al., *Flow Matching for Generative Modeling*,
  arXiv:2210.02747.
- Albergo et al., *Stochastic Interpolants*,
  arXiv:2209.15571.
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data
  with Rectified Flow*, arXiv:2209.03003.
- Wang, Belkin, and Wang, *General and Efficient Steering of Diffusion Models*,
  arXiv:2602.11395.
