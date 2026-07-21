# Toy-Data Course Module: Flow Matching and Steering

This folder contains a four-notebook teaching sequence on flow matching,
continuous-time generation, and inference-time steering. Two-dimensional
targets make the learned vector field and individual trajectories directly
visible.

> **Scope and terminology**
>
> - These notebooks use a small 2D velocity MLP. They do not implement the
>   EPiC point-cloud architecture used elsewhere in this repository.
> - Notebook `03` is a paper-inspired **state-space steering surrogate**. It
>   studies timing, strength, evaluation, and control-versus-fidelity trade-offs.
>   It does not reproduce the internal activation intervention or high-noise
>   noise-alignment stage of arXiv:2602.11395.
> - In arXiv:2602.11395, **RFM** means *Recursive Feature Machine*. It should
>   not be confused with Flow Matching or Rectified Flow.

See [`COURSE_GUIDE.md`](COURSE_GUIDE.md) for prerequisites, readings, lab
questions, deliverables, grading criteria, and the advanced extension.

## Learning outcomes

After completing the sequence, students should be able to:

1. Explain the conditional flow-matching training target and its ODE sampler.
2. Train and diagnose a toy flow-matching model on several support geometries.
3. Implement a time-gated perturbation to a learned vector field.
4. Design a paired ablation using fixed initial samples and valid distributional
   metrics.
5. Distinguish toy state-space steering from block-level activation steering in
   diffusion models.

## Four-notebook path

1. `00_tutorial_flow_matching_principles.ipynb`
   - Introduces probability paths, conditional flow matching, and ODE sampling.
   - Checkpoint: derive the OT-constant path and target velocity.

2. `01_toy_flow_matching_baseline.ipynb`
   - Trains baseline models on a 12-point circle, a uniform disk, and a Swiss
     roll.
   - Checkpoint: determine whether the endpoint cloud matches the target support,
     rather than relying only on training loss.

3. `02_toy_flow_matching_time_steering.ipynb`
   - Adds a hand-designed, time-windowed control field.
   - Checkpoint: separate intervention success from preservation of the learned
     target distribution.

4. `03_toy_activation_steering_arxiv_ablations.ipynb`
   - Runs paired, paper-inspired ablations over intervention time, strength, and
     schedule.
   - Reports control, target-fidelity, diversity, trajectory, and runtime
     measurements.
   - Connects the toy experiment to the components that would be required for
     genuine activation steering.

## Environment

Install the repository dependencies:

```bash
pip install -r requirements.txt
```

Verify the notebook dependencies before class:

```bash
python -c "import torch, torchdyn, matplotlib, numpy; print(torch.__version__)"
```

Full baseline training benefits from a GPU. Reduced-step settings are useful for
checking code paths, but conclusions should be based on a baseline that visibly
covers the target support.

## Recommended workflow

1. Read and run notebook `00`; write down definitions of `x_t`, `u_t`, and
   `v_theta`.
2. In notebook `01`, record one successful baseline figure and its training
   configuration.
3. Before notebook `02`, predict how early and late steering will affect the
   trajectory.
4. In notebook `03`, use identical initial noise for every configuration and
   compare intervention control against target-distribution fidelity.
5. Report negative or unstable results rather than selecting only favorable
   trajectories.

## Completion evidence

A complete submission contains:

* a reproducibility record with seed, device, training steps, sample count, and
  ODE resolution;
* a paired ablation table including the baseline;
* one figure supporting a qualitative claim;
* one evidence-based conclusion and one limitation;
* a short explanation of what the toy notebook omits from NA-RFM.

## Primary steering reference

Qingsong Wang, Mikhail Belkin, and Yusu Wang,  
*General and Efficient Steering of Diffusion Models*,
arXiv:2602.11395.
