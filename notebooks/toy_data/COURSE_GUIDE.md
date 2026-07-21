# Flow-Matching Steering Course Notes

## Why this module exists

This module gives a compact, reproducible sequence of notebooks for introducing
flow-matching, trajectory steering, and ablation-based experimentation on simple
2D toy data. It is designed to be readable in seminar/lab settings and
translatable to higher-dimensional experiments in papers.

## Learning objectives

- Understand the conditional flow-matching ODE view and the OT-constant baseline.
- Implement and visualize temporal steering fields on top of a learned flow.
- Compare steering strategies across schedule shape, timing, and magnitude.
- Measure behavior with path-level metrics, endpoint error, and runtime checks.
- Translate toy-level findings into hypotheses for larger image/text/latent models.

## Recommended background

- Diffusion / flow generative models and continuous normalizing flows.
- Ordinary differential equations and intuition for vector fields.
- Autograd and PyTorch basics (module/loss/eval loops).
- Basic scientific plotting and interpretation of trajectory plots.

## Suggested pre-readings

- Lipman et al., *Flow Matching* (ICLR 2023), arXiv:2210.02747
- Albergo et al., *Stochastic Interpolants* (arXiv:2209.15571)
- Liu et al., *Rectified Flow* (arXiv:2209.03003)
- Wang et al., *Activation Steering* (arXiv:2602.11395)

## Module flow for classes

1. `00_tutorial_flow_matching_principles.ipynb`
   - Concepts and notation.
2. `01_toy_flow_matching_baseline.ipynb`
   - Baseline transport behavior on toy targets.
3. `02_toy_flow_matching_time_steering.ipynb`
   - Time-window steering insertion and qualitative behavior.
4. `03_toy_activation_steering_arxiv_ablations.ipynb`
   - Structured ablations, schedules, and baseline/steered comparison.

## Suggested lab questions

- How sensitive is endpoint accuracy to steering start/end time choices?
- Do stronger steering coefficients always improve alignment with the target?
- Which profile (constant/triangular/cosine) yields the best path-length vs. accuracy trade-off?
- How does the toy pattern change when target geometry is discontinuous (Swiss roll vs disk)?

## Expected deliverables

- A short results table (best five settings) with scalar metrics.
- One trajectory figure that justifies a qualitative claim.
- A short interpretation paragraph connecting profile shape to ODE path behavior.
