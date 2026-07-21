# Toy-Data Notebook Series

## Course notes and materials

These notebooks are organized as a compact, instructor-friendly course path on
flow-matching and trajectory steering.

The companion guide is in [`COURSE_GUIDE.md`](COURSE_GUIDE.md), with learning
outcomes, prerequisites, references, and suggested exercises.

## Sequence overview

This folder contains lightweight toy-system notebooks for flow-matching experiments on
2D geometry data. The series is organized as follows:

- `00_tutorial_flow_matching_principles.ipynb`
  - Theoretical foundation for flow matching and conditional flow matching objectives.
- `01_toy_flow_matching_baseline.ipynb`
  - Baseline OT-constant flow matching on three targets (circle, disk, Swiss roll).
- `02_toy_flow_matching_time_steering.ipynb`
  - Time-window steering extension to the same baseline, with a minimal steering term
    and side-by-side trajectories.
- `03_toy_activation_steering_arxiv_ablations.ipynb`
  - Activation-steering A/B studies inspired by arXiv:2602.11395.
  - Ablations over strength, time-window bounds, and schedule profile.
  - Includes path-length, endpoint-distance, and runtime metrics.

Run from a stable environment with dependencies from `requirements.txt` (`torchdyn`,
`torch`, `matplotlib`, `numpy`), preferably with a GPU for faster training.
For the new arXiv-steering notebook, these dependencies are enough for CPU defaults.

Suggested workflow:

1. Start with notebook `00...` to align notation and training objective.
2. Reproduce baseline behavior with notebook `01...`.
3. Test temporal steering behavior with notebook `02...` and keep notes on
   hyperparameters and loss behavior.
4. Expand to activation ablations in notebook `03...` and compare steer schedules
   before scaling to higher-dimensional latent experiments.

Use this as the foundation for a future extension to the `arXiv:2602.11395` direction
in paper-level experiments.
