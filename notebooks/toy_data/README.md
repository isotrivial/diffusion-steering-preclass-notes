# Toy Course: Flow Matching and Post-Hoc Steering

This folder contains a six-notebook course sequence for undergraduates. It
starts with visual intuition, trains one unconditional 2D flow model, and then
compares three post-hoc steering families:

- class Gaussian/PCA denoiser guidance;
- inference-time class-energy gradients;
- learned hidden-feature directions.

The series intentionally contains **no attraction-to-a-point controller**. A
class is represented by examples and distributions, not by a chosen target
coordinate.

## Notebook order

| # | Notebook | Main question |
|---|---|---|
| 00 | `00_diffusion_and_flow_matching_foundations.ipynb` | How do noising paths, velocities, denoisers, and samplers differ? |
| 01 | `01_train_unconditional_flow_matching.ipynb` | Can an unconditional MLP learn the noise-to-data flow? |
| 02 | `02_denoisers_and_deterministic_samplers.ipynb` | How does velocity define a denoised estimate, and how do Euler, Heun, and RK4 compare? |
| 03 | `03_gaussian_denoisers_and_noise_alignment.ipynb` | Why can class/full Gaussian denoisers provide coarse high-noise control? |
| 04 | `04_training_free_gradient_guidance.ipynb` | What does post-hoc gradient guidance gain and cost? |
| 05 | `05_activation_steering_and_capstone.ipynb` | When is class information readable in hidden features, and can a fixed direction steer generation? |

The older source notebooks (`flow_matching_*.ipynb`) are retained as research
provenance. They are not part of the maintained course sequence.

## What students learn

After the series, students should be able to:

1. construct an I-CFM training minibatch and state its target velocity;
2. convert a velocity prediction into a clean-data estimate for a linear path;
3. test deterministic solver error with paired initial noise;
4. distinguish paired-noise evaluation from the NA-RFM noise-alignment method;
5. compare forward-only, gradient-based, and activation-space steering;
6. report control, fidelity, diversity, and runtime without collapsing them into
   one unsupported claim;
7. state precisely which parts are toy analogues and which parts would require
   a real diffusion architecture.

## Environment

From the repository root:

```bash
python -m venv --system-site-packages .venv-toy-course
source .venv-toy-course/bin/activate
pip install -r notebooks/toy_data/requirements-course.txt
python scripts/run_toy_course_notebooks.py
```

The runner executes notebooks in order and writes executed copies plus a JSON
report under `notebooks/toy_data/executed/`. Notebook `01` creates
`notebooks/toy_data/artifacts/toy_flow_model.pt`; later notebooks load it. If a
later notebook is run first, it trains the same checkpoint automatically.

The default training budget is 2,500 steps. Override it only for a code-path
check:

```bash
TOY_COURSE_TRAIN_STEPS=400 python scripts/run_toy_course_notebooks.py
```

Do not use reduced training to draw experimental conclusions.

## Course documents

- `COURSE_GUIDE.md`: prerequisites, schedule, assessments, and grading.
- `BACKGROUND.md`: compact derivations and terminology.
- `toy_course.py`: shared implementation used by all six notebooks.
- `scripts/build_toy_course_notebooks.py`: readable notebook cell source.

## Primary reference

Qingsong Wang, Mikhail Belkin, and Yusu Wang,
[*General and Efficient Steering of Diffusion Models*](https://arxiv.org/abs/2602.11395),
arXiv:2602.11395, 2026.
