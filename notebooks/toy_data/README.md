# Pre-Class Notes: Flow Matching and Post-Hoc Steering

This folder contains six preparatory notebooks to read before class. They start
with a visible noise-to-data process, train unconditional flows on several 2D
geometries, connect velocity to denoising, and only then introduce three places
to steer a frozen generator. A seventh notebook carries the first steering
mechanism to CIFAR-10. The progression matters: Gaussian/PCA,
objective-gradient, and hidden-feature controls make more sense after the
model, denoiser, and sampler are separated.

The series intentionally contains **no attraction-to-a-point controller**. A
class is represented by examples and distributions, not by a chosen target
coordinate.

## Notebook order

| # | Read with outputs | Main question |
|---|---|---|
| 00 | [Generative-model foundations](executed/00_diffusion_and_flow_matching_foundations.ipynb) | How do noising paths, velocities, denoisers, and samplers differ? |
| 01 | [Unconditional flow matching](executed/01_train_unconditional_flow_matching.ipynb) | Can an unconditional MLP learn clusters, a continuous circle, and eight separated modes? |
| 02 | [Denoisers and deterministic samplers](executed/02_denoisers_and_deterministic_samplers.ipynb) | Why is the velocity conversion a conditional clean estimate, and how do Euler, Heun, and RK4 compare? |
| 03 | [Gaussian/PCA denoisers and noise alignment](executed/03_gaussian_denoisers_and_noise_alignment.ipynb) | Can Gaussian/PCA structure denoise MNIST pixels and provide coarse high-noise control? |
| 04 | [Post-hoc objective-gradient guidance](executed/04_training_free_gradient_guidance.ipynb) | What does a state-dependent objective gradient gain, and what does online backpropagation cost? |
| 05 | [Activation steering and method comparison](executed/05_activation_steering_and_method_comparison.ipynb) | When is class information readable in hidden features, and how do the steering methods compare? |
| 06 | [Noise alignment in an unconditional CIFAR-10 EDM](executed/cifar10_steering_with_unconditional_edm.ipynb) | Does the high-noise distributional correction produce a measurable, class-specific image shift? |

The links above always open the published notebooks with inline outputs. The
same filenames one directory above are output-free implementation sources.

The older source notebooks (`flow_matching_*.ipynb`) are retained as research
provenance. They are not part of the pre-class reading sequence.

Notebook `06` uses NVIDIA's unconditional CIFAR-10 EDM checkpoint and a
class-minus-full PCA denoiser correction. The learner-facing notebook focuses
on the images, trajectory, metrics, and limits. Reproduction details, pinned
assets, and the calibration record remain in `optional/CIFAR10_BRIDGE.md`.

## Why the sequence is ordered this way

Notebook `00` supplies the vocabulary needed to read the training code.
Notebook `01` makes geometric successes and failures visible. Notebook `02`
then translates the velocity model into the denoiser language used by the
steering methods. Notebook `03` tests a simple covariance prior on pixels before
using it for guidance. Notebooks `04` and `05` relax that prior through an
objective gradient and an internal feature edit. Notebook `06` is last because
it asks which parts of the visible toy argument survive in a pretrained image
generator.

## Environment

From the repository root:

```bash
python -m venv --system-site-packages .venv-toy-notes
source .venv-toy-notes/bin/activate
pip install -r notebooks/toy_data/requirements-notes.txt
python scripts/run_toy_notes.py
```

The runner executes notebooks in order and writes executed copies plus a JSON
report under `notebooks/toy_data/executed/`. Notebook `01` creates the main,
circle, and eight-Gaussian checkpoints; later steering notebooks load only the
main three-cluster checkpoint. If a later notebook is run first, it trains that
checkpoint automatically.

The three-mode, circle, and eight-mode default training budgets are 2,500,
3,000, and 3,000 steps. Override them only for a code-path check:

```bash
TOY_NOTES_TRAIN_STEPS=400 \
TOY_NOTES_CIRCLE_STEPS=400 \
TOY_NOTES_EIGHT_STEPS=400 \
python scripts/run_toy_notes.py
```

Do not use reduced training to draw experimental conclusions.

## Supporting files

- `PRECLASS_NOTES.md`: scope, prerequisites, reading order, and discussion questions.
- `BACKGROUND.md`: compact derivations and terminology.
- `toy_notes.py`: shared implementation used by all six notebooks.
- `scripts/build_toy_notes.py`: readable notebook cell source.

## Primary reference

Qingsong Wang, Mikhail Belkin, and Yusu Wang,
[*General and Efficient Steering of Diffusion Models*](https://arxiv.org/abs/2602.11395),
arXiv:2602.11395, 2026.
