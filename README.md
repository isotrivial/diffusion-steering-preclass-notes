# Diffusion Steering: Pre-Class Notes

Preparatory notebooks for undergraduates studying diffusion models, flow
matching, deterministic sampling, and post-hoc steering. The notes build the
ideas first in a fully visible 2D setting and keep claims tied to paired
experiments and explicit controls.

The maintained notes do not use EPiC layers. This repository began from an
EPiC-FM research-code tree because the original toy notebooks lived there; the
teaching sequence itself uses small ordinary MLPs.

## Reading sequence

| # | Notebook | Main idea |
|---|---|---|
| 00 | `00_diffusion_and_flow_matching_foundations.ipynb` | Noising paths, vector fields, and flow-matching targets |
| 01 | `01_train_unconditional_flow_matching.ipynb` | Train an unlabeled model and inspect a learned eight-mode flow map |
| 02 | `02_denoisers_and_deterministic_samplers.ipynb` | Convert velocity to a denoised estimate; compare Euler, Heun, and RK4 |
| 03 | `03_gaussian_denoisers_and_noise_alignment.ipynb` | Use class-minus-full Gaussian/PCA denoisers for coarse steering |
| 04 | `04_training_free_gradient_guidance.ipynb` | Differentiate a class objective during sampling |
| 05 | `05_activation_steering_and_method_comparison.ipynb` | Probe hidden features and test matched activation interventions |

Start with [the reading guide](notebooks/toy_data/PRECLASS_NOTES.md). Compact
derivations and terminology are in
[the background notes](notebooks/toy_data/BACKGROUND.md).

There is no attraction-to-a-point controller in the maintained sequence. A
class is represented by examples, fitted distributions, class objectives, or
held-out activation directions, never by a chosen target point.

## Optional CIFAR-10 bridge

The optional image notebook uses NVIDIA's official **unconditional** CIFAR-10
EDM checkpoint, deterministic 18-step sampling, and a class-minus-full PCA
denoiser correction. It includes paired baseline, zero-strength, target-class,
and wrong-class controls. Its manifest-locked A6000 run passed all predeclared
checks. It is intentionally separate from the six core notes.

See [the CIFAR-10 bridge contract](notebooks/toy_data/optional/CIFAR10_BRIDGE.md).

## Run the core notes

```bash
python -m venv --system-site-packages .venv-toy-notes
source .venv-toy-notes/bin/activate
pip install -r notebooks/toy_data/requirements-notes.txt
python scripts/run_toy_notes.py
```

Executed copies and an execution report are written to
`notebooks/toy_data/executed/`. Default training budgets are evidence settings;
reduced environment-variable overrides are only smoke tests.

The optional CIFAR-10 notebook needs the external assets and GPU environment
locked by its manifest. On the UCSD host:

```bash
bash scripts/validate_cifar10_edm_bridge_ucsd.sh
```

## Repository layout

- `notebooks/toy_data/`: maintained notes, reading material, tests, and shared helpers;
- `notebooks/toy_data/optional/`: manifest-gated image bridge;
- `scripts/build_toy_notes.py`: readable source for the six generated notebooks;
- `scripts/run_toy_notes.py`: ordered notebook runner;
- `scripts/validate_toy_notes_ucsd.sh`: full core validation on one GPU;
- `src/`, `configs/`, `checkpoints/`: retained upstream EPiC-FM research code.

## References

- Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747).
- Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364).
- Song, Meng, and Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502).
- Wang, Belkin, and Wang, [*General and Efficient Steering of Diffusion Models*](https://arxiv.org/abs/2602.11395).
- Buhmann et al., [*EPiC-ly Fast Particle Cloud Generation with Flow-Matching and Diffusion*](https://arxiv.org/abs/2310.00049), retained upstream code provenance.
