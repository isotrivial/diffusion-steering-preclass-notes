# CIFAR-10 Image Steering Experiment

## Status

The bridge is implemented as
`cifar10_steering_with_unconditional_edm.ipynb`. It follows the six lightweight
notes as the image-scale test of their Gaussian/PCA steering idea. It passed the
revised manifest-locked A6000 validation on 2026-07-22. The complete
machine-readable result is preserved in `cifar10_edm_evaluation.json`.

## Locked generator

The notebook uses NVIDIA's official unconditional CIFAR-10 VP checkpoint:

- source: <https://github.com/NVlabs/edm>;
- checkpoint: `edm-cifar10-32x32-uncond-vp.pkl`;
- checkpoint SHA-256:
  `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`;
- sampler: deterministic EDM Heun, 18 steps, 35 network evaluations, no churn.

The network has `label_dim=0`. It cannot receive a CIFAR class label. NVIDIA's
EDM code and pretrained material use the CC BY-NC-SA 4.0 license, so this bridge
is intended for noncommercial teaching and research.

The motivation for this experiment is not that a Gaussian image model is a
complete generator. It is that coarse covariance information may remain useful
when the nonlinear EDM denoiser is still operating at high noise. This connects
the MNIST posterior-mean illustration to a real pretrained diffusion model and
the Gaussian-structure evidence of [Li, Dai, and Qu](https://arxiv.org/abs/2410.24060).

## Steering mechanism

The intervention is a class-minus-full Gaussian denoiser correction:

```text
D_guided(x, sigma) = D_EDM(x, sigma)
                     + strength * (D_class_PCA(x, sigma) - D_full_PCA(x, sigma))
```

The PCA statistics are fitted to CIFAR-10 training images in `[-1, 1]`. They
define full-image Wiener denoisers. The correction is applied to EDM's denoised
estimate during a fixed high-noise window. There is no target image, target
coordinate, centroid pull, or point-attraction controller.

This is noise-alignment-style, distribution-level guidance. It is not called
activation steering. An image activation experiment would need a separately
specified EDM block, held-out probe, shuffled-direction control, and hook audit.

## Paired experiment

Every method uses the same unconditional checkpoint, initial noise tensors,
18-step sampler, and frozen ResNet-56 evaluator:

1. unconditional baseline;
2. zero strength through the complete PCA correction path;
3. target-class correction (`cat`);
4. predeclared wrong-class correction (`ship`).

Strength and window length are chosen once on seeds `10000-10063` using the rule
in the manifest. The final evaluation uses new seeds `30000-30255`, which were
not used in calibration or either earlier evaluation. The wrong-class choice
and revised thresholds are fixed before this evaluation.

The final calibration rule requires at least `0.70` feature-covariance trace as
a margin above the `0.60` evaluation floor. It selected strength `3.0` through
transition step `7` (starting noise level about `5.32`). The selected candidate
raised the calibration cat rate from `0.109` to `0.531`, retained `0.712` of
baseline feature-covariance trace, and produced no near duplicates. These
calibration seeds are not used below. The complete 30-setting sweep is
preserved in `cifar10_edm_calibration.json`.

This margin was introduced after a first revised candidate retained `0.598` on
fresh evaluation data and therefore failed the locked `0.600` floor. That run
was rejected; the floor was not rounded or lowered. The margin candidate is
evaluated below on another unused seed range.

## Reported evidence

The executed notebook embeds a fixed evenly spaced paired image grid with
evaluator annotations, clean-image estimates along a deterministic trajectory,
and a metric comparison. It reports:

- frozen-evaluator test accuracy;
- target prediction rate and mean target probability;
- distance to real target-class evaluator features;
- evaluator-feature covariance trace and class-histogram entropy;
- near-duplicate rate;
- paired bootstrap uncertainty for target-rate gain;
- exact zero-strength and deterministic-repeat errors;
- runtime and peak GPU memory.

The bridge passes only if every locked check in
`cifar10_bridge_manifest.json` succeeds. A failed check remains a
`NEGATIVE RESULT`; the notebook does not search for new seeds or classes.

On the 256 untouched final evaluation seeds, the cat prediction rate increased
from `0.066` to `0.418`; the paired gain was `0.352` with 95% interval
`[0.285, 0.418]`. Target-feature distance fell from `5.229` to `2.839`, the
feature-covariance trace ratio was `0.709`, zero-strength and deterministic
repeat errors were exactly zero, runtime was `201` seconds, and peak allocated
GPU memory was `3.00` GiB. The wrong-class ship correction had cat rate `0.000`.

## Run on the UCSD host

The validation script expects the locked assets listed in the manifest and
creates clean, pinned source checkouts under `.external/`:

```bash
bash scripts/validate_cifar10_edm_bridge_ucsd.sh
```

It writes the executed notebook and logs to `.validation/cifar10-edm/` and
creates `VALIDATION_PASSED` only after the notebook has embedded its figures and
passed every locked protocol check.
