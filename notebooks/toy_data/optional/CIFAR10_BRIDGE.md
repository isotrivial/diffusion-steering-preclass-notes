# CIFAR-10 Image Steering Experiment

## Status

The bridge is implemented as
`cifar10_steering_with_unconditional_edm.ipynb`. It follows the six lightweight
notes as the image-scale test of their Gaussian/PCA steering idea. It passed the
manifest-locked A6000 validation on 2026-07-21. The complete machine-readable
result is preserved in `cifar10_edm_evaluation.json`.

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

Strength is chosen once on seeds `10000-10063` using the rule in the manifest.
Evaluation then uses untouched seeds `0-255`. The wrong-class choice and all
thresholds are fixed before evaluation.

The predeclared calibration selected strength `4.0`; it raised the calibration
cat rate from `0.109` to `0.375`, moved the mean evaluator features toward real
cat features, retained `0.876` of the baseline feature-covariance trace, and
produced no near duplicates. These calibration seeds are not used below. The
complete result is preserved in `cifar10_edm_calibration.json`.

## Reported evidence

The executed notebook embeds paired image grids, clean-image estimates along a
deterministic trajectory, and a metric comparison. It reports:

- frozen-evaluator test accuracy;
- target prediction rate and mean target probability;
- distance to real target-class evaluator features;
- evaluator-feature covariance trace and class-histogram entropy;
- near-duplicate rate;
- paired bootstrap uncertainty for target-rate gain;
- exact zero-strength and deterministic-repeat errors;
- runtime and peak GPU memory.

The bridge passes only if every predeclared check in
`cifar10_bridge_manifest.json` succeeds. A failed check remains a
`NEGATIVE RESULT`; the notebook does not search for new seeds or classes.

On the 256 untouched evaluation seeds, the cat prediction rate increased from
`0.086` to `0.324`; the paired gain was `0.238` with 95% interval
`[0.176, 0.301]`. Target-feature distance fell from `5.022` to `3.444`, the
feature-covariance trace ratio was `0.766`, zero-strength and deterministic
repeat errors were exactly zero, runtime was `140` seconds, and peak allocated
GPU memory was `3.00` GiB. The wrong-class ship correction reduced the cat rate
to `0.004`.

## Run on the UCSD host

The validation script expects the locked assets listed in the manifest and
creates clean, pinned source checkouts under `.external/`:

```bash
bash scripts/validate_cifar10_edm_bridge_ucsd.sh
```

It writes the executed notebook and logs to `.validation/cifar10-edm/` and
creates `VALIDATION_PASSED` only after the notebook has embedded its figures and
passed every scientific control.
