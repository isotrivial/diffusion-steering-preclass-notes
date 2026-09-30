# Optional MNIST diffusion speedrun

## Start here

Open the [student notebook](MNIST_SPEEDRUN.ipynb), or inspect the
[executed example](MNIST_SPEEDRUN.executed.ipynb). The
[setup instructions](README.md) include a command-line alternative.
The conditional model, evaluator, and real-data reference are included; students
only run inference. Notes 00–02 provide the preparation for this optional exercise.

## Learning goal

Understand the quality–cost tradeoff created by diffusion sampling choices.
Find a configuration that meets a fixed quality band with fewer denoiser
evaluations (NFE), then explain the tradeoff using metrics and images.

## What is fixed

- conditional MNIST model architecture and checkpoint;
- cosine noise schedule, clean-image prediction, and uniform time grid;
- frozen evaluator and statistics from all 60,000 training images;
- 10,000 generated samples, with 1,000 requests for each digit;
- initial-noise seed and sampling batch size (128).

The notebook verifies the three artifact hashes in
[release.json](release.json). Retraining, fine-tuning, changing
weights or the evaluator, choosing favorable seeds, and discarding generated
images are outside the challenge. The 100-image preview is for inspecting
images, not for leaderboard FID-M.

## What students may change

1. guidance scale `w`, using `u + w*(c-u)`: 0 is unconditional, 1 conditional;
2. sampling-step count;
3. sampler: DDIM, Euler, or Heun;
4. stochastic DDIM `eta` in [0,1]; Euler and Heun require `eta=0`.

Students submit at most five configurations. Record every value. Euler and
Heun integrate velocity; DDIM uses the clean/noise transport. All finish with
a denoising step instead of evaluating a singular velocity at the data endpoint.

## Required scorecard

Each configuration reports:

- FID-M in the frozen MNIST feature space (lower is better);
- requested-digit agreement and mean requested-digit confidence;
- nonduplicate fraction in that feature space;
- NFE per generated image;
- median sampling time over three runs after warm-up;
- sample count, seed, batch size, device, library version, artifact hashes;
- a fixed-seed image grid.

FID-M is not comparable to published Inception FID. Nonduplicate fraction
checks for close generated neighbors; it is not a measurement of real-mode
coverage. Both quality and diversity diagnostics matter.

Requested-digit agreement means the fraction the frozen classifier labels as
the digit requested from the generator (the JSON field is named `accuracy`).
Asking for a 7 and getting an image classified as 7 counts as agreement. This
checks conditioning, not overall image quality: one repeated prototype per
digit could score 100%. The evaluator's own test accuracy is a separate
measurement on real MNIST images with known labels.

NFE is the primary cost measure. With `w=2`, DDIM/Euler cost `2*steps` and
Heun costs `2*(2*steps-1)`. At `w=0` or `w=1`, these counts are halved.
For example, 7-step DDIM and 4-step Heun both use 14 NFE at `w=2`.
Wall time is comparable only on the same device class and batch size. The
notebook defaults to one measured run for practice; set `REPEATS=3` for timing.

## Practice rule

The reference is 16-step DDIM with `w=2`, `eta=0`, seed 0. Its measured
[scorecard](results/reference-seed0.json) uses 32 NFE and scores
FID-M 91.13, digit agreement 99.81%, and nonduplicate fraction 96.89%.

A practice configuration qualifies if all three conditions hold at the fixed
10,000-sample protocol:

- FID-M ≤120;
- requested-digit agreement ≥98%;
- nonduplicate fraction ≥95%.

Among qualifying configurations, lower NFE ranks first, followed by lower median
wall time on the designated hardware, then higher nonduplicate fraction.
These bands support the released practice example. They are not a validated
hidden-seed grading rule.

## Student submission

- executed notebook and up to five complete scorecards;
- one fixed-seed grid for the selected configuration;
- a 200-word explanation of why the knobs changed quality and cost;
- one failure case or negative result.

## Before a graded leaderboard

The runnable practice package includes frozen artifacts, an executed example,
metric sanity checks, NFE regression checks, and deterministic replay checks.
Before assigning grades, instructors must additionally fix hidden evaluation
seeds, calibrate their quality bands, and measure a complete run on the target
student hardware. The full 10,000-image notebook has been exercised on Apple
MPS; a CPU or Colab class-time budget is not certified.
