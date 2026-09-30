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

Keep these settings fixed for every configuration and score all generated
images. Use the 100-image preview to inspect samples, then generate 10,000
images for the scorecard.

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
- sample count, seed, batch size, device, and library version;
- a fixed-seed image grid.

FID-M measures distributional distance in the frozen MNIST feature space.
Nonduplicate fraction checks for close generated neighbors.

Requested-digit agreement means the fraction the frozen classifier labels as
the digit requested from the generator (the JSON field is named `accuracy`).
Asking for a 7 and getting an image classified as 7 counts as agreement.
The evaluator's own test accuracy is measured on real MNIST images with known
labels.

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

## Student submission

- executed notebook and up to five complete scorecards;
- one fixed-seed grid for the selected configuration;
- a 200-word explanation of why the knobs changed quality and cost;
- one failure case or negative result.

## Before a graded leaderboard

For graded runs, choose the evaluation seeds and quality bands, then measure
the runtime on the student hardware.
