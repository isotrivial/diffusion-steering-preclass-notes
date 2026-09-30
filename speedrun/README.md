# MNIST diffusion speedrun

Generate recognizable, varied digits with fewer denoiser evaluations. Change
the sampler, step count, and guidance strength while using the same trained
model and frozen evaluator.

Start with the [student notebook](MNIST_SPEEDRUN.ipynb), or read the
[worked example](MNIST_SPEEDRUN.executed.ipynb). The model and evaluator are
included, so you can begin sampling immediately. Notes 00–02 provide the
background for this exercise.

## Run it

From the repository root:

```sh
python -m pip install -r speedrun/requirements.txt
```

Open the notebook and run its cells. It starts with a 100-image preview, then
scores 10,000 generated images.

You can also run a configuration from the command line:

```sh
python speedrun/benchmark.py --order heun --steps 4 --repeats 1 \
  --out speedrun/results/local/heun4.json
```

This saves the scorecard and a digit grid. Use `--repeats 3` to report median
sampling time across three runs.

## Turn the knobs

| Setting | What it changes |
|---|---|
| `steps` | Number of sampling intervals |
| `w` | Guidance strength: 0 is unconditional, 1 is conditional, and values above 1 strengthen conditioning |
| `order` | Sampler: `ddim`, `euler`, or `heun` |
| `eta` | Noise added during DDIM sampling, from 0 to 1; use 0 with Euler or Heun |

Time runs from noise at `t=0` to clean data at `t=1`. The model predicts a clean
image from the current noisy image, time, and requested digit. DDIM updates the
clean-image and noise estimates along the cosine schedule; Euler and Heun
integrate the corresponding velocity.

With guidance `w=2`, DDIM and Euler cost `2*steps` denoiser evaluations per image;
Heun costs `2*(2*steps-1)`. Compare 7-step DDIM with 4-step Heun for an equal-cost
experiment: both use 14 evaluations. At `w=0` or `w=1`, only one prediction
branch is needed, halving these counts.

## Read the scores

- **FID-M:** distance between generated and real MNIST feature distributions,
  measured by the frozen classifier. Lower is better. The reference uses all
  60,000 training images and the classifier's 128-dimensional representation.
- **Requested-digit agreement:** fraction the classifier labels as the requested
  digit. Requesting a 7 and generating an image classified as 7 counts as
  agreement. **Target confidence** is the mean probability assigned to the
  requested digit.
- **Nonduplicate fraction:** fraction of generated images whose nearest neighbor
  is farther than the fixed threshold in classifier feature space.
- **NFE:** number of denoiser evaluations per generated image. Conditional and
  unconditional predictions each count as one evaluation.

## Example comparison

Both configurations below use guidance `w=2`, seed 0, and 10,000 images:

| Sampler | Steps | NFE | FID-M | Digit agreement | Nonduplicate fraction |
|---|---:|---:|---:|---:|---:|
| DDIM | 16 | 32 | 91.13 | 99.81% | 96.89% |
| Heun | 4 | 14 | 79.29 | 99.78% | 97.36% |

![Generated digits, requested rows 0 through 9](results/heun4-seed0.png)

## Your challenge

Try up to five configurations. Keep the model, evaluator, initial noise, batch
size, and sample count fixed. Aim for FID-M ≤120, requested-digit agreement ≥98%,
and nonduplicate fraction ≥95% with the fewest NFE.

Submit your scorecards, a fixed-seed image grid, and a short explanation of
which choices improved quality and cost. Include one configuration that failed
to meet the quality band. See the [exercise instructions](SPEC.md) for details.
