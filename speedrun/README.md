# MNIST diffusion speedrun

A runnable inference exercise with a trained conditional generator, a frozen
MNIST evaluator, fixed real-data statistics, and a student notebook. No training
or dataset download is needed for sampling and scoring.

Open [MNIST_SPEEDRUN.ipynb](MNIST_SPEEDRUN.ipynb). The
[executed example](MNIST_SPEEDRUN.executed.ipynb) includes the image grids and a
10,000-image scorecard. The [course specification](SPEC.md)
explains the challenge and submission.

## Run it

From the repository root, in a Python environment with PyTorch:

```sh
python -m pip install -r speedrun/requirements.txt
python speedrun/benchmark.py --order heun --steps 4 --repeats 1 \
  --out speedrun/results/local/heun4.json
```

The command generates 10,000 balanced digit requests, checks the release hashes,
warms up the sampler, and saves a JSON scorecard and a digit grid. It selects
CUDA, Apple MPS, or CPU automatically; use `--device cpu` to choose explicitly.
For median timing, use `--repeats 3` (the CLI default). Sampling time excludes
model loading and evaluator scoring. A single run is sufficient to try the
exercise; compare timing only on the same hardware.

For a quick installation check, add `--n 100 --repeats 1`. That score is marked
as a smoke test and cannot pass the practice band. The notebook starts with a
100-image preview and then scores 10,000 images. Full CPU sampling takes longer
than accelerator sampling; no CPU class-time budget has been certified.

## Frozen scoring

[release.json](release.json) records SHA-256 hashes for the evaluator, generator,
and reference statistics. All three small artifacts are included under
`checkpoints/` (about 9 MB total). `benchmark.Speedrun` verifies them before
loading and before each run, puts both networks in evaluation mode, and disables
gradients. Notebook comparisons use the same CPU-generated initial noise bank.

- **FID-M:** Gaussian Fréchet distance in the evaluator's 128-dimensional
  post-ReLU representation. Preprocessing is grayscale 28×28, scaled to [-1,1].
  The reference uses all 60,000 MNIST training images, with float64 covariance
  and a fixed 1e-6 ridge. This is a course metric, not published Inception FID.
- **Requested-digit agreement** (the scorecard's `accuracy` field): fraction
  the frozen classifier labels as the requested digit. For example, request a
  7 and count success if the classifier predicts 7. **Target confidence** is
  its mean probability for the requested digit. Repeating one recognizable
  image per digit could achieve 100% agreement; this is not image quality.
- **Nonduplicate fraction:** fraction whose closest other generated sample is
  farther than the fixed real-data threshold. This detects near-duplicates;
  it does not establish coverage of the real distribution.
- **NFE:** conditional and unconditional predictions per generated image.
  A combined two-branch batch still costs two evaluations.

Separately, the evaluator has 99.32% accuracy on labeled real test images.
Against the frozen training reference,
the full 10,000-image test set has FID-M 5.5844. The
[evaluator checks](results/evaluator-validation.json) also verify increasing
scores under progressive blur and zero nonduplicate fraction for a repeated
image. This real-data score is a reference comparison, not a universal FID floor.

The generator is the existing 15-minute, 2,942-step EMA checkpoint. The
[reference scorecard](results/reference-seed0.json) records 16-step DDIM with
`w=2`, seed 0, and 10,000 samples: FID-M **91.13**, digit agreement **99.81%**,
nonduplicate fraction **96.89%**, **32 NFE**. One measured sampling run took
181 seconds on this Mac's MPS device. That timing is not a three-run median.

The executed notebook's [4-step Heun example](results/heun4-seed0.json) uses
**14 NFE** and scores FID-M **79.29**, digit agreement **99.78%**, and nonduplicate
fraction **97.36%** on the same public seed. Its measured sampling time was
79 seconds. Both configurations pass the practice band. This is a comparison
at one fixed seed, not a claim that Heun always wins.

![Generated digits, requested rows 0 through 9](results/heun4-seed0.png)

Practice thresholds are FID-M ≤120, digit agreement ≥98%, and nonduplicate fraction
≥95%. A qualifying configuration with lower NFE wins; use three timing runs
for a same-device tie. These thresholds support practice. Hidden evaluation
seeds and their acceptance bands remain an instructor decision before grading.

## Sampler conventions

The course clock runs from noise at `t=0` to data at `t=1`. Guidance is
`unconditional + w*(conditional - unconditional)`: `w=0` is unconditional,
`w=1` is conditional, and `w>1` extrapolates. The [deterministic-sampler note](../notebooks/toy_data/executed/02_denoisers_and_deterministic_samplers.ipynb)
provides the solver background. This optional exercise uses a separate,
class-conditional MNIST model; the main notes use an unconditional 2D model.

`ddim` follows the cosine clean/noise transport. `euler` and `heun` integrate the
implied velocity. All use a final clean-image prediction to avoid evaluating
the singular velocity at `t=1`. With `w=2`, DDIM/Euler cost `2*steps` NFE and
Heun costs `2*(2*steps-1)`. At `w=0` or `w=1`, only one branch is needed.
The stochastic knob `eta` is supported only for DDIM and uses the
[authors' DDIM variance formula](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py).
The time grid, batch size, noise seeds, and sample count are fixed for comparisons.

The earlier Claude sweep is historical: it called DDIM “Euler,” treated `w=0`
as conditional, and included a Heun result from before its solver repair.
Its prefix-of-test-set comparisons also changed the class mix, so they did not
isolate sample-count effects. The released code fixes those issues; use the
new scorecards for this exercise.

## Instructor checks and training

```sh
python speedrun/test_speedrun.py
python speedrun/validate.py --device cpu
```

The tests cover CFG endpoints, actual per-image NFE across batches, velocity
integration, paired noise, deterministic replay, metric identities, duplicate
rejection, and changed-artifact rejection. `validate.py` downloads MNIST if it
is not cached; the student notebook does not need it.

`train_evaluator.py` and `train_diffusion.py` are optional instructor training
scripts. They write to `checkpoints/training/` and refuse to overwrite an
existing file. Their outputs do not replace this release. The original
training seeds were not saved, so the distributed weights define the exact
baseline. `freeze_reference.py` records the reference once and also refuses
to overwrite a release.
