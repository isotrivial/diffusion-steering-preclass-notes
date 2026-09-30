"""Check the released evaluator against real digits, blur, noise and duplicates."""
import argparse
import json
import time
import torch
import torch.nn.functional as F
import benchmark
import common
import fidm


def main(device=None, out=None):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    bench = benchmark.Speedrun(device)
    x, labels = common.mnist(split="test")
    start = time.perf_counter()
    real = bench.score(x, labels)
    g = torch.Generator().manual_seed(20260930)
    # Draw a random subset; the first 5,000 IDX images have a different label
    # mix from the last 5,000. That is not an isolated sample-size experiment.
    idx = torch.randperm(len(x), generator=g)[:1000]
    a = x[idx]
    blur_fid = []
    for k in (1, 2, 4):
        blur = a.clone()
        for _ in range(k):
            blur = F.avg_pool2d(F.pad(blur, (1, 1, 1, 1), mode="replicate"), 3, 1)
        f = fidm.features(bench.evaluator, blur, bench.device)
        blur_fid.append(fidm.frechet(bench.reference, fidm.stats(f)))
    noise = torch.randn(a.shape, generator=g).clamp(-1, 1)
    noise_fid = fidm.frechet(bench.reference, fidm.stats(fidm.features(bench.evaluator, noise, bench.device)))
    duplicate = a[:1].repeat(1000, 1, 1, 1)
    duplicate_fraction = fidm.coverage(fidm.features(bench.evaluator, duplicate, bench.device), bench.threshold)
    checks = dict(real_accuracy_above_99=real["accuracy"] >= .99,
                  real_fid_below_10=real["fid_m"] < 10,
                  blur_increases=blur_fid[0] < blur_fid[1] < blur_fid[2],
                  noise_worse_than_real=noise_fid > real["fid_m"],
                  duplicates_rejected=duplicate_fraction == 0.)
    result = dict(protocol=bench.manifest["protocol"], device=bench.device,
                  real_test_10000=real, blurred_1000_fid_m=blur_fid,
                  noise_1000_fid_m=noise_fid,
                  repeated_image_nonduplicate_fraction=duplicate_fraction,
                  checks=checks, seconds=time.perf_counter()-start)
    if out:
        from pathlib import Path
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not all(checks.values()):
        raise AssertionError("Evaluator sanity check failed; inspect the reported checks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--out")
    main(**vars(ap.parse_args()))
