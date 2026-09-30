"""Instructor-only: freeze the real-data statistics for the supplied evaluator.

Refuses to replace a release. Students use benchmark.py; they never run this.
"""
import argparse
import json
import time
import numpy as np
import torch
import common
import fidm


def main(device):
    root = common.ROOT
    refpath = root / "checkpoints/reference.npz"
    manifestpath = root / "release.json"
    if refpath.exists() or manifestpath.exists():
        raise FileExistsError("Reference or release already exists; preserve the released evaluator")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ck = torch.load(root / "checkpoints/evaluator.pt", map_location="cpu", weights_only=True)
    evaluator = common.Evaluator().to(device).eval().requires_grad_(False)
    evaluator.load_state_dict(ck["state"])
    xtr, _ = common.mnist(split="train")
    xte, yte = common.mnist(split="test")
    start = time.perf_counter()
    f = fidm.features(evaluator, xtr, device)
    mu, cov = fidm.stats(f)
    print("Extracted all 60,000 training features", flush=True)
    # A shuffled subset avoids coupling the duplicate threshold to IDX ordering.
    subset = np.random.default_rng(20260930).permutation(len(f))[:5000]
    threshold = fidm.coverage_threshold(f[subset])
    test_f = fidm.features(evaluator, xte, device)
    test_fid = fidm.frechet((mu, cov), fidm.stats(test_f))
    acc = fidm.accuracy(evaluator, xte, yte, device)
    np.savez_compressed(refpath, mu=mu, cov=cov, duplicate_threshold=threshold)
    manifest = dict(
        protocol="mnist-fidm-v1", sample_count=10000, batch_size=128,
        preprocessing="float32 [N,1,28,28]; pixel/255*2-1; no resize",
        features="Evaluator.features: 128-dimensional post-ReLU activations",
        covariance="unbiased sample covariance in float64 plus 1e-6 I",
        reference_split="MNIST training split, all 60000 images",
        duplicate_calibration=dict(split="train", n=5000, seed=20260930,
                                   threshold=threshold, multiplier=0.25),
        evaluator_test_accuracy=acc, real_test_fid_m=test_fid,
        generator_training=dict(steps=2942, minutes=15, weights="EMA"),
        reference_config=dict(steps=16, w=2.0, eta=0.0, order="ddim"),
        practice_quality_band=dict(fid_m_max=120.0, accuracy_min=0.98,
                                   nonduplicate_min=0.95),
        band_status="Practice thresholds; hidden-seed graded leaderboard not calibrated",
        public_seed=0, reference_device=device, torch_version=str(torch.__version__),
        sha256={str(p.relative_to(root)): common.sha256(p) for p in
                (root / "checkpoints/evaluator.pt", root / "checkpoints/generator.pt", refpath)},
    )
    manifestpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen reference: test FID-M={test_fid:.4f}, accuracy={acc:.4f}, "
          f"duplicate threshold={threshold:.4f}, {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=common.device(), choices=["cpu", "cuda", "mps"])
    main(**vars(ap.parse_args()))
