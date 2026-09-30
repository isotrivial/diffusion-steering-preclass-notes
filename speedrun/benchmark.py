"""Load the released weights/reference and run a complete speedrun scorecard."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
import common
import fidm
import sample


def verify_release(root=common.ROOT):
    root = Path(root)
    manifest = json.loads((root / "release.json").read_text())
    for name, digest in manifest["sha256"].items():
        path = root / name
        if not path.is_file() or common.sha256(path) != digest:
            raise ValueError(f"Released file missing or changed: {path}. Restore the course copy.")
    return manifest


class Speedrun:
    def __init__(self, device=None, root=common.ROOT):
        self.root = Path(root)
        self.manifest = verify_release(self.root)
        self.device = device or common.device()
        ck = torch.load(self.root / "checkpoints/evaluator.pt", map_location="cpu", weights_only=True)
        self.evaluator = common.Evaluator().to(self.device)
        self.evaluator.load_state_dict(ck["state"])
        ck = torch.load(self.root / "checkpoints/generator.pt", map_location="cpu", weights_only=True)
        self.net = common.UNet(ch=ck["ch"]).to(self.device)
        self.net.load_state_dict(ck["state"])
        for model in (self.evaluator, self.net):
            model.eval().requires_grad_(False)
        with np.load(self.root / "checkpoints/reference.npz", allow_pickle=False) as ref:
            self.reference = ref["mu"], ref["cov"]
            self.threshold = float(ref["duplicate_threshold"])

    def score(self, x, labels):
        x = torch.as_tensor(x, device="cpu", dtype=torch.float32)
        labels = torch.as_tensor(labels, device="cpu", dtype=torch.long)
        if (x.shape != (len(labels), 1, 28, 28) or len(x) < 2
                or not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1
                or labels.ndim != 1 or (labels < 0).any() or (labels > 9).any()):
            raise ValueError("Expected finite [N,1,28,28] images in [-1,1] and N digit labels")
        f = fidm.features(self.evaluator, x, self.device)
        accuracy, confidence = fidm.classification(self.evaluator, x, labels, self.device)
        return dict(fid_m=fidm.frechet(self.reference, fidm.stats(f)),
                    accuracy=accuracy, target_confidence=confidence,
                    nonduplicate_fraction=fidm.coverage(f, self.threshold))

    def sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            torch.mps.synchronize()

    def run(self, config=None, *, n=None, seed=0, repeats=3, batch=None):
        config = {**self.manifest["reference_config"], **(config or {})}
        if set(config) != {"steps", "w", "eta", "order"}:
            raise ValueError("Only steps, w, eta and order may change")
        n = self.manifest["sample_count"] if n is None else n
        batch = self.manifest["batch_size"] if batch is None else batch
        if repeats < 1 or n < 10 or n % 10:
            raise ValueError("Use positive repeats and an n divisible by 10")
        verify_release(self.root)
        labels = torch.arange(n) % 10
        sample.sample(self.net, min(n, batch), labels[:batch], self.device,
                      seed=seed, batch=batch, **config)
        self.sync()
        timings = []
        for _ in range(repeats):
            self.sync()
            start = time.perf_counter()
            x, nfe = sample.sample(self.net, n, labels, self.device, seed=seed,
                                   batch=batch, **config)
            self.sync()
            timings.append(time.perf_counter() - start)
        metrics = self.score(x, labels)
        full = n == self.manifest["sample_count"] and batch == self.manifest["batch_size"]
        band = self.manifest["practice_quality_band"]
        meets_band = (metrics["fid_m"] <= band["fid_m_max"]
                      and metrics["accuracy"] >= band["accuracy_min"]
                      and metrics["nonduplicate_fraction"] >= band["nonduplicate_min"])
        row = dict(protocol=self.manifest["protocol"], config=config, n=n, seed=seed,
                   batch_size=batch, nfe_per_sample=nfe, **metrics,
                   sampling_seconds_median=statistics.median(timings),
                   sampling_seconds=timings, timing_repeats=repeats,
                   full_protocol=full, meets_practice_band=bool(full and meets_band),
                   device=self.device, torch_version=str(torch.__version__),
                   platform=platform.platform(), sha256=self.manifest["sha256"])
        return row, x


def save_grid(x, labels, path, per_class=10):
    from PIL import Image
    rows = []
    for digit in range(10):
        selected = x[labels == digit][:per_class, 0]
        rows.append(torch.cat(list(selected), dim=1))
    pixels = ((torch.cat(rows, dim=0) + 1) * 127.5).clamp(0, 255).byte().numpy()
    Image.fromarray(pixels).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--w", type=float)
    ap.add_argument("--eta", type=float)
    ap.add_argument("--order", choices=["ddim", "euler", "heun"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=common.ROOT / "results/scorecard.json")
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    bench = Speedrun(args.device)
    config = {k: getattr(args, k) for k in ("steps", "w", "eta", "order")
              if getattr(args, k) is not None}
    row, x = bench.run(config, n=args.n, seed=args.seed, repeats=args.repeats, batch=args.batch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")
    save_grid(x, torch.arange(len(x)) % 10, args.out.with_suffix(".png"))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
