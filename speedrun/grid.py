"""Fixed-seed preview using the released, verified checkpoint."""
import sys
import torch
import benchmark
import sample


def main(out="samples.png", steps=4, w=2.0, per_class=10):
    bench = benchmark.Speedrun()
    labels = torch.arange(10).repeat_interleave(per_class)
    x, nfe = sample.sample(bench.net, len(labels), labels, bench.device,
                           steps=steps, w=w, order="heun", seed=7)
    benchmark.save_grid(x, labels, out, per_class)
    print(f"{out}: rows 0-9; Heun steps={steps}, w={w}, NFE={nfe}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
