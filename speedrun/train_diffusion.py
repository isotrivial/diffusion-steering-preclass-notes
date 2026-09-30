"""Train the class-conditional generator the speedrun freezes.

Predicts the clean image x_1 from (x_t, t, y), the deck's f_theta. Labels are
dropped 10% of the time to give classifier-free guidance an unconditional
branch (label 10). Time is sampled uniformly in t, which is the p(t) knob the
deck's Module 3 discusses -- fixed here, since the speedrun is about sampling.
"""
import argparse, os, time
import torch, torch.nn.functional as F
import common
from pathlib import Path


def main(minutes=15.0, bs=256, lr=2e-4, ch=64, drop=0.1, out=None, seed=0):
    out = Path(out) if out else common.ROOT / "checkpoints/training/generator.pt"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    x1_all, y_all = common.mnist(split="train")
    x1_all, y_all = x1_all.to(dev), y_all.to(dev)
    net = common.UNet(ch=ch).to(dev)
    ema = common.UNet(ch=ch).to(dev); ema.load_state_dict(net.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    print(f"  device {dev} | {sum(p.numel() for p in net.parameters())/1e6:.2f}M params "
          f"| budget {minutes:.0f} min")
    t0, step, running = time.time(), 0, 0.0
    while (time.time() - t0) < minutes * 60:
        idx = torch.randint(0, len(x1_all), (bs,), device=dev)
        x1, y = x1_all[idx], y_all[idx].clone()
        y[torch.rand(bs, device=dev) < drop] = 10           # unconditional branch
        t = torch.rand(bs, device=dev)
        a, s = common.alpha_sigma(t)
        x0 = torch.randn_like(x1)
        xt = a[:, None, None, None] * x1 + s[:, None, None, None] * x0
        loss = F.mse_loss(net(xt, t, y), x1)                # target is the clean image
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            d = 0.999 if step > 1000 else 0.9
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(pn, alpha=1 - d)
        running += loss.item(); step += 1
        if step % 500 == 0:
            el = time.time() - t0
            print(f"  step {step:6d}  loss {running/500:.4f}  {el/60:.1f} min "
                  f"({el/(minutes*60)*100:.0f}%)")
            running = 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": ema.state_dict(), "ch": ch, "steps": step, "seed": seed}, out)
    print(f"  saved {out} after {step} steps, {(time.time()-t0)/60:.1f} min")
    print(f"  sha256 {common.sha256(out)[:16]}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))
