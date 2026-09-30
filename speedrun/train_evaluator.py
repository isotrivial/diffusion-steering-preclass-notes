"""Train and freeze the evaluator. Everything the speedrun scores comes from
this one network, so it is trained once, hashed, and never touched again."""
import argparse, json, time
import torch, torch.nn.functional as F
import common
from pathlib import Path

def main(epochs=6, bs=256, lr=2e-3, out=None, seed=0):
    out = Path(out) if out else common.ROOT / "checkpoints/training/evaluator.pt"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    xtr, ytr = common.mnist(split="train"); xte, yte = common.mnist(split="test")
    m = common.Evaluator().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * (len(xtr) // bs + 1))
    t0 = time.time()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            logits, _ = m(xtr[idx].to(dev))
            loss = F.cross_entropy(logits, ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            acc = sum((m(xte[i:i+1000].to(dev))[0].argmax(1).cpu() == yte[i:i+1000]).sum().item()
                      for i in range(0, len(xte), 1000)) / len(xte)
        print(f"  epoch {ep+1}/{epochs}  test acc {acc:.4f}  ({time.time()-t0:.0f}s)")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": m.state_dict(), "test_acc": acc, "seed": seed}, out)
    print(f"  saved {out}  sha256 {common.sha256(out)[:16]}...  final acc {acc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))
