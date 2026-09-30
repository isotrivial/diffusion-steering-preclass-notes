"""Shared pieces for the MNIST speedrun.

Course convention throughout, matching Topic 1:
    x_1 = clean data at t = 1,  x_0 = noise at t = 0,
    x_t = alpha_t * x_1 + sigma_t * x_0,
    f_theta(x_t, t) predicts the clean image x_1.

Schedule is cosine (alpha = sin(pi t / 2), sigma = cos(pi t / 2)), the VP row of
the deck's (alpha, sigma) table -- chosen because the deck itself notes cosine
keeps information longer than linear-beta on images.
"""
import gzip, hashlib, os, urllib.request
from pathlib import Path
import numpy as np
import torch

# Optional fallback for a host with an incompatible cuDNN installation.
if os.environ.get("NO_CUDNN") == "1":
    torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz",  "test_y": "t10k-labels-idx1-ubyte.gz"}


ROOT = Path(__file__).resolve().parent


def device():
    return "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")


def mnist(root=None, split="train"):
    root = ROOT / "data" if root is None else Path(root)
    os.makedirs(root, exist_ok=True)
    out = []
    for key in (f"{split}_x", f"{split}_y"):
        p = os.path.join(root, FILES[key])
        if not os.path.exists(p):
            urllib.request.urlretrieve(MIRROR + FILES[key], p)
        with gzip.open(p) as fh:
            raw = fh.read()
        if key.endswith("_x"):
            out.append(np.frombuffer(raw[16:], np.uint8).reshape(-1, 1, 28, 28).copy())
        else:
            out.append(np.frombuffer(raw[8:], np.uint8).copy())
    x, y = out
    return torch.from_numpy(x).float() / 255.0 * 2 - 1, torch.from_numpy(y).long()


def alpha_sigma(t):
    """cosine schedule; t=0 is noise, t=1 is data"""
    return torch.sin(t * torch.pi / 2), torch.cos(t * torch.pi / 2)


def alpha_sigma_dot(t):
    """d/dt of the cosine schedule -- needed for the velocity form, which is
    what a second-order solver has to average."""
    return (torch.pi / 2) * torch.cos(t * torch.pi / 2), -(torch.pi / 2) * torch.sin(t * torch.pi / 2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- evaluator
class Evaluator(nn.Module):
    """Frozen MNIST CNN. Penultimate features drive FID-M and the coverage
    diagnostic; the logits give digit accuracy. One forward pass serves all
    three, so scoring a configuration is cheap."""
    FEAT = 128

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, self.FEAT)
        self.head = nn.Linear(self.FEAT, 10)

    def features(self, x):
        return F.relu(self.fc(self.body(x).flatten(1)))

    def forward(self, x):
        f = self.features(x)
        return self.head(f), f


# ---------------------------------------------------------------- generator
class Block(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.emb = nn.Linear(emb, cout)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.emb(e)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Class-conditional, predicts the clean image x_1. Label 10 = 'no class',
    used for the unconditional branch of classifier-free guidance."""
    def __init__(self, ch=64, emb=256):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cemb = nn.Embedding(11, emb)
        self.d1, self.d2, self.d3 = Block(1, ch, emb), Block(ch, ch * 2, emb), Block(ch * 2, ch * 2, emb)
        self.mid = Block(ch * 2, ch * 2, emb)
        self.u3 = Block(ch * 4, ch * 2, emb)
        self.u2 = Block(ch * 4, ch, emb)
        self.u1 = Block(ch * 2, ch, emb)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x, t, y):
        e = self.temb(t[:, None]) + self.cemb(y)
        h1 = self.d1(x, e)                              # 28
        h2 = self.d2(F.avg_pool2d(h1, 2), e)            # 14
        h3 = self.d3(F.avg_pool2d(h2, 2), e)            # 7
        m = self.mid(h3, e)
        u = self.u3(torch.cat([m, h3], 1), e)
        u = F.interpolate(u, size=h2.shape[-2:], mode="nearest")
        u = self.u2(torch.cat([u, h2], 1), e)
        u = F.interpolate(u, size=h1.shape[-2:], mode="nearest")
        u = self.u1(torch.cat([u, h1], 1), e)
        return self.out(u)
