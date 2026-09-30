"""Frozen-denoiser sampling, with noise at t=0 and data at t=1.

Guidance is u + w*(c-u): w=0 unconditional, w=1 conditional, w>1 guided.
DDIM transports the predicted clean image and noise along the cosine schedule.
Euler and Heun integrate the implied velocity; all use a final denoising step
instead of evaluating the singular velocity at t=1. NFE counts each conditional
or unconditional evaluation per image, even when both branches share a batch.
"""
import math
import torch
import common


@torch.no_grad()
def sample(net, n, labels, device, steps=16, w=2.0, eta=0.0, order="ddim", seed=0,
           batch=128):
    if not isinstance(n, int) or n < 1 or not isinstance(steps, int) or steps < 1:
        raise ValueError("n and steps must be positive integers")
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if order not in ("ddim", "euler", "heun"):
        raise ValueError("order must be ddim, euler or heun")
    if not math.isfinite(w) or w < 0 or not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("w must be nonnegative and eta must be in [0,1]")
    if eta and order != "ddim":
        raise ValueError("eta is supported only by DDIM; Euler/Heun are deterministic")
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    if labels.shape != (n,) or (labels < 0).any() or (labels > 9).any():
        raise ValueError("supply n digit labels in [0,9]")
    # One CPU noise bank pairs configurations, independent of device and batch.
    g = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(n, 1, 28, 28, generator=g)
    outs = []
    for i in range(0, n, batch):
        x, nfe = _sample_chunk(net, initial[i:i + batch], labels[i:i + batch],
                               device, steps, w, eta, order, seed + i + 1)
        outs.append(x.cpu())
    return torch.cat(outs), nfe


@torch.no_grad()
def _sample_chunk(net, initial, labels, device, steps, w, eta, order, noise_seed):
    g = torch.Generator(device="cpu").manual_seed(noise_seed)
    x, y = initial.to(device), labels.to(device)
    y_un = torch.full_like(y, 10)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    nfe = 0

    def x1_hat(xt, t):
        nonlocal nfe
        tb = t.expand(xt.shape[0])
        if w in (0, 1):
            nfe += 1
            return net(xt, tb, y_un if w == 0 else y).clamp(-1, 1)
        out = net(torch.cat([xt, xt]), torch.cat([tb, tb]), torch.cat([y, y_un]))
        nfe += 2
        c, u = out.chunk(2)
        return (u + w * (c - u)).clamp(-1, 1)

    def velocity(xt, t):
        a, sg = common.alpha_sigma(t)
        x1 = x1_hat(xt, t)
        x0 = (xt - a * x1) / sg
        ad, sd = common.alpha_sigma_dot(t)
        return ad * x1 + sd * x0

    for i, (t, s) in enumerate(zip(ts[:-1], ts[1:])):
        if i == steps - 1:
            x = x1_hat(x, t)  # exact data endpoint, no division by sigma(1)
        elif order in ("euler", "heun"):
            v = velocity(x, t)
            pred = x + (s - t) * v
            x = pred if order == "euler" else x + (s - t) * (v + velocity(pred, s)) / 2
        else:
            at, st = common.alpha_sigma(t)
            a_s, s_s = common.alpha_sigma(s)
            x1 = x1_hat(x, t)
            x0 = (x - at * x1) / st
            # DDIM Eq. 16, translated from data->noise to course time.
            # https://github.com/ermongroup/ddim/blob/main/functions/denoising.py
            sig = eta * s_s / st * (1 - (at / a_s).square()).clamp_min(0).sqrt()
            keep = (s_s.square() - sig.square()).clamp_min(0).sqrt()
            x = a_s * x1 + keep * x0
            if eta:
                x += sig * torch.randn(x.shape, generator=g).to(device)
    return x.clamp(-1, 1), nfe
