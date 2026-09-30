"""Frechet distance in the frozen CNN's 128-D MNIST feature space (FID-M).

Use the same evaluator, preprocessing, reference statistics and sample count
for every comparison. These scores are not comparable to Inception FID.
The duplicate diagnostic measures separation within generated samples, not
coverage of the real distribution. Random noise can pass that diagnostic.
"""
import numpy as np
import torch

RIDGE = 1e-6


@torch.no_grad()
def features(evaluator, x, device, bs=256):
    return torch.cat([evaluator.features(x[i:i + bs].to(device)).cpu()
                      for i in range(0, len(x), bs)]).double().numpy()


def stats(f):
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2 or len(f) < 2 or not np.isfinite(f).all():
        raise ValueError("statistics require at least two finite feature vectors")
    return f.mean(0), np.cov(f, rowvar=False) + RIDGE * np.eye(f.shape[1])


def frechet(s1, s2):
    """Stable symmetric PSD square root; never discard a complex component."""
    (m1, c1), (m2, c2) = s1, s2
    eig, vec = np.linalg.eigh((c1 + c1.T) / 2)
    root = (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
    prod = root @ c2 @ root
    cross = np.sqrt(np.maximum(np.linalg.eigvalsh((prod + prod.T) / 2), 0)).sum()
    value = float(np.square(m1 - m2).sum() + np.trace(c1) + np.trace(c2) - 2 * cross)
    if not np.isfinite(value):
        raise ValueError("non-finite FID-M")
    return max(0.0, value)


def nearest_distances(f, batch=256):
    """Blockwise distances: avoid allocating a 10,000 x 10,000 matrix."""
    x = torch.as_tensor(f, dtype=torch.float64)
    if x.ndim != 2 or len(x) < 2 or not torch.isfinite(x).all():
        raise ValueError("duplicate diagnostic requires at least two finite samples")
    out = []
    for i in range(0, len(x), batch):
        d = torch.cdist(x[i:i + batch], x)
        j = torch.arange(len(d))
        d[j, i + j] = float("inf")
        out.append(d.min(1).values)
    return torch.cat(out)


def coverage_threshold(f_real, q=0.25):
    """Freeze 0.25 times the real calibration set's median NN distance."""
    return float(q * nearest_distances(f_real).median())


def coverage(f_gen, thresh):
    """Legacy API name: fraction without a near-duplicate, not mode coverage."""
    return float((nearest_distances(f_gen) > thresh).double().mean())


@torch.no_grad()
def classification(evaluator, x, y, device, bs=256):
    ok, confidence = 0, 0.0
    for i in range(0, len(x), bs):
        logits, _ = evaluator(x[i:i + bs].to(device))
        labels = y[i:i + bs].to(device)
        ok += (logits.argmax(1) == labels).sum().item()
        confidence += logits.softmax(1).gather(1, labels[:, None]).sum().item()
    return ok / len(x), confidence / len(x)


def accuracy(evaluator, x, y, device, bs=256):
    return classification(evaluator, x, y, device, bs)[0]
