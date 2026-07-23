"""Shared utilities for the toy diffusion and flow-matching pre-class notes.

The notebooks keep the conceptual code visible and use this module for repeated
training, sampling, metrics, and plotting.  The generative model is always
unconditional; class labels are used only by post-hoc steering methods.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


NOTES_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = NOTES_DIR / "artifacts"
CHECKPOINT_PATH = ARTIFACT_DIR / "toy_flow_model.pt"
EIGHT_GAUSSIAN_CHECKPOINT_PATH = ARTIFACT_DIR / "eight_gaussian_flow_model.pt"
CIRCLE_CHECKPOINT_PATH = ARTIFACT_DIR / "circle_flow_model.pt"
MODEL_FORMAT_VERSION = 2

CLASS_NAMES = ("left", "right", "top")
CLASS_COLORS = ("#0072B2", "#D55E00", "#009E73")
BASE_STD = 1.8
CLASS_CENTERS = torch.tensor(
    [[-2.2, -1.0], [2.2, -1.0], [0.0, 2.35]], dtype=torch.float32
)
CLASS_SCALES = torch.tensor(
    [[0.45, 0.25], [0.32, 0.48], [0.55, 0.30]], dtype=torch.float32
)
CLASS_ANGLES = torch.tensor([0.35, -0.45, 0.0], dtype=torch.float32)
EIGHT_GAUSSIAN_STD = 0.20
CIRCLE_RADIUS = 3.0
CIRCLE_RADIAL_STD = 0.12
_EIGHT_GAUSSIAN_ANGLES = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8.0)
EIGHT_GAUSSIAN_CENTERS = 3.2 * torch.stack(
    [torch.cos(_EIGHT_GAUSSIAN_ANGLES), torch.sin(_EIGHT_GAUSSIAN_ANGLES)], dim=1
)
EIGHT_GAUSSIAN_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#332288",
    "#777777",
)
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rotation(angle: torch.Tensor, device: torch.device) -> torch.Tensor:
    c = torch.cos(angle).to(device)
    s = torch.sin(angle).to(device)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def sample_labeled_mixture(
    n: int,
    *,
    device: Optional[torch.device] = None,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a three-class anisotropic Gaussian mixture."""
    device = device or choose_device()
    if labels is None:
        labels = torch.randint(len(CLASS_NAMES), (n,), device=device)
    else:
        labels = labels.to(device=device, dtype=torch.long)
        n = labels.numel()

    points = torch.empty(n, 2, device=device)
    centers = CLASS_CENTERS.to(device)
    scales = CLASS_SCALES.to(device)
    angles = CLASS_ANGLES.to(device)
    for class_id in range(len(CLASS_NAMES)):
        mask = labels == class_id
        count = int(mask.sum())
        if count == 0:
            continue
        raw = torch.randn(count, 2, device=device) * scales[class_id]
        points[mask] = raw @ _rotation(angles[class_id], device).T + centers[class_id]
    return points, labels


def sample_class(
    n: int, class_id: int, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    labels = torch.full((n,), int(class_id), dtype=torch.long, device=device or choose_device())
    return sample_labeled_mixture(n, device=device, labels=labels)[0]


def sample_eight_gaussians(
    n: int,
    *,
    device: Optional[torch.device] = None,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample eight equally weighted isotropic Gaussians on a ring."""
    device = device or choose_device()
    if labels is None:
        labels = torch.randint(8, (n,), device=device)
    else:
        labels = labels.to(device=device, dtype=torch.long)
        n = labels.numel()
    centers = EIGHT_GAUSSIAN_CENTERS.to(device)
    points = centers[labels] + EIGHT_GAUSSIAN_STD * torch.randn(n, 2, device=device)
    return points, labels


def sample_noisy_circle(
    n: int,
    *,
    device: Optional[torch.device] = None,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a continuous noisy circle; labels are angular bins for plotting only."""
    device = device or choose_device()
    if labels is None:
        angles = 2.0 * math.pi * torch.rand(n, device=device)
        labels = torch.floor(8.0 * angles / (2.0 * math.pi)).long().clamp_max(7)
    else:
        labels = labels.to(device=device, dtype=torch.long)
        n = labels.numel()
        angles = (labels + torch.rand(n, device=device)) * (2.0 * math.pi / 8.0)
    radii = CIRCLE_RADIUS + CIRCLE_RADIAL_STD * torch.randn(n, device=device)
    points = radii[:, None] * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    return points, labels


def sample_base(n: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
    return BASE_STD * torch.randn(n, 2, device=device or choose_device())


def _resolve_mnist_file(data_dir: Path, filename: str, *, download: bool) -> Path:
    compressed = data_dir / filename
    uncompressed = data_dir / filename.removesuffix(".gz")
    if uncompressed.exists():
        return uncompressed
    if compressed.exists():
        return compressed
    if not download:
        raise FileNotFoundError(f"missing MNIST file: {compressed}")
    data_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{MNIST_MIRROR}/{filename}", compressed)
    return compressed


def _read_mnist_idx(path: Path) -> torch.Tensor:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        payload = handle.read()
    magic = int(np.frombuffer(payload[:4], dtype=">i4")[0])
    if magic == 2051:
        count, rows, columns = np.frombuffer(payload[4:16], dtype=">i4")
        values = np.frombuffer(payload, dtype=np.uint8, offset=16)
        return torch.from_numpy(values.copy()).reshape(int(count), int(rows) * int(columns))
    if magic == 2049:
        count = int(np.frombuffer(payload[4:8], dtype=">i4")[0])
        values = np.frombuffer(payload, dtype=np.uint8, offset=8, count=count)
        return torch.from_numpy(values.copy()).long()
    raise ValueError(f"unsupported MNIST IDX magic number {magic} in {path}")


def load_mnist_split(
    *,
    train: bool,
    data_dir: Optional[Path | str] = None,
    device: Optional[torch.device] = None,
    download: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load flattened MNIST without importing torchvision."""
    if data_dir is None:
        data_dir = Path(os.environ.get("MNIST_DATA_DIR", ARTIFACT_DIR / "mnist"))
    data_dir = Path(data_dir)
    prefix = "train" if train else "test"
    image_path = _resolve_mnist_file(data_dir, MNIST_FILES[f"{prefix}_images"], download=download)
    label_path = _resolve_mnist_file(data_dir, MNIST_FILES[f"{prefix}_labels"], download=download)
    images = _read_mnist_idx(image_path).float().div_(255.0)
    labels = _read_mnist_idx(label_path)
    if device is not None:
        images = images.to(device)
        labels = labels.to(device)
    return images, labels


def forward_noising(
    clean: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Linear noising path: x_t = t*x_data + (1-t)*x_noise."""
    if noise is None:
        noise = sample_base(clean.shape[0], device=clean.device)
    t = expand_time(t, clean)
    return t * clean + (1.0 - t) * noise


def expand_time(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=x.dtype)
    if t.ndim == 0:
        t = t.repeat(x.shape[0])
    return t.reshape(x.shape[0], 1)


class VelocityMLP(nn.Module):
    """Small unconditional velocity network with an inspectable hidden layer."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input = nn.Linear(5, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def _time_features(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = expand_time(t, x)
        return torch.cat([x, t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=1)

    def features(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.input(self._time_features(t, x)))
        return F.silu(self.hidden1(h))

    def velocity_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Continue the forward pass from the inspectable hidden layer."""
        return self.output(F.silu(self.hidden2(features)))

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        *,
        feature_direction: Optional[torch.Tensor] = None,
        feature_strength: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        h = self.features(t, x)
        if feature_direction is not None and feature_strength is not None:
            direction = feature_direction.to(device=h.device, dtype=h.dtype)
            direction = direction / direction.norm().clamp_min(1e-8)
            strength = expand_time(feature_strength, h)
            rms = h.square().mean(dim=1, keepdim=True).sqrt().detach()
            h = h + strength * rms * direction.reshape(1, -1)
        return self.velocity_from_features(h)


def train_flow_model(
    model: VelocityMLP,
    *,
    steps: int = 2500,
    batch_size: int = 1024,
    learning_rate: float = 2e-3,
    seed: int = 2026,
    progress_every: int = 500,
    data_sampler: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = sample_labeled_mixture,
) -> list[float]:
    """Train independent conditional flow matching on an unlabeled 2D dataset."""
    set_seed(seed)
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    losses: list[float] = []

    for step in range(1, steps + 1):
        x_data, _ = data_sampler(batch_size, device=device)
        x_noise = sample_base(batch_size, device=device)
        t = torch.rand(batch_size, device=device)
        xt = forward_noising(x_data, t, x_noise)
        target_velocity = x_data - x_noise

        loss = F.mse_loss(model(t, xt), target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if progress_every and (step == 1 or step % progress_every == 0 or step == steps):
            print(f"step {step:4d}/{steps} | flow-matching loss {losses[-1]:.5f}")

    model.eval()
    return losses


def save_checkpoint(model: VelocityMLP, losses: Iterable[float], path: Path = CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "hidden_dim": model.hidden_dim,
            "model_state": model.state_dict(),
            "losses": list(losses),
            "class_centers": CLASS_CENTERS,
        },
        path,
    )


def load_or_train_model(
    *,
    device: Optional[torch.device] = None,
    steps: Optional[int] = None,
    force_retrain: bool = False,
    checkpoint_path: Path = CHECKPOINT_PATH,
    trainer: Optional[Callable[[VelocityMLP, int], list[float]]] = None,
) -> Tuple[VelocityMLP, list[float], bool]:
    """Load the shared checkpoint, or train it when a notebook is run alone."""
    device = device or choose_device()
    if checkpoint_path.exists() and not force_retrain:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("format_version") == MODEL_FORMAT_VERSION:
            model = VelocityMLP(hidden_dim=int(payload["hidden_dim"])).to(device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            return model, [float(v) for v in payload.get("losses", [])], False

    if steps is None:
        steps = int(os.environ.get("TOY_NOTES_TRAIN_STEPS", "2500"))
    model = VelocityMLP().to(device)
    losses = trainer(model, steps) if trainer is not None else train_flow_model(model, steps=steps)
    save_checkpoint(model, losses, checkpoint_path)
    return model, losses, True


def load_or_train_eight_gaussian_model(
    *,
    device: Optional[torch.device] = None,
    steps: Optional[int] = None,
    force_retrain: bool = False,
    checkpoint_path: Path = EIGHT_GAUSSIAN_CHECKPOINT_PATH,
) -> Tuple[VelocityMLP, list[float], bool]:
    """Load or train the separate model used only for the eight-mode flow map."""
    device = device or choose_device()
    if steps is None:
        steps = int(os.environ.get("TOY_NOTES_EIGHT_STEPS", "3000"))
    if checkpoint_path.exists() and not force_retrain:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (
            payload.get("format_version") == MODEL_FORMAT_VERSION
            and payload.get("dataset") == "eight_gaussian_ring"
            and payload.get("training_steps") == steps
            and payload.get("component_std") == EIGHT_GAUSSIAN_STD
            and payload.get("base_std") == BASE_STD
            and torch.is_tensor(payload.get("centers"))
            and payload["centers"].shape == EIGHT_GAUSSIAN_CENTERS.shape
            and torch.allclose(
                payload["centers"].cpu(),
                EIGHT_GAUSSIAN_CENTERS,
            )
        ):
            model = VelocityMLP(hidden_dim=int(payload["hidden_dim"])).to(device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            return model, [float(v) for v in payload.get("losses", [])], False

    model = VelocityMLP().to(device)
    training_seed = 2106
    losses = train_flow_model(
        model,
        steps=steps,
        data_sampler=sample_eight_gaussians,
        seed=training_seed,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "dataset": "eight_gaussian_ring",
            "hidden_dim": model.hidden_dim,
            "model_state": model.state_dict(),
            "losses": losses,
            "centers": EIGHT_GAUSSIAN_CENTERS,
            "component_std": EIGHT_GAUSSIAN_STD,
            "base_std": BASE_STD,
            "training_seed": training_seed,
            "training_steps": steps,
        },
        checkpoint_path,
    )
    return model, losses, True


def load_or_train_circle_model(
    *,
    device: Optional[torch.device] = None,
    steps: Optional[int] = None,
    force_retrain: bool = False,
    checkpoint_path: Path = CIRCLE_CHECKPOINT_PATH,
) -> Tuple[VelocityMLP, list[float], bool]:
    """Load or train the separate unconditional model for the circle example."""
    device = device or choose_device()
    if steps is None:
        steps = int(os.environ.get("TOY_NOTES_CIRCLE_STEPS", "3000"))
    if checkpoint_path.exists() and not force_retrain:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (
            payload.get("format_version") == MODEL_FORMAT_VERSION
            and payload.get("dataset") == "noisy_circle"
            and payload.get("training_steps") == steps
            and payload.get("radius") == CIRCLE_RADIUS
            and payload.get("radial_std") == CIRCLE_RADIAL_STD
            and payload.get("base_std") == BASE_STD
        ):
            model = VelocityMLP(hidden_dim=int(payload["hidden_dim"])).to(device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            return model, [float(v) for v in payload.get("losses", [])], False

    model = VelocityMLP().to(device)
    training_seed = 2116
    losses = train_flow_model(
        model,
        steps=steps,
        data_sampler=sample_noisy_circle,
        seed=training_seed,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "dataset": "noisy_circle",
            "hidden_dim": model.hidden_dim,
            "model_state": model.state_dict(),
            "losses": losses,
            "radius": CIRCLE_RADIUS,
            "radial_std": CIRCLE_RADIAL_STD,
            "base_std": BASE_STD,
            "training_seed": training_seed,
            "training_steps": steps,
        },
        checkpoint_path,
    )
    return model, losses, True


VelocityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def integrate_ode(
    velocity: VelocityFunction,
    x0: torch.Tensor,
    *,
    steps: int = 64,
    method: str = "heun",
    return_path: bool = True,
) -> torch.Tensor:
    """Deterministic Euler, Heun, or RK4 integration on t in [0, 1]."""
    if method not in {"euler", "heun", "rk4"}:
        raise ValueError("method must be 'euler', 'heun', or 'rk4'")
    x = x0.detach().clone()
    path = [x.detach().cpu()]
    dt = 1.0 / steps

    for index in range(steps):
        t0 = torch.full((x.shape[0],), index * dt, device=x.device, dtype=x.dtype)
        t1 = torch.full((x.shape[0],), (index + 1) * dt, device=x.device, dtype=x.dtype)
        if method == "euler":
            x = x + dt * velocity(t0, x)
        elif method == "heun":
            k1 = velocity(t0, x)
            predictor = x + dt * k1
            k2 = velocity(t1, predictor)
            x = x + 0.5 * dt * (k1 + k2)
        else:
            tm = t0 + 0.5 * dt
            k1 = velocity(t0, x)
            k2 = velocity(tm, x + 0.5 * dt * k1)
            k3 = velocity(tm, x + 0.5 * dt * k2)
            k4 = velocity(t1, x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x.detach()
        if return_path:
            path.append(x.cpu())

    return torch.stack(path) if return_path else x.cpu()


def base_velocity(model: VelocityMLP) -> VelocityFunction:
    def velocity(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(t, x)

    return velocity


def denoised_from_velocity(x: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """For the linear path, x_data_hat = x_t + (1-t)*v_theta(x_t,t)."""
    return x + (1.0 - expand_time(t, x)) * velocity


def additive_noise_coordinates(x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rewrite the path as y=x_data+sigma*epsilon for standard-normal epsilon."""
    tt = expand_time(t, x).clamp_min(0.04)
    return x / tt, BASE_STD * (1.0 - tt) / tt


def window_gate(
    t: torch.Tensor,
    start: float,
    end: float,
    *,
    ramp_fraction: float = 0.15,
) -> torch.Tensor:
    """A flat window with short cosine ramps at both edges."""
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("expected 0 <= start < end <= 1")
    u = (t - start) / (end - start)
    active = ((u >= 0.0) & (u <= 1.0)).to(t.dtype)
    ramp = max(float(ramp_fraction), 1e-6)
    rise = 0.5 - 0.5 * torch.cos(math.pi * (u / ramp).clamp(0.0, 1.0))
    fall = 0.5 - 0.5 * torch.cos(math.pi * ((1.0 - u) / ramp).clamp(0.0, 1.0))
    return active * torch.minimum(rise, fall)


@dataclass
class GaussianStats:
    means: torch.Tensor
    covariances: torch.Tensor
    full_mean: torch.Tensor
    full_covariance: torch.Tensor

    def to(self, device: torch.device) -> "GaussianStats":
        return GaussianStats(
            self.means.to(device),
            self.covariances.to(device),
            self.full_mean.to(device),
            self.full_covariance.to(device),
        )


def estimate_gaussian_stats(
    samples: torch.Tensor, labels: torch.Tensor, *, ridge: float = 1e-4
) -> GaussianStats:
    samples = samples.float()
    labels = labels.long()
    means = []
    covariances = []
    eye = torch.eye(samples.shape[1], device=samples.device, dtype=samples.dtype)
    for class_id in range(len(CLASS_NAMES)):
        xc = samples[labels == class_id]
        mean = xc.mean(dim=0)
        centered = xc - mean
        cov = centered.T @ centered / max(xc.shape[0] - 1, 1) + ridge * eye
        means.append(mean)
        covariances.append(cov)
    full_mean = samples.mean(dim=0)
    centered = samples - full_mean
    full_covariance = centered.T @ centered / max(samples.shape[0] - 1, 1) + ridge * eye
    return GaussianStats(torch.stack(means), torch.stack(covariances), full_mean, full_covariance)


def make_reference_stats(
    *, n: int = 30000, device: Optional[torch.device] = None, seed: int = 31415
) -> GaussianStats:
    previous_state = torch.random.get_rng_state()
    set_seed(seed)
    samples, labels = sample_labeled_mixture(n, device=device or choose_device())
    stats = estimate_gaussian_stats(samples, labels)
    torch.random.set_rng_state(previous_state)
    return stats


def gaussian_denoise(
    y: torch.Tensor, sigma: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    """Posterior mean for y=x+sigma*epsilon under a Gaussian data model."""
    mean = mean.to(device=y.device, dtype=y.dtype)
    covariance = covariance.to(device=y.device, dtype=y.dtype)
    sigma = expand_time(sigma, y)
    eye = torch.eye(y.shape[1], device=y.device, dtype=y.dtype).expand(y.shape[0], -1, -1)
    cov = covariance.expand(y.shape[0], -1, -1)
    system = cov + sigma.square().reshape(-1, 1, 1) * eye
    centered = (y - mean).unsqueeze(-1)
    solved = torch.linalg.solve(system, centered)
    return mean + (cov @ solved).squeeze(-1)


@dataclass
class LowRankGaussianStats:
    mean: torch.Tensor
    components: torch.Tensor
    variances: torch.Tensor

    def to(self, device: torch.device) -> "LowRankGaussianStats":
        return LowRankGaussianStats(
            self.mean.to(device),
            self.components.to(device),
            self.variances.to(device),
        )


def fit_low_rank_gaussian(samples: torch.Tensor, *, rank: int = 64) -> LowRankGaussianStats:
    """Fit a Gaussian using the leading covariance eigenvectors."""
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("samples must have shape (n, dimension) with n >= 2")
    rank = min(int(rank), samples.shape[0] - 1, samples.shape[1])
    if rank < 1:
        raise ValueError("rank must be positive")
    samples = samples.float()
    mean = samples.mean(dim=0)
    centered = samples - mean
    q = min(rank + 8, samples.shape[0] - 1, samples.shape[1])
    _, singular_values, components = torch.pca_lowrank(
        centered, q=q, center=False, niter=3
    )
    variances = singular_values[:rank].square() / (samples.shape[0] - 1)
    return LowRankGaussianStats(mean, components[:, :rank], variances)


def low_rank_gaussian_denoise(
    observations: torch.Tensor,
    sigma: torch.Tensor | float,
    stats: LowRankGaussianStats,
) -> torch.Tensor:
    """Posterior mean under a low-rank Gaussian image model."""
    stats = stats.to(observations.device)
    centered = observations - stats.mean
    coordinates = centered @ stats.components
    sigma_squared = expand_time(sigma, observations).square()
    shrinkage = stats.variances.reshape(1, -1) / (
        stats.variances.reshape(1, -1) + sigma_squared
    )
    return stats.mean + (coordinates * shrinkage) @ stats.components.T


def noise_alignment_delta(
    x: torch.Tensor, t: torch.Tensor, target_class: int, stats: GaussianStats
) -> torch.Tensor:
    """Class PCA/Gaussian denoiser minus full-data PCA/Gaussian denoiser."""
    y, sigma = additive_noise_coordinates(x, t)
    class_denoised = gaussian_denoise(
        y, sigma, stats.means[target_class], stats.covariances[target_class]
    )
    full_denoised = gaussian_denoise(y, sigma, stats.full_mean, stats.full_covariance)
    return class_denoised - full_denoised


def make_noise_aligned_velocity(
    model: VelocityMLP,
    stats: GaussianStats,
    *,
    target_class: int,
    strength: float = 1.0,
    start: float = 0.04,
    end: float = 0.35,
) -> VelocityFunction:
    return add_noise_alignment(
        base_velocity(model),
        stats,
        target_class=target_class,
        strength=strength,
        start=start,
        end=end,
    )


def add_noise_alignment(
    base_velocity_fn: VelocityFunction,
    stats: GaussianStats,
    *,
    target_class: int,
    strength: float = 1.0,
    start: float = 0.04,
    end: float = 0.35,
) -> VelocityFunction:
    """Apply noise alignment to any base velocity function."""
    def guided_velocity(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            v = base_velocity_fn(t, x)
            gate = window_gate(t, start, end).reshape(-1, 1)
            remaining = (1.0 - expand_time(t, x)).clamp_min(0.04)
            clean_delta = noise_alignment_delta(x, t, target_class, stats.to(x.device))
            velocity_delta = float(strength) * gate * clean_delta / remaining
            return v + velocity_delta

    return guided_velocity


def class_log_probabilities(x: torch.Tensor, stats: GaussianStats) -> torch.Tensor:
    stats = stats.to(x.device)
    logits = []
    constant = x.shape[1] * math.log(2.0 * math.pi)
    for class_id in range(len(CLASS_NAMES)):
        mean = stats.means[class_id]
        cov = stats.covariances[class_id]
        delta = x - mean
        solve = torch.linalg.solve(cov, delta.T).T
        quadratic = (delta * solve).sum(dim=1)
        logdet = torch.logdet(cov)
        logits.append(-0.5 * (constant + logdet + quadratic))
    return torch.stack(logits, dim=1)


def predict_classes(x: torch.Tensor, stats: GaussianStats) -> torch.Tensor:
    return class_log_probabilities(x, stats).argmax(dim=1)


def make_gradient_guided_velocity(
    model: VelocityMLP,
    stats: GaussianStats,
    *,
    target_class: int,
    strength: float = 0.8,
    start: float = 0.25,
    end: float = 0.92,
) -> VelocityFunction:
    """Inference-time class guidance through the clean-estimate gradient."""
    stats = stats.to(next(model.parameters()).device)

    def velocity(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = window_gate(t, start, end).reshape(-1, 1)
        if float(gate.max()) == 0.0:
            with torch.no_grad():
                return model(t, x)

        x_req = x.detach().requires_grad_(True)
        v = model(t, x_req)
        clean = denoised_from_velocity(x_req, t, v)
        log_posterior = class_log_probabilities(clean, stats).log_softmax(dim=1)
        objective = log_posterior[:, target_class].sum()
        gradient = torch.autograd.grad(objective, x_req)[0]
        direction = gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return v.detach() + float(strength) * gate * direction.detach()

    return velocity


@dataclass
class LinearProbe:
    mean: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.mean) / self.scale
        augmented = torch.cat(
            [normalized, torch.ones(normalized.shape[0], 1, device=features.device)], dim=1
        )
        return (augmented @ self.weights).argmax(dim=1)


def fit_linear_probe(
    features: torch.Tensor, labels: torch.Tensor, *, ridge: float = 1e-2
) -> LinearProbe:
    mean = features.mean(dim=0, keepdim=True)
    scale = features.std(dim=0, keepdim=True).clamp_min(1e-4)
    normalized = (features - mean) / scale
    augmented = torch.cat(
        [normalized, torch.ones(normalized.shape[0], 1, device=features.device)], dim=1
    )
    targets = F.one_hot(labels, num_classes=len(CLASS_NAMES)).to(features.dtype)
    eye = torch.eye(augmented.shape[1], device=features.device, dtype=features.dtype)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(augmented.T @ augmented + ridge * eye, augmented.T @ targets)
    return LinearProbe(mean, scale, weights)


def collect_forward_activations(
    model: VelocityMLP,
    *,
    t_value: float,
    n_per_class: int = 800,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(n_per_class)
    clean, labels = sample_labeled_mixture(labels.numel(), device=device, labels=labels)
    noise = sample_base(clean.shape[0], device=device)
    t = torch.full((clean.shape[0],), float(t_value), device=device)
    xt = forward_noising(clean, t, noise)
    with torch.no_grad():
        features = model.features(t, xt)
    return features, labels


def discriminant_direction(
    features: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
    *,
    ridge: float = 0.1,
) -> torch.Tensor:
    """A covariance-aware target-vs-rest direction in hidden-feature space."""
    positive = features[labels == target_class]
    negative = features[labels != target_class]
    mean_delta = positive.mean(dim=0) - negative.mean(dim=0)
    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(features.shape[0] - 1, 1)
    eye = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
    direction = torch.linalg.solve(covariance + ridge * eye, mean_delta)
    return direction / direction.norm().clamp_min(1e-8)


def make_activation_steered_velocity(
    model: VelocityMLP,
    direction: torch.Tensor,
    *,
    strength: float = 1.0,
    start: float = 0.35,
    end: float = 0.90,
) -> VelocityFunction:
    direction = direction.to(next(model.parameters()).device)

    def velocity(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = window_gate(t, start, end)
        with torch.no_grad():
            return model(
                t,
                x,
                feature_direction=direction,
                feature_strength=float(strength) * gate,
            )

    return velocity


def make_combined_velocity(
    model: VelocityMLP,
    stats: GaussianStats,
    direction: torch.Tensor,
    *,
    target_class: int,
    noise_strength: float = 1.0,
    activation_strength: float = 1.0,
    noise_window: Tuple[float, float] = (0.04, 0.35),
    activation_window: Tuple[float, float] = (0.35, 0.90),
) -> VelocityFunction:
    stats = stats.to(next(model.parameters()).device)
    direction = direction.to(next(model.parameters()).device)

    def velocity(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        na_gate = window_gate(t, *noise_window).reshape(-1, 1)
        act_gate = window_gate(t, *activation_window)
        with torch.no_grad():
            v = model(
                t,
                x,
                feature_direction=direction,
                feature_strength=float(activation_strength) * act_gate,
            )
            remaining = (1.0 - expand_time(t, x)).clamp_min(0.04)
            clean_delta = noise_alignment_delta(x, t, target_class, stats)
            velocity_delta = float(noise_strength) * na_gate * clean_delta / remaining
            return v + velocity_delta

    return velocity


def wasserstein_distance(
    samples: torch.Tensor, reference: torch.Tensor, *, max_samples: int = 256
) -> float:
    """Finite-sample Wasserstein-1 estimate from minimum-cost point matching."""
    n = min(samples.shape[0], reference.shape[0], int(max_samples))
    if n < 1:
        raise ValueError("both sample sets must be nonempty")
    sample_indexes = torch.linspace(0, samples.shape[0] - 1, n).round().long()
    reference_indexes = torch.linspace(0, reference.shape[0] - 1, n).round().long()
    x = samples.detach().float()[sample_indexes.to(samples.device)]
    y_all = reference.detach().to(samples.device).float()
    y = y_all[reference_indexes.to(samples.device)]
    costs = torch.cdist(x, y).cpu().numpy()
    rows, columns = linear_sum_assignment(costs)
    return float(costs[rows, columns].mean())


def evaluate_samples(
    samples: torch.Tensor,
    target_reference: torch.Tensor,
    stats: GaussianStats,
    *,
    target_class: int,
) -> Dict[str, float]:
    samples = samples.float()
    target_reference = target_reference.to(samples).float()
    predictions = predict_classes(samples, stats.to(samples.device))
    target_rate = float((predictions == target_class).float().mean().cpu())
    target_mean = target_reference.mean(dim=0)
    sample_mean = samples.mean(dim=0)
    mean_error = float(torch.linalg.norm(sample_mean - target_mean).cpu())
    sample_trace = torch.var(samples, dim=0, unbiased=False).sum()
    target_trace = torch.var(target_reference, dim=0, unbiased=False).sum()
    diversity_ratio = float((sample_trace / target_trace.clamp_min(1e-8)).cpu())
    return {
        "target_rate": target_rate,
        "target_wasserstein": wasserstein_distance(samples, target_reference),
        "mean_error": mean_error,
        "diversity_ratio": diversity_ratio,
    }


def timed_sample(
    velocity: VelocityFunction,
    x0: torch.Tensor,
    *,
    steps: int,
    method: str,
) -> Tuple[torch.Tensor, float]:
    if x0.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    path = integrate_ode(velocity, x0, steps=steps, method=method, return_path=True)
    if x0.device.type == "cuda":
        torch.cuda.synchronize()
    return path, time.perf_counter() - started


def pca_project(features: torch.Tensor) -> torch.Tensor:
    return fit_pca_projection(features).transform(features)


@dataclass
class PCAProjection:
    mean: torch.Tensor
    components: torch.Tensor

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) @ self.components.T


def fit_pca_projection(features: torch.Tensor, *, n_components: int = 2) -> PCAProjection:
    """Fit display-only PCA parameters that can be reused on held-out features."""
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return PCAProjection(mean=mean, components=vh[:n_components])


def model_state_digest(model: nn.Module) -> str:
    """Hash model tensors so notebooks can show that post-hoc sampling is frozen."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_labeled_points(
    points: torch.Tensor,
    labels: torch.Tensor,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    alpha: float = 0.55,
) -> plt.Axes:
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    points_np = points.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    for class_id, name in enumerate(CLASS_NAMES):
        subset = points_np[labels_np == class_id]
        ax.scatter(subset[:, 0], subset[:, 1], s=9, alpha=alpha, color=CLASS_COLORS[class_id], label=name)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def plot_trajectory_comparison(
    trajectories: Dict[str, torch.Tensor],
    *,
    target_reference: Optional[torch.Tensor] = None,
    n_lines: int = 24,
) -> None:
    fig, axes = plt.subplots(1, len(trajectories), figsize=(5 * len(trajectories), 4.5), squeeze=False)
    for ax, (name, path) in zip(axes[0], trajectories.items()):
        path_np = path.detach().cpu().numpy()
        if target_reference is not None:
            reference_np = target_reference.detach().cpu().numpy()
            ax.scatter(
                reference_np[:, 0],
                reference_np[:, 1],
                s=8,
                alpha=0.13,
                color="#009E73",
                label="target examples",
            )
        count = min(n_lines, path_np.shape[1])
        indexes = np.linspace(0, path_np.shape[1] - 1, count, dtype=int)
        for index in indexes:
            ax.plot(path_np[:, index, 0], path_np[:, index, 1], color="#666666", alpha=0.32, lw=0.8)
        ax.scatter(path_np[-1, :, 0], path_np[-1, :, 1], s=7, alpha=0.38, color="#0072B2")
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.2, 4.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if target_reference is not None:
            ax.legend(frameon=False, loc="upper left", fontsize=8)
    plt.tight_layout()


def environment_summary(device: torch.device) -> Dict[str, str]:
    gpu_name = "none"
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "torch": torch.__version__,
        "device": str(device),
        "gpu": gpu_name,
        "cuda": str(torch.version.cuda),
    }
