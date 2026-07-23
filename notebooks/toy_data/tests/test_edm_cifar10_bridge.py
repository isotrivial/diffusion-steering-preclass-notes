from __future__ import annotations

import sys
from pathlib import Path

import torch


OPTIONAL_DIR = Path(__file__).resolve().parents[1] / "optional"
sys.path.insert(0, str(OPTIONAL_DIR))

from edm_cifar10_bridge import PCAWienerDenoiser, StackedRandomGenerator, edm_heun_sample


class TinyDenoiserNetwork(torch.nn.Module):
    sigma_min = 0.0
    sigma_max = float("inf")
    label_dim = 0

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, class_labels=None):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        return x / (1 + sigma.square())


def make_filter(path: Path, mean: float) -> PCAWienerDenoiser:
    dimension = 12
    torch.save(
        {
            "mean_patch": torch.full((dimension,), mean),
            "eigenvectors": torch.eye(dimension),
            "eigenvalues": torch.linspace(0.2, 1.4, dimension),
            "img_resolution": 2,
            "img_channels": 3,
        },
        path,
    )
    return PCAWienerDenoiser(path, torch.device("cpu"))


def test_stacked_random_generator_repeats_seedwise():
    seeds = [4, 9, 15]
    first = StackedRandomGenerator(torch.device("cpu"), seeds).randn([3, 3, 2, 2])
    second = StackedRandomGenerator(torch.device("cpu"), seeds).randn([3, 3, 2, 2])
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_zero_strength_executes_correction_path_without_changing_endpoint(tmp_path):
    network = TinyDenoiserNetwork()
    latents = torch.randn(4, 3, 2, 2, generator=torch.Generator().manual_seed(7))
    schedule = torch.tensor([2.0, 0.8, 0.2, 0.0], dtype=torch.float64)
    full = make_filter(tmp_path / "full.pt", mean=0.0)
    target = make_filter(tmp_path / "target.pt", mean=0.4)

    baseline, _ = edm_heun_sample(network, latents, schedule)
    zero, _ = edm_heun_sample(
        network,
        latents,
        schedule,
        class_denoiser=target,
        full_denoiser=full,
        strength=0.0,
        start_step=0,
        end_step=2,
    )

    combiner_calls = []

    def recording_combiner(network_estimate, class_estimate, full_estimate, strength):
        combiner_calls.append(float(strength))
        return network_estimate + strength * (class_estimate - full_estimate)

    steered, _ = edm_heun_sample(
        network,
        latents,
        schedule,
        class_denoiser=target,
        full_denoiser=full,
        strength=0.5,
        start_step=0,
        end_step=2,
        denoiser_combiner=recording_combiner,
    )

    assert torch.equal(baseline, zero)
    assert not torch.equal(baseline, steered)
    assert combiner_calls
