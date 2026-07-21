from __future__ import annotations

import sys
from pathlib import Path

import torch


COURSE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE_DIR))

from toy_course import (  # noqa: E402
    BASE_STD,
    VelocityMLP,
    additive_noise_coordinates,
    base_velocity,
    denoised_from_velocity,
    discriminant_direction,
    estimate_gaussian_stats,
    fit_linear_probe,
    forward_noising,
    gaussian_denoise,
    integrate_ode,
    make_activation_steered_velocity,
    make_noise_aligned_velocity,
    sample_labeled_mixture,
    window_gate,
)


def test_forward_noising_has_correct_endpoints():
    clean = torch.randn(16, 2)
    noise = torch.randn(16, 2)
    assert torch.allclose(forward_noising(clean, torch.zeros(16), noise), noise)
    assert torch.allclose(forward_noising(clean, torch.ones(16), noise), clean)


def test_all_integrators_solve_constant_field():
    x0 = torch.randn(12, 2)

    def constant_velocity(t, x):
        return torch.ones_like(x) * torch.tensor([0.75, -0.25])

    expected = x0 + torch.tensor([0.75, -0.25])
    for method in ("euler", "heun", "rk4"):
        endpoint = integrate_ode(constant_velocity, x0, steps=7, method=method)[-1]
        assert torch.allclose(endpoint, expected, atol=1e-6)


def test_deterministic_integrator_repeats_exactly():
    torch.manual_seed(8)
    model = VelocityMLP(hidden_dim=16)
    x0 = torch.randn(20, 2)
    first = integrate_ode(base_velocity(model), x0, steps=10, method="heun")
    second = integrate_ode(base_velocity(model), x0, steps=10, method="heun")
    assert torch.equal(first, second)


def test_velocity_to_denoiser_identity_for_exact_pair():
    x_noise = torch.randn(20, 2)
    x_data = torch.randn(20, 2)
    t = torch.rand(20)
    xt = forward_noising(x_data, t, x_noise)
    exact_velocity = x_data - x_noise
    recovered = denoised_from_velocity(xt, t, exact_velocity)
    assert torch.allclose(recovered, x_data, atol=1e-6)


def test_additive_coordinates_include_base_scale():
    torch.manual_seed(12)
    clean = torch.randn(20, 2)
    standard_noise = torch.randn(20, 2)
    base_noise = BASE_STD * standard_noise
    t = torch.linspace(0.1, 0.9, 20)
    xt = forward_noising(clean, t, base_noise)
    y, sigma = additive_noise_coordinates(xt, t)
    expected = clean + sigma * standard_noise
    assert torch.allclose(y, expected, atol=2e-6)


def test_gaussian_denoiser_approaches_observation_at_low_noise():
    y = torch.randn(32, 2)
    mean = torch.tensor([1.0, -1.0])
    covariance = torch.tensor([[1.5, 0.2], [0.2, 0.8]])
    denoised = gaussian_denoise(y, torch.full((32,), 1e-4), mean, covariance)
    assert torch.allclose(denoised, y, atol=1e-5)


def test_window_gate_is_zero_outside_window():
    t = torch.linspace(0, 1, 101)
    gate = window_gate(t, 0.25, 0.75)
    assert torch.all(gate[t < 0.25] == 0)
    assert torch.all(gate[t > 0.75] == 0)
    assert torch.isfinite(gate).all()
    assert gate.min() >= 0 and gate.max() <= 1


def test_zero_strength_guidance_matches_baseline():
    torch.manual_seed(9)
    model = VelocityMLP(hidden_dim=24)
    samples, labels = sample_labeled_mixture(600, device=torch.device("cpu"))
    stats = estimate_gaussian_stats(samples, labels)
    x = torch.randn(30, 2)
    t = torch.full((30,), 0.2)
    baseline = model(t, x)
    guided = make_noise_aligned_velocity(
        model, stats, target_class=1, strength=0.0
    )(t, x)
    assert torch.allclose(guided, baseline, atol=1e-6)


def test_linear_probe_and_direction_use_labels():
    torch.manual_seed(10)
    class0 = torch.randn(120, 12) - 0.6
    class1 = torch.randn(120, 12) + 0.6
    features = torch.cat([class0, class1])
    labels = torch.cat([torch.zeros(120), torch.ones(120)]).long()
    multiclass_labels = labels.clone()
    probe = fit_linear_probe(features, multiclass_labels)
    accuracy = (probe.predict(features) == multiclass_labels).float().mean()
    direction = discriminant_direction(features, labels, target_class=1)
    assert accuracy > 0.85
    assert torch.allclose(direction.norm(), torch.tensor(1.0), atol=1e-5)
    assert (features[labels == 1] @ direction).mean() > (features[labels == 0] @ direction).mean()


def test_zero_activation_strength_matches_baseline():
    torch.manual_seed(11)
    model = VelocityMLP(hidden_dim=20)
    direction = torch.randn(20)
    x = torch.randn(25, 2)
    t = torch.full((25,), 0.5)
    baseline = model(t, x)
    guided = make_activation_steered_velocity(
        model, direction, strength=0.0
    )(t, x)
    assert torch.allclose(guided, baseline, atol=1e-6)
