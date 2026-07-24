"""Plotting and repeated evaluation used by the pre-class notebooks.

The notebooks keep the scientific transformations visible.  This module holds
figure construction, metric sweeps, and repeated experiment plumbing so those
details do not obscure the computation being introduced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import linear_sum_assignment

from toy_notes import (
    BASE_STD,
    CIRCLE_RADIUS,
    CLASS_COLORS,
    CLASS_NAMES,
    EIGHT_GAUSSIAN_CENTERS,
    EIGHT_GAUSSIAN_COLORS,
    GaussianStats,
    VelocityMLP,
    base_velocity,
    denoised_from_velocity,
    discriminant_direction,
    estimate_gaussian_stats,
    evaluate_samples,
    fit_linear_probe,
    integrate_ode,
    plot_labeled_points,
    predict_classes,
    sample_base,
    set_seed,
    timed_sample,
    wasserstein_distance,
    window_gate,
)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    fontsize: float = 10,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#333333",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.4,
            color=color,
        )
    )


def plot_generator_pipeline() -> None:
    """Draw the train/generate distinction without exposing patch boilerplate."""
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.6))
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")

    axes[0].set_title(
        "Training: use examples to learn a reusable velocity rule",
        loc="left",
        fontsize=12,
    )
    _box(axes[0], (0.02, 0.56), 0.20, 0.25, "data examples\n$x_{data}$", "#D9EAD3")
    _box(axes[0], (0.02, 0.13), 0.20, 0.25, "noise samples\n$x_{noise}$", "#E7E6E6")
    _box(
        axes[0],
        (0.36, 0.34),
        0.25,
        0.28,
        "build a training pair\n$(x_t, x_{data}-x_{noise})$",
        "#FCE5CD",
    )
    _box(
        axes[0],
        (0.75, 0.34),
        0.21,
        0.28,
        "fit velocity model\n$v_\\theta(t,x_t)$",
        "#CFE2F3",
    )
    _arrow(axes[0], (0.22, 0.68), (0.36, 0.51))
    _arrow(axes[0], (0.22, 0.25), (0.36, 0.45))
    _arrow(axes[0], (0.61, 0.48), (0.75, 0.48))
    axes[0].text(
        0.02,
        0.02,
        "The network input contains $(t,x_t)$, not class labels.",
        fontsize=9,
        color="#555555",
    )

    axes[1].set_title(
        "Generation: start from fresh noise; no clean target is supplied",
        loc="left",
        fontsize=12,
    )
    _box(axes[1], (0.02, 0.34), 0.20, 0.28, "fresh noise seed\n$x_{noise}$", "#E7E6E6")
    _box(
        axes[1],
        (0.38, 0.34),
        0.25,
        0.28,
        "frozen $v_\\theta$\n+ ODE sampler",
        "#CFE2F3",
    )
    _box(axes[1], (0.78, 0.34), 0.18, 0.28, "new generated\nsample", "#EAD1DC")
    _arrow(axes[1], (0.22, 0.48), (0.38, 0.48))
    _arrow(axes[1], (0.63, 0.48), (0.78, 0.48))
    axes[1].text(
        0.02, 0.12, "Different seeds provide diversity.", fontsize=9, color="#555555"
    )
    axes[1].text(
        0.66,
        0.12,
        "Same seed + same settings = same endpoint.",
        fontsize=9,
        color="#555555",
    )
    fig.suptitle("How a flow model learns and generates", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_flow_matching_batch(
    x_noise: torch.Tensor,
    x_data: torch.Tensor,
    x_t: torch.Tensor,
    target_velocity: torch.Tensor,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        x_noise[:, 0].cpu(),
        x_noise[:, 1].cpu(),
        s=18,
        color="#777777",
        label="noise endpoint",
    )
    ax.scatter(
        x_data[:, 0].cpu(),
        x_data[:, 1].cpu(),
        s=28,
        color="#0072B2",
        label="data endpoint",
    )
    ax.scatter(
        x_t[:, 0].cpu(), x_t[:, 1].cpu(), s=24, color="#E69F00", label="training state"
    )
    ax.quiver(
        x_t[:, 0].cpu(),
        x_t[:, 1].cpu(),
        target_velocity[:, 0].cpu(),
        target_velocity[:, 1].cpu(),
        angles="xy",
        scale_units="xy",
        scale=3.2,
        width=0.004,
        color="#222222",
        alpha=0.75,
    )
    ax.set_title("One flow-matching minibatch")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    plt.show()


def plot_loss_curve(losses: Iterable[float], *, title: str) -> None:
    values = np.asarray(list(losses), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(values, color="#0072B2", alpha=0.35, lw=0.8)
    if len(values) >= 100:
        smooth = np.convolve(values, np.ones(100) / 100, mode="valid")
        ax.plot(
            np.arange(99, len(values)),
            smooth,
            color="#D55E00",
            lw=2,
            label="100-step mean",
        )
        ax.legend(frameon=False)
    ax.set(xlabel="optimization step", ylabel="MSE", title=title)
    ax.grid(alpha=0.2)
    plt.show()


def plot_flow_endpoints(
    source: torch.Tensor,
    target: torch.Tensor,
    target_labels: torch.Tensor,
    generated: torch.Tensor,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].scatter(
        source[:, 0].cpu(), source[:, 1].cpu(), s=6, alpha=0.25, color="#777777"
    )
    axes[0].set_title("Initial Gaussian noise")
    plot_labeled_points(
        target, target_labels, ax=axes[1], title="Target data", alpha=0.35
    )
    axes[2].scatter(
        generated[:, 0].cpu(), generated[:, 1].cpu(), s=7, alpha=0.35, color="#0072B2"
    )
    axes[2].set_title("Generated endpoints")
    for ax in axes:
        ax.set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.2, 4.5))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_flow_snapshots(
    path: torch.Tensor,
    generated: torch.Tensor,
    target: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    count: int = 1500,
) -> None:
    """Show how one fixed batch is partitioned by its eventual destination."""
    stats = estimate_gaussian_stats(target, target_labels)
    final_labels = predict_classes(generated, stats).cpu()
    indexes = torch.linspace(0, path.shape[1] - 1, min(count, path.shape[1])).long()
    snapshot_steps = np.linspace(0, path.shape[0] - 1, 5).round().astype(int)

    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(16, 3.4))
    for ax, step in zip(axes, snapshot_steps):
        points = path[step][indexes]
        labels = final_labels[indexes]
        for class_id, color in enumerate(CLASS_COLORS):
            mask = labels == class_id
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=6,
                alpha=0.34,
                color=color,
            )
        ax.set(
            title=f"t={step / (path.shape[0] - 1):.2f}",
            aspect="equal",
            xlim=(-4.5, 4.5),
            ylim=(-4.2, 4.5),
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "One deterministic flow; color is assigned from each final endpoint",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def flow_endpoint_diagnostics(
    generated: torch.Tensor,
    target: torch.Tensor,
    target_labels: torch.Tensor,
) -> list[dict[str, float | bool]]:
    stats = estimate_gaussian_stats(target, target_labels)
    predicted = predict_classes(generated, stats)
    balance = torch.bincount(predicted, minlength=len(CLASS_NAMES)).float()
    balance /= balance.sum()
    return [
        {
            "diagnostic": "empirical matching distance to full target",
            "value": wasserstein_distance(generated, target),
        },
        {
            "diagnostic": "generated class balance max error",
            "value": float(torch.max(torch.abs(balance - 1 / len(CLASS_NAMES))).cpu()),
        },
        {
            "diagnostic": "smallest generated mode fraction",
            "value": float(balance.min().cpu()),
        },
        {
            "diagnostic": "all samples finite",
            "value": bool(torch.isfinite(generated).all()),
        },
    ]


def plot_wasserstein_matching(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    count: int = 120,
) -> float:
    generated_indexes = torch.linspace(0, generated.shape[0] - 1, count).round().long()
    target_indexes = torch.linspace(0, target.shape[0] - 1, count).round().long()
    x = generated[generated_indexes.to(generated.device)]
    y = target[target_indexes.to(target.device)]
    costs = torch.cdist(x, y.to(x.device)).cpu().numpy()
    rows, columns = linear_sum_assignment(costs)
    matched = costs[rows, columns]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].scatter(
        y[:, 0].cpu(), y[:, 1].cpu(), s=16, alpha=0.45, color="#D55E00", label="target"
    )
    axes[0].scatter(
        x[:, 0].cpu(),
        x[:, 1].cpu(),
        s=16,
        alpha=0.45,
        color="#0072B2",
        label="generated",
    )
    axes[0].set(
        title="1. Two empirical distributions",
        aspect="equal",
        xlim=(-4.5, 4.5),
        ylim=(-4.2, 4.5),
    )
    axes[0].legend(frameon=False, loc="upper left")

    for row, column in zip(rows, columns):
        endpoints = torch.stack([x[row], y[column].to(x.device)]).cpu()
        axes[1].plot(
            endpoints[:, 0], endpoints[:, 1], color="#999999", alpha=0.22, lw=0.7
        )
    axes[1].scatter(y[:, 0].cpu(), y[:, 1].cpu(), s=13, alpha=0.55, color="#D55E00")
    axes[1].scatter(x[:, 0].cpu(), x[:, 1].cpu(), s=13, alpha=0.55, color="#0072B2")
    axes[1].set(
        title="2. Minimum-cost one-to-one matching",
        aspect="equal",
        xlim=(-4.5, 4.5),
        ylim=(-4.2, 4.5),
    )

    axes[2].hist(matched, bins=18, color="#0072B2", alpha=0.82)
    axes[2].axvline(
        matched.mean(), color="#D55E00", lw=2, label="mean matched distance"
    )
    axes[2].set(
        title="3. Distances in the optimal matching",
        xlabel="matched Euclidean distance",
        ylabel="number of pairs",
    )
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.18, axis="y")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Empirical Wasserstein matching distance = {matched.mean():.3f}", y=1.02
    )
    plt.tight_layout()
    plt.show()
    return float(matched.mean())


def plot_velocity_field(
    model: VelocityMLP,
    device: torch.device,
    *,
    times: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> None:
    grid_1d = torch.linspace(-4, 4, 19, device=device)
    gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    fig, axes = plt.subplots(1, len(times), figsize=(4.7 * len(times), 4.2))
    for ax, t_value in zip(np.atleast_1d(axes), times):
        t = torch.full((grid.shape[0],), t_value, device=device)
        with torch.no_grad():
            vectors = model(t, grid)
        ax.quiver(
            grid[:, 0].cpu(),
            grid[:, 1].cpu(),
            vectors[:, 0].cpu(),
            vectors[:, 1].cpu(),
            angles="xy",
            scale_units="xy",
            scale=18,
            width=0.003,
            color="#333333",
        )
        ax.set(
            title=f"Learned velocity at t={t_value:.2f}",
            aspect="equal",
            xlim=(-4.3, 4.3),
            ylim=(-4.3, 4.3),
        )
    plt.tight_layout()
    plt.show()


def run_flow_case(
    model_loader: Callable[..., tuple[VelocityMLP, list[float], bool]],
    target_sampler: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    seed: int,
    count: int = 4096,
    steps: int = 80,
    sampler_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model, losses, _ = model_loader(device=device)
    set_seed(seed)
    source = sample_base(count, device=device)
    target, labels = target_sampler(count, device=device, **dict(sampler_kwargs or {}))
    path = integrate_ode(base_velocity(model), source, steps=steps, method="heun")
    return {
        "model": model,
        "losses": losses,
        "source": source,
        "target": target,
        "labels": labels,
        "path": path,
        "generated": path[-1].to(device),
        "steps": steps,
    }


def circle_flow_metrics(case: Mapping[str, Any]) -> list[dict[str, float]]:
    target = case["target"]
    generated = case["generated"]
    source = case["source"]
    target_radii = torch.linalg.norm(target, dim=1)
    generated_radii = torch.linalg.norm(generated, dim=1)
    target_angles = torch.remainder(torch.atan2(target[:, 1], target[:, 0]), 2 * np.pi)
    generated_angles = torch.remainder(
        torch.atan2(generated[:, 1], generated[:, 0]), 2 * np.pi
    )
    target_bins = torch.floor(16 * target_angles / (2 * np.pi)).long().clamp_max(15)
    generated_bins = (
        torch.floor(16 * generated_angles / (2 * np.pi)).long().clamp_max(15)
    )
    target_occupancy = torch.bincount(target_bins, minlength=16).float()
    generated_occupancy = torch.bincount(generated_bins, minlength=16).float()
    target_occupancy /= target_occupancy.sum()
    generated_occupancy /= generated_occupancy.sum()
    source_distance = wasserstein_distance(source, target)
    generated_distance = wasserstein_distance(generated, target)
    return [
        {"diagnostic": "target mean radius", "value": float(target_radii.mean().cpu())},
        {
            "diagnostic": "generated mean radius",
            "value": float(generated_radii.mean().cpu()),
        },
        {
            "diagnostic": "generated radial MAE",
            "value": float(
                torch.mean(torch.abs(generated_radii - CIRCLE_RADIUS)).cpu()
            ),
        },
        {
            "diagnostic": "angular occupancy total variation",
            "value": float(
                (0.5 * torch.abs(generated_occupancy - target_occupancy).sum()).cpu()
            ),
        },
        {
            "diagnostic": "generated-to-target empirical matching distance",
            "value": generated_distance,
        },
        {
            "diagnostic": "generated/source distance ratio",
            "value": generated_distance / source_distance,
        },
    ]


def plot_circle_flow(case: Mapping[str, Any]) -> None:
    source = case["source"]
    target = case["target"]
    generated = case["generated"]
    path = case["path"]
    steps = int(case["steps"])
    target_angles = torch.remainder(torch.atan2(target[:, 1], target[:, 0]), 2 * np.pi)
    generated_angles = torch.remainder(
        torch.atan2(generated[:, 1], generated[:, 0]), 2 * np.pi
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].scatter(
        source[:, 0].cpu(), source[:, 1].cpu(), s=5, alpha=0.20, color="#777777"
    )
    axes[0].set_title("Gaussian source")
    axes[1].scatter(
        target[:, 0].cpu(),
        target[:, 1].cpu(),
        c=target_angles.cpu(),
        cmap="twilight",
        s=5,
        alpha=0.35,
    )
    axes[1].set_title("Noisy-circle target")
    axes[2].scatter(
        generated[:, 0].cpu(),
        generated[:, 1].cpu(),
        c=generated_angles.cpu(),
        cmap="twilight",
        s=5,
        alpha=0.35,
    )
    axes[2].set_title("Generated endpoints")
    for ax in axes:
        ax.set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    snapshot_steps = np.linspace(0, steps, 5).round().astype(int)
    indexes = torch.linspace(0, path.shape[1] - 1, 1800).round().long()
    colors = generated_angles[indexes.to(generated.device)].cpu()
    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(16, 3.4))
    for ax, step in zip(axes, snapshot_steps):
        points = path[step][indexes]
        ax.scatter(
            points[:, 0], points[:, 1], c=colors, cmap="twilight", s=5, alpha=0.38
        )
        ax.set(
            title=f"t={step / steps:.2f}",
            aspect="equal",
            xlim=(-4.5, 4.5),
            ylim=(-4.5, 4.5),
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "A deterministic flow forms a continuous ring; color follows final angle",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def eight_gaussian_metrics(
    case: Mapping[str, Any],
) -> tuple[list[dict[str, float | int | bool]], list[dict[str, float | int]]]:
    generated = case["generated"]
    target = case["target"]
    target_labels = case["labels"]
    source = case["source"]
    centers = EIGHT_GAUSSIAN_CENTERS.to(generated.device)
    assignments = torch.cdist(generated, centers).argmin(dim=1)
    fractions = torch.bincount(assignments, minlength=8).float()
    fractions /= fractions.sum()
    mode_rows: list[dict[str, float | int]] = []
    for component in range(8):
        generated_mode = generated[assignments == component]
        target_mode = target[target_labels == component]
        target_mean = target_mode.mean(dim=0)
        target_rms = torch.sqrt((target_mode - target_mean).square().sum(dim=1).mean())
        centroid_error = torch.linalg.norm(
            generated_mode.mean(dim=0) - target_mean
        ) / target_rms.clamp_min(1e-8)
        spread_ratio = torch.var(
            generated_mode, dim=0, unbiased=False
        ).sum() / torch.var(target_mode, dim=0, unbiased=False).sum().clamp_min(1e-8)
        mode_rows.append(
            {
                "mode": component,
                "generated share": float(fractions[component].cpu()),
                "normalized centroid error": float(centroid_error.cpu()),
                "within-mode trace ratio": float(spread_ratio.cpu()),
            }
        )
    source_distance = wasserstein_distance(source, target)
    generated_distance = wasserstein_distance(generated, target)
    summary = [
        {
            "diagnostic": "modes with at least 5% generated mass",
            "value": int((fractions >= 0.05).sum().cpu()),
        },
        {
            "diagnostic": "occupancy total variation from uniform",
            "value": float((0.5 * torch.abs(fractions - 1 / 8).sum()).cpu()),
        },
        {
            "diagnostic": "generated-to-target empirical matching distance",
            "value": generated_distance,
        },
        {
            "diagnostic": "generated/source distance ratio",
            "value": generated_distance / source_distance,
        },
        {
            "diagnostic": "mean normalized centroid error",
            "value": float(
                np.mean([row["normalized centroid error"] for row in mode_rows])
            ),
        },
        {
            "diagnostic": "median within-mode trace ratio",
            "value": float(
                np.median([row["within-mode trace ratio"] for row in mode_rows])
            ),
        },
        {
            "diagnostic": "all endpoints finite",
            "value": bool(torch.isfinite(generated).all()),
        },
    ]
    return summary, mode_rows


def plot_eight_gaussian_flow(case: Mapping[str, Any]) -> None:
    source = case["source"]
    target = case["target"]
    target_labels = case["labels"]
    generated = case["generated"]
    path = case["path"]
    steps = int(case["steps"])
    centers = EIGHT_GAUSSIAN_CENTERS.to(generated.device)
    assignments = torch.cdist(generated, centers).argmin(dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].scatter(
        source[:, 0].cpu(), source[:, 1].cpu(), s=5, alpha=0.22, color="#777777"
    )
    axes[0].set_title("Fixed Gaussian source")
    for component in range(8):
        points = target[target_labels == component].cpu()
        axes[1].scatter(
            points[:, 0],
            points[:, 1],
            s=6,
            alpha=0.30,
            color=EIGHT_GAUSSIAN_COLORS[component],
        )
    axes[1].set_title("Balanced eight-Gaussian target")
    for ax in axes:
        ax.set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    snapshot_steps = np.linspace(0, steps, 5).round().astype(int)
    indexes = torch.linspace(0, path.shape[1] - 1, 1600).long()
    colors = assignments[indexes.to(generated.device)].cpu()
    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(16, 3.4))
    for ax, step in zip(axes, snapshot_steps):
        points = path[step][indexes]
        for component in range(8):
            mask = colors == component
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=5,
                alpha=0.33,
                color=EIGHT_GAUSSIAN_COLORS[component],
            )
        ax.set(
            title=f"t={step / steps:.2f}",
            aspect="equal",
            xlim=(-4.5, 4.5),
            ylim=(-4.5, 4.5),
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "One deterministic learned flow map; color is assigned from the final endpoint",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

def plot_velocity_to_denoiser(
    x_t: torch.Tensor,
    pair_velocity: torch.Tensor,
    learned_velocity: torch.Tensor,
    clean_estimate: torch.Tensor,
    paired_clean: torch.Tensor,
    *,
    t_value: float,
) -> None:
    subset = torch.arange(
        0, x_t.shape[0], max(1, x_t.shape[0] // 30), device=x_t.device
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax in axes:
        ax.scatter(x_t[:, 0].cpu(), x_t[:, 1].cpu(), s=8, alpha=0.16, color="#777777")
        ax.set(aspect="equal", xlim=(-5, 5), ylim=(-5, 5))
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].quiver(
        x_t[subset, 0].cpu(),
        x_t[subset, 1].cpu(),
        pair_velocity[subset, 0].cpu(),
        pair_velocity[subset, 1].cpu(),
        angles="xy",
        scale_units="xy",
        scale=3.0,
        width=0.004,
        color="#D55E00",
    )
    axes[0].set_title("Training target: endpoint pair is known")
    axes[1].quiver(
        x_t[subset, 0].cpu(),
        x_t[subset, 1].cpu(),
        learned_velocity[subset, 0].cpu(),
        learned_velocity[subset, 1].cpu(),
        angles="xy",
        scale_units="xy",
        scale=3.0,
        width=0.004,
        color="#0072B2",
    )
    axes[1].set_title("Learned velocity: only (t, x_t) is known")
    for index in subset:
        line = torch.stack([x_t[index], clean_estimate[index]]).cpu()
        axes[2].plot(line[:, 0], line[:, 1], color="#999999", alpha=0.45, lw=0.8)
    axes[2].scatter(
        clean_estimate[:, 0].cpu(),
        clean_estimate[:, 1].cpu(),
        s=10,
        alpha=0.35,
        color="#0072B2",
        label="$D_\\theta$",
    )
    axes[2].scatter(
        paired_clean[:, 0].cpu(),
        paired_clean[:, 1].cpu(),
        s=14,
        alpha=0.25,
        facecolors="none",
        edgecolors="#D55E00",
        label="paired clean endpoint",
    )
    axes[2].set_title("$D_\\theta =$ current state + remaining velocity")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"The velocity-to-denoiser conversion at t={t_value:.2f}", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_denoiser_times(
    model: VelocityMLP,
    clean: torch.Tensor,
    noise: torch.Tensor,
    *,
    times: tuple[float, ...] = (0.15, 0.45, 0.75),
) -> None:
    fig, axes = plt.subplots(1, len(times), figsize=(4.7 * len(times), 4.2))
    for ax, t_value in zip(np.atleast_1d(axes), times):
        t = torch.full((clean.shape[0],), t_value, device=clean.device)
        x_t = t[:, None] * clean + (1 - t[:, None]) * noise
        with torch.no_grad():
            velocity = model(t, x_t)
            denoised = denoised_from_velocity(x_t, t, velocity)
        subset = torch.arange(0, clean.shape[0], 15, device=clean.device)
        ax.scatter(
            x_t[:, 0].cpu(),
            x_t[:, 1].cpu(),
            s=7,
            alpha=0.18,
            color="#777777",
            label="current state",
        )
        ax.scatter(
            denoised[:, 0].cpu(),
            denoised[:, 1].cpu(),
            s=8,
            alpha=0.35,
            color="#0072B2",
            label="denoised estimate",
        )
        arrows = denoised[subset] - x_t[subset]
        ax.quiver(
            x_t[subset, 0].cpu(),
            x_t[subset, 1].cpu(),
            arrows[:, 0].cpu(),
            arrows[:, 1].cpu(),
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
            color="#D55E00",
            alpha=0.8,
        )
        ax.set(
            title=f"Denoiser view at t={t_value:.2f}",
            aspect="equal",
            xlim=(-5, 5),
            ylim=(-5, 5),
        )
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()


def solver_metrics_from_paths(
    paths: Mapping[tuple[str, int], torch.Tensor],
    reference_endpoint: torch.Tensor,
    target: torch.Tensor,
) -> tuple[list[dict[str, float | int | str]], dict[str, torch.Tensor]]:
    rows: list[dict[str, float | int | str]] = []
    saved_paths: dict[str, torch.Tensor] = {}
    for (method, steps), path in paths.items():
        rows.append(
            {
                "solver": method,
                "steps": steps,
                "model evaluations": steps * {"euler": 1, "heun": 2, "rk4": 4}[method],
                "paired endpoint RMSE": float(
                    torch.sqrt(torch.mean((path[-1] - reference_endpoint) ** 2))
                ),
                "empirical matching distance to target": wasserstein_distance(
                    path[-1], target.cpu()
                ),
            }
        )
        if (method, steps) in {("euler", 8), ("heun", 16), ("rk4", 16)}:
            saved_paths[f"{method}, {steps} steps"] = path
    return rows, saved_paths


def plot_solver_accuracy(table: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in table.groupby("solver"):
        ax.loglog(
            group["model evaluations"],
            group["paired endpoint RMSE"],
            marker="o",
            label=method,
        )
    ax.set(
        xlabel="model evaluations",
        ylabel="paired endpoint RMSE",
        title="Numerical accuracy at fixed initial noise",
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    plt.show()


def plot_pca_components(
    full_stats: Any,
    class_stats: Any,
    *,
    image_shape: tuple[int, int] = (28, 28),
    components: int = 3,
) -> None:
    fig, axes = plt.subplots(2, components + 1, figsize=(10, 3.8))
    axes[0, 0].imshow(
        full_stats.mean.reshape(image_shape).cpu(), cmap="gray", vmin=0, vmax=1
    )
    axes[0, 0].set_title("all-digit mean")
    axes[1, 0].imshow(
        class_stats.mean.reshape(image_shape).cpu(), cmap="gray", vmin=0, vmax=1
    )
    axes[1, 0].set_title("digit-3 mean")
    for column in range(1, components + 1):
        for row, stats, label in ((0, full_stats, "all"), (1, class_stats, "digit 3")):
            component = stats.components[:, column - 1].reshape(image_shape).cpu()
            scale = float(component.abs().max().clamp_min(1e-8))
            axes[row, column].imshow(component, cmap="RdBu_r", vmin=-scale, vmax=scale)
            axes[row, column].set_title(f"{label} PC {column}")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(
        "Means and leading covariance directions learned from MNIST training images",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_pca_shrinkage(stats: Any, *, components: int = 4) -> None:
    """Show how observation weight changes with noise in leading PCA directions."""
    sigma = torch.linspace(0, 1.5, 200, device=stats.variances.device)
    fig, ax = plt.subplots(figsize=(7, 4))
    for index, variance in enumerate(stats.variances[:components]):
        weight = variance / (variance + sigma.square())
        ax.plot(sigma.cpu(), weight.cpu(), label=f"PC {index + 1}")
    ax.set(
        xlabel="additive noise level sigma",
        ylabel="observation weight",
        title="High-variance directions retain influence longer",
        ylim=(-0.02, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    plt.show()


def plot_image_rows(
    rows: Iterable[tuple[str, torch.Tensor]],
    *,
    count: int = 10,
    image_shape: tuple[int, int] = (28, 28),
    title: str,
) -> None:
    rows = list(rows)
    fig, axes = plt.subplots(len(rows), count, figsize=(13, 1.35 * len(rows)))
    for row_index, (label, images) in enumerate(rows):
        for column in range(count):
            axes[row_index, column].imshow(
                images[column].reshape(image_shape).detach().cpu().clamp(0, 1),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row_index, column].axis("off")
        axes[row_index, 0].text(
            -0.16,
            0.5,
            label,
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
    fig.suptitle(title, y=1.01)
    plt.tight_layout(rect=(0.13, 0, 1, 0.97))
    plt.show()


def plot_class_minus_full_field(
    grid: torch.Tensor,
    delta: torch.Tensor,
    reference_data: torch.Tensor,
    reference_labels: torch.Tensor,
    *,
    target_class: int,
    t_value: float,
    start: float,
    end: float,
) -> None:
    gate_times = torch.linspace(0, 1, 300, device=grid.device)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(
        gate_times.cpu(),
        window_gate(gate_times, start, end).cpu(),
        color="#0072B2",
        lw=2,
    )
    axes[0].set(
        xlabel="t: noise to data",
        ylabel="gate g(t)",
        title="The class-minus-full correction is used at high noise",
    )
    axes[0].grid(alpha=0.2)
    axes[1].quiver(
        grid[:, 0].cpu(),
        grid[:, 1].cpu(),
        delta[:, 0].cpu(),
        delta[:, 1].cpu(),
        angles="xy",
        scale_units="xy",
        scale=1.7,
        width=0.004,
        color=CLASS_COLORS[target_class],
        alpha=0.8,
    )
    plot_labeled_points(
        reference_data[:2500],
        reference_labels[:2500],
        ax=axes[1],
        title=f"Class-minus-full denoiser change at t={t_value:.2f}",
        alpha=0.18,
    )
    axes[1].set(xlim=(-4.8, 4.8), ylim=(-4.8, 4.8))
    plt.tight_layout()
    plt.show()


plot_noise_alignment_field = plot_class_minus_full_field


def compare_steering_methods(
    methods: Mapping[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    x0: torch.Tensor,
    target_reference: torch.Tensor,
    stats: GaussianStats,
    *,
    target_class: int,
    steps: int,
) -> tuple[list[dict[str, float | str]], dict[str, torch.Tensor]]:
    rows: list[dict[str, float | str]] = []
    paths: dict[str, torch.Tensor] = {}
    for name, velocity in methods.items():
        path, seconds = timed_sample(velocity, x0, steps=steps, method="heun")
        metrics = evaluate_samples(
            path[-1].to(x0.device), target_reference, stats, target_class=target_class
        )
        rows.append({"method": name, **metrics, "seconds": seconds})
        paths[name] = path
    return rows, paths


def plot_class_minus_full_comparison(
    paths: Mapping[str, torch.Tensor],
    table: Any,
    target_reference: torch.Tensor,
    stats: GaussianStats,
    *,
    strength: float,
) -> None:
    baseline = paths["baseline"][-1].to(target_reference.device)
    steered = paths[f"class-minus-full {strength:.1f}"][-1].to(target_reference.device)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, endpoints, title in (
        (axes[0], baseline, "Baseline endpoints"),
        (axes[1], steered, f"Class-minus-full correction, strength {strength:.1f}"),
    ):
        predicted = predict_classes(endpoints, stats)
        ax.scatter(
            target_reference[:, 0].cpu(),
            target_reference[:, 1].cpu(),
            s=8,
            alpha=0.12,
            color="#555555",
        )
        for class_id, class_name in enumerate(CLASS_NAMES):
            subset = endpoints[predicted == class_id].cpu()
            ax.scatter(
                subset[:, 0],
                subset[:, 1],
                s=7,
                alpha=0.38,
                color=CLASS_COLORS[class_id],
                label=class_name,
            )
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    indexes = torch.linspace(0, baseline.shape[0] - 1, 70).long().to(baseline.device)
    changes = steered[indexes] - baseline[indexes]
    axes[2].quiver(
        baseline[indexes, 0].cpu(),
        baseline[indexes, 1].cpu(),
        changes[:, 0].cpu(),
        changes[:, 1].cpu(),
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
        color="#0072B2",
        alpha=0.55,
    )
    axes[2].set_title("Paired baseline-to-steered endpoint changes")
    for ax in axes:
        ax.set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.2, 4.5))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    sweep = table.dropna(subset=["strength"]).sort_values("strength")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    specifications = (
        ("target_rate", "target-class rate (higher)", None),
        (
            "target_wasserstein",
            "empirical matching distance to target (lower)",
            None,
        ),
        ("diversity_ratio", "diversity ratio", 1.0),
    )
    for ax, (column, label, ideal) in zip(axes, specifications):
        ax.plot(sweep["strength"], sweep[column], marker="o", color="#0072B2")
        if ideal is not None:
            ax.axhline(ideal, color="#777777", ls="--")
        ax.set(xlabel="correction strength", ylabel=label)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


plot_noise_alignment_comparison = plot_class_minus_full_comparison


def plot_activation_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.01, 0.80, "Ordinary forward pass", fontsize=11, weight="bold")
    _box(ax, (0.03, 0.56), 0.14, 0.18, "state\n$(t,x_t)$", "#E7E6E6", fontsize=9.5)
    _box(ax, (0.25, 0.56), 0.16, 0.18, "hidden feature\n$h_t$", "#CFE2F3", fontsize=9.5)
    _box(ax, (0.51, 0.56), 0.18, 0.18, "later MLP\nlayers", "#D9EAD3", fontsize=9.5)
    _box(
        ax,
        (0.79, 0.56),
        0.17,
        0.18,
        "velocity\n$v_\\theta(t,x_t)$",
        "#EAD1DC",
        fontsize=9.5,
    )
    _arrow(ax, (0.17, 0.65), (0.25, 0.65))
    _arrow(ax, (0.41, 0.65), (0.51, 0.65))
    _arrow(ax, (0.69, 0.65), (0.79, 0.65))

    ax.text(0.01, 0.34, "Steered forward pass", fontsize=11, weight="bold")
    _box(ax, (0.03, 0.10), 0.14, 0.18, "same state\n$(t,x_t)$", "#E7E6E6", fontsize=9.5)
    _box(ax, (0.25, 0.10), 0.16, 0.18, "same hidden\n$h_t$", "#CFE2F3", fontsize=9.5)
    _box(
        ax,
        (0.48, 0.10),
        0.24,
        0.18,
        "edited feature\n$h'_t=h_t+\\alpha g(t)\\,\\mathrm{RMS}(h_t)d$",
        "#FCE5CD",
        fontsize=9.2,
    )
    _box(ax, (0.79, 0.10), 0.17, 0.18, "changed\nvelocity", "#EAD1DC", fontsize=9.5)
    _arrow(ax, (0.17, 0.19), (0.25, 0.19))
    _arrow(ax, (0.41, 0.19), (0.48, 0.19))
    _arrow(ax, (0.72, 0.19), (0.79, 0.19))
    ax.text(
        0.60, 0.41, "learned direction $d$", ha="center", fontsize=9.5, color="#D55E00"
    )
    _arrow(ax, (0.60, 0.39), (0.60, 0.29), color="#D55E00")
    fig.suptitle(
        "Activation steering edits a hidden feature, not the sample position",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()
    plt.show()


def activation_probe_sweep(
    model: VelocityMLP,
    *,
    target_class: int,
    times: Iterable[float],
    seed: int,
    collect: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> tuple[list[dict[str, float]], dict[float, torch.Tensor]]:
    rows: list[dict[str, float]] = []
    directions: dict[float, torch.Tensor] = {}
    for index, t_value in enumerate(times):
        set_seed(seed + 2 * index)
        train_features, train_labels = collect(model, t_value=t_value, n_per_class=600)
        set_seed(seed + 2 * index + 1)
        test_features, test_labels = collect(model, t_value=t_value, n_per_class=400)
        probe = fit_linear_probe(train_features, train_labels)
        accuracy = float(
            (probe.predict(test_features) == test_labels).float().mean().cpu()
        )
        directions[float(t_value)] = discriminant_direction(
            train_features, train_labels, target_class
        )
        rows.append(
            {
                "t": float(t_value),
                "effective noise std": BASE_STD * (1 - float(t_value)) / float(t_value),
                "held-out probe accuracy": accuracy,
            }
        )
    return rows, directions


def plot_activation_probe_sweep(
    table: Any,
    directions: Mapping[float, torch.Tensor],
    reference_direction: torch.Tensor,
    *,
    reference_t: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    axes[0].plot(
        table["t"], table["held-out probe accuracy"], marker="o", color="#0072B2"
    )
    axes[0].axhline(1 / len(CLASS_NAMES), color="#777777", ls="--", label="chance")
    axes[0].set(
        xlabel="t (noise to data)",
        ylabel="held-out accuracy",
        ylim=(0.25, 1.02),
        title="Linear decodability across time",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    times = list(directions)
    cosines = [
        float(torch.dot(directions[t], reference_direction).cpu()) for t in times
    ]
    axes[1].plot(times, cosines, marker="o", color="#0072B2")
    axes[1].axhline(0, color="#777777", lw=1, ls="--")
    axes[1].axvspan(0.40, 0.90, color="#E69F00", alpha=0.16)
    axes[1].scatter([reference_t], [1.0], s=80, color="#D55E00", zorder=3)
    axes[1].set(
        xlabel="t (noise to data)",
        ylabel=f"cosine with t={reference_t:.2f} direction",
        title="A late direction is not the local direction at every time",
        ylim=(-0.35, 1.08),
    )
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_method_tradeoff(table: Any) -> None:
    marker_by_method = {
        "baseline": "o",
        "shuffled activation": "s",
        "activation steering": "D",
        "class-minus-full": "^",
        "gradient guidance": "v",
        "noise + activation": "P",
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, row in table[table["method"] != "activation 0.0"].iterrows():
        name = row["method"]
        ax.scatter(
            row["target_wasserstein"],
            row["target_rate"],
            s=80,
            marker=marker_by_method.get(name, "o"),
        )
        ax.annotate(
            name,
            (row["target_wasserstein"], row["target_rate"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        xlabel="empirical matching distance to target class (lower is better)",
        ylabel="target-class rate (higher is better)",
        title="Control-quality trade-off",
    )
    ax.grid(alpha=0.2)
    plt.show()
