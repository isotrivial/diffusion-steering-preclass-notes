"""Runnable CIFAR-10 bridge using an unconditional NVIDIA EDM checkpoint.

The bridge keeps the image experiment separate from the six core toy notes. It
adds a class-minus-full PCA/Wiener denoiser correction to EDM's clean-image
estimate during a fixed early sampling window. No target point is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import subprocess
import sys
import tarfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path.resolve())
    return manifest


def _git_commit(path: str | Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_assets(manifest: dict[str, Any], verify_hashes: bool = True) -> dict[str, Any]:
    """Fail plainly when a source checkout or data artifact is not the locked one."""

    required_sections = {
        "generator",
        "checkpoint",
        "data",
        "pca_statistics",
        "evaluator",
        "sampler",
        "steering",
        "evaluation",
    }
    missing = sorted(required_sections - manifest.keys())
    if missing:
        raise ValueError(f"manifest is missing sections: {missing}")

    source_checks = {}
    for name, section in [
        ("edm", manifest["generator"]),
        ("evaluator", manifest["evaluator"]),
    ]:
        source_path = Path(section["source_path"])
        if not source_path.is_dir():
            raise FileNotFoundError(f"missing {name} source checkout: {source_path}")
        observed = _git_commit(source_path)
        expected = section["source_commit"]
        if observed != expected:
            raise RuntimeError(
                f"{name} source revision mismatch: expected {expected}, observed {observed}"
            )
        source_checks[name] = observed

    files = {
        "checkpoint": manifest["checkpoint"],
        "cifar10_test_archive": manifest["data"]["archive"],
        "pca_full": manifest["pca_statistics"]["full"],
        "pca_target": manifest["pca_statistics"]["target"],
        "pca_wrong": manifest["pca_statistics"]["wrong"],
        "evaluator_checkpoint": manifest["evaluator"]["checkpoint"],
    }
    file_checks = {}
    for name, spec in files.items():
        artifact_path = Path(spec["path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing {name}: {artifact_path}")
        observed = sha256_file(artifact_path) if verify_hashes else "not-checked"
        if verify_hashes and observed != spec["sha256"]:
            raise RuntimeError(
                f"{name} SHA-256 mismatch: expected {spec['sha256']}, observed {observed}"
            )
        file_checks[name] = {"path": str(artifact_path), "sha256": observed}

    return {"sources": source_checks, "files": file_checks}


class PCAWienerDenoiser:
    """Full-image Gaussian denoiser in a fitted PCA basis."""

    def __init__(self, filter_path: str | Path, device: torch.device):
        payload = torch.load(filter_path, map_location="cpu", weights_only=True)
        self.mean = payload["mean_patch"].float().to(device)
        self.eigenvectors = payload["eigenvectors"].float().to(device)
        self.eigenvalues = payload["eigenvalues"].float().to(device)
        self.name = str(payload.get("name", Path(filter_path).stem))
        self.num_images = int(payload.get("num_images", -1))

    @torch.no_grad()
    def __call__(self, noisy: torch.Tensor, sigma: float) -> torch.Tensor:
        shape = noisy.shape
        flat = noisy.float().reshape(shape[0], -1)
        centered = flat - self.mean.unsqueeze(0)
        coefficients = centered @ self.eigenvectors
        gains = self.eigenvalues / (self.eigenvalues + float(sigma) ** 2 + 1e-8)
        estimate = (coefficients * gains.unsqueeze(0)) @ self.eigenvectors.T
        return (estimate + self.mean.unsqueeze(0)).reshape(shape).to(noisy.dtype)


class StackedRandomGenerator:
    """One independent torch.Generator per seed, matching NVIDIA EDM."""

    def __init__(self, device: torch.device, seeds: Iterable[int]):
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size: Iterable[int], **kwargs: Any) -> torch.Tensor:
        size = tuple(size)
        if size[0] != len(self.generators):
            raise ValueError("the leading batch dimension must match the number of seeds")
        return torch.stack(
            [torch.randn(size[1:], generator=generator, **kwargs) for generator in self.generators]
        )


def load_edm_network(manifest: dict[str, Any], device: torch.device) -> torch.nn.Module:
    source_path = str(Path(manifest["generator"]["source_path"]).resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    with Path(manifest["checkpoint"]["path"]).open("rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            network = pickle.load(handle)["ema"].to(device).eval()
    if int(network.label_dim) != 0:
        raise RuntimeError(f"expected an unconditional EDM network, label_dim={network.label_dim}")
    return network


def make_edm_schedule(
    network: torch.nn.Module,
    device: torch.device,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    sigma_min = max(float(sigma_min), float(network.sigma_min))
    sigma_max = min(float(sigma_max), float(network.sigma_max))
    indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    schedule = (
        sigma_max ** (1 / rho)
        + indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return torch.cat([network.round_sigma(schedule), torch.zeros_like(schedule[:1])])


def combine_denoised_estimates(
    network_estimate: torch.Tensor,
    class_estimate: torch.Tensor,
    full_estimate: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Add the class-minus-full distribution correction to the EDM estimate."""
    return network_estimate + float(strength) * (class_estimate - full_estimate)


@torch.no_grad()
def edm_heun_sample(
    network: torch.nn.Module,
    latents: torch.Tensor,
    schedule: torch.Tensor,
    class_denoiser: PCAWienerDenoiser | None = None,
    full_denoiser: PCAWienerDenoiser | None = None,
    strength: float = 0.0,
    start_step: int = 0,
    end_step: int = 5,
    capture_steps: Iterable[int] = (),
    capture_count: int = 8,
    denoiser_combiner: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor
    ] = combine_denoised_estimates,
) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
    """NVIDIA EDM Algorithm 2 with a fixed clean-estimate correction.

    The correction is evaluated on the same noisy state as the neural denoiser:
    D_guided = D_network + strength * (D_class - D_full).
    """

    if (class_denoiser is None) != (full_denoiser is None):
        raise ValueError("class and full denoisers must be supplied together")

    capture_steps = set(int(step) for step in capture_steps)
    captures: dict[int, dict[str, Any]] = {}
    x_next = latents.to(torch.float64) * schedule[0]

    def corrected_denoised(x: torch.Tensor, sigma: torch.Tensor, step: int) -> torch.Tensor:
        denoised = network(x, sigma, None).to(torch.float64)
        if class_denoiser is not None and start_step <= step <= end_step:
            class_estimate = class_denoiser(x.float(), float(sigma))
            full_estimate = full_denoiser(x.float(), float(sigma))
            denoised = denoiser_combiner(
                denoised,
                class_estimate.to(torch.float64),
                full_estimate.to(torch.float64),
                strength,
            )
        return denoised

    for step, (sigma_cur, sigma_next) in enumerate(zip(schedule[:-1], schedule[1:])):
        x_cur = x_next
        denoised = corrected_denoised(x_cur, sigma_cur, step)
        if step in capture_steps:
            captures[step] = {
                "sigma": float(sigma_cur),
                "denoised": denoised[:capture_count].float().clamp(-1, 1).cpu(),
            }

        derivative = (x_cur - denoised) / sigma_cur
        x_next = x_cur + (sigma_next - sigma_cur) * derivative

        if step < len(schedule) - 2:
            denoised_next = corrected_denoised(x_next, sigma_next, step)
            derivative_next = (x_next - denoised_next) / sigma_next
            x_next = x_cur + (sigma_next - sigma_cur) * (
                0.5 * derivative + 0.5 * derivative_next
            )

    captures[len(schedule) - 1] = {
        "sigma": 0.0,
        "denoised": x_next[:capture_count].float().clamp(-1, 1).cpu(),
    }
    return x_next.float(), captures


def load_cifar10_test(archive_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    with tarfile.open(archive_path, "r:gz") as archive:
        member = next(member for member in archive.getmembers() if member.name.endswith("/test_batch"))
        handle = archive.extractfile(member)
        if handle is None:
            raise RuntimeError("could not read CIFAR-10 test_batch")
        payload = pickle.load(handle, encoding="bytes")
    images = torch.from_numpy(payload[b"data"].reshape(-1, 3, 32, 32)).float() / 127.5 - 1
    labels = torch.tensor(payload[b"labels"], dtype=torch.long)
    return images, labels


def load_evaluator(manifest: dict[str, Any], device: torch.device) -> torch.nn.Module:
    source_path = str(Path(manifest["evaluator"]["source_path"]).resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from pytorch_cifar_models.resnet import cifar10_resnet56

    model = cifar10_resnet56(pretrained=False)
    state = torch.load(
        manifest["evaluator"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def _preprocess_for_evaluator(images: torch.Tensor) -> torch.Tensor:
    images = (images.float().clamp(-1, 1) + 1) / 2
    mean = images.new_tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = images.new_tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def classify_with_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_logits = []
    all_features = []
    feature_box: list[torch.Tensor] = []

    def capture_features(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        feature_box.append(inputs[0].detach())

    hook = model.fc.register_forward_pre_hook(capture_features)
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            feature_box.clear()
            logits = model(_preprocess_for_evaluator(batch))
            all_logits.append(logits.cpu())
            all_features.append(feature_box[0].cpu())
    finally:
        hook.remove()

    logits = torch.cat(all_logits)
    probabilities = torch.softmax(logits, dim=1)
    return logits.argmax(dim=1), probabilities, torch.cat(all_features)


def evaluator_test_accuracy(
    model: torch.nn.Module,
    archive_path: str | Path,
    device: torch.device,
    batch_size: int,
) -> float:
    images, labels = load_cifar10_test(archive_path)
    predictions, _, _ = classify_with_features(model, images, device, batch_size)
    return float((predictions == labels).float().mean())


def _normalized_entropy(predictions: torch.Tensor) -> float:
    counts = torch.bincount(predictions, minlength=10).float()
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * nonzero.log()).sum() / math.log(10))


def _near_duplicate_rate(images: torch.Tensor, threshold: float = 0.02) -> float:
    flat = ((images.float().clamp(-1, 1) + 1) / 2).reshape(len(images), -1)
    distances = torch.cdist(flat, flat) / math.sqrt(flat.shape[1])
    distances.fill_diagonal_(float("inf"))
    return float((distances.min(dim=1).values < threshold).float().mean())


def evaluate_samples(
    images: torch.Tensor,
    evaluator: torch.nn.Module,
    target_class: int,
    target_reference_mean: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    finite = bool(torch.isfinite(images).all())
    predictions, probabilities, features = classify_with_features(
        evaluator, images, device, batch_size
    )
    centered = features - features.mean(dim=0, keepdim=True)
    covariance_trace = float(centered.square().sum() / max(len(features) - 1, 1))
    metrics = {
        "finite": finite,
        "target_rate": float((predictions == target_class).float().mean()),
        "target_probability": float(probabilities[:, target_class].mean()),
        "target_feature_mean_distance": float(
            torch.linalg.vector_norm(features.mean(dim=0) - target_reference_mean)
        ),
        "feature_covariance_trace": covariance_trace,
        "class_histogram": torch.bincount(predictions, minlength=10).tolist(),
        "class_histogram_entropy": _normalized_entropy(predictions),
        "near_duplicate_rate": _near_duplicate_rate(images.cpu()),
    }
    raw = {
        "predictions": predictions,
        "target_probabilities": probabilities[:, target_class],
        "features": features,
    }
    return metrics, raw


def paired_bootstrap_interval(
    target_indicator: torch.Tensor,
    baseline_indicator: torch.Tensor,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    differences = (target_indicator.float() - baseline_indicator.float()).numpy()
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    gains = differences[sample_indices].mean(axis=1)
    return float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))


def _method_spec(
    method: str,
    target_denoiser: PCAWienerDenoiser,
    wrong_denoiser: PCAWienerDenoiser,
    full_denoiser: PCAWienerDenoiser,
    strength: float,
) -> tuple[PCAWienerDenoiser | None, PCAWienerDenoiser | None, float]:
    if method == "baseline":
        return None, None, 0.0
    if method == "zero":
        return target_denoiser, full_denoiser, 0.0
    if method == "target":
        return target_denoiser, full_denoiser, strength
    if method == "wrong":
        return wrong_denoiser, full_denoiser, strength
    raise ValueError(f"unknown method: {method}")


def run_bridge(
    manifest: dict[str, Any],
    verify_hashes: bool = True,
    verify_evaluator: bool = True,
    denoiser_combiner: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor
    ] = combine_denoised_estimates,
) -> dict[str, Any]:
    """Run the locked paired experiment and return plots plus JSON-ready metrics."""

    asset_checks = validate_assets(manifest, verify_hashes=verify_hashes)
    if not torch.cuda.is_available():
        raise RuntimeError("the CIFAR-10 bridge requires a CUDA GPU")
    device = torch.device("cuda")
    torch.manual_seed(int(manifest["evaluation"]["sampling_seed"]))
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    network = load_edm_network(manifest, device)
    evaluator_device = torch.device("cpu")
    evaluator = load_evaluator(manifest, evaluator_device)
    full_denoiser = PCAWienerDenoiser(manifest["pca_statistics"]["full"]["path"], device)
    target_denoiser = PCAWienerDenoiser(
        manifest["pca_statistics"]["target"]["path"], device
    )
    wrong_denoiser = PCAWienerDenoiser(
        manifest["pca_statistics"]["wrong"]["path"], device
    )

    evaluation = manifest["evaluation"]
    sampler = manifest["sampler"]
    steering = manifest["steering"]
    sample_count = int(evaluation["sample_count"])
    seed_start = int(evaluation["sampling_seed"])
    seeds = list(range(seed_start, seed_start + sample_count))
    rng = StackedRandomGenerator(device, seeds)
    latents = rng.randn([sample_count, 3, 32, 32], device=device)
    schedule = make_edm_schedule(
        network,
        device,
        num_steps=int(sampler["steps"]),
        sigma_min=float(sampler["sigma_min"]),
        sigma_max=float(sampler["sigma_max"]),
        rho=float(sampler["rho"]),
    )

    test_images, test_labels = load_cifar10_test(manifest["data"]["archive"]["path"])
    target_reference = test_images[test_labels == int(steering["target_class"])]
    _, _, target_features = classify_with_features(
        evaluator, target_reference, evaluator_device, int(evaluation["batch_size"])
    )
    target_reference_mean = target_features.mean(dim=0)

    measured_accuracy = None
    if verify_evaluator:
        measured_accuracy = evaluator_test_accuracy(
            evaluator,
            manifest["data"]["archive"]["path"],
            evaluator_device,
            int(evaluation["batch_size"]),
        )

    images = {}
    endpoints = {}
    captures = {}
    metrics = {}
    raw = {}
    runtimes = {}
    for method in ["baseline", "zero", "target", "wrong"]:
        class_denoiser, comparison_denoiser, method_strength = _method_spec(
            method,
            target_denoiser,
            wrong_denoiser,
            full_denoiser,
            float(steering["strength"]),
        )
        method_started = time.perf_counter()
        generated, method_captures = edm_heun_sample(
            network,
            latents,
            schedule,
            class_denoiser=class_denoiser,
            full_denoiser=comparison_denoiser,
            strength=method_strength,
            start_step=int(steering["start_step"]),
            end_step=int(steering["end_step"]),
            capture_steps=steering["capture_steps"] if method in {"baseline", "target"} else (),
            denoiser_combiner=denoiser_combiner,
        )
        torch.cuda.synchronize(device)
        runtimes[method] = time.perf_counter() - method_started
        endpoints[method] = generated.cpu()
        images[method] = generated.clamp(-1, 1).cpu()
        captures[method] = method_captures
        metrics[method], raw[method] = evaluate_samples(
            generated,
            evaluator,
            int(steering["target_class"]),
            target_reference_mean,
            evaluator_device,
            int(evaluation["batch_size"]),
        )
        metrics[method]["runtime_seconds"] = runtimes[method]

    repeat, _ = edm_heun_sample(network, latents, schedule)
    deterministic_repeat_max_abs = float(
        (repeat.cpu() - endpoints["baseline"]).abs().max()
    )
    zero_endpoint_max_abs = float(
        (endpoints["zero"] - endpoints["baseline"]).abs().max()
    )
    baseline_rate = metrics["baseline"]["target_rate"]
    target_gain = metrics["target"]["target_rate"] - baseline_rate
    wrong_gain = metrics["wrong"]["target_rate"] - baseline_rate
    diversity_ratio = (
        metrics["target"]["feature_covariance_trace"]
        / metrics["baseline"]["feature_covariance_trace"]
    )
    interval = paired_bootstrap_interval(
        raw["target"]["predictions"] == int(steering["target_class"]),
        raw["baseline"]["predictions"] == int(steering["target_class"]),
        seed=int(evaluation["bootstrap_seed"]),
        draws=int(evaluation["bootstrap_draws"]),
    )

    total_seconds = time.perf_counter() - started
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    minimum_accuracy = float(manifest["evaluator"]["minimum_test_accuracy"])
    checks = {
        "evaluator_accuracy": measured_accuracy is not None and measured_accuracy >= minimum_accuracy,
        "all_outputs_finite": all(method_metrics["finite"] for method_metrics in metrics.values()),
        "deterministic_repeat": deterministic_repeat_max_abs < 1e-6,
        "zero_strength_identity": zero_endpoint_max_abs < 1e-6,
        "absolute_target_rate": metrics["target"]["target_rate"]
        >= float(evaluation["minimum_target_rate"]),
        "target_rate_gain": target_gain >= float(evaluation["minimum_target_rate_gain"]),
        "paired_gain_lower_bound_positive": interval[0] > 0,
        "target_feature_distance_improves": (
            metrics["target"]["target_feature_mean_distance"]
            < metrics["baseline"]["target_feature_mean_distance"]
        ),
        "feature_diversity_preserved": diversity_ratio >= float(
            evaluation["minimum_feature_trace_ratio"]
        ),
        "wrong_class_is_specific_control": wrong_gain < 0.5 * target_gain,
        "runtime_within_budget": total_seconds < float(evaluation["maximum_runtime_seconds"]),
        "memory_within_budget": peak_memory_gb < float(evaluation["maximum_peak_memory_gb"]),
    }
    release_status = "PASS" if all(checks.values()) else "NEGATIVE RESULT"

    return {
        "manifest_path": manifest.get("_manifest_path"),
        "asset_checks": asset_checks,
        "device": torch.cuda.get_device_name(device),
        "evaluator_test_accuracy": measured_accuracy,
        "schedule": [float(value) for value in schedule.cpu()],
        "seeds": seeds,
        "target_class": int(steering["target_class"]),
        "wrong_class": int(steering["wrong_class"]),
        "strength": float(steering["strength"]),
        "metrics": metrics,
        "target_rate_gain": target_gain,
        "wrong_class_target_rate_gain": wrong_gain,
        "paired_target_rate_gain_95ci": list(interval),
        "feature_covariance_trace_ratio": diversity_ratio,
        "zero_endpoint_max_abs": zero_endpoint_max_abs,
        "deterministic_repeat_max_abs": deterministic_repeat_max_abs,
        "total_runtime_seconds": total_seconds,
        "peak_gpu_memory_gb": peak_memory_gb,
        "checks": checks,
        "release_status": release_status,
        "images": images,
        "captures": captures,
        "diagnostics": {
            method: {
                "predictions": raw[method]["predictions"],
                "target_probabilities": raw[method]["target_probabilities"],
            }
            for method in raw
        },
    }


def calibrate_strengths(
    manifest: dict[str, Any],
    strengths: Iterable[float],
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Evaluate candidate strengths on calibration seeds, never evaluation seeds."""

    validate_assets(manifest, verify_hashes=verify_hashes)
    if not torch.cuda.is_available():
        raise RuntimeError("calibration requires a CUDA GPU")
    device = torch.device("cuda")
    network = load_edm_network(manifest, device)
    evaluator_device = torch.device("cpu")
    evaluator = load_evaluator(manifest, evaluator_device)
    full_denoiser = PCAWienerDenoiser(manifest["pca_statistics"]["full"]["path"], device)
    target_denoiser = PCAWienerDenoiser(
        manifest["pca_statistics"]["target"]["path"], device
    )
    calibration = manifest["calibration"]
    sampler = manifest["sampler"]
    steering = manifest["steering"]
    seeds = list(
        range(
            int(calibration["sampling_seed"]),
            int(calibration["sampling_seed"]) + int(calibration["sample_count"]),
        )
    )
    rng = StackedRandomGenerator(device, seeds)
    latents = rng.randn([len(seeds), 3, 32, 32], device=device)
    schedule = make_edm_schedule(
        network,
        device,
        num_steps=int(sampler["steps"]),
        sigma_min=float(sampler["sigma_min"]),
        sigma_max=float(sampler["sigma_max"]),
        rho=float(sampler["rho"]),
    )
    test_images, test_labels = load_cifar10_test(manifest["data"]["archive"]["path"])
    target_reference = test_images[test_labels == int(steering["target_class"])]
    _, _, target_features = classify_with_features(
        evaluator, target_reference, evaluator_device, int(calibration["batch_size"])
    )
    target_reference_mean = target_features.mean(dim=0)

    baseline, _ = edm_heun_sample(network, latents, schedule)
    baseline_metrics, _ = evaluate_samples(
        baseline,
        evaluator,
        int(steering["target_class"]),
        target_reference_mean,
        evaluator_device,
        int(calibration["batch_size"]),
    )
    candidates = []
    strengths = [float(strength) for strength in strengths]
    end_steps = calibration.get("candidate_end_steps", [steering["end_step"]])
    for end_step in end_steps:
        for strength in strengths:
            generated, _ = edm_heun_sample(
                network,
                latents,
                schedule,
                class_denoiser=target_denoiser,
                full_denoiser=full_denoiser,
                strength=float(strength),
                start_step=int(steering["start_step"]),
                end_step=int(end_step),
            )
            candidate_metrics, _ = evaluate_samples(
                generated,
                evaluator,
                int(steering["target_class"]),
                target_reference_mean,
                evaluator_device,
                int(calibration["batch_size"]),
            )
            candidate_metrics["strength"] = float(strength)
            candidate_metrics["end_step"] = int(end_step)
            candidate_metrics["end_sigma"] = float(schedule[int(end_step)])
            candidate_metrics["target_rate_gain"] = (
                candidate_metrics["target_rate"] - baseline_metrics["target_rate"]
            )
            candidate_metrics["feature_trace_ratio"] = (
                candidate_metrics["feature_covariance_trace"]
                / baseline_metrics["feature_covariance_trace"]
            )
            candidates.append(candidate_metrics)

    eligible = [
        candidate
        for candidate in candidates
        if candidate["finite"]
        and candidate["feature_trace_ratio"]
        >= float(calibration["minimum_feature_trace_ratio"])
        and candidate["near_duplicate_rate"] <= float(calibration["maximum_near_duplicate_rate"])
    ]
    selected = max(eligible, key=lambda item: item["target_probability"]) if eligible else None
    return {
        "sampling_seed": int(calibration["sampling_seed"]),
        "sample_count": int(calibration["sample_count"]),
        "selection_rule": calibration["selection_rule"],
        "baseline": baseline_metrics,
        "candidates": candidates,
        "selected_strength": None if selected is None else selected["strength"],
        "selected_end_step": None if selected is None else selected["end_step"],
    }


def json_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"images", "captures", "diagnostics"}
    }


def write_json_summary(result: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(json_summary(result), indent=2) + "\n", encoding="utf-8")


def plot_method_grid(result: dict[str, Any], count: int = 12):
    import matplotlib.pyplot as plt

    methods = ["baseline", "target", "wrong"]
    labels = {
        "baseline": "unconditional",
        "target": f"target: {CIFAR10_CLASSES[result['target_class']]}",
        "wrong": f"wrong: {CIFAR10_CLASSES[result['wrong_class']]}",
    }
    sample_count = len(result["images"]["baseline"])
    count = min(int(count), sample_count)
    indices = torch.linspace(0, sample_count - 1, count).round().long().tolist()
    target_name = CIFAR10_CLASSES[result["target_class"]]
    fig, axes = plt.subplots(len(methods), count, figsize=(1.5 * count, 5.7))
    for row, method in enumerate(methods):
        batch = (result["images"][method] + 1) / 2
        predictions = result["diagnostics"][method]["predictions"]
        target_probabilities = result["diagnostics"][method]["target_probabilities"]
        for column, sample_index in enumerate(indices):
            axes[row, column].imshow(batch[sample_index].permute(1, 2, 0).numpy())
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(
                    f"seed {result['seeds'][sample_index]}", fontsize=8
                )
            prediction = CIFAR10_CLASSES[int(predictions[sample_index])]
            probability = float(target_probabilities[sample_index])
            axes[row, column].text(
                0.5,
                -0.06,
                f"{prediction}\nP({target_name})={probability:.2f}",
                transform=axes[row, column].transAxes,
                ha="center",
                va="top",
                fontsize=6.5,
            )
        fig.text(
            0.012,
            1 - (row + 0.5) / len(methods),
            labels[method],
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )
    fig.suptitle(
        "Paired EDM samples on fixed, evenly spaced seeds (not selected by outcome)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0.035, 0.025, 1, 0.95], h_pad=1.4)
    return fig


def first_target_conversion_index(result: dict[str, Any]) -> int:
    """Return the first seed where target steering changes non-target to target."""

    target_class = int(result["target_class"])
    baseline = result["diagnostics"]["baseline"]["predictions"]
    target = result["diagnostics"]["target"]["predictions"]
    converted = torch.nonzero(
        (baseline != target_class) & (target == target_class), as_tuple=False
    ).flatten()
    return int(converted[0]) if len(converted) else 0


def plot_denoised_trajectory(result: dict[str, Any], sample_index: int = 0):
    import matplotlib.pyplot as plt

    methods = ["baseline", "target"]
    steps = sorted(result["captures"]["baseline"])
    fig, axes = plt.subplots(2, len(steps), figsize=(2.2 * len(steps), 4.8))
    for row, method in enumerate(methods):
        for column, step in enumerate(steps):
            record = result["captures"][method][step]
            image = (record["denoised"][sample_index] + 1) / 2
            axes[row, column].imshow(image.permute(1, 2, 0).numpy())
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(
                    f"step {step}\nsigma={record['sigma']:.2g}", fontsize=9
                )
        fig.text(
            0.012,
            1 - (row + 0.5) / len(methods),
            method,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )
    seed = result.get("seeds", list(range(len(result["images"]["baseline"]))))[
        sample_index
    ]
    fig.suptitle(
        f"EDM clean-image estimates along deterministic seed {seed}", fontsize=13
    )
    plt.tight_layout(rect=[0.035, 0, 1, 0.92])
    return fig


def plot_metric_comparison(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    methods = ["baseline", "zero", "target", "wrong"]
    labels = ["baseline", "zero", "target", "wrong"]
    target_rates = [result["metrics"][method]["target_rate"] for method in methods]
    feature_distances = [
        result["metrics"][method]["target_feature_mean_distance"] for method in methods
    ]
    trace_ratios = [
        result["metrics"][method]["feature_covariance_trace"]
        / result["metrics"]["baseline"]["feature_covariance_trace"]
        for method in methods
    ]
    panels = [
        (target_rates, "Target prediction rate\n(higher is better)", None),
        (feature_distances, "Distance to target features\n(lower is better)", None),
        (trace_ratios, "Retained feature variation\n(1.0 matches baseline)", 1.0),
    ]
    colors = ["#666666", "#999999", "#0072B2", "#D55E00"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for axis, (values, title, reference) in zip(axes, panels):
        axis.bar(labels, values, color=colors)
        if reference is not None:
            axis.axhline(reference, color="#222222", ls="--", lw=1)
        axis.set_title(title, fontsize=10)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    return fig
