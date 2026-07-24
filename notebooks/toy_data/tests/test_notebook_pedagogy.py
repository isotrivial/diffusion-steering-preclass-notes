from __future__ import annotations

import json
import re
from pathlib import Path


NOTES_DIR = Path(__file__).resolve().parents[1]
TOY_NOTEBOOKS = sorted(NOTES_DIR.glob("[0-9][0-9]_*.ipynb"))
CIFAR_NOTEBOOK = NOTES_DIR / "optional" / "cifar10_steering_with_unconditional_edm.ipynb"
MAINTAINED_NOTEBOOKS = [*TOY_NOTEBOOKS, CIFAR_NOTEBOOK]
EXECUTED_DIR = NOTES_DIR / "executed"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def markdown_text(notebook: dict) -> str:
    return "\n".join(
        cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )


def source_text(cell: dict) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_maintained_notebooks_have_prediction_before_observation_structure():
    assert len(TOY_NOTEBOOKS) == 6
    for path in MAINTAINED_NOTEBOOKS:
        text = markdown_text(load_notebook(path))
        assert len(re.findall(r"^# ", text, flags=re.MULTILINE)) == 1, path.name
        assert text.count("**Before you ") >= 2, path.name

    for path in TOY_NOTEBOOKS:
        text = markdown_text(load_notebook(path))
        assert "**Change one thing:**" in text, path.name


def test_reader_notebooks_avoid_stale_or_misleading_scaffolding():
    banned = (
        "learning goals",
        "what this notebook establishes",
        "self-audit",
        "capstone",
        "paired initial noise",
        "point anchor",
        "lambda(t)(a-x)",
        "lambda(t) (a-x)",
        "\N{GREEK SMALL LETTER LAMDA}(t)(a-x)",
        "tanh(a-x)",
        "swd",
    )
    for path in MAINTAINED_NOTEBOOKS:
        text = markdown_text(load_notebook(path)).lower()
        for phrase in banned:
            assert phrase not in text, f"{phrase!r} appears in {path.name}"


def test_source_notebooks_are_output_free_and_keep_code_cells_short():
    for path in MAINTAINED_NOTEBOOKS:
        notebook = load_notebook(path)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert not cell.get("outputs"), path.name
            nonblank_lines = sum(bool(line.strip()) for line in cell["source"].splitlines())
            assert nonblank_lines <= 35, (
                f"{path.name} has a {nonblank_lines}-line code cell; split the "
                "mechanism from plotting or bookkeeping"
            )


def test_cifar_reader_notebook_omits_protocol_bookkeeping():
    notebook = load_notebook(CIFAR_NOTEBOOK)
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    for implementation_name in (
        "calibration_result",
        "selected_calibration",
        "evidence_summary",
    ):
        assert implementation_name not in source


def test_published_notebooks_match_sources_and_contain_outputs():
    expected_images = {
        "00_diffusion_and_flow_matching_foundations.ipynb": 4,
        "01_train_unconditional_flow_matching.ipynb": 9,
        "02_denoisers_and_deterministic_samplers.ipynb": 4,
        "03_gaussian_denoisers_and_noise_alignment.ipynb": 6,
        "04_training_free_gradient_guidance.ipynb": 2,
        "05_activation_steering_and_method_comparison.ipynb": 4,
        "cifar10_steering_with_unconditional_edm.ipynb": 4,
    }
    for source_path in MAINTAINED_NOTEBOOKS:
        executed_path = EXECUTED_DIR / source_path.name
        source = load_notebook(source_path)
        executed = load_notebook(executed_path)
        assert len(source["cells"]) == len(executed["cells"]), source_path.name

        for source_cell, executed_cell in zip(source["cells"], executed["cells"]):
            assert source_cell["id"] == executed_cell["id"], source_path.name
            assert source_cell["cell_type"] == executed_cell["cell_type"], source_path.name
            assert source_text(source_cell) == source_text(executed_cell), source_path.name
            assert not any(
                output.get("output_type") == "error"
                for output in executed_cell.get("outputs", [])
            ), source_path.name

        image_count = sum(
            "image/png" in output.get("data", {})
            for cell in executed["cells"]
            for output in cell.get("outputs", [])
        )
        assert image_count >= expected_images[source_path.name], source_path.name
