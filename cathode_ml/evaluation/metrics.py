"""Unified result loading and comparison table generation.

Loads JSON result artifacts from all four model types (RF, XGBoost,
CGCNN, MEGNet), normalizes them into a unified comparison structure,
and generates publication-quality markdown + JSON comparison tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Wong colour-blind-safe palette (locked decision)
MODEL_COLORS: dict[str, str] = {
    "rf": "#0072B2",
    "xgb": "#D55E00",
    "cgcnn": "#009E73",
    "megnet": "#CC79A7",
}

# Display labels (dagger for pretrained MEGNet)
MODEL_LABELS: dict[str, str] = {
    "rf": "RF",
    "xgb": "XGBoost",
    "cgcnn": "CGCNN",
    "megnet": "MEGNet\u2020",
}

MODELS_ORDER: list[str] = ["rf", "xgb", "cgcnn", "megnet"]

PROPERTIES: list[str] = [
    "formation_energy_per_atom",
    "voltage",
    "capacity",
    "energy_above_hull",
]

# Footnote for pretrained MEGNet
_MEGNET_FOOTNOTE = "\u2020 Fine-tuned from pretrained MEGNet-MP-2019.4.1"


def load_all_results(results_base: str = "data/results") -> dict:
    """Load results from all model types into a unified dictionary.

    Reads JSON result files from baselines (RF, XGBoost), CGCNN, and
    MEGNet subdirectories and normalizes into a single structure:
    ``{property: {model: {mae, rmse, r2, n_train, n_test}}}``.

    Args:
        results_base: Root directory containing model result subdirectories.

    Returns:
        Unified results dictionary. Empty dict if no results found.
    """
    base = Path(results_base)
    unified: dict[str, dict] = {}

    # --- Baselines (RF + XGBoost) ---
    baselines_path = base / "baselines" / "baseline_results.json"
    if baselines_path.exists():
        try:
            with open(baselines_path) as f:
                baselines = json.load(f)
            for prop, models in baselines.items():
                unified.setdefault(prop, {}).update(models)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load baselines: %s", exc)
    else:
        logger.warning("Baseline results not found: %s", baselines_path)

    # --- CGCNN ---
    cgcnn_path = base / "cgcnn" / "cgcnn_results.json"
    if cgcnn_path.exists():
        try:
            with open(cgcnn_path) as f:
                cgcnn = json.load(f)
            for prop, models in cgcnn.items():
                unified.setdefault(prop, {}).update(models)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load CGCNN results: %s", exc)
    else:
        logger.warning("CGCNN results not found: %s", cgcnn_path)

    # --- MEGNet ---
    megnet_path = base / "megnet" / "megnet_results.json"
    if megnet_path.exists():
        try:
            with open(megnet_path) as f:
                megnet = json.load(f)
            for prop, models in megnet.items():
                unified.setdefault(prop, {}).update(models)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load MEGNet results: %s", exc)
    else:
        logger.warning("MEGNet results not found: %s", megnet_path)

    return unified


def generate_comparison_table(all_results: dict, property_name: str) -> str:
    """Generate a markdown comparison table for a single property.

    Creates a table with models as rows and MAE, RMSE, R-squared as
    columns. Best values are bolded (lowest MAE/RMSE, highest R2).
    Models not present for this property are skipped.

    Args:
        all_results: Unified results dict from :func:`load_all_results`.
        property_name: Target property to generate table for.

    Returns:
        Markdown-formatted comparison table string.
    """
    prop_results = all_results.get(property_name, {})

    # Collect rows for models that have results
    rows: list[tuple[str, str, float, float, float]] = []
    for model_key in MODELS_ORDER:
        if model_key not in prop_results:
            continue
        metrics = prop_results[model_key]
        label = MODEL_LABELS[model_key]
        rows.append((
            model_key,
            label,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        ))

    if not rows:
        return f"No results available for {property_name}.\n"

    # Find best values
    best_mae = min(r[2] for r in rows)
    best_rmse = min(r[3] for r in rows)
    best_r2 = max(r[4] for r in rows)

    def _fmt(val: float, is_best: bool) -> str:
        formatted = f"{val:.4f}"
        return f"**{formatted}**" if is_best else formatted

    # Build markdown table
    lines: list[str] = []
    lines.append(f"### {property_name}\n")
    lines.append("| Model | MAE | RMSE | R-squared |")
    lines.append("|-------|-----|------|-----------|")

    for _key, label, mae, rmse, r2 in rows:
        mae_str = _fmt(mae, mae == best_mae)
        rmse_str = _fmt(rmse, rmse == best_rmse)
        r2_str = _fmt(r2, r2 == best_r2)
        lines.append(f"| {label} | {mae_str} | {rmse_str} | {r2_str} |")

    # Add footnote if MEGNet is present
    has_megnet = any(r[0] == "megnet" for r in rows)
    if has_megnet:
        lines.append("")
        lines.append(f"_{_MEGNET_FOOTNOTE}_")

    lines.append("")
    return "\n".join(lines)


def generate_all_tables(results_base: str = "data/results") -> None:
    """Generate comparison tables for all properties.

    Loads all results, generates a markdown table for each property,
    and writes combined markdown and JSON output files to the
    ``comparison/`` subdirectory.

    Args:
        results_base: Root directory containing model result subdirectories.
    """
    all_results = load_all_results(results_base)

    if not all_results:
        logger.warning("No results found; skipping table generation.")
        return

    base = Path(results_base)
    out_dir = base / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate markdown
    md_parts: list[str] = ["# Model Comparison Results\n"]
    json_data: dict = {}

    for prop in PROPERTIES:
        if prop not in all_results:
            continue
        table = generate_comparison_table(all_results, prop)
        md_parts.append(table)

        # Build JSON structure for this property
        prop_json: dict = {}
        for model_key in MODELS_ORDER:
            if model_key in all_results[prop]:
                prop_json[model_key] = all_results[prop][model_key]
        json_data[prop] = prop_json

    # Write markdown
    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(md_parts))
    logger.info("Comparison markdown written to %s", md_path)

    # Write JSON
    json_path = out_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("Comparison JSON written to %s", json_path)
