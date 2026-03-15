"""Publication-quality figure generation for cathode ML evaluation.

Generates parity plots (2x2 per property), bar chart model comparison,
and learning curves from training metrics CSVs. All figures use the
Wong colorblind-safe palette and Nature/Science journal minimal style.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cathode_ml.evaluation.metrics import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS_ORDER,
    PROPERTIES,
)

logger = logging.getLogger(__name__)

# Nature/Science minimal style (locked decision)
NATURE_STYLE: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def apply_nature_style() -> None:
    """Apply Nature/Science journal style to matplotlib rcParams."""
    matplotlib.rcParams.update(NATURE_STYLE)


def _property_display_name(prop: str) -> str:
    """Convert property key to display name (title-cased, underscores to spaces)."""
    return prop.replace("_", " ").title()


def plot_parity(
    property_name: str,
    model_predictions: dict,
    output_path: str,
) -> None:
    """Generate a 2x2 parity plot for a single property.

    Creates a figure with one panel per model. Missing models get invisible panels.

    Args:
        property_name: Target property key.
        model_predictions: Dict mapping model keys to dicts with
            y_true, y_pred (arrays), mae, r2 (floats).
        output_path: File path for the output PNG.
    """
    n_models = len(MODELS_ORDER)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes_flat = axes.flatten()

    # Hide extra subplots
    for i in range(n_models, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for idx, model_key in enumerate(MODELS_ORDER):
        ax = axes_flat[idx]

        if model_key not in model_predictions:
            ax.set_visible(False)
            continue

        data = model_predictions[model_key]
        y_true = np.asarray(data["y_true"])
        y_pred = np.asarray(data["y_pred"])
        mae = data["mae"]
        r2 = data["r2"]

        # Scatter plot
        ax.scatter(
            y_true,
            y_pred,
            c=MODEL_COLORS[model_key],
            s=10,
            alpha=0.6,
            edgecolors="none",
        )

        # Diagonal reference line
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Labels and title
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(MODEL_LABELS[model_key])

        # R-squared and MAE annotation (upper-left)
        ax.text(
            0.05,
            0.95,
            f"R\u00b2 = {r2:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=7,
        )

    fig.suptitle(_property_display_name(property_name), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_bar_comparison(all_results: dict, output_path: str) -> None:
    """Generate per-property bar charts comparing MAE across models.

    Uses separate subplots per property so each has its own y-axis scale,
    avoiding the problem of capacity MAE (~50) dwarfing formation energy
    MAE (~0.03).

    Args:
        all_results: Unified results dict from load_all_results().
        output_path: File path for the output PNG.
    """
    # Collect properties that have results
    props_with_data = [p for p in PROPERTIES if p in all_results]
    if not props_with_data:
        logger.warning("No results to plot for bar comparison.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        return

    # Collect models present in any property
    models_present = [
        m for m in MODELS_ORDER
        if any(m in all_results.get(p, {}) for p in props_with_data)
    ]

    n_props = len(props_with_data)
    n_models = len(models_present)

    # Property units for axis labels
    prop_units = {
        "voltage": "V",
        "capacity": "mAh/g",
        "formation_energy_per_atom": "eV/atom",
        "energy_above_hull": "eV/atom",
    }

    fig, axes = plt.subplots(1, n_props, figsize=(3 * n_props, 4))
    if n_props == 1:
        axes = [axes]

    for ax_idx, prop in enumerate(props_with_data):
        ax = axes[ax_idx]
        prop_results = all_results.get(prop, {})

        # Only plot models that have reasonable results (exclude wildly negative R2)
        models_for_prop = []
        mae_values = []
        colors = []
        for m in models_present:
            if m in prop_results:
                r2 = prop_results[m].get("r2", 0)
                # Skip models with extremely poor fits (R2 < -1)
                if r2 < -1:
                    continue
                models_for_prop.append(m)
                mae_values.append(prop_results[m]["mae"])
                colors.append(MODEL_COLORS[m])

        if not models_for_prop:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(_property_display_name(prop))
            continue

        x = np.arange(len(models_for_prop))
        bars = ax.bar(x, mae_values, color=colors, width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, mae_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}" if val < 1 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in models_for_prop],
            rotation=45, ha="right", fontsize=7,
        )
        unit = prop_units.get(prop, "")
        ax.set_ylabel(f"MAE ({unit})" if unit else "MAE")
        ax.set_title(_property_display_name(prop))

    fig.suptitle("Model Comparison (MAE, lower is better)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_learning_curves(results_base: str, output_path: str) -> None:
    """Generate learning curves grid from per-epoch training CSVs.

    Reads CSV files at ``{results_base}/{model}/{property}_metrics.csv``
    for CGCNN, M3GNet, and TensorNet. Grid: rows=properties, cols=models.
    Missing files show "No data" text.

    Args:
        results_base: Root directory containing model result subdirectories.
        output_path: File path for the output PNG.
    """
    base = Path(results_base)
    gnn_models = ["cgcnn", "m3gnet", "tensornet"]
    n_rows = len(PROPERTIES)
    n_cols = len(gnn_models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2.5 * n_rows))
    # Ensure axes is 2D even with single row
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, prop in enumerate(PROPERTIES):
        for col_idx, model in enumerate(gnn_models):
            ax = axes[row_idx, col_idx]
            csv_path = base / model / f"{prop}_metrics.csv"

            if not csv_path.exists():
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=8,
                )
                ax.set_title(f"{MODEL_LABELS[model]} - {_property_display_name(prop)}", fontsize=7)
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", csv_path, exc)
                ax.text(
                    0.5, 0.5, "Read error",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=8,
                )
                ax.set_title(f"{MODEL_LABELS[model]} - {_property_display_name(prop)}", fontsize=7)
                continue

            epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))

            if "train_loss" in df.columns:
                ax.plot(epochs, df["train_loss"].values, color="#333333", label="Train", linewidth=1)

            if "val_loss" in df.columns:
                ax.plot(
                    epochs, df["val_loss"].values,
                    color=MODEL_COLORS[model],
                    label="Val",
                    linewidth=1,
                )

            ax.set_title(f"{MODEL_LABELS[model]} - {_property_display_name(prop)}", fontsize=7)
            ax.set_xlabel("Epoch", fontsize=6)
            ax.set_ylabel("Loss", fontsize=6)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
