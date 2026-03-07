"""Plotly chart factory functions for the Streamlit dashboard.

Provides consistent styling using the Wong colorblind-safe palette
defined in cathode_ml.evaluation.metrics.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cathode_ml.evaluation.metrics import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS_ORDER,
)


def make_bar_comparison(
    all_results: dict,
    properties: list[str],
) -> go.Figure:
    """Create a grouped bar chart of MAE per model per property.

    Args:
        all_results: Unified results dict from :func:`get_all_results`.
        properties: List of property names to include.

    Returns:
        Plotly Figure with grouped bars.
    """
    fig = go.Figure()

    # Collect models that appear in any property
    models_present = []
    for model in MODELS_ORDER:
        for prop in properties:
            if model in all_results.get(prop, {}):
                if model not in models_present:
                    models_present.append(model)
                break

    for model in models_present:
        mae_values = []
        prop_labels = []
        for prop in properties:
            prop_data = all_results.get(prop, {})
            if model in prop_data:
                mae_values.append(prop_data[model]["mae"])
                prop_labels.append(prop.replace("_", " ").title())
            else:
                mae_values.append(None)
                prop_labels.append(prop.replace("_", " ").title())

        fig.add_trace(go.Bar(
            name=MODEL_LABELS.get(model, model),
            x=prop_labels,
            y=mae_values,
            marker_color=MODEL_COLORS.get(model, "#999999"),
        ))

    fig.update_layout(
        barmode="group",
        yaxis_title="MAE",
        xaxis_title="Property",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_white",
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig


def make_training_curves(
    df: Optional[pd.DataFrame],
    model: str,
    prop: str,
) -> Optional[go.Figure]:
    """Create a training/validation loss curve plot.

    Args:
        df: Training metrics DataFrame with columns: epoch, train_loss, val_loss.
            If None, returns None.
        model: Model key (e.g., ``"cgcnn"``).
        prop: Property name.

    Returns:
        Plotly Figure with train and val loss traces, or None if df is None.
    """
    if df is None:
        return None

    prop_title = prop.replace("_", " ").title()
    model_label = MODEL_LABELS.get(model, model)
    model_color = MODEL_COLORS.get(model, "#999999")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["epoch"],
        y=df["train_loss"],
        mode="lines",
        name="Train Loss",
        line=dict(color="#333333", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=df["epoch"],
        y=df["val_loss"],
        mode="lines",
        name="Val Loss",
        line=dict(color=model_color, width=2),
    ))

    fig.update_layout(
        title=f"{model_label} - {prop_title}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig


def make_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: str,
    prop: str,
    ids: Optional[Sequence[str]] = None,
) -> go.Figure:
    """Create an interactive parity plot (predicted vs actual).

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values.
        model: Model key.
        prop: Property name.
        ids: Optional material IDs for hover text.

    Returns:
        Plotly Figure with scatter and diagonal reference line.
    """
    prop_title = prop.replace("_", " ").title()
    model_label = MODEL_LABELS.get(model, model)
    model_color = MODEL_COLORS.get(model, "#999999")

    hover_text = ids if ids is not None else None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(color=model_color, size=6, opacity=0.7),
        text=hover_text,
        hovertemplate=(
            "True: %{x:.4f}<br>Pred: %{y:.4f}<br>ID: %{text}"
            if hover_text is not None
            else "True: %{x:.4f}<br>Pred: %{y:.4f}"
        ),
    ))

    # Diagonal reference line
    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    margin = (vmax - vmin) * 0.05
    fig.add_trace(go.Scatter(
        x=[vmin - margin, vmax + margin],
        y=[vmin - margin, vmax + margin],
        mode="lines",
        name="Perfect",
        line=dict(color="#BBBBBB", dash="dash", width=1),
        showlegend=False,
    ))

    fig.update_layout(
        title=f"{model_label} - {prop_title}",
        xaxis_title="True",
        yaxis_title="Predicted",
        template="plotly_white",
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig
