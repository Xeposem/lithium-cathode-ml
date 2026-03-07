"""Data Explorer page -- dataset distributions and property correlations.

Provides per-property histograms and an interactive scatter matrix for
exploring the cleaned cathode materials dataset (DASH-03).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import PROPERTIES, get_cached_records

# Human-readable labels for properties
_PROPERTY_LABELS: dict[str, str] = {
    "formation_energy_per_atom": "Formation Energy (eV/atom)",
    "voltage": "Voltage (V)",
    "capacity": "Capacity (mAh/g)",
    "energy_above_hull": "Energy Above Hull (eV/atom)",
}

_HIST_COLOR = "#0072B2"


# ---------------------------------------------------------------------------
# Helper functions (importable for testing)
# ---------------------------------------------------------------------------


def _make_histogram(df: pd.DataFrame, column: str, color: str = _HIST_COLOR) -> go.Figure:
    """Create a Plotly histogram for a single property column.

    Args:
        df: DataFrame containing the column.
        column: Column name to plot.
        color: Bar colour.

    Returns:
        Plotly Figure.
    """
    label = _PROPERTY_LABELS.get(column, column)
    fig = go.Figure(
        go.Histogram(
            x=df[column].dropna(),
            marker_color=color,
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=label,
        xaxis_title=label,
        yaxis_title="Count",
        bargap=0.05,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _make_scatter_matrix(
    df: pd.DataFrame,
    columns: list[str],
    color_col: str | None = "source",
) -> go.Figure:
    """Create a Plotly scatter matrix for pairwise correlations.

    Args:
        df: DataFrame.
        columns: Numeric columns to include.
        color_col: Column to colour points by (default: source).

    Returns:
        Plotly Figure.
    """
    labels = {c: _PROPERTY_LABELS.get(c, c) for c in columns}
    color = color_col if color_col and color_col in df.columns else None
    fig = px.scatter_matrix(
        df.dropna(subset=columns),
        dimensions=columns,
        color=color,
        labels=labels,
        height=600,
    )
    fig.update_traces(diagonal_visible=True, showupperhalf=False)
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    return fig


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def _render() -> None:
    """Render the Data Explorer page."""
    st.title("Data Explorer")

    records = get_cached_records()
    if not records:
        st.warning("No cached data found. Run the data pipeline first.")
        return

    df = pd.DataFrame(records)

    # ---- Dataset summary metrics ----
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))

    # Records per source
    if "source" in df.columns:
        src_counts = df["source"].value_counts()
        col2.metric("MP", src_counts.get("materials_project", 0))
        col3.metric("OQMD", src_counts.get("oqmd", 0))
        col4.metric("BDG", src_counts.get("battery_data_genome", 0))

    # Count available properties
    available_props = [p for p in PROPERTIES if p in df.columns and df[p].notna().any()]
    st.caption(f"Properties with data: {len(available_props)} of {len(PROPERTIES)}")

    # ---- Property histograms (2 per row) ----
    st.subheader("Property Distributions")
    for i in range(0, len(available_props), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(available_props):
                prop = available_props[idx]
                with col:
                    fig = _make_histogram(df, prop)
                    st.plotly_chart(fig, use_container_width=True)

    # ---- Scatter matrix ----
    st.subheader("Property Correlations")
    default_cols = [p for p in available_props]
    selected = st.multiselect(
        "Properties to include",
        options=available_props,
        default=default_cols,
        key="scatter_props",
    )
    if len(selected) >= 2:
        fig = _make_scatter_matrix(df, selected)
        st.plotly_chart(fig, use_container_width=True)
    elif selected:
        st.info("Select at least 2 properties for the scatter matrix.")


_render()
