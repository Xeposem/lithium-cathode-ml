"""Model Comparison page: Per-property metrics and training curves.

Provides detailed comparison tables and GNN training curves (DASH-01, DASH-04).
"""

from __future__ import annotations

import streamlit as st

from dashboard.utils.charts import make_training_curves
from dashboard.utils.data_loader import (
    MODEL_LABELS,
    MODELS_ORDER,
    PROPERTIES,
    get_all_results,
    get_training_csv,
)


def _render() -> None:
    """Render the Model Comparison page."""
    st.title("Model Comparison")

    all_results = get_all_results()

    if not all_results:
        st.warning("No results found. Run the evaluation pipeline first.")
        return

    # --- Property selector ---
    available_props = [p for p in PROPERTIES if p in all_results]
    if not available_props:
        st.info("No property results available.")
        return

    selected_prop = st.selectbox(
        "Property",
        available_props,
        format_func=lambda p: p.replace("_", " ").title(),
    )

    if selected_prop is None:
        return

    # --- Full metrics table for selected property ---
    st.subheader("Metrics")

    prop_data = all_results.get(selected_prop, {})
    metrics_rows = []
    for model in MODELS_ORDER:
        if model not in prop_data:
            continue
        m = prop_data[model]
        metrics_rows.append({
            "Model": MODEL_LABELS.get(model, model),
            "MAE": f"{m['mae']:.4f}",
            "RMSE": f"{m['rmse']:.4f}",
            "R-squared": f"{m['r2']:.4f}",
            "N Train": m.get("n_train", "N/A"),
            "N Test": m.get("n_test", "N/A"),
        })

    if metrics_rows:
        st.dataframe(metrics_rows, width="stretch")

    # --- Parity plots placeholder ---
    st.subheader("Parity Plots")
    st.info(
        "Parity plots require prediction arrays (available after full pipeline run). "
        "These will be displayed once prediction data is saved during evaluation."
    )

    # --- Training curves (GNN models only, DASH-04) ---
    st.subheader("Training Curves")
    st.caption("Baselines (RF, XGBoost) do not have per-epoch training curves.")

    gnn_models = ["cgcnn", "m3gnet", "tensornet"]
    cols = st.columns(len(gnn_models))

    for col, model in zip(cols, gnn_models):
        with col:
            df = get_training_csv("data/results", model, selected_prop)
            if df is not None:
                fig = make_training_curves(df, model, selected_prop)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                prop_title = selected_prop.replace("_", " ").title()
                model_label = MODEL_LABELS.get(model, model)
                st.caption(
                    f"No training CSV found for {model_label} on {prop_title}"
                )


if not getattr(st, "_is_test_mock", False):
    _render()
