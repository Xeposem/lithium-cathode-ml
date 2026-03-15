"""Overview page: Model Performance Summary.

Landing page for the Cathode ML Dashboard (DASH-01).
Displays key findings, best-model summary table, and MAE bar chart.
"""

from __future__ import annotations

import streamlit as st

from dashboard.utils.charts import make_bar_comparison
from dashboard.utils.data_loader import (
    MODELS_ORDER,
    MODEL_LABELS,
    PROPERTIES,
    get_all_results,
)


def _render() -> None:
    """Render the Overview page."""
    st.title("Cathode ML: Model Performance Overview")

    st.markdown("""
    This dashboard presents the results of a comparative study of machine learning
    models for predicting lithium cathode material properties from crystal structure.
    Five models were evaluated -- Random Forest, XGBoost, CGCNN, M3GNet, and TensorNet --
    across four target properties: formation energy, voltage, capacity, and energy
    above hull.
    """)

    # Load results
    all_results = get_all_results()

    if not all_results:
        st.warning("No results found. Run the evaluation pipeline first.")
        return

    # --- Best model summary table ---
    st.subheader("Best Model per Property")

    summary_rows = []
    for prop in PROPERTIES:
        prop_data = all_results.get(prop, {})
        if not prop_data:
            continue
        # Find model with lowest MAE
        best_model = min(
            prop_data.keys(),
            key=lambda m: prop_data[m]["mae"],
        )
        summary_rows.append({
            "Property": prop.replace("_", " ").title(),
            "Best Model": MODEL_LABELS.get(best_model, best_model),
            "MAE": f"{prop_data[best_model]['mae']:.4f}",
            "R-squared": f"{prop_data[best_model]['r2']:.4f}",
        })

    if summary_rows:
        st.dataframe(summary_rows, width="stretch")

    # --- MAE bar comparison ---
    st.subheader("MAE Comparison Across Models")

    available_props = [p for p in PROPERTIES if p in all_results]
    if available_props:
        fig = make_bar_comparison(all_results, available_props)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No property results available for comparison.")


if not getattr(st, "_is_test_mock", False):
    _render()
