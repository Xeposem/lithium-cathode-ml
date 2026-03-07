"""Cathode ML Dashboard - Streamlit entrypoint.

Multi-page Streamlit application for exploring lithium cathode
ML model results, data, and predictions.

Launch: ``streamlit run dashboard/app.py``
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Cathode ML Dashboard",
    page_icon=":material/battery_charging_full:",
    layout="wide",
)

# Define pages grouped by section
results_pages = [
    st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:"),
    st.Page("pages/model_comparison.py", title="Model Comparison", icon=":material/compare:"),
]

explore_pages = [
    st.Page("pages/data_explorer.py", title="Data Explorer", icon=":material/search:"),
    st.Page("pages/materials_explorer.py", title="Materials Explorer", icon=":material/science:"),
]

tools_pages = [
    st.Page("pages/predict.py", title="Predict", icon=":material/calculate:"),
    st.Page("pages/crystal_viewer.py", title="Crystal Viewer", icon=":material/view_in_ar:"),
]

pg = st.navigation({
    "Results": results_pages,
    "Explore": explore_pages,
    "Tools": tools_pages,
})

pg.run()
