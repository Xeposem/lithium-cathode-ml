"""Cathode ML Dashboard - Streamlit entrypoint.

Multi-page Streamlit application for exploring lithium cathode
ML model results, data, and predictions.

Launch: ``streamlit run dashboard/app.py``
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Suppress cosmetic warnings that appear during normal dashboard use.
# pymatgen: noble gases (Ar, He, Ne) have no Pauling electronegativity.
warnings.filterwarnings(
    "ignore",
    message=".*No Pauling electronegativity.*",
    category=UserWarning,
)
# matminer: MagpieData impute_nan default change notice.
warnings.filterwarnings(
    "ignore",
    message=".*impute_nan.*",
    category=UserWarning,
)
# matminer: general featurizer warnings.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="matminer",
)
# pymatgen: general structure/composition warnings.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymatgen",
)

# Ensure the project root is on sys.path so that `from dashboard.utils...`
# imports work regardless of the working directory Streamlit uses.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
