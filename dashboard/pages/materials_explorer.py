"""Materials Explorer page -- filterable table and discovery panel.

Provides interactive filtering by property ranges, elements, stability,
and source, plus a discovery panel ranking top candidate materials
(DASH-05, DASH-06).
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import pandas as pd
import streamlit as st

from dashboard.utils.data_loader import get_cached_records

# Columns to display in the materials table
_DISPLAY_COLS = [
    "material_id",
    "formula",
    "source",
    "voltage",
    "capacity",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
]

_RANKING_OPTIONS = {
    "voltage": ("Voltage (V)", False),  # descending
    "capacity": ("Capacity (mAh/g)", False),  # descending
    "formation_energy_per_atom": ("Formation Energy (eV/atom)", True),  # ascending (lower = more stable)
}


# ---------------------------------------------------------------------------
# Helper functions (importable for testing)
# ---------------------------------------------------------------------------


def _extract_elements(formula: str) -> set[str]:
    """Extract unique element symbols from a chemical formula.

    Args:
        formula: Chemical formula string, e.g. "LiFePO4".

    Returns:
        Set of element symbols, e.g. {"Li", "Fe", "P", "O"}.
    """
    return set(re.findall(r"[A-Z][a-z]?", formula))


def filter_materials(
    df: pd.DataFrame,
    voltage_range: Optional[tuple[float, float]] = None,
    energy_range: Optional[tuple[float, float]] = None,
    capacity_range: Optional[tuple[float, float]] = None,
    hull_range: Optional[tuple[float, float]] = None,
    elements: Optional[Sequence[str]] = None,
    stable_only: bool = False,
    sources: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Apply filters to materials DataFrame.

    All filters are optional; omitting a filter means no restriction
    on that dimension.

    Args:
        df: Full materials DataFrame.
        voltage_range: (min, max) voltage filter.
        energy_range: (min, max) formation energy filter.
        capacity_range: (min, max) capacity filter.
        hull_range: (min, max) energy above hull filter.
        elements: Required elements (all must be present).
        stable_only: If True, keep only stable materials (is_stable == True).
        sources: Allowed data sources.

    Returns:
        Filtered DataFrame.
    """
    mask = pd.Series(True, index=df.index)

    if voltage_range is not None and "voltage" in df.columns:
        vmin, vmax = voltage_range
        mask &= df["voltage"].between(vmin, vmax) | df["voltage"].isna()

    if energy_range is not None and "formation_energy_per_atom" in df.columns:
        emin, emax = energy_range
        mask &= df["formation_energy_per_atom"].between(emin, emax) | df["formation_energy_per_atom"].isna()

    if capacity_range is not None and "capacity" in df.columns:
        cmin, cmax = capacity_range
        mask &= df["capacity"].between(cmin, cmax) | df["capacity"].isna()

    if hull_range is not None and "energy_above_hull" in df.columns:
        hmin, hmax = hull_range
        mask &= df["energy_above_hull"].between(hmin, hmax) | df["energy_above_hull"].isna()

    if elements and "formula" in df.columns:
        required = set(elements)
        mask &= df["formula"].apply(lambda f: required.issubset(_extract_elements(f)))

    if stable_only and "is_stable" in df.columns:
        mask &= df["is_stable"] == True  # noqa: E712

    if sources and "source" in df.columns:
        mask &= df["source"].isin(sources)

    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def _render() -> None:
    """Render the Materials Explorer page."""
    st.title("Materials Explorer")

    records = get_cached_records()
    if not records:
        st.warning("No cached data found. Run the data pipeline first.")
        return

    df = pd.DataFrame(records)

    # ---- Filter controls ----
    with st.expander("Filters", expanded=True):
        col1, col2 = st.columns(2)

        voltage_range = None
        if "voltage" in df.columns and df["voltage"].notna().any():
            vmin, vmax = float(df["voltage"].min()), float(df["voltage"].max())
            with col1:
                voltage_range = st.slider(
                    "Voltage (V)", vmin, vmax, (vmin, vmax), key="filt_voltage"
                )

        energy_range = None
        if "formation_energy_per_atom" in df.columns and df["formation_energy_per_atom"].notna().any():
            emin, emax = float(df["formation_energy_per_atom"].min()), float(df["formation_energy_per_atom"].max())
            with col2:
                energy_range = st.slider(
                    "Formation Energy (eV/atom)", emin, emax, (emin, emax), key="filt_energy"
                )

        col3, col4 = st.columns(2)

        capacity_range = None
        if "capacity" in df.columns and df["capacity"].notna().any():
            cmin, cmax = float(df["capacity"].min()), float(df["capacity"].max())
            with col3:
                capacity_range = st.slider(
                    "Capacity (mAh/g)", cmin, cmax, (cmin, cmax), key="filt_capacity"
                )

        hull_range = None
        if "energy_above_hull" in df.columns and df["energy_above_hull"].notna().any():
            hmin, hmax = float(df["energy_above_hull"].min()), float(df["energy_above_hull"].max())
            with col4:
                hull_range = st.slider(
                    "Energy Above Hull (eV/atom)", hmin, hmax, (hmin, hmax), key="filt_hull"
                )

        col5, col6 = st.columns(2)

        elements = None
        if "formula" in df.columns:
            all_elements = sorted(
                set().union(*(
                    _extract_elements(f) for f in df["formula"].dropna()
                ))
            )
            with col5:
                elements = st.multiselect(
                    "Must contain elements", all_elements, key="filt_elements"
                )

        sources = None
        if "source" in df.columns:
            available_sources = sorted(df["source"].dropna().unique())
            with col6:
                sources = st.multiselect(
                    "Data source", available_sources, key="filt_sources"
                )

        stable_only = st.checkbox("Stable only (E_hull = 0)", key="filt_stable")

    # ---- Apply filters ----
    filtered = filter_materials(
        df,
        voltage_range=voltage_range,
        energy_range=energy_range,
        capacity_range=capacity_range,
        hull_range=hull_range,
        elements=elements if elements else None,
        stable_only=stable_only,
        sources=sources if sources else None,
    )

    st.caption(f"Showing {len(filtered)} of {len(df)} materials")

    # ---- Filterable table ----
    display_cols = [c for c in _DISPLAY_COLS if c in filtered.columns]
    st.dataframe(filtered[display_cols], width="stretch")

    # ---- Discovery panel ----
    st.subheader("Top Candidates")

    rank_col = st.selectbox(
        "Rank by",
        options=list(_RANKING_OPTIONS.keys()),
        format_func=lambda k: _RANKING_OPTIONS[k][0],
        key="rank_by",
    )
    top_n = st.slider("Show top N", 5, 50, 10, key="top_n")

    label, ascending = _RANKING_OPTIONS[rank_col]

    # Only rank materials that have a value for the ranking column
    rankable = filtered.dropna(subset=[rank_col])
    ranked = rankable.sort_values(rank_col, ascending=ascending).head(top_n)

    st.dataframe(ranked[display_cols], width="stretch")

    direction = "lower" if ascending else "higher"
    st.info(f"Materials ranked by {label} ({direction} is better). Showing top {len(ranked)} of {len(rankable)} with data.")


if not getattr(st, "_is_test_mock", False):
    _render()
