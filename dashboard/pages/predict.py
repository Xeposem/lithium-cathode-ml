"""Predict Cathode Properties page.

Dual-mode prediction interface: composition string input for baseline models
and CIF file upload for all models (baselines + GNNs) with 3D structure preview.
"""

from __future__ import annotations

import logging
import traceback

import streamlit as st

logger = logging.getLogger(__name__)

# Property display configuration
PROPERTY_UNITS = {
    "voltage": "V",
    "capacity": "mAh/g",
    "formation_energy_per_atom": "eV/atom",
    "energy_above_hull": "eV/atom",
}

PROPERTY_LABELS = {
    "voltage": "Voltage",
    "capacity": "Capacity",
    "formation_energy_per_atom": "Formation Energy",
    "energy_above_hull": "Energy Above Hull",
}

MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "cgcnn": "CGCNN",
    "megnet": "MEGNet",
}


def _render_prediction_cards(results: dict) -> None:
    """Render prediction results as styled cards."""
    if not results:
        st.info("No predictions available. Ensure models have been trained.")
        return

    for prop, model_preds in results.items():
        label = PROPERTY_LABELS.get(prop, prop)
        unit = PROPERTY_UNITS.get(prop, "")

        with st.container(border=True):
            st.subheader(label)
            cols = st.columns(len(model_preds))
            for i, (model_name, value) in enumerate(model_preds.items()):
                with cols[i]:
                    display_name = MODEL_LABELS.get(model_name, model_name)
                    st.metric(
                        label=display_name,
                        value=f"{value:.4f} {unit}",
                    )


def _render_3d_preview(cif_content: str) -> None:
    """Render a 3D crystal structure preview from CIF content."""
    try:
        import py3Dmol
        from stmol import showmol

        viewer = py3Dmol.view(width=700, height=400)
        viewer.addModel(cif_content, "cif")
        viewer.setStyle(
            {"sphere": {"radius": 0.4}, "stick": {"radius": 0.15}}
        )
        viewer.zoomTo()
        showmol(viewer, height=400, width=700)
    except ImportError:
        st.warning("3D viewer requires py3Dmol and stmol packages.")
    except Exception as exc:
        st.warning(f"Could not render 3D preview: {exc}")


def main() -> None:
    """Main entry point for the Predict page."""
    st.title("Predict Cathode Properties")
    st.markdown(
        "Enter a composition formula or upload a CIF file to predict "
        "cathode material properties using trained ML models."
    )

    tab_comp, tab_cif = st.tabs(["Composition Input", "CIF Upload"])

    # ----- Tab 1: Composition-based prediction (baselines only) -----
    with tab_comp:
        col_input, col_results = st.columns([1, 2])

        with col_input:
            formula = st.text_input(
                "Composition formula",
                placeholder="e.g. LiFePO4",
                help="Enter a chemical formula to predict properties using baseline models.",
            )
            predict_btn = st.button("Predict", key="predict_composition")

        with col_results:
            if predict_btn and formula:
                with st.spinner("Running composition-based predictions..."):
                    try:
                        from dashboard.utils.model_loader import predict_from_composition

                        results = predict_from_composition(formula)
                        _render_prediction_cards(results)
                    except Exception:
                        st.error(
                            f"Prediction failed:\n```\n{traceback.format_exc()}\n```"
                        )
            elif predict_btn and not formula:
                st.warning("Please enter a composition formula.")

    # ----- Tab 2: CIF upload (all models) -----
    with tab_cif:
        uploaded_file = st.file_uploader(
            "Upload CIF file for predictions",
            type=["cif"],
            help="Upload a CIF file to predict properties using all available models.",
        )

        if uploaded_file is not None:
            try:
                cif_bytes = uploaded_file.read()
                cif_content = cif_bytes.decode("utf-8")
            except UnicodeDecodeError:
                st.error("Could not decode CIF file. Ensure it is UTF-8 encoded.")
                return

            # Parse structure
            try:
                from pymatgen.core import Structure

                structure = Structure.from_str(cif_content, fmt="cif")
                st.success(
                    f"Parsed structure: {structure.composition.reduced_formula}"
                )
            except Exception:
                st.error(f"Invalid CIF file:\n```\n{traceback.format_exc()}\n```")
                return

            # Inline 3D preview
            with st.expander("3D Structure Preview", expanded=True):
                _render_3d_preview(cif_content)

            # Structure info
            with st.expander("Structure Information"):
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.write(
                        f"**Formula:** {structure.composition.reduced_formula}"
                    )
                    st.write(f"**Num atoms:** {len(structure)}")
                with info_col2:
                    lattice = structure.lattice
                    st.write(f"**a:** {lattice.a:.3f} A")
                    st.write(f"**b:** {lattice.b:.3f} A")
                    st.write(f"**c:** {lattice.c:.3f} A")

            # Predictions
            st.subheader("Predictions")

            with st.spinner("Running predictions from all available models..."):
                all_results: dict = {}

                # Baseline predictions from composition
                try:
                    from dashboard.utils.model_loader import predict_from_composition

                    comp_results = predict_from_composition(
                        structure.composition.reduced_formula
                    )
                    for prop, preds in comp_results.items():
                        all_results.setdefault(prop, {}).update(preds)
                except Exception as exc:
                    logger.error("Baseline predictions failed: %s", exc)
                    st.warning(f"Baseline predictions failed: {exc}")

                # GNN predictions from structure
                try:
                    from dashboard.utils.model_loader import predict_from_structure

                    gnn_results = predict_from_structure(structure)
                    for prop, preds in gnn_results.items():
                        all_results.setdefault(prop, {}).update(preds)
                except Exception as exc:
                    logger.error("GNN predictions failed: %s", exc)
                    st.warning(f"GNN predictions failed: {exc}")

                _render_prediction_cards(all_results)


if __name__ == "__main__":
    main()
