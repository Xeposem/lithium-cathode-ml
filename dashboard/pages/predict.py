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
    "m3gnet": "M3GNet",
    "tensornet": "TensorNet",
}


def _render_prediction_cards(results: dict, best_models: dict | None = None) -> None:
    """Render prediction results as styled cards, highlighting the best model."""
    if not results:
        st.info("No predictions available. Ensure models have been trained.")
        return

    for prop, model_preds in results.items():
        label = PROPERTY_LABELS.get(prop, prop)
        unit = PROPERTY_UNITS.get(prop, "")
        best_model = best_models.get(prop) if best_models else None

        with st.container(border=True):
            st.subheader(label)
            cols = st.columns(len(model_preds))
            for i, (model_name, value) in enumerate(model_preds.items()):
                with cols[i]:
                    display_name = MODEL_LABELS.get(model_name, model_name)
                    is_best = model_name == best_model and model_name in model_preds
                    if is_best:
                        display_name = f":green[{display_name}  (best)]"
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


def _navigate_to_crystal() -> None:
    """Callback for the View Crystal Structure button."""
    formula = st.session_state.get("_predict_formula", "")
    if formula:
        st.session_state["crystal_formula"] = formula


def main() -> None:
    """Main entry point for the Predict page."""
    # Handle navigation after callback sets crystal_formula
    if "crystal_formula" in st.session_state and st.session_state.get("_navigate_crystal"):
        st.session_state.pop("_navigate_crystal", None)
        st.switch_page("pages/crystal_viewer.py")

    st.title("Predict Cathode Properties")
    st.markdown(
        "Enter a composition formula or upload a CIF file to predict "
        "cathode material properties using trained ML models."
    )

    tab_comp, tab_cif = st.tabs(["Composition Input", "CIF Upload"])

    # ----- Tab 1: Composition-based prediction (all models) -----
    with tab_comp:
        col_input, col_results = st.columns([1, 2])

        with col_input:
            formula = st.text_input(
                "Composition formula",
                placeholder="e.g. LiFePO4",
                help="Enter a chemical formula to predict properties using all available models.",
            )
            predict_btn = st.button("Predict", key="predict_composition")

        # Run predictions and store in session state
        if predict_btn and formula:
            with st.spinner("Running predictions..."):
                try:
                    from dashboard.utils.model_loader import get_best_models, predict_from_composition

                    best_models = get_best_models()
                    results, structure_info = predict_from_composition(formula)
                    st.session_state["_predict_results"] = results
                    st.session_state["_predict_structure_info"] = structure_info
                    st.session_state["_predict_best_models"] = best_models
                    st.session_state["_predict_formula"] = formula
                except Exception:
                    st.session_state.pop("_predict_results", None)
                    st.error(
                        f"Prediction failed:\n```\n{traceback.format_exc()}\n```"
                    )
        elif predict_btn and not formula:
            st.warning("Please enter a composition formula.")

        # Display stored results
        with col_results:
            results = st.session_state.get("_predict_results")
            if results is not None:
                structure_info = st.session_state.get("_predict_structure_info")
                best_models = st.session_state.get("_predict_best_models")
                if structure_info:
                    source = structure_info.get("source", "unknown")
                    mat_id = structure_info.get("material_id", "?")
                    e_hull = structure_info.get("energy_above_hull")
                    sg = structure_info.get("space_group")
                    info_parts = [f"**{mat_id}** ({source})"]
                    if sg is not None:
                        info_parts.append(f"SG {sg}")
                    if e_hull is not None:
                        info_parts.append(f"E_hull={e_hull:.4f} eV/atom")
                    st.caption(
                        "GNN predictions use matched structure: "
                        + " | ".join(info_parts)
                    )
                else:
                    st.caption(
                        "No known crystal structure found for this composition. "
                        "Showing baseline model predictions only."
                    )
                _render_prediction_cards(results, best_models)
                if structure_info:
                    def _on_view_crystal():
                        st.session_state["crystal_formula"] = st.session_state.get("_predict_formula", "")
                        st.session_state["_navigate_crystal"] = True

                    st.button(
                        "View Crystal Structure",
                        icon=":material/view_in_ar:",
                        key="view_crystal_link",
                        on_click=_on_view_crystal,
                    )

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
                from dashboard.utils.model_loader import get_best_models
                best_models = get_best_models()

                # Baseline predictions from composition
                try:
                    from dashboard.utils.model_loader import predict_from_composition

                    comp_results, _ = predict_from_composition(
                        structure.composition.reduced_formula
                    )
                    # Only take baseline results — GNN will use the uploaded structure
                    from dashboard.utils.model_loader import GNN_TYPES
                    for prop, preds in comp_results.items():
                        baseline_preds = {k: v for k, v in preds.items() if k not in GNN_TYPES}
                        if baseline_preds:
                            all_results.setdefault(prop, {}).update(baseline_preds)
                except Exception as exc:
                    logger.error("Baseline predictions failed: %s", exc)
                    st.warning(f"Baseline predictions failed: {exc}")

                # GNN predictions from the uploaded structure
                try:
                    from dashboard.utils.model_loader import predict_from_structure

                    gnn_results = predict_from_structure(structure)
                    for prop, preds in gnn_results.items():
                        all_results.setdefault(prop, {}).update(preds)
                except Exception as exc:
                    logger.error("GNN predictions failed: %s", exc)
                    st.warning(f"GNN predictions failed: {exc}")

                _render_prediction_cards(all_results, best_models)


main()
