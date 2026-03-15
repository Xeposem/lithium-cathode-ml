"""Crystal Structure Viewer page.

Interactive 3D crystal structure visualization from uploaded CIF files
or composition formula lookup, using py3Dmol and stmol for rendering.
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


def _render_structure(structure, cif_content: str) -> None:
    """Render structure info and 3D viewer for a pymatgen Structure."""
    # Structure information
    with st.expander("Structure Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Formula:** {structure.composition.reduced_formula}")
            st.write(f"**Num atoms:** {len(structure)}")
            st.write(f"**Num elements:** {len(structure.composition.elements)}")
        with col2:
            lattice = structure.lattice
            st.write(f"**a:** {lattice.a:.4f} A")
            st.write(f"**b:** {lattice.b:.4f} A")
            st.write(f"**c:** {lattice.c:.4f} A")
        with col3:
            st.write(f"**alpha:** {lattice.alpha:.2f} deg")
            st.write(f"**beta:** {lattice.beta:.2f} deg")
            st.write(f"**gamma:** {lattice.gamma:.2f} deg")

    # 3D viewer
    st.subheader("3D Crystal Structure")

    try:
        import py3Dmol
        from stmol import showmol

        viewer = py3Dmol.view(width=700, height=500)
        viewer.addModel(cif_content, "cif")
        viewer.setStyle(
            {"sphere": {"radius": 0.4}, "stick": {"radius": 0.15}}
        )
        viewer.zoomTo()
        showmol(viewer, height=500, width=700)
    except ImportError:
        st.error(
            "3D viewer requires py3Dmol and stmol packages. "
            "Install with: pip install py3Dmol stmol"
        )
    except Exception as exc:
        st.error(f"Could not render 3D structure: {exc}")

    # Download button
    st.download_button(
        label="Download CIF",
        data=cif_content,
        file_name=f"{structure.composition.reduced_formula}.cif",
        mime="text/plain",
    )


def main() -> None:
    """Main entry point for the Crystal Viewer page."""
    st.title("Crystal Structure Viewer")
    st.markdown(
        "Enter a composition formula to look up the most stable known structure, "
        "or upload a CIF file to visualize in 3D."
    )

    tab_formula, tab_cif = st.tabs(["Composition Lookup", "CIF Upload"])

    # ----- Tab 1: Formula-based structure lookup -----
    with tab_formula:
        # Pre-fill from session state (e.g. navigated from predict page)
        default_formula = st.session_state.pop("crystal_formula", "")

        formula = st.text_input(
            "Composition formula",
            value=default_formula,
            placeholder="e.g. LiFePO4",
            help="Look up the most stable known crystal structure for this composition.",
        )
        view_btn = st.button("View Structure", key="view_formula")

        # Auto-show if pre-filled from predict page, or on button click
        if (default_formula and formula) or (view_btn and formula):
            with st.spinner("Looking up structure..."):
                try:
                    from dashboard.utils.model_loader import lookup_structure

                    match = lookup_structure(formula)
                    if match is None:
                        st.warning(
                            f"No known crystal structure found for '{formula}' "
                            "in the materials database."
                        )
                    else:
                        structure, record = match
                        source = record.get("source", "unknown")
                        mat_id = record.get("material_id", "?")
                        e_hull = record.get("energy_above_hull")
                        sg = record.get("space_group")
                        info_parts = [f"**{mat_id}** ({source})"]
                        if sg is not None:
                            info_parts.append(f"SG {sg}")
                        if e_hull is not None:
                            info_parts.append(f"E_hull={e_hull:.4f} eV/atom")
                        st.caption(
                            "Matched structure: " + " | ".join(info_parts)
                        )

                        from pymatgen.io.cif import CifWriter

                        cif_content = str(CifWriter(structure))
                        _render_structure(structure, cif_content)
                except Exception as exc:
                    st.error(f"Structure lookup failed: {exc}")
        elif view_btn and not formula:
            st.warning("Please enter a composition formula.")

    # ----- Tab 2: CIF file upload -----
    with tab_cif:
        uploaded_file = st.file_uploader(
            "Upload CIF file",
            type=["cif"],
            help="Upload a Crystallographic Information File (.cif) to visualize.",
        )

        if uploaded_file is None:
            st.info(
                "Upload a CIF file to visualize its crystal structure in 3D."
            )
            return

        try:
            cif_bytes = uploaded_file.read()
            cif_content = cif_bytes.decode("utf-8")
        except UnicodeDecodeError:
            st.error("Could not decode CIF file. Ensure it is UTF-8 encoded.")
            return

        try:
            from pymatgen.core import Structure

            structure = Structure.from_str(cif_content, fmt="cif")
        except Exception as exc:
            st.error(f"Invalid CIF file: {exc}")
            return

        _render_structure(structure, cif_content)


main()
