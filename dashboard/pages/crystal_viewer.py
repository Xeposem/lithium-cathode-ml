"""Crystal Structure Viewer page.

Interactive 3D crystal structure visualization from uploaded CIF files
using py3Dmol and stmol for rendering.
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the Crystal Viewer page."""
    st.title("Crystal Structure Viewer")
    st.markdown(
        "Upload a CIF file to visualize its crystal structure in 3D. "
        "The viewer supports rotation, zoom, and element-colored atoms."
    )

    uploaded_file = st.file_uploader(
        "Upload CIF file",
        type=["cif"],
        help="Upload a Crystallographic Information File (.cif) to visualize.",
    )

    if uploaded_file is None:
        st.info(
            "Upload a CIF file to visualize its crystal structure in 3D. "
            "Supported format: .cif (Crystallographic Information File)."
        )
        return

    # Decode CIF content
    try:
        cif_bytes = uploaded_file.read()
        cif_content = cif_bytes.decode("utf-8")
    except UnicodeDecodeError:
        st.error("Could not decode CIF file. Ensure it is UTF-8 encoded.")
        return

    # Parse structure with pymatgen
    try:
        from pymatgen.core import Structure

        structure = Structure.from_str(cif_content, fmt="cif")
    except Exception as exc:
        st.error(f"Invalid CIF file: {exc}")
        return

    # Structure information in sidebar/expander
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


if __name__ == "__main__":
    main()
