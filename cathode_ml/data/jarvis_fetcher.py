"""JARVIS-DFT data fetcher for lithium-containing materials (DATA-04).

Pulls the ``dft_3d`` dataset from JARVIS-Tools, filters to lithium
compounds, and converts to MaterialRecord objects.
"""

import logging
from dataclasses import asdict
from typing import List

from cathode_ml.data.cache import DataCache
from cathode_ml.data.schemas import MaterialRecord


def _get_jarvis_data():
    """Lazy import of jarvis data loading function."""
    from jarvis.db.figshare import data as jarvis_data
    return jarvis_data


class JARVISFetcher:
    """Fetches lithium-containing materials from the JARVIS-DFT database.

    Downloads the ``dft_3d`` dataset via jarvis-tools, filters to entries
    containing lithium, and converts to MaterialRecord objects.

    Note: JARVIS uses the OptB88vdW functional (different from PBE used
    by Materials Project and OQMD). Formation energies are not directly
    comparable across databases without normalization.

    Usage:
        config = load_config()
        cache = DataCache("data/raw")
        fetcher = JARVISFetcher(config, cache)
        records = fetcher.fetch()
    """

    def __init__(self, config: dict, cache: DataCache) -> None:
        self.config = config["data_sources"]["jarvis"]
        self.cache = cache
        self.logger = logging.getLogger("cathode_ml.data.jarvis_fetcher")

    def _cache_key(self) -> str:
        return self.cache.cache_key("jarvis", {
            "dataset": self.config.get("dataset", "dft_3d"),
            "element": self.config.get("element", "Li"),
        })

    def _deserialize_records(self, data: dict) -> List[MaterialRecord]:
        return [MaterialRecord(**rec) for rec in data["records"]]

    def _serialize_records(self, records: List[MaterialRecord]) -> dict:
        return {"records": [asdict(r) for r in records]}

    def _build_structure_dict(self, atoms_dict: dict) -> dict:
        """Convert JARVIS Atoms dict to pymatgen-compatible structure dict."""
        try:
            from jarvis.core.atoms import Atoms as JarvisAtoms
            from pymatgen.io.jarvis import JarvisAtomsAdaptor

            jarvis_atoms = JarvisAtoms.from_dict(atoms_dict)
            structure = JarvisAtomsAdaptor.get_structure(jarvis_atoms)
            return structure.as_dict()
        except Exception:
            pass

        # Fallback: manual conversion
        try:
            from pymatgen.core import Lattice, Structure

            lattice_mat = atoms_dict.get("lattice_mat")
            coords = atoms_dict.get("coords")
            elements = atoms_dict.get("elements")
            cartesian = atoms_dict.get("cartesian", True)

            if not lattice_mat or not coords or not elements:
                return {}

            lattice = Lattice(lattice_mat)
            structure = Structure(
                lattice, elements, coords,
                coords_are_cartesian=cartesian,
            )
            return structure.as_dict()
        except Exception:
            return {}

    def _entry_to_record(self, entry: dict) -> MaterialRecord:
        """Convert a single JARVIS entry to MaterialRecord."""
        jid = entry.get("jid", "unknown")
        material_id = f"jarvis-{jid}"
        formula = entry.get("formula", entry.get("composition", "unknown"))

        formation_energy = entry.get("formation_energy_peratom")
        if formation_energy is not None:
            try:
                formation_energy = float(formation_energy)
            except (ValueError, TypeError):
                formation_energy = None

        ehull = entry.get("ehull")
        if ehull is not None:
            try:
                ehull = float(ehull)
            except (ValueError, TypeError):
                ehull = None

        sg = entry.get("spacegroup_number")
        if sg is not None:
            try:
                sg = int(sg)
            except (ValueError, TypeError):
                sg = None

        atoms_dict = entry.get("atoms", {})
        structure_dict = self._build_structure_dict(atoms_dict) if atoms_dict else {}

        return MaterialRecord(
            material_id=material_id,
            formula=formula,
            structure_dict=structure_dict,
            source="jarvis",
            formation_energy_per_atom=formation_energy,
            energy_above_hull=ehull,
            voltage=None,
            capacity=None,
            is_stable=None,
            space_group=sg,
        )

    def fetch(self, force_refresh: bool = False) -> List[MaterialRecord]:
        """Fetch lithium-containing materials from JARVIS-DFT.

        Args:
            force_refresh: If True, bypass cache and re-fetch.

        Returns:
            List of MaterialRecord with JARVIS data.
        """
        cache_key = self._cache_key()

        if not force_refresh and self.cache.has(cache_key):
            self.logger.info("Loading JARVIS data from cache")
            data = self.cache.load(cache_key)
            return self._deserialize_records(data)

        self.logger.info("Fetching JARVIS dft_3d dataset...")
        jarvis_data = _get_jarvis_data()

        dataset_name = self.config.get("dataset", "dft_3d")
        element = self.config.get("element", "Li")

        try:
            dataset = jarvis_data(dataset_name)
        except Exception as e:
            self.logger.error("Failed to load JARVIS %s: %s", dataset_name, e)
            return []

        self.logger.info("JARVIS %s: %d total entries, filtering for %s...",
                         dataset_name, len(dataset), element)

        records = []
        for entry in dataset:
            # Filter: must contain target element
            formula = entry.get("formula", entry.get("composition", ""))
            elements = entry.get("atoms", {}).get("elements", [])
            if element not in formula and element not in elements:
                continue

            try:
                record = self._entry_to_record(entry)
                records.append(record)
            except Exception as e:
                self.logger.debug("Skipping JARVIS entry %s: %s",
                                  entry.get("jid", "?"), e)

        self.logger.info("Created %d JARVIS MaterialRecords", len(records))

        self.cache.save(
            cache_key,
            self._serialize_records(records),
            metadata={"source": "jarvis", "count": len(records)},
        )
        self.logger.info("Cached JARVIS data with key: %s", cache_key)

        return records
