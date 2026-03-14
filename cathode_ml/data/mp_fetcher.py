"""Materials Project data fetcher with electrode data joining (DATA-01).

Fetches lithium cathode materials from the Materials Project API,
joins insertion electrode data (voltage, capacity) onto material
summaries, and caches results to avoid repeated API calls.
"""

import logging
import os
from dataclasses import asdict
from typing import List

from cathode_ml.data.cache import DataCache
from cathode_ml.data.schemas import MaterialRecord


def _get_mprester():
    """Lazy import of MPRester to avoid import-time dependency issues."""
    from mp_api.client import MPRester
    return MPRester


class MPFetcher:
    """Fetches cathode material data from the Materials Project API.

    Uses MPRester to query:
    1. Material summaries (structure, formation energy, stability)
    2. Insertion electrode data (voltage, capacity)

    Joins electrode data onto summaries by material_id and returns
    a list of MaterialRecord objects. Results are cached to avoid
    repeated API calls.

    Usage:
        config = load_config()
        cache = DataCache("data/raw")
        fetcher = MPFetcher(config, cache)
        records = fetcher.fetch()
    """

    def __init__(self, config: dict, cache: DataCache) -> None:
        """Initialize with config and cache.

        Args:
            config: Full pipeline config dict (must contain
                    data_sources.materials_project section).
            cache: DataCache instance for caching API responses.
        """
        self.config = config["data_sources"]["materials_project"]
        self.cache = cache
        self.logger = logging.getLogger("cathode_ml.data.mp_fetcher")

    def _cache_key(self) -> str:
        """Build deterministic cache key from config params."""
        return self.cache.cache_key("mp", {
            "elements": self.config["elements_must_contain"],
            "energy_above_hull_max": self.config["energy_above_hull_max"],
        })

    def _deserialize_records(self, data: dict) -> List[MaterialRecord]:
        """Convert cached dict data back into MaterialRecord list."""
        return [MaterialRecord(**rec) for rec in data["records"]]

    def _serialize_records(self, records: List[MaterialRecord]) -> dict:
        """Convert MaterialRecord list to serializable dict."""
        return {"records": [asdict(r) for r in records]}

    def _fetch_electrodes_raw(self, api_key: str) -> List[dict]:
        """Fetch electrode data via raw REST API as a fallback.

        The typed ``InsertionElectrodeDoc`` search can fail with ``'@class'``
        deserialization errors.  This method pages through the REST endpoint
        directly and returns plain dicts.
        """
        import requests

        base_url = "https://api.materialsproject.org/materials/insertion_electrodes/"
        headers = {"X-API-KEY": api_key}
        fields = "battery_id,material_ids,average_voltage,capacity_grav,working_ion"
        all_docs = []
        skip = 0
        limit = 1000

        while True:
            resp = requests.get(
                base_url,
                params={"_fields": fields, "_skip": skip, "_limit": limit},
                headers=headers,
                timeout=120,
            )
            if not resp.ok:
                self.logger.warning("Raw electrode API returned %d", resp.status_code)
                break
            body = resp.json()
            data = body.get("data", [])
            if not data:
                break
            all_docs.extend(data)
            skip += limit
            self.logger.info("  Fetched %d electrode docs so far...", len(all_docs))
            # Check if we've gotten all of them
            total = body.get("meta", {}).get("total_doc", None)
            if total is not None and len(all_docs) >= total:
                break

        self.logger.info("Raw REST: fetched %d electrode docs total", len(all_docs))
        return all_docs

    def fetch(self, force_refresh: bool = False) -> List[MaterialRecord]:
        """Fetch lithium cathode materials from Materials Project.

        Checks cache first unless force_refresh is True. On cache miss,
        queries the MP API for material summaries and insertion electrode
        data, joins them by material_id, and caches the result.

        Args:
            force_refresh: If True, bypass cache and re-fetch from API.

        Returns:
            List of MaterialRecord with MP data and electrode properties.
        """
        cache_key = self._cache_key()

        # Check cache
        if not force_refresh and self.cache.has(cache_key):
            self.logger.info("Loading MP data from cache")
            data = self.cache.load(cache_key)
            return self._deserialize_records(data)

        # Fetch from API
        self.logger.info("Fetching materials from Materials Project API...")
        api_key = os.environ.get("MP_API_KEY", "")

        MPRester = _get_mprester()
        with MPRester(api_key) as mpr:
            # Fetch material summaries
            summary_docs = mpr.materials.summary.search(
                elements=self.config["elements_must_contain"],
                energy_above_hull=(0, self.config["energy_above_hull_max"]),
                fields=self.config["fields"],
            )
            self.logger.info("Fetched %d material summaries", len(summary_docs))

            # Fetch insertion electrode data.
            # The typed InsertionElectrodeDoc search can fail with '@class'
            # deserialization errors in some mp_api versions.  Fall back to
            # the raw REST API which returns plain dicts.
            electrode_docs = []
            try:
                electrode_docs = mpr.insertion_electrodes.search(
                    fields=[
                        "battery_id", "material_ids", "average_voltage",
                        "capacity_grav", "working_ion", "framework_formula",
                    ],
                )
                self.logger.info("Fetched %d electrode docs", len(electrode_docs))
            except Exception as e:
                self.logger.warning(
                    "Typed electrode fetch failed (%s), trying raw REST API...", e
                )
                electrode_docs = self._fetch_electrodes_raw(api_key)

        # Filter Li-ion electrodes and build lookup map
        electrode_map = {}
        for edoc in electrode_docs:
            try:
                if isinstance(edoc, dict):
                    working_ion = edoc.get("working_ion")
                    material_ids = edoc.get("material_ids", []) or []
                    voltage = edoc.get("average_voltage")
                    capacity = edoc.get("capacity_grav")
                else:
                    working_ion = getattr(edoc, "working_ion", None)
                    material_ids = getattr(edoc, "material_ids", []) or []
                    voltage = getattr(edoc, "average_voltage", None)
                    capacity = getattr(edoc, "capacity_grav", None)
                if working_ion != "Li":
                    continue
                for mid in material_ids:
                    electrode_map[mid] = {
                        "voltage": voltage,
                        "capacity": capacity,
                    }
            except Exception:
                continue

        self.logger.info(
            "Built electrode map: %d materials with Li-ion electrode data",
            len(electrode_map),
        )

        # Convert to MaterialRecord list
        records = []
        for doc in summary_docs:
            # Extract space group safely
            space_group = None
            if hasattr(doc, "symmetry") and doc.symmetry is not None:
                space_group = getattr(doc.symmetry, "number", None)

            # Lookup electrode data
            elec = electrode_map.get(doc.material_id, {})

            record = MaterialRecord(
                material_id=doc.material_id,
                formula=doc.formula_pretty,
                structure_dict=doc.structure.as_dict(),
                source="materials_project",
                formation_energy_per_atom=doc.formation_energy_per_atom,
                energy_above_hull=doc.energy_above_hull,
                voltage=elec.get("voltage"),
                capacity=elec.get("capacity"),
                is_stable=doc.is_stable,
                space_group=space_group,
            )
            records.append(record)

        electrode_count = sum(1 for r in records if r.voltage is not None)
        self.logger.info(
            "Created %d MaterialRecords (%d with electrode data)",
            len(records), electrode_count,
        )

        # Cache results
        self.cache.save(
            cache_key,
            self._serialize_records(records),
            metadata={"source": "materials_project", "count": len(records)},
        )
        self.logger.info("Cached MP data with key: %s", cache_key)

        return records
