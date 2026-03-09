"""OQMD data fetcher with HTTP fallback (DATA-02).

Fetches lithium-containing materials from the Open Quantum Materials
Database. Tries qmpy_rester first, falls back to direct HTTP if
qmpy_rester fails (it is unmaintained since 2019).
"""

import logging
from dataclasses import asdict
from typing import List

import requests

from cathode_ml.data.cache import DataCache
from cathode_ml.data.schemas import MaterialRecord


def _get_qmpy_rester():
    """Lazy import of qmpy_rester to avoid import-time issues."""
    import qmpy_rester as qr
    return qr


class OQMDFetcher:
    """Fetches material data from the Open Quantum Materials Database.

    Tries qmpy_rester Python client first. If that fails (package
    issues, connection errors), falls back to direct HTTP requests
    to the OQMD REST API with pagination.

    Usage:
        config = load_config()
        cache = DataCache("data/raw")
        fetcher = OQMDFetcher(config, cache)
        records = fetcher.fetch()
    """

    OQMD_API_URL = "http://oqmd.org/oqmdapi/formationenergy"

    def __init__(self, config: dict, cache: DataCache) -> None:
        """Initialize with config and cache.

        Args:
            config: Full pipeline config dict (must contain
                    data_sources.oqmd section).
            cache: DataCache instance for caching API responses.
        """
        self.config = config["data_sources"]["oqmd"]
        self.cache = cache
        self.logger = logging.getLogger("cathode_ml.data.oqmd_fetcher")

    def _cache_key(self) -> str:
        """Build deterministic cache key from config params."""
        return self.cache.cache_key("oqmd", {
            "element_set": self.config["element_set"],
            "stability_max": self.config["stability_max"],
        })

    def _deserialize_records(self, data: dict) -> List[MaterialRecord]:
        """Convert cached dict data back into MaterialRecord list."""
        return [MaterialRecord(**rec) for rec in data["records"]]

    def _serialize_records(self, records: List[MaterialRecord]) -> dict:
        """Convert MaterialRecord list to serializable dict."""
        return {"records": [asdict(r) for r in records]}

    @staticmethod
    def _parse_site(site_str: str):
        """Parse OQMD site string like 'Li @ 0.5 0.5 0.5' into (species, coords).

        Returns:
            Tuple of (species_str, [x, y, z]) or None if unparseable.
        """
        parts = site_str.split("@")
        if len(parts) != 2:
            return None
        species = parts[0].strip()
        try:
            coords = [float(x) for x in parts[1].strip().split()]
        except ValueError:
            return None
        if len(coords) != 3:
            return None
        return species, coords

    def _build_structure_dict(self, entry: dict) -> dict:
        """Build a pymatgen-compatible structure dict from OQMD unit_cell + sites.

        Args:
            entry: Single OQMD API entry with 'unit_cell' and 'sites' fields.

        Returns:
            Dict compatible with pymatgen Structure.from_dict(), or empty dict
            if the entry lacks sufficient structural data.
        """
        unit_cell = entry.get("unit_cell")
        sites_raw = entry.get("sites")
        if not unit_cell or not sites_raw:
            return {}

        parsed = [self._parse_site(s) for s in sites_raw]
        if not all(parsed):
            return {}

        species = [p[0] for p in parsed]
        frac_coords = [p[1] for p in parsed]

        try:
            from pymatgen.core import Lattice, Structure
            lattice = Lattice(unit_cell)
            structure = Structure(lattice, species, frac_coords)
            return structure.as_dict()
        except Exception:
            return {}

    def _entry_to_record(self, entry: dict) -> MaterialRecord:
        """Convert a single OQMD entry to MaterialRecord.

        Args:
            entry: Dict from OQMD API response.

        Returns:
            MaterialRecord with OQMD data.
        """
        entry_id = entry.get("entry_id", "")
        material_id = f"oqmd-{entry_id}" if entry_id else f"oqmd-{entry.get('name', 'unknown')}"
        formula = entry.get("name", entry.get("composition", "unknown"))
        space_group = entry.get("spacegroup", None)

        structure_dict = self._build_structure_dict(entry)

        return MaterialRecord(
            material_id=material_id,
            formula=formula,
            structure_dict=structure_dict,
            source="oqmd",
            formation_energy_per_atom=entry.get("delta_e"),
            energy_above_hull=entry.get("stability"),
            voltage=None,
            capacity=None,
            is_stable=None,
            space_group=space_group,
        )

    def _fetch_via_qmpy(self) -> List[dict]:
        """Fetch data using qmpy_rester Python client.

        Returns:
            List of entry dicts from OQMD.

        Raises:
            Exception: If qmpy_rester fails for any reason.
        """
        qr = _get_qmpy_rester()
        with qr.QMPYRester() as q:
            data = q.get_oqmd_phases(
                element_set=self.config["element_set"],
                stability=f"<{self.config['stability_max']}",
                verbose=False,
            )
        return data.get("data", [])

    def _fetch_via_http(self) -> List[dict]:
        """Fetch data via direct HTTP to OQMD REST API with pagination.

        Returns:
            List of entry dicts from OQMD.
        """
        results = []
        data_available = None
        filter_expr = (
            f"element_set={self.config['element_set']}"
            f" AND stability<{self.config['stability_max']}"
        )
        params = {
            "filter": filter_expr,
            "limit": 100,
            "offset": 0,
        }

        while True:
            for attempt in range(3):
                resp = requests.get(self.OQMD_API_URL, params=params)
                if resp.status_code == 502 and attempt < 2:
                    import time
                    wait = 5 * (attempt + 1)
                    self.logger.warning(
                        "502 Bad Gateway, retrying in %ds (attempt %d/3)",
                        wait, attempt + 1,
                    )
                    time.sleep(wait)
                    continue
                break
            resp.raise_for_status()
            page = resp.json()
            results.extend(page.get("data", []))

            if data_available is None:
                data_available = page.get("meta", {}).get("data_available", "?")
                self.logger.info("OQMD reports %s total records", data_available)

            self.logger.info(
                "OQMD progress: %d / %s fetched", len(results), data_available
            )

            # Check for next page
            next_url = page.get("next")
            if not next_url:
                links = page.get("links", {})
                next_url = links.get("next") if isinstance(links, dict) else None

            if not next_url:
                break

            params["offset"] += params["limit"]

        return results

    def fetch(self, force_refresh: bool = False) -> List[MaterialRecord]:
        """Fetch lithium-containing materials from OQMD.

        Checks cache first unless force_refresh is True. Tries
        qmpy_rester, falls back to HTTP on any failure.

        Args:
            force_refresh: If True, bypass cache and re-fetch.

        Returns:
            List of MaterialRecord with OQMD data.
        """
        cache_key = self._cache_key()

        # Check cache
        if not force_refresh and self.cache.has(cache_key):
            self.logger.info("Loading OQMD data from cache")
            data = self.cache.load(cache_key)
            return self._deserialize_records(data)

        # Try qmpy_rester first, fall back to HTTP
        entries = []
        try:
            self.logger.info("Fetching OQMD data via qmpy_rester...")
            entries = self._fetch_via_qmpy()
            self.logger.info(
                "qmpy_rester success: %d entries", len(entries)
            )
        except Exception as e:
            self.logger.warning(
                "qmpy_rester failed (%s), falling back to HTTP", e
            )
            entries = self._fetch_via_http()
            self.logger.info(
                "HTTP fallback success: %d entries", len(entries)
            )

        # Convert to MaterialRecord
        records = [self._entry_to_record(entry) for entry in entries]
        self.logger.info("Created %d OQMD MaterialRecords", len(records))

        # Cache results
        self.cache.save(
            cache_key,
            self._serialize_records(records),
            metadata={"source": "oqmd", "count": len(records)},
        )
        self.logger.info("Cached OQMD data with key: %s", cache_key)

        return records
