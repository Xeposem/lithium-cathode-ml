"""AFLOW data fetcher for lithium-containing materials (DATA-03).

Queries the AFLOW database via the AFLUX REST API for lithium
compounds, fetches relaxed structures (CONTCAR), and maps results
to MaterialRecord objects.
"""

import logging
import time
from dataclasses import asdict
from typing import List

import requests

from cathode_ml.data.cache import DataCache
from cathode_ml.data.schemas import MaterialRecord


# AFLUX API endpoint
AFLUX_URL = "http://aflowlib.org/API/aflux/"

# Page size for AFLUX queries (max 64 per AFLOW docs)
_PAGE_SIZE = 64


class AFLOWFetcher:
    """Fetches lithium-containing materials from the AFLOW database.

    Uses the AFLUX REST API to query for Li-containing entries,
    fetches relaxed CONTCAR structures, and converts to
    MaterialRecord objects.

    Usage:
        config = load_config()
        cache = DataCache("data/raw")
        fetcher = AFLOWFetcher(config, cache)
        records = fetcher.fetch()
    """

    def __init__(self, config: dict, cache: DataCache) -> None:
        self.config = config["data_sources"]["aflow"]
        self.cache = cache
        self.logger = logging.getLogger("cathode_ml.data.aflow_fetcher")

    def _cache_key(self) -> str:
        return self.cache.cache_key("aflow", {
            "element": self.config.get("element", "Li"),
            "max_entries": self.config.get("max_entries", 5000),
        })

    def _deserialize_records(self, data: dict) -> List[MaterialRecord]:
        return [MaterialRecord(**rec) for rec in data["records"]]

    def _serialize_records(self, records: List[MaterialRecord]) -> dict:
        return {"records": [asdict(r) for r in records]}

    def _fetch_contcar(self, aurl: str, session: requests.Session) -> str:
        """Fetch the relaxed CONTCAR (POSCAR format) for an AFLOW entry.

        Args:
            aurl: AFLOW URL like 'aflowlib.duke.edu:AFLOWDATA/...'
            session: requests Session for connection reuse.

        Returns:
            POSCAR string, or empty string on failure.
        """
        # aurl uses ':' as path separator, not port
        url = "http://" + aurl.replace(":", "/") + "/CONTCAR.relax.vasp"
        try:
            resp = session.get(url, timeout=30)
            if resp.ok:
                return resp.text
        except Exception:
            pass
        return ""

    def _poscar_to_structure_dict(self, poscar_str: str) -> dict:
        """Parse POSCAR string into pymatgen structure dict."""
        if not poscar_str:
            return {}
        try:
            from pymatgen.io.vasp import Poscar
            poscar = Poscar.from_str(poscar_str)
            return poscar.structure.as_dict()
        except Exception:
            return {}

    def _entry_to_record(self, entry: dict, structure_dict: dict) -> MaterialRecord:
        """Convert an AFLUX entry dict + structure to MaterialRecord."""
        auid = entry.get("auid", "unknown")
        material_id = f"aflow-{auid}"
        formula = entry.get("compound", "unknown")

        formation_energy = entry.get("enthalpy_formation_atom")
        if formation_energy is not None:
            try:
                formation_energy = float(formation_energy)
            except (ValueError, TypeError):
                formation_energy = None

        sg = entry.get("spacegroup_relax")
        if sg is not None:
            try:
                sg = int(sg)
            except (ValueError, TypeError):
                sg = None

        return MaterialRecord(
            material_id=material_id,
            formula=formula,
            structure_dict=structure_dict,
            source="aflow",
            formation_energy_per_atom=formation_energy,
            energy_above_hull=None,
            voltage=None,
            capacity=None,
            is_stable=None,
            space_group=sg,
        )

    def _query_aflux(self, page: int) -> dict:
        """Query AFLUX API for one page of Li-containing entries.

        Args:
            page: 1-based page number.

        Returns:
            AFLUX JSON response dict, or empty dict on failure.
        """
        element = self.config.get("element", "Li")
        q = (
            f"species({element}),enthalpy_formation_atom,"
            f"paging({page}),format(json)"
        )
        for attempt in range(3):
            try:
                resp = requests.get(
                    AFLUX_URL + "?" + q, timeout=60,
                )
                if resp.ok:
                    return resp.json()
                self.logger.warning(
                    "AFLUX HTTP %d on page %d", resp.status_code, page,
                )
            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    "AFLUX request failed (attempt %d/3): %s", attempt + 1, e,
                )
                time.sleep(5 * (attempt + 1))
        return {}

    def fetch(self, force_refresh: bool = False) -> List[MaterialRecord]:
        """Fetch lithium-containing materials from AFLOW.

        Uses the AFLUX REST API with pagination to get metadata,
        then fetches CONTCAR structure files for each entry.

        Args:
            force_refresh: If True, bypass cache and re-fetch.

        Returns:
            List of MaterialRecord with AFLOW data.
        """
        cache_key = self._cache_key()

        if not force_refresh and self.cache.has(cache_key):
            self.logger.info("Loading AFLOW data from cache")
            data = self.cache.load(cache_key)
            return self._deserialize_records(data)

        self.logger.info("Fetching AFLOW data via AFLUX REST API...")
        max_entries = self.config.get("max_entries", 5000)

        entries = []
        page = 1
        total = None

        while True:
            data = self._query_aflux(page)
            if not data:
                self.logger.warning("Empty response on page %d, stopping", page)
                break

            # Parse the "N of total" keyed dict
            for key, entry in data.items():
                if isinstance(entry, dict):
                    entries.append(entry)
                    # Extract total from key like "1 of 146080"
                    if total is None:
                        parts = key.split(" of ")
                        if len(parts) == 2:
                            try:
                                total = int(parts[1])
                            except ValueError:
                                pass

            self.logger.info(
                "AFLUX page %d: %d entries so far (total: %s)",
                page, len(entries), total or "?",
            )

            if len(entries) >= max_entries:
                entries = entries[:max_entries]
                break

            if total is not None and len(entries) >= total:
                break

            page += 1

        self.logger.info(
            "AFLUX query complete: %d entries, fetching structures...",
            len(entries),
        )

        # Fetch structures via CONTCAR files
        session = requests.Session()
        records = []
        skipped = 0

        for i, entry in enumerate(entries):
            aurl = entry.get("aurl", "")
            if not aurl:
                skipped += 1
                continue

            poscar_str = self._fetch_contcar(aurl, session)
            structure_dict = self._poscar_to_structure_dict(poscar_str)

            record = self._entry_to_record(entry, structure_dict)
            records.append(record)

            if (i + 1) % 500 == 0:
                self.logger.info(
                    "AFLOW structure fetch: %d / %d (skipped %d)",
                    i + 1, len(entries), skipped,
                )

        self.logger.info(
            "Created %d AFLOW MaterialRecords (%d skipped)",
            len(records), skipped,
        )

        self.cache.save(
            cache_key,
            self._serialize_records(records),
            metadata={"source": "aflow", "count": len(records)},
        )
        self.logger.info("Cached AFLOW data with key: %s", cache_key)

        return records
