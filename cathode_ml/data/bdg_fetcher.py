"""Battery Data Genome file downloader and parser.

Downloads battery material data from NREL datasets and parses
into MaterialRecord objects. BDG is a supplementary source --
the pipeline works with MP + OQMD alone if BDG is unavailable.
"""

import csv
import io
import logging
from dataclasses import asdict
from typing import Optional

import requests

from cathode_ml.data.cache import DataCache
from cathode_ml.data.schemas import MaterialRecord


class BDGFetcher:
    """Fetches battery material data from Battery Data Genome datasets.

    This is a file downloader + CSV parser, NOT an API client.
    Downloads CSV data from configured source URLs and parses rows
    into MaterialRecord objects.

    If the source URL is unavailable or data format unexpected,
    logs a warning and returns empty list -- the pipeline works
    with MP + OQMD alone.

    Usage:
        fetcher = BDGFetcher(config, cache)
        records = fetcher.fetch()
    """

    def __init__(self, config: dict, cache: DataCache) -> None:
        """Initialize with config and cache.

        Args:
            config: Full pipeline config dict. BDG settings read from
                    config["data_sources"]["battery_data_genome"].
            cache: DataCache instance for caching downloaded data.
        """
        self.config = config["data_sources"]["battery_data_genome"]
        self.cache = cache
        self.logger = logging.getLogger("cathode_ml.data.bdg_fetcher")

    def fetch(self, force_refresh: bool = False) -> list:
        """Fetch battery material data from BDG source.

        Checks cache first; downloads if needed. Returns empty list
        if source is disabled or unavailable.

        Args:
            force_refresh: If True, bypass cache and re-download.

        Returns:
            List of MaterialRecord objects.
        """
        if not self.config.get("enabled", True):
            self.logger.info("BDG source disabled in config")
            return []

        source_url = self.config.get("source_url", "")
        cache_key = self.cache.cache_key("bdg", {"source_url": source_url})

        # Check cache unless force refresh
        if not force_refresh and self.cache.has(cache_key):
            self.logger.info("Loading BDG data from cache")
            cached = self.cache.load(cache_key)
            return self._records_from_dicts(cached.get("records", []))

        # Download fresh data
        try:
            self.logger.info(f"Downloading BDG data from {source_url}")
            response = requests.get(source_url, timeout=60)
            response.raise_for_status()
            records = self._parse_csv(response.text)
        except Exception as e:
            self.logger.warning(f"Failed to fetch BDG data: {e}")
            return []

        # Cache the results
        record_dicts = [self._record_to_dict(r) for r in records]
        self.cache.save(cache_key, {"records": record_dicts},
                        metadata={"source": "battery_data_genome", "url": source_url})

        self.logger.info(f"Fetched {len(records)} records from BDG")
        return records

    def _parse_csv(self, text: str) -> list:
        """Parse CSV text into MaterialRecord objects.

        Args:
            text: CSV content with headers.

        Returns:
            List of MaterialRecord objects.
        """
        reader = csv.DictReader(io.StringIO(text))
        records = []
        for i, row in enumerate(reader):
            try:
                record = MaterialRecord(
                    material_id=row.get("material_id", f"bdg-{i:04d}"),
                    formula=row.get("formula", ""),
                    structure_dict={},  # Text-mined entries lack crystal structures
                    source="battery_data_genome",
                    formation_energy_per_atom=self._safe_float(row.get("formation_energy_per_atom")),
                    energy_above_hull=self._safe_float(row.get("energy_above_hull")),
                    voltage=self._safe_float(row.get("voltage")),
                    capacity=self._safe_float(row.get("capacity")),
                    is_stable=None,
                    space_group=self._safe_int(row.get("space_group")),
                )
                records.append(record)
            except Exception as e:
                self.logger.warning(f"Skipping row {i}: {e}")
                continue

        return records

    def _records_from_dicts(self, dicts: list) -> list:
        """Convert list of dicts back to MaterialRecord objects.

        Args:
            dicts: List of dicts with MaterialRecord fields.

        Returns:
            List of MaterialRecord objects.
        """
        records = []
        for d in dicts:
            records.append(MaterialRecord(
                material_id=d.get("material_id", ""),
                formula=d.get("formula", ""),
                structure_dict=d.get("structure_dict", {}),
                source=d.get("source", "battery_data_genome"),
                formation_energy_per_atom=d.get("formation_energy_per_atom"),
                energy_above_hull=d.get("energy_above_hull"),
                voltage=d.get("voltage"),
                capacity=d.get("capacity"),
                is_stable=d.get("is_stable"),
                space_group=d.get("space_group"),
            ))
        return records

    @staticmethod
    def _record_to_dict(record: MaterialRecord) -> dict:
        """Convert MaterialRecord to dict for caching."""
        return asdict(record)

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float, returning None on failure."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert value to int, returning None on failure."""
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
