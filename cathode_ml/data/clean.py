"""Validation, deduplication, outlier removal, and filter logging.

Provides CleaningPipeline that applies a series of data quality
filters to MaterialRecord lists, documenting every step with
FilterRecord for reproducibility (DATA-05, DATA-06).
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Callable, List, Optional, Tuple

import numpy as np

from cathode_ml.data.schemas import FilterRecord, MaterialRecord


class CleaningPipeline:
    """Comprehensive cleaning pipeline for material records.

    Applies validation, outlier removal, and deduplication filters.
    Every filter step is documented with a FilterRecord for full
    reproducibility and audit trail.

    Usage:
        pipeline = CleaningPipeline()
        cleaned = pipeline.run(records, config)
        pipeline.save_log("data/logs/cleaning_log.json")
    """

    def __init__(self) -> None:
        """Initialize with empty filter log."""
        self.log: List[FilterRecord] = []
        self.logger = logging.getLogger("cathode_ml.cleaning")

    def apply_filter(
        self,
        records: List[MaterialRecord],
        filter_fn: Callable[[MaterialRecord], bool],
        name: str,
        rationale: str,
    ) -> List[MaterialRecord]:
        """Apply a filter function and log the result.

        Args:
            records: Input material records.
            filter_fn: Function returning True to keep record.
            name: Name of the filter for logging.
            rationale: Why this filter is applied.

        Returns:
            Filtered list of MaterialRecord objects.
        """
        count_before = len(records)
        records = [r for r in records if filter_fn(r)]
        count_after = len(records)
        count_removed = count_before - count_after

        record = FilterRecord(
            filter_name=name,
            description=f"Applied {name}",
            rationale=rationale,
            count_before=count_before,
            count_after=count_after,
            count_removed=count_removed,
        )
        self.log.append(record)
        self.logger.info(f"{name}: {count_before} -> {count_after} ({count_removed} removed)")
        return records

    def validate_structure(self, structure_dict: dict) -> Tuple[bool, str]:
        """Validate a pymatgen structure dictionary.

        Checks:
        - Non-empty structure dict
        - Positive volume
        - 2 to 200 sites
        - No overlapping atoms (< 0.5 Angstrom apart)
        - Valid species

        Args:
            structure_dict: pymatgen Structure.as_dict() format.

        Returns:
            Tuple of (is_valid, reason). reason is empty string if valid.
        """
        if not structure_dict:
            return False, "No structure data"

        try:
            from pymatgen.core import Structure

            structure = Structure.from_dict(structure_dict)
        except Exception as e:
            return False, f"Cannot parse structure: {e}"

        # Check volume
        if structure.volume <= 0:
            return False, "Non-positive volume"

        # Check site count
        num_sites = len(structure.sites)
        if num_sites < 2:
            return False, f"Too few sites ({num_sites})"
        if num_sites > 200:
            return False, f"Too many sites ({num_sites})"

        # Check for overlapping atoms (distance < 0.5 Angstrom)
        try:
            dist_matrix = structure.distance_matrix
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix)
            if min_dist < 0.5:
                return False, f"Overlapping atoms (min distance {min_dist:.3f} A)"
        except Exception:
            pass  # If distance calculation fails, skip this check

        return True, ""

    def remove_outliers(
        self,
        records: List[MaterialRecord],
        property_name: str,
        iqr_multiplier: float = 1.5,
    ) -> List[MaterialRecord]:
        """Remove outliers using IQR method.

        Values outside [Q1 - iqr*IQR, Q3 + iqr*IQR] are removed.

        Args:
            records: Input material records.
            property_name: Attribute name to check (e.g., "formation_energy_per_atom").
            iqr_multiplier: Multiplier for IQR range (default 1.5).

        Returns:
            Records with outliers removed.
        """
        values = []
        for r in records:
            val = getattr(r, property_name, None)
            if val is not None:
                values.append(val)

        if len(values) < 4:
            # Not enough data to compute meaningful IQR
            self.log.append(FilterRecord(
                filter_name=f"outlier_{property_name}",
                description=f"Outlier removal on {property_name}",
                rationale=f"IQR-based outlier detection (multiplier={iqr_multiplier})",
                count_before=len(records),
                count_after=len(records),
                count_removed=0,
            ))
            return records

        arr = np.array(values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        count_before = len(records)

        def in_range(r: MaterialRecord) -> bool:
            val = getattr(r, property_name, None)
            if val is None:
                return True  # Keep records without the property
            return lower <= val <= upper

        records = [r for r in records if in_range(r)]
        count_after = len(records)

        self.log.append(FilterRecord(
            filter_name=f"outlier_{property_name}",
            description=f"Outlier removal on {property_name} [{lower:.4f}, {upper:.4f}]",
            rationale=f"IQR-based outlier detection (multiplier={iqr_multiplier})",
            count_before=count_before,
            count_after=count_after,
            count_removed=count_before - count_after,
        ))
        self.logger.info(
            f"Outlier removal ({property_name}): {count_before} -> {count_after}"
        )
        return records

    def deduplicate(self, records: List[MaterialRecord]) -> List[MaterialRecord]:
        """Remove duplicate records by reduced formula + space group.

        For duplicates, prefers materials_project source.

        Args:
            records: Input material records.

        Returns:
            Deduplicated list of records.
        """
        count_before = len(records)

        # Group by (formula, space_group)
        groups: dict = defaultdict(list)
        for r in records:
            key = (r.formula, r.space_group)
            groups[key].append(r)

        # For each group, pick the best record
        SOURCE_PRIORITY = {
            "materials_project": 0,
            "oqmd": 1,
            "battery_data_genome": 2,
        }

        result = []
        for key, group in groups.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Sort by source priority, pick first
                group.sort(key=lambda r: SOURCE_PRIORITY.get(r.source, 99))
                result.append(group[0])

        count_after = len(result)
        self.log.append(FilterRecord(
            filter_name="deduplicate",
            description="Deduplicate by reduced formula + space group",
            rationale="Remove cross-source duplicates, prefer Materials Project",
            count_before=count_before,
            count_after=count_after,
            count_removed=count_before - count_after,
        ))
        self.logger.info(f"Deduplication: {count_before} -> {count_after}")
        return result

    def run(self, records: List[MaterialRecord], config: dict) -> List[MaterialRecord]:
        """Run the full cleaning pipeline.

        Applies filters in order:
        1. Validate structures (remove empty/invalid)
        2. Filter by site count
        3. Filter by formation energy range
        4. Remove outliers on formation_energy_per_atom
        5. Deduplicate

        Args:
            records: Raw material records from all sources.
            config: Pipeline configuration dict with "filters" section.

        Returns:
            Cleaned list of MaterialRecord objects.
        """
        filters_config = config["filters"]
        self.logger.info(f"Starting cleaning pipeline with {len(records)} records")

        # 1. Validate structures -- remove entries with no/invalid structure
        records = self.apply_filter(
            records,
            lambda r: self.validate_structure(r.structure_dict)[0],
            "validate_structure",
            "Remove records with invalid or missing crystal structures",
        )

        # 2. Filter by formation energy range
        e_min, e_max = filters_config.get("formation_energy_range", [-5.0, 0.5])
        records = self.apply_filter(
            records,
            lambda r: (
                r.formation_energy_per_atom is not None
                and e_min <= r.formation_energy_per_atom <= e_max
            ),
            "formation_energy_range",
            f"Keep records with formation energy in [{e_min}, {e_max}] eV/atom",
        )

        # 3. Remove outliers on formation_energy_per_atom
        iqr_mult = filters_config.get("outlier_iqr_multiplier", 1.5)
        records = self.remove_outliers(
            records, "formation_energy_per_atom", iqr_multiplier=iqr_mult
        )

        # 4. Deduplicate
        records = self.deduplicate(records)

        self.logger.info(f"Cleaning complete: {len(records)} records remaining")
        return records

    def save_log(self, path: str) -> None:
        """Save cleaning log as JSON.

        Args:
            path: Output file path for the cleaning log.
        """
        from pathlib import Path

        log_dir = Path(path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.log], f, indent=2)
        self.logger.info(f"Cleaning log saved to {path}")
