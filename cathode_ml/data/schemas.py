"""Data validation dataclasses for materials."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MaterialRecord:
    """Record for a single material from any data source.

    Required fields identify the material. Optional fields hold
    computed properties that may or may not be available depending
    on the data source.
    """

    material_id: str
    formula: str
    structure_dict: dict
    source: str
    formation_energy_per_atom: Optional[float] = None
    energy_above_hull: Optional[float] = None
    voltage: Optional[float] = None
    capacity: Optional[float] = None
    is_stable: Optional[bool] = None
    space_group: Optional[int] = None


@dataclass
class FilterRecord:
    """Record documenting a single filter step in the cleaning pipeline.

    Used to build a structured cleaning log for reproducibility (DATA-06).
    """

    filter_name: str
    description: str
    rationale: str
    count_before: int
    count_after: int
    count_removed: int
