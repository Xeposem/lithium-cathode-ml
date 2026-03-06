"""Shared pytest fixtures for cathode_ml tests."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_config():
    """Return a minimal valid config dict."""
    return {
        "random_seeds": {
            "python": 42,
            "numpy": 42,
        },
        "data_sources": {
            "materials_project": {
                "enabled": True,
                "elements_must_contain": ["Li"],
                "fields": [
                    "material_id",
                    "formula_pretty",
                    "structure",
                    "formation_energy_per_atom",
                    "energy_above_hull",
                    "is_stable",
                    "symmetry",
                ],
                "energy_above_hull_max": 0.1,
            },
            "oqmd": {
                "enabled": True,
                "element_set": "Li",
                "stability_max": 0.1,
            },
            "battery_data_genome": {
                "enabled": True,
                "source_url": "https://example.com/bdg",
            },
        },
        "filters": {
            "min_sites": 2,
            "max_sites": 200,
            "formation_energy_range": [-5.0, 0.5],
            "required_properties": ["formation_energy_per_atom"],
            "remove_noble_gases": True,
            "outlier_iqr_multiplier": 1.5,
        },
        "cache": {
            "directory": "data/raw",
            "use_cache": True,
        },
    }


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def sample_structure_dict():
    """Return a pymatgen Structure.as_dict() for a simple LiCoO2-like structure."""
    return {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": 0,
        "lattice": {
            "matrix": [
                [2.8, 0.0, 0.0],
                [-1.4, 2.4249, 0.0],
                [0.0, 0.0, 14.05],
            ],
            "pbc": [True, True, True],
        },
        "sites": [
            {
                "species": [{"element": "Li", "occu": 1}],
                "abc": [0.0, 0.0, 0.5],
                "xyz": [0.0, 0.0, 7.025],
            },
            {
                "species": [{"element": "Co", "occu": 1}],
                "abc": [0.0, 0.0, 0.0],
                "xyz": [0.0, 0.0, 0.0],
            },
            {
                "species": [{"element": "O", "occu": 1}],
                "abc": [0.0, 0.0, 0.2604],
                "xyz": [0.0, 0.0, 3.6586],
            },
        ],
    }


@pytest.fixture
def sample_material_record():
    """Return a MaterialRecord with test data."""
    from cathode_ml.data.schemas import MaterialRecord

    return MaterialRecord(
        material_id="mp-22526",
        formula="LiCoO2",
        structure_dict={"lattice": {"matrix": [[2.8, 0, 0], [-1.4, 2.42, 0], [0, 0, 14.05]]}},
        source="materials_project",
        formation_energy_per_atom=-2.1,
        energy_above_hull=0.0,
        voltage=3.9,
        capacity=140.0,
        is_stable=True,
        space_group=166,
    )
