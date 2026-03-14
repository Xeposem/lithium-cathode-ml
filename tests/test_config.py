"""Tests for cathode_ml.config module."""

import random

import numpy as np
import pytest


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_returns_dict(self):
        """load_config returns a dict with all expected top-level keys."""
        from cathode_ml.config import load_config

        config = load_config("configs/data.yaml")
        assert isinstance(config, dict)
        assert "random_seeds" in config
        assert "data_sources" in config
        assert "filters" in config
        assert "cache" in config

    def test_load_config_random_seeds(self):
        """Config contains random_seeds with python and numpy keys."""
        from cathode_ml.config import load_config

        config = load_config("configs/data.yaml")
        seeds = config["random_seeds"]
        assert "python" in seeds
        assert "numpy" in seeds
        assert isinstance(seeds["python"], int)
        assert isinstance(seeds["numpy"], int)

    def test_load_config_nonexistent_path_raises(self):
        """load_config raises FileNotFoundError for nonexistent path."""
        from cathode_ml.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/path.yaml")

    def test_load_config_data_sources(self):
        """Config contains data_sources section with expected sources."""
        from cathode_ml.config import load_config

        config = load_config("configs/data.yaml")
        sources = config["data_sources"]
        assert "materials_project" in sources
        assert "oqmd" in sources
        assert "aflow" in sources
        assert "jarvis" in sources


class TestSetSeeds:
    """Tests for the set_seeds function."""

    def test_set_seeds_python_reproducible(self):
        """set_seeds makes random.random() produce same value on consecutive calls."""
        from cathode_ml.config import set_seeds

        config = {"random_seeds": {"python": 42, "numpy": 42}}

        set_seeds(config)
        val1 = random.random()

        set_seeds(config)
        val2 = random.random()

        assert val1 == val2

    def test_set_seeds_numpy_reproducible(self):
        """set_seeds makes np.random.random() produce same value on consecutive calls."""
        from cathode_ml.config import set_seeds

        config = {"random_seeds": {"python": 42, "numpy": 42}}

        set_seeds(config)
        val1 = np.random.random()

        set_seeds(config)
        val2 = np.random.random()

        assert val1 == val2

    def test_set_seeds_different_seeds_different_output(self):
        """Different seed values produce different random output."""
        from cathode_ml.config import set_seeds

        config_a = {"random_seeds": {"python": 42, "numpy": 42}}
        config_b = {"random_seeds": {"python": 99, "numpy": 99}}

        set_seeds(config_a)
        val_a = random.random()

        set_seeds(config_b)
        val_b = random.random()

        assert val_a != val_b


class TestSchemas:
    """Tests for data schemas."""

    def test_material_record_creation(self):
        """MaterialRecord dataclass holds all expected fields."""
        from cathode_ml.data.schemas import MaterialRecord

        record = MaterialRecord(
            material_id="mp-12345",
            formula="LiFePO4",
            structure_dict={"lattice": {}},
            source="materials_project",
            formation_energy_per_atom=-1.5,
            energy_above_hull=0.01,
            voltage=3.4,
            capacity=170.0,
            is_stable=True,
            space_group=62,
        )
        assert record.material_id == "mp-12345"
        assert record.formula == "LiFePO4"
        assert record.source == "materials_project"
        assert record.formation_energy_per_atom == -1.5
        assert record.voltage == 3.4

    def test_material_record_optional_none(self):
        """MaterialRecord allows None for optional property fields."""
        from cathode_ml.data.schemas import MaterialRecord

        record = MaterialRecord(
            material_id="mp-99999",
            formula="LiMnO2",
            structure_dict={},
            source="oqmd",
        )
        assert record.formation_energy_per_atom is None
        assert record.energy_above_hull is None
        assert record.voltage is None
        assert record.capacity is None
        assert record.is_stable is None
        assert record.space_group is None

    def test_filter_record_creation(self):
        """FilterRecord dataclass holds filter metadata."""
        from cathode_ml.data.schemas import FilterRecord

        record = FilterRecord(
            filter_name="min_sites",
            description="Remove structures with fewer than 2 sites",
            rationale="Single-atom structures are not useful for GNN",
            count_before=1000,
            count_after=950,
            count_removed=50,
        )
        assert record.filter_name == "min_sites"
        assert record.count_before == 1000
        assert record.count_removed == 50
