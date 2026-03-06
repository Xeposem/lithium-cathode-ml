"""Tests for Materials Project data fetcher (DATA-01).

Tests use mocked MPRester to avoid real API calls.
"""

import pytest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from cathode_ml.data.cache import DataCache
from cathode_ml.data.schemas import MaterialRecord


@pytest.fixture
def mp_config(sample_config):
    """Return full config dict for MPFetcher."""
    return sample_config


@pytest.fixture
def mp_cache(tmp_cache_dir):
    """Return a DataCache instance in a temp directory."""
    return DataCache(str(tmp_cache_dir))


def _make_mock_summary_doc(material_id, formula, structure_dict,
                           formation_energy=-2.0, ehull=0.01,
                           is_stable=True, space_group=166):
    """Create a mock summary doc matching MPRester response shape."""
    doc = MagicMock()
    doc.material_id = material_id
    doc.formula_pretty = formula
    mock_structure = MagicMock()
    mock_structure.as_dict.return_value = structure_dict
    doc.structure = mock_structure
    doc.formation_energy_per_atom = formation_energy
    doc.energy_above_hull = ehull
    doc.is_stable = is_stable
    mock_symmetry = MagicMock()
    mock_symmetry.number = space_group
    doc.symmetry = mock_symmetry
    return doc


def _make_mock_electrode_doc(battery_id, material_ids, voltage, capacity,
                             working_ion="Li"):
    """Create a mock electrode doc."""
    doc = MagicMock()
    doc.battery_id = battery_id
    doc.material_ids = material_ids
    doc.average_voltage = voltage
    doc.capacity_grav = capacity
    doc.working_ion = working_ion
    doc.framework_formula = "CoO2"
    return doc


@pytest.fixture
def mock_summary_docs(sample_structure_dict):
    """Two mock summary docs."""
    return [
        _make_mock_summary_doc("mp-22526", "LiCoO2", sample_structure_dict,
                               formation_energy=-2.1, ehull=0.0,
                               is_stable=True, space_group=166),
        _make_mock_summary_doc("mp-19017", "LiFePO4", sample_structure_dict,
                               formation_energy=-1.8, ehull=0.05,
                               is_stable=True, space_group=62),
    ]


@pytest.fixture
def mock_electrode_docs():
    """Electrode docs -- one matches mp-22526, one is non-Li."""
    return [
        _make_mock_electrode_doc("bat-1", ["mp-22526"], 3.9, 140.0, "Li"),
        _make_mock_electrode_doc("bat-2", ["mp-99999"], 2.5, 100.0, "Na"),
    ]


class TestMPFetcherInit:
    """Test MPFetcher initialization."""

    def test_init_stores_config_and_cache(self, mp_config, mp_cache):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)
        assert fetcher.config == mp_config["data_sources"]["materials_project"]
        assert fetcher.cache is mp_cache


class TestMPFetcherCachePath:
    """Test that cached data is returned without API calls."""

    def test_cached_data_returns_without_api_call(self, mp_config, mp_cache,
                                                   sample_structure_dict):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)

        # Pre-populate cache with serialized records
        cache_key = mp_cache.cache_key("mp", {
            "elements": mp_config["data_sources"]["materials_project"]["elements_must_contain"],
            "energy_above_hull_max": mp_config["data_sources"]["materials_project"]["energy_above_hull_max"],
        })
        cached_records = [
            asdict(MaterialRecord(
                material_id="mp-22526",
                formula="LiCoO2",
                structure_dict=sample_structure_dict,
                source="materials_project",
                formation_energy_per_atom=-2.1,
                energy_above_hull=0.0,
                voltage=3.9,
                capacity=140.0,
                is_stable=True,
                space_group=166,
            ))
        ]
        mp_cache.save(cache_key, {"records": cached_records})

        with patch("cathode_ml.data.mp_fetcher.MPRester") as mock_rester:
            result = fetcher.fetch()

        # MPRester should NOT have been called
        mock_rester.assert_not_called()
        assert len(result) == 1
        assert isinstance(result[0], MaterialRecord)
        assert result[0].material_id == "mp-22526"
        assert result[0].voltage == 3.9


class TestMPFetcherAPIPath:
    """Test the uncached API path."""

    def test_fetch_calls_api_and_returns_material_records(
        self, mp_config, mp_cache, mock_summary_docs, mock_electrode_docs
    ):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)

        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = mock_summary_docs
        mock_mpr.insertion_electrodes.search.return_value = mock_electrode_docs
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)

        with patch("cathode_ml.data.mp_fetcher.MPRester", return_value=mock_mpr):
            with patch.dict("os.environ", {"MP_API_KEY": "test-key"}):
                result = fetcher.fetch()

        # Should have called both search endpoints
        mock_mpr.materials.summary.search.assert_called_once()
        mock_mpr.insertion_electrodes.search.assert_called_once()

        assert len(result) == 2
        assert all(isinstance(r, MaterialRecord) for r in result)

    def test_electrode_data_joined_by_material_id(
        self, mp_config, mp_cache, mock_summary_docs, mock_electrode_docs
    ):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)

        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = mock_summary_docs
        mock_mpr.insertion_electrodes.search.return_value = mock_electrode_docs
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)

        with patch("cathode_ml.data.mp_fetcher.MPRester", return_value=mock_mpr):
            with patch.dict("os.environ", {"MP_API_KEY": "test-key"}):
                result = fetcher.fetch()

        # mp-22526 has electrode data (bat-1), mp-19017 does not
        mp22526 = next(r for r in result if r.material_id == "mp-22526")
        mp19017 = next(r for r in result if r.material_id == "mp-19017")

        assert mp22526.voltage == 3.9
        assert mp22526.capacity == 140.0
        assert mp19017.voltage is None
        assert mp19017.capacity is None

    def test_source_is_materials_project(
        self, mp_config, mp_cache, mock_summary_docs, mock_electrode_docs
    ):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)

        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = mock_summary_docs
        mock_mpr.insertion_electrodes.search.return_value = mock_electrode_docs
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)

        with patch("cathode_ml.data.mp_fetcher.MPRester", return_value=mock_mpr):
            with patch.dict("os.environ", {"MP_API_KEY": "test-key"}):
                result = fetcher.fetch()

        assert all(r.source == "materials_project" for r in result)

    def test_fetch_caches_results(
        self, mp_config, mp_cache, mock_summary_docs, mock_electrode_docs
    ):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)

        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = mock_summary_docs
        mock_mpr.insertion_electrodes.search.return_value = mock_electrode_docs
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)

        with patch("cathode_ml.data.mp_fetcher.MPRester", return_value=mock_mpr):
            with patch.dict("os.environ", {"MP_API_KEY": "test-key"}):
                fetcher.fetch()

        # Cache should now have the data
        cache_key = mp_cache.cache_key("mp", {
            "elements": mp_config["data_sources"]["materials_project"]["elements_must_contain"],
            "energy_above_hull_max": mp_config["data_sources"]["materials_project"]["energy_above_hull_max"],
        })
        assert mp_cache.has(cache_key)


class TestMPFetcherForceRefresh:
    """Test force_refresh bypasses cache."""

    def test_force_refresh_bypasses_cache(
        self, mp_config, mp_cache, mock_summary_docs, mock_electrode_docs,
        sample_structure_dict
    ):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)

        # Pre-populate cache
        cache_key = mp_cache.cache_key("mp", {
            "elements": mp_config["data_sources"]["materials_project"]["elements_must_contain"],
            "energy_above_hull_max": mp_config["data_sources"]["materials_project"]["energy_above_hull_max"],
        })
        mp_cache.save(cache_key, {"records": []})

        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = mock_summary_docs
        mock_mpr.insertion_electrodes.search.return_value = mock_electrode_docs
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)

        with patch("cathode_ml.data.mp_fetcher.MPRester", return_value=mock_mpr):
            with patch.dict("os.environ", {"MP_API_KEY": "test-key"}):
                result = fetcher.fetch(force_refresh=True)

        # API should have been called despite cache
        mock_mpr.materials.summary.search.assert_called_once()
        assert len(result) == 2


class TestMPFetcherStructureConversion:
    """Test structure dict conversion."""

    def test_structure_dict_from_as_dict(
        self, mp_config, mp_cache, mock_summary_docs, mock_electrode_docs,
        sample_structure_dict
    ):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)

        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = mock_summary_docs
        mock_mpr.insertion_electrodes.search.return_value = mock_electrode_docs
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)

        with patch("cathode_ml.data.mp_fetcher.MPRester", return_value=mock_mpr):
            with patch.dict("os.environ", {"MP_API_KEY": "test-key"}):
                result = fetcher.fetch()

        # structure_dict should be from structure.as_dict()
        assert result[0].structure_dict == sample_structure_dict


class TestMPFetcherSpaceGroup:
    """Test space group extraction."""

    def test_space_group_extracted_from_symmetry(
        self, mp_config, mp_cache, mock_summary_docs, mock_electrode_docs
    ):
        from cathode_ml.data.mp_fetcher import MPFetcher

        fetcher = MPFetcher(mp_config, mp_cache)

        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = mock_summary_docs
        mock_mpr.insertion_electrodes.search.return_value = mock_electrode_docs
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)

        with patch("cathode_ml.data.mp_fetcher.MPRester", return_value=mock_mpr):
            with patch.dict("os.environ", {"MP_API_KEY": "test-key"}):
                result = fetcher.fetch()

        assert result[0].space_group == 166
        assert result[1].space_group == 62
