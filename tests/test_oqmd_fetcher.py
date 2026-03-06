"""Tests for OQMD data fetcher with HTTP fallback (DATA-02).

Tests use mocked qmpy_rester and requests to avoid real API calls.
"""

import json

import pytest
from dataclasses import asdict
from unittest.mock import MagicMock, patch, call

from cathode_ml.data.cache import DataCache
from cathode_ml.data.schemas import MaterialRecord


@pytest.fixture
def oqmd_config(sample_config):
    """Return full config dict for OQMDFetcher."""
    return sample_config


@pytest.fixture
def oqmd_cache(tmp_cache_dir):
    """Return a DataCache instance in a temp directory."""
    return DataCache(str(tmp_cache_dir))


@pytest.fixture
def sample_oqmd_entries():
    """Sample OQMD data entries as returned by qmpy_rester."""
    return [
        {
            "entry_id": 12345,
            "name": "Li2MnO3",
            "delta_e": -1.5,
            "stability": 0.02,
            "spacegroup": 12,
        },
        {
            "entry_id": 67890,
            "name": "LiCoO2",
            "delta_e": -2.1,
            "stability": 0.0,
            "spacegroup": 166,
        },
    ]


@pytest.fixture
def sample_oqmd_http_page1():
    """First page of HTTP API response."""
    return {
        "data": [
            {
                "entry_id": 12345,
                "name": "Li2MnO3",
                "delta_e": -1.5,
                "stability": 0.02,
                "spacegroup": 12,
            },
        ],
        "next": "http://oqmd.org/oqmdapi/formationenergy?offset=100",
        "links": {"next": "http://oqmd.org/oqmdapi/formationenergy?offset=100"},
    }


@pytest.fixture
def sample_oqmd_http_page2():
    """Second (last) page of HTTP API response."""
    return {
        "data": [
            {
                "entry_id": 67890,
                "name": "LiCoO2",
                "delta_e": -2.1,
                "stability": 0.0,
                "spacegroup": 166,
            },
        ],
        "next": None,
        "links": {"next": None},
    }


class TestOQMDFetcherInit:
    """Test OQMDFetcher initialization."""

    def test_init_stores_config_and_cache(self, oqmd_config, oqmd_cache):
        from cathode_ml.data.oqmd_fetcher import OQMDFetcher

        fetcher = OQMDFetcher(oqmd_config, oqmd_cache)
        assert fetcher.config == oqmd_config["data_sources"]["oqmd"]
        assert fetcher.cache is oqmd_cache


class TestOQMDFetcherCachePath:
    """Test cached data returned without API calls."""

    def test_cached_data_returns_without_api_call(self, oqmd_config, oqmd_cache):
        from cathode_ml.data.oqmd_fetcher import OQMDFetcher

        fetcher = OQMDFetcher(oqmd_config, oqmd_cache)

        # Pre-populate cache
        cache_key = oqmd_cache.cache_key("oqmd", {
            "element_set": oqmd_config["data_sources"]["oqmd"]["element_set"],
            "stability_max": oqmd_config["data_sources"]["oqmd"]["stability_max"],
        })
        cached_records = [
            asdict(MaterialRecord(
                material_id="oqmd-12345",
                formula="Li2MnO3",
                structure_dict={},
                source="oqmd",
                formation_energy_per_atom=-1.5,
                energy_above_hull=None,
                voltage=None,
                capacity=None,
                is_stable=None,
                space_group=12,
            ))
        ]
        oqmd_cache.save(cache_key, {"records": cached_records})

        with patch("cathode_ml.data.oqmd_fetcher._get_qmpy_rester") as mock_qr:
            result = fetcher.fetch()

        mock_qr.assert_not_called()
        assert len(result) == 1
        assert isinstance(result[0], MaterialRecord)
        assert result[0].material_id == "oqmd-12345"


class TestOQMDFetcherQmpyPath:
    """Test qmpy_rester success path."""

    def test_qmpy_rester_success_returns_records(
        self, oqmd_config, oqmd_cache, sample_oqmd_entries
    ):
        from cathode_ml.data.oqmd_fetcher import OQMDFetcher

        fetcher = OQMDFetcher(oqmd_config, oqmd_cache)

        mock_q = MagicMock()
        mock_q.get_oqmd_phases.return_value = {"data": sample_oqmd_entries}
        mock_q.__enter__ = MagicMock(return_value=mock_q)
        mock_q.__exit__ = MagicMock(return_value=False)

        mock_qr_module = MagicMock()
        mock_qr_module.QMPYRester.return_value = mock_q

        with patch("cathode_ml.data.oqmd_fetcher._get_qmpy_rester",
                    return_value=mock_qr_module):
            result = fetcher.fetch()

        assert len(result) == 2
        assert all(isinstance(r, MaterialRecord) for r in result)
        assert all(r.source == "oqmd" for r in result)

    def test_entries_convert_to_material_record(
        self, oqmd_config, oqmd_cache, sample_oqmd_entries
    ):
        from cathode_ml.data.oqmd_fetcher import OQMDFetcher

        fetcher = OQMDFetcher(oqmd_config, oqmd_cache)

        mock_q = MagicMock()
        mock_q.get_oqmd_phases.return_value = {"data": sample_oqmd_entries}
        mock_q.__enter__ = MagicMock(return_value=mock_q)
        mock_q.__exit__ = MagicMock(return_value=False)

        mock_qr_module = MagicMock()
        mock_qr_module.QMPYRester.return_value = mock_q

        with patch("cathode_ml.data.oqmd_fetcher._get_qmpy_rester",
                    return_value=mock_qr_module):
            result = fetcher.fetch()

        li2mno3 = next(r for r in result if "Li2MnO3" in r.formula)
        assert li2mno3.material_id == "oqmd-12345"
        assert li2mno3.formation_energy_per_atom == -1.5
        assert li2mno3.space_group == 12
        assert li2mno3.structure_dict == {}
        assert li2mno3.voltage is None
        assert li2mno3.capacity is None


class TestOQMDFetcherHTTPFallback:
    """Test HTTP fallback when qmpy_rester fails."""

    def test_qmpy_failure_triggers_http_fallback(
        self, oqmd_config, oqmd_cache, sample_oqmd_http_page1, sample_oqmd_http_page2
    ):
        from cathode_ml.data.oqmd_fetcher import OQMDFetcher

        fetcher = OQMDFetcher(oqmd_config, oqmd_cache)

        # qmpy_rester raises an exception
        mock_qr_module = MagicMock()
        mock_q = MagicMock()
        mock_q.get_oqmd_phases.side_effect = Exception("Connection refused")
        mock_q.__enter__ = MagicMock(return_value=mock_q)
        mock_q.__exit__ = MagicMock(return_value=False)
        mock_qr_module.QMPYRester.return_value = mock_q

        # HTTP fallback returns 2 pages
        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = sample_oqmd_http_page1
        mock_resp1.raise_for_status = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = sample_oqmd_http_page2
        mock_resp2.raise_for_status = MagicMock()

        with patch("cathode_ml.data.oqmd_fetcher._get_qmpy_rester",
                    return_value=mock_qr_module):
            with patch("cathode_ml.data.oqmd_fetcher.requests") as mock_requests:
                mock_requests.get.side_effect = [mock_resp1, mock_resp2]
                result = fetcher.fetch()

        # Should have 2 records (one from each page)
        assert len(result) == 2
        assert all(isinstance(r, MaterialRecord) for r in result)

    def test_http_pagination_stops_on_no_next(
        self, oqmd_config, oqmd_cache, sample_oqmd_http_page2
    ):
        from cathode_ml.data.oqmd_fetcher import OQMDFetcher

        fetcher = OQMDFetcher(oqmd_config, oqmd_cache)

        mock_qr_module = MagicMock()
        mock_q = MagicMock()
        mock_q.get_oqmd_phases.side_effect = Exception("Timeout")
        mock_q.__enter__ = MagicMock(return_value=mock_q)
        mock_q.__exit__ = MagicMock(return_value=False)
        mock_qr_module.QMPYRester.return_value = mock_q

        # Only one page with no "next"
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_oqmd_http_page2
        mock_resp.raise_for_status = MagicMock()

        with patch("cathode_ml.data.oqmd_fetcher._get_qmpy_rester",
                    return_value=mock_qr_module):
            with patch("cathode_ml.data.oqmd_fetcher.requests") as mock_requests:
                mock_requests.get.side_effect = [mock_resp]
                result = fetcher.fetch()

        # Only 1 HTTP call (no pagination)
        assert mock_requests.get.call_count == 1
        assert len(result) == 1


class TestOQMDFetcherForceRefresh:
    """Test force_refresh bypasses cache."""

    def test_force_refresh_bypasses_cache(
        self, oqmd_config, oqmd_cache, sample_oqmd_entries
    ):
        from cathode_ml.data.oqmd_fetcher import OQMDFetcher

        fetcher = OQMDFetcher(oqmd_config, oqmd_cache)

        # Pre-populate cache
        cache_key = oqmd_cache.cache_key("oqmd", {
            "element_set": oqmd_config["data_sources"]["oqmd"]["element_set"],
            "stability_max": oqmd_config["data_sources"]["oqmd"]["stability_max"],
        })
        oqmd_cache.save(cache_key, {"records": []})

        mock_q = MagicMock()
        mock_q.get_oqmd_phases.return_value = {"data": sample_oqmd_entries}
        mock_q.__enter__ = MagicMock(return_value=mock_q)
        mock_q.__exit__ = MagicMock(return_value=False)

        mock_qr_module = MagicMock()
        mock_qr_module.QMPYRester.return_value = mock_q

        with patch("cathode_ml.data.oqmd_fetcher._get_qmpy_rester",
                    return_value=mock_qr_module):
            result = fetcher.fetch(force_refresh=True)

        # Should have fetched from API despite cache
        mock_q.get_oqmd_phases.assert_called_once()
        assert len(result) == 2


class TestOQMDFetcherCaching:
    """Test that results are cached after fetch."""

    def test_fetch_caches_results(
        self, oqmd_config, oqmd_cache, sample_oqmd_entries
    ):
        from cathode_ml.data.oqmd_fetcher import OQMDFetcher

        fetcher = OQMDFetcher(oqmd_config, oqmd_cache)

        mock_q = MagicMock()
        mock_q.get_oqmd_phases.return_value = {"data": sample_oqmd_entries}
        mock_q.__enter__ = MagicMock(return_value=mock_q)
        mock_q.__exit__ = MagicMock(return_value=False)

        mock_qr_module = MagicMock()
        mock_qr_module.QMPYRester.return_value = mock_q

        with patch("cathode_ml.data.oqmd_fetcher._get_qmpy_rester",
                    return_value=mock_qr_module):
            fetcher.fetch()

        cache_key = oqmd_cache.cache_key("oqmd", {
            "element_set": oqmd_config["data_sources"]["oqmd"]["element_set"],
            "stability_max": oqmd_config["data_sources"]["oqmd"]["stability_max"],
        })
        assert oqmd_cache.has(cache_key)
