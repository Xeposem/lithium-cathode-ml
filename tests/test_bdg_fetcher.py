"""Tests for Battery Data Genome fetcher."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from cathode_ml.data.schemas import MaterialRecord


class TestBDGFetcherInit:
    """Test BDGFetcher initialization."""

    def test_init_extracts_bdg_config(self, sample_config, tmp_cache_dir):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        fetcher = BDGFetcher(sample_config, cache)
        assert fetcher.config == sample_config["data_sources"]["battery_data_genome"]

    def test_init_stores_cache(self, sample_config, tmp_cache_dir):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        fetcher = BDGFetcher(sample_config, cache)
        assert fetcher.cache is cache


class TestBDGFetcherDisabled:
    """Test BDGFetcher when source is disabled."""

    def test_disabled_source_returns_empty(self, sample_config, tmp_cache_dir):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        sample_config["data_sources"]["battery_data_genome"]["enabled"] = False
        cache = DataCache(str(tmp_cache_dir))
        fetcher = BDGFetcher(sample_config, cache)
        result = fetcher.fetch()
        assert result == []

    def test_disabled_source_logs_info(self, sample_config, tmp_cache_dir, caplog):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        sample_config["data_sources"]["battery_data_genome"]["enabled"] = False
        cache = DataCache(str(tmp_cache_dir))
        fetcher = BDGFetcher(sample_config, cache)
        with caplog.at_level(logging.INFO):
            fetcher.fetch()
        assert "disabled" in caplog.text.lower()


class TestBDGFetcherCache:
    """Test cache behavior."""

    def test_returns_cached_data(self, sample_config, tmp_cache_dir):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        # Pre-populate cache with a list of material record dicts
        cached_records = [
            {
                "material_id": "bdg-001",
                "formula": "LiCoO2",
                "structure_dict": {},
                "source": "battery_data_genome",
                "voltage": 3.9,
            }
        ]
        cache_key = cache.cache_key("bdg", {"source_url": sample_config["data_sources"]["battery_data_genome"]["source_url"]})
        cache.save(cache_key, {"records": cached_records})

        fetcher = BDGFetcher(sample_config, cache)
        result = fetcher.fetch()
        assert len(result) == 1
        assert isinstance(result[0], MaterialRecord)
        assert result[0].formula == "LiCoO2"

    def test_force_refresh_bypasses_cache(self, sample_config, tmp_cache_dir):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        # Pre-populate cache
        cache_key = cache.cache_key("bdg", {"source_url": sample_config["data_sources"]["battery_data_genome"]["source_url"]})
        cache.save(cache_key, {"records": [{"material_id": "bdg-old", "formula": "Old", "structure_dict": {}, "source": "battery_data_genome"}]})

        fetcher = BDGFetcher(sample_config, cache)
        # Mock the download to return different data
        with patch("cathode_ml.data.bdg_fetcher.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "material_id,formula,voltage,capacity\nbdg-new,LiFePO4,3.4,170\n"
            mock_response.raise_for_status = MagicMock()
            mock_requests.get.return_value = mock_response

            result = fetcher.fetch(force_refresh=True)
            mock_requests.get.assert_called_once()
            assert any(r.formula == "LiFePO4" for r in result)


class TestBDGFetcherDownload:
    """Test download and parse behavior."""

    def test_successful_download_parses_csv(self, sample_config, tmp_cache_dir):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        fetcher = BDGFetcher(sample_config, cache)

        csv_content = (
            "material_id,formula,voltage,capacity\n"
            "bdg-001,LiCoO2,3.9,140\n"
            "bdg-002,LiFePO4,3.4,170\n"
        )
        with patch("cathode_ml.data.bdg_fetcher.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = csv_content
            mock_response.raise_for_status = MagicMock()
            mock_requests.get.return_value = mock_response

            result = fetcher.fetch()

        assert len(result) == 2
        assert all(isinstance(r, MaterialRecord) for r in result)
        assert result[0].source == "battery_data_genome"
        assert result[0].formula == "LiCoO2"
        assert result[0].voltage == 3.9
        assert result[1].capacity == 170.0

    def test_download_failure_returns_empty_with_warning(self, sample_config, tmp_cache_dir, caplog):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        fetcher = BDGFetcher(sample_config, cache)

        with patch("cathode_ml.data.bdg_fetcher.requests") as mock_requests:
            mock_requests.get.side_effect = Exception("Connection refused")

            with caplog.at_level(logging.WARNING):
                result = fetcher.fetch()

        assert result == []
        assert "warning" in caplog.text.lower() or "error" in caplog.text.lower() or "failed" in caplog.text.lower()

    def test_empty_csv_returns_empty_list(self, sample_config, tmp_cache_dir):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        fetcher = BDGFetcher(sample_config, cache)

        with patch("cathode_ml.data.bdg_fetcher.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "material_id,formula,voltage\n"  # headers only
            mock_response.raise_for_status = MagicMock()
            mock_requests.get.return_value = mock_response

            result = fetcher.fetch()

        assert result == []

    def test_structure_dict_is_empty_for_text_mined(self, sample_config, tmp_cache_dir):
        from cathode_ml.data.bdg_fetcher import BDGFetcher
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        fetcher = BDGFetcher(sample_config, cache)

        csv_content = "material_id,formula,voltage,capacity\nbdg-001,LiCoO2,3.9,140\n"
        with patch("cathode_ml.data.bdg_fetcher.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = csv_content
            mock_response.raise_for_status = MagicMock()
            mock_requests.get.return_value = mock_response

            result = fetcher.fetch()

        assert result[0].structure_dict == {}
