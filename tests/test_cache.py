"""Tests for cathode_ml.data.cache module."""

import json

import pytest


class TestDataCacheInit:
    """Tests for DataCache initialization."""

    def test_creates_directory_if_not_exists(self, tmp_path):
        """DataCache creates the cache directory if it does not exist."""
        from cathode_ml.data.cache import DataCache

        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        cache = DataCache(str(cache_dir))
        assert cache_dir.exists()

    def test_works_with_existing_directory(self, tmp_cache_dir):
        """DataCache works when directory already exists."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        assert tmp_cache_dir.exists()


class TestCacheKey:
    """Tests for cache_key generation."""

    def test_deterministic_same_inputs(self, tmp_cache_dir):
        """cache_key returns the same string for identical source + params."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        key1 = cache.cache_key("mp", {"elements": ["Li"]})
        key2 = cache.cache_key("mp", {"elements": ["Li"]})
        assert key1 == key2

    def test_different_source_different_key(self, tmp_cache_dir):
        """Different source names produce different keys."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        key1 = cache.cache_key("mp", {"elements": ["Li"]})
        key2 = cache.cache_key("oqmd", {"elements": ["Li"]})
        assert key1 != key2

    def test_different_params_different_key(self, tmp_cache_dir):
        """Different params produce different keys."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        key1 = cache.cache_key("mp", {"elements": ["Li"]})
        key2 = cache.cache_key("mp", {"elements": ["Na"]})
        assert key1 != key2

    def test_key_contains_source_prefix(self, tmp_cache_dir):
        """cache_key starts with the source name."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        key = cache.cache_key("mp", {"elements": ["Li"]})
        assert key.startswith("mp_")


class TestCacheHas:
    """Tests for has() method."""

    def test_returns_false_for_missing_key(self, tmp_cache_dir):
        """has() returns False when key has not been saved."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        assert cache.has("nonexistent_key") is False

    def test_returns_true_after_save(self, tmp_cache_dir):
        """has() returns True after data is saved with that key."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        cache.save("test_key", {"value": 42})
        assert cache.has("test_key") is True


class TestCacheSaveLoad:
    """Tests for save() and load() round-trip."""

    def test_round_trip_preserves_data(self, tmp_cache_dir):
        """Saving and loading returns the original data dict."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        data = {"materials": [{"id": "mp-1", "formula": "LiCoO2"}], "count": 1}
        cache.save("test_key", data)

        loaded = cache.load("test_key")
        assert loaded == data

    def test_load_returns_data_not_wrapper(self, tmp_cache_dir):
        """load() returns just the data, not the wrapper with timestamp/metadata."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        cache.save("test_key", {"value": 99}, metadata={"source": "test"})

        loaded = cache.load("test_key")
        assert loaded == {"value": 99}
        assert "timestamp" not in loaded
        assert "metadata" not in loaded

    def test_save_writes_valid_json(self, tmp_cache_dir):
        """Saved cache files are valid JSON on disk."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        cache.save("test_key", {"value": 1})

        cache_file = tmp_cache_dir / "test_key.json"
        assert cache_file.exists()

        with open(cache_file) as f:
            raw = json.load(f)

        assert "timestamp" in raw
        assert "metadata" in raw
        assert "data" in raw
        assert raw["data"] == {"value": 1}

    def test_save_with_metadata(self, tmp_cache_dir):
        """Saved file includes the provided metadata."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        cache.save("test_key", {"value": 1}, metadata={"version": "2025.10"})

        cache_file = tmp_cache_dir / "test_key.json"
        with open(cache_file) as f:
            raw = json.load(f)

        assert raw["metadata"] == {"version": "2025.10"}

    def test_load_nonexistent_key_raises(self, tmp_cache_dir):
        """load() raises FileNotFoundError for a key that was never saved."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        with pytest.raises(FileNotFoundError):
            cache.load("nonexistent_key")

    def test_overwrite_on_duplicate_save(self, tmp_cache_dir):
        """Saving the same key twice overwrites the previous data."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        cache.save("test_key", {"version": 1})
        cache.save("test_key", {"version": 2})

        loaded = cache.load("test_key")
        assert loaded == {"version": 2}


class TestCacheClear:
    """Tests for clear() method."""

    def test_clear_removes_cached_file(self, tmp_cache_dir):
        """clear() removes the cache file so has() returns False."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        cache.save("test_key", {"value": 1})
        assert cache.has("test_key") is True

        cache.clear("test_key")
        assert cache.has("test_key") is False

    def test_clear_nonexistent_does_not_raise(self, tmp_cache_dir):
        """clear() on a nonexistent key does not raise."""
        from cathode_ml.data.cache import DataCache

        cache = DataCache(str(tmp_cache_dir))
        cache.clear("nonexistent_key")  # Should not raise
