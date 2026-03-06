"""File-based data cache with metadata tracking.

Prevents duplicate API calls by caching responses as JSON files
with timestamp and metadata for reproducibility tracking (DATA-04).
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class DataCache:
    """File-based cache for API responses and processed data.

    Each cached item is stored as a JSON file containing:
    - timestamp: when the data was cached
    - metadata: source info, query params, db version, etc.
    - data: the actual cached data

    Usage:
        cache = DataCache("data/raw")
        key = cache.cache_key("mp", {"elements": ["Li"]})
        if not cache.has(key):
            data = fetch_from_api(...)
            cache.save(key, data, metadata={"version": "2025.10"})
        data = cache.load(key)
    """

    def __init__(self, cache_dir: str) -> None:
        """Initialize cache with the given directory.

        Creates the directory if it does not exist.

        Args:
            cache_dir: Path to the cache directory.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, source: str, params: dict) -> str:
        """Generate a deterministic cache key from source and params.

        Args:
            source: Data source name (e.g., "mp", "oqmd").
            params: Query parameters dict.

        Returns:
            String key in format "{source}_{md5_hash}".
        """
        param_str = json.dumps(params, sort_keys=True)
        digest = hashlib.md5(param_str.encode()).hexdigest()
        return f"{source}_{digest}"

    def has(self, key: str) -> bool:
        """Check if a cache entry exists.

        Args:
            key: Cache key to check.

        Returns:
            True if cached data exists for this key.
        """
        return (self.cache_dir / f"{key}.json").exists()

    def save(self, key: str, data: dict, metadata: Optional[dict] = None) -> None:
        """Save data to cache with timestamp and metadata.

        Args:
            key: Cache key.
            data: Data dict to cache.
            metadata: Optional metadata dict (source info, version, etc.).
        """
        payload = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "data": data,
        }
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, key: str) -> dict:
        """Load cached data for the given key.

        Returns only the data field, not the wrapper with timestamp/metadata.

        Args:
            key: Cache key to load.

        Returns:
            The cached data dict.

        Raises:
            FileNotFoundError: If no cache entry exists for this key.
        """
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            raise FileNotFoundError(f"No cache entry for key: {key}")
        with open(cache_file) as f:
            payload = json.load(f)
        return payload["data"]

    def clear(self, key: str) -> None:
        """Remove a cache entry if it exists.

        Args:
            key: Cache key to remove.
        """
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
