"""CLI orchestrator for fetching and cleaning cathode material data.

Runs all enabled fetchers (Materials Project, OQMD, AFLOW, JARVIS),
merges results, applies the cleaning pipeline, and saves processed data
with a cleaning log.

Usage:
    python -m cathode_ml.data.fetch
    python -m cathode_ml.data.fetch --refresh mp aflow jarvis
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from cathode_ml.config import load_config, set_seeds
from cathode_ml.data.cache import DataCache
from cathode_ml.data.clean import CleaningPipeline

logger = logging.getLogger("cathode_ml.data.fetch")

# Canonical source keys used for --refresh and config lookup
SOURCE_KEYS = ("mp", "oqmd", "aflow", "jarvis")


def run_fetch(
    config_path: str = "configs/data.yaml",
    refresh_sources: set | None = None,
) -> None:
    """Run the full data fetching and cleaning pipeline.

    Args:
        config_path: Path to the data YAML config file.
        refresh_sources: Set of source keys to force-refresh (``"mp"``,
            ``"oqmd"``, ``"aflow"``, ``"jarvis"``).  ``None`` or empty
            means use cache for everything.
    """
    if refresh_sources is None:
        refresh_sources = set()

    config = load_config(config_path)
    set_seeds(config)

    cache = DataCache(config["cache"]["directory"])
    all_records = []

    # Fetch from each enabled source
    sources = config.get("data_sources", {})

    if sources.get("materials_project", {}).get("enabled", False):
        try:
            from cathode_ml.data.mp_fetcher import MPFetcher

            mp = MPFetcher(config, cache)
            mp_records = mp.fetch(force_refresh="mp" in refresh_sources)
            all_records.extend(mp_records)
            logger.info(f"Materials Project: {len(mp_records)} records")
        except ImportError:
            logger.warning("MPFetcher not available -- skipping Materials Project")
        except Exception as e:
            logger.error(f"Materials Project fetch failed: {e}")

    if sources.get("oqmd", {}).get("enabled", False):
        try:
            from cathode_ml.data.oqmd_fetcher import OQMDFetcher

            oqmd = OQMDFetcher(config, cache)
            oqmd_records = oqmd.fetch(force_refresh="oqmd" in refresh_sources)
            all_records.extend(oqmd_records)
            logger.info(f"OQMD: {len(oqmd_records)} records")
        except ImportError:
            logger.warning("OQMDFetcher not available -- skipping OQMD")
        except Exception as e:
            logger.error(f"OQMD fetch failed: {e}")

    if sources.get("aflow", {}).get("enabled", False):
        try:
            from cathode_ml.data.aflow_fetcher import AFLOWFetcher

            af = AFLOWFetcher(config, cache)
            af_records = af.fetch(force_refresh="aflow" in refresh_sources)
            all_records.extend(af_records)
            logger.info(f"AFLOW: {len(af_records)} records")
        except ImportError:
            logger.warning("aflow package not installed -- skipping AFLOW")
        except Exception as e:
            logger.error(f"AFLOW fetch failed: {e}")

    if sources.get("jarvis", {}).get("enabled", False):
        try:
            from cathode_ml.data.jarvis_fetcher import JARVISFetcher

            jv = JARVISFetcher(config, cache)
            jv_records = jv.fetch(force_refresh="jarvis" in refresh_sources)
            all_records.extend(jv_records)
            logger.info(f"JARVIS: {len(jv_records)} records")
        except ImportError:
            logger.warning("jarvis-tools not installed -- skipping JARVIS")
        except Exception as e:
            logger.error(f"JARVIS fetch failed: {e}")

    logger.info(f"Total records fetched: {len(all_records)}")

    # Clean
    pipeline = CleaningPipeline()
    cleaned = pipeline.run(all_records, config)

    # Save processed data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "materials.json", "w") as f:
        json.dump([asdict(r) for r in cleaned], f, indent=2)

    # Save cleaning log
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_log(str(logs_dir / "cleaning_log.json"))

    logger.info(f"Cleaned records: {len(cleaned)}, saved to data/processed/materials.json")
    logger.info(f"Cleaning log: data/logs/cleaning_log.json")


def main() -> None:
    """CLI entry point for standalone data fetching."""
    parser = argparse.ArgumentParser(
        description="Fetch and clean cathode material data from multiple sources"
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to YAML config file (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--refresh",
        nargs="+",
        choices=["all"] + list(SOURCE_KEYS),
        default=[],
        metavar="SOURCE",
        help="Force-refresh specific sources (mp, oqmd, aflow, jarvis, or all)",
    )
    args, _ = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    refresh = set(args.refresh)
    if "all" in refresh:
        refresh = set(SOURCE_KEYS)
    run_fetch(config_path=args.config, refresh_sources=refresh)


if __name__ == "__main__":
    main()
