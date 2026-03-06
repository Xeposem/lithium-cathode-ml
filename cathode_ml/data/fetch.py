"""CLI orchestrator for fetching and cleaning cathode material data.

Runs all enabled fetchers (Materials Project, OQMD, Battery Data Genome),
merges results, applies the cleaning pipeline, and saves processed data
with a cleaning log.

Usage:
    python -m cathode_ml.data.fetch
    python -m cathode_ml.data.fetch --config configs/data.yaml --force-refresh
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


def main() -> None:
    """Run the full data fetching and cleaning pipeline.

    1. Load config and set seeds
    2. Fetch from each enabled source
    3. Merge all records
    4. Run cleaning pipeline
    5. Save processed data and cleaning log
    """
    parser = argparse.ArgumentParser(
        description="Fetch and clean cathode material data from multiple sources"
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to YAML config file (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cache and re-download all data",
    )
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    set_seeds(config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    cache = DataCache(config["cache"]["directory"])
    all_records = []

    # Fetch from each enabled source
    sources = config.get("data_sources", {})

    if sources.get("materials_project", {}).get("enabled", False):
        try:
            from cathode_ml.data.mp_fetcher import MPFetcher

            mp = MPFetcher(config, cache)
            mp_records = mp.fetch(force_refresh=args.force_refresh)
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
            oqmd_records = oqmd.fetch(force_refresh=args.force_refresh)
            all_records.extend(oqmd_records)
            logger.info(f"OQMD: {len(oqmd_records)} records")
        except ImportError:
            logger.warning("OQMDFetcher not available -- skipping OQMD")
        except Exception as e:
            logger.error(f"OQMD fetch failed: {e}")

    if sources.get("battery_data_genome", {}).get("enabled", False):
        try:
            from cathode_ml.data.bdg_fetcher import BDGFetcher

            bdg = BDGFetcher(config, cache)
            bdg_records = bdg.fetch(force_refresh=args.force_refresh)
            all_records.extend(bdg_records)
            logger.info(f"Battery Data Genome: {len(bdg_records)} records")
        except ImportError:
            logger.warning("BDGFetcher not available -- skipping BDG")
        except Exception as e:
            logger.error(f"BDG fetch failed: {e}")

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


if __name__ == "__main__":
    main()
