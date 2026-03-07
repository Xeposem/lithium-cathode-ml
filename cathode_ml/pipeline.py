"""CLI pipeline orchestrator for the full cathode ML workflow.

Orchestrates fetch -> featurize -> train -> evaluate with a single
command, supporting --skip-fetch, --skip-train, --models, and --seed
flags. All heavy imports are lazy (inside stage functions) to keep
``--help`` fast.

Usage:
    python -m cathode_ml.pipeline
    python -m cathode_ml.pipeline --skip-fetch --models rf cgcnn
    python -m cathode_ml.pipeline --skip-train --seed 123
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("cathode_ml.pipeline")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the pipeline CLI.

    Returns:
        Configured ArgumentParser with --models, --skip-fetch,
        --skip-train, --seed, and --config-dir flags.
    """
    parser = argparse.ArgumentParser(
        prog="cathode-ml-pipeline",
        description="Run the full cathode ML pipeline end-to-end",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rf", "xgb", "cgcnn", "megnet"],
        default=["rf", "xgb", "cgcnn", "megnet"],
        help="Models to train/evaluate (default: all four)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching (use cached data)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (use saved checkpoints/results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing config YAML files (default: configs)",
    )
    return parser


# ---------------------------------------------------------------------------
# Stage functions (lazy imports inside each)
# ---------------------------------------------------------------------------


def run_fetch_stage(args: argparse.Namespace) -> None:
    """Fetch and clean cathode data from all sources."""
    from cathode_ml.data.fetch import main as fetch_main  # noqa: C0415

    fetch_main()


def run_featurize_stage(args: argparse.Namespace) -> None:
    """Log that featurization occurs inline during training.

    In the current architecture, featurization is performed by each
    training orchestrator (composition features for baselines, graph
    construction for GNNs). No separate featurization step is needed.
    """
    logger.info("Featurization is performed inline by each training orchestrator.")


def run_train_stage(args: argparse.Namespace) -> None:
    """Train selected models on processed data with separate config files."""
    import json as _json  # noqa: C0415

    from cathode_ml.config import load_config  # noqa: C0415
    from cathode_ml.data.schemas import MaterialRecord  # noqa: C0415

    config_dir = Path(args.config_dir)

    # Load separate config files (not monolithic data.yaml)
    features_config = load_config(str(config_dir / "features.yaml"))

    # Load processed records from JSON (not DataCache)
    processed_path = Path("data/processed/materials.json")
    with open(processed_path) as f:
        raw_records = _json.load(f)
    records = [MaterialRecord(**r) for r in raw_records]

    # Baseline models (RF, XGBoost)
    baseline_models = [m for m in args.models if m in ("rf", "xgb")]
    if baseline_models:
        from cathode_ml.models.baselines import run_baselines  # noqa: C0415

        baselines_config = load_config(str(config_dir / "baselines.yaml"))
        run_baselines(records, features_config, baselines_config, seed=args.seed)

    # CGCNN
    if "cgcnn" in args.models:
        from cathode_ml.models.train_cgcnn import train_cgcnn  # noqa: C0415

        cgcnn_config = load_config(str(config_dir / "cgcnn.yaml"))
        train_cgcnn(
            records=records,
            features_config=features_config,
            cgcnn_config=cgcnn_config,
            seed=args.seed,
        )

    # MEGNet
    if "megnet" in args.models:
        from cathode_ml.models.train_megnet import train_megnet  # noqa: C0415

        megnet_config = load_config(str(config_dir / "megnet.yaml"))
        train_megnet(
            records=records,
            features_config=features_config,
            megnet_config=megnet_config,
            seed=args.seed,
        )


def run_evaluate_stage(args: argparse.Namespace) -> None:
    """Run evaluation: generate comparison tables and plots (if available)."""
    from cathode_ml.evaluation.metrics import (  # noqa: C0415
        generate_all_tables,
        load_all_results,
    )

    results_base = "data/results"
    generate_all_tables(results_base)

    # Plots module may not exist yet (created in plan 05-02)
    try:
        from cathode_ml.evaluation.plots import (  # noqa: C0415
            apply_nature_style,
            plot_bar_comparison,
            plot_learning_curves,
        )

        apply_nature_style()
        all_results = load_all_results(results_base)
        figures_dir = str(Path(results_base) / "figures")
        plot_bar_comparison(all_results, str(Path(figures_dir) / "bar_comparison.png"))
        plot_learning_curves(results_base, str(Path(figures_dir) / "learning_curves.png"))
    except ImportError:
        logger.info("Plots module not available; skipping plot generation.")

    logger.info("Evaluation complete.")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the pipeline stages in order, respecting skip flags.

    Args:
        args: Parsed CLI arguments from build_parser().
    """
    stages: list[tuple[str, object, bool]] = [
        ("Fetching Data", run_fetch_stage, not args.skip_fetch),
        ("Featurizing", run_featurize_stage, True),
        ("Training Models", run_train_stage, not args.skip_train),
        ("Evaluating", run_evaluate_stage, True),
    ]

    total = len(stages)
    for i, (name, func, should_run) in enumerate(stages, start=1):
        if should_run:
            logger.info("=== Stage %d/%d: %s ===", i, total, name)
            func(args)
        else:
            logger.info("=== Stage %d/%d: %s (SKIPPED) ===", i, total, name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args and run the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
