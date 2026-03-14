"""CLI pipeline orchestrator for the full cathode ML workflow.

Orchestrates fetch -> featurize -> train -> evaluate with a single
command, supporting --stage, --skip-fetch, --skip-train, --models,
and --seed flags. Supports RF, XGBoost, CGCNN, M3GNet, and TensorNet
models. All heavy imports are lazy (inside stage functions) to keep
``--help`` fast.

Usage:
    python -m cathode_ml --list-stages
    python -m cathode_ml --stage fetch train
    python -m cathode_ml --stage train evaluate --models rf cgcnn
    python -m cathode_ml --skip-fetch --models rf cgcnn m3gnet
    python -m cathode_ml --skip-train --seed 123
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("cathode_ml.pipeline")

# Ordered list of (stage_name, description) — canonical stage definitions.
STAGES = [
    ("fetch", "Fetch and clean cathode data from all sources"),
    ("featurize", "Featurize data (currently inline during training)"),
    ("train", "Train selected models on processed data"),
    ("evaluate", "Generate comparison tables and plots"),
]

STAGE_NAMES = [name for name, _ in STAGES]


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
        "--list-stages",
        action="store_true",
        help="List available pipeline stages and exit",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=STAGE_NAMES,
        default=None,
        metavar="STAGE",
        help=(
            "Run only these stages (choose from: "
            + ", ".join(STAGE_NAMES)
            + "). Default: all stages."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rf", "xgb", "cgcnn", "m3gnet", "tensornet"],
        default=["rf", "xgb", "cgcnn", "m3gnet", "tensornet"],
        help="Models to train/evaluate (default: all five)",
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
    parser.add_argument(
        "--refresh",
        nargs="+",
        choices=["all", "mp", "oqmd", "aflow", "jarvis"],
        default=[],
        metavar="SOURCE",
        help="Force-refresh specific data sources (mp, oqmd, aflow, jarvis, or all)",
    )
    return parser


# ---------------------------------------------------------------------------
# Stage functions (lazy imports inside each)
# ---------------------------------------------------------------------------


def run_fetch_stage(args: argparse.Namespace) -> None:
    """Fetch and clean cathode data from all sources."""
    from cathode_ml.data.fetch import run_fetch  # noqa: C0415

    refresh = set(args.refresh)
    if "all" in refresh:
        refresh = {"mp", "oqmd", "bdg"}
    run_fetch(
        config_path=str(Path(args.config_dir) / "data.yaml"),
        refresh_sources=refresh,
    )


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

    # M3GNet
    if "m3gnet" in args.models:
        from cathode_ml.models.train_m3gnet import train_m3gnet  # noqa: C0415

        m3gnet_config = load_config(str(config_dir / "m3gnet.yaml"))
        logger.info("Training M3GNet models...")
        train_m3gnet(
            records=records,
            features_config=features_config,
            m3gnet_config=m3gnet_config,
            seed=args.seed,
        )

    # TensorNet
    if "tensornet" in args.models:
        from cathode_ml.models.train_tensornet import train_tensornet  # noqa: C0415

        tensornet_config = load_config(str(config_dir / "tensornet.yaml"))
        logger.info("Training TensorNet models...")
        train_tensornet(
            records=records,
            features_config=features_config,
            tensornet_config=tensornet_config,
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


def _resolve_stages(args: argparse.Namespace) -> list[tuple[str, object, bool]]:
    """Build the ordered stage list with run/skip flags.

    ``--stage`` takes priority: if provided, only the listed stages run.
    Otherwise the legacy ``--skip-fetch`` / ``--skip-train`` flags apply.
    """
    stage_funcs = {
        "fetch": ("Fetching Data", run_fetch_stage),
        "featurize": ("Featurizing", run_featurize_stage),
        "train": ("Training Models", run_train_stage),
        "evaluate": ("Evaluating", run_evaluate_stage),
    }

    selected = set(args.stage) if args.stage else None
    result: list[tuple[str, object, bool]] = []

    for stage_name in STAGE_NAMES:
        label, func = stage_funcs[stage_name]
        if selected is not None:
            should_run = stage_name in selected
        else:
            # Legacy skip flags
            if stage_name == "fetch":
                should_run = not args.skip_fetch
            elif stage_name == "train":
                should_run = not args.skip_train
            else:
                should_run = True
        result.append((label, func, should_run))

    return result


def list_stages() -> None:
    """Print available pipeline stages to stdout."""
    print("Available pipeline stages:\n")
    for i, (name, desc) in enumerate(STAGES, start=1):
        print(f"  {i}. {name:12s} - {desc}")
    print(f"\nUsage: python -m cathode_ml --stage {' '.join(STAGE_NAMES[:2])}")


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the pipeline stages in order, respecting skip/stage flags.

    Args:
        args: Parsed CLI arguments from build_parser().
    """
    stages = _resolve_stages(args)
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

    if args.list_stages:
        list_stages()
        return

    run_pipeline(args)


if __name__ == "__main__":
    main()
