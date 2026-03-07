"""CLI entry point for cathode ML evaluation.

Run with: ``python -m cathode_ml.evaluation``

Generates comparison tables, bar charts, and learning curves from
existing model result artifacts. Parity plots require prediction
arrays and are skipped unless prediction data is available.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate evaluation tables and figures for cathode ML models.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Path to results directory (default: data/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/figures",
        help="Path for figures output (default: data/results/figures)",
    )
    return parser


def main(results_dir: str = "data/results", output_dir: str = "data/results/figures") -> None:
    """Run full evaluation pipeline: tables, bar chart, learning curves.

    Args:
        results_dir: Path to results directory.
        output_dir: Path for figure output.
    """
    from cathode_ml.evaluation.metrics import (
        PROPERTIES,
        generate_all_tables,
        load_all_results,
    )
    from cathode_ml.evaluation.plots import (
        apply_nature_style,
        plot_bar_comparison,
        plot_learning_curves,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    apply_nature_style()

    # Generate comparison tables
    logger.info("Generating comparison tables...")
    generate_all_tables(results_dir)

    # Load results for plotting
    all_results = load_all_results(results_dir)

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parity plots require y_true/y_pred arrays not available in JSON results.
    # They will be generated when the pipeline runs inference (Plan 03).
    for prop in PROPERTIES:
        logger.info("=== Parity plot: %s === (skipped -- no prediction arrays)", prop)

    # Bar chart comparison
    if all_results:
        logger.info("Generating bar chart comparison...")
        plot_bar_comparison(all_results, str(out / "bar_comparison.png"))
    else:
        logger.warning("No results found; skipping bar chart.")

    # Learning curves
    logger.info("Generating learning curves...")
    plot_learning_curves(results_dir, str(out / "learning_curves.png"))

    logger.info("Evaluation complete. Figures saved to %s", output_dir)


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir)
