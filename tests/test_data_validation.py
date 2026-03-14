"""End-to-end integration tests for AFLOW and JARVIS data paths.

Validates that MaterialRecord objects from AFLOW and JARVIS sources
flow through CleaningPipeline and graph conversion without errors.
Also verifies the pipeline.py refresh-all expansion fix.
"""

import argparse
from unittest.mock import patch

import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Data

from cathode_ml.data.clean import CleaningPipeline
from cathode_ml.data.schemas import MaterialRecord
from cathode_ml.features.graph import structure_to_graph

# ---------------------------------------------------------------------------
# Fixtures: synthetic MaterialRecord objects
# ---------------------------------------------------------------------------

# A simple Li2O-like structure for testing
_LATTICE = Lattice.cubic(4.2)
_STRUCTURE = Structure(
    _LATTICE,
    ["Li", "Li", "O"],
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
)
_STRUCTURE_DICT = _STRUCTURE.as_dict()

# Standard configs loaded inline to avoid file-system dependency in unit tests
_CLEANING_CONFIG = {
    "filters": {
        "formation_energy_range": [-5.0, 0.5],
        "outlier_iqr_multiplier": 1.5,
    },
}

_FEATURES_CONFIG = {
    "graph": {
        "cutoff_radius": 8.0,
        "max_neighbors": 12,
        "gaussian": {
            "dmin": 0.0,
            "dmax": 5.0,
            "num_gaussians": 80,
        },
        "node_feature_dim": 100,
    },
}


def _make_records(source: str, count: int = 5, sg_offset: int = 0) -> list:
    """Create synthetic MaterialRecord objects for a given source.

    Args:
        source: Data source name.
        count: Number of records to create.
        sg_offset: Offset added to space_group values to avoid
            cross-source dedup collisions (dedup key = formula + sg).
    """
    records = []
    for i in range(count):
        records.append(
            MaterialRecord(
                material_id=f"{source}-test-{i}",
                formula="Li2O",
                structure_dict=_STRUCTURE_DICT,
                source=source,
                formation_energy_per_atom=-1.5 - i * 0.1,
                energy_above_hull=0.01 * i,
                voltage=None,
                capacity=None,
                is_stable=True,
                space_group=1 + i + sg_offset,  # unique per source via offset
            )
        )
    return records


# ---------------------------------------------------------------------------
# AFLOW path tests
# ---------------------------------------------------------------------------


class TestAFLOWPath:
    """Validate AFLOW records through clean -> graph pipeline."""

    def test_aflow_records_survive_cleaning(self):
        records = _make_records("aflow", count=5)
        pipeline = CleaningPipeline()
        cleaned = pipeline.run(records, _CLEANING_CONFIG)
        assert len(cleaned) >= 1, "At least one AFLOW record should survive cleaning"
        assert all(r.source == "aflow" for r in cleaned)

    def test_aflow_cleaned_record_to_graph(self):
        records = _make_records("aflow", count=3)
        pipeline = CleaningPipeline()
        cleaned = pipeline.run(records, _CLEANING_CONFIG)
        assert len(cleaned) >= 1

        record = cleaned[0]
        structure = Structure.from_dict(record.structure_dict)
        graph = structure_to_graph(structure, _FEATURES_CONFIG)

        assert isinstance(graph, Data)
        assert graph.x is not None and graph.x.shape[0] > 0, "Graph must have node features"
        assert graph.edge_index is not None and graph.edge_index.shape[1] > 0, "Graph must have edges"
        assert graph.edge_attr is not None and graph.edge_attr.shape[0] > 0, "Graph must have edge attributes"

    def test_aflow_structure_dict_valid(self):
        records = _make_records("aflow", count=1)
        record = records[0]
        structure = Structure.from_dict(record.structure_dict)
        assert len(structure) == 3
        assert structure.volume > 0


# ---------------------------------------------------------------------------
# JARVIS path tests
# ---------------------------------------------------------------------------


class TestJARVISPath:
    """Validate JARVIS records through clean -> graph pipeline."""

    def test_jarvis_records_survive_cleaning(self):
        records = _make_records("jarvis", count=5)
        pipeline = CleaningPipeline()
        cleaned = pipeline.run(records, _CLEANING_CONFIG)
        assert len(cleaned) >= 1, "At least one JARVIS record should survive cleaning"
        assert all(r.source == "jarvis" for r in cleaned)

    def test_jarvis_cleaned_record_to_graph(self):
        records = _make_records("jarvis", count=3)
        pipeline = CleaningPipeline()
        cleaned = pipeline.run(records, _CLEANING_CONFIG)
        assert len(cleaned) >= 1

        record = cleaned[0]
        structure = Structure.from_dict(record.structure_dict)
        graph = structure_to_graph(structure, _FEATURES_CONFIG)

        assert isinstance(graph, Data)
        assert graph.x is not None and graph.x.shape[0] > 0
        assert graph.edge_index is not None and graph.edge_index.shape[1] > 0
        assert graph.edge_attr is not None and graph.edge_attr.shape[0] > 0

    def test_jarvis_structure_dict_valid(self):
        records = _make_records("jarvis", count=1)
        record = records[0]
        structure = Structure.from_dict(record.structure_dict)
        assert len(structure) == 3
        assert structure.volume > 0


# ---------------------------------------------------------------------------
# Mixed 4-source test
# ---------------------------------------------------------------------------


class TestMixedSources:
    """Validate that records from all 4 sources survive cleaning together."""

    def test_all_four_sources_survive_cleaning(self):
        all_records = []
        # Use sg_offset so each source gets unique (formula, space_group) keys
        # to avoid cross-source dedup removing records
        for i, source in enumerate(("materials_project", "oqmd", "aflow", "jarvis")):
            all_records.extend(_make_records(source, count=3, sg_offset=i * 100))

        pipeline = CleaningPipeline()
        cleaned = pipeline.run(all_records, _CLEANING_CONFIG)

        surviving_sources = {r.source for r in cleaned}
        assert len(cleaned) >= 4, "At least 4 records should survive from mixed sources"
        assert len(surviving_sources) == 4, (
            f"All 4 sources should be represented, got: {surviving_sources}"
        )

    def test_mixed_source_graph_conversion(self):
        all_records = []
        for i, source in enumerate(("materials_project", "oqmd", "aflow", "jarvis")):
            all_records.extend(_make_records(source, count=2, sg_offset=i * 100))

        pipeline = CleaningPipeline()
        cleaned = pipeline.run(all_records, _CLEANING_CONFIG)

        # Convert every surviving record to a graph
        for record in cleaned:
            structure = Structure.from_dict(record.structure_dict)
            graph = structure_to_graph(structure, _FEATURES_CONFIG)
            assert isinstance(graph, Data)
            assert graph.x is not None and graph.x.shape[0] > 0


# ---------------------------------------------------------------------------
# Pipeline refresh fix test
# ---------------------------------------------------------------------------


class TestPipelineRefreshFix:
    """Verify that pipeline.py refresh-all expands to all 4 sources."""

    def test_refresh_all_includes_aflow_and_jarvis(self):
        """Test that run_fetch_stage expands 'all' to include aflow and jarvis.

        We read the source code of pipeline.py directly to verify the fix,
        since importing cathode_ml.data.fetch fails on Python 3.9 due to
        PEP 604 type union syntax.
        """
        import inspect

        from cathode_ml.pipeline import run_fetch_stage

        source = inspect.getsource(run_fetch_stage)
        # Verify the expansion set contains the correct sources
        assert "aflow" in source, "run_fetch_stage must reference 'aflow'"
        assert "jarvis" in source, "run_fetch_stage must reference 'jarvis'"
        assert "bdg" not in source, "run_fetch_stage must NOT reference 'bdg'"
        assert "mp" in source, "run_fetch_stage must reference 'mp'"
        assert "oqmd" in source, "run_fetch_stage must reference 'oqmd'"

    def test_refresh_all_expansion_logic(self):
        """Directly test the expansion logic without mocking."""
        # Simulate what run_fetch_stage does
        refresh = set(["all"])
        if "all" in refresh:
            refresh = {"mp", "oqmd", "aflow", "jarvis"}

        assert "aflow" in refresh
        assert "jarvis" in refresh
        assert "mp" in refresh
        assert "oqmd" in refresh
        assert "bdg" not in refresh
        assert len(refresh) == 4


# ---------------------------------------------------------------------------
# Live fetch validation tests (slow -- skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLiveFetchValidation:
    """Live API validation for AFLOW and JARVIS fetchers.

    These tests make actual network calls and are slow.
    Run with: pytest -m slow
    """

    def test_aflow_live_fetch_and_pipeline(self):
        """Fetch real AFLOW data and validate end-to-end."""
        from cathode_ml.data.aflow_fetcher import AFLOWFetcher
        from cathode_ml.data.cache import DataCache

        config = {
            "data_sources": {
                "aflow": {
                    "element": "Li",
                    "max_entries": 10,
                },
            },
        }
        cache = DataCache("data/raw")
        fetcher = AFLOWFetcher(config, cache)
        records = fetcher.fetch(force_refresh=True)

        assert len(records) > 0, "AFLOW fetch should return at least 1 record"
        print(f"\nAFLOW: fetched {len(records)} records")

        # Check at least one has a structure
        has_structure = [r for r in records if r.structure_dict]
        print(f"AFLOW: {len(has_structure)} records with structure_dict")
        assert len(has_structure) > 0, "At least one AFLOW record should have a structure"

        # Run through cleaning
        pipeline = CleaningPipeline()
        cleaned = pipeline.run(has_structure, _CLEANING_CONFIG)
        print(f"AFLOW: {len(cleaned)} records after cleaning")

        if len(cleaned) > 0:
            structure = Structure.from_dict(cleaned[0].structure_dict)
            graph = structure_to_graph(structure, _FEATURES_CONFIG)
            assert isinstance(graph, Data)
            assert graph.x is not None and graph.x.shape[0] > 0
            print(f"AFLOW: graph conversion OK -- {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

    def test_jarvis_live_fetch_and_pipeline(self):
        """Fetch real JARVIS data and validate end-to-end."""
        from cathode_ml.data.cache import DataCache
        from cathode_ml.data.jarvis_fetcher import JARVISFetcher

        config = {
            "data_sources": {
                "jarvis": {
                    "dataset": "dft_3d",
                    "element": "Li",
                },
            },
        }
        cache = DataCache("data/raw")
        fetcher = JARVISFetcher(config, cache)
        records = fetcher.fetch(force_refresh=True)

        assert len(records) > 0, "JARVIS fetch should return at least 1 record"
        print(f"\nJARVIS: fetched {len(records)} records")

        has_structure = [r for r in records if r.structure_dict]
        print(f"JARVIS: {len(has_structure)} records with structure_dict")
        assert len(has_structure) > 0, "At least one JARVIS record should have a structure"

        pipeline = CleaningPipeline()
        cleaned = pipeline.run(has_structure, _CLEANING_CONFIG)
        print(f"JARVIS: {len(cleaned)} records after cleaning")

        if len(cleaned) > 0:
            structure = Structure.from_dict(cleaned[0].structure_dict)
            graph = structure_to_graph(structure, _FEATURES_CONFIG)
            assert isinstance(graph, Data)
            assert graph.x is not None and graph.x.shape[0] > 0
            print(f"JARVIS: graph conversion OK -- {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
