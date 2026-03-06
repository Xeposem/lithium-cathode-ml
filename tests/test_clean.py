"""Tests for cleaning pipeline."""

import json
import logging
from pathlib import Path

import pytest

from cathode_ml.data.schemas import FilterRecord, MaterialRecord


def _make_record(
    material_id="mp-001",
    formula="LiCoO2",
    structure_dict=None,
    source="materials_project",
    formation_energy_per_atom=-2.1,
    energy_above_hull=0.0,
    voltage=3.9,
    capacity=140.0,
    is_stable=True,
    space_group=166,
):
    """Helper to create MaterialRecord with defaults."""
    if structure_dict is None:
        structure_dict = {
            "@module": "pymatgen.core.structure",
            "@class": "Structure",
            "charge": 0,
            "lattice": {
                "matrix": [
                    [2.8, 0.0, 0.0],
                    [-1.4, 2.4249, 0.0],
                    [0.0, 0.0, 14.05],
                ],
                "pbc": [True, True, True],
            },
            "sites": [
                {
                    "species": [{"element": "Li", "occu": 1}],
                    "abc": [0.0, 0.0, 0.5],
                    "xyz": [0.0, 0.0, 7.025],
                },
                {
                    "species": [{"element": "Co", "occu": 1}],
                    "abc": [0.0, 0.0, 0.0],
                    "xyz": [0.0, 0.0, 0.0],
                },
                {
                    "species": [{"element": "O", "occu": 1}],
                    "abc": [0.0, 0.0, 0.2604],
                    "xyz": [0.0, 0.0, 3.6586],
                },
            ],
        }
    return MaterialRecord(
        material_id=material_id,
        formula=formula,
        structure_dict=structure_dict,
        source=source,
        formation_energy_per_atom=formation_energy_per_atom,
        energy_above_hull=energy_above_hull,
        voltage=voltage,
        capacity=capacity,
        is_stable=is_stable,
        space_group=space_group,
    )


class TestValidateStructure:
    """Test structure validation."""

    def test_valid_structure_passes(self, sample_structure_dict):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        valid, reason = pipeline.validate_structure(sample_structure_dict)
        assert valid is True
        assert reason == ""

    def test_empty_dict_fails(self):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        valid, reason = pipeline.validate_structure({})
        assert valid is False
        assert "no structure" in reason.lower()

    def test_too_few_sites_fails(self):
        from cathode_ml.data.clean import CleaningPipeline

        # Structure with only 1 site
        structure_dict = {
            "@module": "pymatgen.core.structure",
            "@class": "Structure",
            "charge": 0,
            "lattice": {
                "matrix": [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]],
                "pbc": [True, True, True],
            },
            "sites": [
                {
                    "species": [{"element": "Li", "occu": 1}],
                    "abc": [0.0, 0.0, 0.0],
                    "xyz": [0.0, 0.0, 0.0],
                },
            ],
        }
        pipeline = CleaningPipeline()
        valid, reason = pipeline.validate_structure(structure_dict)
        assert valid is False
        assert "sites" in reason.lower() or "few" in reason.lower()


class TestApplyFilter:
    """Test filter application and logging."""

    def test_apply_filter_creates_filter_record(self):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        records = [
            _make_record(material_id="mp-001", formation_energy_per_atom=-2.0),
            _make_record(material_id="mp-002", formation_energy_per_atom=-1.0),
            _make_record(material_id="mp-003", formation_energy_per_atom=5.0),
        ]
        result = pipeline.apply_filter(
            records,
            lambda r: r.formation_energy_per_atom < 0,
            "negative_energy",
            "Keep only negative formation energies",
        )
        assert len(result) == 2
        assert len(pipeline.log) == 1
        assert isinstance(pipeline.log[0], FilterRecord)
        assert pipeline.log[0].filter_name == "negative_energy"
        assert pipeline.log[0].count_before == 3
        assert pipeline.log[0].count_after == 2
        assert pipeline.log[0].count_removed == 1


class TestRemoveOutliers:
    """Test IQR-based outlier removal."""

    def test_removes_extreme_values(self):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        # Create records with mostly normal values and one extreme outlier
        records = []
        for i, e in enumerate([-2.0, -2.1, -1.9, -2.05, -1.95, -2.15, -1.85, -2.2, -1.8, 50.0]):
            records.append(_make_record(
                material_id=f"mp-{i:03d}",
                formation_energy_per_atom=e,
            ))

        result = pipeline.remove_outliers(records, "formation_energy_per_atom", iqr_multiplier=1.5)
        # The outlier (50.0) should be removed
        assert len(result) < len(records)
        assert all(r.formation_energy_per_atom != 50.0 for r in result)
        # Should have logged a FilterRecord
        assert len(pipeline.log) == 1

    def test_no_values_for_property_returns_all(self):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        records = [_make_record(formation_energy_per_atom=None)]
        result = pipeline.remove_outliers(records, "formation_energy_per_atom", iqr_multiplier=1.5)
        # All records returned when no values to compute IQR on
        assert len(result) == len(records)


class TestDeduplicate:
    """Test deduplication logic."""

    def test_deduplicate_prefers_mp_source(self):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        records = [
            _make_record(material_id="mp-001", formula="LiCoO2", source="materials_project", space_group=166),
            _make_record(material_id="oqmd-001", formula="LiCoO2", source="oqmd", space_group=166),
        ]
        result = pipeline.deduplicate(records)
        assert len(result) == 1
        assert result[0].source == "materials_project"
        assert len(pipeline.log) == 1

    def test_deduplicate_different_spacegroups_kept(self):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        records = [
            _make_record(material_id="mp-001", formula="LiCoO2", source="materials_project", space_group=166),
            _make_record(material_id="mp-002", formula="LiCoO2", source="materials_project", space_group=225),
        ]
        result = pipeline.deduplicate(records)
        assert len(result) == 2


class TestRunFullPipeline:
    """Test full pipeline run."""

    def test_run_applies_all_filters_and_produces_log(self):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        records = [
            _make_record(material_id="mp-001", formation_energy_per_atom=-2.0),
            _make_record(material_id="mp-002", formation_energy_per_atom=-1.5),
            _make_record(material_id="mp-003", formation_energy_per_atom=None, structure_dict={}),
        ]
        config = {
            "filters": {
                "min_sites": 2,
                "max_sites": 200,
                "formation_energy_range": [-5.0, 0.5],
                "required_properties": ["formation_energy_per_atom"],
                "remove_noble_gases": True,
                "outlier_iqr_multiplier": 1.5,
            }
        }
        result = pipeline.run(records, config)
        # At least some filters logged
        assert len(pipeline.log) > 0
        # All remaining records should be valid
        assert isinstance(result, list)


class TestSaveLog:
    """Test cleaning log persistence."""

    def test_save_log_creates_valid_json(self, tmp_path):
        from cathode_ml.data.clean import CleaningPipeline

        pipeline = CleaningPipeline()
        pipeline.log.append(FilterRecord(
            filter_name="test_filter",
            description="Test filter",
            rationale="Testing",
            count_before=10,
            count_after=8,
            count_removed=2,
        ))
        log_path = tmp_path / "cleaning_log.json"
        pipeline.save_log(str(log_path))

        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["filter_name"] == "test_filter"
        assert data[0]["count_removed"] == 2
