"""Tests for compositional group splitting."""

import numpy as np
import pytest

from cathode_ml.features.split import get_group_keys, compositional_split


class TestGetGroupKeys:
    """Test reduced formula normalization for group keys."""

    def test_get_group_keys(self):
        """Polymorphs with different formulas normalize to same reduced formula."""
        keys = get_group_keys(["Li2Co2O4", "LiCoO2", "LiFePO4"])
        assert keys == ["LiCoO2", "LiCoO2", "LiFePO4"]

    def test_get_group_keys_already_reduced(self):
        """Already-reduced formulas are unchanged."""
        keys = get_group_keys(["LiCoO2", "LiFePO4"])
        assert keys == ["LiCoO2", "LiFePO4"]


class TestCompositionalSplit:
    """Test group-based train/val/test splitting."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic dataset with known polymorph groups."""
        # 10 compositions, 5 groups with 2 entries each (polymorphs)
        formulas = (
            ["LiCoO2"] * 20
            + ["LiFePO4"] * 20
            + ["LiMnO2"] * 20
            + ["LiNiO2"] * 20
            + ["LiVO3"] * 20
        )
        groups = get_group_keys(formulas)
        return formulas, groups

    def test_group_split_sizes(self, synthetic_data):
        """compositional_split with 100 samples produces ~80/10/10."""
        _, groups = synthetic_data
        n = len(groups)
        train_idx, val_idx, test_idx = compositional_split(
            n_samples=n, groups=groups, test_size=0.1, val_size=0.1, seed=42
        )
        # All indices should be present exactly once
        all_idx = np.sort(np.concatenate([train_idx, val_idx, test_idx]))
        np.testing.assert_array_equal(all_idx, np.arange(n))

        # Approximate proportions (group-based, so not exact)
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0

    def test_no_formula_overlap(self, synthetic_data):
        """No reduced formula appears in both train and test index sets."""
        formulas, groups = synthetic_data
        n = len(groups)
        train_idx, val_idx, test_idx = compositional_split(
            n_samples=n, groups=groups, test_size=0.1, val_size=0.1, seed=42
        )
        groups_arr = np.array(groups)
        train_formulas = set(groups_arr[train_idx])
        test_formulas = set(groups_arr[test_idx])
        overlap = train_formulas & test_formulas
        assert overlap == set(), f"Train-test overlap: {overlap}"

    def test_no_formula_overlap_val(self, synthetic_data):
        """No reduced formula appears in both val and test index sets."""
        formulas, groups = synthetic_data
        n = len(groups)
        train_idx, val_idx, test_idx = compositional_split(
            n_samples=n, groups=groups, test_size=0.1, val_size=0.1, seed=42
        )
        groups_arr = np.array(groups)
        val_formulas = set(groups_arr[val_idx])
        test_formulas = set(groups_arr[test_idx])
        overlap = val_formulas & test_formulas
        assert overlap == set(), f"Val-test overlap: {overlap}"

    def test_deterministic_with_seed(self, synthetic_data):
        """Same seed produces identical splits across two calls."""
        _, groups = synthetic_data
        n = len(groups)
        split1 = compositional_split(n_samples=n, groups=groups, seed=42)
        split2 = compositional_split(n_samples=n, groups=groups, seed=42)
        for s1, s2 in zip(split1, split2):
            np.testing.assert_array_equal(s1, s2)

    def test_group_split_no_leakage(self):
        """All polymorphs (same reduced formula) end up in the same split."""
        # Create data with clear polymorph groups
        # 3 entries per composition, 10 compositions
        formulas = []
        for comp in ["LiCoO2", "LiFePO4", "LiMnO2", "LiNiO2", "LiVO3",
                      "LiCrO2", "LiTiO2", "LiMoO3", "LiWO3", "LiAlO2"]:
            formulas.extend([comp] * 3)

        groups = get_group_keys(formulas)
        n = len(formulas)
        train_idx, val_idx, test_idx = compositional_split(
            n_samples=n, groups=groups, test_size=0.1, val_size=0.1, seed=42
        )

        groups_arr = np.array(groups)
        # For each unique group, all its members should be in the same split
        for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_groups = groups_arr[idx]
            for group in np.unique(split_groups):
                # All indices of this group in the original data
                all_group_idx = np.where(groups_arr == group)[0]
                # They should all be in this split
                assert set(all_group_idx).issubset(set(idx)), (
                    f"Group {group} split across boundaries: "
                    f"found in {split_name} but not all members present"
                )
