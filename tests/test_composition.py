"""Tests for Magpie composition featurization."""

import numpy as np
import pytest

from cathode_ml.features.composition import featurize_compositions


class TestMagpieFeatures:
    """Test Magpie descriptor featurization."""

    def test_magpie_features_shape(self):
        """featurize_compositions returns ndarray of shape (2, 132) for two formulas."""
        X, labels = featurize_compositions(["LiCoO2", "LiFePO4"])
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 2
        assert X.shape[1] == 132
        assert len(labels) == 132

    def test_magpie_feature_labels(self):
        """Returns list of 132 feature label strings."""
        _, labels = featurize_compositions(["LiCoO2"])
        assert isinstance(labels, list)
        assert all(isinstance(lbl, str) for lbl in labels)
        assert len(labels) == 132

    def test_nan_handling(self):
        """Single-element compositions produce finite values after imputation."""
        X, _ = featurize_compositions(["Li", "LiCoO2"])
        # Li is single-element, some Magpie stats (e.g., range, std) are NaN
        # After imputation, no NaN should remain
        assert np.all(np.isfinite(X)), f"NaN found in output: {np.argwhere(~np.isfinite(X))}"

    def test_all_nan_column_dropped(self):
        """If a feature column is entirely NaN, it is dropped and feature count < 132."""
        # When all formulas are single-element, many columns will be all-NaN
        X, labels = featurize_compositions(["Li", "Fe", "Co"])
        # Single-element formulas have NaN for range/std/etc -- those columns should be dropped
        assert X.shape[1] < 132
        assert len(labels) == X.shape[1]
        assert np.all(np.isfinite(X))

    def test_empty_input(self):
        """featurize_compositions([]) returns empty array with correct dimensions."""
        X, labels = featurize_compositions([])
        assert isinstance(X, np.ndarray)
        assert X.shape == (0, 0)
        assert labels == []
