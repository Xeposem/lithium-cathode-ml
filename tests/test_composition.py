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
        # Matminer v0.9.3 does not produce all-NaN columns for single-element
        # formulas. We test the dropping logic by patching the featurizer output
        # to inject an all-NaN column, verifying it gets removed.
        from unittest.mock import patch, MagicMock
        import pandas as pd
        from pymatgen.core import Composition

        original_featurize = None

        def patched_featurize(df, col_id, ignore_errors=True):
            """Call original featurizer then inject an all-NaN column."""
            from matminer.featurizers.composition import ElementProperty
            feat = ElementProperty.from_preset("magpie")
            result = feat.featurize_dataframe(df.copy(), col_id=col_id, ignore_errors=ignore_errors)
            # Inject an all-NaN column at position of first feature
            first_label = feat.feature_labels()[0]
            result[first_label] = np.nan
            return result

        with patch(
            "cathode_ml.features.composition.ElementProperty"
        ) as MockEP:
            from matminer.featurizers.composition import ElementProperty as RealEP
            mock_feat = RealEP.from_preset("magpie")
            mock_feat.featurize_dataframe = patched_featurize
            MockEP.from_preset.return_value = mock_feat

            X, labels = featurize_compositions(["LiCoO2", "LiFePO4"])

        # One column was all-NaN, so should be dropped
        assert X.shape[1] == 131
        assert len(labels) == 131
        assert np.all(np.isfinite(X))

    def test_empty_input(self):
        """featurize_compositions([]) returns empty array with correct dimensions."""
        X, labels = featurize_compositions([])
        assert isinstance(X, np.ndarray)
        assert X.shape == (0, 0)
        assert labels == []
