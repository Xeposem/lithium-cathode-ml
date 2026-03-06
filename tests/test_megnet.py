"""Tests for MEGNet model wrapper with lazy matgl imports.

Tests cover lazy import error handling, pretrained model loading,
available model listing, and state dict extraction. Tests requiring
matgl are skipped when the dependency is not installed.
"""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

HAS_MATGL = importlib.util.find_spec("matgl") is not None
skip_no_matgl = pytest.mark.skipif(not HAS_MATGL, reason="matgl not installed")


class TestLazyImportError:
    """MEGNet functions raise helpful ImportError when matgl is missing."""

    def test_lazy_import_error_load(self):
        """load_megnet_model raises ImportError with install instructions when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.megnet import load_megnet_model

            with pytest.raises(ImportError, match="pip install matgl==1.3.0 dgl==2.2.0"):
                load_megnet_model("MEGNet-MP-2018.6.1-Eform")

    def test_lazy_import_error_available(self):
        """get_available_megnet_models raises ImportError when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.megnet import get_available_megnet_models

            with pytest.raises(ImportError, match="pip install matgl==1.3.0 dgl==2.2.0"):
                get_available_megnet_models()

    def test_lazy_import_error_predict(self):
        """predict_with_megnet raises ImportError when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.megnet import predict_with_megnet

            with pytest.raises(ImportError, match="pip install matgl==1.3.0 dgl==2.2.0"):
                predict_with_megnet(MagicMock(), [])


@skip_no_matgl
class TestLoadPretrained:
    """MEGNet pretrained model loading via matgl."""

    def test_load_pretrained(self):
        """load_megnet_model returns a model object with cutoff attribute."""
        from cathode_ml.models.megnet import load_megnet_model

        model = load_megnet_model("MEGNet-MP-2018.6.1-Eform")
        assert hasattr(model, "cutoff"), "Loaded model should have a cutoff attribute"

    def test_get_available_models(self):
        """get_available_megnet_models returns list of MEGNet model strings."""
        from cathode_ml.models.megnet import get_available_megnet_models

        models = get_available_megnet_models()
        assert isinstance(models, list)
        assert len(models) > 0, "Should find at least one MEGNet model"
        for name in models:
            assert "MEGNet" in name, f"Model {name!r} should contain 'MEGNet'"

    def test_get_state_dict(self):
        """get_megnet_state_dict returns a dict-like object from loaded model."""
        from cathode_ml.models.megnet import (
            get_megnet_state_dict,
            load_megnet_model,
        )

        model = load_megnet_model("MEGNet-MP-2018.6.1-Eform")
        state_dict = get_megnet_state_dict(model)
        assert hasattr(state_dict, "keys"), "State dict should be dict-like"
        assert len(state_dict) > 0, "State dict should not be empty"
