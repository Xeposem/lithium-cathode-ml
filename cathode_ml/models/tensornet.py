"""TensorNet model wrapper with lazy matgl imports.

Provides functions for constructing TensorNet models from config,
extracting state dicts, and running predictions. TensorNet is an
O(3)-equivariant tensor network architecture trained from scratch
(no pretrained property prediction models available).

All matgl imports are lazy (inside function bodies) so the rest
of the package works without matgl installed.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_INSTALL_MSG = (
    "matgl>=2.0.0 is required for TensorNet training. "
    "Install with: pip install 'matgl>=2.0.0'"
)


def _import_matgl():
    """Lazy import of matgl with helpful error message."""
    try:
        import matgl
    except ImportError:
        raise ImportError(_INSTALL_MSG)
    return matgl


def build_tensornet_from_config(model_config: dict, element_types: list):
    """Construct a TensorNet model from a configuration dictionary.

    TensorNet has no pretrained property prediction models, so it is
    always built from scratch using architecture parameters in
    *model_config*.

    Args:
        model_config: Dictionary of architecture parameters. Supported
            keys (with defaults): ``units`` (64), ``nblocks`` (2),
            ``num_rbf`` (32), ``cutoff`` (5.0), ``rbf_type``
            ("Gaussian"), ``readout_type`` ("weighted_atom"),
            ``activation_type`` ("swish"),
            ``equivariance_invariance_group`` ("O(3)"),
            ``is_intensive`` (True), ``ntargets`` (1).
        element_types: List of element symbols present in the dataset,
            e.g. ``["Li", "Co", "O"]``.

    Returns:
        A ``matgl.models.TensorNet`` instance ready for training.

    Raises:
        ImportError: If matgl is not installed.
    """
    _import_matgl()
    from matgl.models import TensorNet

    params = {
        "element_types": element_types,
        "units": model_config.get("units", 64),
        "nblocks": model_config.get("nblocks", 2),
        "num_rbf": model_config.get("num_rbf", 32),
        "cutoff": model_config.get("cutoff", 5.0),
        "rbf_type": model_config.get("rbf_type", "Gaussian"),
        "readout_type": model_config.get("readout_type", "weighted_atom"),
        "activation_type": model_config.get("activation_type", "swish"),
        "equivariance_invariance_group": model_config.get(
            "equivariance_invariance_group", "O(3)"
        ),
        "is_intensive": model_config.get("is_intensive", True),
        "ntargets": model_config.get("ntargets", 1),
    }

    logger.info(
        "Building TensorNet: units=%d, nblocks=%d, cutoff=%.1f, group=%s",
        params["units"],
        params["nblocks"],
        params["cutoff"],
        params["equivariance_invariance_group"],
    )

    model = TensorNet(**params)
    logger.info("TensorNet constructed with %d element types", len(element_types))
    return model


def get_tensornet_state_dict(model) -> dict:
    """Extract the inner torch Module state dict from a matgl model.

    Lightning checkpoints (``.ckpt``) differ from the project standard
    ``.pt`` format.  This function extracts the raw ``state_dict()``
    suitable for ``torch.save()``.

    Args:
        model: A TensorNet model returned by
            :func:`build_tensornet_from_config`.

    Returns:
        An ``OrderedDict`` of parameter tensors.
    """
    return model.model.state_dict()


def predict_with_tensornet(model, structures: list) -> List[float]:
    """Run predictions on a list of pymatgen Structure objects.

    Args:
        model: A TensorNet model (matgl potential wrapper).
        structures: List of ``pymatgen.core.Structure`` objects.

    Returns:
        List of float predictions, one per structure.

    Raises:
        ImportError: If matgl is not installed.
    """
    # Ensure matgl is available (needed for internal tensor ops)
    _import_matgl()

    predictions = []
    for struct in structures:
        pred = model.predict_structure(struct)
        predictions.append(float(pred))
    return predictions
