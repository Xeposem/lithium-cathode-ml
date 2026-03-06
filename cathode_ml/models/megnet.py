"""MEGNet model wrapper with lazy matgl imports.

Provides functions for loading pretrained MEGNet models, listing
available models, extracting state dicts, and running predictions.
All matgl/DGL imports are lazy (inside function bodies) so the rest
of the package works without matgl installed.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_INSTALL_MSG = (
    "matgl is required for MEGNet training. "
    "Install with: pip install matgl==1.3.0 dgl==2.2.0"
)


def _import_matgl():
    """Lazy import of matgl with helpful error message."""
    try:
        import matgl
    except ImportError:
        raise ImportError(_INSTALL_MSG)
    return matgl


def load_megnet_model(model_name: str):
    """Load a pretrained MEGNet model from matgl.

    Args:
        model_name: Pretrained model identifier, e.g.
            ``"MEGNet-MP-2018.6.1-Eform"``. Use
            :func:`get_available_megnet_models` to list valid names.

    Returns:
        A matgl model object with ``.cutoff`` and ``.predict_structure``
        attributes.

    Raises:
        ImportError: If matgl is not installed.
    """
    matgl = _import_matgl()
    logger.info("Loading pretrained MEGNet model: %s", model_name)
    model = matgl.load_model(model_name)
    logger.info("Model loaded (cutoff=%.2f)", model.cutoff)
    return model


def get_available_megnet_models() -> List[str]:
    """Return list of pretrained MEGNet model names available in matgl.

    Returns:
        List of model name strings containing ``"MEGNet"``.

    Raises:
        ImportError: If matgl is not installed.
    """
    matgl = _import_matgl()
    all_models = matgl.get_available_pretrained_models()
    megnet_models = [m for m in all_models if "MEGNet" in m]
    return megnet_models


def get_megnet_state_dict(model) -> dict:
    """Extract the inner torch Module state dict from a matgl model.

    Lightning checkpoints (``.ckpt``) differ from the project standard
    ``.pt`` format.  This function extracts the raw ``state_dict()``
    suitable for ``torch.save()``.

    Args:
        model: A matgl model returned by :func:`load_megnet_model`.

    Returns:
        An ``OrderedDict`` of parameter tensors.
    """
    return model.model.state_dict()


def predict_with_megnet(model, structures: list) -> List[float]:
    """Run predictions on a list of pymatgen Structure objects.

    Args:
        model: A matgl model returned by :func:`load_megnet_model`.
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
