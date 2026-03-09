"""M3GNet model wrapper with lazy matgl imports.

Provides functions for loading pretrained M3GNet models, listing
available models, extracting state dicts, and running predictions.
All matgl imports are lazy (inside function bodies) so the rest
of the package works without matgl installed.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_INSTALL_MSG = (
    "matgl>=2.0.0 is required for M3GNet training. "
    "Install with: pip install 'matgl>=2.0.0' dgl"
)


def _import_matgl():
    """Lazy import of matgl with DGL backend for M3GNet compatibility.

    M3GNet is a DGL-only model in matgl 2.x.  The ``MATGL_BACKEND``
    environment variable must be set **before** the first matgl import
    so that ``matgl.layers.__init__`` exports the DGL-specific classes
    (e.g. ``EmbeddingBlock``) that M3GNet requires.
    """
    import os

    os.environ.setdefault("MATGL_BACKEND", "dgl")
    try:
        import matgl
    except ImportError:
        raise ImportError(_INSTALL_MSG)
    return matgl


def load_m3gnet_model(model_name: str = "M3GNet-MP-2018.6.1-Eform"):
    """Load a pretrained M3GNet model from matgl.

    Args:
        model_name: Pretrained model identifier. Default is
            ``"M3GNet-MP-2018.6.1-Eform"``. Use
            :func:`get_available_m3gnet_models` to list valid names.

    Returns:
        A matgl model object with ``.cutoff`` and ``.predict_structure``
        attributes.

    Raises:
        ImportError: If matgl is not installed.
    """
    matgl = _import_matgl()
    logger.info("Loading pretrained M3GNet model: %s", model_name)
    model = matgl.load_model(model_name)
    # matgl 2.x wraps models in TransformedTargetModel; cutoff is on the inner model
    cutoff = getattr(model, "cutoff", None) or getattr(model.model, "cutoff", None)
    logger.info("Model loaded (cutoff=%.2f)", cutoff)
    return model


def get_available_m3gnet_models() -> List[str]:
    """Return list of pretrained M3GNet model names available in matgl.

    Returns:
        List of model name strings containing ``"M3GNet"``.

    Raises:
        ImportError: If matgl is not installed.
    """
    matgl = _import_matgl()
    all_models = matgl.get_available_pretrained_models()
    m3gnet_models = [m for m in all_models if "M3GNet" in m]
    return m3gnet_models


def get_m3gnet_state_dict(model) -> dict:
    """Extract the inner torch Module state dict from a matgl model.

    Lightning checkpoints (``.ckpt``) differ from the project standard
    ``.pt`` format.  This function extracts the raw ``state_dict()``
    suitable for ``torch.save()``.

    Args:
        model: A matgl model returned by :func:`load_m3gnet_model`.

    Returns:
        An ``OrderedDict`` of parameter tensors.
    """
    return model.model.state_dict()


def predict_with_m3gnet(model, structures: list) -> List[float]:
    """Run predictions on a list of pymatgen Structure objects.

    Args:
        model: A matgl model returned by :func:`load_m3gnet_model`.
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
