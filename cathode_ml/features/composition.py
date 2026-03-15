"""Magpie composition featurization for cathode materials.

Uses matminer's ElementProperty featurizer with the 'magpie' preset
to generate 132-dimensional composition descriptors. Handles NaN values
via median imputation and drops all-NaN columns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition


def featurize_compositions(
    formulas: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Generate Magpie composition descriptors for a list of chemical formulas.

    Args:
        formulas: List of chemical formula strings (e.g., ["LiCoO2", "LiFePO4"]).

    Returns:
        Tuple of (X, labels) where:
            X: ndarray of shape (n_samples, n_features) with Magpie descriptors.
               n_features is 132 unless all-NaN columns were dropped.
            labels: List of feature label strings matching X columns.
    """
    if not formulas:
        return np.empty((0, 0)), []

    # Build dataframe with composition objects
    df = pd.DataFrame({"formula": formulas})
    df["composition"] = df["formula"].apply(Composition)

    # Featurize using Magpie preset (132 descriptors)
    # Suppress tqdm deprecation warnings from matminer internals
    from matminer.featurizers.composition import ElementProperty

    featurizer = ElementProperty.from_preset("magpie")
    featurizer.set_n_jobs(1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        df = featurizer.featurize_dataframe(
            df, col_id="composition", ignore_errors=True, pbar=False
        )

    # Extract feature columns
    feature_labels = featurizer.feature_labels()
    X = df[feature_labels].values.astype(float)

    # Drop columns that are entirely NaN
    all_nan_mask = np.all(np.isnan(X), axis=0)
    X = X[:, ~all_nan_mask]
    feature_labels = [lbl for lbl, is_nan in zip(feature_labels, all_nan_mask) if not is_nan]

    # Impute remaining NaN with column median
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            col[nan_mask] = median_val

    return X, feature_labels
