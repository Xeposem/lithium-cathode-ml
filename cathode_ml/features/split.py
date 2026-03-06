"""Compositional group splitting to prevent polymorph leakage.

Ensures that all entries sharing the same reduced chemical formula
(polymorphs, different stoichiometries of the same composition) are
placed in the same train/val/test fold. This prevents data leakage
where a model memorizes compositions seen during training.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Composition
from sklearn.model_selection import GroupShuffleSplit


def get_group_keys(formulas: list[str]) -> list[str]:
    """Normalize chemical formulas to reduced formula for group-based splitting.

    Polymorphs and non-reduced formulas (e.g., Li2Co2O4 -> LiCoO2) are
    mapped to the same group key.

    Args:
        formulas: List of chemical formula strings.

    Returns:
        List of reduced formula strings (same length as input).
    """
    return [Composition(f).reduced_formula for f in formulas]


def compositional_split(
    n_samples: int,
    groups: list[str],
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test sets grouped by composition.

    All entries sharing the same reduced formula are placed in the same
    fold, preventing polymorph leakage between splits.

    Args:
        n_samples: Total number of samples.
        groups: List of group keys (reduced formulas) for each sample.
        test_size: Fraction of data for test set (default 0.1).
        val_size: Fraction of data for validation set (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices) as numpy arrays.
    """
    indices = np.arange(n_samples)
    groups_arr = np.array(groups)

    # First split: separate test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(indices, groups=groups_arr))

    # Second split: separate val from trainval
    # Adjust val_size relative to trainval remainder
    val_frac = val_size / (1 - test_size)
    groups_trainval = groups_arr[trainval_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_rel_idx, val_rel_idx = next(
        gss_val.split(trainval_idx, groups=groups_trainval)
    )

    # Map relative indices back to original indices
    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    return train_idx, val_idx, test_idx
