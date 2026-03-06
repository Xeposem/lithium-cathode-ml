"""Crystal structure to PyG graph conversion with Gaussian distance expansion.

Converts pymatgen Structure objects into torch_geometric Data objects
suitable for graph neural networks (CGCNN, MEGNet).

Uses scipy KDTree with periodic images for PBC-aware neighbor finding
(avoids pymatgen Cython buffer dtype issue on some platforms).
"""

from typing import List, Tuple

import numpy as np
import torch
from pymatgen.core import Structure
from scipy.spatial import KDTree
from torch_geometric.data import Data


def gaussian_expansion(
    distances: torch.Tensor,
    dmin: float = 0.0,
    dmax: float = 5.0,
    num_gaussians: int = 80,
) -> torch.Tensor:
    """Expand scalar distances into Gaussian basis functions.

    Args:
        distances: 1D tensor of interatomic distances, shape (N,).
        dmin: Minimum center for Gaussian expansion.
        dmax: Maximum center for Gaussian expansion.
        num_gaussians: Number of Gaussian basis functions.

    Returns:
        Expanded distances, shape (N, num_gaussians).
    """
    centers = torch.linspace(dmin, dmax, num_gaussians, dtype=distances.dtype)
    gamma = 1.0 / (centers[1] - centers[0])
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)


def structure_to_graph(structure: Structure, config: dict) -> Data:
    """Convert a pymatgen Structure to a PyG Data object.

    Node features are one-hot encoded atomic numbers.
    Edge features are Gaussian-expanded interatomic distances.
    Edges are determined by a distance cutoff with periodic boundary conditions.

    Args:
        structure: pymatgen Structure with periodic boundary conditions.
        config: Configuration dict with 'graph' section containing
            cutoff_radius, max_neighbors, gaussian params, node_feature_dim.

    Returns:
        torch_geometric Data object with x, edge_index, edge_attr.
    """
    graph_cfg = config["graph"]
    cutoff_radius = graph_cfg["cutoff_radius"]
    max_neighbors = graph_cfg["max_neighbors"]
    gauss_cfg = graph_cfg["gaussian"]
    node_feature_dim = graph_cfg["node_feature_dim"]

    num_atoms = len(structure)

    # Node features: one-hot atomic number
    x = torch.zeros(num_atoms, node_feature_dim)
    for i, site in enumerate(structure):
        atomic_num = site.specie.Z
        if atomic_num < node_feature_dim:
            x[i, atomic_num] = 1.0

    # Edges via PBC-aware neighbor finding using scipy KDTree with periodic images
    # Generate periodic images of all atom positions within cutoff range
    lattice = structure.lattice
    cart_coords = structure.cart_coords  # (num_atoms, 3)

    # Determine which periodic images to consider
    # For each lattice vector, we need enough images to cover cutoff_radius
    frac_cutoff = lattice.get_fractional_coords(
        np.array([cutoff_radius, cutoff_radius, cutoff_radius])
    )
    max_images = np.ceil(np.abs(frac_cutoff)).astype(int) + 1

    # Build expanded coordinate array with periodic images
    image_coords: List[np.ndarray] = []
    image_atom_indices: List[int] = []

    for ia in range(-max_images[0], max_images[0] + 1):
        for ib in range(-max_images[1], max_images[1] + 1):
            for ic in range(-max_images[2], max_images[2] + 1):
                shift = lattice.get_cartesian_coords(np.array([ia, ib, ic]))
                for atom_idx in range(num_atoms):
                    image_coords.append(cart_coords[atom_idx] + shift)
                    image_atom_indices.append(atom_idx)

    image_coords_arr = np.array(image_coords)
    tree = KDTree(image_coords_arr)

    src_list = []
    dst_list = []
    dist_list = []

    for atom_idx in range(num_atoms):
        # Query neighbors within cutoff
        dists, indices = tree.query(
            cart_coords[atom_idx], k=len(image_coords_arr), distance_upper_bound=cutoff_radius
        )
        # Filter out self (distance ~0) and inf (no neighbor)
        neighbors = []
        for d, idx in zip(dists, indices):
            if d == np.inf or idx >= len(image_atom_indices):
                break
            dst_idx = image_atom_indices[idx]
            if d < 1e-8 and dst_idx == atom_idx:
                continue  # Skip self-loop at origin
            neighbors.append((dst_idx, d))

        # Sort by distance, take up to max_neighbors
        neighbors.sort(key=lambda x: x[1])
        for dst_idx, distance in neighbors[:max_neighbors]:
            src_list.append(atom_idx)
            dst_list.append(dst_idx)
            dist_list.append(distance)

    if len(src_list) == 0:
        # Return graph with empty edges (will fail validation)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, gauss_cfg["num_gaussians"])
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        distances = torch.tensor(dist_list, dtype=torch.float32)
        edge_attr = gaussian_expansion(
            distances,
            dmin=gauss_cfg["dmin"],
            dmax=gauss_cfg["dmax"],
            num_gaussians=gauss_cfg["num_gaussians"],
        )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def validate_graph(data: Data) -> Tuple[bool, str]:
    """Validate that a PyG Data object represents a valid molecular graph.

    Checks:
    - x (node features) exists and is non-empty
    - edge_index exists and is non-empty
    - All nodes appear in edge_index (no isolated nodes)
    - edge_attr count matches edge count

    Args:
        data: PyG Data object to validate.

    Returns:
        Tuple of (is_valid, reason). reason is empty string if valid.
    """
    if data.x is None or data.x.shape[0] == 0:
        return False, "Node features (x) are empty or missing"

    if data.edge_index is None or data.edge_index.shape[1] == 0:
        return False, "Edge index is empty - graph has no edges"

    num_nodes = data.x.shape[0]
    nodes_in_edges = torch.unique(data.edge_index).tolist()
    missing_nodes = [i for i in range(num_nodes) if i not in nodes_in_edges]
    if missing_nodes:
        return False, f"Isolated nodes found: {missing_nodes}"

    num_edges = data.edge_index.shape[1]
    if data.edge_attr is not None and data.edge_attr.shape[0] != num_edges:
        return False, (
            f"Edge attr count ({data.edge_attr.shape[0]}) "
            f"doesn't match edge count ({num_edges})"
        )

    return True, ""
