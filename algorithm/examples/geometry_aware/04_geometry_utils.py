"""
Geometry utility showcase.

Highlights:
  * Construct grid, k-NN, and radius graphs with `MeshGraphBuilder`.
  * Inspect Laplacian properties and characteristic mesh length.
  * Compute gradient-based weights used by geometry-aware QR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from TBMD.core.geometry import (
    GeometricWeightComputer,
    MeshGraphBuilder,
    estimate_characteristic_length,
)


@dataclass
class MeshSummary:
    name: str
    nodes: int
    edges: int
    avg_degree: float
    characteristic_length: float


def summarize_mesh(builder: MeshGraphBuilder, spatial: Tuple[int, ...]) -> MeshSummary:
    mesh = builder.build_from_shape(spatial)
    adjacency = mesh.adjacency_matrix
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    char_len = estimate_characteristic_length(mesh)
    return MeshSummary(
        name=builder.connectivity_type,
        nodes=adjacency.shape[0],
        edges=adjacency.nnz // 2,
        avg_degree=float(degrees.mean()),
        characteristic_length=float(char_len),
    )


def demo_grid_mesh() -> None:
    print("=== Grid connectivity ===")
    for spatial in [(12, 18), (10, 10, 6)]:
        builder = MeshGraphBuilder(connectivity_type="grid")
        summary = summarize_mesh(builder, spatial)
        print(
            f"{summary.name:>7} mesh {spatial}: "
            f"{summary.nodes} nodes, {summary.edges} edges, "
            f"avg degree {summary.avg_degree:.2f}, "
            f"h_char={summary.characteristic_length:.3f}"
        )


def demo_coordinate_mesh() -> None:
    print("\n=== Coordinate-based connectivity ===")
    rng = np.random.default_rng(42)
    coords = rng.uniform(-1, 1, size=(200, 2))

    for connectivity, kwargs in [
        ("knn", {"k": 8}),
        ("radius", {"radius": 0.35}),
        ("delaunay", {}),
    ]:
        builder = MeshGraphBuilder(connectivity_type=connectivity, **kwargs)
        mesh = builder.build_from_coordinates(coords)
        adjacency = mesh.adjacency_matrix
        degrees = np.asarray(adjacency.sum(axis=1)).flatten()
        print(
            f"{connectivity:>7}: nodes={adjacency.shape[0]}, "
            f"edges={adjacency.nnz // 2}, "
            f"avg degree={degrees.mean():.2f}, "
            f"h_char={estimate_characteristic_length(mesh):.3f}"
        )


def demo_gradient_weights() -> None:
    print("\n=== Gradient weight computation ===")
    builder = MeshGraphBuilder(connectivity_type="grid")
    mesh = builder.build_from_shape((40, 40))

    # Field with strong gradient in centre
    x = np.linspace(-1.0, 1.0, 40)
    y = np.linspace(-1.0, 1.0, 40)
    X, Y = np.meshgrid(x, y, indexing="ij")
    field = np.tanh(5 * X) + 0.5 * np.sin(3 * Y)

    computer = GeometricWeightComputer(mesh)
    weights_fd = computer.compute_gradient_weights(field.ravel(), method="fd")
    weights_graph = computer.compute_gradient_weights(field.ravel(), method="graph")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(field, cmap="viridis")
    axes[0].set_title("Field")
    axes[1].imshow(weights_fd.reshape(40, 40), cmap="magma")
    axes[1].set_title("FD gradient weights")
    axes[2].imshow(weights_graph.reshape(40, 40), cmap="magma")
    axes[2].set_title("Graph gradient weights")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()

    print(
        f"FD weights: min={weights_fd.min():.3f}, max={weights_fd.max():.3f}, "
        f"median={np.median(weights_fd):.3f}"
    )
    print(
        f"Graph weights: min={weights_graph.min():.3f}, max={weights_graph.max():.3f}, "
        f"median={np.median(weights_graph):.3f}"
    )


def main() -> None:
    demo_grid_mesh()
    demo_coordinate_mesh()
    demo_gradient_weights()


if __name__ == "__main__":
    main()
