
import numpy as np


def prepare_exploded_view(components):
    if not components:
        raise ValueError("No model components supplied.")

    vertex_arrays = [
        np.asarray(component["mesh"].vertices).copy()
        for component in components
    ]

    nonempty = [v for v in vertex_arrays if len(v)]
    if not nonempty:
        raise ValueError("Model contains no vertices.")

    model_min = np.min(
        np.vstack([v.min(axis=0) for v in nonempty]),
        axis=0,
    )
    model_max = np.max(
        np.vstack([v.max(axis=0) for v in nonempty]),
        axis=0,
    )

    center = (model_min + model_max) / 2.0
    diagonal = float(np.linalg.norm(model_max - model_min))

    if diagonal < 1e-9:
        raise ValueError("Model has zero size.")

    for component, vertices in zip(components, vertex_arrays):
        component["_base_vertices"] = vertices

        if len(vertices):
            component_center = vertices.mean(axis=0)
            direction = component_center - center
            length = float(np.linalg.norm(direction))

            if length > 1e-9:
                direction /= length
            else:
                direction = np.array([0.0, 1.0, 0.0])
        else:
            direction = np.array([0.0, 1.0, 0.0])

        component["_explode_direction"] = direction
        component["_explode_distance"] = diagonal * 0.12

    return center, diagonal
