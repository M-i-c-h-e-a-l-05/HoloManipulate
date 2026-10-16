from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def load_multipart_model(model_path, color=(0.62, 0.65, 0.69)):
    """
    Load a multipart GLB as separate Open3D meshes.

    Each component retains its original scene-graph world transform.
    Materials and textures are intentionally ignored.
    """
    path = Path(model_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    scene = trimesh.load(
        str(path),
        force="scene",
        process=False,
    )

    if not isinstance(scene, trimesh.Scene):
        raise TypeError("Expected a multipart GLB scene.")

    components = []

    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        source_mesh = scene.geometry[geometry_name]

        if not isinstance(source_mesh, trimesh.Trimesh):
            continue
        if len(source_mesh.vertices) == 0 or len(source_mesh.faces) == 0:
            continue

        # Copy before transforming so the source scene remains unchanged.
        mesh = source_mesh.copy()
        mesh.apply_transform(transform)

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        geometry = o3d.geometry.TriangleMesh()
        geometry.vertices = o3d.utility.Vector3dVector(vertices)
        geometry.triangles = o3d.utility.Vector3iVector(faces)
        geometry.compute_vertex_normals()
        geometry.paint_uniform_color(color)

        components.append({
            "name": str(node_name),
            "geometry_name": str(geometry_name),
            "mesh": geometry,
        })

    if not components:
        raise ValueError(f"No usable mesh components found in {path}")

    return components
