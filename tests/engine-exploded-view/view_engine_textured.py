import os
import numpy as np
import trimesh
import open3d as o3d
from open3d.visualization import gui, rendering

GLB_PATH = "tests/engine-exploded-view/source/untitled.glb"

if not os.path.isfile(GLB_PATH):
    raise FileNotFoundError(GLB_PATH)

scene_data = trimesh.load(GLB_PATH, force="scene", process=False)

app = gui.Application.instance
app.initialize()

window = app.create_window(
    "HoloManipulate - Textured Engine Test", 1280, 800
)
widget = gui.SceneWidget()
widget.scene = rendering.Open3DScene(window.renderer)
widget.scene.set_background([0.08, 0.08, 0.10, 1.0])

loaded = 0
textured = 0
failed = []

for node_name in scene_data.graph.nodes_geometry:
    try:
        transform, geometry_name = scene_data.graph[node_name]
        source = scene_data.geometry[geometry_name]
        visual = source.visual

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(source.vertices, dtype=np.float64).copy()
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(source.faces, dtype=np.int32).copy()
        )

        # Open3D expects one UV coordinate per triangle vertex.
        uv = getattr(visual, "uv", None)
        if uv is not None:
            uv = np.asarray(uv)
            faces = np.asarray(source.faces)
            if uv.ndim == 2 and len(uv) == len(source.vertices):
                mesh.triangle_uvs = o3d.utility.Vector2dVector(
                    np.ascontiguousarray(uv[faces].reshape(-1, 2))
                )

        mesh.transform(np.asarray(transform, dtype=np.float64))
        mesh.compute_vertex_normals()

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [1.0, 1.0, 1.0, 1.0]

        source_material = getattr(visual, "material", None)
        texture = getattr(source_material, "baseColorTexture", None)

        if texture is not None and uv is not None:
            pixels = np.asarray(texture)

            if pixels.ndim == 2:
                pixels = np.repeat(pixels[:, :, None], 3, axis=2)

            if pixels.ndim == 3 and pixels.shape[2] in (3, 4):
                if pixels.dtype != np.uint8:
                    if np.issubdtype(pixels.dtype, np.floating):
                        pixels = np.clip(pixels * 255, 0, 255)
                    pixels = pixels.astype(np.uint8)

                pixels = np.ascontiguousarray(pixels)
                material.base_color_map = o3d.geometry.Image(pixels)
                textured += 1

        widget.scene.add_geometry(
            f"{loaded:03d}_{node_name}", mesh, material
        )
        loaded += 1

    except Exception as exc:
        failed.append((node_name, str(exc)))

window.add_child(widget)

def on_layout(context):
    widget.frame = window.content_rect

window.set_on_layout(on_layout)

bounds = scene_data.bounds
minimum = bounds[0]
maximum = bounds[1]

bbox = o3d.geometry.AxisAlignedBoundingBox(minimum, maximum)
widget.setup_camera(60.0, bbox, bbox.get_center())

print("Geometry nodes:", len(scene_data.graph.nodes_geometry))
print("Meshes loaded:", loaded)
print("Components with textures assigned:", textured)
print("Failed components:", len(failed))

for name, error in failed[:10]:
    print("FAILED:", name, error)

print("Close the viewer window to finish.")
app.run()
