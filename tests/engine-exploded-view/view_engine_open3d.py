import os
import numpy as np
import trimesh
import open3d as o3d

GLB_PATH = "tests/engine-exploded-view/source/untitled.glb"

if not os.path.isfile(GLB_PATH):
    raise FileNotFoundError(f"Engine GLB not found: {GLB_PATH}")

print("Loading engine:", os.path.abspath(GLB_PATH))
scene = trimesh.load(GLB_PATH, force="scene", process=False)

viewer = o3d.visualization.Visualizer()

if not viewer.create_window(
    window_name="HoloManipulate - Engine Import Test",
    width=1280,
    height=800,
):
    raise RuntimeError("Open3D could not create its visualization window.")

loaded = 0
failed = []

for node_name in scene.graph.nodes_geometry:
    try:
        # Get the component's transform in the scene's base frame.
        transform, geometry_name = scene.graph[node_name]

        # Copy the mesh so the original GLB scene remains untouched.
        source_mesh = scene.geometry[geometry_name]
        vertices = np.asarray(source_mesh.vertices)
        faces = np.asarray(source_mesh.faces)

        if len(vertices) == 0 or len(faces) == 0:
            failed.append((node_name, "Empty mesh"))
            continue

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.copy())
        mesh.triangles = o3d.utility.Vector3iVector(faces.copy())

        # Preserve each part's position and orientation from the GLB.
        mesh.transform(np.asarray(transform, dtype=float))
        mesh.compute_vertex_normals()

        # Preserve vertex colors where available; otherwise use the
        # material's base color as an approximation.
        visual = getattr(source_mesh, "visual", None)
        colors = None

        if visual is not None:
            vertex_colors = getattr(visual, "vertex_colors", None)

            if vertex_colors is not None:
                vertex_colors = np.asarray(vertex_colors)
                if (
                    len(vertex_colors) == len(vertices)
                    and vertex_colors.ndim == 2
                    and vertex_colors.shape[1] >= 3
                ):
                    colors = vertex_colors[:, :3].astype(float) / 255.0

            if colors is None:
                material = getattr(visual, "material", None)
                main_color = getattr(material, "main_color", None)

                if main_color is not None:
                    rgba = np.asarray(main_color, dtype=float).reshape(-1)
                    if len(rgba) >= 3:
                        color = np.clip(rgba[:3] / 255.0, 0.0, 1.0)
                        colors = np.tile(color, (len(vertices), 1))

        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        if not viewer.add_geometry(mesh):
            failed.append((node_name, "Open3D rejected geometry"))
            continue

        loaded += 1

    except Exception as exc:
        failed.append((node_name, str(exc)))

print("\n--- VIEWER RESULTS ---")
print("GLB geometry count:", len(scene.geometry))
print("Geometry nodes:", len(scene.graph.nodes_geometry))
print("Meshes added to Open3D:", loaded)
print("Failed components:", len(failed))

if failed:
    for name, reason in failed[:10]:
        print(f"  {name}: {reason}")

if loaded == 0:
    viewer.destroy_window()
    raise RuntimeError("No engine components were displayed.")

bounds = scene.bounds
print("Original scene bounds:", bounds)
print("Original scene extents:", scene.extents)

# Center the camera on the imported engine.
center = (bounds[0] + bounds[1]) / 2.0
view = viewer.get_view_control()
view.set_lookat(center.tolist())
view.set_front([0.0, 0.0, -1.0])
view.set_up([0.0, 1.0, 0.0])
view.set_zoom(0.65)

print("\nInspect the engine in the Open3D window.")
print("Close the window to finish the test.")
viewer.run()
viewer.destroy_window()
print("Viewer closed.")
