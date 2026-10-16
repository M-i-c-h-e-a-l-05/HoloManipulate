import open3d as o3d


def create_city():
    """Build a simple, CPU-friendly 3D city diorama."""
    city = o3d.geometry.TriangleMesh()

    def add_box(x, y, z, sx, sy, sz, color):
        nonlocal city

        part = o3d.geometry.TriangleMesh.create_box(
            width=sx, height=sy, depth=sz
        )
        part.translate([x - sx / 2, y - sy / 2, z])
        part.paint_uniform_color(color)
        part.compute_vertex_normals()
        city += part

    # Base platform
    add_box(0, 0, -0.15, 8.0, 6.0, 0.3, (0.20, 0.35, 0.24))

    # Roads
    add_box(0, 0, 0.005, 7.5, 0.65, 0.015, (0.20, 0.21, 0.23))
    add_box(0, 0, 0.008, 0.65, 5.5, 0.015, (0.20, 0.21, 0.23))

    # Buildings: x, y, width, depth, height, color
    buildings = [
        (-2.5, -1.7, 0.8, 0.8, 1.3, (0.30, 0.55, 0.85)),
        (-1.4, -1.8, 0.7, 0.9, 2.0, (0.65, 0.70, 0.78)),
        ( 1.3, -1.7, 0.9, 0.8, 1.5, (0.85, 0.52, 0.30)),
        ( 2.5, -1.6, 0.8, 0.9, 2.6, (0.40, 0.63, 0.85)),
        (-2.5,  1.5, 0.9, 0.9, 2.2, (0.65, 0.48, 0.78)),
        (-1.3,  1.6, 0.8, 0.8, 1.1, (0.85, 0.75, 0.40)),
        ( 1.4,  1.5, 0.9, 0.9, 2.8, (0.30, 0.68, 0.68)),
        ( 2.6,  1.5, 0.8, 0.8, 1.6, (0.75, 0.43, 0.43)),
        ( 0.0,  2.0, 0.7, 0.8, 1.0, (0.72, 0.75, 0.80)),
    ]

    for x, y, sx, sy, height, color in buildings:
        add_box(x, y, 0.02, sx, sy, height, color)

    city.compute_vertex_normals()
    return city
