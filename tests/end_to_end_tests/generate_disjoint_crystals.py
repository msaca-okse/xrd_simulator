import matplotlib.pyplot as plt
import numpy as np
from xrd_simulator.detector import Detector
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.phase import Phase
from xrd_simulator.templates import get_uniform_powder_sample
from scipy.spatial.transform import Rotation
import multiprocessing
import os
import h5py

pixel_size = 150.
detector_size = pixel_size * 512
detector_distance = 142938.28756189224
det_corner_0 = np.array([detector_distance, -detector_size / 2., -detector_size / 2.])
det_corner_1 = np.array([detector_distance, detector_size / 2., -detector_size / 2.])
det_corner_2 = np.array([detector_distance, -detector_size / 2., detector_size / 2.])

detector = Detector(pixel_size, pixel_size, det_corner_0, det_corner_1, det_corner_2)
sample_bounding_radius = 0.00001 * detector_size
polycrystal = get_uniform_powder_sample(
    sample_bounding_radius=sample_bounding_radius,
    number_of_grains=2,
    unit_cell=[4.926, 4.926, 5.4189, 90., 100., 120.],
    sgname='P3221'
)
polycrystal.save('my_polycrystal', save_mesh_as_xdmf=False)
poly_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'poly_disjoint_crystal.pc'))



def generate_2d_spaced_tetrahedra(nx=3, ny=3, spacing=4.0, jitter=1.0, bounding_radius=5):
    """
    Generate tetrahedral elements placed on a 2D grid, then shift the entire mesh to fit within a bounding sphere.

    Parameters:
        nx (int): Number of elements along the x-axis.
        ny (int): Number of elements along the y-axis.
        spacing (float): Base spacing between grid points.
        jitter (float): Random displacement for element nodes.
        bounding_radius (float or None): If set, shifts the whole mesh so it fits within this radius from the origin.

    Returns:
        coords (np.ndarray): Node coordinates, shape (num_nodes, 3).
        enod (np.ndarray): Element connectivity, shape (num_elements, 4).
    """
    coords = []
    enod = []

    for i in range(nx):
        for j in range(ny):
            base_center = np.array([i * spacing, j * spacing, 0.0])
            element_nodes = base_center + np.random.uniform(-jitter, jitter, (4, 3))
            start_index = len(coords)
            coords.extend(element_nodes)
            enod.append([start_index + k for k in range(4)])

    coords = np.array(coords)

    # If bounding is requested, shift the mesh to fit within the sphere
    if bounding_radius is not None:
        centroid = np.mean(coords, axis=0)
        coords -= centroid  # Center the mesh at the origin
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > bounding_radius:
            coords *= (bounding_radius / max_dist) * 0.95  # Scale slightly to avoid touching the boundary

    return coords, np.array(enod)

# Example usage
coord, enod = generate_2d_spaced_tetrahedra(nx=3, ny=3, spacing=4.0, jitter=1.0, bounding_radius=3)
print(coord)

# Pass to your mesh wrapper
mesh = TetraMesh.generate_mesh_from_vertices(coord, enod)
orientation = Rotation.random(mesh.number_of_elements).as_matrix()
sgname='P3221'
path_to_cif_file = None


N = 9
def generate_random_unit_cells(N):
    lengths = np.random.uniform(3.0, 6.0, size=(N, 3))        # a, b, c
    angles = np.random.uniform(80.0, 130.0, size=(N, 3))      # α, β, γ
    unit_cells = [np.concatenate([l, a]) for l, a in zip(lengths, angles)]
    return unit_cells

# Example usage
unit_cells = generate_random_unit_cells(N)
phases = []
for i in range(N):
    phases.append(Phase(unit_cells[i], sgname, path_to_cif_file))


strain_tensor=np.zeros((3, 3))
element_phase_map = np.arange(0,9).astype(int)


polycrystal = Polycrystal(
        mesh,
        orientation,
        strain=strain_tensor,
        phases=phases,
        element_phase_map=element_phase_map,
    )
polycrystal.save('poly_disjoint_crystal', save_mesh_as_xdmf=False)


import pyvista as pv
import meshio
# Load the tetrahedral mesh
mesh = meshio.read("poly_disjoint_crystal.xdmf")

# Convert to pyvista mesh
points = mesh.points
cells = mesh.cells_dict.get("tetra")
if cells is None:
    raise ValueError("No tetrahedral cells found.")

# Create an unstructured grid with tetrahedral cells
grid = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, points)

# Extract the outer surface (triangle mesh)
surface = grid.extract_surface()

# Save as STL or PLY for Blender
surface.save("poly_disjoint_crystal.ply")  # or .ply