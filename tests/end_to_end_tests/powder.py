import matplotlib.pyplot as plt
import numpy as np
from xrd_simulator.detector import Detector
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.beam import Beam
from xrd_simulator.motion import RigidBodyMotion
from xrd_simulator.templates import get_uniform_powder_sample
import os

pixel_size = 150.
detector_size = pixel_size * 512
detector_distance = 142938.28756189224
det_corner_0 = np.array([detector_distance, -detector_size / 2., -detector_size / 2.])
det_corner_1 = np.array([detector_distance, detector_size / 2., -detector_size / 2.])
det_corner_2 = np.array([detector_distance, -detector_size / 2., detector_size / 2.])

detector = Detector(pixel_size, pixel_size, det_corner_0, det_corner_1, det_corner_2)
sample_bounding_radius = 0.00001 * detector_size
# polycrystal = get_uniform_powder_sample(
#     sample_bounding_radius=sample_bounding_radius,
#     number_of_grains=2,
#     unit_cell=[4.926, 4.926, 5.4189, 90., 100., 120.],
#     sgname='P3221'
# )
# polycrystal.save('my_polycrystal', save_mesh_as_xdmf=False)
polycrystal = Polycrystal.load('my_polycrystal.pc')
print(polycrystal.mesh_lab.coord.shape)
print(polycrystal.mesh_lab.coord[:5])  # show first 5 mesh points

print(dir(polycrystal.mesh_lab))



# orientation_lab = list(polycrystal.orientation_lab)
# orientation_lab[0] = None  # set first grain to powder
# polycrystal.orientation_lab = orientation_lab

# polycrystal.save('my_polycrystal', save_mesh_as_xdmf=False)
# import pyvista as pv
# import meshio
# # Load the tetrahedral mesh
# mesh = meshio.read("my_polycrystal.xdmf")

# # Convert to pyvista mesh
# points = mesh.points
# cells = mesh.cells_dict.get("tetra")
# if cells is None:
#     raise ValueError("No tetrahedral cells found.")

# # Create an unstructured grid with tetrahedral cells
# grid = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, points)

# # Extract the outer surface (triangle mesh)
# surface = grid.extract_surface()

# # Save as STL or PLY for Blender
# surface.save("my_polycrystal.ply")  # or .ply

w = 0.01 * sample_bounding_radius  # full field beam
print(w)
beam_vertices = np.array([
    [-detector_distance, -w, -w],
    [-detector_distance, w, -w],
    [-detector_distance, w, w],
    [-detector_distance, -w, w],
    [detector_distance, -w, -w],
    [detector_distance, w, -w],
    [detector_distance, w, w],
    [detector_distance, -w, w]])
wavelength = 0.285227
xray_propagation_direction = np.array([1, 0, 0]) * 2 * np.pi / wavelength
polarization_vector = np.array([0, 1, 0])
beam = Beam(
    beam_vertices,
    xray_propagation_direction,
    wavelength,
    polarization_vector)

rotation_axis = np.array([0, 0, 1])
translation = np.array([0, 0.3, 0])

Nx=8
N_angle = 3


rotation_angle_placement = 60/N_angle * np.pi / 180.01
rotation_angle_per_scan = 10 * np.pi / 180.01

for i in range(N_angle):
    for j in range(Nx):
        detector = Detector(pixel_size, pixel_size, det_corner_0, det_corner_1, det_corner_2)
        rot = rotation_angle_placement*i+0.0000001
        trans = translation*j
        placement = RigidBodyMotion(rotation_axis, rot, trans)

        motion1deg = RigidBodyMotion(rotation_axis, rotation_angle_per_scan, np.array([0,0,0]))
        polycrystal = Polycrystal.load('my_polycrystal.pc')
        orientation_lab = list(polycrystal.orientation_lab)
        orientation_lab[0] = None  # set first grain to powder
        polycrystal.orientation_lab = orientation_lab
        # BB_intersection=False must work for powder samples
        polycrystal.transform(placement, time=1.0)
        polycrystal.diffract(beam,
                            detector,
                            motion1deg,
                            BB_intersection=False)
        diffraction_pattern = detector.render(frames_to_render=0,
                                            method='project',
                                            lorentz=False,
                                            polarization=False,
                                            structure_factor=False)
        print(np.sum(diffraction_pattern))

        plt.figure(figsize=(10, 10))
        plt.imshow(diffraction_pattern, cmap='gray')
        plt.colorbar()
        #plt.clim([0,0.0001])
        plt.savefig("data/diffraction_pattern_" + str(i)+'_' + str(j) + ".png", dpi=200)
        plt.close('all')
        del diffraction_pattern, polycrystal, detector
        print(i,j)
#plt.show()
