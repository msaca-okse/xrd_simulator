import sys
import os
import numpy as np
#from multiprocessing import Pool
import h5py
import os
device = 'cpu'

if device == 'cpu':
    cp = np




def cartesian_to_polar_cupy(matrix, num_phi=360, num_rad=None,factor = 2):
    """ Convert a 2D CuPy matrix from Cartesian to Polar coordinates. """
    num_phi_old = num_phi
    num_rad_old = num_rad
    num_phi = factor*num_phi
    num_rad = factor*num_rad
    rows, cols = matrix.shape
    bias_col, bias_row = 0, 0
    center_x, center_y = cols // 2 + bias_col, rows // 2 + bias_row
    max_radius = (min(cols, rows) // 2)
    
    if num_rad is None:
        num_rad = max_radius

    # Create polar coordinate grid
    theta = cp.linspace(0, 2 * cp.pi, num_phi)  # Angles
    r = cp.linspace(0, max_radius, num_rad)  # Radii
    R, Theta = cp.meshgrid(r, theta)  # Create grid

    # Convert polar to Cartesian coordinates
    X = center_x + R * cp.cos(Theta)
    Y = center_y + R * cp.sin(Theta)

    # Bilinear interpolation
    X = cp.clip(X, 0, cols - 1)
    Y = cp.clip(Y, 0, rows - 1)
    x0, y0 = X.astype(cp.int32), Y.astype(cp.int32)  # Floor values
    x1, y1 = cp.clip(x0 + 1, 0, cols - 1), cp.clip(y0 + 1, 0, rows - 1)  # Ceiling values

    # Get pixel values from the original image
    Ia = matrix[y0, x0]
    Ib = matrix[y0, x1]
    Ic = matrix[y1, x0]
    Id = matrix[y1, x1]

    # Compute bilinear interpolation weights
    wa = (x1 - X) * (y1 - Y)
    wb = (X - x0) * (y1 - Y)
    wc = (x1 - X) * (Y - y0)
    wd = (X - x0) * (Y - y0)

    # Compute interpolated values
    polar_matrix = wa * Ia + wb * Ib + wc * Ic + wd * Id
    polar_matrix = polar_matrix.reshape(num_phi_old, factor, num_rad_old, factor)

    # Take the mean over the (4,4) blocks
    polar_matrix = polar_matrix.mean(axis=(1, 3))

    return polar_matrix

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import multiprocessing
import re

# Step 1: Get all HDF5 files sorted by angle index
# Extract numeric index from filename
def extract_index(filename):
    match = re.search(r'angle_(\d+)\.h5', filename)
    return int(match.group(1)) if match else -1

# File list, sorted numerically by angle index
files = sorted(glob.glob("data/diffraction_patterns_angle_*.h5"), key=extract_index)

# Load and stack patterns from one file
def load_one_file(filename):
    with h5py.File(filename, 'r') as f:
        patterns = [f[key][:] for key in sorted(f.keys())]
        return np.stack(patterns)  # shape: (num_j, H, W)



# --- Per-angle integration task ---
def process_one_angle(i):
    num_rad = 256
    partial_result = np.empty((num_j, num_rad), dtype=np.float32)


    r_nonunif = cp.arcsin(cp.linspace(0,0.262,256))
    r_unif = cp.linspace(0,1, 256)*0.262


    for j in range(num_j):
        polar_diffraction = cartesian_to_polar_cupy(diffraction_4d[i, j], num_phi = 360, num_rad=num_rad)

        integral = polar_diffraction.sum(axis=0)
        integral_corrected  = cp.interp(r_nonunif, r_unif, integral).astype(cp.float32)

        partial_result[j] = integral_corrected
    return partial_result

if __name__ == '__main__':
    # Step 1: Load all diffraction pattern files in parallel
    #files = sorted(glob.glob("/home/msaca/Data/simulated_diffraction_data/single_powder_simulation/diffraction_patterns_angle_*.h5"), key=extract_index)
    files = sorted(glob.glob("/home/msaca/Data/simulated_diffraction_data/single_crystal_simulation/diffraction_patterns_crystal_angle_*.h5"), key=extract_index)

    #files = files[60:65]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        all_patterns = pool.map(load_one_file, files)

    diffraction_4d = np.stack(all_patterns)  # (num_angles, num_j, H, W)
    print("Loaded shape:", diffraction_4d.shape)

    # Step 2: Set shape globals for worker access
    num_angles, num_j, H, W = diffraction_4d.shape

    # Step 3: Parallel processing of outer loop
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        result_list = pool.map(process_one_angle, range(num_angles))

    integrated_data = np.stack(result_list)  # (num_angles, num_j, num_rad)
    #with h5py.File("/home/msaca/Data/simulated_diffraction_data/single_powder_simulation/integrated_data.h5", "w") as f:
    #    f.create_dataset("integrated", data=integrated_data)
    with h5py.File("/home/msaca/Data/simulated_diffraction_data/single_crystal_simulation/integrated_data.h5", "w") as f:
        f.create_dataset("integrated", data=integrated_data)

    print("Integrated shape:", integrated_data.shape)
