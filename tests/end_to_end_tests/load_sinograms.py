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

if __name__ == '__main__':
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        all_patterns = pool.map(load_one_file, files)

    # Final 4D array: (num_angles, num_j, H, W)
    diffraction_4d = np.stack(all_patterns)
    print("Loaded shape:", diffraction_4d.shape)

    num_angles, num_j, H, W = np.shape(diffraction_4d)


    plt.figure()
    plt.imshow(diffraction_4d[:,:,H//2+50, W//2+150])
    plt.show()

    plt.figure()
    plt.imshow(diffraction_4d[:,:,H//2+50, W//2+200])
    plt.show()

    plt.figure()
    plt.imshow(diffraction_4d[:,:,H//2+50, W//2+230])
    plt.show()

    plt.figure()
    plt.imshow(diffraction_4d[:,:,H//2+50, W//2+120])
    plt.show()