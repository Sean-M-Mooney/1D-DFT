import numpy as np

######## Parameters #########

# Computational Parameters
N_basis = 100  # Number of basis functions
n_grid_points = 2000  # Number of real space grid points
mixing_beta = 0.5
normalization_tolerance = 1e-05
tol = 1e-06
max_iter = 300

# System
L = 20.0  # Box lengths
n_electrons = 12  # Number of electrons
x = np.linspace(-L/2, L/2, n_grid_points)  # Real-space grid
alpha = 0
n_projectors = 1
A = 1
k = 1
ewald_sums = 0

# Modes
write_dens = False
write_proj = False
read_dens = False
hubbard = False
