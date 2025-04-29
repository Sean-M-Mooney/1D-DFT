import numpy as np

######## Parameters #########

# Computational Parameters
N_basis = 100  # Number of basis functions
n_grid_points = 2000  # Number of real space grid points
mixing_beta = 0.75
normalization_tolerance = 1e-05
tol = 1e-06
max_iter = 300

# System
L = 20.0  # Box lengths
n_electrons = 12  # Number of electrons
x = np.linspace(-L/2, L/2, n_grid_points)  # Real-space grid
alpha = 0
n_projectors = 1

# Modes
write_dens = False
write_proj = False
read_dens = False
hubbard = False

# total_energy = [-119.89834626367814, -119.89900284697951, -119.89965407431389, -119.90029915445098, -119.90093769831965]
# occ = [0.8463610371144896, 0.8440089957291335, 0.8416128755176623, 0.8391724541304246, 0.8366873766349565]
