import numpy as np
from config import n_electrons, n_grid_points, L, N_basis, x, read_dens
from basis import basis
import matplotlib.pyplot as plt

# Initialize density
def initial_density():
    rho = np.zeros(n_grid_points)
    if read_dens:
        rho = np.load('dens.npy')
    else:
        rho[:] = n_electrons / L  # Example starting density mode
    return rho


# Calculate real space charge density from basis coeffs
def calculate_density(coeffs):
    dens = np.zeros(n_grid_points)
    for j in range(n_electrons):
        wavefunction = np.zeros(n_grid_points, dtype=complex)
        for i in range(N_basis + 1):
            wavefunction += coeffs[i][j] * basis[i]

        dens += (wavefunction * np.conjugate(wavefunction)).real

    return dens


def fourier_trans_density(rho):
    rho_ft = np.zeros(2 * N_basis + 1, dtype=complex)
    for k in range(int(-N_basis), int(N_basis) + 1):
        rho_ft[N_basis + k] = (1/L) * np.trapezoid(np.exp(-1j * 2 * np.pi * k * x / L) * rho, x)
    return rho_ft