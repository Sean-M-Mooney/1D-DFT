import numpy as np
from config import N_basis, n_grid_points, n_electrons, x, alpha, n_projectors, hubbard
from basis import basis

if hubbard:
    projector = np.load("projector.npy")

    projection = np.zeros(n_grid_points, dtype=complex)

    # Construct the projection across the real space grid
    for j in range(n_projectors):
        for i in range(N_basis + 1):
            projection += projector[i, j] * basis[i]


def occupancy_matrix(coeffs):
    # Calculates the occupation of the hubbard subspace, currently limited to one subspace.
    occ = 0
    for i in range(n_electrons):
        wavefunction = np.zeros(n_grid_points, dtype=complex)
        for j in range(N_basis + 1):
            wavefunction += coeffs[j, i] * basis[j]
        overlap = np.trapezoid(np.conjugate(wavefunction) * projection, x)
        occ += np.conjugate(overlap) * overlap

    return occ.real


def perturbation_matrix():
    # Calculates the alpha perturbation contribution to the Hamiltonian
    p_v = np.zeros([N_basis + 1, N_basis + 1], dtype=complex)

    for i in range(N_basis + 1):
        for j in range(N_basis + 1):
            p_v[i, j] = alpha * np.trapezoid(np.conjugate(basis[i]) * projection * np.conjugate(projection) * basis[j], x)

    return p_v
