import numpy as np
from config import n_grid_points, N_basis, L, x, n_electrons, hubbard, A, k
from basis import basis
from density import fourier_trans_density, calculate_density, initial_density
from scipy.linalg import eigh
from hubbard import perturbation_matrix
from plotting import plot_potential

#This class calculates, stores and solves the Hamiltonian matrix. The external potential is also defined here in external_potential().


class Hamiltonian:
    def __init__(self):
        self.rho = initial_density()
        self.v_ext = self.external_potential()
        self.v_x = self.exchange_potential()
        self.v_har = self.hartree_potential()

        self.T = self.kinetic_energy_matrix()
        self.V = self.potential_matrix(self.v_ext)
        self.HAR = self.hartree_matrix()
        self.X = self.exchange_matrix(self.v_x)
        if hubbard:
            self.p_v = perturbation_matrix()

        self.H = self.T + self.V + self.HAR + self.X
        if hubbard:
            self.H += self.p_v
        self.eigvals, self.eigvecs = eigh(self.H)
        self.coeffs = self.eigvecs[:, :n_electrons]

    def update_hamiltonian(self, new_rho: np.ndarray):
        self.rho = new_rho
        self.v_x = self.exchange_potential()
        self.v_har = self.hartree_potential()

        self.HAR = self.hartree_matrix()
        self.X = self.exchange_matrix(self.v_x)

    def solve(self) -> np.ndarray:
        self.H = self.T + self.V + self.HAR + self.X
        if hubbard:
            self.H += self.p_v
        self.eigvals, self.eigvecs = eigh(self.H)
        self.coeffs = self.eigvecs[:, :n_electrons]
        self.rho = calculate_density(self.coeffs)

    # Compute kinetic energy matrix
    def kinetic_energy_matrix(self) -> np.ndarray:
        k_vals = (2 * np.pi * np.arange(int(-N_basis / 2), int(N_basis / 2) + 1) / L) ** 2
        K_q = np.zeros((N_basis + 1, N_basis + 1), dtype=complex)
        for i in range(N_basis + 1):
            K_q[i, i] = k_vals[i] / 2 + 1j * 0

        return K_q

    # Compute the external potential across the real space grid
    def external_potential(self):
        #v_ext = np.zeros(n_grid_points)
        center = x[100 * int(n_grid_points / 200)]
        width = 12
        v_ext = -60 * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        plot_potential(v_ext)
        return v_ext

    # External potential matrix
    def potential_matrix(self, v_ext: np.ndarray) -> np.ndarray:
        V_q = np.zeros((N_basis + 1, N_basis + 1), dtype=complex)
        for i in range(N_basis + 1):
            for j in range(N_basis + 1):
                V_q[i, j] = np.trapezoid(np.conjugate(basis[i]) * v_ext * basis[j], x)
        return V_q

    # Hartree potential matrix
    def hartree_matrix(self) -> np.ndarray:
        H_q = np.zeros((N_basis + 1, N_basis + 1), dtype=complex)
        for i in range(N_basis + 1):
            for j in range(N_basis + 1):
                H_q[i, j] = np.trapezoid(np.conjugate(basis[i]) * self.v_har * basis[j], x)

        return H_q

    def v_ee(self, r):
        v_ee = A * np.exp(-k * np.abs(r))
        return v_ee

    def hartree_potential(self) -> np.ndarray:
        v_har = np.zeros_like(self.rho)
        for i in range(n_grid_points):
            r = x - x[i]
            v_har[i] = np.trapezoid(self.rho * self.v_ee(r), x)

        return v_har

    # Compute the exchange potential across the real space grid
    def exchange_potential(self) -> np.ndarray:
        v_x = (-np.pi / 2) * self.rho
        return v_x

    # Construct the exchange potential matrix
    def exchange_matrix(self, v_x: np.ndarray) -> np.ndarray:
        ex = np.zeros((N_basis + 1, N_basis + 1), dtype=complex)
        for i in range(N_basis + 1):
            for j in range(N_basis + 1):
                ex[i][j] = np.trapezoid(np.conjugate(basis[i]) * v_x * basis[j], x)

        return ex

    def total_energy(self):
        total_energy = 0

        # Kinetic component
        T_eigs = eigh(self.T)[0]
        E_kin = np.sum(T_eigs[:n_electrons])
        print('Kinetic energy: ', E_kin)
        total_energy += E_kin

        # External component
        E_ext = np.trapezoid(self.v_ext * self.rho, x)
        print('External potential energy: ', E_ext)
        total_energy += E_ext

        # Hartree component
        E_har = np.trapezoid(self.v_har * self.rho, x)
        print('Harte energy: ', E_har)
        total_energy += E_har

        # Exchange contribution
        E_ex = np.trapezoid(self.v_x * self.rho, x)
        print('Exchange energy: ', E_ex)
        total_energy += E_ex

        return total_energy
