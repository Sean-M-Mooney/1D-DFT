import numpy as np
from config import n_electrons, mixing_beta, x, max_iter, tol, normalization_tolerance, write_proj, hubbard, write_dens
from hamiltonian import Hamiltonian
from plotting import plot_density, plot_potential
from hubbard import occupancy_matrix

# This module handles the self-consistency loop and printouts during and after the calculation


class KS_solver:
    def __init__(self, hamiltonian: Hamiltonian):
        self.ham = hamiltonian
        self.rho = self.ham.rho
        self.hubbard_occ = 0

    def self_consistent_loop(self):

        for i in range(max_iter):
            print('\n Iteration #' + str(i + 1))

            self.ham.solve() # Solve KS equations and calculate new density
            rho_new = mixing_beta * self.ham.rho + (1 - mixing_beta) * self.rho  # Mixing

            if hubbard:
                # Compute occupancy of hubbard subspace (limited to one subspace currently)
                self.hubbard_occ = occupancy_matrix(self.ham.coeffs)
                print('Occupancy of hubbard site: ', self.hubbard_occ)

            #Check normalization
            norm = np.trapezoid(rho_new, x)
            if abs(norm - n_electrons) > normalization_tolerance:
                print("Error: Density not normalized")
                break

            # Check convergence
            delta = np.linalg.norm(rho_new - self.rho)
            print("Convergence delta: ", delta)

            if delta < tol:
                print('\n ######  Calculation converged!  ###### \n')
                self.end_calculation()
                break

            total_energy = self.ham.total_energy()
            energy_gradient = total_energy - self.ham.tot_energy
            if energy_gradient > 0:
                print('Warning: Energy gradient positive')
            print('Total energy gradient: ', energy_gradient)
            self.ham.tot_energy = total_energy

            self.rho = rho_new
            self.ham.update_hamiltonian(self.rho)

    def end_calculation(self):
        # Perform tasks once the density has converged

        if write_dens:
            print('Writing density to file')
            np.save('dens.npy', self.rho)

        if write_proj:
            print('Writing projectors to file')
            np.save('projector.npy', self.ham.coeffs)

        # Calculate and print total energy
        print('Total energy: ', self.ham.total_energy())
        if hubbard:
            print('Hubbard occupancy: ', self.hubbard_occ)

        print("Energy eigenvalues")
        for i in range(n_electrons):
            print('Occupancy: 1.0      Energy : ', self.ham.eigvals[i])
        for i in range(n_electrons + 1, n_electrons + 5):
            print('Occupancy: 0.0      Energy : ', self.ham.eigvals[i])

        # Plot charge density
        plot_density(self.rho)
