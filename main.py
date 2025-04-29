from ks_solver import KS_solver
from hamiltonian import Hamiltonian

def main():
    ham = Hamiltonian()
    solver = KS_solver(ham)
    solver.self_consistent_loop()

if __name__ == "__main__":
    main()
