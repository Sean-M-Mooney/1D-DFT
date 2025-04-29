import numpy as np
from config import x, L, N_basis

# Our basis functions are numerically calculated and then stored in a list, makes indexing in the code easier.
basis = []

# Complex exponential basis function
def func(n):
    return (1/np.sqrt(L)) * np.exp(1j * 2 * np.pi * (n) * x / L)

# Calculates basis functions as numerical arrays and stores them
def make_basis():
    for i in range(int(-N_basis/2), int(N_basis/2) + 1):
        basis.append(func(i))

make_basis()
