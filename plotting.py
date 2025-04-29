import matplotlib.pyplot as plt
from config import x

# Plotting functions

def plot_density(rho):
    plt.plot(x, rho)
    plt.ylim(0, max(max(rho) + .2, 1))
    plt.xlabel('x')
    plt.ylabel('density')
    plt.show()

def plot_potential(v):
    plt.plot(x, v)
    plt.ylabel('potential')
    plt.xlabel('x')
    plt.show()
