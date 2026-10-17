# Adapted from: https://github.com/ml-jku/DiffUCO/blob/main/IsingTheoryBaselines/IsingTheory.py

import numpy as np

def calculate_ising_internal_energy( beta, L):
    """
    Calculate the internal energy per spin of the NxN lattice.
    Uses U = -d(log Z)/d(beta) / L^2 via central finite differences.

    https://journals.aps.org/pr/pdf/10.1103/PhysRev.185.832
    """
    eps = beta * 1e-5
    log_Z_plus,  _ = calculate_ising_partition_function(beta + eps, L)
    log_Z_minus, _ = calculate_ising_partition_function(beta - eps, L)
    return -(log_Z_plus - log_Z_minus) / (2 * eps) / L ** 2


def calculate_ising_free_energy( beta, L):
    """
    Calculate the free energy per spin of the NxN lattice
    """
    log_Z, _ = calculate_ising_partition_function(beta, L)
    return -1 / beta * log_Z / L ** 2


def calculate_ising_partition_function( beta, L):
    """
    Calculate the partition function of the NxN lattice (log-space to avoid overflow).

    https://journals.aps.org/pr/pdf/10.1103/PhysRev.185.832
    """
    log_Z_1 = sum(np.log(2) + np.log(np.cosh(0.5 * L * calculate_gamma(2*r+1, beta, L)))
                  for r in range(L))
    sign_Z_1 = 1.0
    log_Z_2, sign_Z_2 = _log_abs_prod_sinh([2*r+1 for r in range(L)], beta, L)
    log_Z_3 = sum(np.log(2) + np.log(np.cosh(0.5 * L * calculate_gamma(2*r, beta, L)))
                  for r in range(L))
    sign_Z_3 = 1.0
    log_Z_4, sign_Z_4 = _log_abs_prod_sinh([2*r for r in range(L)], beta, L)

    log_prefactor = np.log(0.5) + (L**2 / 2) * np.log(2 * np.sinh(2 * beta))

    # Signed log-sum-exp: sum sign_i * exp(log_Z_i)
    log_Zs = np.array([log_Z_1, log_Z_2, log_Z_3, log_Z_4])
    signs  = np.array([sign_Z_1, sign_Z_2, sign_Z_3, sign_Z_4])
    base = np.max(log_Zs)
    total = np.sum(signs * np.exp(log_Zs - base))
    log_Z = log_prefactor + base + np.log(np.abs(total))

    return log_Z, (log_Z_1, log_Z_2, log_Z_3, log_Z_4)


def _log_abs_prod_sinh(r_vals, beta, L):
    """Returns (log|product of 2*sinh(L*gamma/2)|, sign) for the given r values."""
    log_abs = 0.0
    sign = 1.0
    for r in r_vals:
        v = np.sinh(0.5 * L * calculate_gamma(r, beta, L))
        log_abs += np.log(2) + np.log(np.abs(v))
        if v < 0:
            sign *= -1
    return log_abs, sign


def calculate_gamma( r, beta, L):
    if r == 0:
        gamma = 2 * beta + np.log(np.tanh(beta, dtype=np.float64))
    else:
        c_r = np.cosh(2 * beta, dtype=np.float64) * coth(2 * beta) - np.cos(np.pi * r / L)
        gamma = np.log(c_r + np.sqrt(c_r ** 2 - 1))
    return gamma


def calculate_gamma_prime( r, beta, L):
    if r == 0:
        gamma_prime = csch(beta)/np.cosh(beta) + 2
    else:
        cl = np.cosh(2 * beta, dtype=np.float64) * coth(2 * beta) - np.cos(np.pi * r / L)

        nenner = 2*(np.cosh(2*beta) - coth(2*beta) * csch(2*beta))
        zaehler = np.sqrt((cl)**2 -1)
        gamma_prime = nenner/zaehler
    return gamma_prime

def calculate_ising_entropy(beta, internal_energy, free_energy):
    """
    Calculate the entropy of the NxN lattice
    """
    return (internal_energy - free_energy) / (1 / beta)

def coth( x):
    """
    Coth function
    """
    return np.cosh(x, dtype=np.float64) / np.sinh(x, dtype=np.float64)


def csch( x):
    """
    Csch function
    """
    return 1 / np.sinh(x, dtype=np.float64)

def make_plots():
    from matplotlib import pyplot as plt

    L = 16
    Ts = np.linspace(0.1, 20, 100, endpoint=True)
    energy_list = []
    free_energy_list = []
    entropies = []
    for T in Ts:
        beta = 1/T
        energy = calculate_ising_internal_energy(beta, L)
        energy_list.append(energy)
        free_energy = calculate_ising_free_energy(beta, L)
        free_energy_list.append(free_energy)
        entropies.append(calculate_ising_entropy(beta, energy, free_energy))

    plt.figure()
    plt.title("Entropy")
    plt.plot(Ts, entropies)

    plt.show()

    plt.figure()
    plt.title("Energy")
    plt.plot(Ts, energy_list)

    plt.show()

    plt.figure()
    plt.title("Free Energy")
    plt.plot(Ts, free_energy_list)

    plt.show()

if __name__== "__main__":

    # NOTE: beta = 2 * ising_sigma in our setting
    betas = [0.4, 0.2, 0.6]
    Ls = [5,15,25, 32]
    # betas = [0.2, 0.4407]
    # Ls = [10]
    for L in Ls:
        for beta in betas:
            print("Lattice Size", L, "x", L, "Beta:", beta)
            print("Free Energy", calculate_ising_free_energy(beta, L) * beta*L*L)
            print("Free Energy", calculate_ising_free_energy(beta, L) * beta*1000)
            # print("Internal Energy", calculate_ising_internal_energy(beta, L))
            # print("Entropy", calculate_ising_entropy(beta, calculate_ising_internal_energy(beta, L), calculate_ising_free_energy(beta, L)))
            print("")