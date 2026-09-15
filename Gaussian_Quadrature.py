"""Quadrature functions copied unchanged from Gaussian_Quadrature.ipynb.

Numerical checks live in test_gaussian_quadrature.py.
"""

import numpy as np
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi


def gauss_radau_left(N, a=-1.0, b=1.0, tol=1e-12):
    
    """
    Left Gauss–Radau quadrature on [a,b] including x = -1.
    Based on:
        roots of (P_{N-1}(x) + P_N(x)) = 0
        w_i = (1 - x_i) / [N * P_{N-1}(x_i)]^2
        w_1 = 2 / N^2
    """
    # Legendre polynomials
    Pn   = Legendre.basis(N)
    Pn_1 = Legendre.basis(N-1)

    # Internal nodes: roots of P_{N-1}(x) + P_N(x)
    poly = Pn_1 + Pn
    roots = np.sort(poly.roots())

    # Remove any root numerically equal to -1
    roots = roots[roots > -1 + tol]

    # Combine with exact endpoint -1
    x_nodes = np.concatenate(([-1.0], roots))

    # Compute weights
    w = np.zeros_like(x_nodes)
    w[0]  = 2.0 / (N**2)
    w[1:] = (1 - roots) / ((N * Pn_1(roots))**2)

    # Map from [-1,1] → [a,b]
    t_nodes = 0.5 * (b - a) * (x_nodes + 1) + a
    w *= 0.5 * (b - a)
    return t_nodes, w


def gauss_radau_left_infinite(N):
    """
    Classical Left Gauss–Radau quadrature on [-1,1]
    where the endpoint -1 is already included in the roots
    of P_N + P_{N-1}, then mapped to [0,∞) using t = (1+τ)/(1−τ).
    """

    # Step 1: Legendre polynomials L_N and L_{N-1}
    L_N   = Legendre.basis(N)
    L_Nm1 = Legendre.basis(N-1)

    # Step 2: roots of L_N + L_{N-1}
    poly = L_N + L_Nm1
    tau_nodes = np.sort(poly.roots())   # includes -1 automatically
    tau_nodes[np.isclose(tau_nodes, -1, atol=1e-12)] = -1.0    # Force exact -1 endpoint

    # Step 3: classical Gauss–Radau weights on [-1,1]
    w_tau = np.zeros_like(tau_nodes)

    # Endpoint weight at τ = -1
    w_tau[0] = 2.0 / (N**2)

    # Interior weights
    w_tau[1:] = (1 - tau_nodes[1:]) / ((N**2) * (L_Nm1(tau_nodes[1:]))**2)

    # Step 4: infinite-horizon mapping τ → t
    t_nodes = (1 + tau_nodes) / (1 - tau_nodes)

    # Step 5: transformed weights: multiply by dt/dτ
    dt_dtau = 2.0 / (1 - tau_nodes)**2
    w_t = w_tau * dt_dtau

    return t_nodes, w_t


def gauss_radau_left_log_mapping_shen(N, s=1.0):
    """
    Left Gauss–Radau quadrature on [-1,1], mapped to [0,+∞)
    using Shen & Wang (2009) logarithmic mapping (Eq. 2.16):

        t = (s/2)*ln((3+tau)/(1-tau))

    Much smoother than the log(4/(1-tau)^2) mapping.
    """

    # Step 1
    L_N   = Legendre.basis(N)
    L_Np1 = Legendre.basis(N-1)

    # Step 2: nodes
    poly = L_N + L_Np1
    tau_nodes = np.sort(poly.roots())
    tau_nodes[np.isclose(tau_nodes, -1, atol=1e-12)] = -1.0

    # Step 3: weights
    w_tau = np.zeros_like(tau_nodes)
    w_tau[0] = 2.0 / N**2
    w_tau[1:] = (1 - tau_nodes[1:]) / (N**2 * (L_N(tau_nodes[1:]))**2)

    # Step 4: Shen–Wang log mapping
    t_nodes = (s/2.0) * np.log((3 + tau_nodes) / (1 - tau_nodes))

    # Step 5: dt/dtau = (2s)/((3+tau)*(1-tau))
    dt_dtau = (2.0 * s) / ((3 + tau_nodes) * (1 - tau_nodes))

    w_t = w_tau * dt_dtau

    return t_nodes, w_t


def gauss_radau_left_log_mapping(N):
    """
    Left Gauss–Radau quadrature on [-1,1] using the nodes
    from roots of L_N + L_{N+1}, mapped to t = log(4 / (1 - tau)^2).
    """

    # Step 1: Legendre polynomials
    L_N   = Legendre.basis(N)
    L_Np1 = Legendre.basis(N-1)

    # Step 2: roots of L_N + L_{N+1}  (includes tau = -1)
    poly = L_N + L_Np1
    tau_nodes = np.sort(poly.roots())

    # force exact -1 for safety
    tau_nodes[np.isclose(tau_nodes, -1, atol=1e-12)] = -1.0

    # Step 3: classical Gauss–Radau weights on [-1,1]
    w_tau = np.zeros_like(tau_nodes)

    # weight at tau = -1
    w_tau[0] = 2.0 / (N**2)

    # interior weights
    w_tau[1:] = (1 - tau_nodes[1:]) / ((N**2) * (L_N(tau_nodes[1:]))**2)

    # Step 4: LOG mapping:  t = log(4 / (1 - tau)^2)
    t_nodes = np.log(4.0 / (1 - tau_nodes)**2)

    # Step 5: weight mapping: dt/dtau = 2 / (1 - tau)
    dt_dtau = 2.0 / (1 - tau_nodes)

    w_t = w_tau * dt_dtau

    return t_nodes, w_t


def gauss_radau_right(N, a=-1.0, b=1.0, tol=1e-12):

    """
    Right Gauss–Radau quadrature on [a,b] including x = +1.
    Based on:
        roots of (P_N(x) - P_{N-1}(x)) = 0
        w_i = (1 + x_i) / [N * P_{N-1}(x_i)]^2
        w_N = 2 / N^2
    """
    
    # Legendre polynomials
    Pn   = Legendre.basis(N)
    Pn_1 = Legendre.basis(N-1)

    # Internal nodes: roots of P_N(x) - P_{N-1}(x)
    poly = Pn - Pn_1
    roots = np.sort(poly.roots())

    #  Remove any root numerically equal to +1
    roots = roots[roots < 1 - tol]

    #Combine with exact endpoint +1
    x_nodes = np.concatenate((roots, [1.0]))
    print(x_nodes[-1])

    # Compute weights
    w = np.zeros_like(x_nodes)
    w[-1] = 2.0 / (N**2)
    w[:-1] = (1 + roots) / ((N * Pn_1(roots))**2)

    # Map from [-1,1] → [a,b]
    t_nodes = 0.5 * (b - a) * (x_nodes + 1) + a
    w *= 0.5 * (b - a)
    return t_nodes, w


def gauss_lobatto_nodes_weights(N, a=-1.0, b=1.0):

    """
    Compute N Gauss–Lobatto nodes and weights on [a,b].
    N = total number of nodes (including endpoints).
    """
    
    if N < 2:
        raise ValueError("Need at least 2 Lobatto nodes")

    n = N - 1  # Polynomial degree (one less than number of nodes)
    Pn = Legendre.basis(n)
    x_int = np.sort(Pn.deriv().roots())  #  Roots of the derivate of P(N-1) or P'(N-1) 
    x_nodes = np.concatenate(([-1.0], x_int, [1.0]))

    w = np.zeros(N)
    w[0] = w[-1] = 2.0 / (n * (n + 1))
    for i in range(1, N-1):
        xi = x_nodes[i]
        w[i] = 2.0 / (n * (n + 1) * (Pn(xi)**2))

    # Map to [a, b]
    t_nodes = 0.5 * (b - a) * (x_nodes + 1) + a
    w = 0.5 * (b - a) * w
    return t_nodes, w


def gauss_legendre_nodes_weights(N, a=0.0, b=1.0):
    """
    N-point Gauss–Legendre quadrature on [a,b].
    Returns nodes t and weights w.
    """
    x, w = leggauss(N)               # nodes/weights on [-1,1]
    t = 0.5*(b-a)*(x+1) + a          # map nodes to [a,b]
    w = 0.5*(b-a)*w                  # scale weights
    return t, w


def LGR_nodes_jacobi(N):
    if N == 1:
        colloc = np.array([-1.0], dtype=float)
        w_lgr  = np.array([2.0], dtype=float)
    else:
        x, w_gj = roots_jacobi(N-1, 0.0, 1.0)
        x   = np.sort(np.asarray(x, float))
        wgj = np.asarray(w_gj, float)

        colloc = np.concatenate(([-1.0], x))
        w_int  = wgj / (1.0 + x)
        w_lgr  = np.concatenate(([2.0/(N**2)], w_int))

    tau_s = np.concatenate((colloc, [1.0]))   # include endpoint

    return colloc, w_lgr, tau_s


def Gauss_radau_left_jacobi(N):
    # SciPy weight: (1 - x)^alpha (1 + x)^beta
    # Left Radau → weight includes (1 + x) → beta = 1
    # Right Radau → weight includes (1 - x) → beta = 0
    
    from scipy.special import roots_jacobi
    import numpy as np

    # Step 1: Jacobi interior nodes and weights
    x_int, w_jac = roots_jacobi(N-1, alpha=0, beta=1)

    # Step 2: Endpoint and its weight
    x0 = -1.0
    w0 = 2.0 / N**2

    # Step 3: Convert Jacobi weights → Radau weights
    w_int = w_jac / (1 + x_int)

    # Step 4: Assemble final nodes and weights
    x = np.concatenate(([x0], x_int))
    w = np.concatenate(([w0], w_int))
    x_full = np.concatenate((x,[1]))

    return x, x_full, w


def Gauss_radau_right_jacobi(N):
    # SciPy weight: (1 - x)^alpha (1 + x)^beta
    # Left Radau → weight includes (1 + x) → beta = 1
    # Right Radau → weight includes (1 - x) → beta = 0
    
    from scipy.special import roots_jacobi
    import numpy as np

    # Step 1: Jacobi interior nodes and weights
    x_int, w_jac = roots_jacobi(N-1, alpha=1, beta=0)

    # Step 2: Endpoint and its weight
    x_end = 1.0
    w_end = 2.0 / N**2

    # Step 3: Convert Jacobi weights → Radau weights
    w_int = w_jac / (1 - x_int)

    # Step 4: Assemble final nodes and weights
    x = np.concatenate((x_int,[x_end]))
    w = np.concatenate((w_int, [w_end]))
    x_full = np.concatenate(([-1],x))

    return x, x_full, w
