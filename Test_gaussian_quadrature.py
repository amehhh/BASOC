"""Simple checks. Run with: python test_gaussian_quadrature.py"""

import numpy as np
import Gaussian_Quadrature as gq

# 1. Finite interval: integral of exp(x) on [-1, 1].
N = 10
exact = np.exp(1) - np.exp(-1)

for rule in [gq.gauss_radau_left, gq.gauss_radau_right,
             gq.gauss_lobatto_nodes_weights, gq.gauss_legendre_nodes_weights]:
    x, w = rule(N, a=-1, b=1)
    integral = np.sum(np.exp(x) * w)
    print(rule.__name__, "integral:", integral, "exact:", exact)
    assert len(x) == len(w) == N
    assert np.isclose(np.sum(w), 2)
    assert np.isclose(integral, exact, rtol=1e-8, atol=1e-10)

# 2. Check which endpoints are included.
x, w = gq.gauss_radau_left(5)
assert x[0] == -1 and x[-1] < 1

x, w = gq.gauss_radau_right(5)
assert x[0] > -1 and x[-1] == 1

x, w = gq.gauss_lobatto_nodes_weights(5)
assert x[0] == -1 and x[-1] == 1

x, w = gq.gauss_legendre_nodes_weights(5, a=-1, b=1)
assert x[0] > -1 and x[-1] < 1
print("Endpoint checks passed.")

# 3. Jacobi versions should give the same nodes and weights.
x_left, w_left = gq.gauss_radau_left(5)
x, x_full, w = gq.Gauss_radau_left_jacobi(5)
assert np.allclose(x, x_left) and np.allclose(w, w_left)
assert np.array_equal(x_full[:-1], x) and x_full[-1] == 1

x, w, x_full = gq.LGR_nodes_jacobi(5)
assert np.allclose(x, x_left) and np.allclose(w, w_left)
assert np.array_equal(x_full[:-1], x) and x_full[-1] == 1

x_right, w_right = gq.gauss_radau_right(5)
x, x_full, w = gq.Gauss_radau_right_jacobi(5)
assert np.allclose(x, x_right) and np.allclose(w, w_right)
assert x_full[0] == -1 and np.array_equal(x_full[1:], x)
print("Jacobi comparisons passed.")

# 4. Infinite interval: integral of exp(-t) on [0, infinity] is 1.
for rule in [gq.gauss_radau_left_infinite, gq.gauss_radau_left_log_mapping]:
    t, w = rule(200)
    integral = np.sum(np.exp(-t) * w)
    print(rule.__name__, "integral:", integral, "exact:", 1)
    assert np.isclose(integral, 1, rtol=1e-8, atol=1e-10)

# 5. Same integral with the Shen mapping, using your N=200, s=50.
t, w = gq.gauss_radau_left_log_mapping_shen(200, s=50)
integral = np.sum(np.exp(-t) * w)
print("Shen mapping integral:", integral, "exact:", 1)
assert np.isclose(integral, 1, rtol=1e-8, atol=1e-10), "Shen integral differs from 1"
