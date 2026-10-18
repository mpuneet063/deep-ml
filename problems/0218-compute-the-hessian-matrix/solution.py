from typing import Callable

def compute_hessian(
    f: Callable[[list[float]], float],
    point: list[float],
    h: float = 1e-5
) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using finite differences.

    Args:
        f: A scalar function that takes a list of floats and returns a float
        point: The point at which to compute the Hessian (list of coordinates)
        h: Step size for finite differences (default: 1e-5)

    Returns:
        The Hessian matrix as a list of lists (n x n where n = len(point))
    """
    n = len(point)
    H = [[0.0 for _ in range(n)] for _ in range(n)]

    f0 = f(point)

    for i in range(n):
        # Second-order partial derivatives (diagonal elements)
        x_plus = point.copy()
        x_minus = point.copy()

        x_plus[i] += h
        x_minus[i] -= h

        H[i][i] = (f(x_plus) - 2 * f0 + f(x_minus)) / (h ** 2)

        # Mixed partial derivatives (off-diagonal elements)
        for j in range(i + 1, n):
            x_pp = point.copy()
            x_pm = point.copy()
            x_mp = point.copy()
            x_mm = point.copy()

            x_pp[i] += h
            x_pp[j] += h

            x_pm[i] += h
            x_pm[j] -= h

            x_mp[i] -= h
            x_mp[j] += h

            x_mm[i] -= h
            x_mm[j] -= h

            value = (
                f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)
            ) / (4 * h ** 2)

            H[i][j] = value
            H[j][i] = value

    return H