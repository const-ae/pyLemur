import numpy as np


def grassmann_map(x, base_point):
    if base_point.shape[0] == 0 or base_point.shape[1] == 0:
        return base_point
    elif np.isnan(x).any():
        # Return an object with the same shape as x filled with nan
        return np.full(x.shape, np.nan)
    else:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return (base_point @ vt.T) @ np.diag(np.cos(s)) @ vt + u @ np.diag(np.sin(s)) @ vt


def grassmann_log(p, q):
    n = p.shape[0]
    k = p.shape[1]

    if n == 0 or k == 0:
        return p
    else:
        z = q.T @ p
        At = q.T - z @ p.T
        # Translate `lm.fit(z, At)$coefficients` to python
        Bt = np.linalg.lstsq(z, At, rcond=None)[0]
        u, s, vt = np.linalg.svd(Bt.T, full_matrices=True)
        u = u[:, :k]
        s = s[:k]
        vt = vt[:k, :]
        return u @ np.diag(np.arctan(s)) @ vt


def grassmann_project(x):
    return np.linalg.qr(x)[0]


def grassmann_project_tangent(x, base_point):
    return x - base_point @ base_point.T @ x


def grassmann_random_point(n, k):
    x = np.random.randn(n, k)
    return grassmann_project(x)


def grassmann_random_tangent(base_point):
    x = np.random.randn(*base_point.shape)
    return grassmann_project_tangent(x, base_point)


def grassmann_angle_from_tangent(x, normalized=True):
    thetas = np.linalg.svd(x, full_matrices=True, compute_uv=False) / np.pi * 180
    if normalized:
        return np.minimum(thetas, 180 - thetas).max()
    else:
        return thetas[0]


def grassmann_angle_from_point(x, y):
    return grassmann_angle_from_tangent(grassmann_log(y, x))
