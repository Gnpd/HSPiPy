import numpy as np

def hansen_center_distance(center1, center2):
    d = np.asarray(center1) - np.asarray(center2)
    return np.sqrt(4.0 * d[0]**2 + d[1]**2 + d[2]**2)

def hansen_distance(X, center):
    """
    Compute Hansen distance from each row in X to a single center.
    X: shape (n_samples, 3)
    center: shape (3,)
    Returns: shape (n_samples,)
    """
    d = X - center[None, :]
    return np.sqrt(4.0 * d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2)

def hansen_pairwise_distance(solvents):
    """
    Compute pairwise Hansen distances for a set of solvents.
    solvents: shape (n,3)
    Returns: (n, n) matrix of Hansen distances.
    """
    solvents = np.asarray(solvents, dtype=float)
    diffs = solvents[:, None, :] - solvents[None, :, :]
    dD = diffs[..., 0]
    dP = diffs[..., 1]
    dH = diffs[..., 2]
    return np.sqrt(4.0 * dD**2 + dP**2 + dH**2)

def compute_datafit(distances, radii, y):
    """
    Compute geometric mean fitness for 1 or more spheres.
    distances: shape (n_samples,) for 1 sphere, (n_samples, n_spheres) for multi-sphere
    radii: float for 1 sphere, array-like for multi-sphere
    y: array-like, binary labels (1=inside, 0=outside)
    Returns: float (DATAFIT)
    """
    inside = (y == 1)
    outside = ~inside

    # Ensure shapes
    distances = np.atleast_2d(distances)
    if distances.shape[0] == 1:
        distances = distances.T
    radii = np.asarray(radii).reshape((1, -1))

    delta = distances - radii  # (n_samples, n_spheres)
    fitness_matrix = np.ones_like(distances)

    # good solvents outside -> exp(R - dist) = exp(-delta) when delta>0
    good_out_mask = inside[:, None] & (distances > radii)
    if np.any(good_out_mask):
        fitness_matrix[good_out_mask] = np.exp(-delta[good_out_mask])

    # bad solvents inside -> exp(dist - R) = exp(delta) when delta<0
    bad_in_mask = outside[:, None] & (distances < radii)
    if np.any(bad_in_mask):
        fitness_matrix[bad_in_mask] = np.exp(delta[bad_in_mask])

    # per-sample Fi is min across spheres (vectorized)
    fitness = np.min(fitness_matrix, axis=1)

    # inside rule: if any sphere correctly classifies a 'good' (dist <= R) => Fi=1
    if np.any(inside):
        correct_any = np.any(inside[:, None] & (distances <= radii), axis=1)
        fitness[correct_any] = 1.0

    # numerical safety for geometric mean
    fitness = np.maximum(fitness, 1e-12)
    datafit = float(np.exp(np.mean(np.log(fitness))))
    return datafit