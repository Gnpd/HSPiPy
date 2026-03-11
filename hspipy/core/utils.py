from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hansen_center_distance(center1: NDArray, center2: NDArray) -> float:
    """Weighted Euclidean distance between two HSP centers (4·ΔD² + ΔP² + ΔH²)^½."""
    d = np.asarray(center1) - np.asarray(center2)
    return float(np.sqrt(4.0 * d[0]**2 + d[1]**2 + d[2]**2))


def hansen_distance(X: NDArray, center: NDArray) -> NDArray:
    """
    Compute Hansen distance from each row in X to a single center.

    The Hansen distance formula is: sqrt(4·ΔD² + ΔP² + ΔH²), where the
    factor of 4 on the dispersion component (D) reflects its empirically
    larger contribution to solubility differences.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 3)
        Solvent coordinates (D, P, H).
    center : ndarray of shape (3,)
        Sphere center (D, P, H).

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    d = X - center[None, :]
    return np.sqrt(4.0 * d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2)


def hansen_pairwise_distance(solvents: NDArray) -> NDArray:
    """
    Compute pairwise Hansen distances for a set of solvents.

    Parameters
    ----------
    solvents : ndarray of shape (n, 3)
        Solvent coordinates (D, P, H).

    Returns
    -------
    ndarray of shape (n, n)
        Symmetric matrix of pairwise Hansen distances.
    """
    solvents = np.asarray(solvents, dtype=float)
    diffs = solvents[:, None, :] - solvents[None, :, :]
    dD = diffs[..., 0]
    dP = diffs[..., 1]
    dH = diffs[..., 2]
    return np.sqrt(4.0 * dD**2 + dP**2 + dH**2)


def compute_datafit(distances: NDArray, radii: NDArray | float, y: NDArray) -> float:
    """
    Compute DATAFIT: the geometric mean of per-sample fitness scores.

    Each sample gets a fitness value in (0, 1]:
    - Correctly classified samples (good inside sphere, bad outside) → 1.0
    - Good solvent outside sphere → exp(R - dist) < 1  (penalises miss distance)
    - Bad solvent inside sphere  → exp(dist - R) < 1  (penalises intrusion depth)

    The geometric mean (exp(mean(log(fitness)))) rewards solutions that are
    uniformly good across all solvents rather than those that excel on a
    subset while failing elsewhere.

    Parameters
    ----------
    distances : ndarray of shape (n_samples,) or (n_samples, n_spheres)
        Hansen distances from each sample to each sphere center.
    radii : float or array-like of shape (n_spheres,)
        Sphere radius/radii.
    y : array-like of shape (n_samples,)
        Binary labels (1 = good solvent, 0 = bad solvent).

    Returns
    -------
    float
        DATAFIT value in (0, 1]. A value of 1.0 indicates perfect classification.
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