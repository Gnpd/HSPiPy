from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .utils import compute_datafit, hansen_distance, hansen_center_distance

class BaseHSPLoss:
    """Base class for HSP loss functions."""
    def __init__(self, size_factor=None, minimize=True):
        self.size_factor = size_factor
        self.minimize = minimize

    def _apply_size_factor(self, datafit, radii):
        """Apply size factor normalization to the datafit."""
        if self.size_factor is None:
            return datafit
        elif self.size_factor == "n_solvents":
            m = len(self.y)
            return datafit * np.prod(radii) ** (-1 / m)
        else:
            m = float(self.size_factor)
            return datafit * np.prod(radii) ** (-1 / m)

    def _finalize_output(self, of_value):
        """Convert to minimization or maximization objective."""
        return 1 - float(of_value) if self.minimize else float(of_value)

class HSPSingleSphereLoss(BaseHSPLoss):
    """Loss function for fitting a single HSP sphere via optimization."""

    def __call__(self, HSP: NDArray, X: NDArray, y: NDArray) -> float | NDArray:
        self.y = y
        if np.ndim(HSP) == 2:
            return self._batch_call(HSP, X, y)
        D, P, H, R = HSP
        center = np.array([D, P, H])
        dist = hansen_distance(X, center)

        datafit = compute_datafit(dist, R, y)

        objective = self._apply_size_factor(datafit, [R])

        return self._finalize_output(objective)

    def _batch_call(self, HSP: NDArray, X: NDArray, y: NDArray) -> NDArray:
        """Evaluate all population members at once.

        Used by scipy's vectorized differential_evolution.
        HSP: (4, S) → returns (S,)
        """
        D, P, H, R = HSP[0], HSP[1], HSP[2], HSP[3]  # each (S,)
        inside = (y == 1)
        outside = ~inside

        dD = X[:, None, 0] - D[None, :]
        dP = X[:, None, 1] - P[None, :]
        dH = X[:, None, 2] - H[None, :]
        dist = np.sqrt(4.0 * dD**2 + dP**2 + dH**2)  # (n_samples, S)

        delta = dist - R[None, :]  # (n_samples, S)
        fitness = np.ones_like(delta)
        fitness[inside[:, None] & (delta > 0)] = np.exp(-delta[inside[:, None] & (delta > 0)])
        fitness[outside[:, None] & (delta < 0)] = np.exp(delta[outside[:, None] & (delta < 0)])
        fitness = np.maximum(fitness, 1e-12)
        datafit_all = np.exp(np.mean(np.log(fitness), axis=0))  # (S,)

        if self.size_factor is not None:
            m = len(y) if self.size_factor == "n_solvents" else float(self.size_factor)
            datafit_all = datafit_all * R ** (-1.0 / m)

        return 1.0 - datafit_all  # (S,)

class HSPDoubleSphereLoss(BaseHSPLoss):
    """Loss function for fitting two HSP spheres via optimization."""

    def __call__(self, HSP: NDArray, X: NDArray, y: NDArray) -> float | NDArray:
        self.y = y
        if np.ndim(HSP) == 2:
            return self._batch_call(HSP, X, y)
        D1, P1, H1, R1 = HSP[0:4]
        D2, P2, H2, R2 = HSP[4:8]

        center1 = np.array([D1, P1, H1])
        center2 = np.array([D2, P2, H2])

        if self._violates_constraints(center1, center2, R1, R2, X, y):
            return 1e10

        dist1 = hansen_distance(X, center1)
        dist2 = hansen_distance(X, center2)
        distances = np.column_stack([dist1, dist2])

        datafit = compute_datafit(distances, [R1, R2], y)

        objective = self._apply_size_factor(datafit, [R1, R2])

        return self._finalize_output(objective)

    def _batch_call(self, HSP: NDArray, X: NDArray, y: NDArray) -> NDArray:
        """Evaluate all population members at once.

        Used by scipy's vectorized differential_evolution.
        HSP: (8, S) → returns (S,)
        """
        D1, P1, H1, R1 = HSP[0], HSP[1], HSP[2], HSP[3]  # each (S,)
        D2, P2, H2, R2 = HSP[4], HSP[5], HSP[6], HSP[7]

        inside = (y == 1)
        outside = ~inside

        def _batch_dist(D, P, H):
            dD = X[:, None, 0] - D[None, :]
            dP = X[:, None, 1] - P[None, :]
            dH = X[:, None, 2] - H[None, :]
            return np.sqrt(4.0 * dD**2 + dP**2 + dH**2)  # (n_samples, S)

        dist1 = _batch_dist(D1, P1, H1)
        dist2 = _batch_dist(D2, P2, H2)
        delta1 = dist1 - R1[None, :]  # (n_samples, S)
        delta2 = dist2 - R2[None, :]

        fit1 = np.ones_like(delta1)
        fit1[inside[:, None] & (delta1 > 0)] = np.exp(-delta1[inside[:, None] & (delta1 > 0)])
        fit1[outside[:, None] & (delta1 < 0)] = np.exp(delta1[outside[:, None] & (delta1 < 0)])

        fit2 = np.ones_like(delta2)
        fit2[inside[:, None] & (delta2 > 0)] = np.exp(-delta2[inside[:, None] & (delta2 > 0)])
        fit2[outside[:, None] & (delta2 < 0)] = np.exp(delta2[outside[:, None] & (delta2 < 0)])

        fitness = np.minimum(fit1, fit2)
        correct = (inside[:, None] & (delta1 <= 0)) | (inside[:, None] & (delta2 <= 0))
        fitness[correct] = 1.0
        fitness = np.maximum(fitness, 1e-12)
        datafit_all = np.exp(np.mean(np.log(fitness), axis=0))  # (S,)

        # Constraint violations → large penalty
        dD_c = D1 - D2
        dP_c = P1 - P2
        dH_c = H1 - H2
        center_dist = np.sqrt(4.0 * dD_c**2 + dP_c**2 + dH_c**2)  # (S,)
        containment = (center_dist + np.minimum(R1, R2)) <= (np.maximum(R1, R2) + 1e-8)
        no_good = (
            ~np.any(inside[:, None] & (delta1 <= 0), axis=0) |
            ~np.any(inside[:, None] & (delta2 <= 0), axis=0)
        )

        result = 1.0 - datafit_all

        if self.size_factor is not None:
            m = len(y) if self.size_factor == "n_solvents" else float(self.size_factor)
            result = 1.0 - datafit_all * (R1 * R2) ** (-1.0 / m)

        result[containment | no_good] = 1e10
        return result

    def _violates_constraints(self, center1, center2, R1, R2, X, y):
        """Check if the double sphere configuration violates constraints."""
        center_distance = hansen_center_distance(center1, center2)
        min_radius, max_radius = min(R1, R2), max(R1, R2)
        if center_distance + min_radius <= max_radius:
            return True

        inside_mask = (y == 1)
        X_good = X[inside_mask]

        if len(X_good) == 0:
            return True

        dist1_good = hansen_distance(X_good, center1)
        dist2_good = hansen_distance(X_good, center2)

        sphere1_has_good = np.any(dist1_good <= R1)
        sphere2_has_good = np.any(dist2_good <= R2)

        return not (sphere1_has_good and sphere2_has_good)
