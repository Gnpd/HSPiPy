import numpy as np

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
    """Custom loss for single HSP sphere."""
    def __call__(self, HSP, X, y):
        self.y = y
        D, P, H, R = HSP
        center = np.array([D, P, H])
        dist = hansen_distance(X, center)

        datafit = compute_datafit(dist, R, y)

        objective = self._apply_size_factor(datafit, [R])

        return self._finalize_output(objective)

class HSPDoubleSphereLoss(BaseHSPLoss):
    """Custom loss for double HSP spheres."""
    def __call__(self, HSP, X, y):
        self.y = y
        D1, P1, H1, R1 = HSP[0:4]
        D2, P2, H2, R2 = HSP[4:8]

        center1 = np.array([D1, P1, H1])
        center2 = np.array([D2, P2, H2])

        # Check constraints
        if self._violates_constraints(center1, center2, R1, R2, X, y):
            return 1e10  # Heavy penalty for constraint violation

        # Compute distances for all samples to both spheres
        dist1 = hansen_distance(X, center1)
        dist2 = hansen_distance(X, center2)
        distances = np.column_stack([dist1, dist2])

        # Calculate datafit
        datafit = compute_datafit(distances, [R1, R2], y)
        
        objective = self._apply_size_factor(datafit, [R1, R2])

        return self._finalize_output(objective)

    
    def _violates_constraints(self, center1, center2, R1, R2, X, y):
        """Check if the double sphere configuration violates constraints."""
        # Prevent one sphere from being completely inside another
        center_distance = hansen_center_distance(center1, center2)
        min_radius, max_radius = min(R1, R2), max(R1, R2)
        if center_distance + min_radius <= max_radius:
            return True
        
        # Check if each sphere has at least one good solvent inside
        inside_mask = (y == 1)
        X_good = X[inside_mask]
        
        if len(X_good) == 0:
            return True
            
        dist1_good = hansen_distance(X_good, center1)
        dist2_good = hansen_distance(X_good, center2)
        
        sphere1_has_good = np.any(dist1_good <= R1)
        sphere2_has_good = np.any(dist2_good <= R2)
        
        return not (sphere1_has_good and sphere2_has_good)