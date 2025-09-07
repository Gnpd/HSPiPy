import numpy as np

class HSPSingleSphereLoss:
    """Custom loss for single HSP sphere."""
    def __init__(self, inside_limit= 1, size_factor= None):
        self.inside_limit = inside_limit
        self.size_factor = size_factor

    def __call__(self, HSP, X, y):
        Ds, Ps, Hs, Rs = HSP[0], HSP[1], HSP[2], HSP[3]

        # Vectorized distance calculation
        dD = X[:, 0] - Ds
        dP = X[:, 1] - Ps
        dH = X[:, 2] - Hs
        dist = np.sqrt(4.0 * dD**2 + dP**2 + dH**2)

        # Initialize penalties
        fi = np.ones_like(dist)

        inside = (y == 1)
        bad_inside = inside & (dist >= Rs)
        fi[bad_inside] = np.exp(Rs - dist[bad_inside])
      
        outside = (y == 0)
        good_outside = outside & (dist < Rs)
        fi[good_outside] = np.exp(dist[good_outside] - Rs)

        # DATAFIT = np.prod(fi) ** (1 / len(fi))
        # Safe geometric mean
        DATAFIT = np.exp(np.mean(np.log(fi)))

        # Handle size factor logic
        if self.size_factor is None:
            OF = DATAFIT
        elif self.size_factor == "n_solvents":
            m = len(y)
            OF = DATAFIT * Rs ** (-1 / m)
        else:
            m = float(self.size_factor)
            OF = DATAFIT * Rs ** (-1 / m)
    
        return 1 - float(OF)

class HSPDoubleSphereLoss:
    """Custom loss for double HSP spheres."""
    def __init__(self, inside_limit=1, size_factor="n_solvents"):
        self.inside_limit = inside_limit
        self.size_factor = size_factor

    def __call__(self, HSP, X, y):
        D1, P1, H1, R1 = HSP[0], HSP[1], HSP[2], HSP[3]
        D2, P2, H2, R2 = HSP[4], HSP[5], HSP[6], HSP[7]

        # Check for sphere containment constraint
        # Distance between centers
        center_distance = np.sqrt(4.0 * (D1 - D2)**2 + (P1 - P2)**2 + (H1 - H2)**2)
        
        # Prevent one sphere from being completely inside another
        # If center_distance + smaller_radius <= larger_radius, one is inside the other
        min_radius = min(R1, R2)
        max_radius = max(R1, R2)
        
        if center_distance + min_radius <= max_radius:
            return 1e6  # Heavy penalty for containment
        
        # Check if each sphere has at least one good solvent inside
        inside_mask = (y == 1)
        X_good = X[inside_mask]
        
        if len(X_good) > 0:
            # Calculate distances for good solvents to each sphere
            dD1_good = X_good[:, 0] - D1
            dP1_good = X_good[:, 1] - P1
            dH1_good = X_good[:, 2] - H1
            dist1_good = np.sqrt(4.0 * dD1_good**2 + dP1_good**2 + dH1_good**2)
            
            dD2_good = X_good[:, 0] - D2
            dP2_good = X_good[:, 1] - P2
            dH2_good = X_good[:, 2] - H2
            dist2_good = np.sqrt(4.0 * dD2_good**2 + dP2_good**2 + dH2_good**2)
            
            # Check if any good solvent is inside each sphere
            sphere1_has_good = np.any(dist1_good <= R1)
            sphere2_has_good = np.any(dist2_good <= R2)
            
            # Penalize if either sphere has no good solvents
            if not sphere1_has_good or not sphere2_has_good:
                return 1e5  # Heavy penalty for empty spheres
        else:
            # If no good solvents at all, return heavy penalty
            return 1e6

        # Vectorized distance calculation sphere 1
        dD1 = X[:, 0] - D1
        dP1 = X[:, 1] - P1
        dH1 = X[:, 2] - H1
        dist1 = np.sqrt(4.0 * dD1**2 + dP1**2 + dH1**2)

        # Vectorized distance calculation sphere 2
        dD2 = X[:, 0] - D2
        dP2 = X[:, 1] - P2
        dH2 = X[:, 2] - H2
        dist2 = np.sqrt(4.0 * dD2**2 + dP2**2 + dH2**2)

        inside = (y == 1)
        outside = (y == 0)

        # Penalties sphere 1
        fi1 = np.ones_like(dist1)
        bad_inside = inside & (dist1 >= R1)
        fi1[bad_inside] = np.exp(R1 - dist1[bad_inside])
        good_outside = outside & (dist1 < R1)
        fi1[good_outside] = np.exp(dist1[good_outside] - R1)

        # Penalties sphere 2
        fi2 = np.ones_like(dist2)
        bad_inside = inside & (dist2 >= R2)
        fi2[bad_inside] = np.exp(R2 - dist2[bad_inside])
        good_outside = outside & (dist2 < R2)
        fi2[good_outside] = np.exp(dist2[good_outside] - R2)

        # Combine penalties
        fi = np.ones(len(y))

        # For samples where y=1 AND any fi equals 1, keep fi_min = 1
        inside_mask = (y == 1)
        any_fi_equals_1 = np.any(np.column_stack([np.isclose(fi1, 1.0), np.isclose(fi2, 1.0)]), axis=1)
        keep_ones = inside_mask & any_fi_equals_1
        
        # For all other cases, use minimum of fi values
        use_min = ~keep_ones
        fi[use_min] = np.minimum(fi1[use_min], fi2[use_min])

        # DATAFIT = np.prod(fi) ** (1 / len(fi))
        # Safe geometric mean
        DATAFIT = np.exp(np.mean(np.log(fi)))

        # Handle size factor logic
        if self.size_factor is None:
            OF = DATAFIT
        elif self.size_factor == "n_solvents":
            m = len(y)
            OF = DATAFIT * (R1 * R2) ** (-1 / m)
        else:
            m = float(self.size_factor)
            OF = DATAFIT * (R1 * R2) ** (-1 / m)

        return 1 - float(OF)