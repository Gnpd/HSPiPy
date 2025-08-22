import numpy as np
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

class HSPEstimator(BaseEstimator, TransformerMixin):
    """
    Hansen Solubility Parameters estimator with sklearn compatibility.
    
    This estimator fits Hansen Solubility Parameter spheres to solvent data
    and can predict solvent compatibility. Fully compatible with sklearn 
    pipelines and cross-validation.
    
    For plotting capabilities and simplified interface, use the HSP class
    which inherits from this estimator.
    """
    DEFAULT_OPTIONS = {
        'differential_evolution': {
            'bounds': [(10, 30), (0, 30), (0, 30), (1, 20)],
            'strategy': 'best1bin',
            'maxiter': 2000,
            'popsize': 15,
            'tol': 1e-4,
            'mutation': 0.075,
            'recombination': 0.7,
            'workers': 1,
        }
    }

    def __init__(self, method='differential_evolution', inside_limit=1, n_spheres=1, options=None):
        """
        Hansen Solubility Parameters Estimator.

        Parameters
        ----------
        method : str, default='differential_evolution'
            Optimization method for fitting HSP spheres.
        inside_limit : float, default=1
            Threshold separating inside vs outside (y <= inside_limit treated as inside).
        n_spheres : int, default=1
            Number of HSP spheres to fit (1 or 2 supported).
        options : dict or None
            Method-specific optimizer parameter overrides.
        """
        self.method = method
        self.inside_limit = inside_limit
        self.n_spheres = n_spheres

        # Get default options for the method
        default_options = self.DEFAULT_OPTIONS.get(method, {})
        options = options or {}
        extra_keys = set(options) - set(default_options)
        if extra_keys:
            raise ValueError(
                f"Unknown parameters for method '{method}': {', '.join(extra_keys)}"
            )
        opts = {**default_options, **options}
        self.options = opts

        # Fitted attributes (scikit-learn convention: trailing underscore)
        self.hsp_ = None          # (D, P, H, R) Hansen Solubility Parameters
        self.error_ = None        # objective value (mean penalty)

    def _objective_single(self, HSP, X, y):
        '''
        Objective function for single sphere (n_spheres=1)
        '''

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

        # Radius penalty: product of radii, same log-trick
        # R_penalty = np.exp(-np.sum(np.log(Rs)) / n_samples)

        OF = DATAFIT * Rs ** (-1 / len(fi))
    
        return 1 - float(OF)
    
    def _objective_double(self, HSP, X, y):
        '''
        Objective function for double sphere (n_spheres=2)
        '''

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

        # Radius penalty: product of radii, same log-trick
        # R_penalty = np.exp(-np.sum(np.log(Rs)) / n_samples)

        OF = DATAFIT * (R1*R2) ** (-1 / len(fi))

        return 1 - float(OF)
    
    def _differential_evolution_fit(self, X, y):
        """Fit using differential evolution"""

        # Binarize labels
        y_bin = ((y <= self.inside_limit) & (y != 0)).astype(int)

        X_good = X[y_bin == 1, :3]
        if X_good.shape[0] < self.n_spheres:
            raise ValueError("Not enough inside solvents to form the required number of spheres.")
       
        options = self.options.copy()
        
        if self.n_spheres == 1:
            objective = self._objective_single
            options['x0'] = np.array([X_good.mean(axis=0)[0], X_good.mean(axis=0)[1], X_good.mean(axis=0)[2], 5.0])

        elif self.n_spheres == 2:
            objective = self._objective_double

            base_bounds = options.get('bounds', [(10, 30), (0, 30), (0, 30), (1, 20)])
            options['bounds'] = base_bounds * self.n_spheres

            # Use KMeans for initial guess sphere centers
            kmeans = KMeans(n_clusters=self.n_spheres, n_init=10, random_state=0)
            kmeans.fit(X_good)
            initial_guess = np.hstack((kmeans.cluster_centers_, np.full((self.n_spheres, 1), 5.0)))
            options['x0'] = initial_guess.flatten()

        else:
            raise ValueError("Only single or double sphere models are currently supported.")

        def fun(array):
            return objective(array, X, y_bin)
          
        print(options)
        result = differential_evolution(
            func=fun,
            **options
        )


        spheres = result.x.reshape(self.n_spheres, 4)
        self.hsp_ = np.array(spheres) 
        self.error_ = result.fun
        
        return self
    
    def fit(self, X, y):
        """Fit HSP model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Training vectors with D, P, H values
        y : array-like of shape (n_samples,)
            Target scores
        
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = ~np.isnan(y)
        Xv = X[valid]
        yv = y[valid]
        
        if self.method == 'differential_evolution':
            return self._differential_evolution_fit(Xv, yv)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def predict(self, X):
        """Predict using the HSP model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Vectors to predict
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        check_is_fitted(self, ['hsp_', 'error_'])
        X = np.asarray(X, dtype=float)

        spheres = np.asarray(self.hsp_, dtype=float)
        Ds, Ps, Hs, Rs = spheres[:, 0], spheres[:, 1], spheres[:, 2], spheres[:, 3]
        # Broadcast over spheres
        dD = X[:, [0]] - Ds  # (n_samples, n_spheres)
        dP = X[:, [1]] - Ps
        dH = X[:, [2]] - Hs
        dist = np.sqrt(4.0 * dD**2 + dP**2 + dH**2)  # (n_samples, n_spheres)
        red = dist / Rs  # (n_samples, n_spheres)
        red_min = red.min(axis=1)  # (n_samples,)

        # Predict: 1 if inside (RED <= 1), else 0
        y_pred = np.where(red_min <= 1.0, 1, 0)
        return y_pred

    def transform(self, X):
        """Transform X using HSP distances.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Input samples with D, P, H values
            
        Returns
        -------
        X_transformed : array of shape (n_samples, 1)
            RED values for each sample
        """
        check_is_fitted(self, ['hsp_', 'error_'])
        X = np.asarray(X, dtype=float)
        spheres = np.asarray(self.hsp_, dtype=float)
        Ds, Ps, Hs, Rs = spheres[:, 0], spheres[:, 1], spheres[:, 2], spheres[:, 3]
        dD = X[:, [0]] - Ds  # (n_samples, n_spheres)
        dP = X[:, [1]] - Ps
        dH = X[:, [2]] - Hs
        dist = np.sqrt(4.0 * dD**2 + dP**2 + dH**2)  # (n_samples, n_spheres)
        red = dist / Rs  # (n_samples, n_spheres)
        red_min = red.min(axis=1)  # (n_samples,)
        return red_min.reshape(-1, 1)
    
    def score(self, X, y):
        """Returns the model accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) with respect to y
        """
        check_is_fitted(self, ['hsp_', 'error_'])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y_bin = ((y <= self.inside_limit) & (y != 0)).astype(int)
        y_pred = self.predict(X)
        self.accuracy_ = np.mean(y_pred == y_bin)
        return self.accuracy_

    def fit_transform(self, X, y):
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
        y : array-like of shape (n_samples,)
        
        Returns
        -------
        X_transformed : array of shape (n_samples, 1)
        """
        return self.fit(X, y).transform(X)






