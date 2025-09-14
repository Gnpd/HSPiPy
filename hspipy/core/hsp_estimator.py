import numpy as np
from scipy.optimize import differential_evolution, minimize, OptimizeResult
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from .loss import HSPSingleSphereLoss, HSPDoubleSphereLoss
from .utils import compute_datafit, hansen_distance, hansen_center_distance

class HSPEstimator(BaseEstimator, TransformerMixin):
    """
    Hansen Solubility Parameters estimator with sklearn compatibility.
    
    This estimator fits Hansen Solubility Parameter spheres to solvent data
    and can predict solvent compatibility. Fully compatible with sklearn 
    pipelines and cross-validation.
    
    For plotting capabilities and simplified interface, use the HSP class
    which inherits from this estimator.
    """
     # Define default parameter ranges for optimization
    DEFAULT_BOUNDS = [(10, 30), (0, 30), (0, 30), (1, 20)]

    def __init__(
        self,
        method='classic',
        inside_limit=1,
        n_spheres=1,
        loss=None,
        size_factor=None,
        # Differential Evolution params
        de_bounds=None,
        de_strategy='best1bin',
        de_maxiter=2000,
        de_popsize=15,
        de_tol=1e-6,
        de_mutation=0.7,
        de_recombination=0.4,
        de_init='latinhypercube',
        de_atol=0,
        de_updating='immediate',
        de_workers=1,
        # Minimize params
        min_options=None,
    ):
        self.method = method
        self.inside_limit = inside_limit
        self.n_spheres = n_spheres
        self.loss = loss
        self.size_factor = size_factor
        # DE params
        self.de_bounds = de_bounds or self.DEFAULT_BOUNDS
        self.de_strategy = de_strategy
        self.de_maxiter = de_maxiter
        self.de_popsize = de_popsize
        self.de_tol = de_tol
        self.de_mutation = de_mutation
        self.de_recombination = de_recombination
        self.de_init = de_init
        self.de_atol = de_atol
        self.de_updating = de_updating
        self.de_workers = de_workers
        # Minimize params
        self.minimize_options = min_options or {}

        # Fitted attributes (scikit-learn convention: trailing underscore)
        self.hsp_ = None          # (D, P, H, R) Hansen Solubility Parameters
        self.error_ = None        # objective value (mean penalty)
        self.accuracy_ = None     # accuracy on training data
        self.optimization_result_ = None  # result of the optimization process
        self.inside_ = None  # solvents classified as inside (y<=inside_limit)
        self.outside_ = None # solvents classified as outside (y>inside_limit)
        self.datafit_ = None  # DATAFIT value of the fitted model

    def _binarize_labels(self, y):
        """Convert labels to binary (1=inside, 0=outside) using inside_limit."""
        y = np.asarray(y, dtype=float)
        return ((y <= self.inside_limit) & (y != 0)).astype(int)
    
    def _get_loss_function(self):
        """Get the appropriate loss function based on configuration."""
        if self.loss is not None:
            return self.loss
        
        if self.n_spheres == 1:
            return HSPSingleSphereLoss(size_factor=self.size_factor)
        elif self.n_spheres == 2:
            return HSPDoubleSphereLoss(size_factor=self.size_factor)
        else:
            raise ValueError("Only 1 or 2 spheres are currently supported")
        
    def _get_initial_guess(self, X_good):
        """Get initial guess for optimization based on good solvents."""
        if self.n_spheres == 1:
            return np.array([
                X_good.mean(axis=0)[0], 
                X_good.mean(axis=0)[1], 
                X_good.mean(axis=0)[2], 
                5.0
            ])
        elif self.n_spheres == 2:
            dists = squareform(pdist(X_good))
            idx = np.unravel_index(np.argmax(dists), dists.shape)
            centers = X_good[list(idx)]
            initial_guess = np.hstack((centers, np.full((self.n_spheres, 1), 2.0)))
            return initial_guess.flatten()
    
    def _get_bounds(self):
        """Get bounds for optimization based on number of spheres."""
        if self.n_spheres == 1:
            return self.de_bounds
        elif self.n_spheres == 2:
            return self.de_bounds * self.n_spheres
    
    def _get_datafit(self, X, y_bin):
        """Compute DATAFIT for the current model on given data."""
        check_is_fitted(self, 'hsp_')

        if self.n_spheres == 1:
            D, P, H, R = self.hsp_[0]
            center = np.array([D, P, H])
            dist = hansen_distance(X, center)
            return compute_datafit(dist, R, y_bin)
        elif self.n_spheres == 2:
            D1, P1, H1, R1 = self.hsp_[0]
            D2, P2, H2, R2 = self.hsp_[1]
            center1 = np.array([D1, P1, H1])
            center2 = np.array([D2, P2, H2])
            dist1 = hansen_distance(X, center1)
            dist2 = hansen_distance(X, center2)
            distances = np.stack((dist1, dist2), axis=1)  # shape (n_samples, 2)
            radii = np.array([R1, R2])
            return compute_datafit(distances, radii, y_bin)
        else:
            raise ValueError("DATAFIT computation only implemented for 1 or 2 spheres.")

        return DATAFIT
     
    def _classic_fit(self, X, y):
        """Classic fitting algorithm for single sphere."""
        X = np.asarray(X, dtype=float)
        if np.sum(y == 1) == 0:
            raise ValueError("No good solvents found for classic fit.")
        
        nfev = 0
        nit_total = 0

        def datafit(dist, R):
            nonlocal nfev
            nfev += 1
            return compute_datafit(dist, R, y)

        def best_radius(dist):
            dists = np.unique(dist)
            candidates = np.r_[max(dists.min() - 1e-6, 0.0), dists, dists + 1e-6, dists.max() + 1.0]
            best_df = -np.inf
            best_R = None
            for R in candidates:
                df = datafit(dist, R)
                if df > best_df or (np.isclose(df, best_df) and (best_R is None or R < best_R)):
                    best_df = df
                    best_R = R
            return best_R, best_df

        # Start at mean of good solvents
        center = X[y == 1, :3].mean(axis=0)
        dist = hansen_distance(X, center)
        best_R, best_df = best_radius(dist)
        best_center = center.copy()

        # Edge lengths 1.0 → 0.3 → 0.1; use ±half-edge offsets for corners
        for edge in (1.0, 0.3, 0.1):
            h = 0.5 * edge
            improved = True
            signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
            iter_count = 0
            while improved:
                improved = False
                iter_count += 1
                corners = best_center[None, :] + h * signs
                for c in corners:
                    dist_c = hansen_distance(X, c)
                    R_c, df_c = best_radius(dist_c)
                    if df_c > best_df or (np.isclose(df_c, best_df) and R_c < best_R):
                        best_df = df_c
                        best_R = R_c
                        best_center = c
                        improved = True
            nit_total += iter_count

        self.hsp_ = np.array([[best_center[0], best_center[1], best_center[2], best_R]])
        self.error_ = 1.0 - best_df
        self.datafit_ = best_df
        self.optimization_result_ = OptimizeResult({
            'method': 'classic', 
            'fun': float(self.error_),
            'x': self.hsp_.flatten().tolist(),
            'success': True,
            'message': f'classic finished, datafit={best_df:.6g}',
            'nit': int(nit_total),
            'nfev': int(nfev),
            })
        return self
  
    def _classic_two_fit(self, X, y):
        """Classic HSP fit restricted to exactly 2 spheres."""
        if self.n_spheres != 2:
            raise ValueError("classic_two requires n_spheres == 2.")
        X = np.asarray(X, dtype=float)

        if np.sum(y == 1) == 0:
            raise ValueError("No good solvents found for classic_two.")

        nfev = 0
        nit_total = 0

        # --- Constraint-aware datafit ---
        def combine_datafit(dists_list, radii, centers):
            nonlocal nfev
            nfev += 1
            distances = np.stack(dists_list, axis=1)  # (n_samples, 2)
            R = np.asarray(radii).reshape((1, 2))
            df = compute_datafit(distances, R, y)

            # Apply constraints as penalties
            penalty = 0.0
            # (1) Sphere containment
            d = hansen_center_distance(centers[0], centers[1])
            if d + min(radii) <= max(radii) + 1e-8:
                penalty += 1000.0
            # (2) Each sphere must contain at least two "good"
            inside = (y == 1)
            if np.any(inside):
               # Count how many good solvents each sphere contains
                good_counts = np.sum(inside[:, None] & (distances <= R), axis=0)
                # Require at least 2 per sphere
                if np.any(good_counts < 2):
                    penalty += 1000.0
            else:
                penalty += 1000.0

            # (3) Minimum radius to avoid collapse
            min_radius = 1
            if np.any(np.asarray(radii) < min_radius):
                penalty += 1000.0

            return df - penalty

        def best_radius_for_sphere(dist_j, dists_list, radii, j, centers):
            # scan candidate radii for sphere j keeping others fixed
            dists = np.unique(dist_j)
            candidates = np.r_[max(dists.min() - 1e-6, 0.0),
                            dists,
                            dists + 1e-6,
                            dists.max() + 1.0]
            best_df = -np.inf
            best_R = radii[j]
            for R in candidates:
                candidate_r = list(radii)
                candidate_r[j] = R
                df = combine_datafit(dists_list, candidate_r, centers)
                if df > best_df or (np.isclose(df, best_df) and R < best_R):
                    best_df = df
                    best_R = R
            return best_R, best_df

        # --- Initialization: farthest pair among good solvents ---
        X_good = X[y == 1, :3]
        if X_good.shape[0] < 2:
            raise ValueError("Not enough good solvents to initialize 2 spheres.")

        initial_guess = self._get_initial_guess(X_good)  # shape (8,)
        centers = [initial_guess[i*4:i*4+3].copy() for i in range(2)]

        dists_list = [hansen_distance(X, c) for c in centers]
        radii = [0.5, 0.5]

        # Initial datafit
        best_df = combine_datafit(dists_list, radii, centers)

        # --- Classic cube schedule ---
        for edge in (1.0, 0.3, 0.1):
            h = 0.5 * edge
            improved = True
            max_iter = 1000
            iter_count = 0
            tol = 1e-6
            while improved and iter_count < max_iter:
                improved = False
                signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                                                for sy in (-1.0, 1.0)
                                                for sz in (-1.0, 1.0)])
                iter_count += 1
                for j in range(2):
                    best_local = None
                    best_local_df = best_df
                    best_local_R = radii[j]
                    for s in signs:
                        c = centers[j] + h * s
                        dist_j = hansen_distance(X, c)
                        cand_dists = list(dists_list)
                        cand_dists[j] = dist_j
                        Rj, dfj = best_radius_for_sphere(dist_j, cand_dists, radii, j, centers)
                        cand_centers = list(centers)
                        cand_centers[j] = c
                        cand_radii = list(radii)
                        cand_radii[j] = Rj
                        df_total = combine_datafit(cand_dists, cand_radii, cand_centers)
                        if df_total > best_local_df or (np.isclose(df_total, best_local_df) and Rj < best_local_R):
                            best_local_df = df_total
                            best_local_R = Rj
                            best_local = (cand_dists, cand_centers, cand_radii)
                    if best_local is not None and (best_local_df > best_df + tol or
                                                (np.isclose(best_local_df, best_df) and best_local_R < radii[j])):
                        dists_list, centers, radii = best_local
                        best_df = best_local_df
                        improved = True
            nit_total += iter_count

        self.hsp_ = np.array([[centers[i][0], centers[i][1], centers[i][2], radii[i]] for i in range(2)])
        self.error_ = 1.0 - best_df
        self.datafit_ = best_df
        self.optimization_result_ = OptimizeResult({
            'method': 'classic_two',
            'datafit': best_df,
            'fun': float(self.error_),
            'x': self.hsp_.flatten().tolist(),
            'success': True,
            'message': f'classic finished, datafit={best_df:.6g}',
            'nit': int(nit_total),
            'nfev': int(nfev),
        })
        return self

    def _minimize_fit(self, X, y):
        """Fit using scipy.optimize.minimize with selected solver."""
     
        X_good = X[y == 1, :3]
        if X_good.shape[0] < self.n_spheres:
            raise ValueError("Not enough inside solvents to form the required number of spheres.")
        
        loss_func = self._get_loss_function()
        x0 = self._get_initial_guess(X_good)
        bounds = self._get_bounds()

        def objective(array):
            return loss_func(array, X, y)

        result = minimize(
            objective,
            x0,
            method=self.method,
            bounds=bounds if self.method in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr'] else None,
            tol=1e-6,
            options=self.minimize_options
        )

        self.hsp_ = result.x.reshape(self.n_spheres, 4)
        self.error_ = result.fun
        self.optimization_result_ = result
        self.datafit_ = self._get_datafit(X, y)

        return self
    
    def _differential_evolution_fit(self, X, y):
        """Fit using differential evolution"""

        X_good = X[y == 1, :3]
        if X_good.shape[0] < self.n_spheres:
            raise ValueError("Not enough inside solvents to form the required number of spheres.")
        
        loss_func = self._get_loss_function()
        bounds = self._get_bounds()
        x0 = self._get_initial_guess(X_good)

        def objective(array):
            return loss_func(array, X, y)
        
        options = {
            'bounds': bounds,
            'strategy': self.de_strategy,
            'maxiter': self.de_maxiter,
            'popsize': self.de_popsize,
            'tol': self.de_tol,
            'mutation': self.de_mutation,
            'recombination': self.de_recombination,
            'init': self.de_init,
            'atol': self.de_atol,
            'updating': self.de_updating,
            'workers': self.de_workers,
            'x0': x0,
            }

        result = differential_evolution(objective, **options)

        self.hsp_ = result.x.reshape(self.n_spheres, 4)
        self.error_ = result.fun
        self.optimization_result_ = result
        self.datafit_ = self._get_datafit(X, y)
        
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
        yv = self._binarize_labels(y[valid])  # Binarize labels
        
        self.inside_ = Xv[yv == 1]
        self.outside_ = Xv[yv == 0]

        # Select appropriate fitting method
        if self.method == 'differential_evolution':
            return self._differential_evolution_fit(Xv, yv)
        elif self.method in [
            'Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B', 'TNC',
            'COBYLA', 'COBYQA', 'SLSQP'
        ]:
            return self._minimize_fit(Xv, yv)
        elif self.method == 'classic' and self.n_spheres == 1:
            return self._classic_fit(Xv, yv)
        elif self.method == 'classic' and self.n_spheres >= 2:
            return self._classic_two_fit(Xv, yv)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def predict(self, X):
        """Predict using the HSP model for 1 or 2 spheres.

        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Vectors to predict

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values (1 if inside any sphere, else 0)
        """
        check_is_fitted(self, ['hsp_', 'error_'])
        X = np.asarray(X, dtype=float)

        spheres = np.asarray(self.hsp_, dtype=float)
        Ds, Ps, Hs, Rs = spheres[:, 0], spheres[:, 1], spheres[:, 2], spheres[:, 3]

        # Calculate distances to all spheres
        dD = X[:, None, 0] - Ds[None, :]
        dP = X[:, None, 1] - Ps[None, :]
        dH = X[:, None, 2] - Hs[None, :]
        dist = np.sqrt(4.0 * dD**2 + dP**2 + dH**2)  # (n_samples, n_spheres)

        red = dist / Rs[None, :]  # (n_samples, n_spheres)

        # If inside any sphere, predict 1
        y_pred = (red <= 1.0).any(axis=1).astype(int)
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
        """Returns the model accuracy on the given test data and labels."""
        check_is_fitted(self, ['hsp_', 'error_'])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Binarize and filter out NaN
        valid = ~np.isnan(y)
        y_bin = self._binarize_labels(y[valid])
        y_pred = self.predict(X[valid])

        self.accuracy_ = accuracy_score(y_bin, y_pred)
        return self.accuracy_






