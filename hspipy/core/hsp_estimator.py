from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, minimize, OptimizeResult
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils._param_validation import Interval, StrOptions, Real, Integral

from .loss import HSPSingleSphereLoss, HSPDoubleSphereLoss
from .utils import compute_datafit, hansen_distance, hansen_center_distance

class HSPEstimator(TransformerMixin, BaseEstimator):
    """
    Hansen Solubility Parameters estimator with sklearn compatibility.
    
    This estimator fits Hansen Solubility Parameter spheres to solvent data
    and can predict solvent compatibility. Fully compatible with sklearn 
    pipelines and cross-validation.
    
    For plotting capabilities and simplified interface, use the HSP class
    which inherits from this estimator.

    Parameters
    ----------
    method : str, default='classic'
        Optimization method to use.
    inside_limit : float, default=1
        Threshold for classifying solvents as inside/outside.
    n_spheres : int, default=1
        Number of spheres to fit.
        
    Attributes
    ----------
    hsp_ : ndarray of shape (n_spheres, 4)
        Fitted Hansen Solubility Parameters (D, P, H, R) for each sphere.
    error_ : float
        Final optimization objective value.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> from hspipy import HSPEstimator
    >>> import numpy as np
    >>> X = np.random.rand(100, 3) * 20  # D, P, H values
    >>> y = np.random.randint(0, 2, 100)  # Binary labels
    >>> estimator = HSPEstimator()
    >>> estimator.fit(X, y)
    >>> predictions = estimator.predict(X)
    """
     # Define default parameter ranges for optimization
    DEFAULT_BOUNDS = [(10, 30), (0, 30), (0, 30), (1, 20)]

    _parameter_constraints = {
        "method": [StrOptions({"classic", "differential_evolution", "Nelder-Mead", "Powell", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "COBYQA", "SLSQP"})],
        "inside_limit": [Interval(Real, 0, None, closed="right")],
        "n_spheres": [Interval(Integral, 1, None, closed="left")],
        "loss": [None, callable],
        "size_factor": [None, StrOptions({"n_solvents"}), Interval(Real, 0, None, closed="right")],
        "de_bounds": [None, list],
        "de_strategy": [StrOptions({"best1bin"})],  # Add all valid options
        "de_maxiter": [Interval(Integral, 1, None, closed="left")],
        "de_popsize": [Interval(Integral, 1, None, closed="left")],
        "de_tol": [Interval(Real, 0, None, closed="left")],
        "de_mutation": [Interval(Real, 0, None, closed="left")],
        "de_recombination": [Interval(Real, 0, 1, closed="both")],
        "de_init": [StrOptions({"latinhypercube"})],  # Add all valid options
        "de_atol": [Interval(Real, 0, None, closed="left")],
        "de_updating": [StrOptions({"immediate", "deferred"})],
        "de_workers": [Interval(Integral, 1, None, closed="left")],
        "min_options": [None, dict],
    }

    def _validate_data(self, X, y=None, reset=False):
        """Validate input data with strict 3-feature requirement."""
        X = check_array(
            X,
            accept_sparse=False,
            dtype=float,
            ensure_2d=True,
            ensure_min_features=3,
            ensure_non_negative=True,
            ensure_all_finite=True,  # use force_all_finite=True for sklearn <1.6
        )
        if X.shape[1] != 3:
            raise ValueError(
                f"X must have exactly 3 features (D, P, H), but got {X.shape[1]} features."
            )

        if y is not None:
            y = np.asarray(y, dtype=float)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y have inconsistent numbers of samples: "
                    f"X has {X.shape[0]} samples, y has {y.shape[0]} samples."
                )
            return X, y
        return X

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
        de_maxiter=1000,
        de_popsize=15,
        de_tol=1e-6,
        de_mutation=0.7,
        de_recombination=0.4,
        de_init='latinhypercube',
        de_atol=0,
        de_updating='deferred',
        de_workers=1,
        # Minimize params
        min_options=None,
    ):
        super().__init__()
        self.method = method
        self.inside_limit = inside_limit
        self.n_spheres = n_spheres
        self.loss = loss
        self.size_factor = size_factor
        # DE params
        self.de_bounds = de_bounds
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
        self.min_options = min_options

    def _binarize_labels(self, y: NDArray) -> NDArray:
        """Convert labels to binary (1=inside, 0=outside) using inside_limit.

        A solvent is classified as "inside" (good) when its score satisfies
        ``0 < score <= inside_limit``.  A score of exactly 0 is treated as
        "outside" (bad) regardless of ``inside_limit``, because 0 is the
        conventional placeholder for an untested or bad solvent in HSP data.
        """
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
        bounds = self.de_bounds if self.de_bounds is not None else self.DEFAULT_BOUNDS
        if self.n_spheres == 1:
            return bounds
        elif self.n_spheres == 2:
            return bounds * self.n_spheres
    
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
     
    def _classic_fit(self, X, y):
        """Classic fitting algorithm for single sphere (vectorized)."""
        X = np.asarray(X, dtype=float)
        if np.sum(y == 1) == 0:
            raise ValueError("No good solvents found for classic fit.")

        nfev = 0
        nit_total = 0
        inside = (y == 1)
        outside = ~inside

        def best_radius_vec(dist):
            """Vectorized radius search: evaluates all candidates in one numpy pass."""
            nonlocal nfev
            unique_dists = np.unique(dist)
            candidates = np.r_[
                max(unique_dists[0] - 1e-6, 0.0),
                unique_dists,
                unique_dists + 1e-6,
                unique_dists[-1] + 1.0,
            ]  # (n_cand,)
            nfev += len(candidates)

            # delta[i, j] = dist[i] - candidates[j],  shape (n_samples, n_cand)
            delta = dist[:, None] - candidates[None, :]
            fitness = np.ones_like(delta)

            # good solvent outside sphere
            good_out = inside[:, None] & (delta > 0)
            fitness[good_out] = np.exp(-delta[good_out])

            # bad solvent inside sphere
            bad_in = outside[:, None] & (delta < 0)
            fitness[bad_in] = np.exp(delta[bad_in])

            fitness = np.maximum(fitness, 1e-12)
            datafit_all = np.exp(np.mean(np.log(fitness), axis=0))  # (n_cand,)

            best_df = float(datafit_all.max())
            tied = np.abs(datafit_all - best_df) < 1e-9
            best_R = float(candidates[tied].min())
            return best_R, best_df

        # Start at mean of good solvents
        center = X[inside].mean(axis=0)
        dist = hansen_distance(X, center)
        best_R, best_df = best_radius_vec(dist)
        best_center = center.copy()

        signs = np.array([[sx, sy, sz]
                          for sx in (-1.0, 1.0)
                          for sy in (-1.0, 1.0)
                          for sz in (-1.0, 1.0)])  # (8, 3)

        # Strict-improvement hill-climb: only move when df strictly increases.
        # R tie-breaking is intentionally excluded from the movement criterion —
        # mixing it in creates a large plateau traversal that never terminates
        # on some datasets. The minimum radius for the final center is resolved
        # once by best_radius_vec after convergence.
        tol = 1e-9
        for edge in (1.0, 0.3, 0.1):
            h = 0.5 * edge
            improved = True
            iter_count = 0
            while improved:
                improved = False
                iter_count += 1
                corners = best_center[None, :] + h * signs  # (8, 3)

                # All 8 corner distances in one broadcast: (n_samples, 8)
                d = X[:, None, :] - corners[None, :, :]
                dist_all = np.sqrt(4.0 * d[..., 0]**2 + d[..., 1]**2 + d[..., 2]**2)

                for k in range(8):
                    R_c, df_c = best_radius_vec(dist_all[:, k])
                    if df_c > best_df + tol:
                        best_df = df_c
                        best_R = R_c
                        best_center = corners[k]
                        improved = True
            nit_total += iter_count

        # Resolve minimum radius for the converged center
        dist = hansen_distance(X, best_center)
        best_R, best_df = best_radius_vec(dist)

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
        """Classic HSP fit restricted to exactly 2 spheres (vectorized)."""
        if self.n_spheres != 2:
            raise ValueError("classic_two requires n_spheres == 2.")
        X = np.asarray(X, dtype=float)
        if np.sum(y == 1) == 0:
            raise ValueError("No good solvents found for classic_two.")

        nfev = 0
        nit_total = 0
        inside = (y == 1)
        outside = ~inside

        def best_radius_for_sphere_vec(candidate_center, dist_j, dist_k, R_k, center_k):
            """Vectorized radius search for one sphere, keeping the other fixed.

            candidate_center is passed explicitly so the containment constraint
            uses the new position rather than the stale centers list.
            """
            nonlocal nfev
            unique_dists = np.unique(dist_j)
            candidates = np.r_[
                max(unique_dists[0] - 1e-6, 0.0),
                unique_dists,
                unique_dists + 1e-6,
                unique_dists[-1] + 1.0,
            ]
            nfev += len(candidates)

            delta_j = dist_j[:, None] - candidates[None, :]  # (n_samples, n_cand)
            delta_k = dist_k - R_k  # (n_samples,)

            # Per-sphere fitness
            fit_j = np.ones_like(delta_j)
            g_out_j = inside[:, None] & (delta_j > 0)
            fit_j[g_out_j] = np.exp(-delta_j[g_out_j])
            b_in_j = outside[:, None] & (delta_j < 0)
            fit_j[b_in_j] = np.exp(delta_j[b_in_j])

            fit_k = np.ones(len(y))
            g_out_k = inside & (delta_k > 0)
            fit_k[g_out_k] = np.exp(-delta_k[g_out_k])
            b_in_k = outside & (delta_k < 0)
            fit_k[b_in_k] = np.exp(delta_k[b_in_k])

            # Combined fitness: worst across both spheres, then override correct good
            fitness = np.minimum(fit_j, fit_k[:, None])
            correct = (inside[:, None] & (delta_j <= 0)) | (inside & (dist_k <= R_k))[:, None]
            fitness[correct] = 1.0
            fitness = np.maximum(fitness, 1e-12)
            datafit_all = np.exp(np.mean(np.log(fitness), axis=0))  # (n_cand,)

            # Constraint penalties (vectorized over candidates)
            cd = hansen_center_distance(candidate_center, center_k)
            min_r = np.minimum(candidates, R_k)
            max_r = np.maximum(candidates, R_k)
            penalty = np.zeros(len(candidates))
            penalty[(cd + min_r) <= (max_r + 1e-8)] += 1000.0
            good_j = np.sum(inside[:, None] & (delta_j <= 0), axis=0)
            good_k = int(np.sum(inside & (dist_k <= R_k)))
            penalty[(good_j < 2) | (good_k < 2)] += 1000.0
            penalty[candidates < 1] += 1000.0

            scored = datafit_all - penalty
            best_scored = float(scored.max())
            tied = np.abs(scored - best_scored) < 1e-9
            best_R = float(candidates[tied].min())
            return best_R, best_scored

        # --- Initialization ---
        X_good = X[y == 1, :3]
        if X_good.shape[0] < 2:
            raise ValueError("Not enough good solvents to initialize 2 spheres.")

        initial_guess = self._get_initial_guess(X_good)
        centers = [initial_guess[i * 4:i * 4 + 3].copy() for i in range(2)]
        dists_list = [hansen_distance(X, c) for c in centers]
        radii = [1.0, 1.0]
        best_df = -np.inf

        signs = np.array([[sx, sy, sz]
                          for sx in (-1.0, 1.0)
                          for sy in (-1.0, 1.0)
                          for sz in (-1.0, 1.0)])  # (8, 3)

        # Strict-improvement hill-climb: same rationale as _classic_fit.
        # R tie-breaking removed from both the inner candidate selection and
        # the outer movement condition to prevent plateau traversal.
        tol = 1e-9
        for edge in (1.0, 0.3, 0.1):
            h = 0.5 * edge
            improved = True
            iter_count = 0
            while improved:
                improved = False
                iter_count += 1
                for j in range(2):
                    k = 1 - j
                    best_local_df = best_df
                    best_local = None

                    corners = centers[j][None, :] + h * signs  # (8, 3)
                    d = X[:, None, :] - corners[None, :, :]
                    dist_j_all = np.sqrt(4.0 * d[..., 0]**2 + d[..., 1]**2 + d[..., 2]**2)

                    for idx in range(8):
                        Rj, dfj = best_radius_for_sphere_vec(
                            corners[idx], dist_j_all[:, idx],
                            dists_list[k], radii[k], centers[k],
                        )
                        if dfj > best_local_df + tol:
                            best_local_df = dfj
                            best_local = (corners[idx], dist_j_all[:, idx], Rj)

                    if best_local is not None:
                        c, dist_j, Rj = best_local
                        centers[j] = c
                        radii[j] = Rj
                        dists_list[j] = dist_j
                        best_df = best_local_df
                        improved = True
            nit_total += iter_count

        # Resolve final radii for both spheres after convergence
        for j in range(2):
            k = 1 - j
            radii[j], _ = best_radius_for_sphere_vec(
                centers[j], dists_list[j], dists_list[k], radii[k], centers[k],
            )

        # Report unpenalized datafit for the final configuration
        distances = np.stack(dists_list, axis=1)
        actual_df = float(compute_datafit(distances, np.array(radii).reshape(1, 2), y))

        self.hsp_ = np.array([[centers[i][0], centers[i][1], centers[i][2], radii[i]] for i in range(2)])
        self.error_ = 1.0 - actual_df
        self.datafit_ = actual_df
        self.optimization_result_ = OptimizeResult({
            'method': 'classic_two',
            'datafit': actual_df,
            'fun': float(self.error_),
            'x': self.hsp_.flatten().tolist(),
            'success': True,
            'message': f'classic finished, datafit={actual_df:.6g}',
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
        
        options = self.min_options if self.min_options is not None else {}
        result = minimize(
            objective,
            x0,
            method=self.method,
            bounds=bounds if self.method in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr'] else None,
            tol=1e-6,
            options=options
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

        # vectorized=True lets scipy pass the entire population (n_params, S) to
        # the objective at once. The loss functions handle the batch dimension via
        # ndim dispatch, replacing ~S Python call overhead with a single numpy
        # operation. scipy requires updating='deferred' with vectorized=True, and
        # raises ValueError if workers > 1 is combined with vectorized, so we
        # clamp workers to 1 here.
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
            'updating': 'deferred',
            'workers': 1,
            'x0': x0,
            'vectorized': True,
        }

        result = differential_evolution(objective, **options)

        self.hsp_ = result.x.reshape(self.n_spheres, 4)
        self.error_ = result.fun
        self.optimization_result_ = result
        self.datafit_ = self._get_datafit(X, y)
        
        return self
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: NDArray, y: NDArray) -> HSPEstimator:
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
        X, y = self._validate_data(X, y, reset=True)
       
        # Filter out NaN values in y
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
        elif self.method == 'classic' and self.n_spheres == 2:
            return self._classic_two_fit(Xv, yv)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict using the HSP model for 1 or 2 spheres.

        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Vectors to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values (1 if inside any sphere, else 0)
        """
        check_is_fitted(self, ['hsp_', 'error_'])
        X = self._validate_data(X)

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

    def transform(self, X: NDArray) -> NDArray:
        """Transform X using HSP distances.

        Returns the minimum Relative Energy Difference (RED = dist/R) across
        all fitted spheres for each sample.  RED < 1 means the solvent lies
        inside a sphere (predicted compatible).

        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Input samples with D, P, H values

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, 1)
            RED values for each sample (minimum across spheres)
        """
        check_is_fitted(self, ['hsp_', 'error_'])
        X = self._validate_data(X)

        spheres = np.asarray(self.hsp_, dtype=float)
        Ds, Ps, Hs, Rs = spheres[:, 0], spheres[:, 1], spheres[:, 2], spheres[:, 3]

        dD = X[:, [0]] - Ds  # (n_samples, n_spheres)
        dP = X[:, [1]] - Ps
        dH = X[:, [2]] - Hs
        dist = np.sqrt(4.0 * dD**2 + dP**2 + dH**2)  # (n_samples, n_spheres)
        red = dist / Rs  # (n_samples, n_spheres)
        red_min = red.min(axis=1)  # (n_samples,)

        return red_min.reshape(-1, 1)
    
    def score(self, X: NDArray, y: NDArray) -> float:
        """Returns the model accuracy on the given test data and labels."""
        check_is_fitted(self, ['hsp_', 'error_'])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Binarize and filter out NaN
        valid = ~np.isnan(y)
        y_bin = self._binarize_labels(y[valid])
        y_pred = self.predict(X[valid])

        self.accuracy_ = accuracy_score(y_bin, y_pred)

        self.n_solvents_in_ = int(np.sum(y_bin == 1))
        self.n_solvents_out_ = int(np.sum(y_bin == 0))
        self.n_total_ = int(len(y_bin))
        self.n_wrong_in_ = int(np.sum((y_bin == 0) & (y_pred == 1)))
        self.n_wrong_out_ = int(np.sum((y_bin == 1) & (y_pred == 0)))

        return self.accuracy_



    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Set attributes directly
        tags.input_tags.positive_only = True
        tags.input_tags.sparse = False
        return tags