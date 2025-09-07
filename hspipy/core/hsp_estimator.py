import numpy as np
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from .loss import HSPSingleSphereLoss, HSPDoubleSphereLoss

class HSPEstimator(BaseEstimator, TransformerMixin):
    """
    Hansen Solubility Parameters estimator with sklearn compatibility.
    
    This estimator fits Hansen Solubility Parameter spheres to solvent data
    and can predict solvent compatibility. Fully compatible with sklearn 
    pipelines and cross-validation.
    
    For plotting capabilities and simplified interface, use the HSP class
    which inherits from this estimator.
    """

    def __init__(
        self,
        method='differential_evolution',
        inside_limit=1,
        n_spheres=1,
        loss=None,
        size_factor="n_solvents",
        # Differential Evolution params
        de_bounds=[(10, 30), (0, 30), (0, 30), (1, 20)],
        de_strategy='best1bin',
        de_maxiter=2000,
        de_popsize=15,
        de_tol=1e-8,
        de_mutation=0.7,
        de_recombination=0.3,
        de_init='latinhypercube',
        de_atol=0,
        de_updating='immediate',
        de_workers=1,
    ):
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

        # Fitted attributes (scikit-learn convention: trailing underscore)
        self.hsp_ = None          # (D, P, H, R) Hansen Solubility Parameters
        self.error_ = None        # objective value (mean penalty)
        self.accuracy_ = None     # accuracy on training data
        self.optimization_result_ = None  # result of the optimization process

    def _binarize_labels(self, y):
        """Convert labels to binary (1=inside, 0=outside) using inside_limit."""
        y = np.asarray(y, dtype=float)
        return ((y <= self.inside_limit) & (y != 0)).astype(int)
    
    def _differential_evolution_fit(self, X, y):
        """Fit using differential evolution"""

        # Binarize labels
        y_bin = self._binarize_labels(y, self.inside_limit)

        X_good = X[y_bin == 1, :3]
        if X_good.shape[0] < self.n_spheres:
            raise ValueError("Not enough inside solvents to form the required number of spheres.")
       
        
        options = {
            'bounds': self.de_bounds,
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
            }
        
        if self.n_spheres == 1:
            objective = self.loss if self.loss is not None else HSPSingleSphereLoss(
                inside_limit=self.inside_limit,
                size_factor=self.size_factor
            )
            options['x0'] = np.array([X_good.mean(axis=0)[0], X_good.mean(axis=0)[1], X_good.mean(axis=0)[2], 5.0])

        elif self.n_spheres == 2:
            objective = self.loss if self.loss is not None else HSPDoubleSphereLoss(
                inside_limit=self.inside_limit,
                size_factor=self.size_factor
            )
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
          
        result = differential_evolution(
            func=fun,
            **options
        )


        spheres = result.x.reshape(self.n_spheres, 4)
        self.hsp_ = np.array(spheres) 
        self.error_ = result.fun
        self.optimization_result_ = result
        
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
        y_bin = self._binarize_labels(y[valid], self.inside_limit)
        y_pred = self.predict(X[valid])
        self.accuracy_ = accuracy_score(y_bin, y_pred)
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






