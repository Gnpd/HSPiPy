[Home](index.md) | [HSP](HSP.md) | [HSPEstimator](HSPEstimator.md) | [Discovery](discovery.md)

# HSPEstimator

`class HSPEstimator`

A scikit-learn compatible estimator for Hansen Solubility Parameter sphere fitting

The `HSPEstimator` fits Hansen Solubility Parameter spheres to solvent data and predicts solvent compatibility. It supports both single and double sphere models with various optimization methods Compatible with `sklearn` pipelines, model selection, and metrics.

[View source code](https://github.com/Gnpd/HSPiPy/blob/main/hspipy/core/hsp_estimator.py)


## Parameters

### Core Parameters

- `method`:`str`, default='classic'
  Optimization method to use. Options:
  - `'classic'`: Classic grid-based search [1] adaptation (supports 1 or 2 spheres)
  - `'differential_evolution'`: Differential evolution global optimization. Uses [scipy.optimize.differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).
  - `Nelder-Mead`, `Powell`, `BFGS`, `L-BFGS-B`, `TNC`, `COBYLA`, `COBYQA`, `SLSQP`: Uses [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize).
  See the Discussion on adding more optimization algorithms:
  https://github.com/Gnpd/HSPiPy/discussions/2
- `inside_limit`: `float`, default=1
  Threshold that defines which labels are considered "inside" (good) solvents. Labels with `0 < y <= inside_limit` are treated as inside; `y == 0` or `y > inside_limit` are outside.
- `n_spheres`:`int`, default=1
  Number of spheres to fit. Supported values: 1 or 2.
- `loss`:`callable` or `None`, default = `None`.
   Custom loss function for non classic methods. If `None`, defaults are used:
  - `HSPSingleSphereLoss` for `n_spheres == 1`
  - `HSPDoubleSphereLoss` for `n_spheres == 2`
- `size_factor`: `float`, `str` or `None`, default=None
  Size factor normalization (for non classic methods):
  - `None`: No size normalization
  - `n_solvents`: Normalize by number of solvents [3,4]
  - `float`: Custom normalization factor [2]

### Differential Evolution Parameters:
*passed through to `scipy.optimize.differential_evolution`*

- `de_bounds`: `list of tuple`, default=`[(10, 30), (0, 30), (0, 30), (1, 20)]`
  Bounds for a single sphere:
  $[(D_min, D_max), (P_min, P_max), (H_min, H_max), (R_min, R_max)]$
  For two spheres, the bounds are repeated twice.
- `de_strategy`: `str`, default='best1bin'
  Differential evolution strategy.
- `de_maxiter`: `int`, default=2000
  Maximum number of iterations.
- `de_popsize`: `int`, default=15
  Population size.
- `de_tol`: `float`, default=1e-6
  Relative tolerance for convergence.
- `de_mutation`: `float` or `tuple`, default=0.7
  Mutation factor.
- `de_recombination`: `float`, default=0.4
  Recombination constant.
- `de_init`: `str` or `array`, default='latinhypercube'
  Initialization method.
- `de_atol`: `float`, default=`0`
  Absolute tolerance for convergence.
- `de_updating`: {`'immediate'`, `'deferred'`}, default='immediate'
  Updating strategy.
- `de_workers`: `int`, default=`1`
  Number of workers for parallel execution.

### Minimize Parameters
- `min_options`:`dict` or `None`, default=None
  Additional options for [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize).

## Attributes

- `hsp_`: `ndarray` of shape (n_spheres, 4)
  The fitted sphere parameters `[D, P, H, R]` for each sphere.
- `error_`: `float`
  Objective value at the optimum (lower is better, 1 - DATAFIT fro classic method).
- `accuracy_`: `float`
  Set by `score(X, y)`. Classification accuracy of inside/outside vs. predicted.
- `optimization_result_`: `OptimizeResult`
  Raw optimizer result / metadata (contains `fun`, `x`, `success`, `message`, etc).
- `inside_` / `outside_`: `ndarray` of the input X rows classified as inside or outside after binarization.
- `datafit_`:`float`
  DATAFIT value of the fitted model.

## Methods

### `fit(X, y)`
Fit the estimator to training data.

**Parameters**:
- `X`:  array-like of shape (n_samples, 3).
  Training vectors with D, P, H values.
- `y`:  array-like of shape (n_samples,).
  Target scores (1 and 0, or 1,2,3,4,...).

**Returns**:
- `self`: object
  Fitted estimator
  
### `predict(X)`
Predict using the fitted HSP model.

**Parameters**:
- `X`:  array-like of shape (n_samples, 3).
  Vectors to predict.

**Returns**:
- `y_pred`: `ndarray` of shape (n_samples,)
  Predicted values (1 if inside any sphere, else 0).

### `transform(X)`
Transform X using HSP distances.

**Parameters**:
- `X`:  array-like of shape (n_samples, 3).
  Input samples with D, P, H values.

**Returns**:
- `X_transformed`: `ndarray` of shape (n_samples,1)
  RED values for each sample.

### `score(X, y)`
Return the model accuracy on the given test data and labels.

**Parameters**:
- `X`:  array-like of shape (n_samples, 3).
  Test samples, vectors with D, P, H values.
- `y`:  array-like of shape (n_samples,).
  True labels.

**Returns**:
- `accuracy`: `float`
  Accuracy score vs. binarized labels using `inside_limit`.


## Examples

Basic single-sphere fit

```python
import pandas as pd
import numpy as np
from hspipy import HSPEstimator

# Example data (provided in the repo)
df = pd.read_csv('hspipy/hsp_example.csv')
X = df[['D', 'P', 'H']].to_numpy()
y = df['Score'].to_numpy()

est = HSPEstimator(inside_limit=1, n_spheres=1)
est.fit(X, y)
print('HSP (D,P,H,R):', est.hsp_)
print('Objective error:', est.error_)
acc = est.score(X, y)
print('Accuracy:', acc)
```

Two-sphere model

```python
est2 = HSPEstimator(inside_limit=1, n_spheres=2)
est2.fit(X, y)
print('Two spheres:', est2.hsp_)
```

## References

1. Hansen, C. M. (2007). Hansen Solubility Parameters: A User's Handbook. CRC Press (2nd ed.), Boca Raton, 2007, [DOI:10.1201/9781420006834](https://doi.org/10.1201/9781420006834).
2. Vebber GC, Pranke P, Pereira CN (2013). Calculating Hansen solubility parameters of polymers with genetic algorithms, J. Appl. Polym. Sci. 131 (1), [DOI:10.1002/app.39696](https://doi.org/10.1002/app.39696).
3. Díaz de los Ríos, M.; Hernández Ramos (2020). Determination of the Hansen solubility parameters and the Hansen sphere radius with the aid of the solver add-in of Microsoft Excel. [DOI:10.1007/s42452-020-2512-y](https://doi.org/10.1007/s42452-020-2512-y).
4. Díaz de los Ríos, M.; Belmonte (2022). Extending Microsoft excel and Hansen solubility parameters relationship to double Hansen’s sphere calculation. [DOI:10.1007/s42452-022-04959-4](https://doi.org/10.1007/s42452-022-04959-4).
