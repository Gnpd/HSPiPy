# HSPEstimator

A scikit-learn compatible estimator that fits 1 or 2 Hansen Solubility Parameter (HSP) sphere(s) to solvent data and predicts solvent compatibility.

- Module: `hspipy.core`
- Class: `hspipy.HSPEstimator`
- Compatible with `sklearn` pipelines, model selection, and metrics.

## Parameters

- method: {`'differential_evolution'`}, default=`'differential_evolution'`
  Optimization algorithm used for fitting.
  See the Discussion on adding more optimization algorithms:
  https://github.com/Gnpd/HSPiPy/discussions/2
- inside_limit: float, default=1
  Threshold that defines which labels are considered "inside" (good) solvents. Labels with `0 < y <= inside_limit` are treated as inside; `y == 0` or `y > inside_limit` are outside.
- n_spheres: int, default=1
  Number of spheres to fit. Supported values: 1 or 2.

Differential evolution options (passed through to `scipy.optimize.differential_evolution`):

- de_bounds: list of tuple, default=`[(10, 30), (0, 30), (0, 30), (1, 20)]`
  Bounds for a single sphere `[D, P, H, R]`. For two spheres, the bounds are repeated twice.
- de_strategy: str, default=`'best1bin'`
- de_maxiter: int, default=`2000`
- de_popsize: int, default=`15`
- de_tol: float, default=`1e-8`
- de_mutation: float or tuple, default=`(0.5, 1)`
- de_recombination: float, default=`0.7`
- de_init: str or array, default=`'latinhypercube'`
- de_atol: float, default=`0`
- de_updating: {`'immediate'`, `'deferred'`}, default=`'immediate'`
- de_workers: int, default=`1`

## Attributes

- hsp_: ndarray of shape (n_spheres, 4)
  The fitted sphere parameters `[D, P, H, R]` for each sphere.
- error_: float
  Objective value at the optimum (lower is better).
- accuracy_: float
  Set by `score(X, y)`. Classification accuracy of inside/outside vs. predicted.

## Methods

- fit(X, y) -> self
  Fit the estimator to training data. `X` shape `(n_samples, 3)` with columns `[D, P, H]`. `y` shape `(n_samples,)` labels/scores.
- predict(X) -> ndarray, shape `(n_samples,)`
  Predict 1 if a point is inside any sphere, else 0.
- transform(X) -> ndarray, shape `(n_samples, 1)`
  Compute RED (Relative Energy Distance) as `min(dist/R)` to the fitted sphere(s).
- score(X, y) -> float
  Returns accuracy vs. binarized labels using `inside_limit`.
- fit_transform(X, y) -> ndarray
  Convenience method calling `fit` then `transform`.
- binarize_labels(y, inside_limit=1) -> ndarray
  Static helper: map labels to binary using the chosen `inside_limit`.

## Input/Output contract

- Input X: array-like with columns `[D, P, H]`.
- Input y: array-like labels/scores; `NaN` entries are ignored for fitting and scoring.
- Fitted params: `hsp_` stores `[D, P, H, R]` for each sphere; distances use `sqrt(4 dD^2 + dP^2 + dH^2)`.

## Notes

- With `n_spheres=2`, the optimizer adds constraints to avoid one sphere being fully contained in the other and to ensure each sphere contains at least one inside solvent.
- If the number of inside solvents is less than `n_spheres`, `fit` raises `ValueError`.
- Accuracy is only computed when you call `score`.

## Objective functions

This estimator fits sphere parameters by minimizing a loss built from per-sample penalties and a radius regularization term.

- Label binarization: labels `y` are converted to inside/outside using `inside_limit`:
  `inside = (0 < y <= inside_limit)`, `outside = otherwise`. `NaN` labels are ignored during fitting.
- Distance metric: for a sample `x = [D, P, H]` and a sphere center `[Ds, Ps, Hs]`,
  `dist = sqrt(4*(D-Ds)^2 + (P-Ps)^2 + (H-Hs)^2)`.

The penalty design and the two-sphere constraints follow formulations described by Díaz de los Ríos and collaborators in the attached references (2020; 2022), adapted here for a scikit‑learn style estimator.

Single sphere (`n_spheres=1`)

- Penalty per sample (`fi`): starts at 1.0.
  - Inside sample misclassified (dist >= R): `fi = exp(R - dist)` (penalizes being outside the sphere).
  - Outside sample misclassified (dist < R): `fi = exp(dist - R)` (penalizes being inside the sphere).
- Data fit term: geometric mean `DATAFIT = exp(mean(log(fi)))` (numerically stable).
- Radius regularization: encourages compact spheres via `R^(-1/n)`, where `n` is the number of valid samples.
- Objective returned to the optimizer: `1 - (DATAFIT * R^(-1/n))` (so minimization maximizes the product).

Double sphere (`n_spheres=2`)

- Compute distances and penalties (`fi1`, `fi2`) for each sphere as above.
- Combine penalties into a single `fi`:
  - For inside samples, if either sphere yields `fi == 1` (i.e., correctly inside), keep `fi = 1`.
  - Otherwise, use `min(fi1, fi2)` (the least penalty across spheres).
- Data fit: `DATAFIT = exp(mean(log(fi)))`.
- Radius regularization: `(R1 * R2)^(-1/n)`.
- Objective minimized: `1 - (DATAFIT * (R1*R2)^(-1/n))`.
- Extra constraints (enforced via large penalties):
  - No full containment between spheres: if `center_distance + min(R1, R2) <= max(R1, R2)`, add a very large penalty.
  - Each sphere must contain at least one inside (good) solvent; otherwise add a large penalty.

Rationale

- Exponential penalties grow smoothly with misclassification distance across the boundary.
- The geometric mean balances contributions and avoids numerical issues.
- The radius term discourages trivial solutions with overly large spheres while allowing enough coverage of inside points.
- The 2-sphere constraints avoid degenerate configurations and ensure each sphere is meaningful.

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

- Díaz de los Ríos, M.; Hernández Ramos (2020). Determination of the Hansen solubility parameters and the Hansen sphere radius with the aid of the solver add-in of Microsoft Excel. [DOI:10.1007/s42452-020-2512-y](https://link.springer.com/article/10.1007/s42452-020-2512-y)
- Díaz de los Ríos, M.; Belmonte (2022). Extending Microsoft excel and Hansen solubility parameters relationship to double Hansen’s sphere calculation. [DOI:10.1007/s42452-022-04959-4](https://doi.org/10.1007/s42452-022-04959-4)
