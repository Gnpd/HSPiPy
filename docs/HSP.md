# HSP

High-level helper for Hansen Solubility Parameters that extends `HSPEstimator`
with data loading and plotting utilities.

- Module: `hspipy.hsp`
- Class: `hspipy.HSP`
- Adds I/O and visualization on top of the scikit-learn compatible estimator.

## Parameters

- inside_limit: float, default=1
  Threshold that defines inside (good) vs. outside solvents. Overrides the estimator if you pass a new value to `get`.
- n_spheres: int, default=1
  Number of spheres to fit. Set to 2 for a double-sphere model.

## Attributes (after `get`)

- hsp: ndarray of shape (3,) or (n_spheres, 3)
  Center(s) `[D, P, H]` of the fitted sphere(s) for convenience.
- radius: float or ndarray of shape (n_spheres,)
  Sphere radius (or radii) `R`.
- error: float
  Objective value at optimum (mirror of `error_`).
- accuracy: float
  Classification accuracy (mirror of `accuracy_`).
- grid: pandas.DataFrame
  Loaded dataset with columns: `Solvent, D, P, H, Score`.
- inside, outside: list
  Lists of entries split for plotting convenience.

## Methods

- read(path)
  Read HSP data. Supports CSV and custom HSP formats handled by `HSPDataReader`.
- get(inside_limit=1, n_spheres=1)
  Fit to the loaded `grid`, compute metrics, and populate `hsp`, `radius`, `inside`, `outside`.
- plot_3d()
  3D plot: centers, wireframe spheres, and solvent points (good=blue, bad=red).
- plot_2d()
  2D projections: (P,H), (H,D), (P,D) with circles for each sphere.
- plots()
  Convenience: run both `plot_3d` and `plot_2d`.

## Examples

Fit and plot (single sphere)

```python
import pandas as pd
from hspipy import HSP

hsp = HSP(inside_limit=1, n_spheres=1)
# Either load from file …
# hsp.read('path/to/file.csv')
# …or assign a DataFrame directly
hsp.grid = pd.read_csv('hspipy/hsp_example.csv')

center, radius, err, acc = hsp.get()
print('Center(s):', center)
print('Radius:', radius)
print('Error:', err)
print('Accuracy:', acc)

hsp.plots()  # shows a 3D view and three 2D projections
```

Double-sphere fit

```python
h2 = HSP(inside_limit=1, n_spheres=2)
h2.grid = pd.read_csv('hspipy/hsp_example.csv')
h2.get()
h2.plot_2d()
```

## Notes

- For plotting, the 3D axes are labelled as H (x), D (y), P (z) and the model centers are converted accordingly.
- When `n_spheres == 2`, both centers and radii are arrays; the plot functions iterate and draw each sphere.
- `get()` prints formatted results and returns `(hsp, radius, error, accuracy)` for convenience.
