[Home](index.md) | [HSP](HSP.md) | [HSPEstimator](HSPEstimator.md) | [Discovery](discovery.md)

# HSP

`class HSP`

A high-level helper for Hansen Solubility Parameters that extends [`HSPEstimator`](HSPEstimator.md) with data loading and plotting utilities.

The `HSP` class provides a simplified interface for typical HSP workflows, including data loading from various formats, automatic model fitting, and visualization capabilities. It maintains full sklearn compatibility while adding domain-specific functionality.

[View source code](https://github.com/Gnpd/HSPiPy/blob/main/hspipy/hsp.py)

## Inheritance

`HSP` inherits from [`HSPEstimator`](HSPEstimator.md) and adds the following capabilities:
- Data loading from multiple file formats
- Automatic data preprocessing
- 2D and 3D visualization
- Simplified parameter access

## Parameters

- `inside_limit`: `float`, default=1  
  Threshold that defines which labels are considered "inside" (good) solvents.
- `n_spheres`: `int`, default=1  
  Number of spheres to fit (1 or 2).

## Additional Attributes

- `grid`: `DataFrame` or `None`  
  The loaded HSP data containing solvents and their parameters.
- `inside`: `list`  
  Solvents classified as inside after fitting.
- `outside`: `list`  
  Solvents classified as outside after fitting.
- `DATAFIT`: `float` or `None`  
  DATAFIT value of the fitted model.
- `d`, `p`, `h`: `float`  
  Convenience access to individual HSP parameters (for single sphere).
- `hsp`: `ndarray`  
  The fitted HSP center coordinates.
- `radius`: `float` or `ndarray`  
  The fitted sphere radius/radii.
- `error`: `float`  
  Objective error value.
- `accuracy`: `float`  
  Classification accuracy.

## Methods

### `read(path)`

Read HSP data from various file formats.

**Parameters**:
- `path`: `str` or `Path`  
  Path to the HSP data file. Supports CSV, HSD, and HSDX formats with automatic format detection.

**Returns**:
- `self`: object  
  Returns self with loaded data.

### `get(inside_limit=1, n_spheres=1)`

Fit HSP spheres to the loaded data and prepare for plotting.

**Parameters**:
- `inside_limit`: `float`, default=1
  Threshold score value for classifying a solvent as inside the sphere (`0 < score <= inside_limit`).
- `n_spheres`: `int`, default=1
  Number of spheres to fit (1 or 2).

**Returns**:
- `HSPResult`
  Result object with the following attributes:
  - `.hsp`: `ndarray` — Fitted HSP center coordinates. Shape `(3,)` for single sphere, `(n_spheres, 3)` for multiple.
  - `.radius`: `float` or `ndarray` — Sphere radius or radii.
  - `.error`: `float` — Objective function value from the optimization (lower is better).
  - `.accuracy`: `float` — Classification accuracy on the input dataset.
  - `.datafit`: `float` — DATAFIT value (geometric mean fitness; 1.0 = perfect classification).
  - `.n_solvents_in`: `int` — Number of good solvents (`0 < score <= inside_limit`).
  - `.n_solvents_out`: `int` — Number of bad solvents (`score == 0` or `score > inside_limit`).
  - `.n_total`: `int` — Total number of solvents.
  - `.n_wrong_in`: `int` — Bad solvents predicted inside the sphere (false positives).
  - `.n_wrong_out`: `int` — Good solvents predicted outside the sphere (false negatives).

  In a Jupyter notebook the result renders as a formatted table automatically. In a script use `print(result)` or access attributes directly.

### `plot_3d()`

Create a 3D plot of the HSP space with spheres and solvents.

**Returns**:
- `matplotlib.figure.Figure`  
  3D visualization of the HSP space.

### `plot_2d()`

Create 2D projections of the HSP space.

**Returns**:
- `matplotlib.figure.Figure`  
  Three-panel 2D visualization showing P vs H, H vs D, and P vs D projections.

### `plots()`

Show both 3D and 2D plots.

**Returns**:
- `tuple` — `(fig_3d, fig_2d)` matplotlib figure objects.

## Examples

### Basic Usage

```python
from hspipy import HSP

# Create HSP instance and load data
hsp = HSP()
hsp.read('solvent_data.csv')

# Fit model — result renders as a table in Jupyter automatically
result = hsp.get()

# Access individual attributes
print(result.hsp)             # fitted D, P, H center
print(result.radius)          # sphere radius
print(result.accuracy)        # classification accuracy
print(result.datafit)         # DATAFIT value
print(result.n_solvents_in)   # number of good solvents
print(result.n_solvents_out)  # number of bad solvents
print(result.n_total)         # total solvents
print(result.n_wrong_in)      # bad solvents inside sphere (false positives)
print(result.n_wrong_out)     # good solvents outside sphere (false negatives)

# Visualize results
hsp.plots()
```

## Custom Parameters

```python
# Fit with custom parameters
hsp = HSP(inside_limit=2, n_spheres=2)
hsp.read('solvent_data.hsd')
result = hsp.get()

# Access individual parameters via the HSP instance (set after get())
print(f"D: {hsp.d}, P: {hsp.p}, H: {hsp.h}")
print(f"Radius: {hsp.radius}")
print(f"DATAFIT: {result.datafit}")
print(f"Accuracy: {result.accuracy}")
```

## File Format Support

The `HSP` class supports multiple file formats through the `HSPDataReader`:

- **CSV**: Comma-separated values with columns for solvent name, D, P, H, and score
- **HSD**: Hansen Solubility Parameter data format
- **HSDX**: Extended HSP format with additional metadata


## Notes

- For plotting, the 3D axes are labelled as H (x), D (y), P (z) and the model centers are converted accordingly.
- `get()` returns an `HSPResult` object. In a Jupyter notebook it renders as a formatted table automatically; in a script use `print(result)`.
- `plots()` returns `(fig_3d, fig_2d)` so figures can be saved or further customised.
