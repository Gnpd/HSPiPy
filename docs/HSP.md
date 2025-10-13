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

### `get(inside_limit=None, n_spheres=None)`

Fit HSP spheres to the loaded data and prepare for plotting.

**Parameters**:
- `inside_limit`: `float` or `None`, default=None  
  Threshold for inside vs outside classification. If None, uses the value from initialization.
- `n_spheres`: `int` or `None`, default=None  
  Number of spheres to fit. If None, uses value from initialization.

**Returns**:
- `hsp`: `ndarray`  
  Fitted HSP parameters.
- `radius`: `float` or `ndarray`  
  Fitted sphere radius/radii.
- `error`: `float`  
  Optimization error.
- `accuracy`: `float`  
  Classification accuracy.
- `DATAFIT`: `float`  
  DATAFIT value.

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
- Displays both 3D and 2D visualizations.

## Examples

### Basic Usage

```python
from hspipy import HSP

# Create HSP instance and load data
hsp = HSP()
hsp.read('solvent_data.csv')

# Fit model and get results
hsp, radius, error, accuracy, datafit = hsp.get()

# Visualize results
hsp.plots()
```

## Custom Parameters

```python
# Fit with custom parameters
hsp = HSP(inside_limit=2, n_spheres=2)
hsp.read('solvent_data.hsd')
hsp.get()

# Access individual parameters
print(f"D: {hsp.d}, P: {hsp.p}, H: {hsp.h}")
print(f"Radius: {hsp.radius}")
print(f"DATAFIT: {hsp.DATAFIT}")
print(f"Accuracy: {hsp.accuracy}")

```

## File Format Support

The `HSP` class supports multiple file formats through the `HSPDataReader`:

- **CSV**: Comma-separated values with columns for solvent name, D, P, H, and score
- **HSD**: Hansen Solubility Parameter data format
- **HSDX**: Extended HSP format with additional metadata


## Notes

- For plotting, the 3D axes are labelled as H (x), D (y), P (z) and the model centers are converted accordingly.
- `get()` prints formatted results and returns `(hsp, radius, error, accuracy, DATAFIT)` for convenience.
