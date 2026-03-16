[![PyPI](https://img.shields.io/pypi/v/HSPiPy.svg)](https://pypi.org/project/HSPiPy/) ![version](https://img.shields.io/badge/version-1.1.7-orange.svg) 

# HSPiPy

#### Hansen Solubility Parameters in Python.

### Introduction
---------------

HSPiPy is a Python library designed for calculating and visualizing Hansen Solubility Parameters (HSP). It provides machine-learning–friendly estimators, convenient data import, and plotting tools for analyzing solvent compatibility in materials science, polymers, and coatings.

### Features
---------------

* Read solvent data from multiple formats (CSV, HSD, HSDX)
* Calculate Hansen Solubility Parameters using robust optimization methods
* Support for single or multiple (up to 2) solubility spheres
* Generate 2D and 3D visualizations of solubility spheres
* Fully compatible with scikit-learn for machine learning workflows


### Documentation
---------------

- Class reference:
    - [HSP (high‑level helper)](https://github.com/Gnpd/HSPiPy/blob/main/docs/HSP.md)
    - [HSPEstimator (scikit‑learn compatible)](https://github.com/Gnpd/HSPiPy/blob/main/docs/HSPEstimator.md)


### Installation
---------------

Install **HSPiPy** easily with pip:

```
pip install HSPiPy
```
#### Dependencies

* numpy
* pandas
* matplotlib
* scipy
* scikit-learn

### Usage
--------

#### Reading HSP Data

To read HSP data from a file (CSV, HSD or HSDX), create an instance of the `HSP` class and use the `read` method:
```python
from hspipy import HSP

hsp = HSP()
hsp.read('path_to_your_hsp_file.csv')

```


#### Calculating HSP

Use the `get` method to calculate the Hansen Solubility Parameters (HSP) from your data:
```python
# For a single sphere model (default)
result = hsp.get(inside_limit=1)

# For a double sphere model
result = hsp.get(inside_limit=1, n_spheres=2)
```

`get()` returns an `HSPResult` object. In a Jupyter notebook it renders as a formatted table automatically. In a script use `print(result)` or access attributes directly:

```python
print(result.hsp)             # fitted D, P, H center coordinates
print(result.radius)          # sphere radius
print(result.accuracy)        # classification accuracy
print(result.datafit)         # DATAFIT value
print(result.n_solvents_in)   # number of good solvents
print(result.n_solvents_out)  # number of bad solvents
print(result.n_total)         # total solvents
print(result.n_wrong_in)      # bad solvents inside sphere (false positives)
print(result.n_wrong_out)     # good solvents outside sphere (false negatives)
```

The `inside_limit` parameter defines the threshold score value to consider a solvent as "inside" the solubility sphere (default: `inside_limit=1`).

#### Visualizing HSP

Use the `plot_3d` and `plot_2d` methods to visualize the HSP data in 3D and 2D formats, respectively:
```python
# Generate individual plots
hsp.plot_3d()
hsp.plot_2d()

# Or generate both plots at once
hsp.plots()

```
![3dHSP](https://github.com/Gnpd/HSPiPy/blob/main/3dPlot.png)
![2dHSP](https://github.com/Gnpd/HSPiPy/blob/main/2dPlot.png)

### `HSP` class methods:
| Method              |      Description                                                                       |
|---------------------|:--------------------------------------------------------------------------------------:|
| read(path)          |  Reads solvent data from a CSV, HSD, or HSDX file.                                    |
| get(inside_limit=1, n_spheres=1) |  Fits HSP sphere(s) and returns an `HSPResult` object.                  |
| plot_3d()           |  Plots the HSP data in 3D. Returns the figure.                                        |
| plot_2d()           |  Plots the HSP data in 2D. Returns the figure.                                        |
| plots()             |  Generates both 2D and 3D plots. Returns `(fig_3d, fig_2d)`.                          |

`get()` returns an `HSPResult` object. The following attributes are available on it and also set on the `HSP` instance after calling `get()`:

| Attribute            | Description                                                                                                                             |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `result.hsp`         | Numpy array — Fitted HSP coordinates. Shape: `(3,)` for single-sphere (D, P, H), or `(n_spheres, 3)` for multiple spheres.             |
| `result.radius`      | Float or array — Radius (or radii) of the solubility sphere(s).                                                                        |
| `result.error`       | Float — Objective function value from the optimization (lower is better).                                                              |
| `result.accuracy`    | Float — Classification accuracy of the fitted model on the dataset.                                                                    |
| `result.datafit`     | Float — DATAFIT value (geometric mean fitness; 1.0 = perfect classification).                                                          |
| `result.n_solvents_in`  | Int — Number of good solvents (`0 < score <= inside_limit`).                                                                       |
| `result.n_solvents_out` | Int — Number of bad solvents (`score == 0` or `score > inside_limit`).                                                             |
| `result.n_total`        | Int — Total number of solvents.                                                                                                     |
| `result.n_wrong_in`     | Int — Bad solvents predicted inside the sphere (false positives).                                                                  |
| `result.n_wrong_out`    | Int — Good solvents predicted outside the sphere (false negatives).                                                                |
| `hsp.d`, `hsp.p`, `hsp.h` | Float — Individual HSP components (δD, δP, δH). Single-sphere only; `None` for multi-sphere.                                   |
| `hsp.inside`         | List — Solvents classified as *inside* the solubility sphere(s), with their HSP values and scores.                                     |
| `hsp.outside`        | List — Solvents classified as *outside* the solubility sphere(s), with their HSP values and scores.                                    |
| `hsp.grid`           | Pandas DataFrame — The full input dataset, standardized with columns: `Solvent`, `D`, `P`, `H`, and `Score`.                           |


### Using scikit-learn style estimator
```python
import numpy as np
from hspipy import HSPEstimator

# Example dataset (D, P, H values with scores)
X = np.array([
    [16.0, 8.0, 5.0],
    [18.0, 7.5, 9.0],
    [20.0, 10.0, 12.0],
])
y = np.array([1, 1, 0])  # Inside/outside labels or scores

est = HSPEstimator(n_spheres=1)
est.fit(X, y)

print("Fitted HSP:", est.hsp_)
print("Accuracy:", est.score(X, y))

```

### Contributing
----------------

Contributions are welcome! If you have any suggestions, feature requests, or bug reports, please open an issue on the [GitHub repository](https://github.com/Gnpd/HSPiPy/issues).


### License
-----------

This library is licensed under the MIT License. See the [LICENSE](https://github.com/Gnpd/HSPiPy/blob/main/LICENSE) file for details.

### Acknowledgements
----------------

HSPiPy was inspired by the well-known HSP software suit [Hansen Solubility Parameters in Practice (HSPiP)](https://www.hansen-solubility.com/HSPiP/) and by the HSP community.




