"""Tests for the HSP high-level API."""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

from hspipy import HSP


CSV_PATH = "examples/hsp_example.csv"


@pytest.fixture
def fitted_hsp():
    h = HSP()
    h.read(CSV_PATH)
    h.get(inside_limit=1, n_spheres=1)
    return h


def test_read_csv():
    h = HSP()
    h.read(CSV_PATH)
    assert hasattr(h, "grid")
    assert h.grid is not None
    assert set(h.grid.columns) >= {"Solvent", "D", "P", "H", "Score"}
    assert len(h.grid) > 0


def test_get_single_sphere(fitted_hsp):
    assert hasattr(fitted_hsp, "hsp_")
    assert fitted_hsp.hsp_.shape == (1, 4), "Single sphere hsp_ should be (1, 4)"
    assert fitted_hsp.hsp_[0][3] > 0, "Radius must be positive"
    assert 0.0 <= fitted_hsp.accuracy_ <= 1.0


def test_get_double_sphere():
    h = HSP()
    h.read(CSV_PATH)
    h.get(inside_limit=1, n_spheres=2)
    assert h.hsp_.shape == (2, 4), "Double sphere hsp_ should be (2, 4)"
    for i in range(2):
        assert h.hsp_[i][3] > 0, f"Radius of sphere {i} must be positive"


def test_get_returns_tuple(fitted_hsp):
    h = HSP()
    h.read(CSV_PATH)
    result = h.get(inside_limit=1, n_spheres=1)
    assert isinstance(result, tuple)
    assert len(result) == 5  # hsp, radius, error, accuracy, DATAFIT


def test_plots_returns_figures(fitted_hsp):
    result = fitted_hsp.plots()
    assert isinstance(result, tuple)
    assert len(result) == 2
    import matplotlib.figure
    fig_3d, fig_2d = result
    assert isinstance(fig_3d, matplotlib.figure.Figure)
    assert isinstance(fig_2d, matplotlib.figure.Figure)


def test_plot_3d_returns_figure(fitted_hsp):
    import matplotlib.figure
    fig = fitted_hsp.plot_3d()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_2d_returns_figure(fitted_hsp):
    import matplotlib.figure
    fig = fitted_hsp.plot_2d()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_read_returns_self():
    h = HSP()
    result = h.read(CSV_PATH)
    assert result is h, "read() should return self for method chaining"


def test_read_missing_file():
    h = HSP()
    with pytest.raises(FileNotFoundError):
        h.read("nonexistent_file.csv")


def test_get_without_read():
    h = HSP()
    with pytest.raises(ValueError, match="No data loaded"):
        h.get()
