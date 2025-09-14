"""
HSPiPy - Hansen Solubility Parameters in Python

A Python package for calculating and visualizing Hansen Solubility Parameters.
"""

__version__ = "1.1.0b1"
__author__ = "Alejandro Gutierrez"
__email__ = "agutierrez@g-npd.com"

from .hsp import HSP
from .core.hsp_estimator import HSPEstimator

__all__ = ["HSP", "HSPEstimator"]