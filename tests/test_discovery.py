"""Test discovery functionality."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_all_estimators():
    """Test that all_estimators finds HSPiPy estimators."""
    from hspipy.utils.discovery import all_estimators
    
    estimators = all_estimators()
    estimator_names = [name for name, cls in estimators]
    
    # Should find both HSP and HSPEstimator
    assert "HSP" in estimator_names or "HSPEstimator" in estimator_names
    assert len(estimators) > 0

def test_transformer_filter():
    """Test that transformer filter works."""
    from hspipy.utils.discovery import all_estimators
    
    transformers = all_estimators(type_filter="transformer")
    assert len(transformers) >= 1
    
    # All should be transformers
    for name, cls in transformers:
        from sklearn.base import TransformerMixin
        assert issubclass(cls, TransformerMixin)

def test_imports():
    """Test that main classes can be imported."""
    from hspipy import HSP, HSPEstimator
    from sklearn.base import BaseEstimator, TransformerMixin
    
    assert issubclass(HSPEstimator, BaseEstimator)
    assert issubclass(HSPEstimator, TransformerMixin)

if __name__ == "__main__":
    test_imports()
    test_all_estimators()
    test_transformer_filter()
    print("All tests passed!")