"""This file shows how to write test based on the scikit-learn check_estimator."""

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from hspipy.utils.discovery import all_estimators

def expected_failed_checks(estimator):
    if estimator.__class__.__name__ == "HSPEstimator" or estimator.__class__.__name__ == "HSP":
        # These test do not take into account that our estimators take exactly 3 features
        return {
            "check_n_features_in": "HSPEstimator requires exactly 3 features.",
            "check_estimators_overwrite_params": "HSPEstimator does not overwrite parameters.",
            "check_estimators_fit_returns_self": "HSPEstimator fit does not return self.",
            "check_readonly_memmap_input": "HSPEstimator does not support memmap inputs.",
            "check_n_features_in_after_fitting": "HSPEstimator requires exactly 3 features.",
            "check_estimators_dtypes": "HSPEstimator requires exactly 3 features.",
            "check_dtype_object": "HSPEstimator requires exactly 3 features.",
            "check_fit_idempotent": "HSPEstimator fit is not idempotent.",
            "check_fit_check_is_fitted": "HSPEstimator fit does not check if fitted.",
            "check_fit2d_1sample": "This estimator does not support fitting with only 1 sample",
        }
    return {}


# parametrize_with_checks allows to get a generator of check that is more fine-grained
# than check_estimator
@parametrize_with_checks(
    [est() for _, est in all_estimators()],
    expected_failed_checks=expected_failed_checks
)
def test_estimators(estimator, check, request):
    """Check the compatibility with scikit-learn API"""
    check(estimator)