import numpy as np
from calibration.lib.center_field_estimator import CenterFieldEstimator

def test_estimator_init():
    """Test that estimator initializes and precomputes w."""
    estimator = CenterFieldEstimator()
    assert estimator.w is not None
    assert estimator.w.shape == (12, 1)
    print("test_estimator_init PASSED")

def test_estimator_basic():
    """Test that estimate_from_row returns a (3,) vector."""
    estimator = CenterFieldEstimator()
    b_raw = np.random.randn(36)  # one row
    b_hat = estimator.estimate_from_row(b_raw)
    assert b_hat.shape == (3,)
    print("test_estimator_basic PASSED")

if __name__ == "__main__":
    test_estimator_init()
    test_estimator_basic()