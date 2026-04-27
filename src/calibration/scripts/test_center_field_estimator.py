import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

def test_r_corr_shape():
    """Test apply_r_corr preserves shape."""
    estimator = CenterFieldEstimator()
    b_raw = np.random.randn(12, 3)
    b_rcorr = estimator.apply_r_corr(b_raw)
    assert b_rcorr.shape == (12, 3)
    print("test_r_corr_shape PASSED")

def test_estimate_known_row():
    """Test estimate_from_row returns finite values."""
    estimator = CenterFieldEstimator()
    b_raw = np.random.randn(36)
    b_hat = estimator.estimate_from_row(b_raw)
    assert b_hat.shape == (3,)
    assert np.all(np.isfinite(b_hat))
    print("test_estimate_known_row PASSED")

def test_batch():
    """Test estimate_batch processes multiple rows."""
    estimator = CenterFieldEstimator()
    b_raw = np.random.randn(5, 36)  # 5 rows
    b_hats = estimator.estimate_batch(b_raw)
    assert b_hats.shape == (5, 3)
    assert np.all(np.isfinite(b_hats))
    print(f"test_batch PASSED: {b_hats.shape}")

def test_real_data():
    """Verify with actual manual_record_5V.csv data."""
    import pandas as pd

    csv_path = Path(__file__).resolve().parent.parent.parent / 'sensor_data_collection/data/manual_x/manual_record_5V.csv'
    df = pd.read_csv(csv_path)
    b_raw = df.values[:100]  # first 100 rows

    estimator = CenterFieldEstimator()
    b_hats = estimator.estimate_batch(b_raw)

    print(f"Input shape: {b_raw.shape}")
    print(f"Output shape: {b_hats.shape}")
    print(f"b_hat stats: mean={np.mean(b_hats, axis=0)}, std={np.std(b_hats, axis=0)}")
    print(f"b_hat magnitude: mean={np.mean(np.linalg.norm(b_hats, axis=1)):.2f}")

    # Sanity: magnitude should be in reasonable range (10-100 µT)
    mags = np.linalg.norm(b_hats, axis=1)
    assert np.all(mags > 1.0), f"Magnitude too small: {mags.min()}"
    assert np.all(mags < 200.0), f"Magnitude too large: {mags.max()}"
    print("test_real_data PASSED")

if __name__ == "__main__":
    test_estimator_init()
    test_estimator_basic()
    test_r_corr_shape()
    test_estimate_known_row()
    test_batch()
    test_real_data()