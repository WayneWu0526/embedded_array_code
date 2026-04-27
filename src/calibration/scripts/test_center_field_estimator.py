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

def test_init_with_sensor_ids():
    """Test init with valid sensor subset."""
    estimator = CenterFieldEstimator(sensor_ids=[1, 2, 3])
    assert estimator.w is not None
    assert estimator.w.shape[0] == 3  # 3 sensors
    print("test_init_with_sensor_ids PASSED")

def test_init_default_all_sensors():
    """Test init without sensor_ids uses all 12."""
    estimator = CenterFieldEstimator()
    assert estimator.w.shape[0] == 12
    print("test_init_default_all_sensors PASSED")

def test_invalid_sensor_id():
    """Test ValueError for out-of-range sensor ID."""
    try:
        CenterFieldEstimator(sensor_ids=[1, 2, 99])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Sensor ID 99 not found" in str(e)
        print("test_invalid_sensor_id PASSED")

def test_too_few_sensors():
    """Test ValueError when fewer than 3 sensors."""
    try:
        CenterFieldEstimator(sensor_ids=[1, 2])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "at least 3 sensors" in str(e)
        print("test_too_few_sensors PASSED")

def test_duplicate_sensor_ids():
    """Test ValueError for duplicate IDs."""
    try:
        CenterFieldEstimator(sensor_ids=[1, 1, 2, 3])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Duplicate" in str(e)
        print("test_duplicate_sensor_ids PASSED")

def test_estimate_with_subset():
    """Test estimate_batch with subset sensors (sensors 1,2,3 only)."""
    import pandas as pd

    estimator = CenterFieldEstimator(sensor_ids=[1, 2, 3])
    csv_path = Path(__file__).resolve().parent.parent.parent / 'sensor_data_collection/data/manual_x/manual_record_5V.csv'
    df = pd.read_csv(csv_path)
    b_raw = df.values[:10]  # 10 rows

    b_hats = estimator.estimate_batch(b_raw)
    assert b_hats.shape == (10, 3)
    assert np.all(np.isfinite(b_hats))
    mags = np.linalg.norm(b_hats, axis=1)
    assert np.all(mags > 1.0)
    print(f"test_estimate_with_subset PASSED: shape={b_hats.shape}, mean_mag={np.mean(mags):.2f}")

if __name__ == "__main__":
    test_estimator_init()
    test_estimator_basic()
    test_r_corr_shape()
    test_estimate_known_row()
    test_batch()
    test_real_data()
    test_init_with_sensor_ids()
    test_init_default_all_sensors()
    test_invalid_sensor_id()
    test_too_few_sensors()
    test_duplicate_sensor_ids()
    test_estimate_with_subset()