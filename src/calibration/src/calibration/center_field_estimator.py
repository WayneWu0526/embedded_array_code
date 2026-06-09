import numpy as np

from sensor_array_config import get_array_config


def null(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Computes null space of matrix A. Same as gels_localization."""
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    n = A.shape[1]
    rank = np.sum(s > rcond * s[0]) if s.size > 0 else 0
    k = n - rank
    if k > 0:
        return Vh[rank:, :].T
    else:
        return np.zeros((n, 0))


class CenterFieldEstimator:
    """Estimates local center magnetic field from sensor array measurements."""

    def __init__(self, sensor_config=None, sensor_ids=None):
        if sensor_config is None:
            sensor_config = get_array_config("qmc6309_12ch_v1")
        self.sensor_config = sensor_config
        self.full_d_list = np.array(sensor_config.hardware.d_list)  # (12, 3)
        self.n_sensors = int(sensor_config.manifest.n_sensors)

        # Default: all 12 sensors
        if sensor_ids is None:
            sensor_ids = list(range(1, self.n_sensors + 1))

        # Validate sensor_ids
        self._validate_sensor_ids(sensor_ids)

        # Filter d_list to selected sensors
        self.sensor_ids = sensor_ids
        indices = [sid - 1 for sid in sensor_ids]
        self.d_list = self.full_d_list[indices, :]  # (N_selected, 3)

        self._precompute_weight_vector()

    def _validate_sensor_ids(self, sensor_ids):
        """Validate sensor_ids list."""
        if not isinstance(sensor_ids, (list, tuple)):
            raise ValueError(f"sensor_ids must be a list or tuple of integers, got {type(sensor_ids).__name__}")

        if len(sensor_ids) == 0:
            raise ValueError(f"sensor_ids must have at least 3 sensors, got 0")

        if len(sensor_ids) < 3:
            raise ValueError(f"sensor_ids must have at least 3 sensors, got {len(sensor_ids)}")

        seen = set()
        for sid in sensor_ids:
            if not isinstance(sid, int):
                raise ValueError(
                    f"sensor_ids must be integers in 1-{self.n_sensors}, got {type(sid).__name__}"
                )
            if sid < 1 or sid > self.n_sensors:
                raise ValueError(f"Sensor ID {sid} not found. Available: 1-{self.n_sensors}")
            if sid in seen:
                raise ValueError(f"Duplicate sensor IDs: {sensor_ids}")
            seen.add(sid)

    def _precompute_weight_vector(self):
        """Precompute w from d_list null space. Called once on init."""
        N = len(self.sensor_ids)
        D_cal = self.d_list.T  # (3, N)
        Q = null(D_cal)
        ones_N = np.ones((N, 1))
        g = Q.T @ ones_N
        g_norm_sq = float((g.T @ g).item())
        if g_norm_sq < 1e-10:
            # Fallback for N=3 case (zero null space): use uniform weights
            if N == 3:
                self.w = ones_N / N  # uniform weights [1/3, 1/3, 1/3]
            else:
                raise ValueError(
                    f"Cannot construct w: 1 has zero projection onto null(D_cal) "
                    f"(N={N}, null_space_dim={Q.shape[1]})"
                )
        else:
            self.w = (Q @ g) / g_norm_sq  # (N, 1)

    def estimate_from_row(self, b_raw_row):
        """Estimate center field for a single row.

        Args:
            b_raw_row: (n_sensors*3,) or (n_sensors, 3) raw sensor data for all sensors.
                        Only sensors in self.sensor_ids are used.

        Returns:
            b_hat: (3,) center field estimate
        """
        N_selected = len(self.sensor_ids)
        # Filter to selected sensor columns from full 12-sensor data
        b_raw_filtered = self._filter_to_selected_sensors(b_raw_row)
        b_aligned = b_raw_filtered
        if b_aligned.ndim == 2 and b_aligned.shape[0] == N_selected:
            B_meas = b_aligned.T  # (3, N_selected)
        else:
            raise ValueError(f"Unexpected selected raw shape: {b_aligned.shape}")
        b_hat = B_meas @ self.w  # (3, 1) -> (3,)
        return b_hat.ravel()

    def _filter_to_selected_sensors(self, b_raw):
        """Filter raw data to only include selected sensor columns.

        Args:
            b_raw: (n_sensors*3,) or (n_sensors, 3) raw data for all sensors

        Returns:
            (N_selected, 3) raw data for selected sensors only
        """
        if b_raw.ndim == 1:
            b_raw = b_raw.reshape(self.n_sensors, 3)
        # self.sensor_ids is 1-indexed, convert to 0-indexed
        indices = [sid - 1 for sid in self.sensor_ids]
        return b_raw[indices]  # (N_selected, 3)

    def estimate_batch(self, b_raw):
        """Estimate center field for multiple rows.

        Args:
            b_raw: (N, 36) or (N, 12, 3) raw sensor data

        Returns:
            b_hats: (N, 3) center field estimates
            b_corr: (N, N_selected, 3) orientation-aligned raw sensor readings
        """
        expected_cols = self.n_sensors * 3
        if b_raw.ndim == 2 and b_raw.shape[1] == expected_cols:
            b_raw = b_raw.reshape(b_raw.shape[0], self.n_sensors, 3)

        N = b_raw.shape[0]
        N_sel = len(self.sensor_ids)
        b_hats = np.zeros((N, 3))
        b_corr = np.zeros((N, N_sel, 3))
        for i in range(N):
            b_raw_filtered = self._filter_to_selected_sensors(b_raw[i])
            b_corr[i] = b_raw_filtered
            B_meas = b_raw_filtered.T  # (3, N_selected)
            b_hats[i] = (B_meas @ self.w).ravel()
        return b_hats, b_corr
