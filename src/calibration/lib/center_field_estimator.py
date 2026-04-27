import numpy as np
import sys
from pathlib import Path

# Add calibration lib path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from sensor_array_config.base import get_config


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

    def __init__(self, sensor_config=None):
        if sensor_config is None:
            sensor_config = get_config("QMC6309")
        self.sensor_config = sensor_config
        self.d_list = np.array(sensor_config.hardware.d_list)  # (12, 3)
        self._load_r_corr()
        self._precompute_weight_vector()

    def _load_r_corr(self):
        """Load R_CORR matrices from sensor config into dict sensor_id -> np.array(3,3)."""
        self.R_CORR = {}
        for entry in self.sensor_config.hardware.R_CORR:
            mat = np.array(entry.matrix).reshape(3, 3, order='F')
            for sid in entry.sensor_ids:
                self.R_CORR[sid] = mat

    def _precompute_weight_vector(self):
        """Precompute w from d_list null space. Called once on init."""
        D_cal = self.d_list.T  # (3, 12)
        Q = null(D_cal)
        ones_N = np.ones((12, 1))
        g = Q.T @ ones_N
        g_norm_sq = float(g.T @ g)
        if g_norm_sq < 1e-10:
            raise ValueError("Cannot construct w: 1 has zero projection onto null(D_cal)")
        self.w = (Q @ g) / g_norm_sq  # (12, 1)

    def apply_r_corr(self, b_raw):
        """Apply R_CORR to raw sensor data.

        Args:
            b_raw: (N, 3) or (N*3,) raw sensor readings

        Returns:
            b_rcorr: (N, 3) with R_CORR applied per sensor
        """
        if b_raw.ndim == 1:
            b_raw = b_raw.reshape(12, 3)
        b_rcorr = np.zeros_like(b_raw)
        for sid in range(1, 13):
            sensor_idx = sid - 1
            if sid in self.R_CORR:
                b_rcorr[sensor_idx] = self.R_CORR[sid] @ b_raw[sensor_idx]
            else:
                b_rcorr[sensor_idx] = b_raw[sensor_idx]
        return b_rcorr

    def estimate_from_row(self, b_raw_row):
        """Estimate center field for a single row.

        Args:
            b_raw_row: (36,) or (12, 3) raw sensor data

        Returns:
            b_hat: (3,) center field estimate
        """
        b_rcorr = self.apply_r_corr(b_raw_row)
        if b_rcorr.ndim == 2 and b_rcorr.shape[0] == 12:
            B_meas = b_rcorr.T  # (3, 12)
        else:
            raise ValueError(f"Unexpected shape after R_CORR: {b_rcorr.shape}")
        b_hat = B_meas @ self.w  # (3, 1) -> (3,)
        return b_hat.ravel()

    def estimate_batch(self, b_raw):
        """Estimate center field for multiple rows.

        Args:
            b_raw: (N, 36) or (N, 12, 3) raw sensor data

        Returns:
            b_hats: (N, 3) center field estimates
        """
        if b_raw.ndim == 2 and b_raw.shape[1] == 36:
            b_raw = b_raw.reshape(b_raw.shape[0], 12, 3)

        N = b_raw.shape[0]
        b_hats = np.zeros((N, 3))
        for i in range(N):
            b_hats[i] = self.estimate_from_row(b_raw[i])
        return b_hats
