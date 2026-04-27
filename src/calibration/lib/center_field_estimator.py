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

    def estimate_from_row(self, b_raw_row):
        """Estimate center field for a single row."""
        raise NotImplementedError
