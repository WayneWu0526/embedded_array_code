import os
from .base import (
    SensorArrayConfig, SensorArrayManifest,
    AffineModelParamsSet, SensorArrayHardwareParams
)

_QMC6309_ROOT = os.path.join(os.path.dirname(__file__), "config", "qmc6309")

class QMC6309Config(SensorArrayConfig):
    def __init__(self, config_root: str = _QMC6309_ROOT):
        self._config_root = config_root

    @property
    def manifest(self) -> SensorArrayManifest:
        import yaml
        path = os.path.join(self._config_root, "calibration_manifest.yaml")
        with open(path) as f:
            d = yaml.safe_load(f)
        return SensorArrayManifest(
            sensor_type=d["sensor_type"],
            bit_width=d["bit_width"],
            range_gs=d["range_gs"],
            n_sensors=d["n_sensors"],
            n_groups=d["n_groups"],
            sensors_per_group=d["sensors_per_group"]
        )

    @property
    def affine_model(self) -> AffineModelParamsSet:
        return AffineModelParamsSet.from_json(
            os.path.join(self._config_root, "affine_model_params.json")
        )

    @property
    def hardware(self) -> SensorArrayHardwareParams:
        return SensorArrayHardwareParams.from_json(
            os.path.join(self._config_root, "sensor_array_params.json")
        )

    @property
    def gs_to_si(self) -> float:
        return 1.0e-4