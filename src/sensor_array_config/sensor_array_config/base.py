from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os
import numpy as np

# ---------- manifest schema ----------
@dataclass
class SensorArrayManifest:
    sensor_type: str
    bit_width: int
    range_gs: float
    n_sensors: int
    n_groups: int
    sensors_per_group: int
    adu_to_gs: float = field(init=False)

    def __post_init__(self):
        self.adu_to_gs = self.range_gs / (2 ** (self.bit_width - 1))

# ---------- consistency params (Phase 2) ----------
# NOTE: Kept because NormalizedParamsSet reuses ConsistencyParams in its type annotation.
# Once NormalizedParamsSet is updated to use its own type, this can be removed.
@dataclass
class ConsistencyParams:
    D_i: List[List[float]]
    e_i: List[float]

# ---------- normalized params (Phase 2 variant: trained on b_ref_norm) ----------
@dataclass
class NormalizedParamsSet:
    params: Dict[int, ConsistencyParams]  # reuses D_i, e_i from ConsistencyParams

    @classmethod
    def from_json(cls, path: str) -> "NormalizedParamsSet":
        with open(path) as f:
            raw = json.load(f)
        params = {}
        if "sensors" in raw:
            for entry in raw["sensors"]:
                params[entry["sensor_id"]] = ConsistencyParams(
                    D_i=entry["D_i"],
                    e_i=entry["e_i"]
                )
        else:
            for sid, entry in raw.items():
                if sid == "amp_factor":
                    continue
                params[int(sid)] = ConsistencyParams(
                    D_i=entry["D_i"],
                    e_i=entry["e_i"]
                )
        return cls(params=params)

    def to_json(self, path: str):
        raw = {str(k): {"D_i": v.D_i, "e_i": v.e_i} for k, v in self.params.items()}
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)

# ---------- sensor array hardware params ----------
@dataclass
class RCorrEntry:
    """Single R_CORR entry: a rotation matrix shared by a list of sensors."""
    sensor_ids: List[int]
    matrix: List[float]  # 9 elements, column-major (Fortran order)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.matrix).reshape(3, 3, order='F')


@dataclass
class SensorArrayHardwareParams:
    d_list: List[List[float]]
    R_CORR: List[RCorrEntry]

    @classmethod
    def from_json(cls, path: str) -> "SensorArrayHardwareParams":
        with open(path) as f:
            raw = json.load(f)
        d_list = raw["d_list"]
        # New format: R_CORR is a list of {sensor_ids, matrix} entries
        if isinstance(raw["R_CORR"], list) and len(raw["R_CORR"]) > 0:
            first = raw["R_CORR"][0]
            if isinstance(first, dict) and "sensor_ids" in first:
                R_CORR = [
                    RCorrEntry(entry["sensor_ids"], entry["matrix"])
                    for entry in raw["R_CORR"]
                ]
            else:
                # Old flat format: list of lists (group matrices concatenated)
                R_CORR = []
                n_groups = len(raw["R_CORR"]) // 9
                for i in range(n_groups):
                    R_CORR.append(RCorrEntry(
                        sensor_ids=list(range(i*3+1, i*3+4)),
                        matrix=raw["R_CORR"][i*9:(i+1)*9]
                    ))
        # Legacy format: r_corr_r1, r_corr_r2, ... keys
        elif "r_corr_r1" in raw:
            R_CORR = []
            for i in range(1, 5):
                key = f"r_corr_r{i}"
                R_CORR.append(RCorrEntry(
                    sensor_ids=list(range((i-1)*3+1, i*3+1)),
                    matrix=raw[key]
                ))
        else:
            R_CORR = []
        return cls(
            d_list=d_list,
            R_CORR=R_CORR
        )

# ---------- sensor array config (facade) ----------
class SensorArrayConfig(ABC):
    """Abstract base for a sensor array configuration."""

    @property
    @abstractmethod
    def manifest(self) -> SensorArrayManifest:
        ...

    @property
    @abstractmethod
    def normalized(self) -> NormalizedParamsSet:
        ...

    @property
    @abstractmethod
    def hardware(self) -> SensorArrayHardwareParams:
        ...

    @property
    @abstractmethod
    def gs_to_si(self) -> float:
        """Conversion from Gauss to SI (Tesla = Gs * this)."""
        ...

    def get_sensor_ids(self) -> List[int]:
        return list(range(1, self.manifest.n_sensors + 1))

    def get_group_for_sensor(self, sensor_id: int) -> int:
        sensors_per_group = self.manifest.sensors_per_group
        return (sensor_id - 1) // sensors_per_group

    def get_sensors_in_group(self, group: int) -> List[int]:
        start = group * self.manifest.sensors_per_group + 1
        end = start + self.manifest.sensors_per_group
        return list(range(start, end))

# ---------- sensor type enum ----------
from enum import Enum

class SensorType(Enum):
    QMC6309 = "QMC6309"

# ---------- registry ----------
_REGISTRY: Dict[str, type] = {}

def get_config(sensor_type: str, **kwargs) -> SensorArrayConfig:
    if sensor_type not in _REGISTRY:
        raise ValueError(f"Unknown sensor type '{sensor_type}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[sensor_type](**kwargs)

def register_sensor_type(name: str, cls: type):
    _REGISTRY[name] = cls

def _lazy_register():
    from .qmc6309 import QMC6309Config
    _REGISTRY["QMC6309"] = QMC6309Config
_lazy_register()