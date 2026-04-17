from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
import json
import os

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

# ---------- intrinsic params (Phase 1) ----------
@dataclass
class IntrinsicParams:
    o_i: List[float]
    C_i: List[List[float]]

@dataclass
class IntrinsicParamsSet:
    params: Dict[int, IntrinsicParams]

    @classmethod
    def from_json(cls, path: str) -> "IntrinsicParamsSet":
        with open(path) as f:
            raw = json.load(f)
        params = {}
        for sid, entry in raw.items():
            params[int(sid)] = IntrinsicParams(
                o_i=entry["o_i"],
                C_i=entry["C_i"]
            )
        return cls(params=params)

    def to_json(self, path: str):
        raw = {str(k): {"o_i": v.o_i, "C_i": v.C_i} for k, v in self.params.items()}
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)

# ---------- consistency params (Phase 2) ----------
@dataclass
class ConsistencyParams:
    D_i: List[List[float]]
    e_i: List[float]

@dataclass
class ConsistencyParamsSet:
    params: Dict[int, ConsistencyParams]

    @classmethod
    def from_json(cls, path: str) -> "ConsistencyParamsSet":
        with open(path) as f:
            raw = json.load(f)
        params = {}
        for sid, entry in raw.items():
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
class SensorArrayHardwareParams:
    d_list: List[List[float]]
    R_CORR: List[List[float]]

    @classmethod
    def from_json(cls, path: str) -> "SensorArrayHardwareParams":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            d_list=raw["d_list"],
            R_CORR=raw["R_CORR"]
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
    def intrinsic(self) -> IntrinsicParamsSet:
        ...

    @property
    @abstractmethod
    def consistency(self) -> ConsistencyParamsSet:
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