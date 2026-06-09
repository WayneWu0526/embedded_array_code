import json
from pathlib import Path
from typing import Any, Dict, List

from .schemas import (
    AffineModelParams,
    AffineModelParamsSet,
    ArrayConfig,
    ArrayManifest,
    BoardProfile,
    HardwareParams,
    ImuConfig,
    RCorrEntry,
    SensorChipConfig,
)
from .validation import (
    validate_array_config,
    validate_imu_config,
    validate_profile,
    validate_sensor_chip_config,
)


def _resolve_config_root() -> Path:
    source_root = Path(__file__).resolve().parents[2] / "config"
    if source_root.exists():
        return source_root

    try:
        import rospkg
    except ImportError as exc:
        raise FileNotFoundError(
            f"sensor_array_config config directory not found at {source_root}"
        ) from exc

    ros_root = Path(rospkg.RosPack().get_path("sensor_array_config")) / "config"
    if ros_root.exists():
        return ros_root
    raise FileNotFoundError(f"sensor_array_config config directory not found at {ros_root}")


CONFIG_ROOT = _resolve_config_root()


def _load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(f"Cannot parse non-JSON YAML without PyYAML: {path}") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}")
        return data


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _sensor_type_path(name: str) -> Path:
    return _require_file(CONFIG_ROOT / "sensor_types" / f"{name.lower()}.yaml")


def _array_path(name: str) -> Path:
    return _require_file(CONFIG_ROOT / "arrays" / name)


def _imu_path(name: str) -> Path:
    return _require_file(CONFIG_ROOT / "imu_types" / f"{name.lower()}.yaml")


def _profile_path(name: str) -> Path:
    return _require_file(CONFIG_ROOT / "profiles" / f"{name.lower()}.yaml")


def _get_sensor_chip_config(name: str) -> SensorChipConfig:
    raw = _load_mapping(_sensor_type_path(name))
    config = SensorChipConfig(**raw)
    validate_sensor_chip_config(config)
    return config


def get_array_config(name: str) -> ArrayConfig:
    root = _array_path(name)
    manifest = ArrayManifest(**_load_mapping(root / "array_manifest.yaml"))
    sensor_type = _get_sensor_chip_config(manifest.sensor_type)
    hardware_raw = _load_json(root / "sensor_array_params.json")
    hardware = HardwareParams(
        description=hardware_raw.get("description", ""),
        d_list=hardware_raw["d_list"],
        R_CORR=[RCorrEntry(**entry) for entry in hardware_raw["R_CORR"]],
    )
    affine_raw = _load_json(root / "affine_model_params.json")
    params = {
        int(entry["sensor_id"]): AffineModelParams(D_i=entry["D_i"], e_i=entry["e_i"])
        for entry in affine_raw["sensors"]
    }
    config = ArrayConfig(
        name=name,
        manifest=manifest,
        sensor_type=sensor_type,
        hardware=hardware,
        affine_model=AffineModelParamsSet(params=params),
    )
    validate_array_config(config)
    return config


def get_imu_config(name: str) -> ImuConfig:
    config = ImuConfig(**_load_mapping(_imu_path(name)))
    validate_imu_config(config)
    return config


def get_profile(name: str) -> BoardProfile:
    profile = BoardProfile(**_load_mapping(_profile_path(name)))
    validate_profile(profile)
    _array_path(profile.default_array_config)
    _imu_path(profile.default_imu_config)
    for array_config in profile.startup_sensors:
        _array_path(array_config)
    return profile


def list_array_configs() -> List[str]:
    root = CONFIG_ROOT / "arrays"
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def list_imu_configs() -> List[str]:
    root = CONFIG_ROOT / "imu_types"
    return sorted(path.stem for path in root.glob("*.yaml"))


def list_profiles() -> List[str]:
    root = CONFIG_ROOT / "profiles"
    return sorted(path.stem for path in root.glob("*.yaml"))
