from __future__ import annotations

from dataclasses import dataclass

def audio_backend_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False
    return True


def list_input_devices() -> list[str]:
    return [option.label for option in list_input_device_options()]


def list_output_devices() -> list[str]:
    return [option.label for option in list_output_device_options()]


@dataclass(frozen=True, slots=True)
class DeviceOption:
    index: int
    name: str
    hostapi: str
    default_sample_rate: int
    max_input_channels: int
    max_output_channels: int
    label: str


def list_input_device_options() -> list[DeviceOption]:
    return _list_device_options("input")


def list_output_device_options() -> list[DeviceOption]:
    return _list_device_options("output")


def _list_device_options(direction: str) -> list[DeviceOption]:
    try:
        import sounddevice as sd
    except ImportError:
        return []

    hostapis = sd.query_hostapis()
    options: list[DeviceOption] = []
    for index, device in enumerate(sd.query_devices()):
        if direction == "input" and int(device.get("max_input_channels", 0)) <= 0:
            continue
        if direction == "output" and int(device.get("max_output_channels", 0)) <= 0:
            continue
        hostapi_index = int(device.get("hostapi", -1))
        hostapi_name = "Unknown API"
        if 0 <= hostapi_index < len(hostapis):
            hostapi_name = str(hostapis[hostapi_index].get("name", "Unknown API"))
        name = str(device.get("name", "Unknown device"))
        label = f"{name} [{hostapi_name}]"
        default_sample_rate = int(float(device.get("default_samplerate", 48_000.0)))
        options.append(
            DeviceOption(
                index=index,
                name=name,
                hostapi=hostapi_name,
                default_sample_rate=default_sample_rate,
                max_input_channels=int(device.get("max_input_channels", 0)),
                max_output_channels=int(device.get("max_output_channels", 0)),
                label=label,
            )
        )
    return options
