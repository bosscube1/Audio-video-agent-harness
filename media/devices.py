"""Audio device discovery and validation helpers."""

from __future__ import annotations

import re
from typing import Any

import sounddevice as sd

_LOOPBACK_INPUT_RE = re.compile(r"CABLE Output|Stereo Mix|What U Hear", re.IGNORECASE)
_LOOPBACK_OUTPUT_RE = re.compile(r"CABLE Input", re.IGNORECASE)


def list_input_devices() -> list[dict[str, Any]]:
    """Return all host audio devices that expose input channels."""
    return [
        dict(device)
        for device in sd.query_devices()
        if device.get("max_input_channels", 0) > 0
    ]


def list_output_devices() -> list[dict[str, Any]]:
    """Return all host audio devices that expose output channels."""
    return [
        dict(device)
        for device in sd.query_devices()
        if device.get("max_output_channels", 0) > 0
    ]


def validate_device(name: str | None, kind: str) -> int | None:
    """Resolve a human-readable device name to its PortAudio index.

    Args:
        name: Device name to look up. ``None`` or an empty string selects the
            PortAudio default.
        kind: ``"input"`` or ``"output"``.

    Returns:
        The device index, or ``None`` when ``name`` is ``None``/empty.

    Raises:
        ValueError: If no matching device is found.
    """
    if not name:
        return None

    if kind == "input":
        candidates = list_input_devices()
    elif kind == "output":
        candidates = list_output_devices()
    else:
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}")

    clean_name = name.strip().lower()
    for device in candidates:
        if device.get("name", "").strip().lower() == clean_name:
            return int(device["index"])

    raise ValueError(f"Unknown {kind} device: {name!r}")


def detect_loopback() -> bool:
    """Return True if the default devices look like a software loopback cable."""
    try:
        default_input = sd.query_devices(kind="input")
        default_output = sd.query_devices(kind="output")
    except Exception:
        return False

    input_name = str(default_input.get("name", ""))
    output_name = str(default_output.get("name", ""))

    return bool(
        _LOOPBACK_INPUT_RE.search(input_name) and _LOOPBACK_OUTPUT_RE.search(output_name)
    )
