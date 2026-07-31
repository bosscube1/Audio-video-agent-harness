"""Media I/O package for the Gemini Live Agent."""

from __future__ import annotations

from media.devices import (
    detect_loopback,
    list_input_devices,
    list_output_devices,
    validate_device,
)
from media.microphone import Microphone
from media.screen import ScreenCapture
from media.speaker import Speaker

__all__ = [
    "Microphone",
    "Speaker",
    "ScreenCapture",
    "detect_loopback",
    "list_input_devices",
    "list_output_devices",
    "validate_device",
]
