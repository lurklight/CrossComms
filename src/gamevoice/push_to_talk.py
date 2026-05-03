from __future__ import annotations

from dataclasses import dataclass
import ctypes


@dataclass(frozen=True, slots=True)
class PushToTalkBinding:
    code: int
    label: str


_BINDINGS = [
    PushToTalkBinding(0x01, "Left Mouse"),
    PushToTalkBinding(0x02, "Right Mouse"),
    PushToTalkBinding(0x04, "Middle Mouse"),
    PushToTalkBinding(0x05, "Mouse 4"),
    PushToTalkBinding(0x06, "Mouse 5"),
    PushToTalkBinding(0x08, "Backspace"),
    PushToTalkBinding(0x09, "Tab"),
    PushToTalkBinding(0x0D, "Enter"),
    PushToTalkBinding(0x10, "Shift"),
    PushToTalkBinding(0x11, "Ctrl"),
    PushToTalkBinding(0x12, "Alt"),
    PushToTalkBinding(0x14, "Caps Lock"),
    PushToTalkBinding(0x1B, "Escape"),
    PushToTalkBinding(0x20, "Space"),
    PushToTalkBinding(0x25, "Left Arrow"),
    PushToTalkBinding(0x26, "Up Arrow"),
    PushToTalkBinding(0x27, "Right Arrow"),
    PushToTalkBinding(0x28, "Down Arrow"),
]
_BINDINGS.extend(
    PushToTalkBinding(code, chr(code)) for code in range(ord("A"), ord("Z") + 1)
)
_BINDINGS.extend(
    PushToTalkBinding(code, str(code - 0x30)) for code in range(0x30, 0x3A)
)
_BINDINGS.extend(
    PushToTalkBinding(code, f"F{code - 0x6F}") for code in range(0x70, 0x7D)
)

SUPPORTED_BINDINGS = tuple(_BINDINGS)
SUPPORTED_BINDINGS_BY_LABEL = {binding.label: binding for binding in SUPPORTED_BINDINGS}
SUPPORTED_BINDINGS_BY_CODE = {binding.code: binding for binding in SUPPORTED_BINDINGS}


def default_push_to_talk_binding() -> PushToTalkBinding:
    return SUPPORTED_BINDINGS_BY_LABEL["Mouse 5"]


class WindowsInputMonitor:
    def __init__(self) -> None:
        self._user32 = getattr(ctypes, "windll", None)
        self._user32 = getattr(self._user32, "user32", None)

    def is_available(self) -> bool:
        return self._user32 is not None

    def is_pressed(self, binding: PushToTalkBinding | None) -> bool:
        if binding is None or self._user32 is None:
            return False
        return bool(self._user32.GetAsyncKeyState(binding.code) & 0x8000)

    def pressed_codes(self) -> set[int]:
        if self._user32 is None:
            return set()
        return {
            binding.code
            for binding in SUPPORTED_BINDINGS
            if self._user32.GetAsyncKeyState(binding.code) & 0x8000
        }

    def capture_new_binding(
        self,
        ignored_codes: set[int] | None = None,
    ) -> PushToTalkBinding | None:
        ignored = ignored_codes or set()
        for binding in SUPPORTED_BINDINGS:
            if binding.code in ignored:
                continue
            if self.is_pressed(binding):
                return binding
        return None
