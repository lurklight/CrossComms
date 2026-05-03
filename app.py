from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OVERRIDE_PACKAGES = ROOT / ".packages-override"
PACKAGES = ROOT / ".packages"
OPUS_PACKAGES = ROOT / ".packages-opus"
SRC = ROOT / "src"

for entry in reversed((OVERRIDE_PACKAGES, OPUS_PACKAGES, PACKAGES, SRC)):
    if entry.exists():
        entry_text = str(entry)
        if entry_text in sys.path:
            sys.path.remove(entry_text)
        sys.path.insert(0, entry_text)

NVIDIA_ROOT = PACKAGES / "nvidia"
if NVIDIA_ROOT.exists():
    runtime_dirs = [
        str(path)
        for path in sorted(NVIDIA_ROOT.glob("*/bin"))
        if path.exists()
    ]
    if runtime_dirs:
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(
            runtime_dirs + ([existing_path] if existing_path else [])
        )

from gamevoice.app import main


if __name__ == "__main__":
    main()
