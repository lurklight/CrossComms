from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / ".packages"
SRC = ROOT / "src"

for entry in (PACKAGES, SRC):
    if entry.exists() and str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from gamevoice.app import main


if __name__ == "__main__":
    main()
