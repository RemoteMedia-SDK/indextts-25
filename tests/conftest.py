"""Use the actual SDK Python types; only the inference model is replaced."""

import os
import sys
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]
SDK = Path(
    os.environ.get("REMOTEMEDIA_PYTHON_SRC", PLUGIN.parents[1] / "clients" / "python")
).resolve()
if (SDK / "remotemedia").is_dir():
    sys.path.insert(0, str(SDK))
    if (SDK.parents[1] / "remotemedia" / "protos").is_dir():
        sys.path.insert(0, str(SDK.parents[1]))
sys.path.insert(0, str(PLUGIN))
