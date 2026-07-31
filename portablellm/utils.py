from pathlib import Path
import sys

def application_path() -> Path:

    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent

    return Path(__file__).parent.parent