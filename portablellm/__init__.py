"""Register project-local CUDA runtime DLLs before importing llama_cpp."""
import os
import sys
from pathlib import Path

_dll_handles = []
if sys.platform == "win32":
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for component in ("cuda_runtime", "cublas"):
        directory = nvidia / component / "bin"
        if directory.is_dir():
            _dll_handles.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
