"""Register project-local CUDA and system HIP runtime DLLs before llama_cpp."""
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

    native_library_roots = []
    for variable in ("HIP_PATH", "ROCM_PATH"):
        value = os.environ.get(variable)
        if value:
            native_library_roots.append(Path(value))

    amd_rocm = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "AMD" / "ROCm"
    if amd_rocm.is_dir():
        native_library_roots.extend([amd_rocm, *amd_rocm.glob("*")])

    visual_studio_roots = (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio",
    )
    for visual_studio_root in visual_studio_roots:
        native_library_roots.extend(
            visual_studio_root.glob(
                "2022/*/VC/Redist/MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM"
            )
        )

    for root in native_library_roots:
        for directory in (root, root / "bin", root / "lib"):
            if directory.is_dir():
                try:
                    _dll_handles.append(os.add_dll_directory(str(directory)))
                except OSError:
                    continue
                os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
