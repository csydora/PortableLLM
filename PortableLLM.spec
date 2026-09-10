# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
)

hiddenimports = (
    collect_submodules("portablellm")
    + collect_submodules("llama_cpp")
    + collect_submodules("sentence_transformers")
    + collect_submodules("transformers")
)

binaries = collect_dynamic_libs("llama_cpp")

datas = [
    ("config.json", "."),
]

a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PortableLLM",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="PortableLLM",
)