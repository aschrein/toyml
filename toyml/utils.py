import numpy as np
from pathlib import Path
import sys
import os

def mkdir_recursive(p: Path):
    if not p.exists():
        p.mkdir(parents=True)

def get_or_create_tmp(name=".tmp"):
    p = Path(name)
    if not p.exists():
        mkdir_recursive(p)
    return p.absolute()

def find_folder_with_file(name, root="."):
    p = Path(root).absolute()
    while os.path.exists(p) and p.parent != p:
        # print(f"Checking {p / name}")
        if os.path.exists(p / name):
            # print(f"Found {p / name}")
            return p / name
        p = p.parent
    return None

def set_build_type(build_type):
    os.environ["__SKBUILD_TYPE"] = build_type

def get_build_type():
    return os.environ.get("__SKBUILD_TYPE", "Release")

def find_skbuild_pyd(module, name, build_type):
    """
        Try to find a local pyd file in the build folder built by scikit-build
        Example:
        build\\temp.win-amd64-cpython-312\\Release\\_skbuild\\native\\Release\\embree_wrapper.cp312-win_amd64.pyd
    """

    build_folder = find_folder_with_file("build")
    if build_folder is None:
        return None
    # print(f"Found build folder: {build_folder}")
    version_suffix = f"{sys.version_info.major}{sys.version_info.minor}"
    tmp_path = build_folder / f"temp.win-amd64-cpython-{version_suffix}"
    if os.path.exists(tmp_path):
        # It's always Release?
        tmp_path = tmp_path / "Release" / "_skbuild"
        tmp_path = tmp_path / module / build_type
        if os.path.exists(tmp_path):
            full_path = tmp_path / f"{name}.cp{version_suffix}-win_amd64.pyd"
            if os.path.exists(full_path):
                return full_path
            else:
                return None
        else:
            return None
    else:
        return None