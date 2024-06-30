"""
    Try local build in case we've built a debug or something
"""
import sys, os
from utils import find_skbuild_pyd, get_build_type

build_type      = get_build_type()
local_pyd_path  = find_skbuild_pyd("native", "embree_wrapper", build_type)
if local_pyd_path:
    # print(f"Using local pyd path: {local_pyd_path}")
    sys.path.insert(0, str(local_pyd_path.parent))
    # native = __import__(str(local_pyd_path))
    import embree_wrapper as native

    # print(f"Using local embree_wrapper: {native.embree_version()}")
else:
    # print("Using installed embree_wrapper")
    import embree_wrapper as native

if __name__ == "__main__":
    print(f"Embree version: {native.embree_version()}")