# try to fetch the binary dependencies for the current platform

import os
import sys
# .zip unpacking
import zipfile
import urllib.request

def add_to_path(p):
    """
        Add a path to the system PATH
        if the directory containst a single folder we add
        child folder to the path
    """
    p = os.path.abspath(p)
    if os.path.exists(p):
        num_entries = len(os.listdir(p))
        if num_entries == 1:
            for f in os.listdir(p):
                if os.path.isdir(p / f):
                    sys.path.append(str(p / f))
                    return
        sys.path.append(str(p))

def fetch_zip(url, dst):
    if os.path.exists(dst):
        add_to_path(dst)
        print(f"Skipping download of {url} to {dst} as it already exists")
        return
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    print(f"Downloading {url} to {dst}")
    urllib.request.urlretrieve(url, dst + ".zip")
    with zipfile.ZipFile(dst + ".zip", 'r') as zip_ref:
        zip_ref.extractall(dst)
    # cleanup
    add_to_path(dst)
    os.remove(dst + ".zip")

def check_if_exe_in_path(exe):
    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        exe_file = os.path.join(path, exe)
        if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
            return True
    return False
if not check_if_exe_in_path("embree.dll"):
    fetch_zip("https://github.com/RenderKit/embree/releases/download/v4.3.2/embree-4.3.2.x64.windows.zip", "bindeps/embree")
    fetch_zip("https://github.com/oneapi-src/oneTBB/releases/download/v2021.13.0/oneapi-tbb-2021.13.0-win.zip", "bindeps/tbb")
if not check_if_exe_in_path("qrenderdoc.exe"):
    fetch_zip("https://renderdoc.org/stable/1.33/RenderDoc_1.33_64.zip", "bindeps/renderdoc")
if not check_if_exe_in_path("blender.exe"):
    fetch_zip("https://mirrors.sahilister.in/blender/release/Blender4.1/blender-4.1.1-windows-x64.zip", "bindeps/blender")
if not check_if_exe_in_path("compressonatorcli.exe"):
    fetch_zip("https://github.com/GPUOpen-Tools/compressonator/releases/download/V4.5.52/compressonatorcli-4.5.52-win64.zip", "bindeps/compressonator")
