# try to fetch the binary dependencies for the current platform

import os
import sys
# .zip unpacking
import zipfile
import urllib.request


def fetch_zip(url, dst):
    if os.path.exists(dst):
        print(f"Skipping download of {url} to {dst} as it already exists")
        return
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    print(f"Downloading {url} to {dst}")
    urllib.request.urlretrieve(url, dst + ".zip")
    with zipfile.ZipFile(dst + ".zip", 'r') as zip_ref:
        zip_ref.extractall(dst)


fetch_zip("https://github.com/RenderKit/embree/releases/download/v4.3.2/embree-4.3.2.x64.windows.zip", "bindeps/embree")
fetch_zip("https://github.com/oneapi-src/oneTBB/releases/download/v2021.13.0/oneapi-tbb-2021.13.0-win.zip", "bindeps/tbb")
fetch_zip("https://renderdoc.org/stable/1.33/RenderDoc_1.33_64.zip", "bindeps/renderdoc")