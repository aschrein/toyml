from pygltflib import GLTF2
import requests
from toyml.utils import get_or_create_tmp

# dowload test gltf file
test_url = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/main/2.0/BarramundiFish/glTF-Binary/BarramundiFish.glb"
dst = get_or_create_tmp() / "BarramundiFish.glb"
if not dst.exists():
    r = requests.get(test_url)
    with open(dst, "wb") as f:
        f.write(r.content)

# load gltf file
gltf = GLTF2().load(dst)
print(gltf)