import pygfx as gfx
import pylinalg as la

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)

rot = la.quat_from_euler((0, 0.01), order="XY")

def animate():
    cube.local.rotation = la.quat_mul(rot, cube.local.rotation)

if __name__ == "__main__":
    gfx.show(cube, before_render=animate)