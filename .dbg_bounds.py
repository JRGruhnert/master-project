import sys, numpy as np, h5py, joblib
sys.path.insert(0, "."); sys.path.insert(0, "conf")
import matplotlib; matplotlib.use("Agg")
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config

# drawer slide/pose bounds from demos (ext -1 = open, +1 = closed per user)
for tag in ["drawer0_a_b", "drawer0_b_a"]:
    f = h5py.File(f"data/scenes/ogbench/scene0/experts/{tag}/demos.h5", "r")
    sca = f["heca_drawer0_sca"][:,0]; mn = f["heca_drawer0_sca_min"][:,0]; mx = f["heca_drawer0_sca_max"][:,0]
    y = f["heca_drawer0_pos"][:,1]
    ext = 2*(sca-mn)/(mx-mn)-1
    print(f"{tag}: sca_min={mn[0]:.4f} sca_max={mx[0]:.4f} | ext range {ext.min():+.3f}..{ext.max():+.3f}")
    for lo, hi, name in [(-1.02, -0.8, "ext~-1 (OPEN)"), (0.8, 1.02, "ext~+1 (CLOSED)")]:
        m = (ext >= lo) & (ext <= hi)
        if m.any():
            print(f"   {name}: drawer pos y {y[m].min():+.3f}..{y[m].max():+.3f}, cube? n/a")
    f.close()

# cube in drawer: demo-end positions vs sampled goals (shallow vs deep)
pair = joblib.load("data/scenes/ogbench/scene0/experts/cube0_base_drawer0/conditions.joblib")
print("cube0_base_drawer0 demo-END cube pos y:", pair.post.data_raw["cube0"][:,1].min(), "..", pair.post.data_raw["cube0"][:,1].max())

scene_cfg = find_scene_config("scene0")
scene = Scene.get(scene_cfg, auto_load=False)
deep = []; shallow = []
for ep in range(400):
    (x,_),(y,_) = scene.sample_task()
    c = y.get("cube0").pos
    if c[2] > 0.5:
        (deep if c[1] < -3.5 else shallow).append(c[1])
print(f"sampled goal cube-in-drawer: shallow y {min(shallow):.3f}..{max(shallow):.3f} (n={len(shallow)}), deep y {min(deep):.3f}..{max(deep):.3f} (n={len(deep)})")
# also: what does the drawer look like at the two cube depths? goal drawer ext
