import sys, numpy as np, joblib
sys.path.insert(0, "."); sys.path.insert(0, "conf")
import matplotlib; matplotlib.use("Agg")
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config
from scipy.stats import chi2, norm
from heca.data.entity import Entity

pair = joblib.load("data/scenes/ogbench/scene0/experts/cube0_base_drawer0/conditions.joblib")
ent = pair.entities["cube0"]
up = pair.post.models["cube0"].get_parameters().copy()
p = ent.secure_mix_parameters(up, add_variance=True)
cfg = ent.cfg
chi_sqrt = float(np.sqrt(chi2.ppf(cfg.z_quantile_joint, 6)))   # joint 0.99 bound
z_dim_cap = float(norm.ppf(0.5 + cfg.z_quantile_dim / 2.0))    # per-dim cap

def check(cube_pos, tag):
    sample = np.concatenate([cube_pos, [0.,0.,0.], [0.]])  # pos + aa(0) + ste
    s = ent.model_value(sample); pose = s[:-1]
    best = None
    for k in range(len(p["weights"])):
        var = np.maximum(p["measurement"]["pose"]["covariances"][k], 1e-15)
        m = p["measurement"]["pose"]["means"][k]
        post = np.log(p["weights"][k]) - 0.5*np.sum(np.log(2*np.pi*var) + (pose-m)**2/var)
        if best is None or post > best[0]:
            best = (post, k)
    k = best[1]
    var = np.maximum(p["measurement"]["pose"]["covariances"][k], 1e-15)
    m = p["measurement"]["pose"]["means"][k]
    zd = np.abs(pose - m)/np.sqrt(var)
    z = float(np.sqrt(np.sum(zd**2)))
    print(f"{tag}: best comp k{k}, joint z={z:.2f} (gate bound {chi_sqrt:.2f}), max|zd|={zd.max():.2f} (cap {z_dim_cap:.2f}) -> {'PASS' if z<=chi_sqrt and zd.max()<=z_dim_cap else 'FAIL'}")

check(np.array([-0.97, -2.56, 0.66]), "shallow goal y=-2.56")
check(np.array([-0.97, -2.80, 0.66]), "shallow goal y=-2.80")
check(np.array([-0.97, -4.05, 0.66]), "deep goal y=-4.05")
check(np.array([-0.92, -4.13, 0.66]), "deep goal y=-4.13")
print("post means y:", np.round(p["measurement"]["pose"]["means"][:,1],3))
