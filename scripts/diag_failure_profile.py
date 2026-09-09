"""Full argmax failure profile for a trained checkpoint on scene0 (BOTH mode).

For every episode: classify the goal by cube location + drawer/window/button
goal states; run the trained argmax policy exactly like Heca.step; record
success/truncation. For every failure, dump the cube's value at start (x),
goal (y) and where the episode ended (z), plus drawer/window/button states,
the option budget used, and the sequence of chosen options.

Usage:
    python scripts/diag_failure_profile.py --network default \
        --tag scene0_test4-default-_gv_b --ckp ckp_100.pt --episodes 300
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from heca.experts.expert import ExpertModel
from heca.graphs.graph import SubgoalMode, Graph
from heca.misc import hardware
from heca.scenes.scene import Scene
from heca.heca_gnn.network import Network

import conf.networks
from scripts.common.scenes import find_scene_config, find_scene_models


def fmt(v):
    return [round(float(x), 3) for x in np.asarray(v).ravel()]


def goal_kind(y):
    cube_y = float(y.get("cube0").value[1])
    dr_ext = float(y.get("drawer0").value[6])
    win_ext = float(y.get("window0").value[6])
    b0 = float(y.get("button0").value[-1])
    b1 = float(y.get("button1").value[-1])
    if cube_y < -3.5:
        cube = "deep"
    elif cube_y > -2.2:
        cube = "out"
    else:
        cube = "shallow"
    tags = []
    tags.append("dr_closed" if dr_ext > 0.5 else "dr_open")
    tags.append("win_closed" if win_ext > 0.5 else "win_open")
    if b0 > 0.5:
        tags.append("b0")
    if b1 > 0.5:
        tags.append("b1")
    return f"{cube}({' '.join(tags)})"


def state_line(label, scene, x):
    e = x.get(label)
    return f"{label}={fmt(e.value)}"


def cube_state(scene, s):
    cube = s.get("cube0").value
    return dict(
        pos=fmt(cube[:3]),
        y=round(float(cube[1]), 3),
        value=fmt(cube),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="default")
    ap.add_argument("--ckp", default="ckp_100.pt")
    ap.add_argument("--tag", default="scene0_test4-default-_gv_b")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net_cfg = getattr(conf.networks, args.network)
    scene = Scene.get(find_scene_config("scene0"), auto_load=False)
    model_cfgs = find_scene_models("scene0")

    network = Network.get(net_cfg)
    ckp = torch.load(
        Path("data/network/standard") / args.tag / args.ckp,
        map_location=hardware.device,
        weights_only=False,
    )
    network.load_state_dict(ckp["network"])
    network.eval()

    for agent_cfg in model_cfgs:
        expert = ExpertModel.get(agent_cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()
    graph = Graph.generate(list(model_cfgs), smode=SubgoalMode.BOTH)
    max_opts = scene.cfg.max_steps

    from collections import Counter, defaultdict

    total = Counter()
    ok = Counter()
    fails = []
    for ep in range(args.episodes):
        (x0, _), (y0, _) = scene.sample_task()
        kind = goal_kind(y0)
        total[kind] += 1
        x, y = x0, y0
        graph.set_goal(y)
        graph.set_start(x)
        n_opts = 0
        chosen = []
        term = trunc = False
        last_fb = None
        for _ in range(max_opts + 5):
            data = graph.export()
            keys = graph.export_keys
            if len(keys) == 0:
                trunc = True
                break
            with torch.inference_mode():
                logits = network.actor(data)
            action = int(logits.argmax(dim=-1))
            chosen.append(keys[action])
            a, s = graph.select(action)
            z, fb = ExpertModel.get(a).act(x, s)
            n_opts += 1
            x = z
            graph.set_start(x)
            if fb.truncated or fb.terminal:
                term, trunc, last_fb = fb.terminal, fb.truncated, fb
                break
        if term and last_fb.reward > 0.0:
            ok[kind] += 1
        else:
            fails.append((ep, kind, x0, x, y0, n_opts, trunc, chosen, last_fb))

    print(f"[{args.tag} {args.ckp} net={args.network} ep={args.episodes} "
          f"seed={args.seed}]")
    print("=== success by goal kind ===")
    for k in sorted(total):
        print(f"  {k:<34} {ok[k]:>3}/{total[k]:<3}  "
              f"({100*ok[k]/total[k]:.0f}%)")
    tot = sum(total.values())
    tot_ok = sum(ok.values())
    print(f"  TOTAL                        {tot_ok}/{tot}  "
          f"({100*tot_ok/tot:.1f}%)")
    print(f"\n=== failures: {len(fails)} ===")
    for ep, kind, x0, xf, y, n_opts, trunc, chosen, fb in fails:
        print(f"\n--- ep {ep}  kind={kind}  options_used={n_opts}"
              f"  end={'truncated' if trunc else 'budget/terminal-fail'}")
        print(f"  goal    cube pos={cube_state(scene, y)['pos']}"
              f"  value={cube_state(scene, y)['value']}")
        print(f"  start   cube pos={cube_state(scene, x0)['pos']}"
              f"  value={cube_state(scene, x0)['value']}")
        print(f"  final   cube pos={cube_state(scene, xf)['pos']}"
              f"  value={cube_state(scene, xf)['value']}")
        print(f"  drawer0 goal={fmt(y.get('drawer0').value)} "
              f"start={fmt(x0.get('drawer0').value)} "
              f"final={fmt(xf.get('drawer0').value)}")
        print(f"  window0 goal={fmt(y.get('window0').value)} "
              f"start={fmt(x0.get('window0').value)} "
              f"final={fmt(xf.get('window0').value)}")
        print(f"  button0 goal={fmt(y.get('button0').value)} "
              f"start={fmt(x0.get('button0').value)}")
        print(f"  button1 goal={fmt(y.get('button1').value)} "
              f"start={fmt(x0.get('button1').value)}")
        # option trace: collapse consecutive repeats
        seq = []
        prev = None
        for k in chosen:
            if k != prev:
                seq.append(k)
                prev = k
        print(f"  options({len(chosen)}): {' -> '.join(seq)}")


if __name__ == "__main__":
    main()
