"""Per-episode: argmax outcome vs oracle-solvable, with goal classification.

For every episode: run trained argmax (one chain); then, if it failed, run
random-chain oracle with env isolation (up to `tries`). Report per-episode
goal type (deep-cube / cube-out / cube-shallow; buttons/window) so we can see
whether the ~10% argmax failures are exactly the oracle-unsolvable slice.

Usage:
    python scripts/diag_oracle_classify.py --network default \
        --tag scene0_test4-default-_gv_b --ckp ckp_100.pt --episodes 150
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


def env_snapshot(scene):
    env = scene.env
    obj_recs = []
    for o in env.objects:
        rec = {}
        for attr in ("_cur_state", "_target_button_states"):
            v = getattr(o, attr, None)
            rec[attr] = v.copy() if v is not None else None
        rec["_target_val"] = getattr(o, "_target_val", None)
        obj_recs.append((o, rec))
    return env._data.qpos.copy(), env._data.qvel.copy(), obj_recs


def env_restore(scene, snap):
    env = scene.env
    qpos, qvel, obj_recs = snap
    env._data.qpos[:] = qpos
    env._data.qvel[:] = qvel
    for o, rec in obj_recs:
        for attr, val in rec.items():
            if val is None:
                continue
            if attr == "_target_val":
                setattr(o, attr, val)
            else:
                cur = getattr(o, attr)
                cur[:] = val
    env._apply_button_states()
    env._success = bool(env._evaluate_success(env._compute_successes()))


def run_chain(x, y, graph, policy, max_opts):
    n = 0
    for _ in range(max_opts):
        data = graph.export()
        keys = graph.export_keys
        if len(keys) == 0:
            return False, n
        i = policy(data, keys)
        a, s = graph.select(i)
        z, fb = ExpertModel.get(a).act(x, s)
        n += 1
        x = z
        graph.set_start(x)
        if fb.terminal:
            return fb.reward > 0.0, n
        if fb.truncated:
            return False, n
    return False, n


def goal_kind(y, scene):
    cube = y.get("cube0").value
    dr = y.get("drawer0").value
    win = y.get("window0").value
    cube_y = float(cube[1])
    kind = "deep" if cube_y < -3.5 else ("out" if cube_y > -2.2 else "shallow")
    tags = []
    if float(dr[6]) > 0.9: tags.append("dr_closed")
    if float(win[6]) > 0.9: tags.append("win_closed")
    for b in ("button0", "button1"):
        if float(y.get(b).value[-1]) > 0.9:
            tags.append(b)
    return kind, "+".join(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="default")
    ap.add_argument("--ckp", default="ckp_100.pt")
    ap.add_argument("--tag", default="scene0_test4-default-_gv_b")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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
    rng = np.random.RandomState(args.seed)

    def argmax_policy(data, keys):
        with torch.inference_mode():
            logits = network.actor(data)
        return int(logits.argmax(dim=-1))

    from collections import Counter
    by_kind = Counter()
    fail_kind = Counter()
    fail_solvable = Counter()
    rows = []
    for ep in range(args.episodes):
        (x0, _), (y0, _) = scene.sample_task()
        kind, tags = goal_kind(y0, scene)
        by_kind[kind] += 1
        snap0 = env_snapshot(scene)
        scene.current_step = 0

        # argmax
        env_restore(scene, snap0)
        scene.current_step = 0
        graph.set_goal(y0)
        graph.set_start(x0)
        ok_am, ln_am = run_chain(x0, y0, graph, argmax_policy, max_opts)

        solved = ok_am
        ln_or = None
        if not ok_am:
            fail_kind[kind] += 1
            for _ in range(args.tries):
                env_restore(scene, snap0)
                scene.current_step = 0
                graph.set_goal(y0)
                graph.set_start(x0)

                def rand_policy(data, keys):
                    return int(rng.randint(len(keys)))

                ok_or, ln_or = run_chain(x0, y0, graph, rand_policy, max_opts)
                if ok_or:
                    solved = True
                    break
            if solved:
                fail_solvable[kind] += 1
            rows.append((ep, kind, tags, "argmax-ok" if ok_am else (
                "oracle-solved" if solved else "UNSOLVABLE"), ln_am, ln_or))

    tot = args.episodes
    print(f"[{args.tag} {args.ckp} tries={args.tries} ep={tot} seed={args.seed}]")
    print(f"goal mix: {dict(by_kind)}")
    print(f"argmax fails by kind: {dict(fail_kind)}")
    print(f"of those oracle-solved: {dict(fail_solvable)}")
    print(f"argmax success: {tot - sum(fail_kind.values())}/{tot} = "
          f"{100*(tot - sum(fail_kind.values()))/tot:.1f}%")
    print(f"unsolvable (neither argmax nor {args.tries}-try oracle): "
          f"{sum(fail_kind.values()) - sum(fail_solvable.values())}")
    print("detail:")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
