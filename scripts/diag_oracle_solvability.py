"""Oracle solvability vs trained argmax on the SAME episode stream.

Proper isolation: after sample_task the env sits at x0. Each random-chain
retry snapshots/restores the env and resets the option counter, so every
retry starts from x0 exactly like a fresh episode. Success = fb.terminal with
reward>0 (matches Buffer.stats + Heca episode end).

Usage:
    python scripts/diag_oracle_solvability.py --network default \
        --tag scene0_test4-default-_gv_b --ckp ckp_100.pt --episodes 120
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


def env_snapshot(scene) -> tuple:
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


def env_restore(scene, snap) -> None:
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
    """Execute one chain from x (env must already sit at x). Returns (ok, opts)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="default")
    ap.add_argument("--ckp", default="ckp_100.pt")
    ap.add_argument("--tag", default="scene0_test4-default-_gv_b")
    ap.add_argument("--episodes", type=int, default=120)
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

    n_solv, n_argmax, both = 0, 0, 0
    lens = []
    for ep in range(args.episodes):
        (x0, _), (y0, _) = scene.sample_task()  # env now at x0
        snap0 = env_snapshot(scene)
        scene.current_step = 0

        # --- oracle: random chains, env restored to x0 each retry
        solved = False
        for _ in range(args.tries):
            env_restore(scene, snap0)
            scene.current_step = 0
            graph.set_goal(y0)
            graph.set_start(x0)

            def rand_policy(data, keys):
                return int(rng.randint(len(keys)))

            ok, ln = run_chain(x0, y0, graph, rand_policy, max_opts)
            if ok:
                solved = True
                lens.append(ln)
                break

        # --- argmax single chain from x0
        env_restore(scene, snap0)
        scene.current_step = 0
        graph.set_goal(y0)
        graph.set_start(x0)
        ok2, _ = run_chain(x0, y0, graph, argmax_policy, max_opts)
        if solved:
            n_solv += 1
        if ok2:
            n_argmax += 1
            if solved:
                both += 1

    tot = args.episodes
    print(f"[{args.tag} {args.ckp} tries={args.tries} ep={tot} seed={args.seed}]")
    print(f"  oracle-solvable (random, <= {args.tries} retries): "
          f"{n_solv}/{tot} = {100*n_solv/tot:.1f}%")
    print(f"  argmax success: {n_argmax}/{tot} = {100*n_argmax/tot:.1f}%")
    if lens:
        print(f"  oracle solved-chain length: mean={np.mean(lens):.1f} "
              f"max={max(lens)} (budget={max_opts})")


if __name__ == "__main__":
    main()
