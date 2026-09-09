"""Head-to-head: does the trained GNN policy (argmax and sampled) solve what
test_graph.py proves solvable? Mirrors c01/Heca execution exactly (same
graph.generate/select, same ExpertModel.act/virtual, same count_option), only
the option source changes.

Usage:
    python scripts/diag_graph_vs_heca.py --network default --ckp ckp_100 \
        --tag scene0_test4-default-_gv_b --episodes 120
    python scripts/diag_graph_vs_heca.py --network both --ckp ckp_100 \
        --tag scene0_test4-both-_gv_b --episodes 120
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)  # must precede site-packages `conf` shadow

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from torch.distributions import Categorical

from heca.agents.heca import Heca
from heca.experts.expert import ExpertModel
from heca.graphs.graph import SubgoalMode
from heca.learning.ppo import PPO
from heca.misc import hardware
from heca.scenes.scene import Scene

import conf.networks
from scripts.common.scenes import find_scene_config, find_scene_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="default")
    ap.add_argument("--ckp", default="ckp_100.pt")
    ap.add_argument("--tag", default="scene0_test4-default-_gv_b")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--mode", choices=["argmax", "sample"], default="argmax")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net_cfg = getattr(conf.networks, args.network)
    scene_cfg = find_scene_config("scene0")
    model_cfgs = find_scene_models("scene0")

    scene = Scene.get(scene_cfg, auto_load=False)

    # --- network weights from the trained checkpoint ---------------------
    from heca.heca_gnn.network import Network

    network = Network.get(net_cfg)  # Configurable.get: no auto_load kwarg
    ckp = torch.load(
        Path("data/network/standard") / args.tag / args.ckp,
        map_location=hardware.device,
        weights_only=False,
    )
    network.load_state_dict(ckp["network"])
    network.eval()

    # --- experts + graph exactly like Heca.__init__ / c01 ----------------
    for agent_cfg in model_cfgs:
        expert = ExpertModel.get(agent_cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()
    from heca.graphs.graph import Graph

    graph = Graph.generate(list(model_cfgs), smode=SubgoalMode.BOTH)

    stats = {"success": 0, "trunc": 0, "term_fail": 0, "opt_tot": 0, "opt_succ": 0}
    failed: list[int] = []
    for ep in range(args.episodes):
        (x, _), (y, _) = scene.sample_task()
        graph.set_goal(y)
        graph.set_start(x)
        n_opts = 0
        term = trunc = False
        last_fb = None
        for _ in range(200):  # hard cap (scene.max_steps governs truncation)
            data = graph.export()
            if data["option"].x.shape[0] == 0:
                term, trunc, last_fb = False, True, None
                break
            with torch.inference_mode():
                logits = network.actor(data)
            if args.mode == "argmax":
                action = int(logits.argmax(dim=-1))
            else:
                action = int(Categorical(logits=logits).sample())
            a, s = graph.select(action)
            z, fb = ExpertModel.get(a).act(x, s)
            n_opts += 1
            x = z
            graph.set_start(x)
            if fb.truncated or fb.terminal:
                term, trunc, last_fb = fb.terminal, fb.truncated, fb
                break
        stats["opt_tot"] += n_opts
        if last_fb is not None and term and last_fb.reward > 0.0:
            stats["success"] += 1
            stats["opt_succ"] += n_opts
        elif trunc:
            stats["trunc"] += 1
            failed.append(ep)
        else:
            stats["term_fail"] += 1
            failed.append(ep)

    tot = args.episodes
    print(
        f"[{args.tag} {args.ckp} net={args.network} mode={args.mode} "
        f"ep={tot} seed={args.seed}]"
    )
    print(
        f"  success={stats['success']} ({100 * stats['success'] / tot:.1f}%)  "
        f"truncated={stats['trunc']}  terminal-fail={stats['term_fail']}"
    )
    if stats["success"]:
        print(
            f"  mean options/successful-ep = {stats['opt_succ'] / stats['success']:.2f}"
        )
    print(f"  mean options/episode = {stats['opt_tot'] / tot:.2f}")
    print(f"  failed episodes: {failed}")


if __name__ == "__main__":
    main()
