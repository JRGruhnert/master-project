"""Ablate the two optional network blocks (interaction, timeline memory) on a
trained checkpoint, to explain why their training curves are indistinguishable.

Runs a fixed, seeded sequence of scene0 episodes with the checkpoint's argmax
(and optionally sampled) policy and reports, per configuration:

* success rate,
* how often the argmax changes when the interaction block is replaced by the
  identity (paired, same states), plus the mean/std of the logit shift,
* pairwise cosine similarity of the option rows before vs after the interaction
  block (if the block makes the option rows *more* similar, it is flattening the
  per-option differences the readout ranks on).

``--memory on|off`` controls whether the recurrent memory is carried across
option steps of an episode the way ``Learner.predict`` does
(``data.mem_step = (previous_option_embedding, previous_memory)``). Old
checkpoints predate the state-aggregation layer, so the script also restores the
mean-pooled goal block they were trained with.

Usage:
    PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl \
        python scripts/diag_network_ablations.py --tag scene0_test8-both-_gv_b \
            --network both --episodes 40
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
from torch import nn
from torch.distributions import Categorical

import conf.networks
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.heca_gnn.network import Network
from heca.misc import hardware
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models


def pairwise_cosine(rows: torch.Tensor) -> np.ndarray:
    x = nn.functional.normalize(rows.float(), dim=-1)
    sim = (x @ x.T).numpy()
    n = sim.shape[0]
    return sim[np.triu_indices(n, k=1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="scene0")
    ap.add_argument("--tag", default="scene0_test8-both-_gv_b")
    ap.add_argument("--checkpoint", default="ckp_500.pt")
    ap.add_argument("--network", default="both")
    ap.add_argument("--smode", default="both")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["argmax", "sample"], default="argmax")
    ap.add_argument("--memory", choices=["on", "off"], default="auto")
    ap.add_argument("--legacy-goal-pool", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    scene = Scene.get(find_scene_config(args.scene), auto_load=False)
    model_cfgs = find_scene_models(args.scene)
    for cfg in model_cfgs:
        expert = ExpertModel.get(cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()
    graph = Graph.generate(list(model_cfgs), smode=SubgoalMode(args.smode))

    network = Network.get(getattr(conf.networks, args.network))
    ckp = torch.load(
        Path("data/network/standard") / args.tag / args.checkpoint,
        map_location=hardware.device,
        weights_only=False,
    )
    network.upgrade(ckp["network"])
    if args.legacy_goal_pool and "state_aggregation" not in " ".join(ckp["network"].keys()):
        # Pre-aggregation checkpoints saw the mean of the canonical goal rows.
        network._goal_slot = lambda entity_x, data: (  # type: ignore[method-assign]
            entity_x[data["entity"].goal_idx].mean(dim=0, keepdim=True)
        )
    network.eval()
    block = network.interaction_layer
    use_mem = network.timeline is not None
    if args.memory == "on":
        use_mem = True
    elif args.memory == "off":
        use_mem = False

    captured: dict[str, torch.Tensor] = {}
    if block is not None:
        block.register_forward_pre_hook(
            lambda _m, inputs: captured.__setitem__("before", inputs[0].detach())
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ok = trunc = steps_total = 0
    cos_before, cos_after = [], []
    shifts, shift_stds = [], []
    same_pick = changed_pick = 0

    for _ep in range(args.episodes):
        # Sample inside the loop: in virtual mode the env has to be reset to the
        # task the scene was taken from, otherwise the rollout is scored against
        # a different task and truncates immediately.
        (x0, _), (y0, _) = scene.sample_task()
        graph.set_goal(y0)
        graph.set_start(x0)
        x = x0
        pending = None
        term = False
        last = None
        for _ in range(scene.cfg.max_steps + 5):
            try:
                data = graph.export()
            except RuntimeError:
                break
            if use_mem and pending is not None:
                data.mem_step = pending
            with torch.inference_mode():
                logits = network.actor(data)
                before = captured.get("before")
                after = network._last_option_x
                if block is not None:
                    network.interaction_layer = nn.Identity()
                    logits_ablated = network.actor(data)
                    network.interaction_layer = block
            steps_total += 1
            if block is not None and before is not None and before.shape[0] > 1:
                cos_before.append(float(pairwise_cosine(before).mean()))
                cos_after.append(float(pairwise_cosine(after).mean()))
                diff = logits.squeeze(0) - logits_ablated.squeeze(0)
                shifts.append(float(diff.abs().mean()))
                shift_stds.append(float(diff.std()))
                if int(torch.argmax(logits)) == int(torch.argmax(logits_ablated)):
                    same_pick += 1
                else:
                    changed_pick += 1
            action = (
                torch.argmax(logits, dim=-1)
                if args.mode == "argmax"
                else Categorical(logits=logits).sample()
            )
            if use_mem:
                pending = (
                    network._last_option_x[action].detach(),
                    network._last_mem.detach(),
                )
            model_cfg, subgoal = graph.select(int(action))
            z, fb = ExpertModel.get(model_cfg).act(x, subgoal)
            x = z
            graph.set_start(x)
            if fb.truncated or fb.terminal:
                term, last = fb.terminal, fb
                trunc += int(fb.truncated)
                break
        ok += int(bool(term and last is not None and last.reward > 0.0))

    print(
        f"tag={args.tag} ckp={args.checkpoint} net={args.network} "
        f"mode={args.mode} memory={'on' if use_mem else 'off'} eps={args.episodes}"
    )
    print(f"  success                : {ok}/{args.episodes}")
    print(f"  truncated episodes     : {trunc}/{args.episodes}")
    print(f"  option steps           : {steps_total}")
    if cos_before:
        print(
            f"  option-row cos BEFORE interaction: {np.mean(cos_before):.4f}"
            f" | AFTER: {np.mean(cos_after):.4f}"
        )
        print(
            f"  interaction ablation: argmax unchanged {same_pick}/"
            f"{same_pick + changed_pick}, logit shift |mean| {np.mean(shifts):.3f} "
            f"vs |std| {np.mean(shift_stds):.3f}"
        )
    else:
        print("  (no interaction block in this network)")


if __name__ == "__main__":
    main()
