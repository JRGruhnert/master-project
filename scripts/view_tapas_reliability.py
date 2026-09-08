import argparse
import os
import sys
from pathlib import Path

# Make ``conf`` and ``scripts.common`` importable when run directly as
# ``python scripts/a05_view_model_reliability.py``.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless (conditions fitting imports matplotlib)

import numpy as np

from heca.data.data import DCEntity, DCScene
from heca.experts.expert import ExpertModel
from heca.experts.tapas import TapasExpert
from heca.graphs.graph import Graph, SubgoalMode
from heca.scenes.ogbench.scene import OGScene
from heca.scenes.scene import Scene, SceneFeedback

from scripts.common.args import (
    add_scene_argument,
    add_smode_argument,
    add_use_gt_argument,
    add_viewer_argument,
)
from scripts.common.scenes import find_scene_config, find_scene_models


def entity_str(label: str, dc: DCEntity) -> str:
    """Compact per-type value formatting for display."""
    if any(p in label for p in ("cube", "lid", "peg")):
        return f"pos={np.round(np.asarray(dc.pos, dtype=float), 4).tolist()}"
    if any(p in label for p in ("button", "box", "shelf")):
        return f"ste={int(dc.ste)}"
    if any(p in label for p in ("faucet", "lever")):
        return f"ang={float(np.arctan2(dc.ext[0], dc.ext[1])):+.4f}"
    if any(p in label for p in ("drawer", "window", "slider")):
        return f"ext={float(dc.ext[0]):+.4f}"
    return f"value={np.round(np.asarray(dc.value, dtype=float), 4).tolist()}"


def fmt_value(v):
    """Round arbitrary info_dict values (arrays / tuples / scalars) for print."""
    if isinstance(v, np.ndarray):
        return np.round(np.asarray(v, dtype=float), 4).tolist()
    if isinstance(v, tuple):
        return tuple(fmt_value(x) for x in v)
    return v


def value_diff(a, b) -> float:
    return float(
        np.linalg.norm(
            np.asarray(a.value, dtype=float) - np.asarray(b.value, dtype=float)
        )
    )


def print_scene(title: str, scene: DCScene, labels: list[str] | None = None):
    print(f"  {title}:")
    for label, entity in scene.entities():
        if labels is not None and label not in labels:
            continue
        print(f"    {label:<22} {entity_str(label, entity)}")


def precondition_status(model: ExpertModel, x: DCScene) -> tuple[bool, list[str]]:
    blocked = []
    for label, entity in model.entities.items():
        valid = entity.score_single(
            x.get(label).value,
            model.conditions.pre.models[label].get_parameters(),
        )
        if not valid:
            blocked.append(label)
    return len(blocked) == 0, blocked


def print_model_profile(model: ExpertModel):
    print(f"\nModel Profile:")
    for label in sorted(model.entities.keys()):
        score = model.conditions.change_scores[label]
        kind = "TARGET" if label in model.conditions.target_entities else "ANCHOR"
        print(f"    {label:<22} change={score:>8.3f}  [{kind}]")


def print_step_headline(headline: str, model: ExpertModel):
    print("\n" + "=" * 75)
    print(f"{headline}: {model.cfg.tag}")
    print("=" * 75)


def run_virtual_step(
    scene: OGScene,
    model: ExpertModel,
    x: DCScene,
    subgoal: DCScene,
    key: str,
) -> tuple[DCScene, SceneFeedback]:
    """Run the graph option's virtual step and report the roundtrip.

    ``subgoal`` is the graph's assembled subgoal (anchors pinned to the start
    scene, targets from the option's post-condition nodes). The info_dict is
    rebuilt from the subgoal the same way ``OGScene._step_virt`` builds it —
    no ogbench internals are touched.
    """
    print_step_headline(f"VIRTUAL STEP [{key}]", model)
    print_model_profile(model)
    model.virtual()
    z, fb = model.act(x, subgoal)

    sent: dict = {}
    for label in model.conditions.target_entities:
        sent.update(
            scene.entities[label].env_state_value(
                label, subgoal, unnormalize_pos=scene.unnormalize_position
            )
        )
    print("\nInfo_dict sent to env:")
    for key_, value in sent.items():
        print(f"    {key_:<28} = {fmt_value(value)}")
    print("\nEnv Validation Check:")
    for label in sorted(model.entities.keys()):
        dzy = value_diff(z.get(label), subgoal.get(label))
        if label in model.conditions.target_entities:
            status = "OK" if dzy < 1e-3 else "MISMATCH"
            note = f"|z-subgoal|={dzy:.5f} (target: should be ~0)"
        else:
            dzx = value_diff(z.get(label), x.get(label))
            status = "OK" if dzx < 1e-3 else "MISMATCH"
            note = f"|z-x|={dzx:.5f} (anchor: should be ~0)"
        print(f"    {label:<22} [{status:<8}] {note}")
    print_feedback(fb)
    print_result(model, x, z)
    return z, fb


def run_real_rollout(
    model: ExpertModel, x: DCScene, subgoal: DCScene, key: str
) -> tuple[DCScene, SceneFeedback]:
    """Execute the real TAPAS policy from x toward the option's subgoal."""
    assert isinstance(model, TapasExpert)
    print_step_headline(f"REAL ROLLOUT [{key}]", model)
    print_model_profile(model)
    model.real()
    z, fb = model.act(x, subgoal)
    print_feedback(fb)
    print_result(model, x, z)
    return z, fb


def print_feedback(fb: SceneFeedback):
    print(
        f"Feedback: terminal={fb.terminal} truncated={fb.truncated} "
        f"success={fb.success} reward={fb.reward:.4f}"
    )


def print_result(model: ExpertModel, x: DCScene, z: DCScene):
    print("\nResult (x -> z):")
    for label in sorted(model.entities.keys()):
        vx, vz = x.get(label), z.get(label)
        print(f"    {label:<22} {entity_str(label, vx)}  ->  {entity_str(label, vz)}")


def sample_task(scene: OGScene) -> tuple[DCScene, DCScene]:
    (s_scene, _), (g_scene, _) = scene.sample_task()
    return s_scene, g_scene


def subgoal_expectation(graph: Graph, i: int, model: ExpertModel, x: DCScene) -> str:
    """What the graph option would set: x -> assembled-subgoal per target."""
    _, subgoal = graph.select(i)
    parts = []
    for label in sorted(model.conditions.target_entities):
        parts.append(
            f"{label}: {entity_str(label, x.get(label))}"
            f" -> {entity_str(label, subgoal.get(label))}"
        )
    return "  ".join(parts)


def print_menu(graph: Graph, x: DCScene):
    print("\n" + "-" * 78)
    print("Options (graph, state-gated)")
    keys = graph.export_keys  # mirrors the options the RL graph can choose
    for i, key in enumerate(keys):
        node = graph.ns_option.get_by_key(key)
        model = ExpertModel.get(node.model)
        ok, blocked = precondition_status(model, x)
        flag = "VALID " if ok else "BAD"
        note = "" if ok else f" (Entities: [{', '.join(blocked)}])"
        exp = subgoal_expectation(graph, i, model, x)
        print(f"  [v{i}]/[r{i}]  {key:<44} | State={flag}{note}")
        if exp:
            print(f"          {exp}")
    print("  [0]   reset / sample a new episode")
    print("  [q]   quit")
    print("-" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser, default="scene1")
    add_use_gt_argument(parser)
    add_viewer_argument(parser)
    add_smode_argument(parser)
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    scene = Scene.get(scene_cfg, auto_load=False)
    assert isinstance(scene, OGScene), "Only OGScene supported."
    if args.viewer:
        scene.cfg.viewer = True

    # Load the agents (policy + conditions) exactly like Heca does, then build
    # the actual option graph.
    agent_cfgs = find_scene_models(args.scene)
    for agent_cfg in agent_cfgs:
        agent = ExpertModel.get(agent_cfg, auto_load=False)
        agent.use_gt(args.gt)
        agent.load()
    graph = Graph.generate(list(agent_cfgs), smode=args.smode)

    try:
        quitting = False
        ep_step = 0
        episode = 1
        x, y = sample_task(scene)
        graph.set_goal(y)
        graph.set_start(x)
        while True:
            print("\n" + "=" * 78)
            print(
                f"Episode {episode} Step {ep_step} "
                f"(use_gt={args.gt}, smode={args.smode.name})"
            )
            print("=" * 78)
            print_scene("obs.", x)
            print_scene("goal", y)
            graph.export()  # prune options to those valid for the current state
            print_menu(graph, x)
            ep_step += 1

            choice = input("> ").strip().lower()
            if choice in ("q", "quit", "exit"):
                quitting = True
                break
            if choice in ("0", "reset", ""):
                x, y = sample_task(scene)
                graph.set_goal(y)
                graph.set_start(x)
                ep_step = 0
                episode += 1
                continue

            kind, idx = choice[0], choice[1:]
            if kind not in ("v", "r") or not idx.isdigit():
                print(f"  unknown command: {choice!r}")
                continue
            i = int(idx)
            if i >= len(graph.export_keys):
                print(f"  no option with index {i}")
                continue
            model_cfg, subgoal = graph.select(i)
            model = ExpertModel.get(model_cfg)
            key = graph.export_keys[i]  # keep in sync with select(i)
            if kind == "v":
                z, fb = run_virtual_step(scene, model, x, subgoal, key)
            else:
                z, fb = run_real_rollout(model, x, subgoal, key)
            x = z  # chain: the env has actually moved
            graph.set_start(x)
            if fb.terminal or fb.truncated:
                print("\n  (episode ended by feedback; sampling a new task)")
                x, y = sample_task(scene)
                graph.set_goal(y)
                graph.set_start(x)
                episode += 1
                ep_step = 0
    except (EOFError, KeyboardInterrupt):
        quitting = True
        print("\nbye.")
    finally:
        scene.close()
        if quitting:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
