"""Temp: verify export raises when no option is feasible (no fallback)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import conf.experts.scene1 as s1
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.graphs.roles import ROLE_CURRENT, ROLE_GOAL, ROLE_POST, ROLE_PRE
from heca.data.data import DCEntity, DCScene

CFG = s1.faucet0_a_b
agent = ExpertModel.get(CFG, auto_load=False)
agent.load()
_ = agent.conditions

graph = Graph.generate([CFG], smode=SubgoalMode.BOTH)


def scene(graph, which):
    pair = agent.conditions
    entities = {}
    src = pair.pre.data_raw if which == "pre" else pair.post.data_raw
    for label, arr in src.items():
        v = np.asarray(arr[0], dtype=np.float64)
        entities[label] = DCEntity(value=v, feature=graph.entities[label].gnn_format(v))
    return DCScene(entities)


graph.set_goal(scene(graph, "post"))
graph.set_start(scene(graph, "pre"))

data = graph.export()
role = data["entity"].role_ids
counts = {n: int((role == r).sum()) for n, r in
          {"current": ROLE_CURRENT, "goal": ROLE_GOAL, "pre": ROLE_PRE, "post": ROLE_POST}.items()}
print("normal export: options:", len(graph.export_keys), "roles:", counts)
assert graph.feasible_keys() == list(graph.export_keys)
assert counts["pre"] + counts["post"] >= 1

# far-away scene: no option feasible -> export must raise
x = scene(graph, "pre")
far = {}
for label, dce in x.entities():
    v = np.asarray(dce.value, dtype=np.float64).copy()
    v[:3] += 100.0
    far[label] = DCEntity(value=v, feature=graph.entities[label].gnn_format(v))
graph.set_start(DCScene(far))
assert graph.feasible_keys() == []
try:
    graph.export()
    raise AssertionError("expected RuntimeError when no option is feasible")
except RuntimeError as e:
    print("raised as expected:", str(e)[:80], "...")
print("NO-FALLBACK TEST PASSED")
