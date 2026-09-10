# Per-step graph bookkeeping — where the wall clock went, and what changed

Status: implemented and verified (behavior-preserving), measured on scene0
(BOTH subgoal mode, `--gt --virtual`). Reproduce with

```bash
export PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl
python scripts/diag_profile_step.py --steps 60      # per-step wall clock + cProfile
python scripts/diag_verify_gate.py --steps 150      # equivalence vs the old code paths
```

---

## 1. Why training was slow (and why "the network variant didn't matter")

One option step in the RL loop is

```
data = graph.export()      # gate every option + rebuild the PyG batch
logits, value = network(data)
model, subgoal = graph.select(i)
z = expert.act(x, subgoal)
graph.set_start(z)         # update value nodes + rebuild all edge attrs
```

Only the second line is the network. Measured per option step (mean over 60
steps, this machine, current code):

| stage | before | after |
|---|---|---|
| `graph.export()` (incl. `feasible_keys`) | 3.87 ms | 1.19 ms |
| `set_start()` + `rebuild()` | 11.35 ms | 1.23 ms |
| `network.forward` (fiLM off) | 1.6 ms | 1.6 ms |
| `expert.act` | 1.1 ms | 1.1 ms |
| **total** | **~18 ms** | **5.2 ms** |

The two graph stages were ~85 % of the step, and neither scales with the network
variant: swapping `--network default/both/film` changes 1.6 ms of an 18 ms step
(≈9 %). That is the whole explanation for "it didn't matter which network
variant I used" — a 3 % end-to-end difference is far below run-to-run noise,
even before asking whether the added blocks help at all (see §5).

An update is `buffer capacity = 2048` option steps, so ~13 ms/step saved is
~26 s less per update ⇒ a 500-update run should drop from ~10 h to roughly
6–7 h. (The identical-code A/B below is the hard number; the extrapolation
assumes the rest of the step is unchanged.)

## 2. What was slow

`graph.export()` → `feasible_keys()` walked every option node and, per option:

1. `con.pre.models[label].get_parameters().copy()` — rebuilds the whole StepMix
   parameter dict (all components × all measurement dims) for **every (option,
   entity) pair on every step**, even though it is a constant of the fitted
   model;
2. `assemble_subgoal(node)` — copies the entire scene (`self.start.copy()`) to
   read one value per constrained entity;
3. `score_single(...)` — re-pads the categorical, re-inflates the covariances
   and re-evaluates two `scipy.stats` quantiles *inside* the test
   (`chi2.ppf` per call, `norm.ppf` per call, ~40 µs each).

`set_start()` → `rebuild()` → `es_condition.build()` ran the residual math
(quaternion log-map, gaussian χ², softmax cross-entropy) **one edge at a time in
Python**: ~140 edges × the full small-array numpy overhead.

## 3. What changed

* `EdgeSet.residual_batch(x_src[N,F], x_dst[N,F], n_states)` — the residual in one
  batched numpy pass, grouped by `n_states` (edges with different state
  vocabularies share the pass). `residual()` is now the one-row call into it.
* `EdgeSet.gather_features()` — slices the node set's already stacked feature
  tensor instead of re-stacking node features.
* `ConditionEdges.build()` — gathers src/dst features, per-edge `n_states` and
  component weights, then produces all `[N, 8]` attributes in one call.
* `Entity.prepare_single()` / `Entity.score_prepared()` — `secure_mix_parameters`
  (padding + covariance inflation) is **not idempotent**, so it is applied once
  per fitted model and the ready dict is then scored repeatedly.
* `Entity._best_component` — vectorized over components (`np.argmax` of the
  component log-posteriors) instead of a Python loop.
* cached `chi2.ppf` / `norm.ppf` quantiles (module-level `lru_cache`).
* `Graph._prepared_params` — `get_parameters()` + preparation cached per
  `(model, entity)` object pair (strong refs, so `id()` keys are safe).
* `Graph._gate_pass()` — per-step memo: the **pre**-gate depends only on the
  current scene and the agent's pre condition, so it is now evaluated once per
  (agent, step) instead of once per option; the **post**-gate is memoized on
  `(condition, entity, value bytes)`.
* `Graph._subgoal_plan()` — the entity → value-node mapping is resolved once per
  option (it is structural), so the scene copy disappears from the gate.

Speedups measured by `diag_verify_gate.py` (scene0, BOTH, 150 steps, same
process, legacy implementation re-implemented next to the new one):

| | legacy | new | speedup |
|---|---|---|---|
| `feasible_keys` | 3.09 ms/step | 0.61 ms/step | 5× |
| `es_condition.build` | 10.06 ms/step | 0.33 ms/step | 30× |

Same-process A/B of the whole affected region (60 steps each): export
3.87 → 1.19 ms, `set_start` 11.35 → 1.23 ms — **84 % less bookkeeping per step**.

## 4. Why it is safe (and how that was checked)

`scripts/diag_verify_gate.py` drives real episodes and compares, at every step,
the optimized result against the previous implementation:

* ordered list of feasible option keys — 0 set mismatches, 0 order mismatches;
* condition edge attributes — `max |diff| = 0.0` (bit-identical).

Checked for `smode ∈ {both, goal, chain}` and for `scene0/1/2` (150/80/50 steps).
`residual` vs `residual_batch` is also bit-identical per row (unit check).

Two legacy behaviors had to be preserved explicitly, both now documented in
code:

* `score_single` used `valid_pose and valid_state`, so the `pis` lookup was
  skipped whenever the pose test failed. `score_prepared` keeps that
  short-circuit (a naive rewrite raised `IndexError` on out-of-range states).
* `assemble_subgoal` iterates `option.sources["entity"]`, a **set**; if several
  value nodes ever fed the same entity, which one wins would depend on iteration
  order. That is now impossible by construction: `Graph._validate_structure`
  asserts, once per build, that every option feeds each entity from exactly one
  value node (and that every source key exists), so the gate can map
  entity → source key directly without any fallback path.

### Where the checks live

Input validation is not the network's job: `Network.forward` reads
`role_ids` / `cur_idx` / `goal_idx`, the state slot ids and the edge sets
unconditionally. The assumptions are asserted where the data is produced:

* `Graph._validate_structure()` — once per `Graph.generate`: option sources
  consistent, exactly two state slots (`current`, `goal`, in that order), and
  non-empty condition/summary edge sets.
* `Graph._validate_export()` — once per `export()`, on the exported batch:
  `cur_idx`/`goal_idx` same length and non-empty, `role_ids` aligned with the
  rows, state roles exactly `[current, goal]`, one aggregation edge per canonical
  row pointing at an existing slot, and at least one PRE and one POST row.
  Cost measured: **~56 µs per export** (~1 % of a step), and it does fire on a
  deliberately corrupted export.

## 5. What this does *not* explain

The gate/edge work was pure overhead removal; it does not change what the
network sees. The remaining per-step cost is now split roughly evenly between
`export` (1.2 ms), `set_start` (1.3 ms), the forward (1.8 ms) and the expert
(1.2 ms), so the next lever, if any, is the forward or the number of options —
not the graph plumbing.

For why the optional blocks (`use_option_interaction`, `use_timeline_memory`)
did not move the training curves, see
`scripts/diag_network_ablations.py`: it ablates the interaction block and the
memory carry on a trained checkpoint and reports the paired change in the
selected option.
