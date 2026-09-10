# When the conditions block a task: two failure modes

Two *different* regimes were measured on the option gate, and they need different
fixes — do not treat them as one bug:

* **§1–5, scene4 / scene7 — empty gate (absorbing state).** A skill places the
  last movable entity (cube into the box, cube onto the shelf, lid onto the box),
  after which no option's pre-condition accepts the state and `feasible_keys()`
  raises. The start state is never like this (0/200 tasks); the values are far
  outside the fitted support (z = 27–149 vs cap 4.74), so re-fitting cannot help.
  The state is a *legitimate failure* — the goal is unreachable — so the episode
  should end as a failure (§8), not crash.
* **§6, scene9 — marginal support.** The gate is never empty, chains simply never
  reach the goal. The blocking rejections are 10–30 % outside the fitted
  acceptance radii (values still inside the training bounds): here re-fitting or
  narrower gating is exactly the right lever. §7 contrasts the two.

Status: investigation result, empirically measured. Reproduce the first mode with

```bash
PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl python scripts/diag_deadend_conditions.py \
    --scene scene4 --episodes 10 --tries 20 --max-deadends 6 \
    --out data/diag/deadend_cond_scene4.json     # same for scene7
```

Artifacts: `data/diag/deadend_cond_scene{4,7}.json`, `data/diag/repair_scene9_comp.json`,
`data/diag/guided_scene9_deep_seeded.json`, `data/diag/loose_gate_scene9.json`,
`data/diag/gate_ratio_1.{1,3}_scene9.json`, logs in `data/diag/logs/`,
older coarse scan in `data/diag/deadends_scene{4,7}.json`.

---

## 1. The rule that makes the gate empty

Every expert in these scenes is a **`base` → something** skill. Taking the level
label straight out of the demonstrations (`heca_<entity>_loc`, values
`base` / `box0` / `shelf0`):

| scene | expert | entity level start → end | box0 state start → end |
|---|---|---|---|
| 4 | `cube0_base_base` | base → base | 0 → 0 (and 1 → 1) |
| 4 | `cube0_base_box0` | base → **box0** | 0 → 0 |
| 4 | `cube0_base_shelf0` | base → **shelf0** | 0 → 0 / 1 → 1 |
| 4 | `lid0_base_base` | base → base | 0 → 0 |
| 4 | `lid0_base_box0` | base → **box0** | 0 → **1 (closed)** |
| 4 | `peg0_base_base` | base → base | 0 → 0 / 1 → 1 |
| 7 | `cube0_base_base`, `cube0_base_box0` | base → base / **box0** | 0 → 0 |
| 7 | `cube1_base_base`, `cube1_base_shelf0` | base → base / **shelf0** | 0 → 0 / 1 → 1 |
| 7 | `lid0_base_base`, `lid0_base_box0` | base → base / **box0** | 0 → 0 / 0 → **1** |

Not a single demonstration starts from `box0`, from `shelf0`, or with the box
closed. So the fitted **pre**-conditions only accept "my entity is at `base`
(and the box is open)", while the **post**-conditions happily put entities at
`box0` / `shelf0` / close the box. Consequence:

> The gate is non-empty **iff at least one option has all of its pre-entities at
> `base`**. A state where *every* movable entity is off its base level at the
> same time is absorbing → `feasible_keys()` raises.

Concretely, all 19 option keys collapse into a disjunction of three one-entity
predicates per scene (`pre=` lists taken from the graph's `ConPair`s; `shelf0` is
static and always passes, `box0` adds "box is open"):

```
scene4:  cube0@base  (8 options: cube0_base_{base,box0,shelf0}[s], 4 chains)
      ∨  lid0@base   (6 options: lid0_base_{base,box0}[s], 2 chains; box0 variants also need box open)
      ∨  peg0@base   (3 options: peg0_base_base[s], 1 chain)
scene7:  cube0@base  (6)  ∨  cube1@base  (6)  ∨  lid0@base  (6, box variants need box open)
```

A dead-end is exactly the state where **all three predicates are false at once**,
i.e. `cube0` placed **and** `lid0` on the box **and** the third movable entity
(`peg0` in scene4 / `cube1` in scene7) placed or displaced.

There is no inverse skill in either scene (no open-lid, no cube-out-of-box, no
cube-off-shelf), so nothing can ever leave that region.

## 2. Measured dead-end signature

6 dead-ends per scene, recorded at the first rollout step whose gate was empty.
Level signature is computed relative to the **episode's own** anchor poses:

* **scene4** — `lid0@on_box` 5/6 (+1 at `free/base`), `cube0@in_box` 2/6,
  `cube0@on_shelf` 4/6, `peg0` displaced 6/6.
* **scene7** — `lid0@on_box` **6/6**, `cube0@in_box` **6/6**,
  `cube1@on_shelf` **6/6**.

Example (normalized graph values, `x` = gate input):

```
scene4 ep0 step10   box0 ste=1.0        scene7 ep0 step12   box0 ste=0.0
  box0  [0.75, -1.0,  0.0] ste=1           box0  [0.75,  3.5, 0.0] ste=0
  lid0  [0.75, -0.94, 1.05]  (on box)      lid0  [0.74,  3.48,0.65] (on box)
  cube0 [0.73, -0.96, 0.26]  (in box)      cube0 [0.76,  3.5, 0.26] (in box)
  peg0  [0.01,  3.0, -0.14]  (moved)       cube1 [-1.1, -3.46,1.7 ] (on shelf)
```

Which entity fails, and how hard (worst pre-margin over the failing options):

| scene | pre-fail counts (6 dead-ends) | worst `z` (cap 4.74) | value inside fitted pre support? |
|---|---|---|---|
| 4 | `cube0` 60, `lid0` 36, `peg0` 18, `box0` 5 | `lid0` **26.9** (zd 23.9, cap 3.29), `cube0` 9.3 | **no** — every failing value is outside `con.pre.data_bounds` |
| 7 | `cube0` 42, `cube1` 36, `lid0` 36, `box0` 20 | `cube1` **149.4** | **no** — same |

No missing-key errors and no `assemble_subgoal` failures were recorded, i.e. this
is purely a condition-score rejection, never a bookkeeping bug.

## 3. Where the `RuntimeError` surfaces

`Graph.feasible_keys` (`src/heca/graphs/graph.py:248`) ends in a **bare
`raise RuntimeError`** — no message — as soon as no option passes. The training
path reaches it unguarded:

```
Heca.step (src/heca/agents/heca.py:65-68)
    graph.set_start(x)
    data = graph.export()      # -> feasible_keys() -> RuntimeError
    option = self.learner.predict(data, new_ep)
    a, s = self.graph.select(option)
```

So the very first step taken from an absorbing state kills the whole process
with an empty exception. Two consequences for the fix decision:

* a run with `--virtual` reaches an absorbing state via *random* chains in
  practically every episode (59/60 scene4, 60/60 scene7), so a crash is a matter
  of when, not if, in that mode;
* a real-env run reaches the *same level configuration* whenever the policy
  places the last movable entity (cube in box / cube on shelf / lid on box) —
  the absorbing state is not a virtual-mode artifact, only its frequency is.
  There it survives only if that state coincides with the goal, i.e. the run
  cannot ask for a further step afterwards;
* `Graph.select(i)` indexes the key list of the **last** `export()`
  (`src/heca/graphs/graph.py:571`), i.e. an index is only valid for the state it
  was exported from. Any caller that mixes indices across states silently picks a
  *different* option (or raises `IndexError` when the newer list is shorter).
  Worth an assertion or an explicit `select(key)` overload.

## 4. Intervention test — who blocks the gate
For each entity, restore *only that entity* to a sample of the base-condition of
an option that wants it, then re-run the gate:

| scene | restore `cube0` | restore `cube1` | restore `lid0` | restore `peg0` | restore `box0` |
|---|---|---|---|---|---|
| 4 | opens gate (6/6, via `cube0_base_base`) | – | opens gate (6/6, via `lid0_base_base`) | opens gate (6/6, via `peg0_base_base`) | no effect |
| 7 | opens gate (6/6) | opens gate (6/6) | opens gate (6/6) | – | no effect |

So in scene4 all three movable entities (`cube0`, `lid0`, `peg0`) block
simultaneously; in scene7 `cube0`, `cube1`, `lid0`. `box0` is never the blocker
(it is only ever an *anchor* of a pre-condition, and `lid0` alone is enough).

## 5. Do the conditions have to be refit?

**No — not for this failure.** Two independent reasons:

1. The failing values are *outside the fitted support* of the pre-conditions, by
   a wide margin (z up to 149 vs cap 4.74; all values outside `data_bounds`).
   These are not borderline misses that a better `n_components`/quantile would
   rescue; the states are a different region of state space (placed vs at base).
2. The pre-conditions are *correct*: they describe "not yet placed", which is
   exactly the region the expert policies were trained on and can act from.
   Making them accept "cube in a closed box" would gate in skills that have never
   seen that state — the policy would emit garbage, not recover.

The actual defect is **vocabulary closure**, not fit quality: the post-conditions
of the option set generate states that no pre-condition of the same option set
accepts. But "missing inverse skills" is the wrong conclusion to draw from that —
see §6: the absorbing state is a **legitimate failure state** (the goal is
unreachable), so what is missing is not skills that undo the placement but a
faithful way to end the episode.

Two side observations worth a decision later:

* **Virtual teleports are per-entity and mutually inconsistent.** Each target
  entity is sampled independently, so rollouts produce e.g. *lid on top of the
  box while `box0.ste == 0` (open)* — 4/6 scene4 and 2/6 scene7 dead-ends show
  exactly this decoupling (measured cross-tab `(box0.ste, lid0 level)`:
  scene4 `{(1,on_box):1, (0,on_box):4, (0,free):1}`,
  scene7 `{(0,on_box):2, (1,on_box):4}`). A real rollout can reach the same
  *level* configuration, so this is not the cause of the hole, but it does make
  random exploration hit it far more often than physics would.
* **Random chains drift into the absorbing corner fast** (scene7 dead-ends at
  step 3–24, scene4 at step 5–30), which is why episodes look "unsolvable" under
  random search even though a good chain exists.

**The start state is never the problem.** `scripts/diag_start_state.py` samples
200 tasks per scene and reports:

| scene | feasible options at start | tasks with empty gate | start levels | box0 at start |
|---|---|---|---|---|
| 4 | 13–16 (mean 15.2) | **0/200** | `lid0`, `peg0`, `cube0` all `free/base` 200/200 | open (0) 200/200 |
| 7 | 14–17 (mean 15.7) | **0/200** | `lid0`, `cube1`, `cube0` all `free/base` 200/200 | open (0) 200/200 |

So the task sampler never emits an already-placed configuration; every dead-end
is *created by the rollout* — an option's post-condition teleports an entity into
the box / onto the shelf (and, for the box, closes the lid), and once **all**
movable entities are placed nothing can be applied any more.

**How often does that happen?** It depends entirely on *who chooses* the options
(30 episodes per scene, 12-step budget):

| option selection | scene4 | scene7 |
|---|---|---|
| uniform random (`diag_deadends.py`, up to 60 chains/episode) | ≥1 chain hits an empty gate in 59/60 episodes | 60/60 episodes |
| goal-directed greedy, outcome-evaluated (`diag_targeted_repair.py`) | **0 empty gates**, 28/30 solved, 2 stuck | **0 empty gates**, 29/30 solved, 1 stuck |

So for a policy that acts toward the goal the empty gate is rare — but note the
"stuck" cases in the last row, they are a *second* way to end up unreachable
without an empty gate:

**Unreachable without an empty gate.** In all three stuck episodes the only entity
still differing from the goal was `cube0`, already placed (inside the box / on the
shelf) while the goal wants it elsewhere: all 10 options that target `cube0` fail
their pre-condition on `cube0` itself, yet `feasible_keys()` is *not* empty
because options touching `lid0` / `peg0` are still feasible. A rollout in that
state just burns its option budget applying irrelevant skills.

A sound unreachability check therefore has to answer *"can this entity ever be
changed?"* — and it must **not** be restricted to the currently feasible options.
An option that targets the missing entity may be gated off only because some
*other* entity is in the wrong state (a button in state 2 blocks a skill whose
demos started at state 0), and that other entity can be fixed in a later step.
Measured on scene9, the feasible-only test flags **92 of 200** episodes while
only **5** are actually unsolvable — it would kill 87 solvable episodes.

The correct test is a **changeability fixpoint** over *all* options
(reference implementation: `scripts/diag_changeability.py`,
`pre_label_status` / `changeable_closure`):

```
R := {}                                        # entities that can ever change
repeat until stable:
    option o is applicable  iff  every pre-entity label l of o is
                                 (passes the gate right now)  OR  (l ∈ R)
    if o is applicable:  R |= targets(o)
```

The current gate statuses only *seed* the iteration; options are never filtered
by feasibility. Adding to `R` only ever enables more options (monotone), and
generously assuming "a wrong entity can always be fixed" only *adds* entities, so
`R` over-approximates what is truly changeable. Hence:

> a still-required entity that is **not** in `R` can never be changed → the goal
> is unreachable (sound; may miss cases).

Validated on scene9 (200 reproducible episodes, `--env-seed 12345`):

| test | episodes flagged | known unsolvable | false alarms | misses |
|---|---|---|---|---|
| union of targets over **feasible** options | 92 | 5 (42, 103, 172, 179, 193) | 87 | 0 |
| **changeability fixpoint** over all options | **5** | same 5 | **0** | **0** |

**Caveat that matters as much as the rule:** "still required" must come from the
*environment's* success notion, never from a raw value difference. On scene4/7 the
raw diff flags `lid0` / `box0.ste` whenever the goal's value differs, yet the env
reports success anyway (e.g. a solved episode left `remaining={'box0': 1.0,
'lid0': 1.50}` and still returned reward 1.0) — the env ignores the lid once the
cube is where the task wants it. Feeding that raw diff into the fixpoint produced
12 false alarms out of 13 solved scene4 episodes. So: either expose a per-scene
hook for the success-relevant entities, or calibrate an "entity matters for
success" mask offline (perturb one entity of a successful goal state and see
whether success survives), and only then run the fixpoint on that mask.

## 6. The other failure mode: *marginal* gate rejections (scene9)

The empty gate above is one way the conditions block a task. The opposite regime
exists too, and scene9 is a clean example — it reproduces a rare, deterministic
failure that looks like "the oracle is just unlucky".

Reproducible setup (`--env-seed` makes the task stream identical across runs):

```bash
python scripts/diag_deep_guided.py --scene scene9 --episodes 200 --tries 1000 \
    --env-seed 12345 --restarts 6 --lookahead 3
# -> random_failures=5 (ep 42, 103, 172, 179, 193), guided_solved=0
```

All five fail with ``opts=32`` — the 32-step budget is exhausted, **no empty gate
is ever hit**, and a full teleport of every entity to the goal *is* a valid
environment success state (reward 1.0). So the tasks are unreachable for the
gated chain search, not ill-formed.

**Dose-response on the gate itself** (same episodes, same stream, 500 tries,
only the acceptance radii scaled):

| acceptance radius | unsolvable episodes (of those 5) |
|---|---|
| ×1.0 (current) | **5** |
| ×1.1 | 1 (ep 42) |
| ×1.3 | 0 |
| gate removed | 0 |

So those rejections are *marginal*: a 10–30 % loosening of the fitted
acceptance boundary is enough. Contrast the scene4/7 dead-ends, where the failing
values sit at z = 27–149 against a cap of 4.74 — a factor 6–30 outside, and
unreachable by any threshold tweak.

**Which condition rejects, concretely** (ep 42, initial state): only 30 of 95
options are feasible. `button1` starts in state 2 and the goal wants state 0; the
only option family that can start from state 2 is `button1_s2_s0*`, and

* its pre-condition on `button1` passes (z = 0.0, state OK, inside bounds),
* its pre-condition on **`faucet1`** — an entity the skill does *not* move —
  fails: `z = 5.69` vs cap `5.111` (11 % over), worst per-dim `2.93` vs cap
  `3.291` (passes), state logit OK, and the value is **inside** the fitted
  training bounds (`0/9` components accept, i.e. it is just outside the union of
  ellipsoids).

Hence no option can touch `button1`, the button can never move, and the chain
search burns the whole budget. The blocking relation is mutual: the options that
*can* touch `faucet1` require `button1` to be in state 0/1 in their own
pre-condition, so the order of fixes matters — the pre-conditions couple entities
that are physically independent.

**Verdict:** for this failure mode the user's hypothesis is right — the fitted
conditions are the lever. Two candidate fixes: refit (more data/components so the
ellipsoid union covers the task distribution), or stop gating on entities the
skill does not manipulate (gate the moved entity, plus its anchor in a relative
frame, instead of every entity the demos happened to observe).

## 7. Telling the two failure modes apart

| | scene4 / scene7 (absorbing state) | scene9 (marginal support) |
|---|---|---|
| symptom | `feasible_keys()` raises (empty gate) | budget exhausted, gate never empty |
| where it appears | *created by the rollout*, never at task start | at task start already |
| failing values | z 27–149 vs cap 4.7, outside `data_bounds` | z ≈ 1.0–1.2× cap, inside `data_bounds` |
| meaning | goal unreachable → legitimate failure | goal reachable, gate blocks the way |
| fix | end the episode as failure (terminal, reward 0) | refit / narrower gating; threshold tweak works |

Diagnostics: `scripts/diag_start_state.py` (is the start state already bad?),
`scripts/diag_deadend_conditions.py` (mode 1),
`scripts/diag_targeted_repair.py` + `scripts/diag_goal_reachability.py` (mode 2),
`scripts/diag_deep_guided.py --gate-ratio` (is the gate the blocker at all?).

## 8. What to do about it

The empty gate is not a bug in the conditions and not a missing skill: it is the
*only* correct verdict for a state from which the goal can no longer be reached.
An empty gate is in fact a **sound absorbing-state detector** within the option
abstraction — if no option's pre-condition accepts the state, no further change
can ever happen, so the remaining requirements can never be satisfied (the
success check runs first, via `fb.terminal`, so "goal already met" is not caught
here).

1. **Recommended: end the episode as a failure.** In `Heca.step` /
   `Heca.act`, catch the empty gate, and additionally run the changeability
   fixpoint from §2 to catch "unreachable with a non-empty gate" (see the caveat
   there: the required set must be the env's success notion):

   ```python
   try:
       keys = self.graph.feasible_keys()
   except RuntimeError:                     # nothing applicable -> absorbing
       return x, SceneFeedback(terminal=True, reward=0.0, truncated=False), True

   required = self.scene.success_relevant_diff(x, y)      # scene hook, NOT raw diff
   if required - graph.changeable_entities():              # fixpoint over ALL options
       return x, SceneFeedback(terminal=True, reward=0.0, truncated=False), True
   ```

   `terminal=True` (not `truncated=True`) is the semantically correct signal: the
   task is unreachable, so there is no future reward to bootstrap — truncation
   would tell the critic the opposite and corrupt the GAE target.
   `graph.changeable_entities()` is the fixpoint from §2, to be lifted into
   `src/heca/graphs/graph.py` from `scripts/diag_changeability.py`.
2. **Make the empty gate loud elsewhere.** It is the *success* signal for "this
   option set cannot solve this task", so it should also be counted (per scene)
   rather than swallowed; use it to measure how often the policy walks into
   absorbing states.
3. **Optional, exploration only: mask the trap.** If random/virtual exploration
   hits absorbing states too often (see the frequency numbers in §2 and the
   virtual teleport caveat above), a static closure check at graph-build time can
   drop chain options whose post can land in an absorbing level unless that post
   already satisfies the goal. This shapes exploration; it does not change what
   is reachable.
4. **Not recommended: adding inverse skills** (`lid0_box0_base`,
   `cube0_box0_base`, `cube1_shelf0_base`) as a reaction to this crash. Placing
   an object is a legitimate way to satisfy a goal, and an episode that ends
   there is a genuine failure, not a state to be undone. Add inverse skills only
   if the *task distribution itself* requires undoing placements.
