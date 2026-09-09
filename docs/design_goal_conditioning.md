# Goal Conditioning — Design Alternatives

Status: proposal for review (not implemented). Read this fresh; decide what fits.

Goal: the network should (a) know what the *goal* is, (b) represent each option by
what it *causes* (its pre→post effect), and (c) select options by whether their
effect moves the scene toward the goal — possibly over several steps. Trained
federated across scenes/tasks/objects, so **no design may hard-code a per-scene
entity/option vocabulary**; everything must be derived from shared type/role ids +
graph structure + condition parameters.

---

## 0. Why the current conditioning is "1-step target-vs-goal"

Current flow (what the network actually receives):

1. `export()` → entity rows `x = node.data.feature` (33-dim: 16 state logits +
   μ/lstd pos + quat + lstd rot + extras) with `type_ids` (free/prismatic/revolute/
   static). **No role id** (GOAL/START/SAMPLE/CHECK/pre/post/sub), **no entity
   identity** (cube0 vs drawer0) — only the value content and edges distinguish
   rows.
2. Option rows are **zeros** (`node_set.py`): `x_option = 0`. All option identity
   comes from summary edges.
3. Summary edge attr = `residual(src.data.feature, option.data[src.entity].feature)`
   — and `option.data` was overwritten by `set_goal()` to the **goal scene**.
   ⇒ the only goal signal is a hand-computed **distance from this option's target
   node to the goal value**. The readout learns `logit ≈ −‖target−goal‖`.
4. Critic = `mean` over option embeddings (`OptionReadout`), not a state value.

Consequences measured: `cube0_base_drawer0s` (the one option that makes +3.95
mismatch progress on a deep goal) scores −4.86, because its *target* is the
open-drawer site, 10σ from the deep goal. Options are indistinguishable when their
target nodes coincide (cos ≈ 1.00 clusters). Multi-step value is inexpressible.

---

## 1. Common problem decomposition (applies to all designs)

| # | Sub-problem | Current state | Needed |
|---|---|---|---|
| P1 | Option = effect, not snapshot | option emb = target−goal residual | goal-free "what does this option change" embedding `a_o` |
| P2 | Goal is known | goal only inside summary residual | explicit global goal repr `h_goal` + per-entity goal binding |
| P3 | Score = value, not distance | logit = −‖target−goal‖ | learn compatibility `f(a_o, h_state, h_goal)` trained by advantage (multi-step) |
| P4 | Roles / identity visible | not in input | small shared role-id vocab; per-graph entity binding (edges/pool) |

Federation constraints:
- Entity sets differ per scene → no global entity-id table; use `type_id` +
  `role_id` (small shared vocabs) + permutation-invariant pooling/edges.
- Option sets differ per scene → option "identity" must be *derived* from its
  graph neighborhood + condition features (never a fixed table).
- Per-graph export (one HeteroData per step) → per-graph indexing is safe.

---

## 2. Design A — "Roles + goal channel, keep structure-based options"

Smallest change that keeps the user's structural-differentiation idea, but fixes
what the structure encodes.

Idea: (1) give every entity row an explicit `role_id` (start/goal/post-goal/
post-sample/pre/comp/sub) so the net can tell "this row is cube0's GOAL value";
(2) pool goal rows into a global `h_goal` and inject it into every block (FiLM);
(3) keep summary edges, but change the summary edge attr to encode the option's
**own post−pre delta** (causal effect), *not* post−goal; (4) score options by a
learned readout of `[option_effect, h_state, h_goal]`.

```python
# ---------- graph/export changes (sketch) ----------
ROLE = {"start":0, "goal":1, "post_goal":2, "post_sample":3,
        "pre":4, "comp":5, "sub":6}
# ns_entity.add(...) stores role_id + entity label internally; export writes:
data["entity"].x        = x_rows            # unchanged, 33-dim
data["entity"].type_ids = type_ids          # unchanged
data["entity"].role_ids = role_ids          # NEW [N], shared vocab
# summary edge attr now = residual(src, dst_post_node) where dst_post_node is the
# option's OWN post/effect node (NOT goal). Option.data stays empty; each option
# points at its post rows via sources (already true).

# ---------- network changes (sketch) ----------
class Network(Configurable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Network.Config):
        role_embed_dim: int = 8              # NEW shared role vocab
        goal_mlp: bool = True                # NEW
        # ...
    def __init__(self, cfg):
        super().__init__(cfg)
        self.role_embedding = nn.Embedding(len(ROLE), cfg.role_embed_dim)
        # entity encoder input becomes input_feat_dim + type_embed + role_embed
        self.goal_encoder = GoalEncoder(cfg.feature_dim)   # NEW: pool goal rows
        # each block gets FiLM-conditioned on h_goal (sketch below)

    def forward(self, data):
        x = data["entity"].x
        type_ids = data["entity"].type_ids
        role_ids = data["entity"].role_ids
        role_emb = self.role_embedding(role_ids)            # [N, role_dim]
        # entity encoders take [x, type_emb, role_emb]
        entity_x = self.encode_entities(x, type_ids, role_emb)   # [N, d]

        h_goal = self.goal_encoder(entity_x, role_ids == ROLE["goal"])  # [1, d]
        # FiLM every block on h_goal:
        for layer in self.stepmix_layers:
            entity_x = layer(entity_x, idx, attr, cond=h_goal)
        for layer in self.tapas_layers:
            entity_x = layer(entity_x, idx, cond=h_goal)

        # summary: entity → option, edge attr = causal post−pre delta
        option_x = zeros(n_opt, d)
        option_x = self.summary_layer(entity_x, option_x, summary_idx, summary_attr)

        # readout: per-option compatibility with (state, goal)
        h_state = pooled_state(entity_x, role_ids != goal)   # [1, d]
        logits  = self.option_readout(option_x, h_state, h_goal)  # [1, n_opt]
        value   = self.critic(h_state, h_goal)               # [1]  (state value!)
        return logits, value
```

Pros: minimal graph surgery; role ids are shared across scenes; goal enters
globally (not per-option-distance); option still differentiated by structure.
Cons: FiLM conditioning adds params to every block; still no explicit
"compare option effect to goal" — the readout must learn it (fine, that's the job).

---

## 3. Design B — "Option action embedding from pre/post (goal-free) + goal attention"

Make option identity *explicitly* its causal effect, computed in feature space:
for each option, delta between its post-condition node rows and its pre/start rows
per touched entity, aggregated into `a_o`. Goal is encoded separately and combined
by attention in the readout. This is the design that most directly answers
"network understands causality from pre/post conditions".

```python
class EffectEncoder(nn.Module):
    """per-option: post rows vs pre/start rows of the SAME entity -> effect vec.
       Permutation invariant over the entities the option touches; no entity
       vocab needed (type_id + role deltas only)."""
    def __init__(self, dim):
        super().__init__()
        self.ent_fc = nn.Sequential(nn.Linear(2*dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.pool   = nn.Linear(dim, dim)

    def forward(self, entity_x, entity_type, role_ids,
                post_idx_per_option, pre_idx_per_option):
        outs = []
        for post_ents, pre_ents in zip(post_idx_per_option, pre_idx_per_option):
            # same-entity binding done in graph (pairs), fed as (post, pre) pairs
            d = entity_x[post_ents] - entity_x[pre_ents]     # causal delta
            e = self.ent_fc(torch.cat([entity_x[post_ents], d], -1)).mean(0)
            outs.append(e)
        return torch.stack(outs)   # [n_opt, dim]  a_o

class GoalReader(nn.Module):
    """pool GOAL-role rows (all entities) into h_goal; also per-entity goal tokens."""
    def forward(self, entity_x, role_ids, entity_group_ids):
        g = entity_x[role_ids == ROLE["goal"]]              # goal rows
        # per-entity binding: same entity's goal row should pair with its start row
        return g.mean(0, keepdim=True)                      # simple global first

class OptionReadoutV2(nn.Module):
    """logit_o = MLP([a_o, attn(a_o, goal rows)])  — compatibility learned."""
    def __init__(self, dim):
        super().__init__()
        self.goal_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2*dim), nn.Linear(2*dim, 1))

    def forward(self, a_o, goal_rows):
        # a_o: [n_opt, d] as queries over goal entity rows [n_goal, d]
        ctx, _ = self.goal_attn(a_o.unsqueeze(0), goal_rows.unsqueeze(0),
                                goal_rows.unsqueeze(0))
        ctx = ctx.squeeze(0)
        logits = self.head(torch.cat([a_o, ctx], -1)).view(1, -1)
        return logits
```

Key properties: `a_o` is goal-free — it answers "what does option o do?" Goal
affects only the *scoring* step, so the net can learn "placing cube in open drawer
is valuable when goal = cube deep in closed drawer" (that link requires value
training, see §5). Works across scenes: deltas are type-relative; goal rows exist
in every scene.

Pros: explicit causality; options comparable across scenes; value learning has a
stable action input.
Cons: needs the graph to expose (post, pre) entity pairs per option (it has them
via `sources`); a bit more plumbing in `export()`.

---

## 4. Design C — "Value-of-outcome scoring (option-critic style)"

Make the *score of an option* equal to the predicted value of the state *after*
executing it — the natural multi-step objective. The net predicts
`V(s'_o | g)` where `s'_o` is the state with the option's target entities set to
their post values. Training uses the PPO critic + reward shaping; intermediate
states (cube in open drawer) then get real value.

```python
class OutcomeScorer(nn.Module):
    """score_o = V(predicted next state after option o | goal)  [learned]"""
    def __init__(self, dim, state_encoder):
        super().__init__()
        self.state_encoder = state_encoder     # shared goal-conditioned encoder
        self.value_head    = nn.Linear(dim, 1)

    def forward(self, entity_x, role_ids, group_ids, option_post_rows, h_goal):
        values = []
        for post_rows in option_post_rows:     # rows option o would set
            # s'_o: start rows, but target entities replaced by their post rows
            s_prime = self.apply_post(entity_x, role_ids, group_ids, post_rows)
            v = self.value_head(self.state_encoder(s_prime, h_goal))
            values.append(v)
        return torch.cat(values).view(1, -1)   # option logits = V(s'_o|g)
```

Implementation note: "apply post" needs per-entity binding (group ids / edges).
Simpler first version: score by encoding the option's post rows pooled + start
rows pooled → `V(f(post_pool, start_pool, h_goal))`. This is an approximation of
V(s') but cheap and goal-conditioned.

Pros: the score *is* a value → multi-step is expressible by construction; deep
tasks become learnable as soon as V(intermediate) is learned (needs shaping, §5).
Cons: heavier; needs the state encoder to be good; option post rows must be
identifiable (role ids).

---

## 5. Companion changes (needed regardless of A/B/C)

1. **Reward shaping on mismatch reduction** (per option step):
   `r += α · (dist(x, goal) − dist(x', goal))`. Without it, the cube step gets
   credit only if the whole chain later succeeds → cold start. Normalize per
   entity (buttons vs position scales).
2. **Fix the critic**: value = V(state|goal), not mean over option embeddings.
3. **Execute chains as sequences** (currently a chain option carries only the
   first skill's model, so "multi-step in one action" is not really available).
   Either drop chains from the option set, or execute them as true 2-step skills
   so the graph's causality is honest.
4. Optionally a **self-supervised auxiliary**: predict from `a_o` which entity
   types/roles the option changes ("effect prediction") to stabilize `a_o`.

---

## 6. Decision matrix

| | A: roles+goal channel | B: effect emb + goal attn | C: value-of-outcome |
|---|---|---|---|
| Goal known globally | yes (FiLM) | yes (attn/readout) | yes (in V) |
| Option = causal effect | partial (edge delta) | explicit `a_o` | via outcome state |
| Multi-step expressible | via learned readout + value | via learned readout + value | by construction |
| Graph plumbing | role ids + edge-attr change | (post,pre) pairs per option | post rows per option |
| Scene/object general | yes | yes | yes |
| Param cost | low | low–med | med–high |
| My pick for "network understands pre/post causality" | good baseline | **best fit** | best *if* reward shaping done |

Suggested path: implement P1–P4 data-side (role ids, goal rows, causal edge
attrs, post-row pairs) once — all three designs consume the same exports — then
prototype **B** readout + **§5.1/5.2**, measure deep-task logit of
`cube0_base_drawer0s` on a deep goal and the deep-task success rate.

---

## Addendum (implemented): selectable residual rulers

The exploding-residual fix and both relative rulers are implemented behind a
flag, defaulting to the safe constant floor:

- `ResidualMode.CONSTANT` (default): fixed `LSTD_FLOOR = -2` log-std floor +
  ±10 clip (the bug fix).
- `ResidualMode.ENTITY`: summary-edge z divided by the *entity's* median fitted
  σ across all its comp nodes in the graph (point-distance in "how much this
  entity usually moves" units). Falls back to constant when no comps exist.
- `ResidualMode.POST`: z divided by the σ of the *single best-matching
  component* of the option's post-condition for the goal value — the same
  best-component rule `score_single`/`_best_component` uses for gating. This
  is DRY: the component selection is delegated to
  `Entity.best_component_cov` (a thin wrapper over `secure_mix_parameters` +
  `_best_component`), and `edge_set` only reads the returned covariance to
  build the [3 pos, 3 rot] ruler. Falls back to the constant floor when the
  source node carries no Condition (chain sub-nodes) or no goal is set.
  Note: anchors are scaled by the option's *post* comps too (set_postcon
  attaches post comps to all its value nodes); if you ever want anchors scaled
  by the pre σ instead, that distinction is not representable today.

Plumbing: `--residual-mode {constant,entity,post}` (scripts/common/args.py →
Heca.Config.residual_mode → Graph.generate(residual_mode=...) → EdgeSet
residual_mode → _summary_ruler). test_graph.py exposes the same flag. Tag suffix
`c`/`e`/`p` appended so runs with different rulers get separate checkpoint dirs.
Note: old dirs without the suffix (e.g. `..._gv_b`) predate this; retrain anyway
(the old checkpoints were trained on the exploding values).
