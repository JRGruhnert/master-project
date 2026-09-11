from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from heca.experts.expert import ExpertModel
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.edges.state_edges import StateEdges
from heca.graphs.edges.summary_edges import SummaryEdges
from heca.graphs.edges.translation_edges import TranslationEdges
from heca.graphs.nodes.canonical_nodes import CanonicalNodes
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import *
from heca.graphs.nodes.option_nodes import OptionNodes
from heca.graphs.nodes.state_nodes import StateNodes
from heca.graphs.roles import ROLE_CURRENT, ROLE_GOAL, ROLE_OTHER, ROLE_POST, ROLE_PRE
from heca.misc import hardware, logger
from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.data.condition import Condition
from heca.data.pair import ConPair


class SubgoalMode(Enum):
    BOTH = "both"
    GOAL = "goal"
    CHAIN = "chain"

    def __str__(self):
        return self.value



class Graph:
    def __init__(self, entities: dict[str, Entity]):
        self.entities: dict[str, Entity] = entities

        self.ns_entity: EntityNodes = EntityNodes()
        self.ns_option: OptionNodes = OptionNodes()
        self.ns_canonical: CanonicalNodes = CanonicalNodes()
        self.ns_state: StateNodes = StateNodes()

        self.es_summary: SummaryEdges = SummaryEdges()
        self.es_condition: ConditionEdges = ConditionEdges()
        self.es_translation: TranslationEdges = TranslationEdges()
        self.es_state: StateEdges = StateEdges()

        self.ns_state.add("state_current", StateNode(role=ROLE_CURRENT))
        self.ns_state.add("state_goal", StateNode(role=ROLE_GOAL))

        for label, entity in self.entities.items():
            for role, tag in ((ROLE_CURRENT, "cur"), (ROLE_GOAL, "goal")):
                self.ns_canonical.add(
                    f"state_{tag}_{label}",
                    CanonicalNode(
                        entity=label,
                        type_id=entity.cfg.type_id,
                        n_states=entity.cfg.n_states,
                        role=role,
                    ),
                )

        self.start_keys: set[str] = set()
        self.goal_keys: set[str] = set()
        self.start: DCScene = DCScene.empty()
        self.goal: DCScene = DCScene.empty()

        self._pair_scores: dict[tuple[str, str], dict[str, float]] = {}
        self._agent_tags: list[str] = []

        self._option_conditions: dict[str, ConPair] = {}
        self._export_keys: list[str] | None = None
        self._start_set = False

        self._start_vals: dict[str, np.ndarray | None] | None = None
        self._pre_memo: dict[int, bool] = {}
        self._post_memo: dict[tuple, bool] = {}
        self._mix_cache: dict[tuple[int, int], tuple[object, dict]] = {}
        self._subgoal_plan_cache: dict[str, tuple[tuple[str, str | None], ...]] = {}

    def export(self) -> HeteroData:
        option_keys = self.feasible_keys()
        self._export_keys = option_keys
        ent_keys = self._entity_closure(option_keys)

        ent_old = [self.ns_entity.get_index(k) for k in ent_keys]
        opt_old = [self.ns_option.get_index(k) for k in option_keys]
        e_map = {old: i for i, old in enumerate(ent_old)}
        o_map = {old: i for i, old in enumerate(opt_old)}

        data = HeteroData()
        data[self.ns_entity.type].x = self.ns_entity.x[ent_old]
        data[self.ns_entity.type].type_ids = self.ns_entity.type_ids[ent_old]
        data[self.ns_option.type].x = self.ns_option.x[opt_old]

        disabled = [k for k in self.ns_option.keys if k not in option_keys]
        if disabled:
            logger.debug(
                f"gated out options ({len(disabled)}/{len(self.ns_option.keys)}): "
                f"{disabled}"
            )

        def _compact(
            es: EdgeSet,
            src_map: dict[int, int],
            dst_map: dict[int, int],
            want_attrs: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rows = [
                i for i, (s, d) in enumerate(es.edges) if s in src_map and d in dst_map
            ]
            if not rows:
                return (
                    torch.empty((2, 0), dtype=torch.long),
                    torch.empty((0, es.edge_attr.shape[1] if want_attrs else 0)),
                )
            idx = (
                torch.tensor(
                    [[src_map[es.edges[i][0]], dst_map[es.edges[i][1]]] for i in rows],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )
            if not want_attrs:
                return idx, torch.empty((0, 0))
            attrs = (
                es.edge_attr[rows].clone()
                if es.edge_attr.ndim == 2
                else torch.empty((0, 0))
            )
            return idx, attrs

        si, sa = _compact(self.es_condition, e_map, e_map)
        data[self.es_condition.type].edge_index = si
        data[self.es_condition.type].edge_attr = sa
        si, _ = _compact(self.es_summary, e_map, o_map, want_attrs=False)
        data[self.es_summary.type].edge_index = si
        si, _ = _compact(self.es_translation, e_map, e_map)
        data[self.es_translation.type].edge_index = si
        data = self._export_state_rows(data, ent_keys)
        self._validate_export(data)
        return data.to(device=hardware.device.type)

    def _validate_export(self, data: HeteroData) -> None:
        ent = data[self.ns_entity.type]
        can = data[self.ns_canonical.type]
        slots = data[self.ns_state.type]
        agg = data[self.es_state.type].edge_index
        counts = torch.bincount(ent.role_ids, minlength=ROLE_OTHER + 1)

        assert can.cur_idx.numel() == can.goal_idx.numel() > 0
        assert can.role_ids.shape[0] == can.x.shape[0] == can.type_ids.shape[0]
        # canonical rows never enter the entity set
        assert ent.role_ids.shape[0] == ent.x.shape[0]
        assert can.x.shape[0] == can.cur_idx.numel() + can.goal_idx.numel()
        assert can.role_ids[can.cur_idx].eq(ROLE_CURRENT).all()
        assert can.role_ids[can.goal_idx].eq(ROLE_GOAL).all()
        assert slots.type_ids.tolist() == [ROLE_CURRENT, ROLE_GOAL]
        assert agg.shape[1] == can.x.shape[0]
        assert int(agg[0].max()) < can.x.shape[0]
        assert int(agg[1].max()) < slots.type_ids.shape[0]
        assert bool((counts[[ROLE_PRE, ROLE_POST]] > 0).all())

    def _row_role(self, node: EntityNode) -> int:
        if isinstance(node, SubgoalNode):
            return ROLE_POST  # chain shared value == the target it drives to
        if isinstance(node, ValueNode):
            return ROLE_PRE if node.vmode == ValueMode.START else ROLE_POST
        return ROLE_OTHER  # e.g. CompNode

    def _state_labels(self) -> list[str]:
        if not self._start_set:
            return []
        start_keys = {k for k, _ in self.start.entities()}
        goal_keys = {k for k, _ in self.goal.entities()}
        return sorted(start_keys & goal_keys & set(self.entities))

    def _export_state_rows(self, data: HeteroData, ent_keys: list[str]) -> HeteroData:
        """Export the canonical rows and the aggregation edges.

        One current and one goal row per entity that appears in both the start
        and the goal, interleaved so that ``cur_idx[i]`` and ``goal_idx[i]`` are
        the same entity. They are not part of the entity node set: the relational
        layers are about conditions and translations between entities, and these
        rows are neither.
        """
        ent_type = self.ns_entity.type
        can_type = self.ns_canonical.type
        actor_roles = [self._row_role(self.ns_entity.get_by_key(k)) for k in ent_keys]
        data[ent_type].role_ids = torch.tensor(actor_roles, dtype=torch.long)

        cur_keys: list[str] = []
        goal_keys: list[str] = []
        for label in self._state_labels():
            cur, goal = f"state_cur_{label}", f"state_goal_{label}"
            if self.ns_canonical.has_key(cur) and self.ns_canonical.has_key(goal):
                cur_keys.append(cur)
                goal_keys.append(goal)

        can_keys: list[str] = []
        for cur, goal in zip(cur_keys, goal_keys):
            can_keys += [cur, goal]
        can_old = self.ns_canonical.get_indices(can_keys)
        c_map = {old: new for new, old in enumerate(can_old)}

        data[can_type].x = self.ns_canonical.x[can_old]
        data[can_type].type_ids = self.ns_canonical.type_ids[can_old]
        data[can_type].role_ids = torch.tensor(
            [self.ns_canonical.idx_get(i).role for i in can_old], dtype=torch.long
        )
        cur_rows = [c_map[self.ns_canonical.get_index(k)] for k in cur_keys]
        goal_rows = [c_map[self.ns_canonical.get_index(k)] for k in goal_keys]
        data[can_type].cur_idx = torch.tensor(cur_rows, dtype=torch.long)
        data[can_type].goal_idx = torch.tensor(goal_rows, dtype=torch.long)

        cur_state = self.ns_state.get_index("state_current")
        goal_state = self.ns_state.get_index("state_goal")
        agg_src = list(cur_rows) + list(goal_rows)
        agg_dst = [cur_state] * len(cur_rows) + [goal_state] * len(goal_rows)
        self.es_state.set_index(agg_src, agg_dst)
        data[self.es_state.type].edge_index = self.es_state.edge_index
        data[self.es_state.type].edge_attr = self.es_state.edge_attr
        data[self.ns_state.type].x = self.ns_state.x
        data[self.ns_state.type].type_ids = torch.tensor(
            [node.role for node in self.ns_state.items], dtype=torch.long
        )
        return data

    def _prepared_params(self, model, entity: Entity) -> dict:
        hit = self._mix_cache.get((id(model), id(entity)))
        if hit is not None and hit[0] is model:
            return hit[1]
        params = entity.prepare_single(model.get_parameters().copy())
        self._mix_cache[(id(model), id(entity))] = (model, params)
        return params

    def _start_values(self) -> dict[str, np.ndarray | None]:
        return {label: self._start_value(label) for label in self.entities}

    def _start_value(self, label: str) -> np.ndarray | None:
        try:
            return self.start.get(label).value
        except KeyError:
            return None

    @contextmanager
    def _gate_pass(self):
        """Per-step memo state shared by all options of one ``feasible_keys``."""
        self._pre_memo = {}
        self._post_memo = {}
        self._start_vals = self._start_values()
        try:
            yield
        finally:
            self._start_vals = None

    def _gate(self, con: Condition, values: dict[str, np.ndarray | None]) -> bool:
        """AND over ``con``'s entities of the per-entity state gate."""
        hit = self._pre_memo.get(id(con))
        if hit is not None:
            return hit
        ok = True
        for label in con.models:
            value = values[label]
            if value is None:
                ok = False
                break
            params = self._prepared_params(con.models[label], self.entities[label])
            if not self.entities[label].score_prepared(value, params):
                ok = False
                break
        self._pre_memo[id(con)] = ok
        return ok

    def _subgoal_plan(self, key: str) -> tuple[tuple[str, str | None], ...]:
        hit = self._subgoal_plan_cache.get(key)
        if hit is not None:
            return hit
        source_of = {
            self.ns_entity.get_by_key(skey).entity: skey
            for skey in self.ns_option.get_by_key(key).sources.get("entity", set())
        }
        plan = tuple(
            (label, source_of.get(label))
            for label in self._option_conditions[key].post.models
        )
        self._subgoal_plan_cache[key] = plan
        return plan

    def _gated(self, key: str) -> bool:
        """Gate check for one option during an active :meth:`_gate_pass`."""
        if not self._start_set:
            return True
        con = self._option_conditions[key]
        if not self._gate(con.pre, self._start_vals):
            return False
        for label, src_key in self._subgoal_plan(key):
            value = (
                self._start_vals[label]
                if src_key is None
                else self.ns_entity.get_by_key(src_key).data.value
            )
            if value is None:
                return False
            params = self._prepared_params(con.post.models[label], self.entities[label])
            memo_key = (id(con.post), label, np.asarray(value).tobytes())
            hit = self._post_memo.get(memo_key)
            if hit is None:
                hit = self.entities[label].score_prepared(value, params)
                self._post_memo[memo_key] = hit
            if not hit:
                return False
        return True

    def _feasible(self, key: str) -> bool:
        if self._start_vals is None:
            with self._gate_pass():
                return self._gated(key)
        return self._gated(key)

    def feasible_keys(self) -> list[str]:
        with self._gate_pass():
            values = [k for k in self.ns_option.keys if self._gated(k)]
        if not values:
            raise RuntimeError
        return values

    @property
    def export_keys(self) -> list[str]:
        if self._export_keys is None:
            return list(self.ns_option.keys)
        return list(self._export_keys)

    def _entity_closure(self, option_keys: list[str]) -> list[str]:
        keep: set[str] = set()
        stack: list[str] = []
        for key in option_keys:
            node = self.ns_option.get_by_key(key)
            stack.extend(node.sources.get("entity", ()))
        while stack:
            key = stack.pop()
            if key in keep or not self.ns_entity.has_key(key):
                continue
            keep.add(key)
            node = self.ns_entity.get_by_key(key)
            stack.extend(node.sources.get("entity", ()))
            stack.extend(node.sources.get("comp", ()))
        return [k for k in self.ns_entity.keys if k in keep]

    def set_start(self, start: DCScene):
        self.start = start.copy()
        self._start_set = True
        for key in self.start_keys:
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, ValueNode)
            self.ns_entity.key_update(key, self.start.get(node.entity))

        self.update_nodes()
        self.rebuild()

    def set_goal(self, goal: DCScene):
        self.goal = goal.copy()
        for node in self.ns_option.items:
            node.data = goal.copy()

    def update_nodes(self):
        for key, node in zip(self.ns_canonical.keys, self.ns_canonical.items):
            scene = self.start if node.role == ROLE_CURRENT else self.goal
            try:
                value = scene.get(node.entity)
            except KeyError:
                continue
            self.ns_canonical.key_update(key, value)

        for key in self.goal_keys:
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, ValueNode)
            if node.vmode == ValueMode.START:
                x = self.start.get(node.entity)
            elif node.vmode == ValueMode.GOAL:
                x = self.goal.get(node.entity)
            elif node.vmode == ValueMode.SAMPLE:
                x = node.con.sample(node.entity)
            else:
                if node.con.test(node.entity, self.goal):
                    x = self.goal.get(node.entity)
                else:
                    x = node.con.sample(node.entity)

            self.ns_entity.key_update(key, x)

    def assemble_subgoal(self, option: OptionNode) -> DCScene:
        subgoal = self.start.copy()
        for key in option.sources.get("entity", set()):
            node = self.ns_entity.get_by_key(key)
            assert isinstance(node, EntityNode)
            subgoal.set(node.entity, node.data.copy())
        return subgoal

    def __str__(self) -> str:
        lines = ["=== Graph ==="]
        lines.append(f"Entities: {len(self.entities)}")
        lines.append(str(self.ns_entity))
        lines.append(str(self.ns_option))
        lines.append(str(self.ns_canonical))
        lines.append(f"StepMix: {self.es_condition}")
        lines.append(f"Summary: {self.es_summary}")
        lines.append(f"Tapas:   {self.es_translation}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def rebuild(self):
        self.ns_entity.build()
        self.ns_option.build()
        self.ns_canonical.build()
        self.ns_state.build()
        self.es_condition.build(self.ns_entity, self.ns_entity)
        self.es_summary.build(self.ns_entity, self.ns_option)
        self.es_translation.build(self.ns_entity, self.ns_entity)

    def set_comps(self, tag: str, con: Condition) -> dict[str, set[str]]:
        keys: dict[str, set[str]] = defaultdict(set[str])
        for entity, comps in con.comp_features().items():
            for idx, (comp, weight) in enumerate(comps):
                key = con.label + entity + tag + f"{idx}"
                keys[entity].add(key)
                self.ns_entity.add(
                    key,
                    CompNode(
                        entity=entity,
                        type_id=self.entities[entity].cfg.type_id,
                        n_states=self.entities[entity].cfg.n_states,
                        data=DCEntity(value=np.empty(0), feature=comp),
                        weight=weight,
                    ),
                )
        return keys

    def set_precon(self, label: str, con: Condition) -> dict[str, str]:
        comp_sources = self.set_comps(label, con)
        pre_sources: dict[str, str] = {}
        for entity, sources in comp_sources.items():
            key = "pre_" + entity + label
            pre_sources[entity] = key
            self.start_keys.add(key)
            self.ns_entity.add(
                key=key,
                value=ValueNode(
                    entity=entity,
                    type_id=self.entities[entity].cfg.type_id,
                    n_states=self.entities[entity].cfg.n_states,
                    data=DCEntity.empty(),
                    sources={"comp": set(sources)},
                    con=con,
                    vmode=ValueMode.START,
                ),
            )
        return pre_sources

    def set_postcon(
        self,
        label: str,
        con: Condition,
        comp_sources: dict[str, set[str]],
        pre_sources: dict[str, str],
        entities: list[str],
        vmode: ValueMode,
    ) -> dict[str, str]:
        post_sources: dict[str, str] = {}
        for entity in entities:
            key = vmode.value + entity + label
            self.goal_keys.add(key)
            self.ns_entity.add(
                key,
                ValueNode(
                    entity=entity,
                    type_id=self.entities[entity].cfg.type_id,
                    n_states=self.entities[entity].cfg.n_states,
                    data=DCEntity.empty(),
                    sources={
                        "comp": set(comp_sources[entity]),
                        "entity": {pre_sources[entity]},
                    },
                    con=con,
                    vmode=vmode,
                ),
            )
            post_sources[entity] = key
        return post_sources

    def set_subgoal(
        self,
        label: str,
        comp_sources: dict[str, set[str]],
        pre_sources: dict[str, str],
        post_sources: dict[str, str],
        subgoal: dict[str, np.ndarray],
    ) -> set[str]:
        temp_sources = dict(post_sources)
        for entity, value in subgoal.items():
            key = "sub_" + entity + label
            sources = set(comp_sources[entity])
            sources.add(pre_sources[entity])
            feat = self.entities[entity].gnn_format(value)
            self.ns_entity.add(
                key,
                SubgoalNode(
                    entity=entity,
                    type_id=self.entities[entity].cfg.type_id,
                    n_states=self.entities[entity].cfg.n_states,
                    data=DCEntity(value=value, feature=feat),
                    sources={
                        "comp": set(comp_sources[entity]),
                        "entity": {pre_sources[entity]},
                    },
                ),
            )
            temp_sources[entity] = key
        return set(temp_sources.values())

    @staticmethod
    def _mean_component_feature(
        comps: list[tuple[np.ndarray, float]],
    ) -> np.ndarray | None:
        """Weighted mean of a condition's fitted component features."""
        if not comps:
            return None
        feats = np.stack([f for f, _ in comps]).astype(np.float64)
        weights = np.asarray([w for _, w in comps], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            return feats.mean(axis=0)
        return (feats * (weights / total)[:, None]).sum(axis=0)

    @classmethod
    def _option_effect(cls, pair: ConPair) -> np.ndarray:
        pre_feats = pair.pre.comp_features()
        post_feats = pair.post.comp_features()
        deltas = []
        for entity in sorted(set(pre_feats) & set(post_feats)):
            pre = cls._mean_component_feature(pre_feats[entity])
            post = cls._mean_component_feature(post_feats[entity])
            if pre is not None and post is not None:
                deltas.append(post - pre)
        if not deltas:
            return np.zeros(Entity.FEATURE_DIM, dtype=np.float32)
        return np.mean(np.stack(deltas), axis=0).astype(np.float32)

    @classmethod
    def generate(cls, cfgs: list[ExpertModel.Config], smode: SubgoalMode) -> "Graph":
        entities = {}
        for cfg in cfgs:
            entities.update(ExpertModel.get(cfg).entities)
        graph = cls(entities=entities)
        agents = [ExpertModel.get(cfg) for cfg in cfgs]
        graph._agent_tags = [a.cfg.tag for a in agents]

        for a in agents:
            ac = a.conditions
            effect = graph._option_effect(ac)
            pre_sources = graph.set_precon(ac.label, ac.pre)
            post_comp_sources = graph.set_comps(ac.label, ac.post)
            post_start_sources = graph.set_postcon(
                ac.label,
                ac.post,
                post_comp_sources,
                pre_sources,
                entities=ac.anchor_entities,
                vmode=ValueMode.START,
            )
            post_goal_sources = graph.set_postcon(
                ac.label,
                ac.post,
                post_comp_sources,
                pre_sources,
                entities=ac.target_entities,
                vmode=ValueMode.GOAL,
            )
            post_sources = post_start_sources | post_goal_sources
            use_sample_variant = smode in (SubgoalMode.GOAL, SubgoalMode.BOTH)
            if use_sample_variant:
                post_sample_sources = graph.set_postcon(
                    ac.label,
                    ac.post,
                    post_comp_sources,
                    pre_sources,
                    entities=ac.target_entities,
                    vmode=ValueMode.SAMPLE,
                )
                post_sources_alt = post_start_sources | post_sample_sources
            for b in agents:
                bc = b.conditions
                if ac.label == bc.label:
                    graph.ns_option.add(
                        ac.label,
                        OptionNode(
                            model=a.cfg,
                            sources={"entity": set(post_sources.values())},
                            effect=effect,
                        ),
                    )
                    graph._option_conditions[ac.label] = ac
                    if use_sample_variant:
                        graph.ns_option.add(
                            ac.label + "s",
                            OptionNode(
                                model=a.cfg,
                                sources={"entity": set(post_sources_alt.values())},
                                effect=effect,
                            ),
                        )
                        graph._option_conditions[ac.label + "s"] = ac
                if smode in (SubgoalMode.CHAIN, SubgoalMode.BOTH):
                    graph._pair_scores[(a.cfg.tag, b.cfg.tag)] = bc.pre.scores(ac.post)
                    subgoal = bc.pre.make_subgoal(ac.post)
                    if subgoal:
                        sources = graph.set_subgoal(
                            ac.label + "--" + bc.label,
                            post_comp_sources,
                            pre_sources,
                            post_sources,
                            subgoal,
                        )
                        graph.ns_option.add(
                            ac.label + "--" + bc.label,
                            OptionNode(
                                model=a.cfg,
                                sources={"entity": sources},
                                effect=effect,
                            ),
                        )
                        graph._option_conditions[ac.label + "--" + bc.label] = ac

        graph.es_condition.edges_from_sets(graph.ns_entity, graph.ns_entity, "comp")
        graph.es_summary.edges_from_sets(graph.ns_entity, graph.ns_option, "entity")
        graph.es_translation.edges_from_sets(graph.ns_entity, graph.ns_entity, "entity")
        graph._validate_structure()

        return graph

    def _validate_structure(self) -> None:
        for key in self.ns_option.keys:
            seen: set[str] = set()
            for skey in self.ns_option.get_by_key(key).sources.get("entity", set()):
                entity = self.ns_entity.get_by_key(skey).entity
                assert entity not in seen, f"{key}: two value nodes for {entity}"
                seen.add(entity)
        assert [node.role for node in self.ns_state.items] == [
            ROLE_CURRENT,
            ROLE_GOAL,
        ]
        assert self.es_condition.size > 0
        assert self.es_summary.size > 0

    def select(self, option: int) -> tuple[ExpertModel.Config, DCScene]:
        option_key = self.export_keys[option]
        index = self.ns_option.get_index(option_key)
        node = self.ns_option.idx_get(index)
        assert isinstance(node, OptionNode)
        subgoal = self.assemble_subgoal(node)
        logger.debug(f"Selected Option: {self.ns_option.key_at(index)}")
        return node.model, subgoal

    def plot(self, path: Path, figsize=(12, 8), show_labels=True):
        """Visualize the heterogeneous graph."""
        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)

        G = nx.MultiDiGraph()  # directed, allows multiple edges

        # Build key lookup: index → key (insertion order matches edge indices)
        entity_keys = self.ns_entity.keys
        option_keys = self.ns_option.keys

        # Add nodes with their type and a label
        for key in entity_keys:
            G.add_node(key, type="entity", label=key)
        for key in option_keys:
            G.add_node(key, type="option", label=key)

        # Add edges with their type (resolve positional indices → keys)
        for src, dst in self.es_condition.edges:
            G.add_edge(entity_keys[src], entity_keys[dst], type="stepmix")
        for src, dst in self.es_summary.edges:
            G.add_edge(entity_keys[src], option_keys[dst], type="summary")
        for src, dst in self.es_translation.edges:
            G.add_edge(entity_keys[src], entity_keys[dst], type="tapas")

        # Separate nodes by type for color coding
        entity_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "entity"]
        option_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "option"]

        # Position nodes (spring layout)
        # pos = nx.spring_layout(G, seed=42, k=2.0)
        shells = [option_nodes, entity_nodes]
        pos = nx.shell_layout(G, nlist=shells, scale=3.0)

        plt.figure(figsize=figsize)
        # Draw entity nodes (blue)
        nx.draw_networkx_nodes(
            G, pos, nodelist=entity_nodes, node_color="lightblue", node_size=800
        )
        # Draw option nodes (green)
        nx.draw_networkx_nodes(
            G, pos, nodelist=option_nodes, node_color="lightgreen", node_size=800
        )

        # Draw edges with different colors for each relation
        edge_colors = {"stepmix": "gray", "summary": "orange", "tapas": "red"}
        for etype, color in edge_colors.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == etype]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color,
                arrows=True,
                arrowsize=10,
                alpha=0.6,
            )

        # Labels (optional)
        if show_labels:
            labels = {n: d["label"] for n, d in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)

        # Legend
        legend_elements = [
            Patch(facecolor="lightblue", label="Entity"),
            Patch(facecolor="lightgreen", label="Option"),
            Patch(facecolor="gray", label="stepmix"),
            Patch(facecolor="orange", label="summary"),
            Patch(facecolor="red", label="tapas"),
        ]
        plt.legend(handles=legend_elements, loc="upper left")
        plt.title("Graph Structure")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(plot_path / f"graph.png", dpi=300, bbox_inches="tight")
        plt.close()

    def log(self):
        """Log graph statistics and key attributes."""
        logger.info("=== Graph Summary ===")
        logger.info(f"Entities: {len(self.entities)}")
        logger.info(f"Entity Nodes: {len(self.ns_entity.items)}")
        logger.info(f"Option Nodes: {len(self.ns_option.items)}")
        logger.info(f"StepMix Edges: {len(self.es_condition.edges)}")
        logger.info(f"Summary Edges: {len(self.es_summary.edges)}")
        logger.info(f"Tapas Edges: {len(self.es_translation.edges)}")

        # Optionally log node details
        entity_lines = []
        for key, idx in self.ns_entity.index.items():
            node = self.ns_entity.items[idx]
            entity_lines.append(f"{idx}:\t\t{node.entity}\t{key}")
        logger.debug(f"Entity Nodes:\n" + "\n".join(entity_lines))

        option_lines = []
        for key, idx in self.ns_option.index.items():
            node = self.ns_option.items[idx]
            option_lines.append(f"{idx}:\tagent={node.model.tag}\t\t{key}")
        logger.debug(f"Option Nodes:\n" + "\n".join(option_lines))

        stepmix_lines = []
        for src, dst in list(self.es_condition.edges):
            stepmix_lines.append(f"({src}->{dst})")
        logger.info("StepMix edges:\n" + ", ".join(stepmix_lines))

        tapas_lines = []
        for src, dst in list(self.es_translation.edges):
            tapas_lines.append(f"({src}->{dst})")
        logger.info("Tapas edges:\n" + ", ".join(tapas_lines))

        summary_lines = []
        for src, dst in list(self.es_summary.edges):
            summary_lines.append(f"({src}->{dst})")
        logger.info("Summary edges:\n" + ", ".join(summary_lines))

    def plot_connections(self, path: Path, figsize=(10, 8)):
        tags = self._agent_tags
        pair_scores = self._pair_scores
        if not tags:
            return

        n = len(tags)
        mat = np.full((n, n), np.nan)
        for (src, dst), entity_scores in pair_scores.items():
            i = tags.index(src)
            j = tags.index(dst)
            values = entity_scores.values()
            mat[i, j] = min(values) if values else np.nan

        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgray")
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tags, fontsize=8)
        ax.set_xlabel("pre-condition (target model)")
        ax.set_ylabel("post-condition (source model)")

        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(
                        j,
                        i,
                        "self",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
                elif not np.isnan(mat[i, j]):
                    val = float(mat[i, j])
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val < 0.5 else "black",
                    )
                else:
                    ax.text(
                        j, i, "—", ha="center", va="center", fontsize=7, color="black"
                    )

        fig.colorbar(im, ax=ax, label="min containment score")
        ax.set_title("Condition connections (min entity containment score)")
        fig.tight_layout()
        fig.savefig(plot_path / "connections.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
