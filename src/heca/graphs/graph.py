from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from heca.experts.expert import ExpertModel
from heca.graphs.node import *
from heca.graphs.node_set import NodeSet
from heca.graphs.edge_set import EdgeSet
from heca.misc import hardware, logger
from heca.data.data import DCScene
from heca.data.entity import Entity
from heca.conditions.condition import Condition
from heca.conditions.pair import ConPair


class SubgoalMode(Enum):
    NONE = "none"
    SIMPLE = "simple"
    CHAIN = "chain"
    BOTH = "both"

    def __str__(self):
        return self.value


class Graph:
    def __init__(self, entities: dict[str, Entity]):
        self.entities: dict[str, Entity] = entities
        self.ns_entity: NodeSet[EntityNode] = NodeSet[EntityNode]("entity")

        self.ns_option: NodeSet[OptionNode] = NodeSet[OptionNode]("option")

        self.es_summary: EdgeSet[EntityNode, OptionNode] = EdgeSet[
            EntityNode, OptionNode
        ](("entity", "summary", "option"))
        self.es_stepmix: EdgeSet[EntityNode, EntityNode] = EdgeSet[
            EntityNode, EntityNode
        ](("entity", "stepmix", "entity"))

        self.es_tapas: EdgeSet[EntityNode, EntityNode] = EdgeSet[
            EntityNode, EntityNode
        ](("entity", "tapas", "entity"))

        self.start_keys: set[str] = set()
        self.goal_keys: set[str] = set()
        self.start: DCScene = DCScene.empty()
        self.goal: DCScene = DCScene.empty()

        self._pair_scores: dict[tuple[str, str], dict[str, float]] = {}
        self._agent_tags: list[str] = []

        self._option_conditions: dict[str, ConPair] = {}
        self._export_keys: list[str] | None = None
        self._start_set = False

    def export(self) -> HeteroData:
        option_keys = self.enabled_keys()
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
            es: EdgeSet, src_map: dict[int, int], dst_map: dict[int, int]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rows = [
                i for i, (s, d) in enumerate(es.edges) if s in src_map and d in dst_map
            ]
            if rows:
                idx = (
                    torch.tensor(
                        [
                            [src_map[es.edges[i][0]], dst_map[es.edges[i][1]]]
                            for i in rows
                        ],
                        dtype=torch.long,
                    )
                    .t()
                    .contiguous()
                )
                attrs = (
                    es.edge_attr[rows]
                    if es.edge_attr.ndim == 2
                    else torch.empty((0, 0))
                )
            else:
                idx = torch.empty((2, 0), dtype=torch.long)
                attrs = torch.empty((0, es.edge_attr.shape[1]))
            return idx, attrs

        si, sa = _compact(self.es_stepmix, e_map, e_map)
        data[self.es_stepmix.type].edge_index = si
        data[self.es_stepmix.type].edge_attr = sa
        si, sa = _compact(self.es_summary, e_map, o_map)
        data[self.es_summary.type].edge_index = si
        data[self.es_summary.type].edge_attr = sa
        si, _ = _compact(self.es_tapas, e_map, e_map)
        data[self.es_tapas.type].edge_index = si
        return data.to(device=hardware.device.type)

    def _feasible(self, key: str) -> bool:
        node = self.ns_option.get_by_key(key)
        con = self._option_conditions.get(key)
        if con is None or not self._start_set:
            return True  # no gate info / no current scene -> never disable
        try:
            subgoal = self.assemble_subgoal(node)
        except KeyError:
            return False
        for label in con.pre.models:
            up = con.pre.models[label].get_parameters().copy()
            try:
                value = self.start.get(label).value
            except KeyError:
                return False
            if not self.entities[label].score_single(value, up):
                return False
        for label in con.post.models:
            up = con.post.models[label].get_parameters().copy()
            if not self.entities[label].score_single(subgoal.get(label).value, up):
                return False
        return True

    def enabled_keys(self) -> list[str]:
        keys = [k for k in self.ns_option.keys if self._feasible(k)]
        if not keys:
            logger.warning(
                "No option passes the state gate for the current scene; "
                f"exporting the ungated full set ({len(self.ns_option.keys)} "
                "options)."
            )
            return list(self.ns_option.keys)
        return keys

    @property
    def export_keys(self) -> list[str]:
        if self._export_keys is None:
            return list(self.ns_option.keys)
        return self._export_keys

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
            self.ns_entity.key_update(key, start[node.entity])

        self.update_nodes()
        self.rebuild()

    def set_goal(self, goal: DCScene):
        self.goal = goal.copy()
        for node in self.ns_option.items:
            node.data = goal.copy()

    def update_nodes(self):
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
            subgoal.set(node.entity, node.data)
        return subgoal

    def __str__(self) -> str:
        lines = ["=== Graph ==="]
        lines.append(f"Entities: {len(self.entities)}")
        lines.append(str(self.ns_entity))
        lines.append(str(self.ns_option))
        lines.append(f"StepMix: {self.es_stepmix}")
        lines.append(f"Summary: {self.es_summary}")
        lines.append(f"Tapas:   {self.es_tapas}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def rebuild(self):
        self.ns_entity.build()
        self.ns_option.build()
        self.es_stepmix.build(self.ns_entity, self.ns_entity)
        self.es_summary.build(self.ns_entity, self.ns_option)
        self.es_tapas.build(self.ns_entity, self.ns_entity)

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
            if smode == SubgoalMode.NONE:
                post_check_sources = graph.set_postcon(
                    ac.label,
                    ac.post,
                    post_comp_sources,
                    pre_sources,
                    entities=ac.target_entities,
                    vmode=ValueMode.CHECK,
                )
                post_sources = post_start_sources | post_check_sources
            else:  # SIMPLE / CHAIN / BOTH all use goal-pinned targets
                post_goal_sources = graph.set_postcon(
                    ac.label,
                    ac.post,
                    post_comp_sources,
                    pre_sources,
                    entities=ac.target_entities,
                    vmode=ValueMode.GOAL,
                )
                post_sources = post_start_sources | post_goal_sources
                use_sample_variant = smode in (SubgoalMode.SIMPLE, SubgoalMode.BOTH)
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
                use_chain = smode in (SubgoalMode.CHAIN, SubgoalMode.BOTH)

            for b in agents:
                bc = b.conditions
                if ac.label == bc.label:
                    graph.ns_option.add(
                        ac.label,
                        OptionNode(
                            model=a.cfg,
                            sources={"entity": set(post_sources.values())},
                        ),
                    )
                    graph._option_conditions[ac.label] = ac
                    if use_sample_variant:
                        graph.ns_option.add(
                            ac.label + "s",
                            OptionNode(
                                model=a.cfg,
                                sources={"entity": set(post_sources_alt.values())},
                            ),
                        )
                        graph._option_conditions[ac.label + "s"] = ac
                if use_chain:
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
                            ),
                        )
                        graph._option_conditions[ac.label + "--" + bc.label] = ac

        graph.es_stepmix.edges_from_sets(graph.ns_entity, graph.ns_entity, "comp")
        graph.es_summary.edges_from_sets(graph.ns_entity, graph.ns_option, "entity")
        graph.es_tapas.edges_from_sets(graph.ns_entity, graph.ns_entity, "entity")

        return graph

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
        for src, dst in self.es_stepmix.edges:
            G.add_edge(entity_keys[src], entity_keys[dst], type="stepmix")
        for src, dst in self.es_summary.edges:
            G.add_edge(entity_keys[src], option_keys[dst], type="summary")
        for src, dst in self.es_tapas.edges:
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
        logger.info(f"StepMix Edges: {len(self.es_stepmix.edges)}")
        logger.info(f"Summary Edges: {len(self.es_summary.edges)}")
        logger.info(f"Tapas Edges: {len(self.es_tapas.edges)}")

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
        for src, dst in list(self.es_stepmix.edges):
            stepmix_lines.append(f"({src}->{dst})")
        logger.info("StepMix edges:\n" + ", ".join(stepmix_lines))

        tapas_lines = []
        for src, dst in list(self.es_tapas.edges):
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
