from dataclasses import dataclass, field
from typing import Sequence
import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from heca.heca_gnn.modules.aggregation import StateAggregation
from heca.heca_gnn.modules.condition import ConditionBlock
from heca.heca_gnn.modules.encoder import (
    FreeEncoder,
    OptionEncoder,
    PrismaticEncoder,
    RevoluteEncoder,
    StaticEncoder,
    _EntityEncoder,
)
from heca.heca_gnn.modules.film import FiLMStack
from heca.heca_gnn.modules.interaction import OptionInteraction
from heca.heca_gnn.modules.readout import OptionReadout
from heca.heca_gnn.modules.state_critic import StateCritic
from heca.heca_gnn.modules.summary import SummaryBlock
from heca.heca_gnn.modules.timeline import TimelineMemory
from heca.heca_gnn.modules.translation import TranslationBlock
from heca.graphs.roles import ROLE_GOAL
from heca.data.entity import Entity
from heca.misc import hardware
from heca.misc.base import Configurable


class Network(Configurable, nn.Module):
    _last_option_x: torch.Tensor
    _last_mem: torch.Tensor

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        feature_dim: int = 256
        attn_heads: int = 4
        use_option_effects: bool = True
        use_option_interaction: bool = False
        use_timeline_memory: bool = False

    @property
    def condenser_names(self) -> tuple[str, ...]:
        """Conditioning inputs, in the order they modulate a site."""
        return ("goal", "memory") if self.cfg.use_timeline_memory else ("goal",)

    @property
    def encoder_map(self) -> dict[str, type[_EntityEncoder]]:
        return {
            "free": FreeEncoder,
            "static": StaticEncoder,
            "prismatic": PrismaticEncoder,
            "revolute": RevoluteEncoder,
        }

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        if tuple(self.encoder_map) != Entity.TYPE_NAMES:
            raise ValueError(
                f"encoder_map order {tuple(self.encoder_map)} does not match "
                f"Entity.TYPE_NAMES {Entity.TYPE_NAMES}"
            )
        self.entity_encoders = nn.ModuleDict(
            {name: cls(cfg.feature_dim) for name, cls in self.encoder_map.items()}
        )
        self.option_encoder = OptionEncoder(cfg.feature_dim)

        self.condition_layer = ConditionBlock(cfg.feature_dim)
        self.translation_layer = TranslationBlock(cfg.feature_dim)
        self.summary_layer = SummaryBlock(cfg.feature_dim)
        self.state_aggregation = StateAggregation(cfg.feature_dim)

        if cfg.use_option_interaction:
            self.interaction_layer = OptionInteraction(cfg.feature_dim, cfg.attn_heads)
        else:
            self.interaction_layer = None

        if cfg.use_timeline_memory:
            self.timeline_layer = TimelineMemory(cfg.feature_dim)
        else:
            self.timeline_layer = None

        self.films = FiLMStack(
            cfg.feature_dim, self.condenser_names, ("actor", "critic")
        )
        self.option_readout = OptionReadout(cfg.feature_dim)
        self.state_critic = StateCritic(cfg.feature_dim)

    def actor(self, data: HeteroData) -> torch.Tensor:
        logits, _ = self.forward(data)
        return logits

    def critic(self, data: HeteroData) -> torch.Tensor:
        _, value = self.forward(data)
        return value

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)

    def evaluate(
        self, data_list: Sequence[HeteroData], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logprobs = []
        state_values = []
        entropies = []

        for i, data in enumerate(data_list):
            logits, value = self.forward(data)
            dist = Categorical(logits=logits)

            action = actions[i : i + 1]
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            logprobs.append(logprob)
            state_values.append(value)
            entropies.append(entropy)

        return (
            torch.cat(logprobs).to(hardware.device),
            torch.cat(state_values).to(hardware.device),
            torch.cat(entropies).to(hardware.device),
        )

    def _memory_from(self, data: HeteroData, ref: torch.Tensor) -> torch.Tensor:
        step = getattr(data, "mem_step", None)
        if step is None or self.timeline_layer is None:
            return ref.new_zeros(1, self.cfg.feature_dim)
        return self.timeline_layer(*step)

    def _encode(self, node_type: str, data: HeteroData) -> torch.Tensor:
        """Per-type encoder pass over one node set's rows."""
        x = data[node_type].x
        type_ids = data[node_type].type_ids
        out = x.new_zeros(x.shape[0], self.cfg.feature_dim)
        for t, name in enumerate(self.encoder_map):
            rows = type_ids == t
            if rows.any():
                out[rows] = self.entity_encoders[name](x[rows])
        return out

    def _goal_slot(self, canonical_x: torch.Tensor, data: HeteroData) -> torch.Tensor:
        """Pooled goal slot over the canonical rows (see ``Graph.export``)."""
        roles = data["state"].type_ids
        pooled = self.state_aggregation(
            canonical_x,
            data[("canonical", "aggregation", "state")].edge_index,
            roles.shape[0],
        )
        return pooled[roles == ROLE_GOAL]

    def forward(
        self,
        data: HeteroData,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entity_x = self._encode("entity", data)
        stepmix = data[("entity", "condition", "entity")]
        entity_x = self.condition_layer(entity_x, stepmix.edge_index, stepmix.edge_attr)

        tapas_idx = data[("entity", "translation", "entity")].edge_index
        entity_x = self.translation_layer(entity_x, tapas_idx)

        canonical_x = self._encode("canonical", data)
        h_goal = self._goal_slot(canonical_x, data)

        effects = data["option"].x
        if not self.cfg.use_option_effects:
            effects = torch.zeros_like(effects)
        option_x = self.option_encoder(effects)
        option_x = self.summary_layer(
            entity_x, option_x, data[("entity", "summary", "option")].edge_index
        )

        if self.interaction_layer is not None:
            option_x = self.interaction_layer(option_x)

        self._last_option_x = option_x

        if self.timeline_layer is None:
            memory = None
        elif memory is None:
            memory = self._memory_from(data, option_x)
        self._last_mem = (
            memory
            if memory is not None
            else option_x.new_zeros(1, self.cfg.feature_dim)
        )

        conds = {"goal": h_goal}
        if memory is not None:
            conds["memory"] = memory

        logits = self.option_readout(option_x, self.films, conds)
        value = self.state_critic(
            canonical_x,
            data["canonical"].cur_idx,
            data["canonical"].goal_idx,
            self.films,
            conds,
        )

        return logits, value
