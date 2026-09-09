from dataclasses import dataclass
from typing import Sequence
import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from heca.heca_gnn.layers import StepMixBlock, TapasBlock, SummaryBlock
from heca.misc import hardware
from heca.misc.base import Configurable


class TimelineMemory(nn.Module):
    def __init__(self, dim: int, extra_in: int = 0):
        super().__init__()
        self.gru = nn.GRUCell(dim + extra_in, dim)

    def forward(self, u: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = u.new_zeros(u.shape[0], self.gru.hidden_size)
        return self.gru(u, h)


class OptionInteraction(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.unsqueeze(0)
        attended, _ = self.attn(xs, xs, xs)
        return self.norm(xs.squeeze(0) + attended.squeeze(0))


class OptionReadout(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_dim = max(int(dim * hidden_ratio), 16)

        self.shared = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(hidden_dim, 1)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)

        actor_out = self.actor_head(shared)
        logits = actor_out.view(1, -1)

        pooled = shared.mean(dim=0, keepdim=True)
        value = self.critic_head(pooled).squeeze(-1)

        return logits, value


class Network(Configurable, nn.Module):
    _last_option_x: torch.Tensor
    _last_mem: torch.Tensor

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        type_embed_dim: int = 8
        mobility_feat_dim: int = 3
        feature_dim: int = 256
        input_feat_dim: int = 33
        num_stepmix_layers: int = 1
        num_tapas_layers: int = 1
        encoder_depth: int = 3
        gnn_mlp_depth: int = 3
        attn_heads: int = 4
        readout_hidden_ratio: float = 0.5
        use_option_interaction: bool = False
        use_timeline_memory: bool = False

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        # Maps TYPE_ID → encoder name
        self._type_names = ["free", "static", "prismatic", "revolute"]

        self.type_embedding = nn.Embedding(len(self._type_names), cfg.type_embed_dim)

        encoder_in = cfg.input_feat_dim + cfg.type_embed_dim
        self.entity_encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(encoder_in),
                    nn.Linear(encoder_in, cfg.feature_dim),
                    nn.LayerNorm(cfg.feature_dim),
                    nn.ReLU(),
                )
                for name in self._type_names
            }
        )
        self.stepmix_layers = nn.ModuleList(
            [
                StepMixBlock(cfg.feature_dim, cfg.gnn_mlp_depth)
                for _ in range(cfg.num_stepmix_layers)
            ]
        )

        self.tapas_layers = nn.ModuleList(
            [
                TapasBlock(cfg.feature_dim, cfg.gnn_mlp_depth)
                for _ in range(cfg.num_tapas_layers)
            ]
        )

        self.summary_layer = SummaryBlock(cfg.feature_dim, cfg.gnn_mlp_depth)

        if cfg.use_option_interaction:
            self.interaction_layer = OptionInteraction(cfg.feature_dim, cfg.attn_heads)
        else:
            self.interaction_layer = None

        if cfg.use_timeline_memory:
            self.timeline = TimelineMemory(cfg.feature_dim)
            # The readout consumes [option embedding, memory] concatenated.
            self.option_readout = OptionReadout(
                2 * cfg.feature_dim, cfg.readout_hidden_ratio
            )
        else:
            self.timeline = None
            self.option_readout = OptionReadout(
                cfg.feature_dim, cfg.readout_hidden_ratio
            )

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
        if step is not None and self.timeline is not None:
            u_prev, h_prev = step
            u_prev = u_prev.clone()
            h_prev = h_prev.clone()
            return self.timeline(u_prev, h_prev)
        return ref.new_zeros(1, self.cfg.feature_dim)

    def forward(
        self,
        data: HeteroData,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one transition.

        With the timeline memory enabled, the memory used at this step is
        either recomputed from the transition's stored ``data.mem_step``
        (rollout path, ``memory=None``) or taken from ``memory`` — the state
        threaded in by a training-time unroll (truncated BPTT). Without the
        timeline, ``memory`` is ignored.
        """

        x = data["entity"].x
        type_ids = data["entity"].type_ids
        type_embeds = self.type_embedding(type_ids)

        # Route each row through its type's encoder
        N = x.shape[0]
        entity_x = torch.zeros(N, self.cfg.feature_dim, device=x.device, dtype=x.dtype)
        for t, name in enumerate(self._type_names):
            mask = type_ids == t
            if mask.any():
                inp = torch.cat([x[mask], type_embeds[mask]], dim=-1)
                entity_x[mask] = self.entity_encoders[name](inp)

        stepmix_idx = data[("entity", "stepmix", "entity")].edge_index
        stepmix_attr = data[("entity", "stepmix", "entity")].edge_attr
        for layer in self.stepmix_layers:
            entity_x = layer(entity_x, stepmix_idx, stepmix_attr)

        tapas_idx = data[("entity", "tapas", "entity")].edge_index
        for layer in self.tapas_layers:
            entity_x = layer(entity_x, tapas_idx)

        summary_idx = data[("entity", "summary", "option")].edge_index
        summary_attr = data[("entity", "summary", "option")].edge_attr
        option_x = data["option"].x
        if option_x.shape[-1] != self.cfg.feature_dim:
            option_x = option_x.new_zeros(option_x.shape[0], self.cfg.feature_dim)
        option_x = self.summary_layer(entity_x, option_x, summary_idx, summary_attr)

        if self.interaction_layer is not None:
            option_x = self.interaction_layer(option_x)

        # Expose the per-option embeddings and the memory used at this step to
        # the rollout loop. Set unconditionally (zeros when the timeline is
        # disabled and therefore unused) so callers can read them directly.
        self._last_option_x = option_x
        if self.timeline is not None:
            if memory is None:
                memory = self._memory_from(data, option_x)
            self._last_mem = memory
            option_x = torch.cat(
                [option_x, memory.expand(option_x.shape[0], -1)], dim=-1
            )
        else:
            self._last_mem = option_x.new_zeros(1, option_x.shape[1])

        logits, value = self.option_readout(option_x)

        return logits, value
