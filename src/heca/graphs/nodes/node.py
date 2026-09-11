from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from heca.experts.expert import ExpertModel
from heca.data.condition import Condition
from heca.data.data import DCEntity, DCScene


class ValueMode(Enum):
    GOAL = "Goal"
    START = "Start"
    SAMPLE = "Sample"


@dataclass(slots=True, kw_only=True)
class GraphNode(ABC):
    data: DCEntity
    sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set[str]))

    def __str__(self) -> str:
        src_str = ", ".join(f"{self.sources}" if self.sources else "∅")
        return f"data={self.data} sources=[{src_str}]"


@dataclass(slots=True, kw_only=True)
class EntityNode(GraphNode):
    entity: str
    type_id: int
    data: DCEntity
    n_states: int

    def __str__(self) -> str:
        src_str = ", ".join(f"{self.sources}" if self.sources else "∅")
        return (
            f"EntityNode\n"
            f"  entity:     {self.entity}\n"
            f"  sources:    [{src_str}]\n"
            f"  data:       {self.data}\n"
            f"  n_states:   {self.n_states}\n"
        )


@dataclass(slots=True, kw_only=True)
class SubgoalNode(EntityNode):
    entity: str
    type_id: int
    data: DCEntity
    n_states: int


@dataclass(slots=True, kw_only=True)
class ValueNode(EntityNode):
    entity: str
    type_id: int
    data: DCEntity
    n_states: int
    #
    con: Condition
    vmode: ValueMode


@dataclass(slots=True, kw_only=True)
class CompNode(EntityNode):
    entity: str
    type_id: int
    data: DCEntity
    n_states: int
    #
    weight: float


@dataclass(slots=True, kw_only=True)
class CanonicalNode(EntityNode):
    """A per-entity current or goal value (the critic's side of the fork)."""

    entity: str
    type_id: int
    n_states: int
    #
    role: int = 0
    data: DCEntity = field(default_factory=DCEntity.empty)


@dataclass(slots=True, kw_only=True)
class StateNode(GraphNode):
    role: int = 0
    data: DCEntity = field(default_factory=DCEntity.empty)


@dataclass(slots=True, kw_only=True)
class OptionNode(GraphNode):
    model: ExpertModel.Config
    data: DCScene = DCScene.empty()
    effect: np.ndarray | None = None

    # OptionNode __str__:
    def __str__(self) -> str:
        src_str = ", ".join(f"{self.sources}" if self.sources else "∅")
        return (
            f"OptionNode" f"  model:      {self.model.tag}" f"  sources:    [{src_str}]"
        )
