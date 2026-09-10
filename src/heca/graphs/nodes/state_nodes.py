import torch

from heca.graphs.nodes.node import StateNode
from heca.graphs.nodes.node_set import NodeSet


class StateNodes(NodeSet[StateNode]):
    @property
    def type(self) -> str:
        return "state"

    def build(self):
        self.x = torch.zeros((len(self.items), 1), dtype=torch.float32)
        self.type_ids = torch.tensor(
            [node.role for node in self.items], dtype=torch.long
        )
