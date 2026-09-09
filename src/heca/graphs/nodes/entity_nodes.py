import numpy as np
import torch

from heca.graphs.nodes.node import EntityNode
from heca.graphs.nodes.node_set import NodeSet


class EntityNodes(NodeSet[EntityNode]):

    @property
    def type(self) -> str:
        return "entity"

    def build(self):
        x_np = np.stack([node.data.feature for node in self.items], axis=0)
        self.type_ids = torch.tensor(
            [node.type_id for node in self.items], dtype=torch.long  # type: ignore
        )

        self.x = torch.from_numpy(x_np).float()
