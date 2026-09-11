import numpy as np
import torch

from heca.graphs.nodes.node import CanonicalNode
from heca.graphs.nodes.node_set import NodeSet


class CanonicalNodes(NodeSet[CanonicalNode]):

    @property
    def type(self) -> str:
        return "canonical"

    def build(self):
        filled = [node.data.feature for node in self.items if node.data.feature.size]
        width = int(filled[0].shape[-1]) if filled else 0
        x_np = np.stack(
            [
                (
                    node.data.feature
                    if node.data.feature.size
                    else np.zeros(width, dtype=np.float32)
                )
                for node in self.items
            ],
            axis=0,
        )
        self.x = torch.from_numpy(x_np).float()
        self.type_ids = torch.tensor(
            [node.type_id for node in self.items], dtype=torch.long  # type: ignore
        )
