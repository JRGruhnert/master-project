import numpy as np
import torch

from heca.graphs.nodes.node import OptionNode
from heca.graphs.nodes.node_set import NodeSet


class OptionNodes(NodeSet[OptionNode]):

    @property
    def type(self) -> str:
        return "option"

    def build(self):
        x_np = np.zeros((len(self.items), 128), dtype=np.float32)
        self.x = torch.from_numpy(x_np).float()
