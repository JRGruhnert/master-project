import numpy as np
import torch

from heca.graphs.nodes.node import OptionNode
from heca.graphs.nodes.node_set import NodeSet


class OptionNodes(NodeSet[OptionNode]):

    @property
    def type(self) -> str:
        return "option"

    def build(self):
        effects = [node.effect for node in self.items]
        widths = [e.shape[-1] for e in effects if e is not None]
        width = widths[0] if widths else 0
        x_np = np.zeros((len(self.items), width), dtype=np.float32)
        for i, effect in enumerate(effects):
            if effect is not None and effect.shape[-1] == width:
                x_np[i] = effect
        self.x = torch.from_numpy(x_np).float()
