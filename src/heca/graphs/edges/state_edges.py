import numpy as np
import torch

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import EntityNode, StateNode


class StateEdges(EdgeSet[EntityNode, StateNode]):
    @property
    def type(self) -> tuple[str, str, str]:
        return ("entity", "aggregation", "state")

    def update_attr(self, src: EntityNode, dst: StateNode, index: int):
        self.attrs[index] = np.empty(1)

    def set_index(self, src_indices: list[int], dst_indices: list[int]) -> None:
        """Set the edges from exported row indices (``Graph._append_state_rows``)."""
        self.edges = list(zip(src_indices, dst_indices))
        self.attrs = [np.empty(1) for _ in self.edges]
        self.edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        self.edge_attr = torch.empty((len(self.edges), 0), dtype=torch.float32)
