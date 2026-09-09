import numpy as np

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import EntityNode


class TranslationEdges(EdgeSet[EntityNode, EntityNode]):

    @property
    def type(self) -> tuple[str, str, str]:
        return ("entity", "translation", "entity")

    def update_attr(self, src: EntityNode, dst: EntityNode, index: int):
        self.attrs[index] = np.empty(1)
