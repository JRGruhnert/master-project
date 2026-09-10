from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import EntityNode, OptionNode


class SummaryEdges(EdgeSet[EntityNode, OptionNode]):
    has_attrs: bool = False

    @property
    def type(self) -> tuple[str, str, str]:
        return ("entity", "summary", "option")
