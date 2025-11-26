from enum import Enum


class NetworkType(Enum):
    PPO_GNN = "gnn"
    PPO_BASELINE = "baseline"
    SEARCH_TREE = "search_tree"
    BASELINE_V2 = "baseline2"
    GNN_V1 = "gnn1"
    GNN_V2 = "gnn2"
    SKILLS = "skills"
    GNN_V5 = "gnn5"
    GNN_V6 = "gnn6"
    GNN_V7 = "gnn7"
    BASELINE_TEST = "test_baseline"
    GNN_TEST = "test_gnn"


def to_nt(network_str):
    return NetworkType[network_str.upper()]


def import_network(network_type):
    if network_type is NetworkType.GNN_TEST:
        from src.networks.gnn.gnn3 import Gnn as Network
    elif network_type is NetworkType.GNN_V1:
        from src.networks.gnn.gnn1 import Gnn as Network
    elif network_type is NetworkType.GNN_V2:
        from src.networks.gnn.gnn2 import Gnn as Network
    elif network_type is NetworkType.SKILLS:
        from src.networks.gnn.gnn3 import Gnn as Network
    elif network_type is NetworkType.PPO_GNN:
        from src.networks.gnn.gnn import Gnn as Network
    elif network_type is NetworkType.GNN_V5:
        from src.networks.gnn.gnn5 import Gnn as Network
    elif network_type is NetworkType.GNN_V6:
        from src.networks.gnn.gnn6 import Gnn as Network
    elif network_type is NetworkType.GNN_V7:
        from src.networks.gnn.gnn7 import Gnn as Network
    elif network_type is NetworkType.BASELINE_TEST:
        from src.networks.baseline.baseline import (
            Baseline as Network,
        )
    elif network_type is NetworkType.PPO_BASELINE:
        from src.networks.baseline.baseline import (
            Baseline as Network,
        )
    elif network_type is NetworkType.BASELINE_V2:
        from src.networks.baseline.baseline2 import (
            Baseline as Network,
        )
    else:
        raise ValueError(f"Invalid network {network_type.value}")
    return Network
