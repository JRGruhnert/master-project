from dataclasses import dataclass

from tapas_gmm.master_project.agent import AgentConfig
from tapas_gmm.master_project.environment import MasterEnvConfig
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace


@dataclass
class ExperimentConfig:
    nt: NetworkType
    agent: AgentConfig
    env: MasterEnvConfig
    verbose: bool
    p_empty: float = 0.0  # Probability of empty skill
    p_rand: float = 0.0  # Probability of random skill


@dataclass
class Exp1Config(ExperimentConfig):
    nt: NetworkType  # Set later
    cross_t = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    t_spaces_mapping = {
        1: {"state_space": StateSpace.Minimal, "task_space": TaskSpace.Minimal},
        2: {"state_space": StateSpace.Normal, "task_space": TaskSpace.Normal},
        3: {"state_space": StateSpace.Normal, "task_space": TaskSpace.Minimal},
    }
    p_max: int = 100
    p_min: int = 0
    p_step: int = 10
    pe_all: bool = False  # If true ignore pe and pr and do all combinations
    pr_all: bool = False
    pe: float = 0.0
    pr: float = 0.0
    workers: int = 4
    verbose: bool = False
    agent: AgentConfig = AgentConfig()
    env: MasterEnvConfig = MasterEnvConfig(
        debug_vis=False,
    )
