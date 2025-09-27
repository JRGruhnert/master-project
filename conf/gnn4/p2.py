from conf.master.shared.train_env import p2_gnn
from conf.master.shared.agent import experiments
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    tag="p2",
    nt=NetworkType.GNN_V4,
    env=p2_gnn,
    agent=experiments,
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Normal,
)
