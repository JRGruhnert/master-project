from conf.master.shared.train_env import env1
from conf.master.shared.agent import experiments
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    tag="alt",
    nt=NetworkType.GNN_V2,
    env=env1,
    agent=experiments,
)
