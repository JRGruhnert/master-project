from conf.master.shared.train_env import env1
from conf.master.shared.agent import experiments
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    tag="experiment1_1",
    nt=NetworkType.GNN_V1,
    env=env1,
    agent=experiments,
)
