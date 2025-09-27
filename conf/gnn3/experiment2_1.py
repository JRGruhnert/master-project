from conf.master.shared.train_env import env2
from conf.master.shared.agent import experiments
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    tag="experiment2_1",
    nt=NetworkType.GNN_V3,
    env=env2,
    agent=experiments,
)
