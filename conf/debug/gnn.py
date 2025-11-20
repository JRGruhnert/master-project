from conf.master.shared.train_env import env1
from conf.master.shared.agent import debug
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    tag="test",
    nt=NetworkType.GNN_TEST,
    env=env1,
    agent=debug,
)
