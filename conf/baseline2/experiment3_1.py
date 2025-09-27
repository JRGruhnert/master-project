from conf.master.shared.agent import experiments
from conf.master.shared.train_env import env3
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    tag="experiment3_1",
    nt=NetworkType.BASELINE_V2,
    env=env3,
    agent=experiments,
)
