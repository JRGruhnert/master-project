from conf.master.shared.agent import experiments
from conf.master.shared.train_env import env2
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    tag="experiment2_1",
    nt=NetworkType.BASELINE_V2,
    env=env2,
    agent=experiments,
)
