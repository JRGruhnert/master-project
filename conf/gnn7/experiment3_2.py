from conf.master.shared.train_env import env
from conf.master.shared.agent import experiment2
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Normal,
    verbose=True,
    tag="normal_normal_large",
    nt=NetworkType.GNN_V7,
    env=env,
    agent=experiment2,
)
