from conf.master.shared.train_env import env
from conf.master.shared.agent import experiments
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_train import TrainConfig


config = TrainConfig(
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Minimal,
    verbose=True,
    tag="normal_min",
    nt=NetworkType.GNN_V7,
    env=env,
    agent=experiments,
)
