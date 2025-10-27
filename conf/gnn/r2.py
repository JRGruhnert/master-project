from conf.master.shared.train_env import re_env_gnn
from conf.master.shared.agent import experiments
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_retrain import RetrainConfig


config = RetrainConfig(
    state_space=StateSpace.Minimal,
    task_space=TaskSpace.Minimal,
    verbose=True,
    tag="r2",
    nt=NetworkType.GNN_V4,
    env=re_env_gnn,
    agent=experiments,
    checkpoint="results/gnn4/p2_pe_0.0_pr_0.0/model_cp_best.pth",
    keep_epoch=False,  # Keep the epoch number in the checkpoint
)
