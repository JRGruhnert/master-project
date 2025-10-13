from conf.master.shared.agent import experiments
from tapas_gmm.master_project.environment import MasterEnvConfig
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from tapas_gmm.master_retrain import RetrainConfig
from conf.env.calvin.env_eval_conf import calvin_env_config

p_origin: int = 1
p_goal: int = 2
pe: int = 0  # percentage of empty positions during training
pr: int = 0  # percentage of random positions during training
nt: NetworkType = NetworkType.GNN_V4
# nt: NetworkType = NetworkType.BASELINE_V1  # --- IGNORE ---

re_env_gnn = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
)

re_env_baseline = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=False,
)

config = RetrainConfig(
    state_space=StateSpace.Normal,
    task_space=TaskSpace.Normal,
    verbose=True,
    tag=f"r{p_origin}{p_goal}",
    nt=NetworkType.GNN_V4,
    env=re_env_gnn,
    agent=experiments,
    checkpoint=f"results/{nt.value}/p{p_origin}_pe_{pe}_pr_{pr}/model_cp_best.pth",
    keep_epoch=False,  # Keep the epoch number in the checkpoint
)
