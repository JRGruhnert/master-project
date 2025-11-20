from tapas_gmm.master_eval import EvalConfig
from tapas_gmm.master_project.state import StateSpace
from tapas_gmm.master_project.task import TaskSpace
from conf.master.shared.train_env import eval_env

config = EvalConfig(
    task_space=TaskSpace.Normal,
    state_space=StateSpace.Normal,
    tag="test",
    env=eval_env,
)
