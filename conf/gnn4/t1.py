from conf.shared.train_env import env
from conf.shared.agent import agent
from hrl.common.experiment import ExperimentConfig
from hrl.master_train import TrainConfig


config = TrainConfig(
    tag="t1",
    experiment=ExperimentConfig(
        states_tag="Minimal",
        skills_tag="Minimial",
        p_empty=0.0,
        p_rand=0.0,
        env=env,
        agent=agent,
    ),
    use_wandb=True,
)
