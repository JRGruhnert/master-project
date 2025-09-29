from hrl.common.agent import AgentConfig
from hrl.common.experiment import ExperimentConfig
from hrl.master_train import TrainConfig
from hrl.networks import NetworkType
from hrl.env.calvin import MasterEnvConfig

config = TrainConfig(
    tag="t2",
    experiment=ExperimentConfig(
        states_tag="Normal",
        skills_tag="Normal",
        p_empty=0.0,
        p_rand=0.0,
        env=MasterEnvConfig(),
        agent=AgentConfig(),
        nt=NetworkType.GNN_V4,
    ),
    use_wandb=True,
)
