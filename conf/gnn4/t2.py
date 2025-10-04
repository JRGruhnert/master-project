from hrl.common.agent import AgentConfig
from hrl.experiments.pepr import PePrConfig
from hrl.master_train import TrainConfig
from hrl.networks import NetworkType
from hrl.env.calvin import MasterEnvConfig

config = TrainConfig(
    tag="t2",
    experiment=PePrConfig(
        states_tag="Normal",
        skills_tag="Normal",
        p_empty=0.8,
        p_rand=0.0,
        env=MasterEnvConfig(),
        agent=AgentConfig(),
        nt=NetworkType.GNN_V4,
    ),
    use_wandb=True,
)
