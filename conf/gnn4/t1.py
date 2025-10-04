from hrl.common.agent import AgentConfig
from hrl.experiments.pepr import PePrConfig
from hrl.master_train import TrainConfig
from hrl.networks import NetworkType
from hrl.env.calvin import MasterEnvConfig

config = TrainConfig(
    tag="t1",
    experiment=PePrConfig(
        states_tag="Minimal",
        skills_tag="Minimal",
        p_empty=0.0,
        p_rand=0.0,
        env=MasterEnvConfig(
            debug_vis=False,
        ),
        agent=AgentConfig(),
        nt=NetworkType.GNN_V4,
    ),
    use_wandb=False,
)
