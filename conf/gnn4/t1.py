from hrl.common.agent import AgentConfig
from hrl.common.reward import EvalConfig
from hrl.common.storage import StorageConfig
from hrl.experiments.pepr import PePrConfig
from hrl.master_train import TrainConfig
from hrl.networks import NetworkType
from hrl.env.environment import EnvironmentConfig

storage = StorageConfig(
    skills_tag="Minimal",
    states_tag="Minimal",
)
config = TrainConfig(
    tag="t1",
    nt=NetworkType.GNN_V4,
    experiment=PePrConfig(
        p_empty=0.8,
        p_rand=0.0,
    ),
    env=EnvironmentConfig(render=False),
    agent=AgentConfig(),
    eval=EvalConfig(
        step_reward=-1.0,
        success_reward=100.0,
    ),
    storage=storage,
    use_wandb=True,
)
