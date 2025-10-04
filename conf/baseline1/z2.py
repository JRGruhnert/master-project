from hrl.common.agent import AgentConfig
from hrl.common.reward import EvalConfig
from hrl.common.storage import StorageConfig
from hrl.experiments.pepr import PePrConfig
from hrl.master_train import TrainConfig
from hrl.networks import NetworkType
from hrl.env.environment import EnvironmentConfig

storage = StorageConfig(
    skills_tag="Normal",
    states_tag="Normal",
)
config = TrainConfig(
    tag="z2",
    nt=NetworkType.BASELINE_V1,
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    env=EnvironmentConfig(render=False),
    agent=AgentConfig(),
    eval=EvalConfig(
        step_reward=0.0,
        success_reward=100.0,
    ),
    storage=storage,
    use_wandb=True,
)
