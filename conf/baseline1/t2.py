from hrl.common.agent import HRLAgentConfig
from hrl.common.modules.reward_modules import RewardConfig
from hrl.common.modules.storage_module import StorageConfig
from hrl.experiments.pepr import PePrConfig
from hrl.master_train import TrainConfig
from hrl.networks import NetworkType
from hrl.env.environment import EnvironmentConfig

storage = StorageConfig(
    skills_tag="Normal",
    states_tag="Normal",
)
config = TrainConfig(
    tag="t2",
    nt=NetworkType.BASELINE_V1,
    experiment=PePrConfig(
        p_empty=0.8,
        p_rand=0.0,
    ),
    env=EnvironmentConfig(render=False),
    agent=HRLAgentConfig(),
    reward=RewardConfig(
        step_reward=-1.0,
        success_reward=100.0,
    ),
    storage=storage,
    use_wandb=True,
)
