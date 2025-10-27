from src.core.agents.agent import HRLAgentConfig
from src.core.modules.reward_module import RewardConfig
from src.core.modules.storage_module import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from src.networks import NetworkType
from src.core.environment import EnvironmentConfig
from conf.machine import use_wandb, device_tag

storage = StorageConfig(
    skills_tag="Normal",
    states_tag="Normal",
)
config = TrainConfig(
    tag="t2",
    nt=NetworkType.GNN_V4,
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
    use_wandb=use_wandb,
    device_tag=device_tag,
)
