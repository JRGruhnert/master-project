from src.core.agent import HRLAgentConfig
from src.core.modules.reward_module import RewardConfig
from src.core.modules.storage_module import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from src.core.networks import NetworkType
from src.core.environment import EnvironmentConfig

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
    agent=HRLAgentConfig(),
    reward=RewardConfig(
        step_reward=-1.0,
        success_reward=100.0,
    ),
    storage=storage,
    use_wandb=True,
    device_tag="gorilla",
)
