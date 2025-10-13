from src.core.agent import HRLAgentConfig
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
    tag="test",
    nt=NetworkType.BASELINE_V1,
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    env=EnvironmentConfig(render=False),
    agent=HRLAgentConfig(
        # batch_size=64,
        # mini_batch_size=8,
    ),
    reward=RewardConfig(
        step_reward=-1.0,
        success_reward=100.0,
    ),
    storage=storage,
    use_wandb=use_wandb,  # Enable Weights & Biases logging
    device_tag=device_tag,  # Specify the device tag for logging
)
