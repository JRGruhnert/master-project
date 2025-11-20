from src.networks.ppo_network import PPOConfig
from src.modules.rewards.reward import EvaluatorConfig, RewardMode
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from src.networks import NetworkType
from src.environments.environment import EnvironmentConfig

storage = StorageConfig(
    skills_tag="Normal",
    states_tag="Normal",
)
config = TrainConfig(
    tag="t2",
    nt=NetworkType.PPO_BASELINE,
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=EnvironmentConfig(render=False),
    agent=PPOConfig(),
    evaluator=EvaluatorConfig(
        step_reward=-0.01,
        success_reward=1.0,
        mode=RewardMode.SPARSE,
    ),
    storage=storage,
    use_wandb=False,
)
