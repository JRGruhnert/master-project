from src.core.agents.search_tree import SearchTreeAgentConfig
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
    tag="tree_small",
    nt=NetworkType.SEARCH_TREE,
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    env=EnvironmentConfig(render=False),
    agent=SearchTreeAgentConfig(),
    reward=RewardConfig(
        step_reward=-0.01,
        success_reward=1.0,
    ),
    storage=storage,
    use_wandb=False,
)
