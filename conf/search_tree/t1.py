from src.agents.search_tree import SearchTreeAgentConfig
from src.modules.rewards.reward import EvaluatorConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from src.networks import NetworkType
from src.environments.environment import EnvironmentConfig

storage = StorageConfig(
    used_skills="Minimal",
    used_states="Minimal",
)
config = TrainConfig(
    tag="t1",
    nt=NetworkType.SEARCH_TREE,
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=EnvironmentConfig(render=False),
    agent=SearchTreeAgentConfig(),
    evaluator=EvaluatorConfig(
        step_reward=-0.01,
        success_reward=1.0,
    ),
    storage=storage,
    use_wandb=False,
)
