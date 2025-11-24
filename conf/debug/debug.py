from src.modules.rewards.reward import EvaluatorConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.debug import DebugConfig
from src.networks import NetworkType
from src.environments.environment import EnvironmentConfig

storage = StorageConfig(
    used_skills="Normal",
    used_states="Normal",
)
config = DebugConfig(
    tag="test",
    nt=NetworkType.PPO_GNN,
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    env=EnvironmentConfig(render=True),
    reward=EvaluatorConfig(
        step_reward=-1.0,
        success_reward=100.0,
    ),
    storage=storage,
)
