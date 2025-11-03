from src.core.agents.ppo import PPOAgentConfig
from src.core.modules.reward_module import RewardConfig
from src.core.modules.storage_module import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from src.core.networks import NetworkType
from src.core.environment import EnvironmentConfig

from src.core.networks import NetworkType
from src.core.environment import EnvironmentConfig

storage = StorageConfig(
    skills_tag="Normal",
    states_tag="Normal",
    checkpoint_path="results/gnn4/gnn_small_pe_0.0_pr_0.0/model_cp_best.pth",
)
config = TrainConfig(
    tag="test_eval_gnn4",
    nt=NetworkType.PPO_GNN,
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    env=EnvironmentConfig(
        render=False,
    ),
    agent=PPOAgentConfig(
        eval=True,
    ),
    reward=RewardConfig(
        step_reward=-0.01,
        success_reward=1.0,
    ),
    storage=storage,
    use_wandb=False,
)
