from scripts.skill_eval import EvalConfig
from src.core.modules.reward_module import RewardConfig
from src.core.modules.storage_module import StorageConfig
from src.core.environment import EnvironmentConfig
from src.core.environment import EnvironmentConfig
from src.experiments.skill_pre_post import SkillEvalConfig

storage = StorageConfig(
    skills_tag="Normal",
    states_tag="Normal",
    checkpoint_path="results/gnn4/gnn_small_pe_0.0_pr_0.0/model_cp_best.pth",
)
config = EvalConfig(
    tag="eval",
    experiment=SkillEvalConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    env=EnvironmentConfig(
        render=False,
    ),
    reward=RewardConfig(
        step_reward=-0.01,
        success_reward=1.0,
    ),
    storage=storage,
)
