from scripts.eval_skills import SkillEvalConfig
from src.modules.rewards.reward import EvaluatorConfig
from src.modules.storage import StorageConfig
from src.environments.environment import EnvironmentConfig
from src.environments.environment import EnvironmentConfig

storage = StorageConfig(
    skills_tag="Normal",
    states_tag="Normal",
    checkpoint_path="results/gnn4/gnn_small_pe_0.0_pr_0.0/model_cp_best.pth",
)
config = SkillEvalConfig(
    tag="eval",
    env=EnvironmentConfig(
        render=False,
    ),
    evaluator=EvaluatorConfig(
        step_reward=-0.01,
        success_reward=1.0,
    ),
    storage=storage,
    iterations=100,
)
