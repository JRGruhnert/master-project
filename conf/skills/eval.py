from scripts.eval_skills import SkillEvalConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.experiments.skill_check import SkillCheckExperimentConfig
from src.modules.evaluators.skill import SkillEvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig

mode = LogMode.TERMINAL
render = True
tag = "eval"

config = SkillEvalConfig(
    tag=tag,
    iterations=5,
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=tag,
    ),
    storage=StorageConfig(
        skills_tag="DrawerBlock",
        states_tag="Debug",
        tag=tag,
        network="none",
    ),
    experiment=SkillCheckExperimentConfig(
        evaluator=SkillEvaluatorConfig(),
        max_sample_attempts=100,
        sample_with_precons=True,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=SparseEvaluatorConfig(
        step_reward=0.0,
        success_reward=0.0,
    ),
)
