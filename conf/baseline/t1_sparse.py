from src.agents.ppo.baseline import BaselineAgentConfig
from src.modules.buffer import BufferConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from src.environments.environment import EnvironmentConfig

mode = LogMode.TERMINAL
render = False
eval = False
tag = "t1_sparse"

config = TrainConfig(
    agent=BaselineAgentConfig(eval=eval),
    buffer=BufferConfig(),
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=tag,
    ),
    storage=StorageConfig(
        skills_tag="Minimal",
        states_tag="Minimal",
        tag=tag,
        network="baseline",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=EnvironmentConfig(render=render),
    evaluator=SparseEvaluatorConfig(
        step_reward=-0.01,
        success_reward=1.0,
    ),
)
