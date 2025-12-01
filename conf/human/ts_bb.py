from src.agents.human import HumanAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.modules.buffer import BufferConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig

mode = LogMode.TERMINAL
render = True
tag = "ts_bb"

config = TrainConfig(
    agent=HumanAgentConfig(),
    buffer=BufferConfig(),
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=tag,
    ),
    storage=StorageConfig(
        used_skills="BaseRedPinkBlue",
        used_states="BaseRedPinkBlue",
        eval_states="BaseRedPinkBlue",
        tag=tag,
        network="baseline",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=SparseEvaluatorConfig(
        step_reward=-0.01,
        success_reward=1.0,
    ),
)
