from src.agents.ppo.baseline import BaselineAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.modules.buffer import BufferConfig
from src.modules.evaluators.dense import DenseEvaluatorConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig

mode = LogMode.WANDB
render = False
eval = False
tag = "td_b"

config = TrainConfig(
    agent=BaselineAgentConfig(eval=eval),
    buffer=BufferConfig(),
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=tag,
    ),
    storage=StorageConfig(
        used_skills="Minimal",
        used_states="Minimal",
        tag=tag,
        network="baseline",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=DenseEvaluatorConfig(
        success_reward=1.0,
        negative_step_reward=-0.05,
        positive_step_reward=0.05,
    ),
)
