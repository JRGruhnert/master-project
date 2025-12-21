from src.agents.ppo.baseline import BaselineAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.modules.buffer import BufferConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from conf.common.evaluator import dense3_evaluator

mode = LogMode.TERMINAL
render = False
eval = False
tag = "td_brpb_brpb_test"

config = TrainConfig(
    agent=BaselineAgentConfig(
        eval=eval,
        target_kl=0.03,
        learning_epochs=30,
        learning_rate=0.0001,
    ),
    buffer=BufferConfig(
        steps=2,
    ),
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
    evaluator=dense3_evaluator,
)
