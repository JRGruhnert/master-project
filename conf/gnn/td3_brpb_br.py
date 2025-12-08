from src.agents.ppo.gnn import GNNAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.modules.buffer import BufferConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from conf.common.evaluator import dense3_evaluator

mode = LogMode.WANDB
render = False
eval = False
tag = "td3_brpb_br"

config = TrainConfig(
    agent=GNNAgentConfig(
        eval=eval,
        max_batches=1000,
        early_stop_patience=50,
        min_batches=100,
    ),
    buffer=BufferConfig(steps=1024),
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=tag,
    ),
    storage=StorageConfig(
        used_skills="BaseRed",
        used_states="BaseRedPinkBlue",
        eval_states="BaseRed",
        tag=tag,
        network="gnn",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
