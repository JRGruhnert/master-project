from src.agents.ppo.baseline import BaselineAgentConfig
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
tag = "rf_brpb_brp"

config = TrainConfig(
    agent=BaselineAgentConfig(
        eval=eval,
        max_batches=1000,
        early_stop_patience=50,
        min_batches=100,
        retrain=True,
    ),
    buffer=BufferConfig(steps=1024),
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
        checkpoint_path="results/tf_brpb_br/baseline/best_model.pth",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
