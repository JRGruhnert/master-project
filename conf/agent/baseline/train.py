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
network = "baseline"
prefix = "t"

skills_eval_states = "b"
used_states = "b"
p_empty = 0.0
p_rand = 1.0

tag = f"{prefix}_{used_states}_{skills_eval_states}"
wandb_tag = f"{network}_{tag}"

config = TrainConfig(
    agent=BaselineAgentConfig(
        eval=eval,
        max_batches=1,
        early_stop_patience=1,
        min_batches=1,
        use_ema_for_early_stopping=False,
    ),
    buffer=BufferConfig(steps=4096),
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=wandb_tag,
    ),
    storage=StorageConfig(
        used_skills=skills_eval_states,
        used_states=used_states,
        eval_states=skills_eval_states,
        tag=tag,
        network=network,
    ),
    experiment=PePrConfig(
        p_empty=p_empty,
        p_rand=p_rand,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
