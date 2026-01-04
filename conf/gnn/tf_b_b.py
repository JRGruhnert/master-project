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
network = "gnn"
prefix = "tf"

skills_eval_states = "b"
used_states = "b"
p_empty = 0.1
p_rand = 0.0

tag = f"{prefix}_{skills_eval_states}_{used_states}"
wandb_tag = f"{network}_{tag}"

config = TrainConfig(
    agent=GNNAgentConfig(
        eval=eval,
        max_batches=300,
        early_stop_patience=50,
        min_batches=100,
    ),
    buffer=BufferConfig(steps=1024),
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
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
