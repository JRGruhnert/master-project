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

retrain = True
retrain_tag = "rf_brpb_brp"

network = "gnn"

skills_eval_states = "brpb"
used_states = "brpb"


prefix = "rf" if retrain else "tf"
tag = f"{prefix}_{skills_eval_states}_{used_states}"
wandb_tag = f"{network}_{tag}"

config = TrainConfig(
    agent=GNNAgentConfig(
        eval=eval,
        max_batches=500,
        early_stop_patience=50,
        min_batches=100,
        retrain=retrain,
        use_ema_for_early_stopping=False,
    ),
    buffer=BufferConfig(steps=1024),
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=wandb_tag,
    ),
    storage=StorageConfig(
        used_skills="BaseRedPink",
        used_states="BaseRedPinkBlue",
        eval_states="BaseRedPink",
        tag=tag,
        network=network,
        checkpoint_path=f"results/{retrain_tag}/{network}/best_model.pth",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
