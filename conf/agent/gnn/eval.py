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
retrain = False
eval = True


network = "gnn"
checkpoint_tag = "t_srpb_srpb_pe0.0_pr0.0"
skills_eval_states = "srpb"
used_states = "srpb"

prefix = "test"
tag = f"{prefix}_{used_states}_{skills_eval_states}"
wandb_tag = f"{network}_{tag}"

config = TrainConfig(
    agent=GNNAgentConfig(
        eval=eval,
        max_batches=10,
        early_stop_patience=10,
        min_batches=10,
        retrain=retrain,
        use_ema_for_early_stopping=False,
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
        checkpoint_path=f"results/{network}/{checkpoint_tag}/model_cp_best.pth",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
