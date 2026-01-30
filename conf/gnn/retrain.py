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
retrain_tag = "tf_brpb_br_pe0.0_pr0.0"

network = "gnn"

skills_eval_states = "brp"
used_states = "brpb"


prefix = "rf" if retrain else "tf"
tag = f"{prefix}_{used_states}_{skills_eval_states}"
wandb_tag = f"{network}_{tag}"

config = TrainConfig(
    agent=GNNAgentConfig(
        eval=eval,
        max_batches=750,
        early_stop_patience=50,
        min_batches=250,
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
        checkpoint_path=f"results/{network}/{retrain_tag}/model_cp_best.pth",
    ),
    experiment=PePrConfig(
        p_empty=0.0,
        p_rand=0.0,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=dense3_evaluator,
)
