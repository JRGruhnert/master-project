from src.agents.search_tree import SearchTreeAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.modules.buffer import BufferConfig
from src.modules.logger import LogMode, LoggerConfig
from src.modules.storage import StorageConfig
from src.experiments.pepr import PePrConfig
from scripts.train import TrainConfig
from conf.common.evaluator import dense3_evaluator

mode = LogMode.TERMINAL
render = False
retrain = False
eval = True


network = "tree"
skills_eval_states = "br"
used_states = "br"
max_depth = 10
# b = 6
# sr, sb, sp = 8
# br, bb, bp = 10
# brp = 12
# brpb = 14

prefix = "ef"
tag = f"{prefix}_{used_states}_{skills_eval_states}"
wandb_tag = f"{network}_{tag}"

config = TrainConfig(
    agent=SearchTreeAgentConfig(
        max_depth=max_depth,
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
