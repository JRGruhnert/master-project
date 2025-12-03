from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pprint
from loguru import logger
from wandb import util as wandb_util
import wandb
from wandb.wandb_run import Run


class LogMode(Enum):
    SWEEP = "sweep"
    WANDB = "wandb"
    TERMINAL = "terminal"
    NONE = "none"


@dataclass
class LoggerConfig:
    mode: LogMode = LogMode.SWEEP
    wandb_tag: str = "untagged_run"
    wandb_entity: str = "jan-gruhnert-universit-t-freiburg"
    wandb_project: str = "master-project"
    log_weights: int = 5  # Log weights every n epochs or -1 to disable


class Logger:
    def __init__(self, config: LoggerConfig, run: Run | None = None):
        self.config = config
        self.run = run
        self.time: int = 0

    def initialize(self, metadata: dict = {}):
        self.start_time = datetime.now().replace(microsecond=0)
        if self.config.mode == LogMode.WANDB:
            self.run = wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                name=self.config.wandb_tag,
                id=wandb_util.generate_id(),
                config=metadata,
            )
        elif self.config.mode == LogMode.TERMINAL:
            logger.info("Logging to terminal.")
            pprint.pprint(metadata)
        elif self.config.mode == LogMode.NONE:
            logger.info("Logging disabled.")
        elif self.config.mode == LogMode.SWEEP:
            logger.info("Logging in sweep mode.")
        else:
            raise ValueError(f"Unknown logging mode: {self.config.mode}")

    def log(self, data: dict):
        # Remove weights if not logging this epoch or mode
        if (
            self.config.log_weights != -1
            or self.time % self.config.log_weights == 0
            or self.config.mode == LogMode.TERMINAL
        ):
            for key in data.keys():
                if isinstance(key, str) and key.startswith("weights/"):
                    data.pop(key)
        if (
            self.config.mode == LogMode.WANDB or self.config.mode == LogMode.SWEEP
        ) and self.run:
            self.run.log(data, step=self.time)
        elif self.config.mode == LogMode.TERMINAL:
            pprint.pprint(data)
        else:
            pass  # No logging

        self.time += 1
