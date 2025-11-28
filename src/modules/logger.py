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


class Logger:
    def __init__(self, config: LoggerConfig, run: Run | None = None):
        self.config = config
        self.run = run
        self.max_success_rate = float("-inf")

    def start(self, metadata: dict = {}):
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

    def end(self):
        end_time = datetime.now().replace(microsecond=0)
        total_time = (end_time - self.start_time).total_seconds() / 60.0 / 60.0
        logger.info(f"Total run time: {total_time} hours.")

    def log_metrics(self, data: dict, epoch: int):
        if "stats/success_rate" in data:
            self.max_success_rate = max(
                self.max_success_rate, data["stats/success_rate"]
            )
            data["stats/max_success_rate"] = self.max_success_rate
        if (
            self.config.mode == LogMode.WANDB or self.config.mode == LogMode.SWEEP
        ) and self.run:
            self.run.log(data, step=epoch)
        elif self.config.mode == LogMode.TERMINAL:
            pprint.pprint(data)
        else:
            pass  # No logging

    def log_weights(self, data: dict, epoch: int):
        if (
            self.config.mode == LogMode.WANDB or self.config.mode == LogMode.SWEEP
        ) and self.run:
            histogram_data = {
                name: wandb.Histogram(param) for name, param in data.items()
            }
            self.run.log(
                histogram_data,
                step=epoch,
            )
