from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pprint
from loguru import logger
from wandb import util as wandb_util
import wandb
from wandb.wandb_run import Run


class LogMode(Enum):
    WANDB = "wandb"
    TERMINAL = "terminal"
    NONE = "none"


@dataclass
class LoggerConfig:
    mode: LogMode
    wandb_tag: str
    wandb_entity: str = "jan-gruhnert-universit-t-freiburg"
    wandb_project: str = "master-project"


class Logger:
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.run: Run | None = None

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
        else:
            raise ValueError(f"Unknown logging mode: {self.config.mode}")

    def end(self):
        end_time = datetime.now().replace(microsecond=0)
        total_time = (end_time - self.start_time).total_seconds() / 60.0 / 60.0
        logger.info(f"Total run time: {total_time} hours.")
        if self.run:
            self.run.finish()

    def log_metrics(self, data: dict, epoch: int):
        print(f"Batch summary of Epoch {epoch}:")
        if self.config.mode == LogMode.WANDB and self.run:
            self.run.log(data, step=epoch)
        elif self.config.mode == LogMode.TERMINAL:
            pprint.pprint(data)
        else:
            pass  # No logging

    def log_weights(self, data: dict, epoch: int):
        if self.config.mode == LogMode.WANDB and self.run:
            histogram_data = {
                name: wandb.Histogram(param) for name, param in data.items()
            }
            self.run.log(
                histogram_data,
                step=epoch,
            )
