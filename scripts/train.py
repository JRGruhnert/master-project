from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from tapas_gmm.utils.argparse import parse_and_build_config
from src.modules.logger import LoggerConfig, Logger
from src.modules.buffer import BufferConfig, Buffer
from src.modules.evaluators.evaluator import EvaluatorConfig
from src.modules.storage import Storage, StorageConfig
from src.environments.environment import EnvironmentConfig
from src.agents.agent import AgentConfig
from src.experiments.experiment import ExperimentConfig
from src.factory import (
    select_agent,
    select_environment,
    select_experiment,
    select_evaluator,
)


@dataclass
class TrainConfig:
    agent: AgentConfig
    buffer: BufferConfig
    logger: LoggerConfig
    storage: StorageConfig
    evaluator: EvaluatorConfig
    experiment: ExperimentConfig
    environment: EnvironmentConfig


class Trainer:
    """Manages training loop"""

    def __init__(self, config: TrainConfig):
        self.storage = Storage(config.storage)
        self.buffer = Buffer(config.buffer)
        self.logger = Logger(config.logger)

        evaluator = select_evaluator(config.evaluator, self.storage)
        env = select_environment(config.environment, evaluator, self.storage)
        self.experiment = select_experiment(config.experiment, env, self.storage)
        self.agent = select_agent(config.agent, self.storage, self.buffer)

        # Initialize epoch counter (First epoch is the untrained performance)
        self.epoch = 0

    def collect_batch(self) -> bool:
        """Collect experiences until batch is ready"""
        while True:
            obs, goal = self.experiment.sample_task()
            episode_ended = False
            while not episode_ended:
                skill = self.agent.act(obs, goal)
                obs, reward, done, episode_ended = self.experiment.step(skill)
                if self.agent.feedback(reward, done, episode_ended):
                    return True

    def train_epoch(self) -> bool:
        """Train one epoch, return True if agent signals to stop training."""
        # Collect batch
        start_batch = datetime.now().replace(microsecond=0)
        if not self.collect_batch():
            return False

        # We get metrics before learning before the current batch is cleared
        # We also log weights here before learning to have weights for the current epoch
        metrics = self.agent.metrics()
        self.logger.log_weights(self.agent.weights(), epoch=self.epoch)

        # Learn
        start_learning = datetime.now().replace(microsecond=0)
        should_stop = self.agent.learn()
        end_learning = datetime.now().replace(microsecond=0)

        # Logging metrics here to include time information
        metrics["time/collecting"] = (
            start_learning - start_batch
        ).total_seconds() / 60.0
        metrics["time/learning"] = (
            end_learning - start_learning
        ).total_seconds() / 60.0
        self.logger.log_metrics(metrics, epoch=self.epoch)

        self.epoch += 1
        return should_stop

    def run(self):
        """Main training loop"""
        metadata = self.experiment.metadata()
        metadata.update(self.agent.metadata())
        self.logger.start(metadata)

        while not self.train_epoch():
            pass

        self.logger.end()
        self.experiment.close()


def train_agent(config: TrainConfig):
    trainer = Trainer(config)
    trainer.run()


def entry_point():
    _, dict_config = parse_and_build_config(data_load=False, need_task=False)
    dict_config["storage"]["tag"] = (
        dict_config["storage"]["tag"]
        + f"_pe_{dict_config['experiment']['p_empty']}_pr_{dict_config['experiment']['p_rand']}"
    )
    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    train_agent(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
