from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from tapas_gmm.utils.argparse import parse_and_build_config
import wandb
from src.agents.ppo.ppo import PPOAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig
from src.experiments.pepr import PePrConfig
from src.modules.evaluators.dense2 import Dense2EvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig
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
                if skill:
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
        self.logger.log_weights(self.agent.weights(), epoch=self.epoch)
        metrics = self.agent.metrics()

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
    if wandb.run:
        # Use wandb.config to override TrainConfig
        if wandb.config.dense_evaluator:
            print("Using Dense2EvaluatorConfig from wandb config.")
            evaluator = Dense2EvaluatorConfig(
                success_reward=100.0,
                positive_step_reward=5.0,
                negative_step_reward=-1.0,
            )
        else:
            print("Using SparseEvaluatorConfig from wandb config.")
            evaluator = SparseEvaluatorConfig(
                success_reward=100.0,
                step_reward=-1.0,
            )
        config = TrainConfig(
            agent=PPOAgentConfig(
                eval=wandb.config.agent.eval,
                early_stop_patience=wandb.config.agent.early_stop_patience,
                use_ema_for_early_stopping=wandb.config.agent.use_ema_for_early_stopping,
                ema_smoothing_factor=wandb.config.agent.ema_smoothing_factor,
                min_batches=wandb.config.agent.min_batches,
                max_batches=wandb.config.agent.max_batches,
                saving_freq=wandb.config.agent.saving_freq,
                save_stats=wandb.config.agent.save_stats,
                mini_batch_size=wandb.config.agent.mini_batch_size,
                learning_epochs=wandb.config.agent.learning_epochs,
                lr_annealing=wandb.config.agent.lr_annealing,
                learning_rate=wandb.config.agent.learning_rate,
                gamma=wandb.config.agent.gamma,
                gae_lambda=wandb.config.agent.gae_lambda,
                eps_clip=wandb.config.agent.eps_clip,
                entropy_coef=wandb.config.agent.entropy_coef,
                value_coef=wandb.config.agent.value_coef,
                max_grad_norm=wandb.config.agent.max_grad_norm,
                target_kl=(
                    wandb.config.agent.target_kl
                    if wandb.config.agent.target_kl is not None
                    else None
                ),
            ),
            buffer=BufferConfig(
                steps=wandb.config.buffer.steps,
            ),
            logger=LoggerConfig(),
            storage=StorageConfig(
                used_skills=wandb.config.storage.used_skills,
                used_states=wandb.config.storage.used_states,
                eval_states=wandb.config.storage.eval_states,
                network=wandb.config.network.constant,
            ),
            experiment=PePrConfig(
                p_empty=wandb.config.experiment.p_empty,
                p_rand=wandb.config.experiment.p_rand,
            ),
            environment=CalvinEnvironmentConfig(render=False),
            evaluator=evaluator,
        )
    else:
        _, dict_config = parse_and_build_config(data_load=False, need_task=False)

        dict_config["storage"]["tag"] = (
            dict_config["storage"]["tag"]
            + f"_pe{dict_config['experiment']['p_empty']}_pr{dict_config['experiment']['p_rand']}"
        )
        dict_config["logger"]["wandb_tag"] = (
            dict_config["logger"]["wandb_tag"]
            + f"_pe{dict_config['experiment']['p_empty']}_pr{dict_config['experiment']['p_rand']}"
        )
        config = OmegaConf.to_container(
            dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
        )
    train_agent(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
