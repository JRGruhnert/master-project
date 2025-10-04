from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode
import wandb

from hrl.common.modules.buffer_module import BufferModule
from hrl.common.modules.reward_modules import RewardConfig, SparseRewardModule
from hrl.common.modules.storage_module import StorageModule, StorageConfig
from hrl.env.environment import EnvironmentConfig
from hrl.experiments.pepr import PePrExperiment, PePrConfig
from hrl.env.calvin import CalvinEnvironment
from hrl.common.agent import HRLAgentConfig, HRLAgent
from tapas_gmm.utils.argparse import parse_and_build_config

from hrl.networks import NetworkType, import_network


@dataclass
class TrainConfig:
    tag: str
    nt: NetworkType
    experiment: PePrConfig
    agent: HRLAgentConfig
    env: EnvironmentConfig
    reward: RewardConfig
    storage: StorageConfig
    # New wandb parameters
    use_wandb: bool
    device_tag: str


def train_agent(config: TrainConfig):
    # Initialize the environment and agent
    storage_module = StorageModule(
        config.storage,
        config.tag,
        config.nt,
    )
    reward_module = SparseRewardModule(
        config.reward,
        storage_module.states,
    )
    buffer_module = BufferModule(
        reward_module,
    )
    experiment = PePrExperiment(
        config.experiment,
        CalvinEnvironment(
            config.env,
            reward_module,
            storage_module,
        ),
    )  # Wrap environment in experiment

    Net = import_network(config.nt)
    agent = HRLAgent(
        config.agent,
        Net(storage_module.states, storage_module.skills),
        buffer_module,
        storage_module,
    )

    # Initialize wandb
    if config.use_wandb:
        wandb_name = config.nt.value + "_" + config.device_tag + "_" + config.tag
        run = wandb.init(
            entity="jan-gruhnert-universit-t-freiburg",
            project="master-project",
            config={
                "state_tag": config.storage.states_tag,
                "task_tag": config.storage.skills_tag,
                "tag": config.tag,
                "nt": config.nt.value,
                "p_empty": config.experiment.p_empty,
                "p_rand": config.experiment.p_rand,
                "device": config.device_tag,
            },
            name=wandb_name,
            id=wandb_name,
        )
        # Log initial weights with step=0
        metrics = {
            "train/reward": 0.0,
            "train/episode_length": 0.0,
            "train/success_rate": 0.0,
            "train/batch_duration": 0.0,
            "train/learn_duration": 0.0,
            "train/total_duration": 0.0,
        }
        for name, param in agent.policy_new.named_parameters():
            clean_name = name.replace(".", "/")  # Better naming for wandb
            metrics.update({f"weights/{clean_name}": wandb.Histogram(param.data.cpu())})
        run.log(metrics, step=0)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    epoch = 0
    start_time_batch = datetime.now().replace(microsecond=0)
    while not stop_training:  # Training loop
        terminal = False
        batch_rdy = False
        obs, goal = experiment.reset()
        while not terminal and not batch_rdy:
            skill = agent.act(obs, goal)
            experiment.step(skill)
            reward, terminal = experiment.evaluate()
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            end_time_batch = datetime.now().replace(microsecond=0)
            start_time_learning = datetime.now().replace(microsecond=0)
            stop_training = agent.learn()
            end_time_learning = datetime.now().replace(microsecond=0)
            epoch += 1
            if config.use_wandb:
                # Log weights every 5 epochs (not every epoch to reduce data)
                total_reward, episode_length, success_rate = agent.buffer_module.stats()
                metrics = {
                    "train/reward": total_reward,
                    "train/episode_length": episode_length,
                    "train/success_rate": success_rate,
                    "train/batch_duration_minutes": (
                        end_time_batch - start_time_batch
                    ).total_seconds()
                    / 60,
                    "train/learn_duration_minutes": (
                        end_time_learning - start_time_learning
                    ).total_seconds()
                    / 60,
                    "train/total_duration_minutes": (
                        end_time_learning - start_time
                    ).total_seconds()
                    / 60,
                }
                for name, param in agent.policy_new.named_parameters():
                    clean_name = name.replace(".", "/")  # Better naming for wandb
                    metrics.update(
                        {f"weights/{clean_name}": wandb.Histogram(param.data.cpu())}
                    )
                run.log(metrics, step=epoch)
                print(f"📊 Epoch {epoch}: {metrics}")  # Debug output

            start_time_batch = datetime.now().replace(microsecond=0)
    experiment.close()
    if config.use_wandb:
        run.finish()
    end_time = datetime.now().replace(microsecond=0)
    print(f"Training ended: Total training time: {end_time - start_time}")


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    dict_config["tag"] = (
        dict_config["tag"]
        + f"_pe_{dict_config['experiment']['p_empty']}_pr_{dict_config['experiment']['p_rand']}"
    )

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


if __name__ == "__main__":
    entry_point()
