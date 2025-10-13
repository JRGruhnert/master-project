from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from omegaconf import OmegaConf, SCMode
import wandb

from src.core.modules.buffer_module import BufferModule
from src.core.modules.reward_module import RewardConfig, SparseRewardModule
from src.core.modules.storage_module import StorageModule, StorageConfig
from src.core.environment import EnvironmentConfig
from src.experiments.pepr import PePrExperiment, PePrConfig
from src.integrations.calvin.environment import CalvinEnvironment
from src.core.agent import HRLAgentConfig, HRLAgent
from tapas_gmm.utils.argparse import parse_and_build_config
from src.core.networks import NetworkType, import_network


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
    experiment = PePrExperiment(
        config.experiment,
        CalvinEnvironment(
            config.env,
            reward_module,
            storage_module,
        ),
    )  # Wrap environment in experiment

    # Initialize wandb
    if config.use_wandb:
        random_id = wandb.util.generate_id()
        print(f"Random ID for wandb: {random_id}")  # Debug output
        wandb_name = config.nt.value + "_" + config.device_tag + "_" + config.tag
        run = wandb.init(
            entity="jan-gruhnert-universit-t-freiburg",
            project="master-project",
            config={
                "state_tag": config.storage.states_tag,
                "task_tag": config.storage.skills_tag,
                "tag": config.tag,
                "nt": "belt",
                "p_empty": config.experiment.p_empty,
                "p_rand": config.experiment.p_rand,
                "device": config.device_tag,
            },
            name=wandb_name,
            id=random_id,
        )
        # Log initial weights with step=0
        metrics = {
            "train/reward": 0.0,
            "train/episode_length": 0.0,
            "train/success_rate": 0.0,
            "train/batch_duration": 0.0,
            "train/learn_duration": 0.0,
        }
        run.log(metrics, step=0)
        w_and_b = {
            f"weights/{name.replace('.', '/')}": wandb.Histogram(param.data.cpu())
            for name, param in agent.policy_new.named_parameters()
        }
        run.log(w_and_b, step=0)

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
            print(
                f"Rewards collected: {len(agent.buffer_module.rewards)}"
            )  # Debug output
            print(
                f"Terminals collected: {len(agent.buffer_module.terminals)}"
            )  # Debug output
            print(
                f"Actions collected: {len(agent.buffer_module.actions)}"
            )  # Debug output
            end_time_batch = datetime.now().replace(microsecond=0)
            start_time_learning = datetime.now().replace(microsecond=0)
            total_reward, episode_length, success_rate = agent.buffer_module.stats()
            stop_training = agent.learn()
            end_time_learning = datetime.now().replace(microsecond=0)
            epoch += 1
            if config.use_wandb:
                # Log weights every 5 epochs (not every epoch to reduce data)
                metrics = {
                    "train/reward": total_reward,
                    "train/episode_length": episode_length,
                    "train/success_rate": success_rate,
                    "train/batch_duration": (
                        end_time_batch - start_time_batch
                    ).total_seconds()
                    / 60,
                    "train/learn_duration": (
                        end_time_learning - start_time_learning
                    ).total_seconds()
                    / 60,
                }
                run.log(metrics, step=epoch)
                w_and_b = {
                    f"weights/{name.replace('.', '/')}": wandb.Histogram(
                        param.data.cpu()
                    )
                    for name, param in agent.policy_new.named_parameters()
                }
                run.log(w_and_b, step=epoch)
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
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    entry_point()
