from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode
import wandb

from src.core.agents.search_tree import SearchTreeAgent, SearchTreeAgentConfig
from src.core.modules.buffer_module import BufferModule
from src.core.modules.reward_module import (
    DenseRewardModule,
    RewardConfig,
    SparseRewardModule,
)
from src.core.modules.storage_module import StorageModule, StorageConfig
from src.core.environment import EnvironmentConfig
from src.core.agents.agent import AgentConfig
from src.core.agents.ppo import BaselinePPOAgent, PPOAgentConfig, GNNPPOAgent
from src.core.networks import NetworkType, import_network
from src.experiments.pepr import PePrExperiment, PePrConfig
from src.integrations.calvin.environment import CalvinEnvironment
from tapas_gmm.utils.argparse import parse_and_build_config
from wandb import util as wandb_util  # Explicit import
from loguru import logger


@dataclass
class TrainConfig:
    tag: str
    nt: NetworkType
    experiment: PePrConfig
    agent: AgentConfig
    env: EnvironmentConfig
    reward: RewardConfig
    storage: StorageConfig
    # New wandb parameters
    use_wandb: bool


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
        config.agent.batch_size,
    )
    experiment = PePrExperiment(
        config.experiment,
        CalvinEnvironment(
            config.env,
            reward_module,
            storage_module,
        ),
    )  # Wrap environment in experiment

    if isinstance(config.agent, PPOAgentConfig):
        Net = import_network(config.nt)
        if config.nt is NetworkType.PPO_GNN:
            agent = GNNPPOAgent(
                config.agent,
                Net(storage_module.states, storage_module.skills),
                buffer_module,
                storage_module,
            )
        else:
            agent = BaselinePPOAgent(
                config.agent,
                Net(storage_module.states, storage_module.skills),
                buffer_module,
                storage_module,
            )
        if config.agent.eval:
            logger.info("Loading checkpoint for evaluation...")
            agent.load()

    elif isinstance(config.agent, SearchTreeAgentConfig):
        agent = SearchTreeAgent(
            config.agent,
            buffer_module,
            storage_module,
            reward_module,
        )
    else:
        raise ValueError(f"Unsupported agent config type: {type(config.agent)}")

    logger.info(f"Initialized agent with network type: {config.nt}")

    # Initialize wandb
    if config.use_wandb:
        random_id = wandb_util.generate_id()
        print(f"Random ID for wandb: {random_id}")  # Debug output
        mode = "eval" if config.agent.eval else "train"
        wandb_name = config.nt.value + "_" + mode + "_" + config.tag
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
        if isinstance(agent, BaselinePPOAgent):
            w_and_b = {
                f"weights/{name.replace('.', '/')}": wandb.Histogram(
                    param.data.cpu().numpy()
                )
                for name, param in agent.policy_new.named_parameters()
            }
            run.log(w_and_b, step=0)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    epoch = 0
    start_time_batch = datetime.now().replace(microsecond=0)
    while not stop_training:  # Training loop
        # print(f"Starting Epoch {epoch}")  # Debug output
        terminal = False
        batch_rdy = False
        obs, goal = experiment.reset()
        while not terminal and not batch_rdy:
            skill = agent.act(obs, goal)
            # print(f"Chosen Skill: {skill.name}")  # Debug output
            obs = experiment.step(skill)
            reward, terminal = experiment.evaluate()
            # print(f"Step Reward: {reward}, Terminal: {terminal}")  # Debug output
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            end_time_batch = datetime.now().replace(microsecond=0)
            start_time_learning = datetime.now().replace(microsecond=0)
            total_reward, episode_length, success_rate = agent.buffer_module.stats()
            if config.agent.eval:
                if epoch == 4:  # Stop after 5 eval epochs (0 to 4)
                    stop_training = True
                else:
                    stop_training = False
            else:  # Training mode
                stop_training = agent.learn()
            end_time_learning = datetime.now().replace(microsecond=0)
            epoch += 1
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
            if config.use_wandb:
                # Log weights every 5 epochs (not every epoch to reduce data)
                run.log(metrics, step=epoch)
                if isinstance(agent, BaselinePPOAgent) and epoch % 5 == 0:
                    w_and_b = {
                        f"weights/{name.replace('.', '/')}": wandb.Histogram(
                            param.data.cpu().numpy()
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

    train_agent(config)  # type: ignore


if __name__ == "__main__":
    print("Starting training script...")
    entry_point()
