from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode

from src.modules.rewards.reward import EvaluatorConfig, SparseRewardModule
from src.modules.storage import Storage, StorageConfig
from src.environments.environment import EnvironmentConfig
from src.networks import NetworkType
from src.experiments.pepr import PePrExperiment, PePrConfig
from src.environments.calvin import CalvinEnvironment
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class DebugConfig:
    tag: str
    nt: NetworkType
    experiment: PePrConfig
    env: EnvironmentConfig
    reward: EvaluatorConfig
    storage: StorageConfig


def train_agent(config: DebugConfig):
    # Initialize the environment and agent
    storage_module = Storage(
        config.storage,
        config.tag,
        config.nt,
    )
    storage_module2 = Storage(
        StorageConfig(
            skills_tag="Normal",
            states_tag="Normal",
            # checkpoint_path="results/gnn4/t1_pe_0.0_pr_0.0/model_cp_best.pth",
        ),
        config.tag,
        config.nt,
    )
    reward_module = SparseRewardModule(
        config.reward,
        storage_module.states,
    )
    reward_module2 = SparseRewardModule(
        config.reward,
        storage_module2.states,
    )
    experiment = PePrExperiment(
        config.experiment,
        CalvinEnvironment(
            config.env,
            reward_module,
            storage_module,
        ),
    )  # Wrap environment in experiment

    obs, goal = experiment.sample()
    while True:
        print(f"{0}: Reset")
        for i, task in enumerate(storage_module.skills, start=1):
            print(f"{i}: {task.name}")
        choice = input("Enter the Task id: ")
        task_id = int(choice)
        if task_id == 0:
            print("Resetting environment...")
            obs, goal = experiment.sample()
        else:
            print(
                f"Executing task {task_id}: {storage_module.skills[task_id - 1].name}"
            )
            obs = experiment.step(
                storage_module.skills[task_id - 1]
            )  # Adjust for zero-based index
            reward2, done = reward_module2.step(obs, goal)
            reward, terminal = experiment.evaluate()
            print(f"Step Reward: {reward}")  # Debug output
            print(f"Step Reward 2: {reward2}")


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
