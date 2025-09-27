from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode

from master_project.code.dloader import DataLoader
from master_project.code.env.environment import MasterEnv, MasterEnvConfig
from master_project.code.agent import AgentConfig
from master_project.code.networks import NetworkType
from master_project.code.state.state import StateSpace
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class MasterConfig:
    task_space: StateSpace
    state_space: StateSpace
    tag: str
    nt: NetworkType
    agent: AgentConfig
    env: MasterEnvConfig
    verbose: bool = True


# 0 - EE
# 1 - Button
# 2 - Drawer
# 3 - Slide
# 4 - Switch
# 5 - Blue
# 6 - Pink
# 7 - Red
# 8 - Led
# 9 - Lightbulb


def train_agent(config: MasterConfig):
    dloader = DataLoader(config.state_space, config.task_space, config.verbose)
    env = MasterEnv(config.env, dloader.states, dloader.tasks)
    env.reset()
    while True:
        print(f"{0}: Reset")
        for i, task in enumerate(dloader.tasks, start=1):
            print(f"{i}: {task.name}")
        choice = input("Enter the Task id: ")
        task_id = int(choice)
        if task_id == 0:
            print("Resetting environment...")
            env.reset()
        else:
            task_id -= 1  # Adjust for zero-based index
            env.step(dloader.tasks[task_id], verbose=True)


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


if __name__ == "__main__":
    entry_point()
