from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from master_project.code.dloader import DataLoader
from master_project.code.common.state import StateSpace
from master_project.code.common.task import TaskSpace
from master_project.code.env.environment import MasterEnv, MasterEnvConfig
from master_project.code.agent import MasterAgent, AgentConfig
from master_project.code.networks import NetworkType
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class RetrainConfig:
    state_space: StateSpace
    task_space: TaskSpace
    tag: str
    nt: NetworkType
    agent: AgentConfig
    env: MasterEnvConfig
    checkpoint: str
    keep_epoch: bool
    verbose: bool = True


def train_agent(config: RetrainConfig):
    # Initialize the environment and agent
    dloader = DataLoader(config.state_space, config.task_space, config.verbose)
    env = MasterEnv(config.env, dloader.states, dloader.tasks)
    agent = MasterAgent(
        config.agent,
        config.nt,
        config.tag,
        dloader.states,
        dloader.tasks,
    )
    agent.load(config.checkpoint)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    while not stop_training:  # Training loop
        start_time_batch = datetime.now().replace(microsecond=0)
        terminal = False
        batch_rdy = False
        obs, goal = env.reset()
        while not terminal and not batch_rdy:
            task = agent.act(obs, goal)
            reward, terminal, obs = env.step_exp1(task, verbose=config.verbose)
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            start_time_learning = datetime.now().replace(microsecond=0)
            stop_training = agent.learn(verbose=config.verbose)
            end_time_learning = datetime.now().replace(microsecond=0)
            print(
                f"""
                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                Batch Duration: {start_time_learning - start_time_batch}
                Learn Duration: {end_time_learning - start_time_learning}
                Elapsed Time:   {end_time_learning - start_time}
                Current Time:   {end_time_learning}
                --------------------------------------------------------------------------------------------
                """
            )
    env.close()
    end_time = datetime.now().replace(microsecond=0)
    print(
        f"""
        ============================================================================================
        Start Time: {start_time}
        End Time:   {end_time}
        Duration:   {end_time - start_time}
        ============================================================================================
        """
    )


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    dict_config["tag"] = (
        dict_config["tag"]
        + f"_pe_{dict_config['env']['p_empty']}_pr_{dict_config['env']['p_rand']}"
    )
    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


"""

if __name__ == "__main__":
    entry_point()
r_config = RetrainConfig(
    state_space=state_space,
    task_space=task_space,
    tag=f"r{p_origin}{p_goal}{suffix}",
    checkpoint=f"results/{nt.value}/t{p_origin}{suffix}/model_cp_best.pth",
    keep_epoch=False,  # Keep the epoch number in the checkpoint
    experiment=experiment1,
)
"""
