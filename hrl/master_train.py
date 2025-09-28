from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from conf.shared.experiment import ExperimentConfig
from hrl.common.experiment_loader import ExperimentLoader
from hrl.env.calvin import CalvinEnvironment
from hrl.common.agent import MasterAgent
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class TrainConfig:
    state_space: str
    task_space: str
    tag: str
    experiment: ExperimentConfig


def train_agent(config: TrainConfig):
    # Initialize the environment and agent
    dloader = ExperimentLoader(config.state_space, config.task_space, "results/")
    env = CalvinEnvironment(config.experiment.env)
    agent = MasterAgent(
        config.experiment.agent,
        config.experiment.nt,
        config.tag,
        dloader.states,
        dloader.skills,
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    while not stop_training:  # Training loop
        start_time_batch = datetime.now().replace(microsecond=0)
        terminal = False
        batch_rdy = False
        obs, goal = env.reset()
        while not terminal and not batch_rdy:
            skill = agent.act(obs, goal)
            reward, terminal, obs = env.step_exp1(
                skill,
                dloader.skills,
                p_empty=config.experiment.p_empty,
                p_rand=config.experiment.p_rand,
            )
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            start_time_learning = datetime.now().replace(microsecond=0)
            stop_training = agent.learn(verbose=config.experiment.verbose)
            end_time_learning = datetime.now().replace(microsecond=0)
            print(
                f"""
                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                Batch Duration: {start_time_batch - start_time_learning}
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


if __name__ == "__main__":
    entry_point()
