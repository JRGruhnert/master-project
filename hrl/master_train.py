from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode
import wandb

from hrl.common.experiment import Experiment, ExperimentConfig
from hrl.env.calvin import CalvinEnvironment
from hrl.common.agent import MasterAgent
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class TrainConfig:
    tag: str
    experiment: ExperimentConfig

    # New wandb parameters
    use_wandb: bool


def train_agent(config: TrainConfig):
    # Initialize the environment and agent
    dloader = Experiment(config.experiment)
    env = CalvinEnvironment(config.experiment.env, dloader.max_steps)
    agent = MasterAgent(
        config.experiment.agent,
        config.experiment.nt,
        config.tag,
        dloader.states,
        dloader.skills,
    )

    # Initialize wandb
    if config.use_wandb:
        wandb_name = config.experiment.nt.value + "-" + config.tag
        run = wandb.init(
            entity="jan-gruhnert-universit-t-freiburg",
            project="master-project",
            config={
                "state_tag": config.experiment.states_tag,
                "task_tag": config.experiment.skills_tag,
                "tag": config.tag,
                "nt": config.experiment.nt.value,
                "p_empty": config.experiment.p_empty,
                "p_rand": config.experiment.p_rand,
            },
            name=wandb_name,
            id=wandb_name,
        )
        # Log initial weights with step=0
        for name, param in agent.policy_new.named_parameters():
            clean_name = name.replace(".", "/")  # Better naming for wandb
            run.log(
                {f"weights/{clean_name}": wandb.Histogram(param.data.cpu())}, step=0
            )
            run.log(
                {
                    "train/reward": total_reward,
                    "train/episode_length": episode_length,
                    "train/success_rate": success_rate,
                },
                step=0,
            )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    epoch = 0
    start_time_batch = datetime.now().replace(microsecond=0)
    while not stop_training:  # Training loop
        terminal = False
        batch_rdy = False
        obs, goal = env.reset(dloader.states)
        while not terminal and not batch_rdy:
            skill = agent.act(obs, goal)
            reward, terminal, obs = env.step_exp1(
                skill,
                dloader.skills,
                dloader.states,
                p_empty=config.experiment.p_empty,
                p_rand=config.experiment.p_rand,
            )
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            start_time_learning = datetime.now().replace(microsecond=0)
            stop_training = agent.learn(verbose=True)
            epoch += 1
            if config.use_wandb:
                # Log weights every 5 epochs (not every epoch to reduce data)
                for name, param in agent.policy_new.named_parameters():
                    clean_name = name.replace(
                        ".", "/"
                    )  # actor/0/weight instead of actor.0.weight
                    run.log(
                        {f"weights/{clean_name}": wandb.Histogram(param.data.cpu())},
                        step=epoch,
                    )

                # Always log training metrics
                if hasattr(agent, "buffer") and hasattr(agent.buffer, "stats"):
                    total_reward, episode_length, success_rate = agent.buffer.stats()
                    run.log(
                        {
                            "train/reward": total_reward,
                            "train/episode_length": episode_length,
                            "train/success_rate": success_rate,
                        },
                        step=epoch,
                    )
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
            start_time_batch = datetime.now().replace(microsecond=0)
    env.close()
    run.finish()
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
        + f"_pe_{dict_config['experiment']['p_empty']}_pr_{dict_config['experiment']['p_rand']}"
    )

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


if __name__ == "__main__":
    entry_point()
