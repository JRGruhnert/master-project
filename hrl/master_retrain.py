from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode
import wandb

from hrl.experiments.pepr import PePrExperiment
from hrl.env.calvin import CalvinEnvironment, MasterEnvConfig
from hrl.common.agent import HRLAgent, HRLAgentConfig
from hrl.networks import NetworkType
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class RetrainConfig:
    state_space: str
    task_space: str
    tag: str
    nt: NetworkType
    agent: HRLAgentConfig
    env: MasterEnvConfig
    checkpoint: str
    keep_epoch: bool
    verbose: bool = True

    # New wandb parameters
    use_wandb: bool = True
    p_empty: float = 0.0  # Probability of empty skill
    p_rand: float = 0.0  # Probability of random skill


def train_agent(config: RetrainConfig):
    # Initialize the environment and agent
    dloader = PePrExperiment(config.state_space, config.task_space, "data/")
    env = CalvinEnvironment(config.env, dloader.max_steps)
    agent = HRLAgent(
        config.agent,
        config.nt,
        config.tag,
        dloader.states,
        dloader.skills,
    )
    agent.load(config.checkpoint)
    # Initialize wandb
    if config.use_wandb:
        run = wandb.init(
            entity="experiments",
            project="retraining",
            config={
                "state_space": config.state_space,
                "task_space": config.task_space,
                "tag": config.tag,
                "nt": config.nt.value,
                "checkpoint": config.checkpoint,
            },
        )
        for name, param in agent.policy_new.named_parameters():
            run.log({f"{name}": wandb.Histogram(param.data.cpu())})

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
                p_empty=config.p_empty,
                p_rand=config.p_rand,
            )
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            start_time_learning = datetime.now().replace(microsecond=0)
            stop_training = agent.learn(verbose=config.verbose)
            if config.use_wandb:
                for name, param in agent.policy_new.named_parameters():
                    run.log({f"{name}": wandb.Histogram(param.data.cpu())})
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
