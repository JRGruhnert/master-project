from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode
import concurrent.futures

from conf.shared.experiment import Exp1Config, ExperimentConfig
from hrl.common.agent import MasterAgent
from hrl.common.experiment import Experiment
from hrl.networks import NetworkType
from hrl.state.state import StateSpace
from hrl.skill.skill import SkillSpace
from hrl.env.environment import BaseEnvironment
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class EvalConfig:
    state_space: StateSpace
    task_space: SkillSpace
    tag: str
    experiment: ExperimentConfig
    checkpoint: str = ""


def eval_agent(config: EvalConfig):
    # Initialize the environment and agent
    dl = Experiment(config.state_space, config.task_space, config.experiment.verbose)
    pe = config.experiment.pe
    pr = config.experiment.pr
    max_steps = int(len(dl.skills) * pe + len(dl.skills) * pr + len(dl.skills))
    env = BaseEnvironment(
        config.experiment.env,
        dl.states,
        dl.skills,
        max_steps,
    )
    agent = MasterAgent(
        config.experiment.agent,
        config.experiment.nt,
        config.tag,
        dl.states,
        dl.skills,
    )

    agent.load(config.checkpoint)

    # track total training time
    stop_evaluating = False
    while not stop_evaluating:  # Training loop
        terminal = False
        obs, goal = env.reset()
        while not terminal:
            task = agent.act(obs, goal, eval=True)
            reward, terminal, obs = env.step_exp1(
                task,
                verbose=config.experiment.verbose,
                p_empty=pe,
                p_rand=pr,
            )
            stop_evaluating = agent.feedback(reward, terminal)
    agent.save()

    env.close()
    del dl
    del env
    del agent  # Free up memory


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config: Exp1Config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    # This will submit tasks as workers become free
    configs = []
    # If negative make all
    all_p = [
        v for v in range(config.p_min, config.p_max + config.p_step, config.p_step)
    ]
    pe_values = all_p if config.pe < 0 else [config.pe]
    for pe in pe_values:
        pr_values = all_p if config.pr < 0 else [config.pr]
        for pr in pr_values:
            if pe + pr > 100:
                break  # No need to continue, pr only gets larger

            config.pe = float(pe / 100)
            config.pr = float(pr / 100)
            for origin, goal in config.cross_t:
                origin = 2
                goal = 2
                suffix = f"_pe_{pe}_pr_{pr}/model_cp_best.pth"
                prefix = f"results/{config.nt.value}/t"
                print(f"Suffix: {suffix} from {prefix}")
                eval_config = EvalConfig(
                    state_space=config.t_spaces_mapping[origin]["state_space"],
                    task_space=config.t_spaces_mapping[goal]["task_space"],
                    checkpoint=f"{prefix}{origin}{suffix}",
                    tag=f"e{origin}{goal}",
                    experiment=config,
                )
                configs.append(eval_config)

    for i, eval_config in enumerate(configs):
        eval_agent(eval_config)
        print(f"Progress: {i}/{len(configs)} Last: pe={pe}, pr={pr}")


if __name__ == "__main__":
    entry_point()
