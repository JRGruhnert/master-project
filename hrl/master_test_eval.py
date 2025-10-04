from dataclasses import dataclass
from datetime import datetime
import os
import numpy as np
from omegaconf import OmegaConf, SCMode
import concurrent.futures

from conf.shared.experiment import ExperimentConfig
from hrl.common.agent import HRLAgent
from hrl.common.buffer import BufferModule
from hrl.experiments.pepr import PePrExperiment
from hrl.networks import NetworkType
from hrl.common.state import StateSpace
from hrl.common.skill import SkillSpace
from hrl.env.environment import BaseEnvironment
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class EvalConfig:
    state_space: StateSpace
    task_space: SkillSpace
    tag: str
    experiment: ExperimentConfig
    checkpoint: str = ""


def eval_task(config: EvalConfig):
    # Initialize the environment and agent
    dloader = PePrExperiment(
        config.state_space, config.task_space, config.experiment.verbose
    )
    env = BaseEnvironment(config.experiment.env, dloader.states, dloader.skills)
    buffer = BufferModule()
    os.makedirs("results/tasks", exist_ok=True)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    task_stats: dict["str", dict[str, float]] = {}
    for task in dloader.skills:
        start_time_batch = datetime.now().replace(microsecond=0)
        task_stats[task.name] = {"reward": [], "terminal": []}
        for _ in range(config.num_samples):
            obs, goal = env.reset(task)
            reward, terminal, obs = env.step_exp1(
                task,
                verbose=config.experiment.verbose,
                p_empty=0.0,
                p_random=0.0,
            )
            task_stats[task.name]["reward"].append(reward)
            task_stats[task.name]["terminal"].append(terminal)
        numpy_stats = {k: np.array(v) for k, v in task_stats[task.name].items()}
        np.savez(f"results/tasks/stats_{task.name}.npz", **numpy_stats)
        end_time_batch = datetime.now().replace(microsecond=0)
        print(
            f"""
            Task: {task.name}
            Batch Duration: {end_time_batch - start_time_batch}
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
    print("Saving Task Stats...")
    for task_name, stats in task_stats.items():
        print(f"Task: {task_name}, Stats: {stats}")


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    pe: float = 0.0  # percentage of empty positions during training
    pr: float = 0.0  # percentage of random positions during training
    nt: NetworkType = NetworkType.GNN_V4

    suffix = f"_pe_{pe}_pr_{pr}/model_cp_best.pth"
    prefix = f"results/{nt.value}/t"
    t_evals = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 1), (2, 3)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # This will submit tasks as workers become free
        for origin, goal in t_evals:
            state_space = StateSpace.Minimal if origin in [1] else StateSpace.Normal
            task_space = SkillSpace.Minimal if goal in [1, 3] else SkillSpace.Normal
            eval_config = EvalConfig(
                state_space=state_space,
                task_space=task_space,
                checkpoint=f"{prefix}{origin}{goal}{suffix}",
                tag=f"e{origin}{goal}" if origin != goal else f"e{origin}",
                experiment=config,
            )
            executor.submit(eval_task, eval_config)
        executor.shutdown(wait=True)  # Wait for all tasks to finish


if __name__ == "__main__":
    entry_point()
