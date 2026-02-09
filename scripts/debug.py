from omegaconf import OmegaConf, SCMode

from tapas_gmm.utils.argparse import parse_and_build_config
from scripts.train import TrainConfig
from src.agents.ppo.ppo import PPOAgent
from src.modules.buffer import Buffer
from src.modules.logger import LoggerConfig, Logger
from src.modules.storage import Storage, StorageConfig
from src.modules.evaluators.evaluator import EvaluatorConfig
from src.environments.environment import EnvironmentConfig
from src.agents.agent import AgentConfig
from src.experiments.experiment import ExperimentConfig
from src.factory import (
    select_agent,
    select_environment,
    select_experiment,
    select_evaluator,
)


class NetworkDebugger:
    """Lightweight debugger for inspecting loaded networks"""

    def __init__(self, config: TrainConfig, storage: Storage):
        self.storage = storage
        self.buffer = Buffer(config.buffer)

        evaluator = select_evaluator(config.evaluator, self.storage)
        env = select_environment(config.environment, evaluator, self.storage)
        self.experiment = select_experiment(config.experiment, env, self.storage)
        self.agent = select_agent(config.agent, self.storage, self.buffer)

    def measure_stats(self):
        """Single-step network inspection (no training)"""
        obs, goal = self.experiment.sample_task()
        if isinstance(self.agent, PPOAgent):
            flops, params = self.agent.measure_flops(obs, goal)
        return flops, params

    def close(self):
        self.experiment.close()


def debug_network(config: TrainConfig):
    test_dict = [
        ("slider", "slider"),
        ("red", "red"),
        ("pink", "pink"),
        ("blue", "blue"),
        ("sr", "sr"),
        ("srp", "srp"),
        ("srpb", "srpb"),
        ("srpb", "slider"),
        ("slider", "srpb"),
    ]
    results = {}
    for item in test_dict:
        overwrite = StorageConfig(
            eval_states=item[1],
            used_skills=item[1],
            used_states=item[0],
            network=config.storage.network,
        )
        storage = Storage(overwrite)
        debugger = NetworkDebugger(config, storage)
        flops, params = debugger.measure_stats()
        results[item] = (flops, params)
        debugger.close()
    print("Results:")
    for key, value in results.items():
        print(f"{key}: FLOPs={value[0]}, Parameters={value[1]}")


def entry_point(p_empty: float | None = None, p_rand: float | None = None):
    _, dict_config = parse_and_build_config(data_load=False, need_task=False)
    if p_empty is not None:
        dict_config["experiment"]["p_empty"] = p_empty
    if p_rand is not None:
        dict_config["experiment"]["p_rand"] = p_rand

    dict_config["storage"]["tag"] = (
        dict_config["storage"]["tag"]
        + f"_pe{dict_config['experiment']['p_empty']}_pr{dict_config['experiment']['p_rand']}"
    )
    dict_config["logger"]["wandb_tag"] = (
        dict_config["logger"]["wandb_tag"]
        + f"_pe{dict_config['experiment']['p_empty']}_pr{dict_config['experiment']['p_rand']}"
    )
    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    debug_network(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
