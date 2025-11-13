from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from src.core.modules.reward_module import (
    RewardConfig,
    SparseRewardModule,
)
from src.core.modules.storage_module import StorageModule, StorageConfig
from src.core.environment import EnvironmentConfig
from src.core.agents.agent import AgentConfig
from src.core.networks import NetworkType
from src.core.skills.skill import BaseSkill
from src.experiments.skill_pre_post import SkillEvalExperiment, SkillEvalConfig
from src.integrations.calvin.environment import CalvinEnvironment
from tapas_gmm.utils.argparse import parse_and_build_config
from loguru import logger


@dataclass
class EvalConfig:
    tag: str
    experiment: SkillEvalConfig
    env: EnvironmentConfig
    reward: RewardConfig
    storage: StorageConfig


def train_agent(config: EvalConfig):
    # Initialize the environment and agent
    storage_module = StorageModule(
        config.storage,
        config.tag,
        NetworkType.NONE,
    )
    reward_module = SparseRewardModule(
        config.reward,
        storage_module.states,
    )

    experiment = SkillEvalExperiment(
        config.experiment,
        CalvinEnvironment(
            config.env,
            reward_module,
            storage_module,
        ),
    )  # Wrap environment in experiment

    result_dict: dict[str, int] = {}
    skills_dict: dict[str, BaseSkill] = {
        skill.name: skill for skill in storage_module.skills
    }
    for skill_name, skill in skills_dict.items():
        counter = 0
        if skill_name.endswith("Back"):
            base_name = skill_name.removesuffix("Back")
            pre_skill = skills_dict.get(base_name)
            if pre_skill:
                for i in range(100):
                    terminal = False
                    while not terminal:
                        _, _ = experiment.reset(pre_skill)
                        _ = experiment.step(pre_skill)
                        _, terminal = experiment.evaluate(pre_skill)
                    _ = experiment.step(skill)
                    _, terminal2 = experiment.evaluate(skill)
                    if terminal2:
                        counter += 1
                result_dict[pre_skill.name] = counter
            else:
                print(f"Couldn't find {base_name}")
        elif skill_name.startswith("Place"):
            base_name = skill_name.removeprefix("Place")
            base_name = "Grab" + base_name
            pre_skill = skills_dict.get(base_name)
            if pre_skill:
                for i in range(100):
                    terminal = False
                    while not terminal:
                        _, _ = experiment.reset(pre_skill)
                        _ = experiment.step(pre_skill)
                        _, terminal = experiment.evaluate(pre_skill)
                    _ = experiment.step(skill)
                    _, terminal2 = experiment.evaluate(skill)
                    if terminal2:
                        counter += 1
                result_dict[pre_skill.name] = counter
            else:
                print(f"Couldn't find {base_name}")
        else:
            for i in range(100):
                _, _ = experiment.reset(skill)
                _ = experiment.step(skill)
                _, terminal = experiment.evaluate(skill)
                if terminal:
                    counter += 1
            result_dict[skill.name] = counter

    for key, value in result_dict.items():
        print(f"{key} has successrate: \t {value/100}")
    experiment.close()


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)  # type: ignore


if __name__ == "__main__":
    print("Starting training script...")
    entry_point()
