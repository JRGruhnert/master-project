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


def evaluate_conditional_skill(
    experiment: SkillEvalExperiment,
    pre_skill: BaseSkill,
    main_skill: BaseSkill,
    iterations: int = 100,
) -> float:
    """Evaluate a skill that requires a prerequisite skill to be executed first"""
    counter = 0
    for i in range(iterations):
        # Execute prerequisite skill until completion
        terminal = False
        while not terminal:
            _, _ = experiment.reset(pre_skill)
            _ = experiment.step(pre_skill)
            _, terminal = experiment.evaluate(pre_skill)

        # Execute main skill and check success
        _ = experiment.step(main_skill)
        _, terminal = experiment.evaluate(main_skill)
        if terminal:
            counter += 1
    return counter / iterations


def evaluate_single_skill(
    experiment: SkillEvalExperiment,
    skill: BaseSkill,
    iterations: int = 100,
) -> float:
    """Evaluate a standalone skill"""
    counter = 0
    for i in range(iterations):
        _, _ = experiment.reset(skill)
        _ = experiment.step(skill)
        _, terminal = experiment.evaluate(skill)
        if terminal:
            counter += 1
    return counter / iterations


def get_prerequisite_skill(skill_name: str) -> tuple[bool, str]:
    """Get the prerequisite skill for a given skill name"""
    if skill_name.endswith("Back"):
        return True, skill_name.removesuffix("Back")
    elif skill_name.startswith("Place"):
        base_name = skill_name.removeprefix("Place")
        grab_name = "Grab" + base_name
        return True, grab_name
    return False, ""


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

    result_dict: dict[str, float] = {}
    skills_dict: dict[str, BaseSkill] = {
        skill.name: skill for skill in storage_module.skills
    }
    for skill_name, skill in skills_dict.items():
        print(f"🔄 Evaluating: {skill_name}")

        has_prerequisite, pre_skill_name = get_prerequisite_skill(skill_name)

        if has_prerequisite:
            print(f"Requires prerequisite: {pre_skill_name}")
            pre_skill = skills_dict.get(pre_skill_name)
            if pre_skill:
                result_dict[skill_name] = evaluate_conditional_skill(
                    experiment,
                    pre_skill,
                    skill,
                )
            else:
                print("This message shouldn't happen!!!")
        else:
            result_dict[skill_name] = evaluate_single_skill(experiment, skill)

        print(f"✅ Success rate: {result_dict[skill_name]:.1%}")

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
