from dataclasses import dataclass
import json
from omegaconf import OmegaConf, SCMode
from tapas_gmm.utils.argparse import parse_and_build_config
from src.factory import select_environment, select_evaluator
from src.modules.evaluators.evaluator import EvaluatorConfig
from src.modules.logger import Logger, LoggerConfig
from src.modules.storage import Storage, StorageConfig
from src.environments.environment import EnvironmentConfig
from src.experiments.skill_check import SkillCheckExperiment, SkillCheckExperimentConfig
from src.skills.skill import Skill


@dataclass
class SkillEvalConfig:
    tag: str
    logger: LoggerConfig
    storage: StorageConfig
    evaluator: EvaluatorConfig
    experiment: SkillCheckExperimentConfig
    environment: EnvironmentConfig
    iterations: int


class SkillEvaluator:
    def __init__(self, config: SkillEvalConfig):
        self.config = config
        self.storage = Storage(config.storage)
        self.logger = Logger(config.logger)
        evaluator = select_evaluator(config.evaluator, self.storage)
        env = select_environment(config.environment, evaluator, self.storage)
        self.experiment = SkillCheckExperiment(config.experiment, env, self.storage)
        self.results: dict[str, float] = {}

    def evaluate_skill(self, skill: Skill) -> float:
        success_count = 0.0
        for _ in range(self.config.iterations):
            if self.experiment.sample_task(skill):
                if self.experiment.step(skill):
                    success_count += 1.0
            else:
                print(f"Could not sample suitable task for skill {skill.name}")
                break
        return success_count / self.config.iterations

    def run(self):
        self.logger.start({"iterations_per_skill": self.config.iterations})

        # Evaluate all skills
        for index, skill in enumerate(self.storage.skills):
            self.results[skill.name] = self.evaluate_skill(skill)
            metrics = {
                "skill_name": skill.name,
                "success_rate": self.results[skill.name],
            }
            self.logger.log_metrics(metrics, epoch=index)

        # Save results to a JSON file
        with open(f"{self.storage.plots_saving_path}/results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        self.logger.end()
        self.experiment.close()


def eval_skills(config: SkillEvalConfig):
    evaluator = SkillEvaluator(config)
    evaluator.run()


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    eval_skills(config)  # type: ignore
