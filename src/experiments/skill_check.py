import torch
from src.modules.evaluators.skill import SkillEvaluator, SkillEvaluatorConfig
from src.modules.storage import Storage
from src.skills.skill import Skill
from src.experiments.experiment import Experiment, ExperimentConfig
from src.environments.environment import Environment
from src.observation.observation import StateValueDict


class SkillCheckExperimentConfig(ExperimentConfig):
    evaluator: SkillEvaluatorConfig


class SkillCheckExperiment(Experiment):
    """Simple Wrapper for centralized data loading and initialisation.
    NOTE: This experiment is very specific to CalvinEnvironment and Tapas Skills and does not generalize.
    """

    def __init__(
        self, config: SkillCheckExperimentConfig, env: Environment, storage: Storage
    ):
        super().__init__(config, env, storage)
        self.config = config
        self.pre_skill = None
        self.evaluator = SkillEvaluator(config.evaluator, storage)

    def sample_task(self, skill: Skill):
        """Samples a new task from the environment that is suitable for the given skill."""
        pre_skill = self._get_prerequisite_skill(skill)
        while True:
            current, goal = self.env.sample_task()
            if pre_skill:
                # If the skill has prerequisite (can only be evaluated by executing prerequisite first)
                obs_like = self._get_conditions_as_observation(
                    pre_skill.precons, current
                )
                if not self.evaluator.is_equal(obs_like, current):
                    continue  # Prerequisite not met, resample
                self.env.step(pre_skill)
            obs_like2 = self._get_conditions_as_observation(skill.precons, current)
            equal = self.evaluator.is_equal(obs_like2, current)
            obs_like3 = self._get_conditions_as_observation(skill.postcons, goal)
            same_areas = self.evaluator.same_areas(obs_like3, goal)
            if equal and same_areas:
                break

    def step(self, skill: Skill) -> bool:
        """Take a step in the environment using the provided skill. Returns True if skill postconditions are met."""
        current, _, _ = self.env.step(skill)
        return self.evaluator.step(
            self._get_conditions_as_observation(skill.postcons, current),
            current,
        )[1]

    def _get_conditions_as_observation(
        self, conditions: dict[str, torch.Tensor], observation: StateValueDict
    ) -> StateValueDict:
        """Convert a dictionary of conditions into a StateValueDict."""
        values = conditions.copy()  # Python passes by reference so..
        # NOTE: Again an exception for the Flip State...
        if "base__button_scalar" in values:
            values["base__button_scalar"] = observation["base__button_scalar"]
        return StateValueDict.from_tensor_dict(values)

    def _get_prerequisite_skill(self, skill: Skill) -> Skill | None:
        """Get the prerequisite skill for a given skill name."""
        skill_name = skill.name
        if skill_name.endswith("Back"):
            pre_skill_name = skill_name.removesuffix("Back")
            return self.storage.get_skill_by_name(pre_skill_name)
        return None

    def metadata(self) -> dict:
        return {}
