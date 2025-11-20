import torch
from src.modules.storage import Storage
from src.skills.skill import Skill
from src.experiments.experiment import Experiment, ExperimentConfig
from src.environments.environment import Environment
from src.observation.observation import StateValueDict


class SkillCheckExperimentConfig(ExperimentConfig):
    pass


class SkillCheckExperiment(Experiment):
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(
        self, config: SkillCheckExperimentConfig, env: Environment, storage: Storage
    ):
        super().__init__(config, env, storage)
        self.config = config
        self.pre_skill = None

    def sample_task(self, skill: Skill) -> tuple[StateValueDict, StateValueDict]:
        self.pre_skill, prerequisites = self._get_prerequisite_skill(skill)
        if (
            self.pre_skill
        ):  # Skill has prerequisite (can only be evaluated by executing prerequisite first)
            while not self.eval_start(skill):
                self.env.sample_skill_prerequisites(self.pre_skill, prerequisites)
                self.env.step(self.pre_skill)

        return self.env.sample_skill_prerequisites(skill)

    def step(self, skill: Skill) -> StateValueDict:

        return self.env.step(skill)

    def _get_prerequisite_skill(
        self, skill: Skill
    ) -> tuple[Skill, tuple[str, torch.Tensor]]:
        """Get the prerequisite skill for a given skill name"""
        if skill.name.endswith("Back"):
            return True, skill.name.removesuffix("Back")
        elif skill.name.startswith("Place"):
            base_name = skill.name.removeprefix("Place")
            grab_name = "Grab" + base_name
            return True, grab_name
        return False, ""

    def evaluate(self, skill: Skill) -> tuple[float, bool]:
        return 0.0, self.env.evaluate_skill(skill.postcons)

    def eval_start(self, skill: Skill) -> bool:
        return self.env.evaluate_skill(skill.precons)

    def eval_end(self, skill: Skill) -> bool:
        return self.env.evaluate_skill(skill.postcons)

    def _check_skill_prerequisite(
        current: CalvinObservation,
        goal: CalvinObservation,
    ) -> tuple[bool, str]:
        """Check if a skill has a prerequisite skill and return its name"""
        if pre_skill.name == "PlaceBack":
            # Check if the object is already at the goal position
            obj_name = "Object"  # Assuming single object named "Object"
            current_pos = current.state_dict[f"{obj_name}_position"]
            goal_pos = goal.state_dict[f"{obj_name}_position"]
            distance = torch.norm(current_pos - goal_pos).item()
            if distance < 0.05:  # Threshold to consider as "at goal"
                return False, ""
            else:
                return True, "GrabBack"
        if skill_name.endswith("Back"):
            return True, skill_name.removesuffix("Back")
        elif skill_name.startswith("Place"):
            base_name = skill_name.removeprefix("Place")
            grab_name = "Grab" + base_name
            return True, grab_name
        return False, ""

    def _evaluate_conditional_skill(
        pre_skill: Skill,
        main_skill: Skill,
        iterations: int,
    ) -> float:
        """Evaluate a skill that requires a prerequisite skill to be executed first"""
        counter = 0
        for i in range(iterations):
            # Execute prerequisite skill until completion
            while not experiment.eval_end(pre_skill):
                current, goal = experiment.sample(pre_skill)
                if not experiment.eval_start(pre_skill, main_skill, current, goal):
                    break
                _ = experiment.step(pre_skill)
                # _ = experiment.eval_end(pre_skill)

            # Execute main skill and check success
            _ = experiment.step(main_skill)
            if experiment.eval_end(main_skill):
                counter += 1
        return counter / iterations

    def _evaluate_single_skill(
        skill: Skill,
        iterations: int,
    ) -> float:
        """Evaluate a standalone skill"""
        counter = 0
        for i in range(iterations):
            _, _ = experiment.sample(skill)
            _ = experiment.step(skill)
            if experiment.eval_end(skill):
                counter += 1
        return counter / iterations

    def metadata(self) -> dict:
        return {}
