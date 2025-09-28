from hrl.skill.tapas import Tapas
from hrl.state.state import State
from hrl.skill.skill import Skill


class ExperimentLoader:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(
        self,
        state_space: str,
        skill_space: str,
        relative_path: str,
    ):
        # We sort based on Id for the baseline network to be consistent
        self._states = State.from_json_list(state_space, relative_path + "/states.json")
        self._states.sort(key=lambda s: s.id)
        self._skills = Tapas.from_json_list(skill_space, relative_path + "/skills.json")
        self._skills.sort(key=lambda s: s.id)

    @property
    def states(self) -> list[State]:
        return self._states

    @property
    def skills(self) -> list[Skill]:
        return self._skills
