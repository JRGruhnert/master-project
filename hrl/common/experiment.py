from dataclasses import dataclass
from data.states.states import STATES_BY_TAG
from data.skills.skills import SKILLS_BY_TAG
from hrl.common.agent import AgentConfig
from hrl.env.calvin import MasterEnvConfig
from hrl.networks import NetworkType


@dataclass
class ExperimentConfig:
    states_tag: str
    skills_tag: str
    p_empty: float
    p_rand: float
    env: MasterEnvConfig
    agent: AgentConfig
    nt: NetworkType


class Experiment:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(
        self,
        config: ExperimentConfig,
    ):
        # We sort based on Id for the baseline network to be consistent
        self.states = sorted(
            STATES_BY_TAG.get(config.states_tag, []), key=lambda s: s.id
        )
        self.skills = sorted(
            SKILLS_BY_TAG.get(config.skills_tag, []), key=lambda s: s.id
        )
        self.p_empty = config.p_empty
        self.p_rand = config.p_rand
        self.max_steps = 5

        for skill in self.skills:
            skill.initialize_conditions(self.states)
            skill.initialize_overrides(self.states)
