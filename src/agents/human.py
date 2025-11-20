from dataclasses import dataclass
from src.agents.agent import Agent, AgentConfig
from src.modules.buffer import Buffer
from src.observation.observation import StateValueDict
from src.skills.skill import Skill


@dataclass
class HumanAgentConfig(AgentConfig):
    pass


class HumanAgent(Agent):
    def __init__(self, config: HumanAgentConfig, buffer: Buffer):
        self.config = config
        self.buffer = buffer

    def act(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> Skill:
        """Select an action given the current observation and goal observation."""
        raise NotImplementedError("Act method not implemented yet.")

    def feedback(self, reward: float, success: bool, terminal: bool) -> bool:
        """Pass feedback from the environment. Returns True if the buffer reached the targeted batch size."""
        raise NotImplementedError("Feedback method not implemented yet.")

    def learn(self) -> bool:
        """Perform learning update. Returns True if training should stop. (Plateau reached)"""
        raise NotImplementedError("Learn method not implemented yet.")

    def save(self, tag: str = ""):
        raise NotImplementedError("Save method not implemented yet.")

    def load(self):
        raise NotImplementedError("Load method not implemented yet.")

    def metadata(self) -> dict:
        """Return agent metadata as a dictionary."""
        raise NotImplementedError("Metadata method not implemented yet.")

    def metrics(self) -> dict[str, float]:
        """Return current agent metrics as a dictionary."""
        raise NotImplementedError("Metrics method not implemented yet.")

    def weights(self) -> dict[str, float]:
        """Return current agent weights as a dictionary."""
        raise NotImplementedError("Weights method not implemented yet.")
