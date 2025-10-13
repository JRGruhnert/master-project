from dataclasses import dataclass
from src.core.agent import BaseAgent
from src.core.agent import BaseAgentConfig
from src.integrations.taco.taco_modified import StateTaco


@dataclass
class TacoAgentConfig(BaseAgentConfig):
    pass


class TacoAgent(BaseAgent):
    def __init__(
        self,
        config: TacoAgentConfig,
    ):
        self.config = config
        self.model = StateTaco()

    def act(self, observation):
        raise NotImplementedError("Act method not implemented yet.")

    def load(self, path: str):
        raise NotImplementedError("Load method not implemented yet.")

    def save(self, path: str):
        raise NotImplementedError("Save method not implemented yet.")

    def train(self):
        raise NotImplementedError("Train method not implemented yet.")

    def eval(self):
        raise NotImplementedError("Eval method not implemented yet.")
