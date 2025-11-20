from dataclasses import dataclass
from loguru import logger
import torch
from src.agents.agent import Agent, AgentConfig
from src.agents.ppo.ppo import PPOAgent, PPOAgentConfig
from src.networks.ppo_network import PPO, PPOConfig
from tapas_gmm.utils.select_gpu import device

from src.modules.buffer import Buffer
from src.modules.storage import Storage
from src.networks.gnn.gnn4 import Gnn


@dataclass
class GNNAgentConfig(PPOAgentConfig):
    # Add any GNN-specific configuration parameters here
    pass


class GNNAgent(PPOAgent):
    def __init__(
        self,
        config: GNNAgentConfig,
        storage: Storage,
        buffer: Buffer,
    ):
        super().__init__(config, Gnn(storage.states, storage.skills), buffer, storage)

    def load(self):
        """
        Load the model from the specified path.
        """
        logger.info(
            "Loading GNN checkpoint from: {}".format(
                self.storage.config.checkpoint_path
            )
        )
        checkpoint = torch.load(
            self.storage.config.checkpoint_path, map_location=device
        )

        self.policy_old.load_state_dict(checkpoint["model_state"])
        self.policy_new.load_state_dict(checkpoint["model_state"])
