from dataclasses import dataclass
from loguru import logger
import torch
from src.agents.ppo.ppo import PPOAgent, PPOAgentConfig
from src.hardware import device

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
        super().__init__(
            config,
            Gnn(storage.states, storage.skills),
            Gnn(storage.states, storage.skills),
            buffer,
            storage,
        )
        self.load()

    def load(self):
        """
        Load the model from the specified path.
        """
        if self.storage.config.checkpoint_path is None:
            # No checkpoint specified
            return

        logger.info(
            "Loading GNN checkpoint from: {}".format(
                self.storage.config.checkpoint_path
            )
        )
        checkpoint = torch.load(
            self.storage.config.checkpoint_path, map_location=device
        )

        self.policy_old.load_state_dict(checkpoint["model_state"], strict=False)
        self.policy_new.load_state_dict(checkpoint["model_state"], strict=False)
