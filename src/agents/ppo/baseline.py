from dataclasses import dataclass

from loguru import logger
import torch
from src.agents.ppo.ppo import PPOAgent, PPOAgentConfig
from src.modules.buffer import Buffer
from src.modules.storage import Storage
from src.hardware import device
from src.networks.baseline.baseline1 import Baseline


@dataclass
class BaselineAgentConfig(PPOAgentConfig):
    pass


class BaselineAgent(PPOAgent):

    def __init__(
        self,
        config: BaselineAgentConfig,
        storage: Storage,
        buffer: Buffer,
    ):
        super().__init__(
            config,
            Baseline(storage.states, storage.skills),
            Baseline(storage.states, storage.skills),
            buffer,
            storage,
        )

        self.load()

    def load(self):
        """Load state_dict and expand dimensions where needed"""
        if self.storage.config.checkpoint_path is None:
            # No checkpoint specified
            return

        logger.info(
            "Loading Baseline checkpoint from: {}".format(
                self.storage.config.checkpoint_path
            )
        )

        checkpoint = torch.load(
            self.storage.config.checkpoint_path, map_location=device
        )

        old_state_dict: dict[str, torch.Tensor] = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )

        # Get current model state dict
        new_state_dict = self.policy_old.state_dict()

        # Copy compatible weights
        for name, old_param in old_state_dict.items():
            if name in new_state_dict:
                new_param = new_state_dict[name]

                # Check if dimensions match
                if old_param.shape == new_param.shape:
                    # Direct copy
                    new_state_dict[name] = old_param
                elif len(old_param.shape) == len(new_param.shape):
                    # Same number of dimensions, try partial copy
                    new_state_dict[name] = self._expand_tensor_dims(
                        old_param, new_param.shape
                    )
                else:
                    print(
                        f"Skipping {name}: incompatible shapes {old_param.shape} -> {new_param.shape}"
                    )

        # Load the modified state dict
        self.policy_old.load_state_dict(new_state_dict, strict=False)
        self.policy_new.load_state_dict(new_state_dict, strict=False)

    def _expand_tensor_dims(self, old_tensor, target_shape):
        """Expand tensor dimensions by copying/padding"""
        old_shape = old_tensor.shape
        new_tensor = torch.zeros(target_shape, dtype=old_tensor.dtype)

        # Copy the overlapping dimensions
        slices = []
        for i, (old_dim, new_dim) in enumerate(zip(old_shape, target_shape)):
            if old_dim <= new_dim:
                slices.append(slice(0, old_dim))
            else:
                slices.append(slice(0, new_dim))

        new_tensor[tuple(slices)] = old_tensor[tuple(slices)]
        return new_tensor
