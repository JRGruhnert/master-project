from dataclasses import dataclass
import numpy as np
from dependencies.tacorl.src.tacorl.modules.tacorl.tacorl_d4rl import TACORL
import torch


@dataclass
class StateTacoConfig:
    play_lmp_dir: str = "~/tacorl/models/play_lmp_state"
    lmp_epoch_to_load: int = 200
    finetune_action_decoder: bool = True


class StateTaco(TACORL):
    def __init__(self, config: StateTacoConfig, state):

        super().__init__(
            play_lmp_dir=config.play_lmp_dir,
            lmp_epoch_to_load=config.lmp_epoch_to_load,
            finetune_action_decoder=config.finetune_action_decoder,
        )
        self.state = state

    def build_networks(self, *args, **kwargs):
        super().build_networks(*args, **kwargs)
        # Additional initialization if needed

    def get_rl_batch(self, batch, latent_plan):
        return super().get_rl_batch(batch, latent_plan)

    def compute_action_decoder_loss(
        self, emb_states, actions, latent_plan, log_type="train"
    ):
        return super().compute_action_decoder_loss(
            emb_states, actions, latent_plan, log_type
        )

    def compute_action_decoder_update(
        self, states, actions, latent_plan, optimize=True, log_type="train"
    ):
        return super().compute_action_decoder_update(
            states, actions, latent_plan, optimize, log_type
        )

    @torch.no_grad()
    def get_pr_latent_plan(self, batch):
        return super().get_pr_latent_plan(batch)

    def training_step(self, batch):

        return super().training_step(batch)

    def configure_optimizers(self):
        return super().configure_optimizers()
