from pathlib import Path
from omegaconf import DictConfig
import pytorch_lightning as pl # type: ignore
from tacorl.datamodule.d4rl_data_module import D4RLDataModule
from src.integrations.taco.old.dataclass import CustomEvalDataset
from src.integrations.taco.transform_manager import CalvinTransformManager
from torch.utils.data import DataLoader

from tacorl.datamodule.dataset.play_dataset import PlayDataset


class CalvinTACORLDataModule(pl.LightningDataModule):

    def __init__(
        self,
        num_workers: int = 4,
        batch_size: int = 32,
        pin_memory: bool = True,
        data_dir: str = "/path/to/calvin/dataset",
        seq_len: int = 16,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dir = data_dir
        self.seq_len = seq_len

    def setup(self, stage=None):
        # Use your custom dataset

        self.train_dataset = PlayDataset(
            data_dir=Path(self.data_dir),
            modalities={}, # pyright: ignore[reportArgumentType]
            action_type="",
            train=True,
            real_world=False,
            min_window_size=16,
            max_window_size=32,
            pad=False,
            transform_manager=CalvinTransformManager(),
            transf_type="train",
            skip_frames=0,
            n_digits=None,
            include_goal= False, #TODO: check
            goal_augmentation: bool = False,
            goal_sampling_prob: float = 0.3,
            goal_strategy_prob: dict[Unknown, Unknown] = { "geometric": 0.5,"similar_robot_obs": 0.5 },
            nn_steps_from_step_path: str = "nn_steps_from_step.pkl",
            num_nn: int = 32,      
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
