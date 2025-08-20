from enum import Enum
import json
import pathlib
import pprint
from loguru import logger
from master_project.data_types.task.task import Task
import numpy as np
import torch
from tapas_gmm.master_project.observation import MasterObservation
from tapas_gmm.master_project.state import State
from tapas_gmm.policy.policy import Policy
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
)


class TaskSpace(Enum):
    Minimal = "Minimal"
    Normal = "Normal"
    Full = "Full"
    Debug = "Debug"


class TapasTask(Task):
    def __init__(
        self,
        name: str,
        reversed: bool,
        conditional: bool,
        overrides: list[str],
    ):
        self._name: str = name
        self._reversed: bool = reversed
        self._conditional: bool = conditional
        self._policy_name: str = "gmm"
        self._overrides_keys: list[str] = overrides
        self._policy: GMMPolicy = self._load_policy()
        self._task_parameters: dict[str, torch.Tensor] = {}
        self._overrides: dict[str, np.ndarray] = {}
        # if self._reversed and len(self._overrides_keys) == 0:
        #     NOTE: Tapas safeguard.
        #     TODO: Remove this restriction in the future.
        # #    raise ValueError(
        # #        "Reversed Tasks without overrides can't work in the current Tapas Setup. Most definitely this is not intended."
        # #    )

    @property
    def name(self) -> str:
        return self._name

    @property
    def reversed(self) -> bool:
        return self._reversed

    @property
    def conditional(self) -> bool:
        return self._conditional

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def policy_name(self) -> str:
        return self._policy_name

    @property
    def overrides(self) -> dict[str, np.ndarray]:
        if len(self._overrides) != len(self._overrides_keys):
            raise ValueError("Task overrides have not been initialized.")
        return self._overrides

    @property
    def task_parameters(self) -> dict[str, torch.Tensor]:
        if len(self._task_parameters) == 0:
            raise ValueError("Task parameters have not been initialized.")
        return self._task_parameters

    @property
    def task_parameters_keys(self) -> set[str]:
        return self._task_parameters.keys()

    def _policy_checkpoint_name(self) -> pathlib.Path:
        return (
            pathlib.Path("data")
            / self.name
            / ("demos" + "_" + "gmm" + "_policy" + "-release")
        ).with_suffix(".pt")

    def _get_config(self) -> GMMPolicyConfig:
        """
        Get the configuration for the OpenDrawer policy.
        """
        return GMMPolicyConfig(
            suffix="release",
            model=AutoTPGMMConfig(
                tpgmm=TPGMMConfig(
                    n_components=20,
                    model_type=ModelType.HMM,
                    use_riemann=True,
                    add_time_component=True,
                    add_action_component=False,
                    position_only=False,
                    add_gripper_action=True,
                    reg_shrink=1e-2,
                    reg_diag=2e-4,
                    reg_diag_gripper=2e-2,
                    reg_em_finish_shrink=1e-2,
                    reg_em_finish_diag=2e-4,
                    reg_em_finish_diag_gripper=2e-2,
                    trans_cov_mask_t_pos_corr=False,
                    em_steps=1,
                    fix_first_component=True,
                    fix_last_component=True,
                    reg_init_diag=5e-4,  # 5
                    heal_time_variance=False,
                ),
            ),
            time_based=True,
            predict_dx_in_xdx_models=False,
            binary_gripper_action=True,
            binary_gripper_closed_threshold=0.95,
            dbg_prediction=False,
            # the kinematics model in RLBench is just to unreliable -> leads to mistakes
            topp_in_t_models=False,
            force_overwrite_checkpoint_config=True,  # TODO:  otherwise it doesnt work
            time_scale=1.0,
            # ---- Changing often ----
            postprocess_prediction=False,  # TODO:  abs quaternions if False else delta quaternions
            return_full_batch=True,
            batch_predict_in_t_models=True,  # Change if visualization is needed
            invert_prediction_batch=self.reversed,
        )

    def _load_policy(self) -> GMMPolicy:
        PolicyClass = import_policy(self.policy_name)
        config = self._get_config()
        policy: GMMPolicy = PolicyClass(config).to(device)

        file_name = self._policy_checkpoint_name()  # type: ignore
        logger.info("Loading policy checkpoint from {}", file_name)
        policy.from_disk(file_name)
        policy.eval()
        return policy

    def initialize_task_parameters(self, states: list[State], verbose: bool = False):
        """
        Initialize the task parameters based on the active states.
        """
        tpgmm: AutoTPGMM = self._policy.model
        # Taskparameters of the AutoTPGMM model
        tapas_tp: set[str] = set()
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                pos_str, rot_str = tpgmm.frame_mapping[frame_idx]
                tapas_tp.add(pos_str)
                tapas_tp.add(rot_str)
        # TODO: Currently assumes tapas tps are euler and quaternion
        # My whole code does not generalize to other Task Parameterized models and state types
        for state in states:
            value = state.make_tp(
                tpgmm.start_values[state.name],
                tpgmm.end_values[state.name],
                self.reversed,
                True if state.name in tapas_tp else False,
            )
            if value is not None:
                self._task_parameters[state.name] = value

        if verbose:
            logger.info(f"Initialized task parameters for {self.name}:")
            logger.info(
                "\n" + pprint.pformat(dict(self._task_parameters), indent=2, width=80)
            )

    def initialize_overrides(self, states: list[State], verbose: bool = False):
        """
        Initialize the task parameters based on the active states.
        """
        # TODO: Its a copy of initialize_task_parameters but only override states get loaded and also in reverse
        # So basically normal since reversed is True
        tpgmm: AutoTPGMM = self._policy.model
        for state in states:
            if state.name in self._overrides_keys:
                value = state.make_tp(
                    tpgmm.start_values[state.name],
                    tpgmm.end_values[state.name],
                    not self.reversed,  # NOTE: We want the opposite of the reverse trajectory
                    True,  # NOTE: All States are True here
                )
                if value is None:
                    raise ValueError(
                        f"Failed to create override for state {state.name}. This should not happen."
                    )
                self._overrides[state.name] = value.numpy()
        if verbose:
            logger.info(f"Initialized overrides for {self.name}:")
            logger.info(
                "\n" + pprint.pformat(dict(self._overrides), indent=2, width=80)
            )

    def state_distances(
        self,
        obs: MasterObservation,
        goal: MasterObservation,
        states: list[State],
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in states:
            if state.name in self.task_parameters_keys:
                value = state.distance_to_tp(
                    obs.states[state.name],
                    goal.states[state.name],
                    self._task_parameters[state.name],
                )
                value = torch.tensor([value, 0.0]) if pad else torch.tensor([value])
                # 0.0 pad for tasks parameters
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
                # 1.0 pad for non-task parameters
            task_features.append(value)
        return torch.stack(task_features, dim=0)
